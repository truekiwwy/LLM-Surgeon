//! P17: reproducible sampling, batched native decoding, the llama-server backend.
//!
//! The batched decoder runs the projections and the MLP on `[B, hidden]` and attention per
//! sequence on its own cache: on the mini-model its logits are compared to the sequential
//! ones, and its tokens must be the same.

use std::cell::Cell;
use std::collections::HashMap;
use std::ops::ControlFlow;
use std::sync::atomic::AtomicBool;

use candle_core::{DType, Device, Tensor};

use weight_surgeon::eval::sampling::SamplingParams;
use weight_surgeon::extract::generate::{
    BatchJob, Batching, GenRequest, Qwen2State, Sequence, TokenSource,
};
use weight_surgeon::extract::{
    parse_config, required_tensors_full, HookedDecoder, KvCache, Qwen2Config,
};

const HIDDEN: usize = 16;

/// Deterministic LCG in [-scale, scale].
struct Lcg(u64);
impl Lcg {
    fn next(&mut self, scale: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
    }
}

fn cfg(kv_heads: usize, window: usize) -> Qwen2Config {
    parse_config(&format!(
        r#"{{"model_type":"qwen2","vocab_size":64,"hidden_size":{HIDDEN},"intermediate_size":32,
        "num_hidden_layers":3,"num_attention_heads":2,"num_key_value_heads":{kv_heads},
        "max_position_embeddings":128,"rms_norm_eps":1e-6,"rope_theta":10000.0,
        "tie_word_embeddings":false,"sliding_window":{window},"hidden_act":"silu"}}"#
    ))
    .unwrap()
}

fn weights(c: &Qwen2Config, seed: u64) -> HashMap<String, Tensor> {
    let mut rng = Lcg(seed);
    required_tensors_full(c)
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            let values: Vec<f32> = (0..n)
                .map(|_| match name.contains("norm") {
                    true => 1.0 + rng.next(0.2),
                    false => rng.next(0.3),
                })
                .collect();
            (name, Tensor::from_vec(values, shape, &Device::Cpu).unwrap())
        })
        .collect()
}

fn state(kv_heads: usize, window: usize) -> Qwen2State {
    let c = cfg(kv_heads, window);
    Qwen2State::from_tensors(weights(&c, 11 + kv_heads as u64), &c, DType::F32).unwrap()
}

/// Prompts of different lengths.
fn prompts() -> Vec<Vec<u32>> {
    vec![
        vec![3, 17, 42, 5, 9],
        vec![1],
        vec![60, 2, 2, 7],
        vec![11, 12, 13, 14, 15, 16, 17, 18, 19],
        vec![33, 8],
    ]
}

fn job(prompt: &[u32], n: usize, eos: &[u32], sequence: Sequence) -> BatchJob {
    BatchJob {
        request: GenRequest {
            prompt_ids: prompt.to_vec(),
            max_new_tokens: n,
            eos_ids: eos.to_vec(),
            steering: None,
        },
        sequence,
    }
}

fn sequential(state: &mut Qwen2State, jobs: &[BatchJob]) -> Vec<Vec<u32>> {
    jobs.iter()
        .map(|j| {
            let mut tokens = Vec::new();
            state
                .generate_sampled(
                    &j.request,
                    &j.sequence,
                    &AtomicBool::new(false),
                    &mut |id| {
                        tokens.push(id);
                        ControlFlow::Continue(())
                    },
                )
                .unwrap();
            tokens
        })
        .collect()
}

fn layout(batch_size: usize, workers: usize) -> Batching<'static> {
    Batching {
        batch_size,
        workers,
        pool: None,
    }
}

fn batched_in(state: &mut Qwen2State, jobs: &[BatchJob], layout: Batching) -> Vec<Vec<u32>> {
    let mut tokens = vec![Vec::new(); jobs.len()];
    let done: Vec<Cell<bool>> = jobs.iter().map(|_| Cell::new(false)).collect();
    state
        .generate_batch(
            jobs,
            layout,
            &AtomicBool::new(false),
            &mut |j, id| {
                assert!(!done[j].get(), "token of job {j} after its end");
                tokens[j].push(id);
                ControlFlow::Continue(())
            },
            &mut |j| done[j].set(true),
        )
        .unwrap();
    assert!(done.iter().all(Cell::get), "every job ends");
    tokens
}

fn batched(state: &mut Qwen2State, jobs: &[BatchJob], batch_size: usize) -> Vec<Vec<u32>> {
    batched_in(state, jobs, layout(batch_size, 1))
}

#[test]
fn batched_greedy_decoding_equals_the_sequential_one() {
    for (kv_heads, window) in [(2, 128), (1, 4)] {
        let mut s = state(kv_heads, window);
        let jobs: Vec<BatchJob> = prompts()
            .iter()
            .enumerate()
            .map(|(i, p)| job(p, 12 + i, &[], Sequence::default()))
            .collect();
        let reference = sequential(&mut s, &jobs);
        // The historical greedy entry point gives the same tokens.
        let mut greedy = Vec::new();
        s.generate(&jobs[0].request, &AtomicBool::new(false), &mut |id| {
            greedy.push(id);
            ControlFlow::Continue(())
        })
        .unwrap();
        assert_eq!(greedy, reference[0]);
        for b in [1, 2, 3, 5, 8] {
            for workers in [1, 3] {
                assert_eq!(
                    batched_in(&mut s, &jobs, layout(b, workers)),
                    reference,
                    "kv {kv_heads} window {window} batch {b} workers {workers}"
                );
            }
        }
        // In a dedicated two-thread pool (`--threads 2`): the same tokens.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let in_pool = Batching {
            pool: Some(&pool),
            ..layout(2, 2)
        };
        assert_eq!(batched_in(&mut s, &jobs, in_pool), reference);
    }
}

#[test]
fn sampled_tokens_do_not_depend_on_the_batch_composition() {
    let mut s = state(2, 128);
    let params = SamplingParams {
        temperature: 0.9,
        top_k: 20,
        top_p: 0.95,
        min_p: 0.02,
        repetition_penalty: 1.1,
        seed: 7,
    };
    let jobs: Vec<BatchJob> = prompts()
        .iter()
        .enumerate()
        .map(|(i, p)| {
            job(
                p,
                15,
                &[],
                Sequence {
                    params,
                    id: i as u64,
                },
            )
        })
        .collect();
    let alone = sequential(&mut s, &jobs);
    for b in [1, 2, 4, 5] {
        assert_eq!(batched(&mut s, &jobs, b), alone, "batch {b}");
    }
    // Reversed order, and the job among other ones: same tokens for the same id.
    let reversed: Vec<BatchJob> = jobs.iter().rev().cloned().collect();
    let mut back = batched(&mut s, &reversed, 3);
    back.reverse();
    assert_eq!(back, alone);
    // Another seed changes the draw; the same seed reproduces it.
    let mut reseeded = jobs.clone();
    reseeded.iter_mut().for_each(|j| j.sequence.params.seed = 8);
    assert_ne!(batched(&mut s, &reseeded, 5), alone);
    assert_eq!(batched(&mut s, &jobs, 5), alone);
}

#[test]
fn eos_max_tokens_and_cancellation_in_a_batch() {
    let mut s = state(2, 128);
    // A sampled sequence (varied tokens); its first token not seen before becomes the EOS:
    // counted, not emitted, the sequence leaves the batch.
    let drawn = Sequence {
        params: SamplingParams {
            temperature: 1.0,
            ..SamplingParams::default()
        },
        id: 3,
    };
    let greedy = sequential(&mut s, &[job(&prompts()[0], 10, &[], drawn)]);
    let k = (1..greedy[0].len())
        .find(|&k| !greedy[0][..k].contains(&greedy[0][k]))
        .unwrap();
    let eos = greedy[0][k];
    let jobs = vec![
        job(&prompts()[0], 10, &[eos], drawn),
        job(&prompts()[1], 0, &[], Sequence::default()),
        job(&prompts()[2], 4, &[], Sequence::default()),
    ];
    let mut tokens = vec![Vec::new(); jobs.len()];
    let stats = s
        .generate_batch(
            &jobs,
            layout(2, 1),
            &AtomicBool::new(false),
            &mut |j, id| {
                tokens[j].push(id);
                ControlFlow::Continue(())
            },
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(tokens[0], greedy[0][..k]);
    assert!(stats[0].stopped_on_eos && stats[0].generated == k + 1);
    assert!(tokens[1].is_empty() && stats[1].generated == 0);
    assert_eq!(stats[2].generated, 4);
    // Cancelled before the first step: nothing generated, no error.
    let go_on = &mut |_, _| ControlFlow::Continue(());
    let stats = s
        .generate_batch(
            &jobs,
            layout(2, 1),
            &AtomicBool::new(true),
            go_on,
            &mut |_| {},
        )
        .unwrap();
    assert!(stats.iter().all(|s| s.generated == 0));
    let not_running = AtomicBool::new(false);
    for wrong in [layout(0, 1), layout(2, 0)] {
        assert!(s
            .generate_batch(&jobs, wrong, &not_running, go_on, &mut |_| {})
            .is_err());
    }
    // `Break` from the callback stops that sequence only.
    let mut seen = vec![0usize; jobs.len()];
    s.generate_batch(
        &jobs,
        layout(3, 1),
        &not_running,
        &mut |j, _| {
            seen[j] += 1;
            match j {
                2 => ControlFlow::Break(()),
                _ => ControlFlow::Continue(()),
            }
        },
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(seen[2], 1);
    assert_eq!(seen[0], k);
}

/// Largest `|batched - sequential| / max(1, |sequential|)` over every logit of every step.
fn worst_relative_gap(window: usize) -> f32 {
    let c = cfg(2, window);
    let model =
        HookedDecoder::from_tensors_full(weights(&c, 13), &c, DType::F32, &Device::Cpu).unwrap();
    let layers = c.num_hidden_layers;
    let ids = |v: &[u32]| Tensor::new(v, &Device::Cpu).unwrap().unsqueeze(0).unwrap();
    let keep = &mut |_: usize, x: Tensor| Ok(x);
    let prompts = prompts();
    // Forced next tokens: the same inputs on both paths.
    let next = |s: usize, step: usize| ((s * 7 + step * 13) % 64) as u32;
    let mut caches: Vec<KvCache> = prompts.iter().map(|_| KvCache::new(layers)).collect();
    let mut sequential = Vec::new();
    for (s, prompt) in prompts.iter().enumerate() {
        let mut cache = KvCache::new(layers);
        let mut logits = Vec::new();
        let hidden = model
            .forward_cached(&ids(prompt), &mut cache, layers, keep)
            .unwrap();
        logits.push(model.logits_last(&hidden).unwrap());
        for step in 0..6 {
            let hidden = model
                .forward_cached(&ids(&[next(s, step)]), &mut cache, layers, keep)
                .unwrap();
            logits.push(model.logits_last(&hidden).unwrap());
        }
        sequential.push(logits);
        model
            .forward_cached(&ids(prompt), &mut caches[s], layers, keep)
            .unwrap();
    }
    let mut worst = 0f32;
    for step in 0..6 {
        let step_ids: Vec<u32> = (0..prompts.len()).map(|s| next(s, step)).collect();
        let mut refs: Vec<&mut KvCache> = caches.iter_mut().collect();
        let batched = model.forward_batch_step(&step_ids, &mut refs).unwrap();
        for (s, row) in batched.iter().enumerate() {
            for (b, q) in row.iter().zip(&sequential[s][step + 1]) {
                worst = worst.max((b - q).abs() / q.abs().max(1.0));
            }
        }
    }
    worst
}

#[test]
fn batched_logits_equal_the_sequential_ones_within_rounding() {
    // A [B, hidden] matrix product may sum in another order than a [1, hidden] one: the
    // documented tolerance is 1e-5 relative (f32); the tokens above are identical.
    for window in [128, 4] {
        let gap = worst_relative_gap(window);
        assert!(gap <= 1e-5, "window {window}: {gap}");
        println!("window {window}: worst relative gap {gap:e}");
    }
}
