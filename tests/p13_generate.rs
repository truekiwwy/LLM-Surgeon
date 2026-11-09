//! S6: greedy generation with a KV cache and activation steering, on a fake Qwen2
//! mini-model (random weights, 4 layers, hidden 16, vocab 64).

use std::collections::HashMap;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use candle_core::{DType, Device, Tensor};
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::Tokenizer;
use weight_surgeon::extract::generate::{
    eos_ids_from_dir, GenRequest, GenStats, Qwen2State, Steering, TokenSource,
};
use weight_surgeon::extract::{
    parse_config, required_tensors_full, ChatTemplate, DTypeChoice, ExtractError, HookedQwen2,
    Qwen2Config, Site,
};
use weight_surgeon::hardware::{GpuInfo, HardwareProbe, MemoryInfo};

const PROMPT: [u32; 5] = [3, 17, 42, 5, 9];
const LAYERS: usize = 4;
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
        "num_hidden_layers":{LAYERS},"num_attention_heads":2,"num_key_value_heads":{kv_heads},
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

fn state(kv_heads: usize, window: usize, dtype: DType) -> (Qwen2State, HookedQwen2) {
    let c = cfg(kv_heads, window);
    let w = weights(&c, 7 + kv_heads as u64);
    let reference = HookedQwen2::from_tensors_full(w.clone(), &c, dtype, &Device::Cpu).unwrap();
    (Qwen2State::from_tensors(w, &c, dtype).unwrap(), reference)
}

fn direction(seed: u64) -> Vec<f32> {
    let mut rng = Lcg(seed);
    (0..HIDDEN).map(|_| rng.next(1.0)).collect()
}

fn steering(layer: usize, alpha_abs: f32) -> Steering {
    Steering {
        layer,
        direction: direction(99),
        alpha_abs,
    }
}

fn request(n: usize, eos: &[u32], steering: Option<Steering>) -> GenRequest {
    GenRequest {
        prompt_ids: PROMPT.to_vec(),
        max_new_tokens: n,
        eos_ids: eos.to_vec(),
        steering,
    }
}

struct Run {
    tokens: Vec<u32>,
    logits: Vec<Vec<f32>>,
    stats: GenStats,
}

fn run(state: &mut Qwen2State, req: &GenRequest) -> Run {
    let (mut tokens, mut logits) = (Vec::new(), Vec::new());
    let stats = state
        .generate_with_logits(
            req,
            &AtomicBool::new(false),
            &mut |id| {
                tokens.push(id);
                ControlFlow::Continue(())
            },
            &mut |l| logits.push(l.to_vec()),
        )
        .unwrap();
    Run {
        tokens,
        logits,
        stats,
    }
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Smallest index among the largest values, as the generator breaks ties.
fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as u32
}

/// Last-position logits of a full forward over `ids`, no cache.
fn full_last_logits(reference: &HookedQwen2, ids: &[u32]) -> Vec<f32> {
    let input = Tensor::new(ids, &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let rows = reference
        .forward_logits(&input)
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    rows.last().unwrap().clone()
}

#[test]
fn kv_cache_matches_full_recomputation() {
    // Window 3 is shorter than the prompt: the sliding window bites during decoding.
    for (kv_heads, window) in [(2, 128), (1, 128), (1, 3)] {
        let (mut state, reference) = state(kv_heads, window, DType::F32);
        let cached = run(&mut state, &request(6, &[], None));
        assert_eq!(cached.tokens.len(), 6);
        let mut context = PROMPT.to_vec();
        for (step, token) in cached.tokens.iter().enumerate() {
            let expected = full_last_logits(&reference, &context);
            let diff = max_diff(&cached.logits[step], &expected);
            assert!(
                diff < 1e-4,
                "kv={kv_heads} window={window} step {step}: {diff}"
            );
            assert_eq!(argmax(&expected), *token, "kv={kv_heads} window={window}");
            context.push(*token);
        }
    }
}

#[test]
fn steered_decoding_matches_steered_prefill_without_cache() {
    let (mut state, _) = state(1, 128, DType::F32);
    let steer = Some(steering(1, 0.8));
    let cached = run(&mut state, &request(6, &[], steer.clone()));
    let baseline = run(&mut state, &request(6, &[], None));
    let spread = cached
        .logits
        .iter()
        .zip(&baseline.logits)
        .map(|(a, b)| max_diff(a, b))
        .fold(0.0, f32::max);
    assert!(spread > 1e-3, "steering must change the logits ({spread})");
    // One new token per call: the whole context goes through the prefill, the cache is
    // never used, and the injection still covers every position.
    let mut context = PROMPT.to_vec();
    for (step, token) in cached.tokens.iter().enumerate() {
        let mut req = request(1, &[], steer.clone());
        req.prompt_ids = context.clone();
        let fresh = run(&mut state, &req);
        assert_eq!(fresh.tokens, [*token], "step {step}");
        let diff = max_diff(&cached.logits[step], &fresh.logits[0]);
        assert!(diff < 1e-4, "step {step}: {diff}");
        context.push(*token);
    }
}

#[test]
fn alpha_zero_is_bit_identical_to_the_baseline() {
    let (mut state, _) = state(2, 128, DType::F32);
    let baseline = run(&mut state, &request(5, &[], None));
    let zero = run(&mut state, &request(5, &[], Some(steering(2, 0.0))));
    assert_eq!(baseline.tokens, zero.tokens);
    assert_eq!(baseline.logits, zero.logits);
    let trait_run: &mut dyn TokenSource = &mut state;
    let mut tokens = Vec::new();
    trait_run
        .generate(&request(5, &[], None), &AtomicBool::new(false), &mut |id| {
            tokens.push(id);
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(tokens, baseline.tokens);
}

#[test]
fn captured_activation_differs_by_exactly_alpha_times_v() {
    for (dtype, tolerance) in [(DType::F32, 1e-5), (DType::BF16, 0.05)] {
        let (mut state, _) = state(2, 128, dtype);
        let steer = steering(2, 1.5);
        let base = state.capture_residual(&PROMPT, 2, None).unwrap();
        let moved = state.capture_residual(&PROMPT, 2, Some(&steer)).unwrap();
        assert_eq!(base.len(), PROMPT.len(), "one row per position");
        for (position, (b, m)) in base.iter().zip(&moved).enumerate() {
            for k in 0..HIDDEN {
                let delta = m[k] - b[k];
                let expected = steer.alpha_abs * steer.direction[k];
                assert!(
                    (delta - expected).abs() < tolerance,
                    "{dtype:?} position {position} dim {k}: {delta} vs {expected}"
                );
            }
        }
        // Before the steered layer nothing moves; after it the change propagates.
        let before = state.capture_residual(&PROMPT, 1, None).unwrap();
        assert_eq!(
            before,
            state.capture_residual(&PROMPT, 1, Some(&steer)).unwrap()
        );
        let after = state.capture_residual(&PROMPT, 3, None).unwrap();
        assert_ne!(
            after,
            state.capture_residual(&PROMPT, 3, Some(&steer)).unwrap()
        );
    }
}

#[test]
fn eos_stops_without_emitting_the_eos_id() {
    let (mut state, _) = state(2, 128, DType::F32);
    let free = run(&mut state, &request(8, &[], None));
    let eos = free.tokens[2];
    let first = free.tokens.iter().position(|&t| t == eos).unwrap();
    let stopped = run(&mut state, &request(8, &[999, eos], None));
    assert_eq!(stopped.tokens, free.tokens[..first]);
    assert!(stopped.stats.stopped_on_eos);
    assert_eq!(
        stopped.stats.generated,
        first + 1,
        "the EOS counts, is not emitted"
    );
    assert!(!free.stats.stopped_on_eos);
}

#[test]
fn token_bound_and_stats() {
    let (mut state, _) = state(2, 128, DType::F32);
    let four = run(&mut state, &request(4, &[], None));
    assert_eq!((four.tokens.len(), four.stats.generated), (4, 4));
    assert!(!four.stats.stopped_on_eos);
    assert!(four.stats.tok_per_s.is_finite() && four.stats.tok_per_s > 0.0);
    let none = run(&mut state, &request(0, &[], None));
    assert!(none.tokens.is_empty() && none.logits.is_empty());
    assert_eq!(none.stats.generated, 0);
}

#[test]
fn chatml_template_and_eos_ids_from_the_model_dir() {
    assert_eq!(
        ChatTemplate::Chatml.apply("salut"),
        "<|im_start|>user\nsalut<|im_end|>\n<|im_start|>assistant\n"
    );
    let tokenizer = |tokens: &[&str]| {
        let vocab = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.to_string(), i as u32))
            .collect();
        Tokenizer::new(WordLevel::builder().vocab(vocab).build().unwrap())
    };
    let full = tokenizer(&["[UNK]", "<|im_end|>", "<|endoftext|>", "a"]);
    let dir = tempfile::tempdir().unwrap();
    let write = |name: &str, content: &str| std::fs::write(dir.path().join(name), content).unwrap();
    // No json at all: only the tokenizer's own markers.
    assert_eq!(eos_ids_from_dir(dir.path(), &full).unwrap(), [1, 2]);
    write("config.json", r#"{"eos_token_id": 1}"#);
    write("generation_config.json", r#"{"eos_token_id": [2, 9]}"#);
    assert_eq!(eos_ids_from_dir(dir.path(), &full).unwrap(), [1, 2, 9]);
    let bare = tokenizer(&["[UNK]", "a"]);
    write("config.json", r#"{"eos_token_id": 5}"#);
    write("generation_config.json", "{}");
    assert_eq!(eos_ids_from_dir(dir.path(), &bare).unwrap(), [5]);
    write("config.json", r#"{"eos_token_id": "x"}"#);
    assert!(matches!(
        eos_ids_from_dir(dir.path(), &bare),
        Err(ExtractError::Config(_))
    ));
}

struct Probe;
impl HardwareProbe for Probe {
    fn system_ram(&self) -> weight_surgeon::hardware::Result<MemoryInfo> {
        Ok(MemoryInfo {
            total_bytes: 1 << 40,
            available_bytes: 1 << 40,
        })
    }
    fn gpus(&self) -> weight_surgeon::hardware::Result<Vec<GpuInfo>> {
        Ok(vec![])
    }
}

#[test]
fn load_from_a_directory_and_refuse_other_architectures() {
    let dir = tempfile::tempdir().unwrap();
    let c = cfg(2, 128);
    let config = format!(
        r#"{{"model_type":"qwen2","vocab_size":64,"hidden_size":{HIDDEN},"intermediate_size":32,
        "num_hidden_layers":{LAYERS},"num_attention_heads":2,"num_key_value_heads":2,
        "max_position_embeddings":128,"rms_norm_eps":1e-6,"rope_theta":10000.0,
        "tie_word_embeddings":false,"sliding_window":128,"hidden_act":"silu"}}"#
    );
    std::fs::write(dir.path().join("config.json"), &config).unwrap();
    let w = weights(&c, 7 + 2);
    candle_core::safetensors::save(&w, dir.path().join("model.safetensors")).unwrap();

    let mut loaded = Qwen2State::load(dir.path(), DTypeChoice::F32, &Probe, 64).unwrap();
    assert_eq!(loaded.config().hidden_size, HIDDEN);
    let mut direct = Qwen2State::from_tensors(w, &c, DType::F32).unwrap();
    let a = run(&mut loaded, &request(4, &[], None));
    let b = run(&mut direct, &request(4, &[], None));
    assert_eq!((a.tokens, a.logits), (b.tokens, b.logits));

    // Another architecture is an invalid argument naming it, from a directory...
    std::fs::write(
        dir.path().join("config.json"),
        config.replace("\"qwen2\"", "\"llama\""),
    )
    .unwrap();
    let err = Qwen2State::load(dir.path(), DTypeChoice::F32, &Probe, 64)
        .err()
        .unwrap();
    assert!(
        matches!(&err, ExtractError::InvalidArgument(m) if m.contains("llama")),
        "{err}"
    );
    // ... and from an in-memory config.
    let mut other = c.clone();
    other.model_type = "qwen3".into();
    let err = Qwen2State::from_tensors(weights(&c, 1), &other, DType::F32)
        .err()
        .unwrap();
    assert!(
        matches!(&err, ExtractError::InvalidArgument(m) if m.contains("qwen3")),
        "{err}"
    );
}

#[test]
fn invalid_requests_are_refused() {
    let (mut state, _) = state(2, 128, DType::F32);
    let refused = |state: &mut Qwen2State, req: GenRequest| {
        let r = state.generate(&req, &AtomicBool::new(false), &mut |_| {
            ControlFlow::Continue(())
        });
        assert!(
            matches!(r, Err(ExtractError::InvalidArgument(_))),
            "{req:?}"
        );
    };
    refused(&mut state, request(2, &[], Some(steering(LAYERS, 1.0))));
    let mut short = steering(0, 1.0);
    short.direction.pop();
    refused(&mut state, request(2, &[], Some(short)));
    refused(&mut state, request(2, &[], Some(steering(0, f32::NAN))));
    refused(
        &mut state,
        request(2, &[], Some(steering(0, f32::INFINITY))),
    );
    let mut nan = steering(0, 1.0);
    nan.direction[3] = f32::NAN;
    refused(&mut state, request(2, &[], Some(nan)));
    let mut empty = request(2, &[], None);
    empty.prompt_ids.clear();
    refused(&mut state, empty);
    let mut outside = request(2, &[], None);
    outside.prompt_ids.push(64);
    refused(&mut state, outside);
    assert!(state.mean_residual_norm(&PROMPT, LAYERS).is_err());
    assert!(state.mean_residual_norm(&[], 0).is_err());
}

#[test]
fn mean_residual_norm_matches_the_early_exit_forward() {
    let c = cfg(2, 128);
    let w = weights(&c, 7 + 2);
    let mut state = Qwen2State::from_tensors(w.clone(), &c, DType::F32).unwrap();
    for layer in 0..LAYERS {
        let early =
            HookedQwen2::from_tensors(w.clone(), &c, layer, DType::F32, &Device::Cpu).unwrap();
        let input = Tensor::new(&PROMPT[..], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let rows = early
            .forward_capture(&input, Site::ResidPost)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let expected = rows
            .iter()
            .map(|r| r.iter().map(|x| x * x).sum::<f32>().sqrt())
            .sum::<f32>()
            / rows.len() as f32;
        let norm = state.mean_residual_norm(&PROMPT, layer).unwrap();
        assert!(
            norm > 0.0 && (norm - expected).abs() < 1e-4 * expected,
            "layer {layer}"
        );
    }
}

/// Counts the layers run, shared with the test.
fn counter(state: &mut Qwen2State) -> Arc<AtomicUsize> {
    let layers = Arc::new(AtomicUsize::new(0));
    let seen = layers.clone();
    state.set_layer_observer(Some(Box::new(move |_| {
        seen.fetch_add(1, Ordering::SeqCst);
    })));
    layers
}

#[test]
fn callback_break_stops_before_the_next_step() {
    let (mut state, _) = state(2, 128, DType::F32);
    let layers = counter(&mut state);
    let mut emitted = Vec::new();
    let stats = state
        .generate(
            &request(10, &[], None),
            &AtomicBool::new(false),
            &mut |id| {
                emitted.push(id);
                match emitted.len() == 2 {
                    true => ControlFlow::Break(()),
                    false => ControlFlow::Continue(()),
                }
            },
        )
        .unwrap();
    assert_eq!((emitted.len(), stats.generated), (2, 2));
    assert_eq!(
        layers.load(Ordering::SeqCst),
        2 * LAYERS,
        "prefill + one decoded step"
    );
}

#[test]
fn flag_raised_mid_step_stops_within_one_layer() {
    let (mut state, _) = state(2, 128, DType::F32);
    let cancel = Arc::new(AtomicBool::new(false));
    let calls = Arc::new(AtomicUsize::new(0));
    let (flag, seen) = (cancel.clone(), calls.clone());
    // Prefill is calls 1..=4, the second step starts at call 5: raise during its layer 1.
    state.set_layer_observer(Some(Box::new(move |layer| {
        let n = seen.fetch_add(1, Ordering::SeqCst) + 1;
        if n == LAYERS + 2 {
            assert_eq!(layer, 1);
            flag.store(true, Ordering::SeqCst);
        }
    })));
    let mut emitted = 0;
    let stats = state
        .generate(&request(10, &[], None), &cancel, &mut |_| {
            emitted += 1;
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        LAYERS + 2,
        "no layer after the raise"
    );
    assert_eq!((emitted, stats.generated), (1, 1));

    // Raised before the call: not a single layer runs.
    let layers = counter(&mut state);
    let done = state
        .generate(&request(10, &[], None), &cancel, &mut |_| {
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!((layers.load(Ordering::SeqCst), done.generated), (0, 0));
}

#[test]
fn flag_raised_from_another_thread_stops_within_one_layer() {
    let (mut state, _) = state(2, 128, DType::F32);
    let cancel = Arc::new(AtomicBool::new(false));
    let armed = Arc::new(AtomicBool::new(false));
    let calls = Arc::new(AtomicUsize::new(0));
    let (flag, ready, seen) = (cancel.clone(), armed.clone(), calls.clone());
    // The observer hands over to the other thread at its 6th layer and waits until the
    // flag is up, so the layer count after the raise is deterministic.
    state.set_layer_observer(Some(Box::new(move |_| {
        if seen.fetch_add(1, Ordering::SeqCst) + 1 == 6 {
            ready.store(true, Ordering::SeqCst);
            let start = Instant::now();
            while !flag.load(Ordering::SeqCst) && start.elapsed() < Duration::from_secs(10) {
                std::thread::yield_now();
            }
        }
    })));
    let (flag, ready) = (cancel.clone(), armed.clone());
    let raiser = std::thread::spawn(move || {
        let start = Instant::now();
        while !ready.load(Ordering::SeqCst) && start.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        flag.store(true, Ordering::SeqCst);
    });
    let stats = state
        .generate(&request(50, &[], None), &cancel, &mut |_| {
            ControlFlow::Continue(())
        })
        .unwrap();
    raiser.join().unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 6);
    assert_eq!(stats.generated, 1, "the second step was cut");
}
