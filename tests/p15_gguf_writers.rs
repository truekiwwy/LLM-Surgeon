//! P15: the residual-stream writers of a quantized GGUF (`o_proj` -> `attn_output`,
//! `embed_tokens` -> `token_embd`, `down_proj` -> `ffn_down`) under `project_out`, and the
//! Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q5_K codecs. References come from `support/gguf.rs` (an
//! independent decoder and keep-scales requantizer) and from the SafeTensors engine on the
//! dequantized weights: `requant(project(dequant(W)))`.

#[path = "support/gguf.rs"]
mod support;

use std::io::{BufRead, BufReader, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use half::f16;
use serde_json::{json, Value};

use support::*;
use weight_surgeon::engine::{self, DType, SubspaceProjection};
use weight_surgeon::patch::gguf::{
    self, quant, Codec, GgmlType, GgufApplyError, GgufOptions, Requant,
};
use weight_surgeon::patch::{
    apply_patch, parse_patch, validate_patch, Axis, LayerSelection, OpStats, OperationKind,
    PatchBackend, PatchError, Projection, ValidationOptions,
};

const VOCAB: usize = 48;
const INTER: usize = 512;
const LAYERS: usize = 2;

// ------------------------------------------------------------------ fixtures

/// Types of the three writers, and whether the file has its own `output.weight`.
#[derive(Clone, Copy)]
struct Types {
    hidden: usize,
    embd: u32,
    attn_output: u32,
    down: u32,
    output: bool,
}

/// A Qwen2-like stack of residual writers: `token_embd` `ne = [hidden, VOCAB]`, per layer
/// `attn_output` `ne = [hidden, hidden]` and `ffn_down` `ne = [INTER, hidden]`, plus an
/// untouched `ffn_gate` and, if asked, `output.weight`.
fn writers_gguf(t: Types, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    let h = t.hidden;
    let mut tensors = vec![T {
        name: "token_embd.weight".into(),
        ne: vec![h as u64, VOCAB as u64],
        ty: t.embd,
        data: tensor_data(t.embd, VOCAB, h, 1.0, &mut rng),
    }];
    if t.output {
        tensors.push(T {
            name: "output.weight".into(),
            ne: vec![h as u64, VOCAB as u64],
            ty: Q8_0,
            data: tensor_data(Q8_0, VOCAB, h, 1.0, &mut rng),
        });
    }
    for layer in 0..LAYERS {
        tensors.push(T {
            name: format!("blk.{layer}.attn_output.weight"),
            ne: vec![h as u64, h as u64],
            ty: t.attn_output,
            data: tensor_data(t.attn_output, h, h, 0.05, &mut rng),
        });
        tensors.push(T {
            name: format!("blk.{layer}.ffn_gate.weight"),
            ne: vec![h as u64, INTER as u64],
            ty: Q8_0,
            data: tensor_data(Q8_0, INTER, h, 0.02, &mut rng),
        });
        tensors.push(T {
            name: format!("blk.{layer}.ffn_down.weight"),
            ne: vec![INTER as u64, h as u64],
            ty: t.down,
            data: tensor_data(t.down, h, INTER, 0.02, &mut rng),
        });
    }
    gguf_bytes(&metadata(), &tensors)
}

/// `k` orthonormal directions of length `len` (Gram-Schmidt in f64), row-major.
fn orthonormal(k: usize, len: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    let mut rows: Vec<Vec<f64>> = Vec::new();
    while rows.len() < k {
        let mut v: Vec<f64> = (0..len).map(|_| f64::from(rng.normal())).collect();
        for r in &rows {
            let dot: f64 = v.iter().zip(r).map(|(x, y)| x * y).sum();
            v.iter_mut().zip(r).for_each(|(x, y)| *x -= dot * y);
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        rows.push(v.iter().map(|x| x / norm).collect());
    }
    rows.concat().iter().map(|&x| x as f32).collect()
}

fn write_directions(path: &Path, k: usize, values: &[f32]) {
    let len = values.len() / k;
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let n = data.len();
    let mut header = json!({
        "a": {"dtype": "F32", "shape": [k, len], "data_offsets": [0, n]},
        "b": {"dtype": "F32", "shape": [k, len], "data_offsets": [n, 2 * n]},
    })
    .to_string();
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&data);
    file.extend_from_slice(&data);
    std::fs::write(path, file).unwrap();
}

struct Fixture {
    dir: tempfile::TempDir,
    model: PathBuf,
    original: Vec<u8>,
    dirs: Vec<f32>,
    k: usize,
}

impl Fixture {
    fn new(t: Types, seed: u64, k: usize) -> Self {
        let bytes = writers_gguf(t, seed);
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("m.gguf");
        std::fs::write(&model, &bytes).unwrap();
        let dirs = orthonormal(k, t.hidden, seed + 100);
        write_directions(&dir.path().join("dirs.safetensors"), k, &dirs);
        Self {
            dir,
            model,
            original: bytes,
            dirs,
            k,
        }
    }

    fn patch(&self, projections: &[&str], layers: &str, alpha: f32) -> PathBuf {
        let path = self.dir.path().join("patch.json");
        std::fs::write(&path, recipe(projections, layers, alpha).to_string()).unwrap();
        path
    }

    fn bytes(&self) -> Vec<u8> {
        std::fs::read(&self.model).unwrap()
    }

    fn backups(&self) -> usize {
        std::fs::read_dir(self.dir.path())
            .unwrap()
            .filter(|e| {
                e.as_ref()
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .contains(".bak-")
            })
            .count()
    }
}

/// The full ablation recipe of the SafeTensors path.
fn recipe(projections: &[&str], layers: &str, alpha: f32) -> Value {
    json!({"schema_version": 1, "backup": true, "operations": [{
        "id": "ablate", "layers": layers, "projections": projections,
        "operation": "project_out", "vector_file": "dirs.safetensors",
        "vector_tensor": "a", "vector_tensor_b": "b", "alpha": alpha}]})
}

const WRITERS: [&str; 3] = ["embed_tokens", "o_proj", "down_proj"];

fn surgeon() -> Command {
    Command::new(env!("CARGO_BIN_EXE_surgeon"))
}

fn run_json(args: &[&str], env: &[(&str, &str)]) -> (i32, Value) {
    let out = surgeon()
        .arg("--json")
        .args(args)
        .envs(env.iter().copied())
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert_eq!(stdout.lines().count(), 1, "{stdout}{stderr}");
    (
        out.status.code().unwrap(),
        serde_json::from_str(&stdout).unwrap(),
    )
}

fn apply(f: &Fixture, patch: &Path, extra: &[&str]) -> (i32, Value) {
    apply_env(f, patch, extra, &[])
}

fn apply_env(f: &Fixture, patch: &Path, extra: &[&str], env: &[(&str, &str)]) -> (i32, Value) {
    let mut args = vec![
        "apply",
        "--gguf",
        "--model",
        f.model.to_str().unwrap(),
        "--patch",
        patch.to_str().unwrap(),
    ];
    args.extend_from_slice(extra);
    run_json(&args, env)
}

fn tensor<'v>(env: &'v Value, name: &str) -> &'v Value {
    env["data"]["gguf"]["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == name)
        .unwrap_or_else(|| panic!("{name} missing in {env}"))
}

// ------------------------------------------------------------------ the reference

struct Decoded {
    ty: u32,
    rows: usize,
    cols: usize,
    range: std::ops::Range<usize>,
}

fn decoded(bytes: &[u8], name: &str) -> Decoded {
    let file = gguf::parse(bytes).unwrap();
    let t = file.tensor(name).unwrap();
    let (ty, cols, rows) = (t.ggml_type.0, t.ne[0] as usize, t.ne[1] as usize);
    let start = t.start as usize;
    let len = codec(ty).row_bytes(cols) * rows;
    Decoded {
        ty,
        rows,
        cols,
        range: start..start + len,
    }
}

/// `project(dequant(W))` by the SafeTensors engine on the independently decoded weights.
fn exact_projection(w: &[f32], d: &Decoded, local: bool, p: &SubspaceProjection<'_>) -> Vec<f32> {
    let mut bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
    let shape = [d.rows, d.cols];
    match local {
        true => engine::project_out_subspace_embedding(&mut bytes, DType::F32, &shape, p),
        false => engine::project_out_subspace(&mut bytes, DType::F32, &shape, p),
    }
    .unwrap();
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Expected file: `requant_keep_scales(project(dequant(W)))` on every target, nothing else.
fn expected_keep_scales(f: &Fixture, names: &[String], alpha: f32) -> Vec<u8> {
    let mut out = f.original.clone();
    let p = SubspaceProjection {
        a: &f.dirs,
        b: &f.dirs,
        alpha,
        k: f.k,
    };
    for name in names {
        let d = decoded(&f.original, name);
        let w = decode_tensor(d.ty, &f.original[d.range.clone()]);
        let exact = exact_projection(&w, &d, name == "token_embd.weight", &p);
        let c = codec(d.ty);
        let blocks = out[d.range.clone()].chunks_mut(c.block_bytes());
        for ((b, before), exact) in blocks
            .zip(w.chunks(c.block_elems()))
            .zip(exact.chunks(c.block_elems()))
        {
            requant_keep_scales(d.ty, b, before, exact);
        }
    }
    out
}

fn writer_names() -> Vec<String> {
    let mut names = vec!["token_embd.weight".to_owned()];
    for layer in 0..LAYERS {
        names.push(format!("blk.{layer}.attn_output.weight"));
        names.push(format!("blk.{layer}.ffn_down.weight"));
    }
    names
}

/// `‖R(X)‖` with `R = Bᵀ·` (writer) or `· B` (embeddings), in f64.
fn read_norm(x: &[f64], rows: usize, cols: usize, b: &[f32], local: bool) -> f64 {
    let mut sq = 0.0;
    match local {
        true => {
            for row in x.chunks(cols) {
                for b_t in b.chunks(cols) {
                    let r: f64 = row.iter().zip(b_t).map(|(x, b)| x * f64::from(*b)).sum();
                    sq += r * r;
                }
            }
        }
        false => {
            for b_t in b.chunks(rows) {
                for j in 0..cols {
                    let r: f64 = (0..rows).map(|i| f64::from(b_t[i]) * x[i * cols + j]).sum();
                    sq += r * r;
                }
            }
        }
    }
    sq.sqrt()
}

fn widen(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| f64::from(v)).collect()
}

// ------------------------------------------------------------------ codecs: known answers

fn put_f16(b: &mut [u8], at: usize, x: f32) {
    b[at..at + 2].copy_from_slice(&f16::from_f32(x).to_le_bytes());
}

/// Library and independent decoders agree, and give the hand-computed values.
fn check_block(ty: u32, b: &[u8], expected: &[(usize, f32)]) {
    let c = codec(ty);
    let mut y = vec![0.0f32; c.block_elems()];
    quant::dequantize_block(c, b, &mut y);
    assert_eq!(widen(&y), widen(&decode_block(ty, b)), "type {ty}");
    for &(i, v) in expected {
        assert_eq!(y[i], v, "type {ty} element {i}");
    }
}

#[test]
fn legacy_hand_built_blocks() {
    // Q4_0: d = 0.5, qs[0] = 0x3A: elem 0 = (10 - 8) * 0.5, elem 16 = (3 - 8) * 0.5.
    let mut b = vec![0u8; 18];
    put_f16(&mut b, 0, 0.5);
    b[2] = 0x3A;
    check_block(Q4_0, &b, &[(0, 1.0), (16, -2.5), (1, -4.0)]);
    // Q4_1: d = 0.25, m = -1, qs[1] = 0xF0: elem 1 = 0 * d + m, elem 17 = 15 * d + m.
    let mut b = vec![0u8; 20];
    put_f16(&mut b, 0, 0.25);
    put_f16(&mut b, 2, -1.0);
    b[5] = 0xF0;
    check_block(Q4_1, &b, &[(1, -1.0), (17, 2.75)]);
    // Q5_0: d = 0.5, qh bits 0 and 17, qs[0] = 0x01, qs[1] = 0x20.
    let mut b = vec![0u8; 22];
    put_f16(&mut b, 0, 0.5);
    b[2..6].copy_from_slice(&(1u32 | 1 << 17).to_le_bytes());
    b[6] = 0x01;
    b[7] = 0x20;
    check_block(Q5_0, &b, &[(0, 0.5), (16, -8.0), (17, 1.0), (1, -8.0)]);
    // Q5_1: d = 1, m = 0.5, qh bit 31, qs[15] = 0x70: elem 31 = 23 + 0.5, elem 15 = 0.5.
    let mut b = vec![0u8; 24];
    put_f16(&mut b, 0, 1.0);
    put_f16(&mut b, 2, 0.5);
    b[4..8].copy_from_slice(&(1u32 << 31).to_le_bytes());
    b[8 + 15] = 0x70;
    check_block(Q5_1, &b, &[(31, 23.5), (15, 0.5)]);
}

#[test]
fn q5_k_hand_built_block() {
    // d = 1, dmin = 0.5; sub-block 0: sc 2, m 4; sub-block 5: sc 1, m 3 (s[9] = 0x31).
    let mut b = vec![0u8; 176];
    put_f16(&mut b, 0, 1.0);
    put_f16(&mut b, 2, 0.5);
    b[4] = 2;
    b[8] = 4;
    b[4 + 9] = 0x31;
    // Element 3: low nibble of qs[3] = 5, bit 0 of qh[3] -> q = 21: 2 * 21 - 0.5 * 4.
    // Element 163 (chunk 2, high half, lane 3): high nibble of qs[67] = 9, bit 5 of qh[3]
    // -> q = 25: 1 * 25 - 0.5 * 3.
    b[48 + 3] = 0x05;
    b[48 + 67] = 0x90;
    b[16 + 3] = 0x21;
    check_block(Q5_K, &b, &[(3, 40.0), (163, 23.5), (0, -2.0), (35, 0.0)]);
}

const ALL_QUANT: [u32; 8] = [Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K];

/// Random integers, finite random scales.
fn random_block(ty: u32, rng: &mut Rng) -> Vec<u8> {
    let c = codec(ty);
    let mut b: Vec<u8> = (0..c.block_bytes()).map(|_| rng.next() as u8).collect();
    let scale = |rng: &mut Rng| 0.1 * rng.normal();
    match ty {
        Q6_K => put_f16(&mut b, 208, scale(rng)),
        _ => {
            put_f16(&mut b, 0, scale(rng));
            if matches!(ty, Q4_1 | Q5_1 | Q4_K | Q5_K) {
                put_f16(&mut b, 2, scale(rng));
            }
        }
    }
    b
}

#[test]
fn library_codecs_match_the_independent_ones_on_random_blocks() {
    let mut rng = Rng(15);
    for ty in ALL_QUANT {
        let c = codec(ty);
        for _ in 0..200 {
            let b = random_block(ty, &mut rng);
            let mut y = vec![0.0f32; c.block_elems()];
            quant::dequantize_block(c, &b, &mut y);
            let theirs = decode_block(ty, &b);
            assert!(
                y.iter()
                    .zip(&theirs)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "dequantize, type {ty}"
            );
            // keep-scales on a random patch, saturation included.
            let exact: Vec<f32> = y
                .iter()
                .map(|&x| match rng.next() % 3 {
                    0 => x,
                    _ => x + 0.05 * rng.normal(),
                })
                .collect();
            let mut ours = b.clone();
            let r = quant::requantize_row(c, Requant::KeepScales, &y, &exact, &mut ours);
            let mut reference = b.clone();
            let sat = requant_keep_scales(ty, &mut reference, &y, &exact);
            assert_eq!(ours, reference, "keep-scales, type {ty}");
            assert_eq!(r.saturated, sat, "saturation, type {ty}");
        }
    }
}

#[test]
fn recompute_round_trips_the_new_types() {
    let mut rng = Rng(16);
    for (ty, bound) in [
        (Q4_0, 0.02),
        (Q4_1, 0.02),
        (Q5_0, 0.006),
        (Q5_1, 0.006),
        (Q5_K, 0.006),
    ] {
        let c = codec(ty);
        let (mut err, mut norm) = (0.0f64, 0.0f64);
        for _ in 0..64 {
            let x: Vec<f32> = (0..c.block_elems()).map(|_| rng.normal()).collect();
            let mut b = vec![0u8; c.block_bytes()];
            quant::quantize_block(c, &x, &mut b).unwrap();
            let y = decode_block(ty, &b);
            err += x
                .iter()
                .zip(&y)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum::<f64>();
            norm += x.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>();
        }
        assert!(err / norm < bound, "type {ty}: nmse {}", err / norm);
    }
    // An f16 overflow of a recomputed scale is refused, never written.
    let x = vec![1e6f32; 32];
    let mut b = vec![0u8; 18];
    assert!(quant::quantize_block(Codec::Q4_0, &x, &mut b).is_err());
}

#[test]
fn every_new_type_is_parsed_with_its_size() {
    for (ty, block, bytes) in [
        (GgmlType::Q4_0, 32, 18),
        (GgmlType::Q4_1, 32, 20),
        (GgmlType::Q5_0, 32, 22),
        (GgmlType::Q5_1, 32, 24),
        (GgmlType::Q5_K, 256, 176),
    ] {
        assert_eq!(ty.block(), Some((block, bytes)));
        let c = Codec::from_type(ty).unwrap();
        assert_eq!(
            (c.block_elems() as u64, c.block_bytes() as u64),
            (block, bytes)
        );
    }
    assert_eq!(GgmlType(11).block(), None, "Q3_K stays unpatchable");
}

// ------------------------------------------------------------------ project_out on the writers

const COMBOS: [Types; 4] = [
    Types {
        hidden: 256,
        embd: Q5_K,
        attn_output: Q4_K,
        down: Q6_K,
        output: true,
    },
    Types {
        hidden: 256,
        embd: Q4_0,
        attn_output: Q5_1,
        down: Q5_K,
        output: true,
    },
    // hidden 96 is not a multiple of 256: llama.cpp falls back to 32-element blocks.
    Types {
        hidden: 96,
        embd: Q8_0,
        attn_output: Q4_1,
        down: Q5_0,
        output: true,
    },
    Types {
        hidden: 96,
        embd: F16,
        attn_output: Q4_0,
        down: F32,
        output: true,
    },
];

#[test]
fn full_ablation_equals_requant_of_the_projection_byte_for_byte() {
    for (i, t) in COMBOS.into_iter().enumerate() {
        let f = Fixture::new(t, 40 + i as u64, 2);
        let patch = f.patch(&WRITERS, "all", 1.0);
        let (code, env) = apply(&f, &patch, &[]);
        assert_eq!(code, 0, "{env}");
        assert_eq!(env["data"]["gguf"]["verified"], true);
        let expected = expected_keep_scales(&f, &writer_names(), 1.0);
        let got = f.bytes();
        // Header, untouched tensors (gate, output.weight) and every target: one hash.
        assert_eq!(sha(&got), sha(&expected), "combo {i}");
        assert_eq!(env["data"]["gguf"]["tensors"].as_array().unwrap().len(), 5);
        for name in writer_names() {
            let t = tensor(&env, &name);
            assert_eq!(t["refine_passes"], 1, "{t}");
            assert!(t["residual_fraction"].as_f64().is_some(), "{t}");
        }
    }
}

#[test]
fn residual_fractions_match_a_brute_force_computation() {
    let t = COMBOS[1];
    let f = Fixture::new(t, 50, 2);
    let patch = f.patch(&WRITERS, "all", 1.0);
    let (code, env) = apply(&f, &patch, &[]);
    assert_eq!(code, 0, "{env}");
    let after = f.bytes();
    let p = SubspaceProjection {
        a: &f.dirs,
        b: &f.dirs,
        alpha: 1.0,
        k: f.k,
    };
    for name in writer_names() {
        let local = name == "token_embd.weight";
        let d = decoded(&f.original, &name);
        let w = decode_tensor(d.ty, &f.original[d.range.clone()]);
        let w2 = decode_tensor(d.ty, &after[d.range.clone()]);
        let exact = exact_projection(&w, &d, local, &p);
        let norm = |x: &[f64]| read_norm(x, d.rows, d.cols, &f.dirs, local);
        let base = norm(&widen(&w));
        let residual = norm(&widen(&w2)) / base;
        let diff: Vec<f64> = w2
            .iter()
            .zip(&exact)
            .map(|(a, x)| f64::from(*a) - f64::from(*x))
            .collect();
        let requant = norm(&diff) / base;
        let t = tensor(&env, &name);
        let close = |key: &str, want: f64| {
            let got = t[key].as_f64().unwrap();
            assert!(
                (got - want).abs() <= 1e-9 * want.max(1e-12) + 1e-12,
                "{name} {key}: {got} vs {want}"
            );
        };
        close("residual_fraction", residual);
        close("requant_residual_fraction", requant);
        // A full ablation of a projector leaves only the requantization noise.
        assert!(residual < 0.5, "{name}: {residual}");
    }
}

#[test]
fn refinement_lowers_the_residual_or_says_so() {
    let t = Types {
        hidden: 256,
        embd: Q4_0,
        attn_output: Q4_0,
        down: Q4_K,
        output: true,
    };
    let run = |passes: &str| {
        let f = Fixture::new(t, 60, 1);
        let patch = f.patch(&WRITERS, "all", 1.0);
        let (code, env) = apply(
            &f,
            &patch,
            &["--refine-passes", passes, "--residual-target", "1e-6"],
        );
        assert_eq!(code, 0, "{env}");
        env
    };
    let (one, four) = (run("1"), run("4"));
    for name in writer_names() {
        let (a, b) = (tensor(&one, &name), tensor(&four, &name));
        let (r1, r4) = (
            a["requant_residual_fraction"].as_f64().unwrap(),
            b["requant_residual_fraction"].as_f64().unwrap(),
        );
        assert!(r4 < r1, "{name}: {r1} -> {r4}");
        assert_eq!(b["refine_passes"], 4, "{b}");
        // The target cannot be met: the report says so, never "complete".
        let warning = b["warning"].as_str().unwrap();
        assert!(warning.contains("> cible"), "{warning}");
    }
}

#[test]
fn recompute_mode_runs_and_verifies() {
    let f = Fixture::new(COMBOS[0], 61, 2);
    let patch = f.patch(&WRITERS, "all", 0.7);
    let (code, env) = apply(
        &f,
        &patch,
        &["--requant", "recompute", "--refine-passes", "2"],
    );
    assert_eq!(code, 0, "{env}");
    assert_eq!(env["data"]["gguf"]["verified"], true);
    // alpha 0.7 keeps 30 % of the read component.
    let r = tensor(&env, "blk.0.ffn_down.weight")["residual_fraction"]
        .as_f64()
        .unwrap();
    assert!((0.2..0.45).contains(&r), "{r}");
}

#[test]
fn alpha_zero_leaves_the_file_identical() {
    let f = Fixture::new(COMBOS[1], 62, 2);
    let patch = f.patch(&WRITERS, "all", 0.0);
    let (code, env) = apply(&f, &patch, &[]);
    assert_eq!(code, 0, "{env}");
    assert_eq!(sha(&f.bytes()), sha(&f.original));
    assert_eq!(tensor(&env, "token_embd.weight")["residual_fraction"], 1.0);
}

#[test]
fn threads_do_not_change_bytes_nor_residuals() {
    let one = Fixture::new(COMBOS[0], 63, 2);
    let many = Fixture::new(COMBOS[0], 63, 2);
    let args = ["--refine-passes", "3"];
    let (_, e1) = apply_env(
        &one,
        &one.patch(&WRITERS, "all", 1.0),
        &args,
        &[("RAYON_NUM_THREADS", "1")],
    );
    let (_, e2) = apply_env(
        &many,
        &many.patch(&WRITERS, "all", 1.0),
        &args,
        &[("RAYON_NUM_THREADS", "7")],
    );
    assert_eq!(sha(&one.bytes()), sha(&many.bytes()));
    for name in writer_names() {
        for key in [
            "residual_fraction",
            "requant_residual_fraction",
            "nmse_vs_exact",
        ] {
            assert_eq!(
                tensor(&e1, &name)[key],
                tensor(&e2, &name)[key],
                "{name} {key}"
            );
        }
    }
}

// ------------------------------------------------------------------ tied embeddings

#[test]
fn tied_embeddings_are_refused_unless_allowed() {
    let tied = Types {
        output: false,
        ..COMBOS[2]
    };
    let f = Fixture::new(tied, 70, 1);
    let patch = f.patch(&WRITERS, "all", 1.0);
    let (code, env) = apply(&f, &patch, &[]);
    assert_eq!(
        (code, env["error"]["kind"].as_str()),
        (2, Some("invalid_input")),
        "{env}"
    );
    let message = env["error"]["message"].as_str().unwrap();
    assert!(message.contains("--allow-tied-embedding") && message.contains("output.weight"));
    assert_eq!(sha(&f.bytes()), sha(&f.original));
    assert_eq!(f.backups(), 0, "refused before the backup");
    // The MLP and o_proj alone never touch token_embd: no refusal.
    let (code, env) = apply(
        &f,
        &f.patch(&["o_proj", "down_proj"], "all", 1.0),
        &["--dry-run"],
    );
    assert_eq!(code, 0, "{env}");

    let patch = f.patch(&WRITERS, "all", 1.0);
    let (code, env) = apply(&f, &patch, &["--allow-tied-embedding"]);
    assert_eq!(code, 0, "{env}");
    let warning = tensor(&env, "token_embd.weight")["warning"]
        .as_str()
        .unwrap();
    assert!(warning.contains("embeddings liés"), "{warning}");
    assert!(env["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|w| w.as_str().unwrap().contains("embeddings liés")));
}

#[test]
fn with_its_own_head_only_token_embd_changes() {
    let f = Fixture::new(COMBOS[2], 71, 1);
    let (code, env) = apply(&f, &f.patch(&["embed_tokens"], "all", 1.0), &[]);
    assert_eq!(code, 0, "{env}");
    let after = f.bytes();
    let out = decoded(&f.original, "output.weight");
    assert_eq!(sha(&after[out.range.clone()]), sha(&f.original[out.range]));
    let embd = decoded(&f.original, "token_embd.weight");
    assert_ne!(
        sha(&after[embd.range.clone()]),
        sha(&f.original[embd.range])
    );
    assert!(tensor(&env, "token_embd.weight")["warning"]
        .as_str()
        .is_none_or(|w| !w.contains("embeddings liés")));
}

// ------------------------------------------------------------------ preflight and safety

/// `blk.1.attn_output` in Q3_K: the last target of the recipe.
fn with_unsupported_last_target() -> Fixture {
    let f = Fixture::new(COMBOS[0], 80, 1);
    let mut rng = Rng(81);
    let h = COMBOS[0].hidden;
    let mut tensors = vec![T {
        name: "token_embd.weight".into(),
        ne: vec![h as u64, VOCAB as u64],
        ty: Q8_0,
        data: tensor_data(Q8_0, VOCAB, h, 1.0, &mut rng),
    }];
    tensors.push(T {
        name: "output.weight".into(),
        ne: vec![h as u64, VOCAB as u64],
        ty: Q8_0,
        data: tensor_data(Q8_0, VOCAB, h, 1.0, &mut rng),
    });
    for layer in 0..LAYERS {
        let (ty, data) = match layer {
            1 => (Q3_K, vec![0x11; h * h / 256 * 110]),
            _ => (Q4_K, tensor_data(Q4_K, h, h, 0.05, &mut rng)),
        };
        tensors.push(T {
            name: format!("blk.{layer}.attn_output.weight"),
            ne: vec![h as u64, h as u64],
            ty,
            data,
        });
    }
    let bytes = gguf_bytes(&metadata(), &tensors);
    std::fs::write(&f.model, &bytes).unwrap();
    Fixture {
        original: bytes,
        ..f
    }
}

#[test]
fn preflight_refuses_an_unsupported_type_before_any_write() {
    let f = with_unsupported_last_target();
    let (code, env) = apply(&f, &f.patch(&WRITERS, "all", 1.0), &[]);
    assert_eq!(code, 2, "{env}");
    assert_eq!(env["error"]["kind"], "unsupported_quant_type");
    let message = env["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("blk.1.attn_output.weight") && message.contains("Q3_K"),
        "{message}"
    );
    assert_eq!(sha(&f.bytes()), sha(&f.original));
    assert_eq!(f.backups(), 0);
    assert!(!gguf::marker_path(&f.model).exists());
}

#[test]
fn preflight_reports_the_work_and_the_backup() {
    let f = Fixture::new(COMBOS[0], 82, 1);
    let (code, env) = apply(
        &f,
        &f.patch(&WRITERS, "all", 1.0),
        &["--dry-run", "--refine-passes", "2"],
    );
    assert_eq!(code, 0, "{env}");
    let p = &env["data"]["gguf"]["preflight"];
    assert_eq!(p["tensors"], 5, "{p}");
    assert_eq!(p["backup_bytes"], f.original.len() as u64);
    let (target, work) = (
        p["target_bytes"].as_u64().unwrap(),
        p["work_bytes"].as_u64().unwrap(),
    );
    // Embeddings: 2 passes + write; writers: R, 2 passes, write.
    assert!(work > 3 * target && work <= 4 * target, "{p}");
    assert_eq!(sha(&f.bytes()), sha(&f.original), "dry run");
    // An option out of range is refused upfront.
    let (code, env) = apply(
        &f,
        &f.patch(&WRITERS, "all", 1.0),
        &["--residual-target", "0"],
    );
    assert_eq!(
        (code, env["error"]["kind"].as_str()),
        (2, Some("invalid_input")),
        "{env}"
    );
    // The GGUF options are refused without --gguf.
    let dirs = f.dir.path().join("dirs.safetensors");
    let patch = f.patch(&WRITERS, "all", 1.0);
    let (code, env) = run_json(
        &[
            "apply",
            "--model",
            dirs.to_str().unwrap(),
            "--patch",
            patch.to_str().unwrap(),
            "--refine-passes",
            "2",
        ],
        &[],
    );
    assert_eq!(code, 2, "{env}");
    assert!(env["error"]["message"]
        .as_str()
        .unwrap()
        .contains("--refine-passes"));
}

/// Fails the second `project_out`: the first tensor is already written.
struct FailSecond<'a> {
    inner: &'a mut gguf::GgufBackend,
    calls: usize,
}

impl PatchBackend for FailSecond<'_> {
    fn resolve_targets(
        &self,
        layers: &LayerSelection,
        projections: &[Projection],
    ) -> weight_surgeon::patch::Result<Vec<String>> {
        self.inner.resolve_targets(layers, projections)
    }
    fn load_vector(&self, file: &Path, t: Option<&str>) -> weight_surgeon::patch::Result<Vec<f32>> {
        self.inner.load_vector(file, t)
    }
    fn load_directions(
        &self,
        file: &Path,
        t: Option<&str>,
    ) -> weight_surgeon::patch::Result<(Option<usize>, Vec<f32>)> {
        self.inner.load_directions(file, t)
    }
    fn shift_tensor(
        &mut self,
        name: &str,
        v: &[f32],
        alpha: f32,
        op: OperationKind,
        axis: Axis,
    ) -> weight_surgeon::patch::Result<OpStats> {
        self.inner.shift_tensor(name, v, alpha, op, axis)
    }
    fn project_out_subspace(
        &mut self,
        name: &str,
        a: &[f32],
        b: &[f32],
        k: usize,
        alpha: f32,
    ) -> weight_surgeon::patch::Result<OpStats> {
        self.calls += 1;
        match self.calls {
            1 => self.inner.project_out_subspace(name, a, b, k, alpha),
            _ => Err(PatchError::Backend("injected".into())),
        }
    }
    fn flush(&mut self) -> weight_surgeon::patch::Result<()> {
        self.inner.flush()
    }
}

#[test]
fn injected_failure_rolls_back_with_the_marker_alive_during_the_write() {
    let f = Fixture::new(COMBOS[1], 90, 2);
    let patch = parse_patch(&recipe(&WRITERS, "all", 1.0).to_string()).unwrap();
    let patch = validate_patch(&patch, f.dir.path(), &ValidationOptions::default()).unwrap();
    let marker = gguf::marker_path(&f.model);
    let mut written_first = false;
    let result = gguf::apply_gguf(
        &f.model,
        &patch,
        &GgufOptions::default(),
        false,
        Arc::new(AtomicBool::new(false)),
        &mut |backend, guard| {
            assert!(marker.exists());
            let mut failing = FailSecond {
                inner: backend,
                calls: 0,
            };
            let report = apply_patch(&patch, &mut failing, guard, false);
            written_first = failing.calls >= 2;
            report
        },
    );
    assert!(written_first);
    match result {
        Err(GgufApplyError::Patch(PatchError::ExecutionFailed { rolled_back, .. })) => {
            assert!(rolled_back)
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(sha(&f.bytes()), sha(&f.original));
    assert!(!marker.exists());
}

#[test]
fn a_non_finite_scale_in_a_target_is_refused_and_restored() {
    let f = Fixture::new(COMBOS[0], 91, 1);
    let mut bytes = f.original.clone();
    let d = decoded(&bytes, "blk.1.attn_output.weight");
    // First block of the second writer: d = NaN (f16 0x7E00).
    bytes[d.range.start..d.range.start + 2].copy_from_slice(&0x7E00u16.to_le_bytes());
    std::fs::write(&f.model, &bytes).unwrap();
    let (code, env) = apply(&f, &f.patch(&WRITERS, "all", 1.0), &[]);
    assert_ne!(code, 0, "{env}");
    let message = env["error"]["message"].as_str().unwrap();
    assert!(message.contains("non finie"), "{message}");
    assert_eq!(sha(&f.bytes()), sha(&bytes), "restored from the backup");
    assert!(!gguf::marker_path(&f.model).exists());
}

#[test]
fn truncated_or_corrupted_files_never_panic() {
    let bytes = writers_gguf(COMBOS[1], 92);
    for cut in (0..bytes.len()).step_by(bytes.len() / 97 + 1) {
        let _ = gguf::parse(&bytes[..cut]);
    }
    let mut rng = Rng(93);
    for _ in 0..200 {
        let mut b = bytes.clone();
        let at = (rng.next() as usize) % b.len();
        b[at] ^= 1 << (rng.next() % 8);
        if let Ok(file) = gguf::parse(&b) {
            for t in &file.tensors {
                let _ = gguf::mapping::layout(t);
            }
        }
    }
    // A truncated model through the CLI: a typed error, no panic (checked by run_json).
    let f = Fixture::new(COMBOS[1], 94, 1);
    std::fs::write(&f.model, &f.original[..f.original.len() / 2]).unwrap();
    let (code, env) = apply(&f, &f.patch(&WRITERS, "all", 1.0), &[]);
    assert_ne!(code, 0, "{env}");
    assert_eq!(env["error"]["kind"], "invalid_gguf");
}

// ------------------------------------------------------------------ inspect and MCP

#[test]
fn inspect_shows_orientation_and_why_a_tensor_is_not_patchable() {
    let f = with_unsupported_last_target();
    let (code, env) = run_json(
        &[
            "inspect",
            "--model",
            f.model.to_str().unwrap(),
            "--proj",
            "o_proj",
            "--proj",
            "embed_tokens",
        ],
        &[],
    );
    assert_eq!(code, 0, "{env}");
    let tensors = env["data"]["tensors"].as_array().unwrap();
    let get = |name: &str| tensors.iter().find(|t| t["name"] == name).unwrap();
    assert_eq!(get("token_embd.weight")["orientation"], "writer_local");
    assert_eq!(get("token_embd.weight")["layer"], Value::Null);
    assert_eq!(get("blk.0.attn_output.weight")["orientation"], "writer");
    assert_eq!(get("blk.0.attn_output.weight")["patchable"], true);
    let q3 = get("blk.1.attn_output.weight");
    assert_eq!(q3["patchable"], false);
    assert!(q3["patchable_reason"].as_str().unwrap().contains("Q3_K"));
    assert_eq!(env["data"]["tied_embeddings"], false);
    // Default: the MLP only, as for SafeTensors.
    let (_, env) = run_json(&["inspect", "--model", f.model.to_str().unwrap()], &[]);
    assert_eq!(env["data"]["tensors"].as_array().unwrap().len(), 0);
}

fn mcp(dir: &Path, read_only: bool, calls: &[Value]) -> Vec<Value> {
    let mut cmd = surgeon();
    cmd.args(["mcp", "--allow-dir", dir.to_str().unwrap()]);
    if read_only {
        cmd.arg("--read-only");
    }
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut lines = vec![
        json!({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "p15", "version": "0"}}}),
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
    ];
    for (i, args) in calls.iter().enumerate() {
        lines.push(json!({"jsonrpc": "2.0", "id": i + 1, "method": "tools/call", "params": args}));
    }
    let mut stdin = child.stdin.take().unwrap();
    for line in &lines {
        writeln!(stdin, "{line}").unwrap();
    }
    drop(stdin);
    let replies: Vec<Value> = BufReader::new(child.stdout.take().unwrap())
        .lines()
        .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
        .collect();
    child.wait().unwrap();
    (1..=calls.len())
        .map(|id| {
            let reply = replies.iter().find(|r| r["id"] == id).unwrap();
            match reply.get("error") {
                Some(e) => json!({"rpc_error": e}),
                None => reply["result"]["structuredContent"].clone(),
            }
        })
        .collect()
}

#[test]
fn mcp_apply_takes_the_new_options_and_keeps_its_guards() {
    let tied = Types {
        output: false,
        ..COMBOS[2]
    };
    let f = Fixture::new(tied, 95, 1);
    let call = |extra: Value| {
        let mut args =
            json!({"model": "m.gguf", "patch": recipe(&WRITERS, "all", 1.0), "gguf": true});
        for (k, v) in extra.as_object().unwrap() {
            args[k] = v.clone();
        }
        json!({"name": "surgeon_apply", "arguments": args})
    };
    let replies = mcp(
        f.dir.path(),
        false,
        &[
            call(json!({"dry_run": true})),
            call(
                json!({"dry_run": true, "allow_tied_embedding": true, "refine_passes": 2, "residual_target": 0.01}),
            ),
            call(json!({"dry_run": true, "allow_tied_embedding": true, "refine_passes": 9})),
            call(json!({"dry_run": true, "refine": 2})),
        ],
    );
    assert_eq!(
        replies[0]["error"]["kind"], "invalid_input",
        "{}",
        replies[0]
    );
    assert_eq!(replies[1]["ok"], true, "{}", replies[1]);
    let tensors = replies[1]["data"]["gguf"]["tensors"].as_array().unwrap();
    assert!(tensors.iter().all(|t| t["refine_passes"]
        .as_u64()
        .is_some_and(|n| (1..=2).contains(&n))));
    assert_eq!(
        replies[2]["error"]["kind"], "invalid_input",
        "{}",
        replies[2]
    );
    assert!(
        replies[3]["ok"] != true,
        "unknown field refused: {}",
        replies[3]
    );
    assert_eq!(sha(&f.bytes()), sha(&f.original), "dry runs only");
    let replies = mcp(
        f.dir.path(),
        true,
        &[call(json!({"allow_tied_embedding": true}))],
    );
    assert_eq!(replies[0]["error"]["kind"], "read_only");
    assert_eq!(sha(&f.bytes()), sha(&f.original));
}

/// Writes two fake GGUF files with residual writers into `$SURGEON_FAKE_GGUF_DIR`, for
/// manual runs of the binary (`cargo test --test p15_gguf_writers -- --ignored
/// make_writers_gguf_on_disk`): `writers.gguf` (hidden 256, Q5_K / Q4_K / Q6_K, its own
/// output head) and `tied.gguf` (hidden 96, Q8_0 / Q4_1 / Q5_0, no output.weight), the
/// directions `dirs.safetensors` of each (`dirs96.safetensors`) and the full ablation
/// recipe `ablate.json` / `ablate96.json`.
#[test]
#[ignore]
fn make_writers_gguf_on_disk() {
    let dir = PathBuf::from(std::env::var("SURGEON_FAKE_GGUF_DIR").unwrap());
    std::fs::create_dir_all(&dir).unwrap();
    let tied = Types {
        output: false,
        ..COMBOS[2]
    };
    for (model, t, dirs, patch) in [
        ("writers.gguf", COMBOS[0], "dirs.safetensors", "ablate.json"),
        ("tied.gguf", tied, "dirs96.safetensors", "ablate96.json"),
    ] {
        std::fs::write(dir.join(model), writers_gguf(t, 7)).unwrap();
        write_directions(&dir.join(dirs), 1, &orthonormal(1, t.hidden, 107));
        let mut recipe = recipe(&WRITERS, "all", 1.0);
        recipe["operations"][0]["vector_file"] = json!(dirs);
        std::fs::write(dir.join(patch), recipe.to_string()).unwrap();
    }
}
