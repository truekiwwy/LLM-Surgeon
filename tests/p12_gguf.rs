//! P12: in-place GGUF patching (parser, codecs, backend, CLI, MCP) and the chunked I/O of
//! `apply`. Every fixture is built in the tests: a minimal GGUF v3 writer, hand-built
//! blocks with hand-computed values, and a second decoder / requantizer written
//! independently of `patch::gguf::quant` (index formulas instead of unpack loops), the
//! last two shared with p15 in `support/gguf.rs`.

use std::io::{BufRead, BufReader, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use half::f16;
use serde_json::{json, Value};

use weight_surgeon::engine::{self, BroadcastAxis, DType, ShiftOp};
use weight_surgeon::patch::gguf::{
    self, quant, Codec, GgmlType, GgufApplyError, GgufOptions, Requant,
};
use weight_surgeon::patch::{
    apply_patch, parse_patch, validate_patch, Axis, LayerSelection, OpStats, OperationKind,
    PatchBackend, PatchError, Projection, ValidatedOperation, ValidatedPatch, ValidationOptions,
};

// ------------------------------------------------------------------ helpers

#[path = "support/gguf.rs"]
mod support;
use support::*;

const HIDDEN: usize = 256;
const INTER: usize = 512;

/// Qwen2-like MLP: gate/up `ne = [HIDDEN, INTER]`, down `ne = [INTER, HIDDEN]`, plus two
/// untouched tensors (one of an unpatchable type).
fn mlp_gguf(types: [u32; 3], layers: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    let mut tensors = vec![T {
        name: "token_embd.weight".into(),
        ne: vec![HIDDEN as u64, 4],
        ty: F32,
        data: tensor_data(F32, 4, HIDDEN, 1.0, &mut rng),
    }];
    for layer in 0..layers {
        for (proj, ty) in ["gate", "up", "down"].iter().zip(types) {
            let (cols, rows) = match *proj {
                "down" => (INTER, HIDDEN),
                _ => (HIDDEN, INTER),
            };
            tensors.push(T {
                name: format!("blk.{layer}.ffn_{proj}.weight"),
                ne: vec![cols as u64, rows as u64],
                ty,
                data: tensor_data(ty, rows, cols, 0.02, &mut rng),
            });
        }
    }
    tensors.push(T {
        name: "blk.0.attn_q.weight".into(),
        ne: vec![32, 4],
        ty: Q3_K,
        data: vec![0x5A; 4 * 18],
    });
    gguf_bytes(&metadata(), &tensors)
}

// ------------------------------------------------------------------ codecs: known answers

#[test]
fn q8_0_hand_built_block() {
    let mut b = vec![0u8; 34];
    b[..2].copy_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
    let qs: [i8; 6] = [0, 1, -1, 127, -127, -128];
    for (i, q) in qs.iter().enumerate() {
        b[2 + i] = *q as u8;
    }
    b[2 + 31] = 7;
    let mut y = vec![0.0f32; 32];
    quant::dequantize_block(Codec::Q8_0, &b, &mut y);
    assert_eq!(&y[..6], &[0.0, 0.5, -0.5, 63.5, -63.5, -64.0]);
    assert_eq!(y[31], 3.5);
    assert_eq!(y, decode_block(Q8_0, &b));
}

/// d = 1, dmin = 0.25, sc = [1,2,3,4,5,37,7,8], m = [0,2,4,6,8,10,50,14] (37 and 50 use
/// the two high bits stored in bytes 1 and 6), q(e) = e % 16.
#[test]
fn q4_k_hand_built_block() {
    let mut b = vec![0u8; 144];
    b[..2].copy_from_slice(&0x3C00u16.to_le_bytes());
    b[2..4].copy_from_slice(&0x3400u16.to_le_bytes());
    b[4..16].copy_from_slice(&[1, 0x82, 3, 4, 0, 2, 0xC4, 6, 0x85, 0xA5, 0x27, 0xE8]);
    for c in 0..4 {
        for i in 0..32 {
            let (lo, hi) = ((64 * c + i) % 16, (64 * c + 32 + i) % 16);
            b[16 + 32 * c + i] = (lo | (hi << 4)) as u8;
        }
    }
    let sc = [1.0, 2.0, 3.0, 4.0, 5.0, 37.0, 7.0, 8.0f32];
    let m = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 50.0, 14.0f32];
    let mut y = vec![0.0f32; 256];
    quant::dequantize_block(Codec::Q4K, &b, &mut y);
    for (e, &v) in y.iter().enumerate() {
        let want = sc[e / 32] * (e % 16) as f32 - 0.25 * m[e / 32];
        assert_eq!(v, want, "element {e}");
    }
    assert_eq!(
        [y[0], y[33], y[170], y[200], y[255]],
        [0.0, 1.5, 367.5, 43.5, 116.5]
    );
    assert_eq!(y, decode_block(Q4_K, &b));
    for j in 0..8 {
        assert_eq!(
            quant::scale_min_k4(j, &b[4..16]),
            (sc[j] as u8, m[j] as u8),
            "sub-block {j}"
        );
    }
}

/// d = 0.5, scales[j] = j - 8 (negative ones included), q(e) = 7e mod 64, value q - 32.
#[test]
fn q6_k_hand_built_block() {
    let mut b = vec![0u8; 210];
    for e in 0..256 {
        let q = ((7 * e) % 64) as u8;
        let ((lo_at, lo_shift), (hi_at, hi_shift)) = (q6k_low(e), q6k_high(e));
        b[lo_at] |= (q & 15) << lo_shift;
        b[hi_at] |= (q >> 4) << hi_shift;
    }
    for j in 0..16 {
        b[192 + j] = (j as i8 - 8) as u8;
    }
    b[208..210].copy_from_slice(&0x3800u16.to_le_bytes());
    let mut y = vec![0.0f32; 256];
    quant::dequantize_block(Codec::Q6K, &b, &mut y);
    for (e, &v) in y.iter().enumerate() {
        let want = 0.5 * (e / 16) as f32 - 4.0;
        let want = want * (((7 * e) % 64) as f32 - 32.0);
        assert_eq!(v, want, "element {e}");
    }
    assert_eq!([y[0], y[1], y[255]], [128.0, 100.0, 87.5]);
    assert_eq!(y, decode_block(Q6_K, &b));
}

#[test]
fn library_decoder_matches_the_independent_one_on_random_blocks() {
    let mut rng = Rng(21);
    for ty in [Q8_0, Q4_K, Q6_K] {
        let c = codec(ty);
        for _ in 0..300 {
            let mut b: Vec<u8> = (0..c.block_bytes()).map(|_| rng.next() as u8).collect();
            // Finite f16 scales (random bits may be NaN/Inf).
            for at in match ty {
                Q8_0 => vec![0],
                Q4_K => vec![0, 2],
                _ => vec![208],
            } {
                let h = f16::from_f32(rng.normal() * 0.01);
                b[at..at + 2].copy_from_slice(&h.to_le_bytes());
            }
            let mut y = vec![0.0f32; c.block_elems()];
            quant::dequantize_block(c, &b, &mut y);
            let want = decode_block(ty, &b);
            let same = y.iter().zip(&want).all(|(a, w)| a.to_bits() == w.to_bits());
            assert!(same, "type {ty}");
        }
    }
}

/// Fresh quantization error is bounded by half a step of the effective grid: Q8_0 adds the
/// f16 rounding of d (reference algorithm), Q4_K / Q6_K round their scales up (no clip).
#[test]
fn quantize_dequantize_round_trip_within_half_a_step() {
    let mut rng = Rng(5);
    for ty in [Q8_0, Q4_K, Q6_K] {
        let c = codec(ty);
        for trial in 0..200 {
            let scale = [0.02f32, 1.0, 1e-4][trial % 3];
            let x: Vec<f32> = (0..c.block_elems())
                .map(|i| match (trial, i) {
                    (t, 0) if t % 7 == 0 => 0.0,
                    _ => scale * rng.normal() + if trial % 5 == 0 { 3.0 * scale } else { 0.0 },
                })
                .collect();
            let mut b = vec![0u8; c.block_bytes()];
            quant::quantize_block(c, &x, &mut b).unwrap();
            let y = decode_block(ty, &b);
            for (e, (&xe, &ye)) in x.iter().zip(&y).enumerate() {
                let bound = match ty {
                    Q8_0 => {
                        let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                        let d = amax / 127.0;
                        d / 2.0 + 127.0 * (d - f16::from_f32(d).to_f32()).abs()
                    }
                    Q4_K => f16v(&b, 0) * q4k_scales(&b[4..16], e / 32).0 / 2.0,
                    _ => f16v(&b, 208) * f32::from(b[192 + e / 16] as i8) / 2.0,
                };
                let err = (xe - ye).abs();
                assert!(
                    err <= bound * (1.0 + 1e-4) + 1e-7 * scale,
                    "type {ty} trial {trial} e {e}: |{xe} - {ye}| = {err} > {bound}"
                );
            }
        }
    }
}

#[test]
fn recomputed_scale_overflow_is_detected() {
    let x = vec![1e7f32; 32];
    let mut b = vec![0u8; 34];
    assert!(quant::quantize_block(Codec::Q8_0, &x, &mut b).is_err());
    let x = vec![-3e7f32; 256];
    let mut b = vec![0u8; 144];
    assert!(quant::quantize_block(Codec::Q4K, &x, &mut b).is_err());
}

// ------------------------------------------------------------------ parser

#[test]
fn parser_reads_the_writer_layout() {
    let bytes = mlp_gguf([Q4_K, Q8_0, Q6_K], 2, 1);
    let file = gguf::parse(&bytes).unwrap();
    assert_eq!(file.version, 3);
    assert_eq!(file.alignment, 32);
    assert_eq!(file.architecture(), Some("qwen2"));
    assert_eq!(file.tensors.len(), 8);
    let down = file.tensor("blk.1.ffn_down.weight").unwrap();
    assert_eq!(down.ne, vec![INTER as u64, HIDDEN as u64]);
    assert_eq!(down.ggml_type, GgmlType::Q6_K);
    assert_eq!(down.size, Some((HIDDEN * INTER / 256 * 210) as u64));
    assert_eq!(file.tensor("blk.0.attn_q.weight").unwrap().size, None);
    assert!(file.data_start.is_multiple_of(32));
    match file.meta("tokenizer.ggml.tokens").unwrap() {
        gguf::parse::MetaValue::Array { len, head, .. } => {
            assert_eq!(*len, 40);
            assert_eq!(head.len(), gguf::parse::ARRAY_HEAD);
        }
        other => panic!("{other:?}"),
    }
    let found = gguf::mapping::targets(
        &file,
        &LayerSelection::List(vec![1]),
        &[Projection::Down, Projection::Gate],
    );
    let names: Vec<&str> = found.iter().map(|(_, _, t)| t.name.as_str()).collect();
    assert_eq!(names, ["blk.1.ffn_gate.weight", "blk.1.ffn_down.weight"]);
}

#[test]
fn layout_errors_name_the_tensor_and_its_type() {
    let tensors = [
        T {
            name: "blk.0.ffn_gate.weight".into(),
            ne: vec![256, 2],
            ty: Q3_K,
            data: vec![0; 2 * 110],
        },
        T {
            // 512 elements: whole Q8_0 blocks overall, but rows of 16 split a block.
            name: "blk.0.ffn_up.weight".into(),
            ne: vec![16, 32],
            ty: Q8_0,
            data: vec![0; 16 * 34],
        },
    ];
    let file = gguf::parse(&gguf_bytes(&metadata(), &tensors)).unwrap();
    let err = gguf::mapping::layout(file.tensor("blk.0.ffn_gate.weight").unwrap()).unwrap_err();
    let text = err.to_string();
    assert!(
        text.contains("unsupported_quant_type") && text.contains("Q3_K"),
        "{text}"
    );
    assert!(text.contains("blk.0.ffn_gate.weight"), "{text}");
    let err = gguf::mapping::layout(file.tensor("blk.0.ffn_up.weight").unwrap()).unwrap_err();
    assert!(err.to_string().contains("multiple du bloc"), "{err}");
}

#[test]
fn parser_refuses_split_big_endian_and_bad_offsets() {
    let mut kv = metadata();
    kv.push(("split.count", Kv::U32(3)));
    assert!(matches!(
        gguf::parse(&gguf_bytes(&kv, &[])),
        Err(gguf::GgufError::Split(_))
    ));
    assert!(gguf::parse::is_split_name("qwen-00001-of-00003.gguf"));
    assert!(!gguf::parse::is_split_name("qwen-q4_k_m.gguf"));
    let mut be = gguf_bytes(&metadata(), &[]);
    be[4..8].copy_from_slice(&3u32.to_be_bytes());
    assert_eq!(gguf::parse(&be), Err(gguf::GgufError::BigEndian));
    let t = T {
        name: "blk.0.ffn_up.weight".into(),
        ne: vec![32, 1],
        ty: Q8_0,
        data: vec![0; 34],
    };
    let mut bytes = gguf_bytes(&metadata(), &[t]);
    let len = bytes.len();
    // Offset field of the only tensor: 8 bytes before the data padding.
    let file = gguf::parse(&bytes).unwrap();
    let off_at = (0..file.data_start as usize - 8)
        .rev()
        .find(|&i| bytes[i..i + 8] == 0u64.to_le_bytes() && bytes[i - 4..i] == Q8_0.to_le_bytes())
        .unwrap();
    bytes[off_at..off_at + 8].copy_from_slice(&(len as u64 * 4).next_multiple_of(32).to_le_bytes());
    assert!(matches!(
        gguf::parse(&bytes),
        Err(gguf::GgufError::Truncated { .. })
    ));
    bytes[off_at..off_at + 8].copy_from_slice(&7u64.to_le_bytes());
    assert!(matches!(
        gguf::parse(&bytes),
        Err(gguf::GgufError::Tensor { .. })
    ));
}

// ------------------------------------------------------------------ files and the binary

struct Fixture {
    dir: tempfile::TempDir,
    model: PathBuf,
    original: Vec<u8>,
}

impl Fixture {
    fn new(bytes: Vec<u8>) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("m.gguf");
        std::fs::write(&model, &bytes).unwrap();
        Self {
            dir,
            model,
            original: bytes,
        }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    fn vector(&self, name: &str, values: &[f32]) {
        std::fs::write(self.path(name), serde_json::to_string(values).unwrap()).unwrap();
    }

    fn patch(&self, ops: Value) -> PathBuf {
        let path = self.path("patch.json");
        let patch = json!({"schema_version": 1, "backup": true, "operations": ops});
        std::fs::write(&path, patch.to_string()).unwrap();
        path
    }

    fn files(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(self.dir.path())
            .unwrap()
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .collect();
        names.sort();
        names
    }
}

fn op(id: &str, layers: &str, proj: &str, vector: &str, axis: &str, alpha: f32) -> Value {
    json!({"id": id, "layers": layers, "projections": [proj], "operation": "subtract",
           "vector_file": vector, "alpha": alpha, "axis": axis})
}

fn surgeon() -> Command {
    Command::new(env!("CARGO_BIN_EXE_surgeon"))
}

/// `surgeon --json <args>`: exit code and the single JSON line of stdout.
fn run_json(args: &[&str], env: &[(&str, &str)]) -> (i32, Value) {
    let out = surgeon()
        .arg("--json")
        .args(args)
        .envs(env.iter().copied())
        .env("CLICOLOR_FORCE", "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 1, "{stdout}");
    assert!(!stdout.contains('\x1b'), "{stdout}");
    (
        out.status.code().unwrap(),
        serde_json::from_str(&stdout).unwrap(),
    )
}

fn apply_gguf_cli(f: &Fixture, patch: &Path, extra: &[&str]) -> (i32, Value) {
    let mut args = vec![
        "apply",
        "--gguf",
        "--model",
        f.model.to_str().unwrap(),
        "--patch",
        patch.to_str().unwrap(),
    ];
    args.extend_from_slice(extra);
    run_json(&args, &[])
}

/// Expected file after a keep-scales patch, computed with the independent decoder and
/// requantizer: `requant(dequant(block) - alpha * v)` on every target, nothing else.
fn expected_keep_scales(
    original: &[u8],
    targets: &[(&str, Axis, &[f32], f32)],
) -> (Vec<u8>, Vec<u64>) {
    let file = gguf::parse(original).unwrap();
    let mut out = original.to_vec();
    let mut saturated = Vec::new();
    for (name, axis, v, alpha) in targets {
        let t = file.tensor(name).unwrap();
        let (ty, cols) = (t.ggml_type.0, t.ne[0] as usize);
        let c = codec(ty);
        let start = t.start as usize;
        let row_bytes = c.row_bytes(cols);
        let mut sat = 0;
        for r in 0..t.ne[1] as usize {
            let row = &mut out[start + r * row_bytes..start + (r + 1) * row_bytes];
            for (k, b) in row.chunks_mut(c.block_bytes()).enumerate() {
                let before = decode_block(ty, b);
                let exact: Vec<f32> = before
                    .iter()
                    .enumerate()
                    .map(|(i, &w)| {
                        let vi = match axis {
                            Axis::Rows => v[k * c.block_elems() + i],
                            Axis::Cols => v[r],
                        };
                        w + -(alpha * vi)
                    })
                    .collect();
                sat += requant_keep_scales(ty, b, &before, &exact);
            }
        }
        saturated.push(sat);
    }
    (out, saturated)
}

fn hidden_vector(seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..HIDDEN).map(|_| scale * rng.normal()).collect()
}

#[test]
fn keep_scales_patch_equals_the_independent_reference_byte_for_byte() {
    let f = Fixture::new(mlp_gguf([Q4_K, Q8_0, Q6_K], 2, 7));
    let v = hidden_vector(3, 1.0);
    f.vector("v.json", &v);
    let patch = f.patch(json!([
        op("gate", "0-1", "gate_proj", "v.json", "rows", 0.004),
        op("up", "1", "up_proj", "v.json", "rows", 0.002),
        op("down", "0", "down_proj", "v.json", "cols", 0.006),
    ]));
    let (code, env) = apply_gguf_cli(&f, &patch, &[]);
    assert_eq!(code, 0, "{env}");
    let targets: Vec<(&str, Axis, &[f32], f32)> = vec![
        ("blk.0.ffn_gate.weight", Axis::Rows, &v, 0.004),
        ("blk.1.ffn_gate.weight", Axis::Rows, &v, 0.004),
        ("blk.1.ffn_up.weight", Axis::Rows, &v, 0.002),
        ("blk.0.ffn_down.weight", Axis::Cols, &v, 0.006),
    ];
    let (want, saturated) = expected_keep_scales(&f.original, &targets);
    let got = std::fs::read(&f.model).unwrap();
    assert!(
        got == want,
        "patched file differs from the independent reference"
    );
    // Header region and untouched tensors: hashed, unchanged (covered by the byte equality,
    // checked again explicitly).
    let file = gguf::parse(&f.original).unwrap();
    let head = file.data_start as usize;
    assert_eq!(sha(&got[..head]), sha(&f.original[..head]));
    assert_eq!(
        env["data"]["gguf"]["header_sha256"],
        sha(&f.original[..head])
    );
    for name in [
        "token_embd.weight",
        "blk.0.ffn_up.weight",
        "blk.1.ffn_down.weight",
    ] {
        let t = file.tensor(name).unwrap();
        let range = t.start as usize..(t.start + t.size.unwrap_or(72)) as usize;
        assert_eq!(sha(&got[range.clone()]), sha(&f.original[range]), "{name}");
    }
    let gguf = &env["data"]["gguf"];
    assert_eq!(gguf["verified"], true);
    assert_eq!(gguf["requant"], "keep-scales");
    let reported: Vec<u64> = gguf["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["saturated_elements"].as_u64().unwrap())
        .collect();
    assert_eq!(reported, saturated);
    assert!(!f.path("m.gguf.surgeon-inprogress").exists());
    let backups: Vec<String> = f
        .files()
        .into_iter()
        .filter(|n| n.contains(".bak-"))
        .collect();
    assert_eq!(backups.len(), 1, "{:?}", f.files());
    assert_eq!(std::fs::read(f.path(&backups[0])).unwrap(), f.original);
    let io = &env["data"]["io"];
    assert!(io["chunks"].as_u64().unwrap() >= 4, "{io}");
    assert_eq!(io["kernel"]["level"], "scalar");
}

#[test]
fn metrics_match_a_brute_force_computation() {
    let f = Fixture::new(mlp_gguf([Q4_K, Q8_0, Q6_K], 1, 8));
    let v = hidden_vector(4, 1.0);
    f.vector("v.json", &v);
    let alpha = 0.003;
    let patch = f.patch(json!([op(
        "down",
        "0",
        "down_proj",
        "v.json",
        "cols",
        alpha
    )]));
    let (code, env) = apply_gguf_cli(&f, &patch, &[]);
    assert_eq!(code, 0, "{env}");
    let file = gguf::parse(&f.original).unwrap();
    let t = file.tensor("blk.0.ffn_down.weight").unwrap();
    let range = t.start as usize..(t.start + t.size.unwrap()) as usize;
    let before = decode_tensor(Q6_K, &f.original[range.clone()]);
    let after = decode_tensor(Q6_K, &std::fs::read(&f.model).unwrap()[range]);
    let (mut i2, mut r2, mut dot, mut e2, mut x2, mut same) = (0.0, 0.0, 0.0, 0.0, 0.0, 0u64);
    for (k, (&b, &a)) in before.iter().zip(&after).enumerate() {
        let d = f64::from(-(alpha * v[k / INTER]));
        let r = f64::from(a) - f64::from(b);
        let x = f64::from(b) + d;
        i2 += d * d;
        r2 += r * r;
        dot += d * r;
        e2 += (f64::from(a) - x).powi(2);
        x2 += x * x;
        same += u64::from(a.to_bits() == b.to_bits());
    }
    let m = &env["data"]["gguf"]["tensors"][0];
    let close = |key: &str, want: f64| {
        let got = m[key].as_f64().unwrap();
        assert!(
            (got - want).abs() <= 1e-9 * want.abs().max(1e-12),
            "{key}: {got} vs {want}"
        );
    };
    close("intended_shift_norm", i2.sqrt());
    close("realized_shift_norm", r2.sqrt());
    close("realized_over_intended", r2.sqrt() / i2.sqrt());
    close(
        "correlation_intended_realized",
        dot / (i2.sqrt() * r2.sqrt()),
    );
    close("nmse_vs_exact", e2 / x2);
    close("unchanged_fraction", same as f64 / before.len() as f64);
    assert_eq!(m["elements"], (HIDDEN * INTER) as u64);
    assert_eq!(m["ggml_type"], "Q6_K");
}

/// Self-calibrated fidelity against a fresh quantization of the exact patched weights
/// (N(0, 0.02²) weights, v ~ N(0, 1)), for each format:
/// - `recompute` stays within 2x the fresh error at any strength;
/// - `keep-scales` stays within 2x while the shift is about one step (alpha 3e-4); a strong
///   shift (alpha 0.01) is clipped at the original grid bounds (Q8_0 has no headroom): the
///   error is then far above 2x, which the report must say (saturation + warning).
///
/// Also the absolute bound of the brief: MSE < 1e-4.
#[test]
fn fidelity_is_within_twice_a_fresh_quantization() {
    for (ty, proj, axis) in [
        (Q4_K, "gate_proj", Axis::Rows),
        (Q8_0, "up_proj", Axis::Rows),
        (Q6_K, "down_proj", Axis::Cols),
    ] {
        for (requant, alpha) in [
            ("keep-scales", 3e-4f32),
            ("recompute", 3e-4),
            ("recompute", 0.01),
            ("keep-scales", 0.01),
        ] {
            let f = Fixture::new(mlp_gguf([Q4_K, Q8_0, Q6_K], 1, 11));
            let v = hidden_vector(9, 1.0);
            f.vector("v.json", &v);
            let axis_name = if axis == Axis::Rows { "rows" } else { "cols" };
            let patch = f.patch(json!([op("x", "0", proj, "v.json", axis_name, alpha)]));
            let (code, env) = apply_gguf_cli(&f, &patch, &["--requant", requant]);
            assert_eq!(code, 0, "{env}");
            let name = format!("blk.0.ffn_{}.weight", proj.trim_end_matches("_proj"));
            let file = gguf::parse(&f.original).unwrap();
            let t = file.tensor(&name).unwrap();
            let c = codec(ty);
            let cols = t.ne[0] as usize;
            let range = t.start as usize..(t.start + t.size.unwrap()) as usize;
            let before = decode_tensor(ty, &f.original[range.clone()]);
            let after = decode_tensor(ty, &std::fs::read(&f.model).unwrap()[range]);
            let delta = |k: usize| {
                -(alpha
                    * if axis == Axis::Rows {
                        v[k % cols]
                    } else {
                        v[k / cols]
                    })
            };
            // f32 exact: what gets requantized; f64 exact: what the report measures against.
            let exact: Vec<f32> = before
                .iter()
                .enumerate()
                .map(|(k, &w)| w + delta(k))
                .collect();
            let exact64: Vec<f64> = before
                .iter()
                .enumerate()
                .map(|(k, &w)| f64::from(w) + f64::from(delta(k)))
                .collect();
            let mut fresh = vec![0u8; c.block_bytes()];
            let mut fresh_err = 0.0f64;
            for (x, x64) in exact
                .chunks(c.block_elems())
                .zip(exact64.chunks(c.block_elems()))
            {
                quant::quantize_block(c, x, &mut fresh).unwrap();
                for (a, b) in decode_block(ty, &fresh).iter().zip(x64) {
                    fresh_err += (f64::from(*a) - b).powi(2);
                }
            }
            let norm: f64 = exact64.iter().map(|x| x * x).sum();
            let err: f64 = after
                .iter()
                .zip(&exact64)
                .map(|(a, x)| (f64::from(*a) - x).powi(2))
                .sum();
            let (nmse, nmse_fresh) = (err / norm, fresh_err / norm);
            let report = &env["data"]["gguf"]["tensors"][0];
            let reported = report["nmse_vs_exact"].as_f64().unwrap();
            assert!((reported - nmse).abs() <= 1e-9 * nmse, "{ty} {requant}");
            assert!(err / exact.len() as f64 <= 1e-4, "absolute MSE, type {ty}");
            let label = format!("type {ty} {requant} alpha {alpha}");
            if requant == "keep-scales" && alpha > 1e-3 {
                if ty == Q8_0 {
                    assert!(nmse > 2.0 * nmse_fresh, "{label}: clipping expected");
                }
                if nmse > 2.0 * nmse_fresh {
                    assert!(
                        report["saturated_elements"].as_u64().unwrap() > 0,
                        "{label}"
                    );
                    let warning = report["warning"].as_str().unwrap_or_default();
                    assert!(warning.contains("--requant recompute"), "{label}: {report}");
                }
            } else {
                assert!(
                    nmse <= 2.0 * nmse_fresh,
                    "{label}: nmse {nmse:.3e} > 2 x fresh {nmse_fresh:.3e}"
                );
            }
            if requant == "recompute" && ty == Q8_0 {
                // Q8_0 recompute is quantize_row_q8_0_ref itself.
                assert!((nmse - nmse_fresh).abs() <= 1e-12, "{nmse} vs {nmse_fresh}");
            }
        }
    }
}

#[test]
fn a_shift_far_below_the_step_is_reported_lost_with_a_warning() {
    let f = Fixture::new(mlp_gguf([Q4_K, Q8_0, Q6_K], 1, 12));
    f.vector("v.json", &hidden_vector(2, 1.0));
    let patch = f.patch(json!([op(
        "tiny",
        "0",
        "gate_proj",
        "v.json",
        "rows",
        1e-5
    )]));
    let (code, env) = apply_gguf_cli(&f, &patch, &[]);
    assert_eq!(code, 0, "{env}");
    let t = &env["data"]["gguf"]["tensors"][0];
    assert!(t["realized_over_intended"].as_f64().unwrap() < 0.5, "{t}");
    assert!(t["warning"].as_str().unwrap().contains("demi-pas"), "{t}");
    let warnings = env["warnings"].as_array().unwrap();
    assert!(warnings
        .iter()
        .any(|w| w.as_str().unwrap().contains("blk.0.ffn_gate.weight")));
}

#[test]
fn saturation_is_counted_and_alpha_zero_leaves_the_file_identical() {
    let f = Fixture::new(mlp_gguf([Q4_K, Q8_0, Q6_K], 1, 13));
    f.vector("v.json", &vec![1.0; HIDDEN]);
    let patch = f.patch(json!([op("zero", "0", "gate_proj", "v.json", "rows", 0.0)]));
    let (code, env) = apply_gguf_cli(&f, &patch, &[]);
    assert_eq!(code, 0, "{env}");
    assert!(
        std::fs::read(&f.model).unwrap() == f.original,
        "alpha 0 changed bytes"
    );
    assert_eq!(env["data"]["gguf"]["tensors"][0]["unchanged_fraction"], 1.0);
    // -alpha * 1 = +0.5: far beyond the top of the Q4_K grid of N(0, 0.02²) weights.
    let patch = f.patch(json!([op("sat", "0", "gate_proj", "v.json", "rows", -0.5)]));
    let (code, env) = apply_gguf_cli(&f, &patch, &[]);
    assert_eq!(code, 0, "{env}");
    let saturated = env["data"]["gguf"]["tensors"][0]["saturated_elements"]
        .as_u64()
        .unwrap();
    let (_, want) = expected_keep_scales(
        &f.original,
        &[("blk.0.ffn_gate.weight", Axis::Rows, &[1.0; HIDDEN], -0.5)],
    );
    assert_eq!(saturated, want[0]);
    assert!(saturated > (HIDDEN * INTER / 2) as u64, "{saturated}");
}

#[test]
fn float_targets_match_the_safetensors_engine() {
    for (ty, dtype) in [(F16, DType::F16), (BF16, DType::BF16), (F32, DType::F32)] {
        let f = Fixture::new(mlp_gguf([ty, ty, ty], 1, 14));
        let v = hidden_vector(6, 0.3);
        f.vector("v.json", &v);
        let patch = f.patch(json!([op("g", "0", "gate_proj", "v.json", "rows", 0.7)]));
        let (code, env) = apply_gguf_cli(&f, &patch, &["--prefetch-chunk-mib", "1"]);
        assert_eq!(code, 0, "{env}");
        let file = gguf::parse(&f.original).unwrap();
        let t = file.tensor("blk.0.ffn_gate.weight").unwrap();
        let range = t.start as usize..(t.start + t.size.unwrap()) as usize;
        let mut want = f.original[range.clone()].to_vec();
        engine::apply_vector_shift(
            &mut want,
            dtype,
            &[INTER, HIDDEN],
            &v,
            0.7,
            ShiftOp::Subtract,
            BroadcastAxis::Rows,
        )
        .unwrap();
        assert!(
            std::fs::read(&f.model).unwrap()[range] == want[..],
            "type {ty}"
        );
    }
}

#[test]
fn chunk_size_and_prefetch_do_not_change_the_bytes() {
    let mut outputs = Vec::new();
    for extra in [
        &[][..],
        &["--prefetch-chunk-mib", "1"][..],
        &["--no-prefetch"][..],
    ] {
        let f = Fixture::new(mlp_gguf([Q4_K, Q8_0, Q6_K], 1, 15));
        f.vector("v.json", &hidden_vector(1, 1.0));
        let patch = f.patch(json!([op("d", "0", "down_proj", "v.json", "cols", 0.01)]));
        let (code, env) = apply_gguf_cli(&f, &patch, extra);
        assert_eq!(code, 0, "{env}");
        if extra.contains(&"--no-prefetch") {
            assert_eq!(env["data"]["io"]["prefetch_requests"], 0);
        }
        outputs.push(std::fs::read(&f.model).unwrap());
    }
    assert!(outputs.windows(2).all(|w| w[0] == w[1]));
}

#[test]
fn dry_run_computes_every_metric_and_writes_nothing() {
    let f = Fixture::new(mlp_gguf([Q4_K, Q8_0, Q6_K], 1, 16));
    f.vector("v.json", &hidden_vector(1, 1.0));
    let patch = f.patch(json!([op("d", "0", "down_proj", "v.json", "cols", 0.01)]));
    let before = f.files();
    let (code, env) = apply_gguf_cli(&f, &patch, &["--dry-run"]);
    assert_eq!(code, 0, "{env}");
    assert!(std::fs::read(&f.model).unwrap() == f.original);
    assert_eq!(f.files(), before, "no backup, no marker");
    assert_eq!(env["data"]["report"]["dry_run"], true);
    assert!(env["data"]["io"].is_null());
    let t = &env["data"]["gguf"]["tensors"][0];
    assert!(t["realized_shift_norm"].as_f64().unwrap() > 0.0, "{t}");
    // The same metrics as a real run.
    let (_, real) = apply_gguf_cli(&f, &patch, &[]);
    assert_eq!(
        real["data"]["gguf"]["tensors"],
        env["data"]["gguf"]["tensors"]
    );
}

#[test]
fn usage_errors_are_explicit() {
    let f = Fixture::new(mlp_gguf([Q4_K, Q8_0, Q6_K], 1, 17));
    f.vector("v.json", &hidden_vector(1, 1.0));
    let patch = f.patch(json!([op("d", "0", "down_proj", "v.json", "cols", 0.01)]));
    let (model, patch) = (f.model.to_str().unwrap(), patch.to_str().unwrap());
    // A .gguf without --gguf: never a lossy operation by surprise.
    let (code, env) = run_json(&["apply", "--model", model, "--patch", patch], &[]);
    assert_eq!(
        (code, env["error"]["kind"].as_str()),
        (2, Some("invalid_input"))
    );
    assert!(env["error"]["message"].as_str().unwrap().contains("--gguf"));
    let (code, env) = run_json(
        &[
            "apply",
            "--gguf",
            "--no-backup",
            "--model",
            model,
            "--patch",
            patch,
        ],
        &[],
    );
    assert_eq!(code, 2, "{env}");
    assert!(env["error"]["message"]
        .as_str()
        .unwrap()
        .contains("--force-no-backup"));
    // Wrong vector length: refused before any backup or write.
    f.vector("short.json", &[1.0; 10]);
    let short = f.patch(json!([op(
        "d",
        "0",
        "down_proj",
        "short.json",
        "cols",
        0.01
    )]));
    let (code, env) = apply_gguf_cli(&f, &short, &[]);
    assert_eq!(code, 2, "{env}");
    assert!(std::fs::read(&f.model).unwrap() == f.original);
    assert!(!f.files().iter().any(|n| n.contains(".bak-")));
    // Unsupported type on a target (Q3_K: 110 bytes per block of 256).
    let bytes = gguf_bytes(
        &metadata(),
        &[T {
            name: "blk.0.ffn_gate.weight".into(),
            ne: vec![HIDDEN as u64, 2],
            ty: Q3_K,
            data: vec![0; 2 * HIDDEN / 256 * 110],
        }],
    );
    let q4 = Fixture::new(bytes);
    q4.vector("v.json", &hidden_vector(1, 1.0));
    let p = q4.patch(json!([op("g", "0", "gate_proj", "v.json", "rows", 0.01)]));
    let (code, env) = apply_gguf_cli(&q4, &p, &[]);
    assert_eq!(code, 2, "{env}");
    assert_eq!(env["error"]["kind"], "unsupported_quant_type");
    assert!(env["error"]["message"].as_str().unwrap().contains("Q3_K"));
}

#[test]
fn force_no_backup_skips_the_copy() {
    let f = Fixture::new(mlp_gguf([Q4_K, Q8_0, Q6_K], 1, 18));
    f.vector("v.json", &hidden_vector(1, 1.0));
    let p = f.path("p.json");
    let patch = json!({"schema_version": 1, "backup": false,
        "operations": [op("d", "0", "down_proj", "v.json", "cols", 0.01)]});
    std::fs::write(&p, patch.to_string()).unwrap();
    let (code, env) = apply_gguf_cli(&f, &p, &["--force-no-backup"]);
    assert_eq!(code, 0, "{env}");
    assert!(
        !f.files().iter().any(|n| n.contains(".bak-")),
        "{:?}",
        f.files()
    );
    assert!(std::fs::read(&f.model).unwrap() != f.original);
}

#[test]
fn an_existing_crash_marker_blocks_the_apply() {
    let f = Fixture::new(mlp_gguf([Q4_K, Q8_0, Q6_K], 1, 19));
    f.vector("v.json", &hidden_vector(1, 1.0));
    let patch = f.patch(json!([op("d", "0", "down_proj", "v.json", "cols", 0.01)]));
    std::fs::write(f.path("m.gguf.surgeon-inprogress"), "{}").unwrap();
    let (code, env) = apply_gguf_cli(&f, &patch, &[]);
    assert_eq!(
        (code, env["error"]["kind"].as_str()),
        (1, Some("apply_in_progress"))
    );
    assert!(env["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Restaurez le backup"));
    assert!(std::fs::read(&f.model).unwrap() == f.original);
}

#[test]
fn an_overflowing_patch_rolls_back_through_the_backup() {
    // F16 target: 65504 is the largest finite value, so -alpha * v = +1e6 overflows.
    let f = Fixture::new(mlp_gguf([F16, F16, F16], 1, 20));
    f.vector("v.json", &vec![-1e6; HIDDEN]);
    let patch = f.patch(json!([op("boom", "0", "gate_proj", "v.json", "rows", 1.0)]));
    let (code, env) = apply_gguf_cli(&f, &patch, &[]);
    assert_eq!(code, 3, "{env}");
    let message = env["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("NaN/Inf") && message.contains("rollback : true"),
        "{message}"
    );
    assert!(std::fs::read(&f.model).unwrap() == f.original);
    assert!(!f.path("m.gguf.surgeon-inprogress").exists());
    // Recompute: a block scale beyond f16 is refused the same way.
    let q = Fixture::new(mlp_gguf([Q8_0, Q8_0, Q8_0], 1, 22));
    q.vector("v.json", &vec![-1e7; HIDDEN]);
    let patch = q.patch(json!([op("big", "0", "gate_proj", "v.json", "rows", 1.0)]));
    let (code, env) = apply_gguf_cli(&q, &patch, &["--requant", "recompute"]);
    assert_eq!(code, 3, "{env}");
    assert!(
        env["error"]["message"].as_str().unwrap().contains("f16"),
        "{env}"
    );
    assert!(std::fs::read(&q.model).unwrap() == q.original);
}

#[test]
fn a_non_finite_scale_in_the_original_is_refused_before_writing() {
    let mut bytes = mlp_gguf([Q8_0, Q8_0, Q8_0], 1, 23);
    let file = gguf::parse(&bytes).unwrap();
    let t = file.tensor("blk.0.ffn_gate.weight").unwrap().start as usize;
    bytes[t + 34 * 5..t + 34 * 5 + 2].copy_from_slice(&0x7E00u16.to_le_bytes()); // NaN d
    let f = Fixture::new(bytes);
    f.vector("v.json", &hidden_vector(1, 1.0));
    let patch = f.patch(json!([op("g", "0", "gate_proj", "v.json", "rows", 0.01)]));
    let (code, env) = apply_gguf_cli(&f, &patch, &[]);
    assert_eq!(code, 3, "{env}");
    assert!(
        env["error"]["message"]
            .as_str()
            .unwrap()
            .contains("non finie"),
        "{env}"
    );
    assert!(std::fs::read(&f.model).unwrap() == f.original);
}

#[test]
fn split_names_and_non_gguf_files_are_refused() {
    let f = Fixture::new(mlp_gguf([Q4_K, Q8_0, Q6_K], 1, 24));
    f.vector("v.json", &hidden_vector(1, 1.0));
    let patch = f.patch(json!([op("d", "0", "down_proj", "v.json", "cols", 0.01)]));
    let split = f.path("m-00001-of-00002.gguf");
    std::fs::copy(&f.model, &split).unwrap();
    let (code, env) = run_json(
        &[
            "apply",
            "--gguf",
            "--model",
            split.to_str().unwrap(),
            "--patch",
            patch.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(
        (code, env["error"]["kind"].as_str()),
        (1, Some("invalid_gguf"))
    );
    let not_gguf = f.path("x.bin");
    std::fs::write(&not_gguf, b"not a model").unwrap();
    let (code, env) = run_json(
        &[
            "apply",
            "--gguf",
            "--model",
            not_gguf.to_str().unwrap(),
            "--patch",
            patch.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 2, "{env}");
}

#[test]
fn inspect_lists_mlp_tensors_with_their_quantization_type() {
    let f = Fixture::new(mlp_gguf([Q4_K, Q8_0, Q6_K], 2, 25));
    let (code, env) = run_json(
        &[
            "inspect",
            "--model",
            f.model.to_str().unwrap(),
            "--layers",
            "1",
        ],
        &[],
    );
    assert_eq!(code, 0, "{env}");
    assert_eq!(env["data"]["format"], "gguf");
    let tensors = env["data"]["tensors"].as_array().unwrap();
    let types: Vec<&str> = tensors
        .iter()
        .map(|t| t["ggml_type"].as_str().unwrap())
        .collect();
    assert_eq!(types, ["Q4_K", "Q8_0", "Q6_K"]);
    assert_eq!(tensors[2]["shape"], json!([HIDDEN, INTER]));
    assert!(tensors.iter().all(|t| t["patchable"] == true));
    // Human mode renders the table.
    let out = surgeon()
        .args(["inspect", "--model", f.model.to_str().unwrap()])
        .output()
        .unwrap();
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(
        text.contains("Q6_K") && text.contains("[INSPECT]"),
        "{text}"
    );
}

#[test]
fn human_output_shows_fidelity_and_verification() {
    let f = Fixture::new(mlp_gguf([Q4_K, Q8_0, Q6_K], 1, 26));
    f.vector("v.json", &hidden_vector(1, 1.0));
    let patch = f.patch(json!([op("d", "0", "down_proj", "v.json", "cols", 0.01)]));
    let out = surgeon()
        .args(["apply", "--gguf", "--model", f.model.to_str().unwrap()])
        .args(["--patch", patch.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0));
    let text = String::from_utf8(out.stdout).unwrap();
    for needle in [
        "[GGUF]",
        "[VERIFY]",
        "relecture OK",
        "blk.0.ffn_down.weight",
        "Q6_K",
    ] {
        assert!(text.contains(needle), "{needle} missing in\n{text}");
    }
}

// ------------------------------------------------------------------ library: guard and marker

fn validated(dir: &Path, kind: OperationKind) -> ValidatedPatch {
    ValidatedPatch {
        description: None,
        backup: true,
        operations: vec![ValidatedOperation {
            id: "op".into(),
            layers: LayerSelection::All,
            projections: vec![Projection::Gate, Projection::Up],
            kind,
            vector_file: dir.join("v.json"),
            vector_tensor: None,
            alpha: 0.01,
            axis: Axis::Rows,
        }],
        warnings: Vec::new(),
    }
}

/// Fails the second shift: the first tensor is already written, the backup restores it.
struct FailSecond<'a> {
    inner: &'a mut gguf::GgufBackend,
    shifts: usize,
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
    fn shift_tensor(
        &mut self,
        name: &str,
        v: &[f32],
        alpha: f32,
        op: OperationKind,
        axis: Axis,
    ) -> weight_surgeon::patch::Result<OpStats> {
        self.shifts += 1;
        match self.shifts {
            1 => self.inner.shift_tensor(name, v, alpha, op, axis),
            _ => Err(PatchError::Backend("injected".into())),
        }
    }
    fn flush(&mut self) -> weight_surgeon::patch::Result<()> {
        self.inner.flush()
    }
}

#[test]
fn injected_failure_rolls_back_and_the_marker_lives_only_during_the_write() {
    let f = Fixture::new(mlp_gguf([Q4_K, Q4_K, Q4_K], 2, 27));
    f.vector("v.json", &hidden_vector(1, 1.0));
    let patch = validated(f.dir.path(), OperationKind::Subtract);
    let marker = gguf::marker_path(&f.model);
    let original_sha = sha(&f.original);
    let mut saw_marker = false;
    let result = gguf::apply_gguf(
        &f.model,
        &patch,
        &GgufOptions::default(),
        false,
        Arc::new(AtomicBool::new(false)),
        &mut |backend, guard| {
            let m: gguf::Marker = serde_json::from_slice(&std::fs::read(&marker).unwrap()).unwrap();
            assert_eq!(m.original_sha256.as_deref(), Some(original_sha.as_str()));
            assert!(m.backup.as_ref().unwrap().exists());
            saw_marker = true;
            apply_patch(
                &patch,
                &mut FailSecond {
                    inner: backend,
                    shifts: 0,
                },
                guard,
                false,
            )
        },
    );
    assert!(saw_marker);
    match result {
        Err(GgufApplyError::Patch(PatchError::ExecutionFailed { rolled_back, .. })) => {
            assert!(rolled_back)
        }
        other => panic!("{other:?}"),
    }
    assert!(
        std::fs::read(&f.model).unwrap() == f.original,
        "restored from the backup"
    );
    assert!(!marker.exists());
}

#[test]
fn project_out_without_its_read_direction_fails_before_any_write() {
    // project_out is supported on a GGUF since the residual writers are (tests/p15); one
    // that skipped validation and lacks its b tensor still never touches the file.
    let f = Fixture::new(mlp_gguf([Q4_K, Q8_0, Q6_K], 1, 28));
    f.vector("v.json", &hidden_vector(1, 1.0));
    let mut patch = validated(f.dir.path(), OperationKind::ProjectOut);
    patch.operations[0].projections = vec![Projection::Down];
    patch.operations[0].axis = Axis::Cols;
    let result = gguf::apply_gguf(
        &f.model,
        &patch,
        &GgufOptions::default(),
        false,
        Arc::new(AtomicBool::new(false)),
        &mut |backend, guard| apply_patch(&patch, backend, guard, false),
    );
    match result {
        Err(GgufApplyError::Patch(PatchError::ExecutionFailed {
            cause,
            left_modified,
            ..
        })) => {
            assert!(cause.contains("project_out"), "{cause}");
            assert!(!left_modified);
        }
        other => panic!("{other:?}"),
    }
    assert!(std::fs::read(&f.model).unwrap() == f.original);
    assert!(!gguf::marker_path(&f.model).exists());
}

#[test]
fn cancellation_between_chunks_rolls_back() {
    let f = Fixture::new(mlp_gguf([Q8_0, Q8_0, Q8_0], 1, 29));
    f.vector("v.json", &hidden_vector(1, 1.0));
    let patch = validated(f.dir.path(), OperationKind::Subtract);
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&cancel);
    let result = gguf::apply_gguf(
        &f.model,
        &patch,
        &GgufOptions::default(),
        false,
        cancel,
        &mut |backend, guard| {
            // Raised after the guard exists: the first shift sees it at its first chunk.
            flag.store(true, std::sync::atomic::Ordering::SeqCst);
            apply_patch(&patch, backend, guard, false)
        },
    );
    assert!(matches!(
        result,
        Err(GgufApplyError::Patch(PatchError::ExecutionFailed {
            rolled_back: true,
            ..
        }))
    ));
    assert!(std::fs::read(&f.model).unwrap() == f.original);
}

#[test]
fn validated_patch_from_json_runs_on_the_library_api() {
    let f = Fixture::new(mlp_gguf([Q4_K, Q8_0, Q6_K], 1, 30));
    f.vector("v.json", &hidden_vector(1, 1.0));
    let patch = parse_patch(
        &json!({"schema_version": 1, "operations": [op("d", "0", "down_proj", "v.json", "cols", 0.01)]})
            .to_string(),
    )
    .unwrap();
    let patch = validate_patch(&patch, f.dir.path(), &ValidationOptions::default()).unwrap();
    let opts = GgufOptions {
        requant: Requant::Recompute,
        ..GgufOptions::default()
    };
    let run = gguf::apply_gguf(
        &f.model,
        &patch,
        &opts,
        false,
        Arc::new(AtomicBool::new(false)),
        &mut |backend, guard| apply_patch(&patch, backend, guard, false),
    )
    .unwrap();
    assert!(run.verified);
    assert_eq!(run.tensors.len(), 1);
    assert_eq!(run.requant, Requant::Recompute);
}

// ------------------------------------------------------------------ SafeTensors I/O, kernel

fn write_safetensors(path: &Path, name: &str, shape: [usize; 2], values: &[f32]) {
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut header =
        json!({ name: {"dtype": "F32", "shape": shape, "data_offsets": [0, data.len()]}})
            .to_string();
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&data);
    std::fs::write(path, file).unwrap();
}

#[test]
fn safetensors_apply_reports_io_and_honours_surgeon_kernel() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model.safetensors");
    let mut rng = Rng(31);
    let w: Vec<f32> = (0..64 * 48).map(|_| rng.normal()).collect();
    write_safetensors(&model, "model.layers.0.mlp.gate_proj.weight", [64, 48], &w);
    std::fs::write(
        dir.path().join("v.json"),
        serde_json::to_string(&vec![0.5f32; 48]).unwrap(),
    )
    .unwrap();
    let patch = dir.path().join("p.json");
    let ops = json!({"schema_version": 1, "backup": false,
        "operations": [op("g", "0", "gate_proj", "v.json", "rows", 0.25)]});
    std::fs::write(&patch, ops.to_string()).unwrap();
    let args = [
        "apply",
        "--model",
        model.to_str().unwrap(),
        "--patch",
        patch.to_str().unwrap(),
        "--prefetch-chunk-mib",
        "1",
    ];
    let original = std::fs::read(&model).unwrap();
    let (code, env) = run_json(&args, &[("SURGEON_KERNEL", "scalar")]);
    assert_eq!(code, 0, "{env}");
    let io = &env["data"]["io"];
    assert_eq!(io["kernel"]["level"], "scalar");
    assert_eq!(io["bytes"], 64 * 48 * 4);
    assert!(io["chunks"].as_u64().unwrap() >= 1);
    assert!(io["page_faults"].is_object(), "{io}");
    let scalar = std::fs::read(&model).unwrap();
    // Unknown level: best available, with a warning; same bytes from the same input.
    std::fs::write(&model, &original).unwrap();
    let (code, env) = run_json(&args, &[("SURGEON_KERNEL", "sse9")]);
    assert_eq!(code, 0, "{env}");
    assert!(env["data"]["io"]["kernel"]["warning"]
        .as_str()
        .unwrap()
        .contains("inconnu"));
    assert!(std::fs::read(&model).unwrap() == scalar);
    // GGUF-only options are refused without --gguf.
    let mut bad = args.to_vec();
    bad.extend(["--requant", "recompute"]);
    let (code, _) = run_json(&bad, &[]);
    assert_eq!(code, 2);
}

// ------------------------------------------------------------------ MCP

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
            "clientInfo": {"name": "p12", "version": "0"}}}),
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
            reply["result"]["structuredContent"].clone()
        })
        .collect()
}

#[test]
fn mcp_apply_gguf_with_sandbox_and_read_only() {
    let f = Fixture::new(mlp_gguf([Q4_K, Q8_0, Q6_K], 1, 32));
    f.vector("v.json", &hidden_vector(1, 1.0));
    let patch = json!({"schema_version": 1, "backup": false,
        "operations": [op("d", "0", "down_proj", "v.json", "cols", 0.01)]});
    let apply = |extra: Value| {
        let mut args = json!({"model": "m.gguf", "patch": patch});
        for (k, v) in extra.as_object().unwrap() {
            args[k] = v.clone();
        }
        json!({"name": "surgeon_apply", "arguments": args})
    };
    let replies = mcp(
        f.dir.path(),
        false,
        &[
            apply(json!({})),
            apply(json!({"gguf": true, "dry_run": true})),
            apply(json!({"gguf": true, "model": "../m.gguf"})),
            json!({"name": "surgeon_inspect", "arguments": {"model": "m.gguf"}}),
            apply(json!({"gguf": true, "requant": "recompute"})),
        ],
    );
    assert_eq!(replies[0]["error"]["kind"], "invalid_input");
    assert_eq!(
        replies[1]["data"]["report"]["dry_run"], true,
        "{}",
        replies[1]
    );
    assert_eq!(replies[2]["error"]["kind"], "path_not_allowed");
    assert_eq!(replies[3]["data"]["format"], "gguf");
    assert_eq!(replies[4]["ok"], true, "{}", replies[4]);
    assert_eq!(replies[4]["data"]["gguf"]["requant"], "recompute");
    // `backup: false` in the patch: an agent still gets a backup.
    assert_eq!(replies[4]["data"]["backups"].as_array().unwrap().len(), 1);
    let replies = mcp(f.dir.path(), true, &[apply(json!({"gguf": true}))]);
    assert_eq!(replies[0]["error"]["kind"], "read_only");
}

/// Writes a fake GGUF, a vector and patches into `$SURGEON_FAKE_GGUF_DIR` for manual runs of
/// the release binary (`cargo test --test p12_gguf -- --ignored make_fake_gguf_on_disk`).
#[test]
#[ignore]
fn make_fake_gguf_on_disk() {
    let dir = PathBuf::from(std::env::var("SURGEON_FAKE_GGUF_DIR").unwrap());
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("model.gguf"), mlp_gguf([Q4_K, Q8_0, Q6_K], 4, 40)).unwrap();
    std::fs::write(dir.join("f16.gguf"), mlp_gguf([F16, F16, F16], 1, 41)).unwrap();
    let v = hidden_vector(1, 1.0);
    std::fs::write(dir.join("v.json"), serde_json::to_string(&v).unwrap()).unwrap();
    std::fs::write(
        dir.join("big.json"),
        serde_json::to_string(&vec![-1e6f32; HIDDEN]).unwrap(),
    )
    .unwrap();
    let write = |name: &str, ops: Value| {
        let patch = json!({"schema_version": 1, "backup": true, "operations": ops});
        std::fs::write(dir.join(name), patch.to_string()).unwrap();
    };
    write(
        "patch.json",
        json!([
            op("gate", "0-3", "gate_proj", "v.json", "rows", 0.004),
            op("down", "1-2", "down_proj", "v.json", "cols", 0.003),
        ]),
    );
    write(
        "overflow.json",
        json!([op("boom", "0", "gate_proj", "big.json", "rows", 1.0)]),
    );
}
