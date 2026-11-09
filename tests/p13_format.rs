//! `.surgery` v1 container: round trips, determinism, hostile input (targeted corruptions,
//! truncation at every length, seeded random corruption) and schema conformance.
//! Files are built with the writer itself; hostile ones are forged from its output.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use weight_surgeon::surgery::error::SurgeryError;
use weight_surgeon::surgery::format::{
    BlobRef, Compression, CompressionAlgo, FactorDtype, FormatError, LowRankFactors, SurgeryHeader,
    TensorKind, FORMAT_VERSION,
};
use weight_surgeon::surgery::reader::{is_surgery_file, SurgeryReader};
use weight_surgeon::surgery::schema::header_schema;
use weight_surgeon::surgery::writer::{HeaderInit, SurgeryWriter, TensorMeta, WriteSummary};

const PRE: usize = 20;
const FOOT: usize = 36;
const SLACK: usize = 4096;

// ---------------------------------------------------------------- helpers

fn val(i: usize) -> f32 {
    // Multiples of 1/8 in [-1, 1]: exact in f16, bf16 and f32.
    ((i * 37 % 17) as f32 - 8.0) / 8.0
}

fn deflate(level: u8) -> Compression {
    Compression {
        algo: CompressionAlgo::Deflate,
        level,
    }
}

fn stored() -> Compression {
    Compression {
        algo: CompressionAlgo::None,
        level: 0,
    }
}

fn init(dtype: FactorDtype, compression: Compression, timestamp: Option<&str>) -> HeaderInit {
    HeaderInit {
        created_with: "test 0.0".into(),
        base_name: "base".into(),
        base_tensor_count: 10,
        tuned_name: "tuned".into(),
        rank: 3,
        factor_dtype: dtype,
        compression,
        timestamp: timestamp.map(str::to_string),
    }
}

fn meta(name: &str, shape: &[usize]) -> TensorMeta {
    TensorMeta {
        name: name.into(),
        shape: shape.to_vec(),
        target_dtype: "BF16".into(),
        base_sha256: "ab".repeat(32),
        rel_error: 0.125,
        energy: 0.5,
    }
}

fn factors(m: usize, n: usize, r: usize, seed: usize) -> LowRankFactors {
    LowRankFactors {
        u: (0..m * r).map(|i| val(i + seed)).collect(),
        s: (0..r).map(|i| val(i + seed + 3).abs() + 0.5).collect(),
        v: (0..r * n).map(|i| val(i + seed + 7)).collect(),
    }
}

/// Two low-rank tensors (ranks 2 and 3) and one 1D full delta.
fn write_sample(
    path: &Path,
    dtype: FactorDtype,
    compression: Compression,
    timestamp: Option<&str>,
) -> WriteSummary {
    let mut w = SurgeryWriter::create(path, init(dtype, compression, timestamp), false).unwrap();
    w.add_lowrank(meta("a.weight", &[6, 5]), &factors(6, 5, 2, 0))
        .unwrap();
    w.add_lowrank(meta("b.weight", &[4, 7]), &factors(4, 7, 3, 5))
        .unwrap();
    let delta: Vec<f32> = (0..8).map(val).collect();
    w.add_full_delta(meta("c.bias", &[8]), &delta).unwrap();
    w.finish().unwrap()
}

fn sample_bytes() -> Vec<u8> {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.surgery");
    write_sample(&path, FactorDtype::F16, deflate(6), None);
    fs::read(path).unwrap()
}

fn header_len(bytes: &[u8]) -> usize {
    u64::from_le_bytes(bytes[12..20].try_into().unwrap()) as usize
}

fn header_of(bytes: &[u8]) -> SurgeryHeader {
    serde_json::from_slice(&bytes[PRE..PRE + header_len(bytes)]).unwrap()
}

fn assemble(json: &[u8], blobs: &[u8]) -> Vec<u8> {
    let mut out = b"SURG".to_vec();
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&[0u8; 6]);
    out.extend_from_slice(&(json.len() as u64).to_le_bytes());
    out.extend_from_slice(json);
    out.extend_from_slice(blobs);
    refresh_footer(&mut out, false);
    out
}

/// Recompute the footer over everything before it (`append` adds the 36 bytes first).
fn refresh_footer(bytes: &mut Vec<u8>, has_footer: bool) {
    if has_footer {
        bytes.truncate(bytes.len() - FOOT);
    }
    let digest = Sha256::digest(&bytes[..]);
    bytes.extend_from_slice(&digest);
    bytes.extend_from_slice(b"SURG");
}

fn for_each_blob(h: &mut SurgeryHeader, mut f: impl FnMut(&mut BlobRef)) {
    for t in &mut h.tensors {
        let blobs = [
            &mut t.blobs.u,
            &mut t.blobs.s,
            &mut t.blobs.v,
            &mut t.blobs.delta,
        ];
        blobs.into_iter().flatten().for_each(&mut f);
    }
}

/// Rebuilds a structurally consistent file around a mutated header and blob section, with a
/// valid footer: only the mutation can make it fail. The header is padded with spaces so that
/// offsets, shifted by `SLACK`, stay valid whatever the mutation does to digit counts.
fn forge(orig: &[u8], mutate: impl FnOnce(&mut SurgeryHeader, &mut Vec<u8>, u64)) -> Vec<u8> {
    let hl = header_len(orig);
    let mut header = header_of(orig);
    for_each_blob(&mut header, |b| b.offset += SLACK as u64);
    let mut blobs = orig[PRE + hl..orig.len() - FOOT].to_vec();
    mutate(&mut header, &mut blobs, (PRE + hl + SLACK) as u64);
    let mut json = header.to_canonical_json().unwrap();
    assert!(
        json.len() <= hl + SLACK,
        "mutation grew the header too much"
    );
    json.resize(hl + SLACK, b' ');
    assemble(&json, &blobs)
}

/// Swaps the stored bytes of the first tensor's `s` blob, shifting the following blobs.
fn replace_s_blob(h: &mut SurgeryHeader, blobs: &mut Vec<u8>, start: u64, new: &[u8]) {
    let old = h.tensors[0].blobs.s.clone().unwrap();
    let from = (old.offset - start) as usize;
    blobs.splice(from..from + old.comp_len as usize, new.iter().copied());
    let delta = new.len() as i64 - old.comp_len as i64;
    for_each_blob(h, |b| {
        if b.offset > old.offset {
            b.offset = (b.offset as i64 + delta) as u64;
        }
    });
    if let Some(s) = h.tensors[0].blobs.s.as_mut() {
        s.comp_len = new.len() as u64;
    }
}

fn open_bytes(dir: &Path, bytes: &[u8]) -> Result<SurgeryReader, SurgeryError> {
    let path = dir.join("x.surgery");
    fs::write(&path, bytes).unwrap();
    SurgeryReader::open(&path)
}

fn open_failure(bytes: &[u8]) -> FormatError {
    let dir = TempDir::new().unwrap();
    match open_bytes(dir.path(), bytes) {
        Err(SurgeryError::Format(e)) => e,
        Err(other) => panic!("expected a format error, got {other}"),
        Ok(_) => panic!("file unexpectedly accepted"),
    }
}

/// Reads every tensor; any failure must be a typed format error.
fn read_all(reader: &mut SurgeryReader) -> Result<(), SurgeryError> {
    for entry in reader.header().tensors.clone() {
        match entry.kind {
            TensorKind::Lowrank => {
                reader.read_lowrank(&entry)?;
            }
            TensorKind::FullDelta => {
                reader.read_full_delta(&entry)?;
            }
        }
    }
    Ok(())
}

fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut x = seed;
    move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    }
}

// ---------------------------------------------------------------- round trip

#[test]
fn round_trip_every_dtype_and_compression() {
    for dtype in [FactorDtype::F16, FactorDtype::Bf16, FactorDtype::F32] {
        for comp in [deflate(1), deflate(6), deflate(10), stored()] {
            let dir = TempDir::new().unwrap();
            let path = dir.path().join("p.surgery");
            let summary = write_sample(&path, dtype, comp, None);
            assert_eq!(summary.tensors, 3);
            assert_eq!(summary.bytes, fs::metadata(&path).unwrap().len());

            assert!(is_surgery_file(&path));
            let mut r = SurgeryReader::open(&path).unwrap();
            let h = r.header().clone();
            assert_eq!(h.factor_dtype, dtype);
            assert_eq!(h.compression, comp);
            assert_eq!(h.tensors.len(), 3);

            let a = r.read_lowrank(&h.tensors[0]).unwrap();
            assert_eq!(a, factors(6, 5, 2, 0), "{dtype:?} {comp:?}");
            let b = r.read_lowrank(&h.tensors[1]).unwrap();
            assert_eq!(b, factors(4, 7, 3, 5));
            let c = r.read_full_delta(&h.tensors[2]).unwrap();
            assert_eq!(c, (0..8).map(val).collect::<Vec<_>>());
            assert_eq!(h.tensors[2].kind, TensorKind::FullDelta);
            assert_eq!(h.tensors[2].rank, 0);
        }
    }
}

#[test]
fn f16_and_bf16_round_to_nearest_even() {
    // (input, f16 expected, bf16 expected): exact ties go to the even mantissa.
    let t11 = 2f32.powi(-11);
    let t8 = 2f32.powi(-8);
    let cases = [
        (1.0 + t11, 1.0, 1.0),
        (1.0 + 3.0 * t11, 1.0 + 4.0 * t11, 1.0),
        (1.0 + t8, 1.0 + t8, 1.0),
        (1.0 + 3.0 * t8, 1.0 + 3.0 * t8, 1.0 + 4.0 * t8),
        (65504.0, 65504.0, 65536.0),
    ];
    for (dtype, pick) in [(FactorDtype::F16, 1), (FactorDtype::Bf16, 2)] {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("r.surgery");
        let mut w = SurgeryWriter::create(&path, init(dtype, stored(), None), false).unwrap();
        let input: Vec<f32> = cases.iter().map(|c| c.0).collect();
        w.add_full_delta(meta("x", &[input.len()]), &input).unwrap();
        // Factor path (U, S, V in the factor dtype): 1 x 5 with rank 1.
        let f = LowRankFactors {
            u: vec![input[0]],
            s: vec![input[1]],
            v: input.clone(),
        };
        w.add_lowrank(meta("y", &[1, 5]), &f).unwrap();
        w.finish().unwrap();
        let mut r = SurgeryReader::open(&path).unwrap();
        let entry = r.header().tensors[1].clone();
        let got = r.read_lowrank(&entry).unwrap();
        let want = |idx: usize| {
            if pick == 1 {
                cases[idx].1
            } else {
                cases[idx].2
            }
        };
        assert_eq!(got.u, vec![want(0)], "{dtype:?}");
        assert_eq!(got.s, vec![want(1)]);
        let want_v: Vec<f32> = (0..5).map(want).collect();
        assert_eq!(got.v, want_v, "{dtype:?}");
    }
}

#[test]
fn many_tensors_cross_offset_digit_boundaries() {
    // Offsets grow past 10, 100, 1000, ... while header_len grows: exercises the fixed point.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("many.surgery");
    let mut w =
        SurgeryWriter::create(&path, init(FactorDtype::F32, stored(), None), false).unwrap();
    for i in 0..300 {
        let delta = vec![i as f32; 3];
        w.add_full_delta(meta(&format!("t{i:03}.bias"), &[3]), &delta)
            .unwrap();
    }
    w.finish().unwrap();
    let mut r = SurgeryReader::open(&path).unwrap();
    for (i, e) in r.header().tensors.clone().iter().enumerate() {
        assert_eq!(r.read_full_delta(e).unwrap(), vec![i as f32; 3]);
    }
}

#[test]
fn deflate_actually_compresses_redundant_factors() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("z.surgery");
    let mut w =
        SurgeryWriter::create(&path, init(FactorDtype::F32, deflate(6), None), false).unwrap();
    w.add_full_delta(meta("z", &[4096]), &vec![0.0; 4096])
        .unwrap();
    w.finish().unwrap();
    let r = SurgeryReader::open(&path).unwrap();
    let blob = r.header().tensors[0].blobs.delta.clone().unwrap();
    assert_eq!(blob.raw_len, 4096 * 4);
    assert!(blob.comp_len < 200, "comp_len = {}", blob.comp_len);
}

// ---------------------------------------------------------------- determinism and header

#[test]
fn identical_bytes_over_two_runs_and_footer_sha() {
    let dir = TempDir::new().unwrap();
    let (p1, p2) = (dir.path().join("1.surgery"), dir.path().join("2.surgery"));
    let s1 = write_sample(&p1, FactorDtype::F16, deflate(6), None);
    let s2 = write_sample(&p2, FactorDtype::F16, deflate(6), None);
    let (b1, b2) = (fs::read(&p1).unwrap(), fs::read(&p2).unwrap());
    assert_eq!(b1, b2);
    assert_eq!(s1.sha256, s2.sha256);

    let body = &b1[..b1.len() - FOOT];
    let digest = Sha256::digest(body);
    assert_eq!(&b1[b1.len() - FOOT..b1.len() - 4], &digest[..]);
    assert_eq!(&b1[b1.len() - 4..], b"SURG");
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(s1.sha256, hex);
    assert_eq!(&b1[..4], b"SURG");
    assert_eq!(u16::from_le_bytes([b1[4], b1[5]]), 1);
    assert_eq!(&b1[6..12], &[0u8; 6]);
}

#[test]
fn header_is_canonical_json() {
    let bytes = sample_bytes();
    let json = &bytes[PRE..PRE + header_len(&bytes)];
    let value: Value = serde_json::from_slice(json).unwrap();
    // `Value` maps are sorted and `to_vec` is compact: equality means sorted keys, no spaces.
    assert_eq!(serde_json::to_vec(&value).unwrap(), json);
    assert!(!String::from_utf8_lossy(json).contains("provenance"));
    // Blob offsets are absolute and point right after the header.
    let h = header_of(&bytes);
    let first = h.tensors[0].blobs.u.clone().unwrap();
    assert_eq!(first.offset as usize, PRE + header_len(&bytes));
}

#[test]
fn timestamp_only_when_requested() {
    let dir = TempDir::new().unwrap();
    let path = |n: &str| dir.path().join(n);
    write_sample(&path("a"), FactorDtype::F16, deflate(6), None);
    write_sample(
        &path("b"),
        FactorDtype::F16,
        deflate(6),
        Some("2026-01-02T03:04:05Z"),
    );
    write_sample(
        &path("c"),
        FactorDtype::F16,
        deflate(6),
        Some("2026-01-02T03:04:05Z"),
    );
    let (a, b, c) = (
        fs::read(path("a")).unwrap(),
        fs::read(path("b")).unwrap(),
        fs::read(path("c")).unwrap(),
    );
    assert!(header_of(&a).provenance.is_none());
    let prov = header_of(&b).provenance.unwrap();
    assert_eq!(prov.timestamp, "2026-01-02T03:04:05Z");
    assert_ne!(a, b);
    assert_eq!(b, c, "same timestamp, same bytes");
    // The file still opens with provenance present.
    SurgeryReader::open(&path("b")).unwrap();
}

#[test]
fn blob_sha_is_over_raw_bytes() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("t.surgery");
    write_sample(&path, FactorDtype::F16, deflate(6), None);
    let mut r = SurgeryReader::open(&path).unwrap();
    let h = r.header().clone();
    let blob = h.tensors[2].blobs.delta.clone().unwrap();
    let raw: Vec<u8> = (0..8).flat_map(|i| val(i).to_le_bytes()).collect();
    let hex: String = Sha256::digest(&raw)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(blob.sha256, hex);
    assert_eq!(blob.raw_len, 32);
    assert!(r.read_full_delta(&h.tensors[2]).is_ok());
}

// ---------------------------------------------------------------- writer behaviour

fn dir_entries(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

#[test]
fn nothing_visible_before_finish_and_no_leftovers() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("out.surgery");
    let mut w =
        SurgeryWriter::create(&path, init(FactorDtype::F16, deflate(6), None), false).unwrap();
    w.add_full_delta(meta("c", &[2]), &[1.0, 2.0]).unwrap();
    assert!(!path.exists(), "target must not exist before finish");
    drop(w);
    assert!(
        dir_entries(dir.path()).is_empty(),
        "abandoned writer leaves nothing"
    );

    write_sample(&path, FactorDtype::F16, deflate(6), None);
    assert_eq!(dir_entries(dir.path()), vec!["out.surgery".to_string()]);
}

#[test]
fn overwrite_flag_is_honoured() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("out.surgery");
    fs::write(&path, b"precious").unwrap();
    let err = SurgeryWriter::create(&path, init(FactorDtype::F16, stored(), None), false)
        .err()
        .unwrap();
    assert!(matches!(err, SurgeryError::Usage(_)));
    assert_eq!(fs::read(&path).unwrap(), b"precious");

    let mut w = SurgeryWriter::create(&path, init(FactorDtype::F16, stored(), None), true).unwrap();
    assert_eq!(
        fs::read(&path).unwrap(),
        b"precious",
        "still untouched before finish"
    );
    w.add_full_delta(meta("c", &[1]), &[1.0]).unwrap();
    w.finish().unwrap();
    assert!(is_surgery_file(&path));
}

#[test]
fn f16_overflow_is_an_error_not_infinity() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("o.surgery");
    let mut w =
        SurgeryWriter::create(&path, init(FactorDtype::F16, stored(), None), false).unwrap();
    let big = LowRankFactors {
        u: vec![1.0],
        s: vec![70000.0],
        v: vec![1.0],
    };
    let err = w.add_lowrank(meta("big", &[1, 1]), &big).err().unwrap();
    assert!(
        matches!(&err, SurgeryError::FactorOverflow { tensor, dtype } if tensor == "big" && *dtype == "f16"),
        "{err}"
    );
    // 65520 is the first value that rounds to infinity; 65519 still fits.
    let edge = |x: f32| LowRankFactors {
        u: vec![1.0],
        s: vec![x],
        v: vec![1.0],
    };
    assert!(w
        .add_lowrank(meta("edge", &[1, 1]), &edge(65520.0))
        .is_err());
    assert!(w.add_lowrank(meta("ok", &[1, 1]), &edge(65519.0)).is_ok());
    drop(w);

    // The same value is fine in bf16 and f32.
    for dtype in [FactorDtype::Bf16, FactorDtype::F32] {
        let mut w = SurgeryWriter::create(&path, init(dtype, stored(), None), true).unwrap();
        w.add_lowrank(meta("big", &[1, 1]), &big).unwrap();
        w.finish().unwrap();
    }
    // bf16 rounding of f32::MAX overflows too.
    let mut w =
        SurgeryWriter::create(&path, init(FactorDtype::Bf16, stored(), None), true).unwrap();
    let max = LowRankFactors {
        u: vec![f32::MAX],
        s: vec![1.0],
        v: vec![1.0],
    };
    assert!(matches!(
        w.add_lowrank(meta("max", &[1, 1]), &max),
        Err(SurgeryError::FactorOverflow { .. })
    ));
}

#[test]
fn writer_rejects_inconsistent_input() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("w.surgery");
    let mut w =
        SurgeryWriter::create(&path, init(FactorDtype::F32, stored(), None), false).unwrap();
    // wrong factor sizes
    assert!(matches!(
        w.add_lowrank(meta("a", &[2, 2]), &factors(2, 3, 1, 0)),
        Err(SurgeryError::Usage(_))
    ));
    // rank above the header rank (3)
    assert!(matches!(
        w.add_lowrank(meta("a", &[4, 4]), &factors(4, 4, 4, 0)),
        Err(SurgeryError::Format(_))
    ));
    // NaN
    let nan = LowRankFactors {
        u: vec![f32::NAN],
        s: vec![1.0],
        v: vec![1.0],
    };
    assert!(matches!(
        w.add_lowrank(meta("a", &[1, 1]), &nan),
        Err(SurgeryError::Usage(_))
    ));
    // bad hash, empty name, 3D lowrank, wrong delta length
    let mut bad = meta("a", &[1, 1]);
    bad.base_sha256 = "zz".into();
    assert!(w.add_lowrank(bad, &factors(1, 1, 1, 0)).is_err());
    assert!(w
        .add_lowrank(meta("", &[1, 1]), &factors(1, 1, 1, 0))
        .is_err());
    assert!(w
        .add_lowrank(meta("a", &[1, 1, 1]), &factors(1, 1, 1, 0))
        .is_err());
    assert!(w.add_full_delta(meta("a", &[3]), &[1.0]).is_err());
    // duplicate name
    w.add_full_delta(meta("dup", &[1]), &[1.0]).unwrap();
    assert!(matches!(
        w.add_full_delta(meta("dup", &[1]), &[1.0]),
        Err(SurgeryError::Usage(_))
    ));
    // The failed calls must not have corrupted the file in progress.
    w.finish().unwrap();
    let mut r = SurgeryReader::open(&path).unwrap();
    assert_eq!(r.header().tensors.len(), 1);
    let e = r.header().tensors[0].clone();
    assert_eq!(r.read_full_delta(&e).unwrap(), vec![1.0]);
}

// ---------------------------------------------------------------- reader: targeted corruptions

#[test]
fn good_forge_is_accepted() {
    // Sanity of the helper itself: an identity mutation must open and read.
    let bytes = forge(&sample_bytes(), |_, _, _| {});
    let dir = TempDir::new().unwrap();
    let mut r = open_bytes(dir.path(), &bytes).unwrap();
    read_all(&mut r).unwrap();
}

#[test]
fn bad_magic_version_flags() {
    let good = sample_bytes();
    let mut b = good.clone();
    b[0] = b'X';
    assert!(matches!(open_failure(&b), FormatError::BadMagic));

    for v in [0u16, 2, 0x0100, u16::MAX] {
        let mut b = good.clone();
        b[4..6].copy_from_slice(&v.to_le_bytes());
        assert!(
            matches!(open_failure(&b), FormatError::UnsupportedVersion { found, .. } if found == v),
            "version {v}"
        );
    }
    let mut b = good.clone();
    b[6] = 1;
    assert!(matches!(
        open_failure(&b),
        FormatError::UnsupportedFlags { flags: 1, .. }
    ));
    let mut b = good.clone();
    b[11] = 0x80;
    assert!(matches!(
        open_failure(&b),
        FormatError::UnsupportedFlags { .. }
    ));

    // footer magic
    let mut b = good.clone();
    let n = b.len();
    b[n - 1] = b'!';
    assert!(matches!(open_failure(&b), FormatError::BadMagic));
}

#[test]
fn header_len_is_checked_against_the_real_size() {
    let good = sample_bytes();
    for declared in [
        u64::MAX,
        1 << 63,
        good.len() as u64,
        (good.len() - PRE - FOOT + 1) as u64,
    ] {
        let mut b = good.clone();
        b[12..20].copy_from_slice(&declared.to_le_bytes());
        assert!(
            matches!(open_failure(&b), FormatError::HeaderLenMismatch { declared: d, .. } if d == declared),
            "declared {declared}"
        );
    }
    // Plausible but wrong values fail in typed ways too (never a panic).
    for declared in [
        0u64,
        1,
        5,
        header_len(&good) as u64 - 1,
        header_len(&good) as u64 + 1,
    ] {
        let mut b = good.clone();
        b[12..20].copy_from_slice(&declared.to_le_bytes());
        open_failure(&b);
    }
}

#[test]
fn offsets_out_of_bounds_and_overlap() {
    let good = sample_bytes();
    for off in [u64::MAX, u64::MAX - 3, 1 << 40, 0, 10] {
        let bytes = forge(&good, |h, _, _| {
            h.tensors[1].blobs.v.as_mut().unwrap().offset = off;
        });
        assert!(
            matches!(open_failure(&bytes), FormatError::OffsetOutOfBounds { .. }),
            "offset {off}"
        );
    }
    // comp_len running past the footer, and offset + comp_len overflowing u64
    for comp in [1u64 << 40, u64::MAX] {
        let bytes = forge(&good, |h, _, _| {
            h.tensors[0].blobs.u.as_mut().unwrap().comp_len = comp;
        });
        assert!(matches!(
            open_failure(&bytes),
            FormatError::InvalidHeader(_) | FormatError::OffsetOutOfBounds { .. }
        ));
    }
    // blob of tensor 1 placed on top of blob of tensor 0
    let bytes = forge(&good, |h, _, _| {
        let target = h.tensors[0].blobs.u.as_ref().unwrap().offset;
        h.tensors[1].blobs.u.as_mut().unwrap().offset = target;
    });
    assert!(matches!(
        open_failure(&bytes),
        FormatError::BlobOverlap { .. }
    ));
    // a one-byte overlap between neighbours
    let bytes = forge(&good, |h, _, _| {
        h.tensors[0].blobs.s.as_mut().unwrap().offset -= 1;
    });
    assert!(matches!(
        open_failure(&bytes),
        FormatError::BlobOverlap { .. }
    ));
}

#[test]
fn gaps_and_unreferenced_bytes_are_refused() {
    let good = sample_bytes();
    let bytes = forge(&good, |_, blobs, _| blobs.extend_from_slice(&[0; 5]));
    assert!(matches!(
        open_failure(&bytes),
        FormatError::InvalidHeader(_)
    ));
}

#[test]
fn lying_raw_len_and_shape() {
    let good = sample_bytes();
    for lie in [0u64, 1, 5, 1 << 62, u64::MAX] {
        let bytes = forge(&good, |h, _, _| {
            h.tensors[0].blobs.u.as_mut().unwrap().raw_len = lie;
        });
        assert!(
            matches!(open_failure(&bytes), FormatError::RawLenMismatch { declared, .. } if declared == lie),
            "raw_len {lie}"
        );
    }
    // Shape that disagrees with the (honest) raw_len of its blobs.
    let bytes = forge(&good, |h, _, _| h.tensors[0].shape = vec![7, 5]);
    assert!(matches!(
        open_failure(&bytes),
        FormatError::RawLenMismatch { .. }
    ));
    // Rank that disagrees with s.
    let bytes = forge(&good, |h, _, _| h.tensors[0].rank = 1);
    assert!(matches!(
        open_failure(&bytes),
        FormatError::RawLenMismatch { .. }
    ));
}

#[test]
fn huge_declared_sizes_never_allocate() {
    let good = sample_bytes();
    // Self-consistent giant tensor: raw_len matches shape x rank but dwarfs the file.
    let bytes = forge(&good, |h, _, _| {
        let t = &mut h.tensors[0];
        t.shape = vec![1 << 40, 5];
        t.blobs.u.as_mut().unwrap().raw_len = (1u64 << 40) * 2 * 2;
    });
    open_failure(&bytes);
    let bytes = forge(&good, |h, _, _| {
        h.tensors[0].shape = vec![usize::MAX, usize::MAX]
    });
    open_failure(&bytes);
    let bytes = forge(&good, |h, _, _| h.rank = usize::MAX);
    assert!(matches!(
        open_failure(&bytes),
        FormatError::InvalidHeader(_)
    ));
}

#[test]
fn implausible_compression_ratio_is_refused() {
    let good = sample_bytes();
    // A 400 kB blob cannot come from a ~40 byte deflate stream (ceiling 1032:1).
    let bytes = forge(&good, |h, _, _| {
        h.tensors[2].shape = vec![100_000];
        h.tensors[2].blobs.delta.as_mut().unwrap().raw_len = 400_000;
    });
    assert!(
        matches!(open_failure(&bytes), FormatError::InvalidHeader(m) if m.contains("comp_len"))
    );
    // With `none`, stored and raw lengths must be equal; the sample's deflated blobs differ.
    let bytes = forge(&good, |h, _, _| h.compression = stored());
    assert!(matches!(
        open_failure(&bytes),
        FormatError::InvalidHeader(_)
    ));
}

#[test]
fn decompression_bomb_is_capped_at_declared_raw_len() {
    let good = sample_bytes();
    let bomb = miniz_oxide::deflate::compress_to_vec(&vec![0u8; 1 << 20], 9);
    let bytes = forge(&good, |h, blobs, start| {
        replace_s_blob(h, blobs, start, &bomb)
    });
    let dir = TempDir::new().unwrap();
    let mut r = open_bytes(dir.path(), &bytes).unwrap();
    let e = r.header().tensors[0].clone();
    // raw_len of `s` for tensor 0 is 2 x 2 bytes: the 1 MiB stream must be cut at 4 bytes.
    match r.read_lowrank(&e) {
        Err(SurgeryError::Format(FormatError::DecompressionBomb {
            limit: 4,
            blob: "s",
            ..
        })) => {}
        other => panic!("expected a bomb error, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn short_and_garbage_streams_are_typed_errors() {
    let good = sample_bytes();
    let dir = TempDir::new().unwrap();
    // Valid stream, fewer bytes than declared.
    let short = miniz_oxide::deflate::compress_to_vec(&[1u8, 2], 6);
    let bytes = forge(&good, |h, blobs, start| {
        replace_s_blob(h, blobs, start, &short)
    });
    let mut r = open_bytes(dir.path(), &bytes).unwrap();
    let e = r.header().tensors[0].clone();
    assert!(matches!(
        r.read_lowrank(&e),
        Err(SurgeryError::Format(FormatError::RawLenMismatch {
            blob: "s",
            ..
        }))
    ));
    // Garbage that is not deflate.
    let bytes = forge(&good, |h, blobs, start| {
        replace_s_blob(h, blobs, start, &[0xFF; 9])
    });
    let mut r = open_bytes(dir.path(), &bytes).unwrap();
    let e = r.header().tensors[0].clone();
    assert!(matches!(
        r.read_lowrank(&e),
        Err(SurgeryError::Format(
            FormatError::DecompressionFailed { .. }
        ))
    ));
}

#[test]
fn blob_hash_mismatch() {
    let good = sample_bytes();
    let bytes = forge(&good, |h, _, _| {
        h.tensors[0].blobs.u.as_mut().unwrap().sha256 = "0".repeat(64);
    });
    let dir = TempDir::new().unwrap();
    let mut r = open_bytes(dir.path(), &bytes).unwrap();
    let e = r.header().tensors[0].clone();
    assert!(matches!(
        r.read_lowrank(&e),
        Err(SurgeryError::Format(FormatError::BlobHashMismatch {
            blob: "u",
            ..
        }))
    ));
    // The other tensors stay readable.
    let e2 = r.header().tensors[2].clone();
    assert!(r.read_full_delta(&e2).is_ok());
}

#[test]
fn global_hash_mismatch_is_detected_at_open() {
    let good = sample_bytes();
    // One flipped bit anywhere in the blob section, footer left alone.
    let mut b = good.clone();
    let at = PRE + header_len(&good) + 3;
    b[at] ^= 1;
    assert!(matches!(open_failure(&b), FormatError::GlobalHashMismatch));
    // Flipped bit in the stored digest itself.
    let mut b = good.clone();
    let n = b.len();
    b[n - FOOT] ^= 0x80;
    assert!(matches!(open_failure(&b), FormatError::GlobalHashMismatch));
}

#[test]
fn non_finite_stored_values_are_refused() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("inf.surgery");
    let mut w =
        SurgeryWriter::create(&path, init(FactorDtype::F32, stored(), None), false).unwrap();
    w.add_full_delta(meta("c", &[1]), &[1.0]).unwrap();
    w.finish().unwrap();
    let good = fs::read(&path).unwrap();
    // Overwrite the single f32 with +inf, fix the blob hash and the footer.
    let bytes = forge(&good, |h, blobs, start| {
        let b = h.tensors[0].blobs.delta.as_mut().unwrap();
        let at = (b.offset - start) as usize;
        blobs[at..at + 4].copy_from_slice(&f32::INFINITY.to_le_bytes());
        b.sha256 = Sha256::digest(f32::INFINITY.to_le_bytes())
            .iter()
            .map(|x| format!("{x:02x}"))
            .collect();
    });
    let mut r = open_bytes(dir.path(), &bytes).unwrap();
    let e = r.header().tensors[0].clone();
    assert!(matches!(
        r.read_full_delta(&e),
        Err(SurgeryError::Format(FormatError::NonFiniteValue { .. }))
    ));
}

#[test]
fn caller_entries_are_not_trusted() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("e.surgery");
    write_sample(&path, FactorDtype::F16, deflate(6), None);
    let mut r = SurgeryReader::open(&path).unwrap();
    let tensors = r.header().tensors.clone();

    let mut lying = tensors[0].clone();
    lying.shape = vec![1 << 30, 1 << 30];
    assert!(matches!(
        r.read_lowrank(&lying),
        Err(SurgeryError::Usage(_))
    ));
    let mut renamed = tensors[0].clone();
    renamed.name = "nope".into();
    assert!(matches!(
        r.read_lowrank(&renamed),
        Err(SurgeryError::Usage(_))
    ));
    // wrong accessor for the kind
    assert!(matches!(
        r.read_full_delta(&tensors[0]),
        Err(SurgeryError::Usage(_))
    ));
    assert!(matches!(
        r.read_lowrank(&tensors[2]),
        Err(SurgeryError::Usage(_))
    ));
}

#[test]
fn invalid_headers_are_rejected_by_validation() {
    let good = sample_bytes();
    type Mutation = Box<dyn Fn(&mut SurgeryHeader)>;
    let cases: Vec<Mutation> = vec![
        Box::new(|h| h.tensors[0].name = String::new()),
        Box::new(|h| h.tensors[0].name = "x".repeat(2000)),
        Box::new(|h| h.tensors[0].name = "a\u{0}b".into()),
        Box::new(|h| h.tensors[1].name = h.tensors[0].name.clone()),
        Box::new(|h| h.tensors[0].shape = vec![0, 5]),
        Box::new(|h| h.tensors[0].shape = vec![]),
        Box::new(|h| h.tensors[0].shape = vec![1; 9]),
        Box::new(|h| h.tensors[0].target_dtype = "I8".into()),
        Box::new(|h| h.tensors[0].base_sha256 = "ABC".into()),
        Box::new(|h| h.tensors[0].base_sha256 = "AB".repeat(32)),
        Box::new(|h| h.tensors[0].rel_error = f64::NAN),
        Box::new(|h| h.tensors[0].energy = -1.0),
        Box::new(|h| h.tensors[0].rank = 0),
        Box::new(|h| h.tensors[0].rank = 9),
        Box::new(|h| h.tensors[2].rank = 1),
        Box::new(|h| h.tensors[0].blobs.delta = h.tensors[0].blobs.u.clone()),
        Box::new(|h| h.tensors[0].blobs.s = None),
        Box::new(|h| h.tensors[2].blobs.delta = None),
        Box::new(|h| h.tensors[0].blobs.u.as_mut().unwrap().sha256 = "xyz".into()),
        Box::new(|h| {
            h.compression = Compression {
                algo: CompressionAlgo::None,
                level: 5,
            }
        }),
        Box::new(|h| h.compression.level = 11),
        Box::new(|h| h.rank = 0),
        Box::new(|h| h.format_version = 2),
    ];
    for (i, mutate) in cases.iter().enumerate() {
        let bytes = forge(&good, |h, _, _| mutate(h));
        let dir = TempDir::new().unwrap();
        match open_bytes(dir.path(), &bytes) {
            Err(SurgeryError::Format(_)) => {}
            Err(other) => panic!("case {i}: wrong error {other}"),
            Ok(_) => panic!("case {i}: accepted"),
        }
    }
}

#[test]
fn malformed_json_headers() {
    let good = sample_bytes();
    let hl = header_len(&good);
    let blobs = &good[PRE + hl..good.len() - FOOT];
    let json = &good[PRE..PRE + hl];
    let text = String::from_utf8(json.to_vec()).unwrap();
    let variants: Vec<Vec<u8>> = vec![
        b"{}".to_vec(),
        b"[]".to_vec(),
        b"null".to_vec(),
        b"{".to_vec(),
        vec![0xFF, 0xFE, 0x00],
        text.replace("\"rank\":3", "\"rank\":-3").into_bytes(),
        text.replace("\"rank\":3", "\"rank\":3.5").into_bytes(),
        text.replace("\"rank\":3", "\"rank\":3,\"rank\":4")
            .into_bytes(),
        text.replace("\"f16\"", "\"f64\"").into_bytes(),
        format!("{}{}", "[".repeat(5000), "]".repeat(5000)).into_bytes(),
        format!("{text} trailing").into_bytes(),
    ];
    for (i, v) in variants.iter().enumerate() {
        let bytes = assemble(v, blobs);
        let dir = TempDir::new().unwrap();
        match open_bytes(dir.path(), &bytes) {
            Err(SurgeryError::Format(FormatError::InvalidHeader(_))) => {}
            Err(other) => panic!("variant {i}: wrong error {other}"),
            Ok(_) => panic!("variant {i}: accepted"),
        }
    }
    // Unknown extra keys are tolerated (forward compatible within a version).
    // Offsets must stay valid, so the added bytes eat the padding of a forged file.
    let padded = forge(&good, |_, _, _| {});
    let hl = header_len(&padded);
    let json = String::from_utf8(padded[PRE..PRE + hl].to_vec()).unwrap();
    let extra = json.replacen('{', "{\"x_future\":1,", 1);
    let extra = &extra[..extra.len() - "\"x_future\":1,".len()];
    let bytes = assemble(extra.as_bytes(), &padded[PRE + hl..padded.len() - FOOT]);
    let dir = TempDir::new().unwrap();
    open_bytes(dir.path(), &bytes).unwrap();
}

#[test]
fn is_surgery_file_cases() {
    let dir = TempDir::new().unwrap();
    let p = |n: &str| -> PathBuf { dir.path().join(n) };
    assert!(!is_surgery_file(&p("missing")));
    fs::write(p("empty"), b"").unwrap();
    assert!(!is_surgery_file(&p("empty")));
    fs::write(p("short"), b"SUR").unwrap();
    assert!(!is_surgery_file(&p("short")));
    fs::write(p("st"), b"{\"a\":1}").unwrap();
    assert!(!is_surgery_file(&p("st")));
    fs::write(p("magic"), b"SURGxxxx").unwrap();
    assert!(is_surgery_file(&p("magic")));
    assert!(!is_surgery_file(dir.path()));
}

// ---------------------------------------------------------------- exhaustive hostile input

#[test]
fn truncation_at_every_length_is_a_typed_error() {
    let good = sample_bytes();
    let dir = TempDir::new().unwrap();
    for n in 0..good.len() {
        match open_bytes(dir.path(), &good[..n]) {
            Err(SurgeryError::Format(_)) => {}
            Err(other) => panic!("length {n}: wrong error {other}"),
            Ok(_) => panic!("length {n}: truncated file accepted"),
        }
    }
    assert!(open_bytes(dir.path(), &good).is_ok());
    // Appended garbage shifts the footer: also refused.
    let mut longer = good.clone();
    longer.push(0);
    open_failure(&longer);
}

#[test]
fn seeded_random_corruption_never_panics() {
    let good = sample_bytes();
    let dir = TempDir::new().unwrap();
    let mut next = rng(0x5EED_1234_ABCD_EF01);
    for round in 0..1500 {
        // Raw corruption: the global hash must catch every real change.
        let mut b = good.clone();
        for _ in 0..1 + next() % 3 {
            let i = (next() % b.len() as u64) as usize;
            b[i] = next() as u8;
        }
        match open_bytes(dir.path(), &b) {
            Err(SurgeryError::Format(_)) => {}
            Err(other) => panic!("round {round}: wrong error {other}"),
            Ok(_) => assert_eq!(b, good, "round {round}: altered file accepted"),
        }
        // Corruption below the hash: footer recomputed, so the parser and the readers face
        // hostile but internally "authentic" content.
        let mut b = good.clone();
        for _ in 0..1 + next() % 3 {
            let i = PRE + (next() % (b.len() - PRE - FOOT) as u64) as usize;
            b[i] = next() as u8;
        }
        refresh_footer(&mut b, true);
        match open_bytes(dir.path(), &b) {
            Err(SurgeryError::Format(_)) => {}
            Err(other) => panic!("round {round}: wrong error {other}"),
            Ok(mut r) => match read_all(&mut r) {
                Ok(()) | Err(SurgeryError::Format(_)) => {}
                Err(other) => panic!("round {round}: wrong read error {other}"),
            },
        }
    }
}

// ---------------------------------------------------------------- schema and dependencies

#[test]
fn schema_validates_produced_headers() {
    let schema = header_schema();
    assert!(jsonschema::draft7::meta::is_valid(&schema));
    let validator = jsonschema::draft7::new(&schema).unwrap();
    let dir = TempDir::new().unwrap();
    for (i, (dtype, comp, ts)) in [
        (FactorDtype::F16, deflate(6), None),
        (FactorDtype::Bf16, stored(), Some("2026-05-06T07:08:09Z")),
        (FactorDtype::F32, deflate(1), None),
    ]
    .into_iter()
    .enumerate()
    {
        let path = dir.path().join(format!("s{i}.surgery"));
        write_sample(&path, dtype, comp, ts);
        let bytes = fs::read(&path).unwrap();
        let value: Value = serde_json::from_slice(&bytes[PRE..PRE + header_len(&bytes)]).unwrap();
        let errors: Vec<String> = validator
            .iter_errors(&value)
            .map(|e| e.to_string())
            .collect();
        assert!(errors.is_empty(), "{errors:?}");
        if ts.is_some() {
            assert!(value.get("provenance").is_some());
        }
    }
}

#[test]
fn schema_rejects_bad_headers() {
    let validator = jsonschema::draft7::new(&header_schema()).unwrap();
    let good: Value = {
        let bytes = sample_bytes();
        serde_json::from_slice(&bytes[PRE..PRE + header_len(&bytes)]).unwrap()
    };
    assert!(validator.is_valid(&good));
    let mut bad = good.clone();
    bad["tensors"][0]["base_sha256"] = Value::from("nothex");
    assert!(!validator.is_valid(&bad));
    let mut bad = good.clone();
    bad["factor_dtype"] = Value::from("f64");
    assert!(!validator.is_valid(&bad));
    let mut bad = good.clone();
    bad["tensors"][0]["blobs"]["u"]["raw_len"] = Value::from(0);
    assert!(!validator.is_valid(&bad));
    let mut bad = good;
    bad.as_object_mut().unwrap().remove("rank");
    assert!(!validator.is_valid(&bad));
}

#[test]
fn compression_is_miniz_oxide_only() {
    let manifest =
        fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")).unwrap();
    assert!(manifest.contains("miniz_oxide"));
    for banned in ["zstd", "flate2", "libz", "lz4", "brotli", "bzip2"] {
        assert!(!manifest.contains(banned), "{banned} is not allowed");
    }
}
