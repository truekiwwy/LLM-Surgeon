# `.surgery` v1 specification

A `.surgery` file stores the difference between a tuned checkpoint and its base checkpoint, as one
low-rank factorisation per 2D weight and an exact delta per 1D tensor. It is produced by
`surgeon diff` and consumed by `surgeon apply`.

Reference implementation: `src/surgery/{format,writer,reader,schema}.rs`. This document is the
normative description of the bytes; the header schema is also available with
`surgeon schema --kind surgery`.

## 1. What a `.surgery` is, and is not

- It is an **approximation** of a fine-tune. For each 2D tensor, `dW = W_tuned - W_base` is replaced
  by its best rank-`r` approximation `U diag(S) V`. The relative error
  `||dW - U diag(S) V||_F / ||dW||_F` is recorded per tensor (`rel_error`) and `energy` records the
  fraction of `||dW||_F^2` captured.
- The patch is **exact only when `dW` has rank <= r**. A full fine-tune usually does not: expect a
  non-zero `rel_error`, larger for tensors that moved in many directions.
- It is **not small**. Size is about `sum over tensors of r (m + n) x bytes_per_factor` (before
  deflate, which gains little on float factors). A 7B model at rank 16 in f16 is in the hundreds of
  MB, not "a few MB". 1D tensors are stored whole in f32.
- Factors are stored in f16, bf16 or f32 (`factor_dtype`, a single value per file). f16 values are
  rounded to nearest even and a value that does not fit is an error (never an infinity); the writer
  caller may then retry with bf16.

## 2. Byte layout

All integers are little-endian. Offsets are measured from the first byte of the file.

| Offset | Size | Field | Value |
|---|---|---|---|
| 0 | 4 | magic | `53 55 52 47` (`"SURG"`) |
| 4 | 2 | `version` (u16) | `1` |
| 6 | 2 | `flags` (u16) | `0` (any other value is refused) |
| 8 | 4 | `reserved` (u32) | `0` (any other value is refused) |
| 12 | 8 | `header_len` (u64) | length `H` of the JSON header in bytes |
| 20 | `H` | header | UTF-8 JSON (section 3) |
| 20 + `H` | `B` | blobs | concatenated blob bytes (section 4) |
| `F - 36` | 32 | `sha256` | SHA-256 of every byte from offset 0 to `F - 36` (exclusive) |
| `F - 4` | 4 | footer magic | `"SURG"` (not hashed) |

`F` is the file size. `B = F - 36 - (20 + H)`. The minimum file is 56 bytes.

The global hash covers the preamble, the header and all blobs, but not the 36 footer bytes.

### Blob layout

Writers pack blobs back to back, in this order: tensors in header order, and inside a tensor
`u`, `s`, `v` (low-rank) or `delta` (full delta). Readers only require that the blobs cover the blob
section exactly, in any order: no gap, no overlap, no unreferenced byte. Every blob is addressed by
its absolute `offset` and `comp_len`.

Because the header (with these offsets) precedes the blobs, the writer spools blobs to a temporary
file, then computes `H` (the digit counts of the offsets depend on `H`; it iterates to the fixed
point) and assembles the final file.

## 3. Header (canonical JSON)

The writer emits **canonical JSON**: UTF-8, no insignificant whitespace, object keys sorted by byte
order at every level, integers in plain decimal, floats in the shortest round-trip form. Equal
inputs therefore give equal bytes. A reader accepts any valid JSON that satisfies the rules below
(whitespace, key order) and ignores unknown keys.

Example: a complete 877-byte file holding one 4 x 3 tensor at rank 2, f16 factors, `compression
none` (produced by the test writer; line breaks added here for reading, the real header has none).
`H = 789`, so the blobs start at offset 809 (`20 + 789`): `u` 16 bytes at 809, `s` 4 bytes at 825,
`v` 12 bytes at 829, then the 32-byte digest at 841 and `"SURG"` at 873.

```json
{"base":{"name":"base","tensor_count":10},
 "compression":{"algo":"none","level":0},
 "created_with":"test 0.0","factor_dtype":"f16","format_version":1,"rank":3,
 "tensors":[{"base_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "blobs":{"s":{"comp_len":4,"offset":825,"raw_len":4,"sha256":"e10013643b25685d17a2eccd73f9894b5272ae4b8b1236394c7cccd701a8282c"},
           "u":{"comp_len":16,"offset":809,"raw_len":16,"sha256":"653c22774370a2d7b11c3b518f16fded131d038d08e7971cb14a2fa98548f1da"},
           "v":{"comp_len":12,"offset":829,"raw_len":12,"sha256":"9a3f4189e5f256d65d3e3cdb92ab733ffc1f1a0f5ffaea7753b25d9f02727cf6"}},
  "energy":0.5,"kind":"lowrank","name":"model.layers.0.mlp.down_proj.weight","rank":2,
  "rel_error":0.125,"shape":[4,3],"target_dtype":"BF16"}],
 "tuned":{"name":"tuned"}}
```

Keys are sorted (`s` before `u`) while the blobs are stored in `u`, `s`, `v` order: the order of the
bytes is given by `offset`, not by the order of the keys. When present, `provenance` sorts between
`format_version` and `rank`.

| Field | Type | Meaning |
|---|---|---|
| `format_version` | u16 | `1`; must equal the preamble `version` |
| `created_with` | string | producer name and version |
| `base.name`, `base.tensor_count` | string, usize | base checkpoint name and its tensor count |
| `tuned.name` | string | tuned checkpoint name |
| `rank` | usize, 1..=65536 | requested rank; the stored rank of any tensor is at most this |
| `factor_dtype` | `"f16"` \| `"bf16"` \| `"f32"` | storage dtype of `u`, `s`, `v` |
| `compression` | `{algo, level}` | `algo` is `"deflate"` (raw deflate stream, no zlib header, `level` 0..=10) or `"none"` (`level` must be 0) |
| `provenance` | `{timestamp}` | present **only** when the producer was asked for a timestamp (`diff --timestamp`); absent otherwise, which keeps the file reproducible |
| `tensors[]` | array | one entry per patched tensor |

Entry of `tensors[]`:

| Field | Type | Meaning |
|---|---|---|
| `name` | string, 1..=1024 bytes, no control character | tensor name in the target checkpoint; unique in the file |
| `shape` | usize array, 1..=8 dims, each >= 1 | shape of the target tensor |
| `kind` | `"lowrank"` \| `"full_delta"` | `lowrank` needs a 2D shape `[m, n]` |
| `rank` | usize | stored rank `r` (1..=header `rank`) for `lowrank`; `0` for `full_delta` |
| `target_dtype` | `"BF16"` \| `"F16"` \| `"F32"` | dtype of the tensor the patch is applied to |
| `base_sha256` | 64 lowercase hex | SHA-256 of the base tensor bytes the delta was computed against |
| `rel_error` | finite f64 >= 0 | relative Frobenius error of the approximation (0 for `full_delta`) |
| `energy` | finite f64 >= 0 | captured energy fraction |
| `blobs` | object | `u`, `s`, `v` for `lowrank`; `delta` for `full_delta` |

Blob reference (`blobs.u|s|v|delta`):

| Field | Type | Meaning |
|---|---|---|
| `offset` | u64 | absolute offset of the stored bytes |
| `comp_len` | u64 >= 1 | stored length (deflated, or raw when `algo = none`) |
| `raw_len` | u64 >= 1 | length after inflation |
| `sha256` | 64 lowercase hex | SHA-256 of the **raw (inflated) bytes** |

### Blob contents

| Blob | Shape | Element type | Expected `raw_len` |
|---|---|---|---|
| `u` | `[m, r]` row-major | `factor_dtype` | `m * r * sz` |
| `s` | `[r]` | `factor_dtype` | `r * sz` |
| `v` | `[r, n]` row-major | `factor_dtype` | `r * n * sz` |
| `delta` | `shape` row-major | f32 | `numel * 4` |

`sz` is 2 for f16/bf16 and 4 for f32. The reconstructed delta is `U diag(S) V`, added to the base
tensor. Conversion f32 to f16/bf16 is round-to-nearest-even (same as `engine::apply_vector_shift`).
Stored values must be finite.

The JSON schema (draft 07, generated from the Rust types by `schemars`) is printed by
`surgeon schema --kind surgery`. It covers the field types, the enums, the hex patterns and the
numeric bounds above; the cross-field rules of section 5 are enforced by the reader.

## 4. Compression and determinism

- `deflate`: raw deflate stream produced by the pure Rust `miniz_oxide` crate (`compress_to_vec`),
  one independent stream per blob, level fixed in the header (default 6). No C library is used.
- Same input and same parameters give the same bytes: no timestamp, no randomness, fixed level, fixed
  tensor order, sorted JSON keys. The output of the deflate encoder is deterministic for a given
  `miniz_oxide` version; a different version of the crate may legitimately produce different (still
  valid) compressed bytes, so reproducibility is guaranteed per build, not across encoder upgrades.
  Readers never depend on the encoder.
- Blob hashes are over the raw bytes (so they do not change with the compressor); the global hash is
  over the stored bytes (it detects any alteration of the file).

## 5. Validation rules (reader)

The file is **untrusted**. `SurgeryReader::open` runs these checks, cheapest first, and returns a
typed `FormatError` (exit code 2, error kind `invalid_surgery`) without ever panicking:

| Check | Error |
|---|---|
| fewer than 4 bytes, or file shorter than 56 bytes | `Truncated` |
| magic is not `SURG` (also footer magic) | `BadMagic` |
| `version != 1` (unknown major refused) | `UnsupportedVersion` |
| `flags != 0` or `reserved != 0` | `UnsupportedFlags` |
| `header_len` larger than `F - 56` | `HeaderLenMismatch` |
| `header_len` above 64 MiB, JSON invalid or off-schema, duplicate key, rule below violated | `InvalidHeader` |
| `raw_len` different from the size implied by shape x dtype x rank | `RawLenMismatch` |
| blob outside `[20 + H, F - 36)`, or `offset + comp_len` overflow | `OffsetOutOfBounds` |
| two blobs overlap | `BlobOverlap` |
| gap or unreferenced bytes in the blob section | `InvalidHeader` |
| global SHA-256 differs, verified by streaming the file in 1 MiB chunks **before** `open` returns | `GlobalHashMismatch` |

Header rules (`InvalidHeader`): `format_version` equals the preamble version; `rank` in 1..=65536;
compression level valid for the algorithm; unique tensor names; name, shape (1..=8 dims, each >= 1,
no overflow of the element count), `target_dtype`, `base_sha256`, finite non-negative `rel_error` and
`energy`; `lowrank` is 2D with `1 <= rank <= header rank` and exactly `u`, `s`, `v`; `full_delta` has
`rank = 0` and exactly `delta`; a blob is at most 2 GiB raw; with `none`, `comp_len = raw_len`; with
`deflate`, `raw_len <= 1032 x comp_len` (the deflate ceiling), so a blob cannot declare a size its
stored bytes could never produce.

Reading a tensor (`read_lowrank`, `read_full_delta`) additionally:

- looks the requested entry up in the validated header (a caller-supplied entry is never trusted);
- reads exactly `comp_len` bytes (already bounded by the real file size);
- inflates with a **hard limit equal to the declared `raw_len`**: a longer stream is
  `DecompressionBomb`, an invalid stream `DecompressionFailed`, a shorter one `RawLenMismatch`;
  memory grows with the bytes actually produced, never from an unverified value;
- verifies the blob SHA-256 over the raw bytes (`BlobHashMismatch`);
- converts to f32 and refuses NaN/Inf bit patterns (`NonFiniteValue`).

Nothing is applied to a model before `open` succeeded, and the global hash is checked at `open`.

The writer applies the same header validation to its own output before writing it (self-check),
rejects non-finite input values, and raises `FactorOverflow` when a finite value does not fit the
factor dtype (f16 above 65504, or bf16 rounding past `f32::MAX`).

## 6. Writer guarantees

- Nothing is visible at the output path before `finish`: blobs go to a hidden spool file next to the
  output, the final file is assembled in a second hidden temporary file, synced, then renamed over
  the target. A writer dropped without `finish` removes its scratch files.
- With `overwrite = false` the writer fails when the target exists (checked at creation and again just
  before the rename; the residual race between the last check and the rename is not closed).
- Blobs are compressed one at a time: peak memory is about one tensor, not the whole patch. A
  `full_delta` of a very large 2D tensor (`--on-high-error full`) needs its raw and compressed bytes
  in memory at once.

## 7. Version policy

- `version` (u16) is the major version of the container. This document describes version 1.
- A change that older readers cannot handle correctly (layout, meaning or hashing of a field, a new
  required key, a new blob role, a new compression algorithm) bumps `version`. Readers refuse any
  version they do not know (`UnsupportedVersion`) instead of guessing.
- Optional keys may be added to the header within a version: readers ignore unknown keys, writers must
  not make their absence change the meaning of existing fields.
- `flags` and `reserved` are zero in version 1 and reserved for future use; readers refuse non-zero
  values so that they cannot be silently misinterpreted.

## 8. Limits of version 1

- Per-blob cap of 2 GiB raw (a tensor whose factors exceed it is refused).
- Header at most 64 MiB.
- Whole-blob inflate: a blob is inflated in memory (bounded by `raw_len`), not streamed.
- A `.surgery` is not exposed through the MCP `surgeon_apply` tool.
