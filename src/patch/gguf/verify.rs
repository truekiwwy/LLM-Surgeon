//! Per-row patching with fidelity measurements, and the read-back check after writing.
//!
//! For each element: `before` = dequantized original, `exact = before + delta` (the single
//! IEEE addition of the SafeTensors kernels, `delta = sign * (alpha * v)`), `after` =
//! dequantized requantized bytes. Sums are accumulated in f64 per row, then folded in row
//! order, so the report does not depend on the thread count.

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::mapping::Layout;
use super::quant::{dequantize_row, requantize_row, scales_finite, Requant};
use crate::kernel::RowDeltas;

pub const WARN_RATIO: f64 = 0.5;
pub const WARN_CORRELATION: f64 = 0.8;
pub const WEAK_PATCH_WARNING: &str = "le pas de quantification est trop grand pour ce patch \
    (un décalage inférieur au demi-pas disparaît ou devient bruit) : augmentez alpha ou \
    patchez en SafeTensors puis re-quantifiez avec llama.cpp";
/// Saturated share above which the clipping is reported (keep-scales keeps the original
/// grid, so a shift past a block's range is clipped; Q8_0 has no headroom at all).
pub const WARN_SATURATED_FRACTION: f64 = 1e-3;
pub const SATURATION_WARNING: &str = "valeurs écrêtées aux bornes de la grille d'origine \
    (keep-scales ne change aucune échelle) : --requant recompute recalcule les échelles des \
    blocs touchés";

/// Why a row cannot be patched. Raised before the row (and its chunk) is written.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RowError {
    #[error("échelle f16 non finie (NaN/Inf) dans le tenseur d'origine")]
    OriginalScale,
    #[error("valeur non finie (NaN/Inf) dans le tenseur d'origine")]
    OriginalValue,
    #[error("échelle recalculée hors du domaine f16 (Inf) : patch trop fort pour ce format")]
    RecomputedScale,
}

/// Running sums of one tensor.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Fidelity {
    pub elements: u64,
    pub intended_sq: f64,
    pub realized_sq: f64,
    pub dot: f64,
    pub err_sq: f64,
    pub exact_sq: f64,
    pub unchanged: u64,
    pub saturated: u64,
    pub nan_or_inf: u64,
    pub max_abs_delta: f32,
}

impl Fidelity {
    pub fn merge(&mut self, o: &Fidelity) {
        self.elements += o.elements;
        self.intended_sq += o.intended_sq;
        self.realized_sq += o.realized_sq;
        self.dot += o.dot;
        self.err_sq += o.err_sq;
        self.exact_sq += o.exact_sq;
        self.unchanged += o.unchanged;
        self.saturated += o.saturated;
        self.nan_or_inf += o.nan_or_inf;
        self.max_abs_delta = self.max_abs_delta.max(o.max_abs_delta);
    }

    pub fn modified(&self) -> u64 {
        self.elements - self.unchanged
    }
}

/// `data.gguf.tensors[]` of the apply report.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TensorFidelity {
    pub name: String,
    pub ggml_type: String,
    pub rows: usize,
    pub cols: usize,
    pub elements: u64,
    /// ‖α·v‖ broadcast over the tensor.
    pub intended_shift_norm: f64,
    /// ‖dequant(after) − dequant(before)‖.
    pub realized_shift_norm: f64,
    /// `None` when nothing was intended (alpha 0).
    pub realized_over_intended: Option<f64>,
    /// Cosine (uncentered correlation) of the intended and realized shifts; 0 when
    /// nothing was realized.
    pub correlation_intended_realized: Option<f64>,
    pub unchanged_fraction: f64,
    pub saturated_elements: u64,
    /// ‖dequant(after) − (before + α·v)‖² / ‖before + α·v‖².
    pub nmse_vs_exact: Option<f64>,
    pub nan_or_inf_produced: u64,
    pub warning: Option<String>,
    /// project_out: `‖R(deq W')‖ / ‖R(W)‖`, `R = Bᵀ·` (embeddings: `· B`), the share of the
    /// read component still there (`1 − α` ideally). Absent for add / subtract.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub residual_fraction: Option<f64>,
    /// project_out: `‖R(deq W' − W_exact)‖ / ‖R(W)‖`, the part brought back by the
    /// requantization, bounded by `--residual-target`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requant_residual_fraction: Option<f64>,
    /// project_out: quantization passes measured (1 + refinements).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refine_passes: Option<usize>,
}

impl TensorFidelity {
    pub fn new(name: &str, ggml_type: String, layout: &Layout, f: &Fidelity) -> Self {
        let intended = f.intended_sq.sqrt();
        let realized = f.realized_sq.sqrt();
        let ratio = (intended > 0.0).then(|| realized / intended);
        let correlation = (intended > 0.0).then(|| match realized > 0.0 {
            true => f.dot / (intended * realized),
            false => 0.0,
        });
        let weak = ratio.is_some_and(|r| r < WARN_RATIO)
            || correlation.is_some_and(|c| c < WARN_CORRELATION);
        let clipped =
            f.elements > 0 && f.saturated as f64 > WARN_SATURATED_FRACTION * f.elements as f64;
        let warnings: Vec<String> = [
            weak.then(|| WEAK_PATCH_WARNING.to_owned()),
            clipped.then(|| format!("{} {SATURATION_WARNING}", f.saturated)),
        ]
        .into_iter()
        .flatten()
        .collect();
        Self {
            name: name.to_owned(),
            ggml_type,
            rows: layout.rows,
            cols: layout.cols,
            elements: f.elements,
            intended_shift_norm: intended,
            realized_shift_norm: realized,
            realized_over_intended: ratio,
            correlation_intended_realized: correlation,
            unchanged_fraction: match f.elements {
                0 => 1.0,
                n => f.unchanged as f64 / n as f64,
            },
            saturated_elements: f.saturated,
            nmse_vs_exact: (f.exact_sq > 0.0).then(|| f.err_sq / f.exact_sq),
            nan_or_inf_produced: f.nan_or_inf,
            warning: (!warnings.is_empty()).then(|| warnings.join(" ; ")),
            residual_fraction: None,
            requant_residual_fraction: None,
            refine_passes: None,
        }
    }

    pub fn push_warning(&mut self, warning: String) {
        self.warning = Some(match self.warning.take() {
            Some(w) => format!("{w} ; {warning}"),
            None => warning,
        });
    }
}

/// Patches one row in place (`row` holds the original bytes, receives the new ones).
pub fn patch_row(
    layout: &Layout,
    mode: Requant,
    row: &mut [u8],
    deltas: RowDeltas<'_>,
) -> Result<Fidelity, RowError> {
    let fill = |_: &[f32], out: &mut [f32]| match deltas {
        RowDeltas::Splat(d) => out.fill(d),
        RowDeltas::Each(ds) => {
            for (c, d) in out.iter_mut().enumerate() {
                *d = ds.get(c).copied().unwrap_or(0.0);
            }
        }
    };
    patch_row_with(layout, mode, row, fill, |_| {}).map(|r| r.fidelity)
}

/// The decoded values of an original row, its scales and values checked finite.
pub fn decode_checked(layout: &Layout, row: &[u8]) -> Result<Vec<f32>, RowError> {
    if !scales_finite(layout.codec, row) {
        return Err(RowError::OriginalScale);
    }
    let mut values = vec![0.0f32; layout.cols];
    dequantize_row(layout.codec, row, &mut values);
    match values.iter().all(|x| x.is_finite()) {
        true => Ok(values),
        false => Err(RowError::OriginalValue),
    }
}

/// A row patched by [`patch_row_with`]: its fidelity sums, the float result and the
/// dequantized written values.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchedRow {
    pub fidelity: Fidelity,
    pub before: Vec<f32>,
    pub exact: Vec<f32>,
    pub after: Vec<f32>,
}

/// Patches one row in place. `deltas(before, out)` fills the float shift, so `exact =
/// before + delta` (one IEEE addition, as the SafeTensors kernels); `correct` turns a copy
/// of `exact` into the values actually requantized (the refinement of `project_out`). The
/// fidelity always measures the written values against `exact`.
pub fn patch_row_with(
    layout: &Layout,
    mode: Requant,
    row: &mut [u8],
    deltas: impl FnOnce(&[f32], &mut [f32]),
    correct: impl FnOnce(&mut [f32]),
) -> Result<PatchedRow, RowError> {
    let codec = layout.codec;
    let before = decode_checked(layout, row)?;
    let mut delta = vec![0.0f32; layout.cols];
    deltas(&before, &mut delta);
    let exact: Vec<f32> = before.iter().zip(&delta).map(|(&w, &d)| w + d).collect();
    let mut target = exact.clone();
    correct(&mut target);
    let requant = requantize_row(codec, mode, &before, &target, row);
    if requant.nonfinite_scale || !scales_finite(codec, row) {
        return Err(RowError::RecomputedScale);
    }
    let mut after = vec![0.0f32; layout.cols];
    dequantize_row(codec, row, &mut after);

    let mut f = Fidelity {
        elements: layout.cols as u64,
        saturated: requant.saturated,
        ..Fidelity::default()
    };
    for (c, (&b, &a)) in before.iter().zip(&after).enumerate() {
        if a.to_bits() == b.to_bits() {
            f.unchanged += 1;
        }
        let d = f64::from(delta[c]);
        f.intended_sq += d * d;
        if !a.is_finite() {
            f.nan_or_inf += 1;
            continue;
        }
        f.max_abs_delta = f.max_abs_delta.max((a - b).abs());
        let realized = f64::from(a) - f64::from(b);
        let exact = f64::from(b) + d;
        let err = f64::from(a) - exact;
        f.realized_sq += realized * realized;
        f.dot += d * realized;
        f.err_sq += err * err;
        f.exact_sq += exact * exact;
    }
    Ok(PatchedRow {
        fidelity: f,
        before,
        exact,
        after,
    })
}

pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

pub fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// A tensor as written: its range, layout and the SHA-256 of the bytes computed while
/// writing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Written {
    pub name: String,
    pub range: std::ops::Range<usize>,
    pub layout: Layout,
    pub sha256: [u8; 32],
}

/// Read-back check of a patched file: header region unchanged, every patched tensor holds
/// exactly the bytes computed during the write, and all its scales and values are finite.
pub fn verify_file(
    bytes: &[u8],
    data_start: usize,
    header_sha256: &[u8; 32],
    written: &[Written],
) -> Result<(), String> {
    let header = bytes
        .get(..data_start)
        .ok_or_else(|| "fichier plus court que son en-tête".to_owned())?;
    if &sha256(header) != header_sha256 {
        return Err("en-tête (métadonnées, table des tenseurs) modifié".into());
    }
    for w in written {
        let data = bytes
            .get(w.range.clone())
            .ok_or_else(|| format!("{} : hors du fichier", w.name))?;
        if sha256(data) != w.sha256 {
            return Err(format!(
                "{} : octets relus différents de ceux écrits",
                w.name
            ));
        }
        let mut values = vec![0.0f32; w.layout.cols];
        for row in data.chunks_exact(w.layout.row_bytes().max(1)) {
            dequantize_row(w.layout.codec, row, &mut values);
            if !scales_finite(w.layout.codec, row) || values.iter().any(|v| !v.is_finite()) {
                return Err(format!("{} : NaN/Inf après écriture", w.name));
            }
        }
    }
    Ok(())
}
