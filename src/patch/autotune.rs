//! `surgeon autotune`: per-layer coefficients α_l of a steering patch, fitted by ridge
//! regression on measured responses. The naive problem `min ‖Y − Σ α_l (W_l v_l)‖² + λ‖α‖²`
//! leaves `Y` and `W_l v_l` undefined; the well-posed version implemented here:
//!
//! ```text
//! readout   s_i = ⟨pool(h_i^R), v⟩   (prompt i, site/pooling/template of the vector, layer R)
//! target    y_i = t_i − s_i⁰         (t: neg_mean_proj for suppress, pos_mean_proj for
//!                                    amplify, --target-projection, or the prompt's target)
//! response  F_il = (s_i(δ_l) − s_i⁰) / δ, where s(δ_l) shifts the in-memory weight `proj` of
//!           layer l by `subtract δ·v` with the engine's own `apply_vector_shift` on the bytes
//!           of the stored dtype (bf16/f16 round to nearest even exactly as on disk)
//! ridge     α = (FᵀF + λI)⁻¹ Fᵀ y (Cholesky; λ > 0 makes it SPD), λ fixed or picked on a
//!           grid 1e-6..1e2 (17 points) by closed-form LOOCV e_i / (1 − h_ii)
//! ```
//!
//! Layers above R cannot change the readout: they are excluded with a warning. The fitted
//! α are applied all at once in memory and the readout is measured again: the report gives
//! the predicted and the actual shift. Side effects (KL, perplexity) are not measured here:
//! use `surgeon bench`. The evaluation prompts serve both to fit and to measure, so
//! `M < 2L` warns of probable overfitting.
//!
//! Unlike the rest of `patch`, this module drives the extraction model (`extract`,
//! `engine`, `hardware`, `linalg`): it produces a `PatchFile`, validated by
//! `validate_patch` before it is written.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::schema::{Axis, LayersField, Operation, OperationKind, PatchFile, Projection};
use super::validate::{validate_patch, ValidationOptions, MAX_ABS_ALPHA};
use crate::engine::{self, BroadcastAxis, ReadOnlyModel, ShiftOp};
use crate::extract::{
    check_tensors, choose_dtype, companion_path, dtype_name, load_config, load_tensors,
    load_tokenizer, plan_memory, pool, required_tensors, tokenize, write_atomic, ChatTemplate,
    DTypeChoice, ExtractError, HookedQwen2, Pooling, Qwen2Config, SafetensorsDir, Site,
    TensorSource,
};
use crate::hardware::HardwareProbe;
use crate::linalg::cholesky::Cholesky;
use crate::linalg::LinalgError;

/// λ grid of `--l2-reg auto`: 10^(-6 + k/2), k = 0..=16.
pub const AUTO_GRID_POINTS: usize = 17;
pub const LINEARITY_TOLERANCE: f64 = 0.10;
const LINEARITY_LAYERS: usize = 3;

#[derive(Debug, Clone, thiserror::Error)]
pub enum AutotuneError {
    #[error("argument invalide : {0}")]
    InvalidArgument(String),
    #[error("{} : {message}", .path.display())]
    Input { path: PathBuf, message: String },
    #[error("{} existe déjà (--overwrite pour le remplacer)", .0.display())]
    OutputExists(PathBuf),
    #[error("ressources insuffisantes : {0}")]
    InsufficientResources(String),
    #[error("modèle : {0}")]
    Model(String),
    #[error("résolution : {0}")]
    Solve(String),
    #[error("calcul : {0}")]
    Compute(String),
    #[error("écriture de {} : {message}", .path.display())]
    Io { path: PathBuf, message: String },
    #[error("autotune annulé, aucun fichier écrit")]
    Cancelled,
}

impl AutotuneError {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::InvalidArgument(_) | Self::Input { .. } => "invalid_input",
            Self::OutputExists(_) => "output_exists",
            Self::InsufficientResources(_) => "insufficient_resources",
            Self::Model(_) => "model_error",
            Self::Solve(_) => "solve_failed",
            Self::Compute(_) | Self::Io { .. } => "autotune_failed",
            Self::Cancelled => "cancelled",
        }
    }
}

impl From<ExtractError> for AutotuneError {
    fn from(e: ExtractError) -> Self {
        match e {
            ExtractError::InvalidArgument(m) | ExtractError::Prompts { message: m, .. } => {
                Self::InvalidArgument(m)
            }
            ExtractError::OutputExists(p) => Self::OutputExists(p),
            ExtractError::InsufficientResources(m) => Self::InsufficientResources(m),
            ExtractError::Cancelled => Self::Cancelled,
            ExtractError::Io { path, message } => Self::Io { path, message },
            ExtractError::Compute(m) => Self::Compute(m),
            other => Self::Model(other.to_string()),
        }
    }
}

impl From<LinalgError> for AutotuneError {
    fn from(e: LinalgError) -> Self {
        Self::Solve(e.to_string())
    }
}

impl From<candle_core::Error> for AutotuneError {
    fn from(e: candle_core::Error) -> Self {
        Self::Compute(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, AutotuneError>;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema, clap::ValueEnum,
)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    /// Amener la projection vers neg_mean_proj.
    #[default]
    Suppress,
    /// Amener la projection vers pos_mean_proj.
    Amplify,
    /// Amener la projection vers --target-projection.
    Target,
}

/// `--l2-reg <f64|auto>`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum L2Reg {
    Fixed(f64),
    Auto,
}

impl std::str::FromStr for L2Reg {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.trim() {
            "auto" => Ok(Self::Auto),
            other => other
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite() && *v > 0.0)
                .map(Self::Fixed)
                .ok_or_else(|| format!("« {other} » : un nombre > 0 ou auto")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AutotuneRequest {
    pub model: PathBuf,
    pub vector: PathBuf,
    pub eval_dataset: PathBuf,
    /// Inclusive `(A, B)`; default every layer up to the readout.
    pub layer_range: Option<(usize, usize)>,
    pub readout_layer: Option<usize>,
    pub proj: Projection,
    pub objective: Objective,
    pub target_projection: Option<f64>,
    pub l2_reg: L2Reg,
    pub probe_alpha: f64,
    pub alpha_max: f64,
    pub min_alpha: f64,
    pub max_forwards: usize,
    pub max_tokens: usize,
    pub dtype: DTypeChoice,
    pub site: Option<Site>,
    pub pooling: Option<Pooling>,
    pub chat_template: Option<ChatTemplate>,
    pub out: Option<PathBuf>,
    pub overwrite: bool,
}

/// Parses `A-B` or `A` (inclusive).
pub fn parse_layer_range(s: &str) -> std::result::Result<(usize, usize), String> {
    let num = |p: &str| {
        p.trim()
            .parse::<usize>()
            .map_err(|_| format!("couche invalide « {p} »"))
    };
    let (a, b) = match s.split_once('-') {
        Some((a, b)) => (num(a)?, num(b)?),
        None => (num(s)?, num(s)?),
    };
    match a <= b {
        true => Ok((a, b)),
        false => Err(format!("plage {a}-{b} décroissante")),
    }
}

pub enum AutotuneEvent {
    Planned { forwards: usize, layers: usize },
    Forward { done: usize, total: usize },
    Solve,
}

/// Counts forwards for the progress events.
struct Progress<'a> {
    done: usize,
    total: usize,
    on_event: &'a mut dyn FnMut(AutotuneEvent),
}

impl Progress<'_> {
    fn tick(&mut self) {
        self.done += 1;
        (self.on_event)(AutotuneEvent::Forward {
            done: self.done,
            total: self.total,
        });
    }
}

// ------------------------------------------------------------------ ridge

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct RidgeFit {
    /// λ retenu.
    pub lambda: f64,
    /// Coefficients avant écrêtage, un par couche candidate.
    pub alpha: Vec<f64>,
    /// R² non centré 1 − Σe²/Σy² (pas d'intercept : part du décalage visé expliquée) ; null si y = 0.
    pub r2: Option<f64>,
    /// RMSE des résidus y − Fα.
    pub rmse: f64,
    /// RMSE leave-one-out (forme fermée e_i/(1−h_ii)) ; null si un h_ii ≈ 1.
    pub loocv_rmse: Option<f64>,
    /// Estimation de κ(FᵀF + λI) (borne inférieure, diagonale de Cholesky).
    pub cond_estimate: f64,
    /// ‖α‖₂ avant écrêtage.
    pub alpha_norm: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct GridPoint {
    pub lambda: f64,
    /// null si la factorisation a échoué ou si LOOCV est indéfini.
    pub loocv_rmse: Option<f64>,
}

/// Ridge on the row-major M×L `f`. Every sum runs in index order.
pub fn ridge(
    f: &[f64],
    m: usize,
    l: usize,
    y: &[f64],
    lambda: f64,
) -> std::result::Result<RidgeFit, LinalgError> {
    if f.len() != m * l || y.len() != m || l == 0 || m == 0 {
        return Err(LinalgError::Shape(format!(
            "F {m}×{l} et y de taille {}",
            y.len()
        )));
    }
    let row = |i: usize| &f[i * l..(i + 1) * l];
    let mut a = vec![0.0; l * l];
    let mut rhs = vec![0.0; l];
    for (i, yi) in y.iter().enumerate() {
        let fi = row(i);
        for p in 0..l {
            rhs[p] += fi[p] * yi;
            for q in 0..l {
                a[p * l + q] += fi[p] * fi[q];
            }
        }
    }
    for p in 0..l {
        a[p * l + p] += lambda;
    }
    let chol = Cholesky::factor(&a, l)?;
    let alpha = chol.solve(&rhs);
    let residuals: Vec<f64> = (0..m)
        .map(|i| y[i] - row(i).iter().zip(&alpha).map(|(x, w)| x * w).sum::<f64>())
        .collect();
    let ss_res: f64 = residuals.iter().map(|e| e * e).sum();
    // Uncentered: the model has no intercept and y is a shift whose mean matters, so the
    // centered SS_tot would make R² negative for fits that explain most of the shift.
    let ss_tot: f64 = y.iter().map(|v| v * v).sum();
    let mut loo = 0.0;
    let mut defined = true;
    for (i, e) in residuals.iter().enumerate() {
        let z = chol.solve(row(i));
        let h: f64 = row(i).iter().zip(&z).map(|(x, w)| x * w).sum();
        match 1.0 - h > 1e-12 {
            true => loo += (e / (1.0 - h)).powi(2),
            false => defined = false,
        }
    }
    Ok(RidgeFit {
        lambda,
        alpha_norm: alpha.iter().map(|x| x * x).sum::<f64>().sqrt(),
        alpha,
        r2: (ss_tot > 0.0).then(|| 1.0 - ss_res / ss_tot),
        rmse: (ss_res / m as f64).sqrt(),
        loocv_rmse: defined.then(|| (loo / m as f64).sqrt()),
        cond_estimate: chol.kappa,
    })
}

pub fn auto_grid() -> Vec<f64> {
    (0..AUTO_GRID_POINTS)
        .map(|k| 10f64.powf(-6.0 + 0.5 * k as f64))
        .collect()
}

/// The grid point with the lowest LOOCV RMSE (the first one on a tie); points whose
/// factorization fails (ill-conditioned) are skipped and reported with `null`.
pub fn ridge_auto(
    f: &[f64],
    m: usize,
    l: usize,
    y: &[f64],
) -> std::result::Result<(RidgeFit, Vec<GridPoint>), LinalgError> {
    let mut best: Option<RidgeFit> = None;
    let mut grid = Vec::with_capacity(AUTO_GRID_POINTS);
    let mut last_error = None;
    for lambda in auto_grid() {
        match ridge(f, m, l, y, lambda) {
            Ok(fit) => {
                grid.push(GridPoint {
                    lambda,
                    loocv_rmse: fit.loocv_rmse,
                });
                let better = match (&best, fit.loocv_rmse) {
                    (_, None) => false,
                    (None, Some(_)) => true,
                    (Some(b), Some(score)) => b.loocv_rmse.is_none_or(|s| score < s),
                };
                if better {
                    best = Some(fit);
                }
            }
            Err(e) => {
                grid.push(GridPoint {
                    lambda,
                    loocv_rmse: None,
                });
                last_error = Some(e);
            }
        }
    }
    match best {
        Some(fit) => Ok((fit, grid)),
        None => Err(last_error.unwrap_or(LinalgError::Shape("aucun λ exploitable".into()))),
    }
}

// ------------------------------------------------------------------ report

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct LayerAlpha {
    pub layer: usize,
    /// Coefficient issu de la régression.
    pub alpha_fitted: f64,
    /// Coefficient écrit (après écrêtage à ±alpha_max) ; 0 si supprimé.
    pub alpha: f64,
    pub clipped: bool,
    /// |α| < min_alpha : couche retirée du patch.
    pub dropped: bool,
    /// ‖F_{:,l}‖ / √M : réponse moyenne de la lecture à cette couche.
    pub response_rms: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct LinearityCheck {
    pub layer: usize,
    /// ‖F(2δ) − F(δ)‖ / ‖F(δ)‖.
    pub relative_deviation: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct Validation {
    /// moyenne(Fα) avec les α écrits.
    pub predicted_mean_shift: f64,
    /// moyenne(s' − s⁰) mesurée, tous les α appliqués ensemble en mémoire.
    pub actual_mean_shift: f64,
    /// RMSE entre le décalage prévu et le décalage mesuré, prompt par prompt.
    pub rmse_pred_vs_actual: f64,
    /// RMSE(t − s⁰) avant patch.
    pub target_rmse_before: f64,
    /// RMSE(t − s') après patch.
    pub target_rmse_after: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct AutotuneTiming {
    pub total: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct AutotuneReport {
    pub model: PathBuf,
    pub vector: PathBuf,
    pub readout_layer: usize,
    pub site: Site,
    pub pooling: Pooling,
    pub chat_template: ChatTemplate,
    pub proj: Projection,
    pub objective: Objective,
    pub dtype_used: String,
    /// Couches candidates (≤ readout_layer), dans l'ordre.
    pub layers: Vec<usize>,
    /// Couches de la plage au-delà de la lecture (sans effet sur elle).
    pub excluded_layers: Vec<usize>,
    pub n_prompts: usize,
    /// Forwards prévus et effectués : M × (1 + L + 1) + M × (vérifications de linéarité).
    pub forwards: usize,
    pub probe_alpha: f64,
    /// Moyenne de s⁰.
    pub baseline_mean: f64,
    /// Moyenne de t.
    pub target_mean: f64,
    /// F (M × L), ligne par prompt.
    pub response_matrix: Vec<Vec<f64>>,
    pub linearity: Vec<LinearityCheck>,
    pub ridge: RidgeFit,
    /// Grille explorée par --l2-reg auto (vide sinon).
    pub lambda_grid: Vec<GridPoint>,
    pub alphas: Vec<LayerAlpha>,
    pub validation: Validation,
    /// Le patch produit ; null si aucune couche n'est retenue.
    pub patch: Option<PatchFile>,
    /// Fichier écrit (--out) ; null sinon.
    pub out: Option<PathBuf>,
    pub recommendation: String,
    pub timing_ms: AutotuneTiming,
}

#[derive(Debug, Clone)]
pub struct AutotuneRun {
    pub report: AutotuneReport,
    pub warnings: Vec<String>,
}

// ------------------------------------------------------------------ pipeline

/// One evaluation prompt and its optional target projection.
#[derive(Debug, Clone, PartialEq)]
pub struct EvalItem {
    pub text: String,
    pub target: Option<f64>,
}

/// JSON array (or JSONL) of strings or of `{text, target?}`.
pub fn load_eval_dataset(path: &Path) -> Result<Vec<EvalItem>> {
    let bad = |m: String| AutotuneError::Input {
        path: path.to_path_buf(),
        message: m,
    };
    let raw = std::fs::read_to_string(path).map_err(|e| bad(e.to_string()))?;
    let content = raw.trim_start_matches('\u{feff}').trim();
    let values: Vec<Value> = match content.starts_with('[') {
        true => serde_json::from_str(content).map_err(|e| bad(format!("JSON invalide : {e}")))?,
        false => content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).map_err(|e| bad(format!("JSONL invalide : {e}"))))
            .collect::<Result<_>>()?,
    };
    if values.is_empty() {
        return Err(bad("aucun prompt".into()));
    }
    values
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            let (text, target) = match &v {
                Value::String(s) => (s.clone(), None),
                Value::Object(o) => (
                    o.get("text")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                        .unwrap_or_default(),
                    match o.get("target") {
                        None | Some(Value::Null) => None,
                        Some(t) => Some(t.as_f64().filter(|x| x.is_finite()).ok_or_else(|| {
                            bad(format!("prompt {i} : target doit être un nombre"))
                        })?),
                    },
                ),
                _ => {
                    return Err(bad(format!(
                        "prompt {i} : chaîne ou objet {{text, target?}} attendu"
                    )))
                }
            };
            match text.trim().is_empty() {
                true => Err(bad(format!("prompt {i} : texte vide"))),
                false => Ok(EvalItem { text, target }),
            }
        })
        .collect()
}

/// What the vector file says about itself.
struct VectorInfo {
    v: Vec<f64>,
    meta: BTreeMap<String, String>,
    pos_mean: Option<f64>,
    neg_mean: Option<f64>,
}

fn read_vector(path: &Path) -> Result<VectorInfo> {
    let bad = |m: String| AutotuneError::Input {
        path: path.to_path_buf(),
        message: m,
    };
    let v =
        engine::load_direction_vector(path, Some("direction")).map_err(|e| bad(e.to_string()))?;
    let meta = ReadOnlyModel::open(path)
        .map_err(|e| bad(e.to_string()))?
        .metadata();
    let num = |k: &str| {
        meta.get(k)
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|x| x.is_finite())
    };
    let (mut pos_mean, mut neg_mean) = (num("pos_mean_proj"), num("neg_mean_proj"));
    // Older files: the companion keeps every projection.
    if pos_mean.is_none() || neg_mean.is_none() {
        if let Ok(text) = std::fs::read_to_string(companion_path(path)) {
            if let Ok(json) = serde_json::from_str::<Value>(&text) {
                let mean = |key: &str| {
                    let list = json.get(key)?.as_array()?;
                    let xs: Vec<f64> = list.iter().filter_map(Value::as_f64).collect();
                    (!xs.is_empty()).then(|| xs.iter().sum::<f64>() / xs.len() as f64)
                };
                pos_mean = pos_mean.or_else(|| mean("pos_projections"));
                neg_mean = neg_mean.or_else(|| mean("neg_projections"));
            }
        }
    }
    Ok(VectorInfo {
        v: v.iter().map(|&x| f64::from(x)).collect(),
        meta,
        pos_mean,
        neg_mean,
    })
}

fn from_meta<T: serde::de::DeserializeOwned>(
    meta: &BTreeMap<String, String>,
    key: &str,
    flag: &str,
    vector: &Path,
) -> Result<T> {
    let raw = meta.get(key).ok_or_else(|| AutotuneError::Input {
        path: vector.to_path_buf(),
        message: format!("métadonnée « {key} » absente : passer {flag}"),
    })?;
    serde_json::from_value(Value::String(raw.clone())).map_err(|_| AutotuneError::Input {
        path: vector.to_path_buf(),
        message: format!("métadonnée {key} = « {raw} » illisible : passer {flag}"),
    })
}

/// The engine's axis for a residual-stream vector (see `engine/mod.rs`): `down_proj`
/// `[hidden, intermediate]` takes v along its rows (`cols` axis), gate/up `[intermediate,
/// hidden]` along their columns (`rows` axis).
pub fn engine_axis(proj: Projection) -> (BroadcastAxis, Axis) {
    match proj {
        Projection::Down => (BroadcastAxis::Cols, Axis::Cols),
        Projection::Gate | Projection::Up => (BroadcastAxis::Rows, Axis::Rows),
    }
}

fn proj_name(proj: Projection) -> &'static str {
    match proj {
        Projection::Gate => "gate_proj",
        Projection::Up => "up_proj",
        Projection::Down => "down_proj",
    }
}

fn candle_dtype(d: &engine::DType) -> Result<DType> {
    match d {
        engine::DType::BF16 => Ok(DType::BF16),
        engine::DType::F16 => Ok(DType::F16),
        engine::DType::F32 => Ok(DType::F32),
        engine::DType::Other(o) => Err(AutotuneError::Model(format!("dtype {o} non supporté"))),
    }
}

/// The weights of one model in memory, and the stored bytes of the shiftable matrices.
struct Bench<'a> {
    cfg: &'a Qwen2Config,
    readout: usize,
    dtype: DType,
    device: Device,
    base: HashMap<String, Tensor>,
    source: &'a dyn TensorSource,
    proj: Projection,
    v32: Vec<f32>,
    site: Site,
    pooling: Pooling,
    inputs: Vec<Tensor>,
    v: Vec<f64>,
}

impl Bench<'_> {
    fn weight_name(&self, layer: usize) -> String {
        format!("model.layers.{layer}.mlp.{}.weight", proj_name(self.proj))
    }

    /// The weight of `layer` after `subtract alpha·v`, computed on the stored bytes by the
    /// engine (same rounding as on disk), then converted to the compute dtype.
    fn shifted(&self, layer: usize, alpha: f64) -> Result<Tensor> {
        let name = self.weight_name(layer);
        let raw = self.source.read(&name)?;
        let mut bytes = raw.bytes.to_vec();
        engine::apply_vector_shift(
            &mut bytes,
            raw.dtype.clone(),
            &raw.shape,
            &self.v32,
            alpha as f32,
            ShiftOp::Subtract,
            engine_axis(self.proj).0,
        )
        .map_err(|e| AutotuneError::Compute(format!("{name} : {e}")))?;
        Ok(
            Tensor::from_raw_buffer(&bytes, candle_dtype(&raw.dtype)?, &raw.shape, &self.device)?
                .to_dtype(self.dtype)?,
        )
    }

    /// Readouts of every prompt with the given layers shifted.
    fn readouts(
        &self,
        shifts: &[(usize, f64)],
        cancel: &AtomicBool,
        progress: &mut Progress<'_>,
    ) -> Result<Vec<f64>> {
        let mut tensors = self.base.clone();
        for &(layer, alpha) in shifts {
            tensors.insert(self.weight_name(layer), self.shifted(layer, alpha)?);
        }
        let model =
            HookedQwen2::from_tensors(tensors, self.cfg, self.readout, self.dtype, &self.device)?;
        let mut out = Vec::with_capacity(self.inputs.len());
        for (i, input) in self.inputs.iter().enumerate() {
            if cancel.load(Ordering::SeqCst) {
                return Err(AutotuneError::Cancelled);
            }
            let activation = model.forward_capture(input, self.site)?;
            let pooled = pool(&activation, self.pooling, &format!("prompt[{i}]"))?;
            out.push(pooled.iter().zip(&self.v).map(|(a, b)| a * b).sum());
            progress.tick();
        }
        Ok(out)
    }
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / a.len().max(1) as f64).sqrt()
}

fn validate_request(req: &AutotuneRequest) -> Result<()> {
    let bad = |m: &str| Err(AutotuneError::InvalidArgument(m.to_owned()));
    if !(req.probe_alpha.is_finite() && req.probe_alpha > 0.0) {
        return bad("--probe-alpha doit être > 0");
    }
    if !(req.alpha_max.is_finite()
        && req.alpha_max > 0.0
        && req.alpha_max <= f64::from(MAX_ABS_ALPHA))
    {
        return bad("--alpha-max doit être dans ]0, 100]");
    }
    if !(req.min_alpha.is_finite() && req.min_alpha >= 0.0) {
        return bad("--min-alpha doit être >= 0");
    }
    if req.max_tokens == 0 {
        return bad("--max-tokens doit être >= 1");
    }
    if let L2Reg::Fixed(l) = req.l2_reg {
        if !(l.is_finite() && l > 0.0) {
            return bad("--l2-reg doit être > 0 (ou auto)");
        }
    }
    if req.objective == Objective::Target && req.target_projection.is_none() {
        return bad("--objective target exige --target-projection");
    }
    if let Some(out) = &req.out {
        if !out
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("json"))
        {
            return Err(AutotuneError::Input {
                path: out.clone(),
                message: "l'extension .json est requise".into(),
            });
        }
        let parent = match out.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };
        if !parent.is_dir() {
            return Err(AutotuneError::Input {
                path: out.clone(),
                message: "dossier de sortie inexistant".into(),
            });
        }
        if out.exists() && !req.overwrite {
            return Err(AutotuneError::OutputExists(out.clone()));
        }
    }
    Ok(())
}

/// Opens the model directory and runs [`run_with`].
pub fn run(
    req: &AutotuneRequest,
    probe: &dyn HardwareProbe,
    cancel: &AtomicBool,
    on_event: &mut dyn FnMut(AutotuneEvent),
) -> Result<AutotuneRun> {
    validate_request(req)?;
    let source = SafetensorsDir::open(&req.model)?;
    run_with(req, &source, probe, cancel, on_event)
}

/// validate -> vector and dataset -> layers -> cost -> RAM (code 1 before any weight) ->
/// baseline, one probe per layer, linearity checks -> ridge -> clip -> validation run ->
/// patch validated, then written atomically (only with `out`).
pub fn run_with(
    req: &AutotuneRequest,
    source: &dyn TensorSource,
    probe: &dyn HardwareProbe,
    cancel: &AtomicBool,
    on_event: &mut dyn FnMut(AutotuneEvent),
) -> Result<AutotuneRun> {
    let started = Instant::now();
    validate_request(req)?;
    let mut warnings = Vec::new();
    let info = read_vector(&req.vector)?;
    let cfg = load_config(&req.model)?;
    if info.v.len() != cfg.hidden_size {
        return Err(AutotuneError::Input {
            path: req.vector.clone(),
            message: format!(
                "vecteur de taille {}, attendu hidden_size = {}",
                info.v.len(),
                cfg.hidden_size
            ),
        });
    }
    let readout = match req.readout_layer {
        Some(r) => r,
        None => from_meta::<String>(&info.meta, "layer", "--readout-layer", &req.vector)?
            .parse::<usize>()
            .map_err(|_| AutotuneError::InvalidArgument("métadonnée layer illisible".into()))?,
    };
    if readout >= cfg.num_hidden_layers {
        return Err(AutotuneError::InvalidArgument(format!(
            "couche de lecture {readout} hors bornes (num_hidden_layers = {})",
            cfg.num_hidden_layers
        )));
    }
    let site = match req.site {
        Some(s) => s,
        None => from_meta(&info.meta, "site", "--site", &req.vector)?,
    };
    let pooling = match req.pooling {
        Some(p) => p,
        None => from_meta(&info.meta, "pooling", "--pooling", &req.vector)?,
    };
    let chat_template = match req.chat_template {
        Some(c) => c,
        None => from_meta(&info.meta, "chat_template", "--chat-template", &req.vector)?,
    };
    let (first, last) = req.layer_range.unwrap_or((0, readout));
    if last >= cfg.num_hidden_layers {
        return Err(AutotuneError::InvalidArgument(format!(
            "plage {first}-{last} hors bornes (num_hidden_layers = {})",
            cfg.num_hidden_layers
        )));
    }
    let layers: Vec<usize> = (first..=last.min(readout)).collect();
    let excluded: Vec<usize> = (readout + 1..=last).filter(|l| *l >= first).collect();
    if !excluded.is_empty() {
        warnings.push(format!(
            "couches {excluded:?} au-delà de la lecture (couche {readout}) : sans effet sur elle, exclues"
        ));
    }
    if layers.is_empty() {
        return Err(AutotuneError::InvalidArgument(format!(
            "aucune couche de la plage {first}-{last} n'est ≤ la couche de lecture {readout}"
        )));
    }
    let items = load_eval_dataset(&req.eval_dataset)?;
    let default_target = match req.objective {
        Objective::Suppress => req.target_projection.or(info.neg_mean),
        Objective::Amplify => req.target_projection.or(info.pos_mean),
        Objective::Target => req.target_projection,
    };
    let targets: Vec<f64> = items
        .iter()
        .map(|it| it.target.or(default_target))
        .collect::<Option<_>>()
        .ok_or_else(|| AutotuneError::Input {
            path: req.vector.clone(),
            message: "projection cible inconnue (ni dans le vecteur, ni dans le dataset) : passer --target-projection".into(),
        })?;

    let (m, l) = (items.len(), layers.len());
    let checks: Vec<usize> = {
        let mut picks = vec![layers[0], layers[l / 2], layers[l - 1]];
        picks.dedup();
        picks.truncate(LINEARITY_LAYERS);
        picks
    };
    let forwards = m * (1 + l + 1) + m * checks.len();
    on_event(AutotuneEvent::Planned {
        forwards,
        layers: l,
    });
    if forwards > req.max_forwards {
        return Err(AutotuneError::InvalidArgument(format!(
            "{forwards} forwards prévus (M = {m} × (1 + L = {l} + 1) + {} de linéarité) > --max-forwards {}",
            m * checks.len(),
            req.max_forwards
        )));
    }
    if m < 2 * l {
        warnings.push(format!(
            "M = {m} prompts < 2L = {} : sur-ajustement probable (le jeu d'évaluation sert à l'ajustement ET à la mesure)",
            2 * l
        ));
    }

    let required = required_tensors(&cfg, readout);
    let totals = check_tensors(source, &required)?;
    let dtype = choose_dtype(
        req.dtype,
        plan_memory(&cfg, totals, req.max_tokens),
        probe,
        &mut warnings,
    )?;
    let tokenizer = load_tokenizer(&req.model)?;
    let device = Device::Cpu;
    let inputs = items
        .iter()
        .enumerate()
        .map(|(i, it)| {
            let t = tokenize(&tokenizer, &chat_template.apply(&it.text), req.max_tokens)?;
            if t.ids.is_empty() {
                return Err(AutotuneError::InvalidArgument(format!(
                    "prompt {i} : aucun token"
                )));
            }
            Ok(Tensor::new(t.ids.as_slice(), &device)?.unsqueeze(0)?)
        })
        .collect::<Result<Vec<_>>>()?;
    let base = load_tensors(source, &required, dtype, &device, &mut |_, _| match cancel
        .load(Ordering::SeqCst)
    {
        true => std::ops::ControlFlow::Break(()),
        false => std::ops::ControlFlow::Continue(()),
    })?;
    let bench = Bench {
        cfg: &cfg,
        readout,
        dtype,
        device,
        base,
        source,
        proj: req.proj,
        v32: info.v.iter().map(|&x| x as f32).collect(),
        site,
        pooling,
        inputs,
        v: info.v.clone(),
    };

    let mut progress = Progress {
        done: 0,
        total: forwards,
        on_event,
    };
    let s0 = bench.readouts(&[], cancel, &mut progress)?;
    let delta = req.probe_alpha;
    let mut f = vec![0.0; m * l];
    for (col, &layer) in layers.iter().enumerate() {
        let s = bench.readouts(&[(layer, delta)], cancel, &mut progress)?;
        for i in 0..m {
            f[i * l + col] = (s[i] - s0[i]) / delta;
        }
    }
    let mut linearity = Vec::new();
    for &layer in &checks {
        let col = layers.iter().position(|x| *x == layer).unwrap_or(0);
        let s2 = bench.readouts(&[(layer, 2.0 * delta)], cancel, &mut progress)?;
        let (mut diff, mut norm) = (0.0, 0.0);
        for i in 0..m {
            let r2 = (s2[i] - s0[i]) / (2.0 * delta);
            diff += (r2 - f[i * l + col]).powi(2);
            norm += f[i * l + col].powi(2);
        }
        let relative_deviation = match norm > 0.0 {
            true => (diff / norm).sqrt(),
            false => 0.0,
        };
        if relative_deviation > LINEARITY_TOLERANCE {
            warnings.push(format!(
                "couche {layer} : réponse non linéaire (écart {:.1} % entre δ et 2δ > {:.0} %)",
                relative_deviation * 100.0,
                LINEARITY_TOLERANCE * 100.0
            ));
        }
        linearity.push(LinearityCheck {
            layer,
            relative_deviation,
        });
    }

    (progress.on_event)(AutotuneEvent::Solve);
    let y: Vec<f64> = targets.iter().zip(&s0).map(|(t, s)| t - s).collect();
    let (fit, lambda_grid) = match req.l2_reg {
        L2Reg::Fixed(lambda) => (ridge(&f, m, l, &y, lambda)?, Vec::new()),
        L2Reg::Auto => ridge_auto(&f, m, l, &y)?,
    };
    let mut alphas = Vec::with_capacity(l);
    let mut clipped_layers = Vec::new();
    for (col, &layer) in layers.iter().enumerate() {
        let fitted = fit.alpha[col];
        let clipped = fitted.abs() > req.alpha_max;
        let value = fitted.clamp(-req.alpha_max, req.alpha_max);
        let dropped = value.abs() < req.min_alpha;
        if clipped {
            clipped_layers.push(layer);
        }
        let column: f64 = (0..m).map(|i| f[i * l + col].powi(2)).sum();
        alphas.push(LayerAlpha {
            layer,
            alpha_fitted: fitted,
            alpha: if dropped { 0.0 } else { value },
            clipped,
            dropped,
            response_rms: (column / m as f64).sqrt(),
        });
    }
    if !clipped_layers.is_empty() {
        warnings.push(format!(
            "α écrêtés à ±{} sur les couches {clipped_layers:?} : prédictions recalculées",
            req.alpha_max
        ));
    }
    // The coefficients are written as f32: predict and validate with those exact values.
    let kept: Vec<(usize, f64)> = alphas
        .iter()
        .filter(|a| !a.dropped)
        .map(|a| (a.layer, f64::from(a.alpha as f32)))
        .collect();
    let written: Vec<f64> = alphas
        .iter()
        .map(|a| match a.dropped {
            true => 0.0,
            false => f64::from(a.alpha as f32),
        })
        .collect();
    let predicted: Vec<f64> = (0..m)
        .map(|i| (0..l).map(|c| f[i * l + c] * written[c]).sum())
        .collect();
    let s_after = match kept.is_empty() {
        true => {
            warnings
                .push("aucune couche retenue (tous les |α| < --min-alpha) : pas de patch".into());
            s0.clone()
        }
        false => bench.readouts(&kept, cancel, &mut progress)?,
    };
    let actual: Vec<f64> = s_after.iter().zip(&s0).map(|(a, b)| a - b).collect();
    let validation = Validation {
        predicted_mean_shift: mean(&predicted),
        actual_mean_shift: mean(&actual),
        rmse_pred_vs_actual: rmse(&predicted, &actual),
        target_rmse_before: rmse(&targets, &s0),
        target_rmse_after: rmse(&targets, &s_after),
    };

    let vector_file = std::path::absolute(&req.vector).map_err(|e| AutotuneError::Input {
        path: req.vector.clone(),
        message: e.to_string(),
    })?;
    let patch = match kept.is_empty() {
        true => None,
        false => Some(PatchFile {
            schema_version: 1,
            description: Some(format!(
                "surgeon autotune : objectif {:?}, lecture couche {readout} ({}, {}), {} prompts, λ = {:e}, R² {}, décalage prévu {:.4}, mesuré {:.4}",
                req.objective,
                site.as_str(),
                pooling.as_str(),
                m,
                fit.lambda,
                fit.r2.map_or("—".to_owned(), |r| format!("{r:.4}")),
                validation.predicted_mean_shift,
                validation.actual_mean_shift
            )),
            backup: true,
            operations: kept
                .iter()
                .map(|&(layer, alpha)| Operation {
                    id: format!("autotune-L{layer}"),
                    layers: LayersField::List(vec![layer]),
                    projections: vec![req.proj],
                    operation: OperationKind::Subtract,
                    vector_file: vector_file.clone(),
                    vector_tensor: Some("direction".into()),
                    vector_tensor_b: None,
                    alpha: alpha as f32,
                    axis: engine_axis(req.proj).1,
                })
                .collect(),
        }),
    };
    if let Some(p) = &patch {
        let base_dir = req
            .out
            .as_ref()
            .and_then(|o| o.parent().map(Path::to_path_buf))
            .unwrap_or_default();
        validate_patch(p, &base_dir, &ValidationOptions::default())
            .map_err(|e| AutotuneError::Compute(format!("patch produit invalide : {e}")))?;
    }
    if cancel.load(Ordering::SeqCst) {
        return Err(AutotuneError::Cancelled);
    }
    let out = match (&req.out, &patch) {
        (Some(out), Some(p)) => {
            let mut bytes =
                serde_json::to_vec_pretty(p).map_err(|e| AutotuneError::Compute(e.to_string()))?;
            bytes.push(b'\n');
            write_atomic(out, &bytes, req.overwrite)?;
            Some(out.clone())
        }
        _ => None,
    };
    Ok(AutotuneRun {
        report: AutotuneReport {
            model: req.model.clone(),
            vector: req.vector.clone(),
            readout_layer: readout,
            site,
            pooling,
            chat_template,
            proj: req.proj,
            objective: req.objective,
            dtype_used: dtype_name(dtype).to_owned(),
            layers,
            excluded_layers: excluded,
            n_prompts: m,
            forwards,
            probe_alpha: delta,
            baseline_mean: mean(&s0),
            target_mean: mean(&targets),
            response_matrix: (0..m).map(|i| f[i * l..(i + 1) * l].to_vec()).collect(),
            linearity,
            ridge: fit,
            lambda_grid,
            alphas,
            validation,
            patch,
            out,
            recommendation: "effets secondaires non mesurés ici : appliquer sur une copie puis surgeon bench (dump, logits, ppl)".into(),
            timing_ms: AutotuneTiming {
                total: started.elapsed().as_millis() as u64,
            },
        },
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic F (M×L) and α_true.
    fn system(m: usize, l: usize) -> (Vec<f64>, Vec<f64>) {
        let f = (0..m * l)
            .map(|k| ((k * 7 % 11) as f64 - 5.0) / 3.0 + (k as f64 * 0.37).sin())
            .collect();
        let alpha = (0..l).map(|c| 0.5 - c as f64 * 0.3).collect();
        (f, alpha)
    }

    fn apply(f: &[f64], m: usize, l: usize, a: &[f64]) -> Vec<f64> {
        (0..m)
            .map(|i| (0..l).map(|c| f[i * l + c] * a[c]).sum())
            .collect()
    }

    #[test]
    fn exact_recovery_noise_and_shrinkage() {
        let (m, l) = (20, 4);
        let (f, truth) = system(m, l);
        let y = apply(&f, m, l, &truth);
        let fit = ridge(&f, m, l, &y, 1e-10).unwrap();
        for (a, t) in fit.alpha.iter().zip(&truth) {
            assert!((a - t).abs() < 1e-8, "{a} vs {t}");
        }
        assert!(fit.r2.unwrap() > 1.0 - 1e-12);
        let noisy: Vec<f64> = y
            .iter()
            .enumerate()
            .map(|(i, v)| v + 1e-3 * ((i as f64) * 1.7).sin())
            .collect();
        let fit = ridge(&f, m, l, &noisy, 1e-8).unwrap();
        assert!(fit
            .alpha
            .iter()
            .zip(&truth)
            .all(|(a, t)| (a - t).abs() < 1e-2));
        let norms: Vec<f64> = [1e-4, 1e-1, 1e1, 1e3]
            .iter()
            .map(|&lam| ridge(&f, m, l, &y, lam).unwrap().alpha_norm)
            .collect();
        assert!(
            norms.windows(2).all(|w| w[0] > w[1]),
            "‖α‖ decreases with λ: {norms:?}"
        );
    }

    #[test]
    fn small_lambda_is_least_squares_and_loocv_matches_brute_force() {
        let (m, l) = (15, 3);
        let (f, _) = system(m, l);
        let y: Vec<f64> = (0..m).map(|i| (i as f64 * 0.9).cos()).collect();
        let lambda = 0.05;
        let fit = ridge(&f, m, l, &y, lambda).unwrap();
        let mut sq = 0.0;
        for out in 0..m {
            let keep: Vec<usize> = (0..m).filter(|&i| i != out).collect();
            let fk: Vec<f64> = keep
                .iter()
                .flat_map(|&i| f[i * l..(i + 1) * l].to_vec())
                .collect();
            let yk: Vec<f64> = keep.iter().map(|&i| y[i]).collect();
            let a = ridge(&fk, m - 1, l, &yk, lambda).unwrap().alpha;
            let pred: f64 = (0..l).map(|c| f[out * l + c] * a[c]).sum();
            sq += (y[out] - pred).powi(2);
        }
        let brute = (sq / m as f64).sqrt();
        assert!(
            (fit.loocv_rmse.unwrap() - brute).abs() < 1e-10,
            "{:?} vs {brute}",
            fit.loocv_rmse
        );

        // λ → 0: the normal equations' solution, whose residual is orthogonal to F.
        let ls = ridge(&f, m, l, &y, 1e-12).unwrap();
        let p = apply(&f, m, l, &ls.alpha);
        for c in 0..l {
            let ortho: f64 = (0..m).map(|i| f[i * l + c] * (y[i] - p[i])).sum();
            assert!(ortho.abs() < 1e-9, "{ortho}");
        }
    }

    #[test]
    fn collinear_columns_auto_grid_and_errors() {
        let m = 12;
        // Two identical columns: FᵀF is singular, λ > 0 restores definiteness.
        let f: Vec<f64> = (0..m).flat_map(|i| [i as f64, i as f64, 1.0]).collect();
        let y: Vec<f64> = (0..m).map(|i| 2.0 * i as f64 + 1.0).collect();
        let fit = ridge(&f, m, 3, &y, 1e-2).unwrap();
        assert!(fit.cond_estimate > 10.0, "{}", fit.cond_estimate);
        assert!(
            (fit.alpha[0] - fit.alpha[1]).abs() < 1e-9,
            "the weight is shared"
        );
        assert!(
            ridge(&f, m, 3, &y, 1e-14).is_err(),
            "too singular for Cholesky"
        );
        let (best, grid) = ridge_auto(&f, m, 3, &y).unwrap();
        assert_eq!(grid.len(), AUTO_GRID_POINTS);
        let again = ridge_auto(&f, m, 3, &y).unwrap();
        assert_eq!(best, again.0, "deterministic choice");
        let scores: Vec<f64> = grid.iter().filter_map(|g| g.loocv_rmse).collect();
        assert!(scores.iter().all(|s| *s >= best.loocv_rmse.unwrap()));
        assert!(ridge(&f, m, 2, &y, 1.0).is_err(), "shape");
        assert_eq!("auto".parse::<L2Reg>(), Ok(L2Reg::Auto));
        assert_eq!("1e-3".parse::<L2Reg>(), Ok(L2Reg::Fixed(1e-3)));
        assert!("-1".parse::<L2Reg>().is_err() && "x".parse::<L2Reg>().is_err());
        assert_eq!(parse_layer_range("2-5"), Ok((2, 5)));
        assert_eq!(parse_layer_range("3"), Ok((3, 3)));
        assert!(parse_layer_range("5-2").is_err());
    }

    #[test]
    fn eval_dataset_formats() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("e.json");
        std::fs::write(
            &p,
            r#"["a b", {"text": "c", "target": -1.5}, {"text": "d"}]"#,
        )
        .unwrap();
        let items = load_eval_dataset(&p).unwrap();
        assert_eq!(
            items[1],
            EvalItem {
                text: "c".into(),
                target: Some(-1.5)
            }
        );
        assert_eq!(items[2].target, None);
        std::fs::write(&p, "{\"text\": \"x\"}\n{\"text\": \"y\", \"target\": 2}\n").unwrap();
        assert_eq!(load_eval_dataset(&p).unwrap().len(), 2);
        for bad in [
            r#"[]"#,
            r#"[{"text": ""}]"#,
            r#"[{"text": "a", "target": "x"}]"#,
            r#"[3]"#,
            "[oops",
        ] {
            std::fs::write(&p, bad).unwrap();
            assert!(load_eval_dataset(&p).is_err(), "{bad}");
        }
    }
}
