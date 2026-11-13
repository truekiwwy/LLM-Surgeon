//! `surgeon graph`: the residual-stream write/read graph of a SafeTensors or GGUF model.
//!
//! A **writer** is a matrix whose columns (or rows, for embeddings) span directions added
//! to the residual stream: `embed_tokens`, `o_proj` and `down_proj` (one node per expert for
//! a fused MoE tensor). A **reader** is a matrix whose rows read that stream: `q_proj`,
//! `k_proj`, `v_proj`, `gate_proj`, `up_proj` and `lm_head`. The graph is causal: an edge
//! points from a writer to a reader that executes strictly later
//! (`embed < q/k/v < attn_out < gate/up < mlp_out` per layer, with the parallel-block
//! variant where attention and MLP share one stage). Edges carry the subspace alignment
//! `||B_w B_r^T||_F^2 / r`, chance (`r/hidden`), its chance-corrected form, an energy-weighted
//! coupling and the cosine of the smallest principal angle.
//!
//! All spectral quantities come from the shared randomized SVD (see
//! [`crate::surgery::rsvd`]); the range certificate gives certified intervals for `sigma1`,
//! the stable rank `D_spec` and the energy-entropy effective rank even when the tail is not
//! captured. `--full-spectrum-max-dim` enables the exact L1 effective rank on small matrices.

pub mod cache;
pub mod diff;
pub mod error;
pub mod layout;
pub mod metrics;
pub mod model;
pub mod output;
pub mod stream;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::{Args, ValueEnum};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::app::{AppDeps, CoreResult, Outcome, ProgressEvent, ProgressSink};
use crate::linalg::jacobi::MAX_N;
use crate::patch::LayerSelection;
use crate::surgery::rsvd::RsvdParams;

use self::cache::Cache;
use self::error::{GraphError, Result};
use self::metrics::{AnalyzeOptions, Basis};
use self::model::{ModelData, TensorRef};
use self::output::{
    EdgeReport, GlobalReport, GraphReport, ModelReport, NodeKind, NodeReport, OutputFile,
    ParametersReport, ToolInfo, SCHEMA_VERSION,
};

// ------------------------------------------------------------------ CLI

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum GraphRole {
    Embed,
    AttnOut,
    Q,
    K,
    V,
    Gate,
    Up,
    MlpOut,
    LmHead,
}

impl GraphRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::AttnOut => "attn_out",
            Self::Q => "q",
            Self::K => "k",
            Self::V => "v",
            Self::Gate => "gate",
            Self::Up => "up",
            Self::MlpOut => "mlp_out",
            Self::LmHead => "lm_head",
        }
    }

    pub fn is_writer(self) -> bool {
        matches!(self, Self::Embed | Self::AttnOut | Self::MlpOut)
    }

    pub fn is_reader(self) -> bool {
        !self.is_writer()
    }
}

/// `--edges`: which causal writer -> reader pairs enter the report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeMode {
    /// A writer to same-layer later readers and to every reader of the next layer.
    Adjacent,
    /// Every writer -> later reader pair.
    All,
    /// Adjacent pairs plus the `N` strongest long-range pairs of each writer.
    TopK(usize),
}

impl EdgeMode {
    pub fn as_str(self) -> String {
        match self {
            Self::Adjacent => "adjacent".into(),
            Self::All => "all".into(),
            Self::TopK(k) => format!("topk:{k}"),
        }
    }
}

impl FromStr for EdgeMode {
    type Err = String;
    fn from_str(text: &str) -> std::result::Result<Self, String> {
        let text = text.trim();
        if text.eq_ignore_ascii_case("adjacent") {
            return Ok(Self::Adjacent);
        }
        if text.eq_ignore_ascii_case("all") {
            return Ok(Self::All);
        }
        if let Some(rest) = text
            .strip_prefix("topk:")
            .or_else(|| text.strip_prefix("topk="))
            .or_else(|| text.strip_prefix("top-k:"))
        {
            let k = rest
                .parse::<usize>()
                .map_err(|_| format!("--edges : nombre invalide « {rest} »"))?;
            if k == 0 {
                return Err("--edges : topk:N exige N >= 1".into());
            }
            return Ok(Self::TopK(k));
        }
        Err(format!(
            "--edges : valeur « {text} » invalide (adjacent, all, topk:N)"
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum EdgeWeight {
    Alignment,
    Coupling,
}

impl EdgeWeight {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Alignment => "alignment",
            Self::Coupling => "coupling",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    Json,
    Dot,
    Mmd,
    Csv,
    Ndjson,
}

impl OutputFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Dot => "dot",
            Self::Mmd => "mmd",
            Self::Csv => "csv",
            Self::Ndjson => "ndjson",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum MatrixArg {
    Q,
    K,
    V,
    O,
    Gate,
    Up,
    Down,
    Embed,
    #[serde(rename = "lm_head")]
    #[value(name = "lm_head")]
    LmHead,
}

impl MatrixArg {
    pub fn role(self) -> GraphRole {
        match self {
            Self::Q => GraphRole::Q,
            Self::K => GraphRole::K,
            Self::V => GraphRole::V,
            Self::O => GraphRole::AttnOut,
            Self::Gate => GraphRole::Gate,
            Self::Up => GraphRole::Up,
            Self::Down => GraphRole::MlpOut,
            Self::Embed => GraphRole::Embed,
            Self::LmHead => GraphRole::LmHead,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ColorScaleArg {
    Quantile,
    Log,
    Linear,
}

impl ColorScaleArg {
    fn scale(self) -> output::ColorScale {
        match self {
            Self::Quantile => output::ColorScale::Quantile,
            Self::Log => output::ColorScale::Log,
            Self::Linear => output::ColorScale::Linear,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum PaletteArg {
    Diverging,
    #[serde(rename = "cb-safe")]
    #[value(name = "cb-safe")]
    CbSafe,
}

impl PaletteArg {
    fn palette(self) -> output::Palette {
        match self {
            Self::Diverging => output::Palette::Diverging,
            Self::CbSafe => output::Palette::CbSafe,
        }
    }
}

#[derive(Debug, Clone, Args)]
pub struct GraphArgs {
    /// Modèle : fichier .safetensors, dossier shardé ou .gguf.
    #[arg(long)]
    pub model: PathBuf,
    /// Dossier des sorties ; sans lui, seule une synthèse bornée est renvoyée.
    #[arg(long)]
    pub out_dir: Option<PathBuf>,
    /// Formats écrits (json, dot, mmd, csv, ndjson), séparés par des virgules.
    #[arg(long, value_enum, value_delimiter = ',', default_value = "json")]
    pub format: Vec<OutputFormat>,
    /// Couches retenues (`0`, `1-3`, `0,2,4` ou `all`).
    #[arg(long)]
    pub layers: Option<String>,
    /// Matrices retenues (`q,k,v,o,gate,up,down,embed,lm_head`), séparées par des virgules.
    #[arg(long, value_enum, value_delimiter = ',')]
    pub matrices: Vec<MatrixArg>,
    /// Rang cible du SVD randomisé.
    #[arg(long, default_value_t = 32)]
    pub rank: usize,
    /// Colonnes supplémentaires du sous-espace aléatoire.
    #[arg(long, default_value_t = 10)]
    pub oversample: usize,
    /// Itérations de puissance (stabilisent le sous-espace).
    #[arg(long = "power-iters", default_value_t = 3)]
    pub power_iters: usize,
    /// Graine du générateur du SVD ; même graine, mêmes nombres.
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Dimension maximale pour le spectre complet exact (0 : désactivé).
    #[arg(long = "full-spectrum-max-dim", default_value_t = 2048)]
    pub full_spectrum_max_dim: usize,
    /// Arêtes : adjacent, all, ou topk:N.
    #[arg(long, default_value = "adjacent", value_parser = EdgeMode::from_str)]
    pub edges: EdgeMode,
    /// Alignement brut minimal (0 garde tout).
    #[arg(long = "min-alignment", default_value_t = 0.0)]
    pub min_alignment: f64,
    /// Poids d'arête : alignment ou coupling.
    #[arg(long = "edge-weight", value_enum, default_value = "alignment")]
    pub edge_weight: EdgeWeight,
    /// Vecteur résiduel (JSON ou .bin f32) pour E_write/E_read.
    #[arg(long)]
    pub vector: Option<PathBuf>,
    /// Kurtosis par canal au-delà duquel un canal est outlier.
    #[arg(long = "kurtosis-threshold", default_value_t = 10.0)]
    pub kurtosis_threshold: f64,
    /// Échelle de couleur des arêtes : quantile, log ou linear.
    #[arg(long = "color-scale", value_enum, default_value = "quantile")]
    pub color_scale: ColorScaleArg,
    /// Palette : diverging ou cb-safe.
    #[arg(long, value_enum, default_value = "diverging")]
    pub palette: PaletteArg,
    /// Nombre maximal de matrices traitées (les plus grosses d'abord).
    #[arg(long = "max-nodes")]
    pub max_nodes: Option<usize>,
    /// Dossier du cache JSONL (désactivé par défaut).
    #[arg(long = "cache-dir")]
    pub cache_dir: Option<PathBuf>,
    /// N'écrit ni ne lit le cache.
    #[arg(long = "no-cache")]
    pub no_cache: bool,
    /// Réutilise un cache partiel d'une exécution interrompue.
    #[arg(long)]
    pub resume: bool,
    /// Estime les passes, octets et la durée sans exécuter les SVD.
    #[arg(long)]
    pub estimate: bool,
    /// Fils rayon (0 : défaut du système).
    #[arg(long, default_value_t = 0)]
    pub jobs: usize,
    /// Remplace les sorties existantes.
    #[arg(long)]
    pub overwrite: bool,
    /// Référence comparée au modèle : dossier ou fichier de modèle, ou graph.json déjà exporté.
    #[arg(long)]
    pub baseline: Option<PathBuf>,
    /// Empreinte rapide (taille, date, 3 fenêtres de 1 Mio par fichier) au lieu du SHA-256 complet.
    #[arg(long = "fast-hash")]
    pub fast_hash: bool,
}

impl GraphArgs {
    /// The effective rope of the command, checked before touching the model.
    fn check(&self) -> Result<Settings> {
        if self.rank == 0 {
            return Err(GraphError::Usage("--rank doit être >= 1".into()));
        }
        if self.rank + self.oversample > MAX_N {
            return Err(GraphError::Usage(format!(
                "--rank + --oversample = {} dépasse {MAX_N}",
                self.rank + self.oversample
            )));
        }
        if self.power_iters > 64 {
            return Err(GraphError::Usage("--power-iters doit être <= 64".into()));
        }
        if !self.min_alignment.is_finite() {
            return Err(GraphError::Usage("--min-alignment doit être fini".into()));
        }
        if !self.kurtosis_threshold.is_finite() || self.kurtosis_threshold <= 0.0 {
            return Err(GraphError::Usage(
                "--kurtosis-threshold doit être un nombre fini > 0".into(),
            ));
        }
        if self.max_nodes == Some(0) {
            return Err(GraphError::Usage("--max-nodes doit être >= 1".into()));
        }
        if self.jobs > 1024 {
            return Err(GraphError::Usage("--jobs doit être <= 1024".into()));
        }
        if self.resume && self.cache_dir.is_none() {
            return Err(GraphError::Usage("--resume exige --cache-dir".into()));
        }
        if self.no_cache && self.cache_dir.is_some() {
            return Err(GraphError::Usage(
                "--no-cache et --cache-dir sont incompatibles".into(),
            ));
        }
        if self.estimate && self.baseline.is_some() {
            return Err(GraphError::Usage(
                "--estimate ne couvre pas --baseline".into(),
            ));
        }
        if self.format.is_empty() {
            return Err(GraphError::Usage("--format est vide".into()));
        }
        let layers = match &self.layers {
            Some(text) => LayerSelection::from_str(text)
                .map_err(|e| GraphError::Usage(format!("--layers : {e}")))?,
            None => LayerSelection::All,
        };
        let roles: Vec<GraphRole> = match self.matrices.is_empty() {
            true => vec![
                GraphRole::Embed,
                GraphRole::AttnOut,
                GraphRole::Q,
                GraphRole::K,
                GraphRole::V,
                GraphRole::Gate,
                GraphRole::Up,
                GraphRole::MlpOut,
                GraphRole::LmHead,
            ],
            false => self.matrices.iter().map(|m| m.role()).collect(),
        };
        // "--format json,dot --format json" would write graph.json twice.
        let mut formats = Vec::new();
        for format in &self.format {
            if !formats.contains(format) {
                formats.push(*format);
            }
        }
        Ok(Settings {
            formats,
            layers,
            roles,
            vector: self.vector.clone(),
            color_scale: self.color_scale.scale(),
            palette: self.palette.palette(),
            cache_dir: self.cache_dir.clone(),
            resume: self.resume,
            no_cache: self.no_cache,
        })
    }
}

struct Settings {
    /// `--format` without duplicates, in the order given.
    formats: Vec<OutputFormat>,
    layers: LayerSelection,
    roles: Vec<GraphRole>,
    vector: Option<PathBuf>,
    color_scale: output::ColorScale,
    palette: output::Palette,
    cache_dir: Option<PathBuf>,
    resume: bool,
    no_cache: bool,
}

impl Settings {
    fn layer_selected(&self, layer: Option<usize>) -> bool {
        let Some(layer) = layer else {
            return true;
        };
        match &self.layers {
            LayerSelection::All => true,
            LayerSelection::List(list) => list.contains(&layer),
            LayerSelection::Range {
                start,
                end_inclusive,
            } => (*start..=*end_inclusive).contains(&layer),
        }
    }
}

fn causal_seq(role: GraphRole, layer: Option<usize>, layers: usize, parallel: bool) -> usize {
    let layer = layer.unwrap_or(0);
    match role {
        GraphRole::Embed => 0,
        GraphRole::LmHead => 4 * layers + 1,
        GraphRole::Q | GraphRole::K | GraphRole::V => 4 * layer + 1,
        GraphRole::AttnOut => 4 * layer + 2,
        GraphRole::Gate | GraphRole::Up if parallel => 4 * layer + 1,
        GraphRole::Gate | GraphRole::Up => 4 * layer + 3,
        GraphRole::MlpOut if parallel => 4 * layer + 2,
        GraphRole::MlpOut => 4 * layer + 4,
    }
}

fn node_id(role: GraphRole, layer: Option<usize>, expert: Option<usize>) -> String {
    let mut id = match (role, layer) {
        (GraphRole::Embed, _) => "embed".to_owned(),
        (GraphRole::LmHead, _) => "lm_head".to_owned(),
        (_, Some(layer)) => format!("L{layer}.{}", role.as_str()),
        (_, None) => role.as_str().to_owned(),
    };
    if let Some(expert) = expert {
        id.push_str(&format!(".e{expert}"));
    }
    id
}

// ------------------------------------------------------------------ core

/// CLI entry point.
pub fn core_graph(args: &GraphArgs, deps: &AppDeps, sink: &mut dyn ProgressSink) -> CoreResult {
    core_graph_with(args, deps, sink, None)
}

/// Core with an optional wall-clock cap (the MCP `max_seconds` argument).
pub fn core_graph_with(
    args: &GraphArgs,
    deps: &AppDeps,
    sink: &mut dyn ProgressSink,
    max_seconds: Option<f64>,
) -> CoreResult {
    if max_seconds.is_some_and(|seconds| !seconds.is_finite() || seconds <= 0.0) {
        return Err(GraphError::Usage("max_seconds doit être un nombre fini > 0".into()).into());
    }
    let started = Instant::now();
    let cancel = Arc::new(AtomicBool::new(false));
    let watchdog = Watchdog::start(max_seconds, Arc::clone(&cancel));
    let settings = args.check()?;
    if args.estimate {
        let model = open_model(&args.model, args, deps, sink, &cancel)?;
        return estimate(args, &settings, &model);
    }
    let mut report = build_report(&args.model, args, &settings, deps, sink, &cancel)?;
    if let Some(path) = &args.baseline {
        let baseline = match diff::is_report_file(path) {
            true => diff::read_report(path)?,
            false => build_report(path, args, &settings, deps, sink, &cancel)?,
        };
        report.baseline_diff = Some(diff::compare(&baseline, &report, path));
    }
    drop(watchdog);
    check_cancel(deps, &cancel)?;

    let outputs = match &args.out_dir {
        Some(dir) => {
            let files = render_files(&report, &settings.formats);
            let paths = output::write_outputs(dir, &files, args.overwrite)?;
            sink.event(ProgressEvent::GraphWrite { files: paths.len() });
            paths
        }
        None => Vec::new(),
    };
    sink.event(ProgressEvent::GraphDone {
        nodes: report.global.nodes,
        edges: report.global.edges,
        seconds: started.elapsed().as_secs_f64(),
    });
    let data = output::summary(&report, &outputs, 200, 20);
    Ok(Outcome::ok(data, output::human_summary(&report, &outputs)))
}

fn open_model(
    path: &Path,
    args: &GraphArgs,
    deps: &AppDeps,
    sink: &mut dyn ProgressSink,
    cancel: &AtomicBool,
) -> Result<ModelData> {
    let model = ModelData::open(path, args.fast_hash)?;
    check_cancel(deps, cancel)?;
    for warning in &model.warnings {
        sink.event(ProgressEvent::Warn(warning.clone()));
    }
    Ok(model)
}

/// The complete report of one model with the command's parameters: the target, or the
/// `--baseline` model.
fn build_report(
    model_path: &Path,
    args: &GraphArgs,
    settings: &Settings,
    deps: &AppDeps,
    sink: &mut dyn ProgressSink,
    cancel: &AtomicBool,
) -> Result<GraphReport> {
    let model = open_model(model_path, args, deps, sink, cancel)?;
    if model.tensors.is_empty() {
        return Err(GraphError::Model("aucune matrice du flux résiduel".into()));
    }

    let vector = match &settings.vector {
        Some(path) => Some(cache::read_vector(path, model.hidden)?),
        None => None,
    };
    let vector_sha = vector.as_deref().map(cache::vector_digest);

    let mut nodes: Vec<&TensorRef> = model
        .tensors
        .iter()
        .filter(|tensor| {
            settings.roles.contains(&tensor.role) && settings.layer_selected(tensor.layer)
        })
        .collect();
    if nodes.is_empty() {
        return Err(GraphError::Usage(
            "aucune matrice ne survit à --layers/--matrices".into(),
        ));
    }
    if let Some(max) = args.max_nodes {
        if nodes.len() > max {
            nodes.sort_by(|a, b| {
                b.weight_bytes
                    .cmp(&a.weight_bytes)
                    .then_with(|| a.name.cmp(&b.name))
                    .then_with(|| a.expert.cmp(&b.expert))
            });
            nodes.truncate(max);
            nodes.sort_by(|a, b| {
                causal_seq(a.role, a.layer, model.layers, model.parallel_blocks)
                    .cmp(&causal_seq(
                        b.role,
                        b.layer,
                        model.layers,
                        model.parallel_blocks,
                    ))
                    .then_with(|| a.name.cmp(&b.name))
                    .then_with(|| a.expert.cmp(&b.expert))
            });
        }
    }

    let mut cache = match (&settings.cache_dir, settings.no_cache) {
        (Some(dir), false) => Cache::open(dir, &model.sha256, settings.resume)?,
        _ => Cache::disabled(),
    };

    sink.event(ProgressEvent::GraphStart {
        model: model_path.to_path_buf(),
        matrices: nodes.len(),
        rank: args.rank,
    });

    let options = AnalyzeOptions {
        params: RsvdParams {
            rank: args.rank,
            oversample: args.oversample,
            power_iters: args.power_iters,
            seed: args.seed,
        },
        full_spectrum_max_dim: args.full_spectrum_max_dim,
        kurtosis_threshold: args.kurtosis_threshold,
        vector: vector.as_deref(),
        cancel,
        external: Some(&deps.interrupted),
    };
    let pool = match args.jobs {
        0 => None,
        threads => Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| GraphError::Compute(format!("--jobs {threads} : {e}")))?,
        ),
    };
    let mut computed: Vec<(NodeReport, Basis, Vec<f32>)> = Vec::with_capacity(nodes.len());
    let mut skipped_full_spectrum = 0usize;
    for (index, tensor) in nodes.iter().enumerate() {
        check_cancel(deps, cancel)?;
        sink.event(ProgressEvent::GraphNode {
            id: node_id(tensor.role, tensor.layer, tensor.expert),
            done: index,
            total: nodes.len(),
        });
        let key = cache::key_for(&model.sha256, tensor, &options, vector_sha.clone())?;
        let analysis = match cache.get(&key) {
            Some(cached) => cached.clone(),
            None => {
                let analysis = match &pool {
                    Some(pool) => {
                        pool.install(|| metrics::analyze(&model, tensor, model.hidden, &options))?
                    }
                    None => metrics::analyze(&model, tensor, model.hidden, &options)?,
                };
                cache.put(&key, &analysis)?;
                analysis
            }
        };
        if args.full_spectrum_max_dim > 0 && !analysis.metrics.full_spectrum {
            skipped_full_spectrum += 1;
        }
        let seq = causal_seq(
            tensor.role,
            tensor.layer,
            model.layers,
            model.parallel_blocks,
        );
        let kind = match tensor.role.is_writer() {
            true => NodeKind::Writer,
            false => NodeKind::Reader,
        };
        let report = NodeReport {
            id: node_id(tensor.role, tensor.layer, tensor.expert),
            role: tensor.role,
            kind,
            layer: tensor.layer,
            expert: tensor.expert,
            tensor: tensor.name.clone(),
            dtype: tensor.dtype.clone(),
            weight_bytes: tensor.weight_bytes,
            rows: tensor.m(),
            cols: tensor.n(),
            seq,
            metrics: analysis.metrics.clone(),
            // Filled by layout::apply once the edges are known.
            pos_column: [0.0; 3],
            pos_constellation: [0.0; 3],
        };
        computed.push((report, analysis.basis, analysis.energy_weights));
    }
    check_cancel(deps, cancel)?;

    let edges = build_edges(
        &computed,
        model.hidden,
        model.layers,
        args.edges,
        args.edge_weight,
        args.min_alignment,
    );
    sink.event(ProgressEvent::GraphEdges {
        nodes: computed.len(),
        edges: edges.len(),
    });

    let outliers: Vec<usize> = computed
        .iter()
        .flat_map(|(node, _, _)| node.metrics.channels.outliers.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut kurtosis: Vec<f64> = computed
        .iter()
        .map(|(n, _, _)| n.metrics.kurtosis)
        .collect();
    kurtosis.sort_by(f64::total_cmp);
    let mean = |values: &mut dyn Iterator<Item = f64>| {
        let mut count = 0usize;
        let mut sum = 0.0;
        for value in values {
            count += 1;
            sum += value;
        }
        match count {
            0 => 0.0,
            count => sum / count as f64,
        }
    };
    let mean_lower = mean(
        &mut computed
            .iter()
            .map(|(n, _, _)| n.metrics.erank_energy_lower),
    );
    let mean_upper = mean(
        &mut computed
            .iter()
            .map(|(n, _, _)| n.metrics.erank_energy_upper),
    );
    let non_finite = computed
        .iter()
        .map(|(n, _, _)| n.metrics.non_finite)
        .sum::<u64>();
    let full_spectrum_nodes = computed
        .iter()
        .filter(|(n, _, _)| n.metrics.full_spectrum)
        .count();
    let structural_edges = edges.iter().filter(|e| e.structural).count();
    let total_weight_bytes = unique_weight_bytes(computed.iter().map(|(n, _, _)| n));
    let mut warnings = model.warnings.clone();
    if skipped_full_spectrum > 0 {
        warnings.push(format!(
            "{skipped_full_spectrum} matrice(s) sans spectre complet : min(m,n) > {} ou min(m,n) > {} ou m·n > {}",
            args.full_spectrum_max_dim,
            MAX_N,
            metrics::FULL_SPECTRUM_PRODUCT_CAP
        ));
    }
    let mut nodes: Vec<NodeReport> = computed.into_iter().map(|(node, _, _)| node).collect();
    layout::apply(&mut nodes, &edges);
    let report = GraphReport {
        schema_version: SCHEMA_VERSION,
        tool: ToolInfo {
            name: "surgeon".into(),
            version: env!("CARGO_PKG_VERSION").into(),
        },
        model: ModelReport {
            path: model.path.display().to_string(),
            sha256: model.sha256.clone(),
            hash_mode: match args.fast_hash {
                true => "fast",
                false => "sha256",
            }
            .into(),
            format: model.format.as_str().into(),
            architecture: model.arch_id.clone(),
            arch_source: model.arch_source.clone(),
            hidden: model.hidden,
            layers: model.layers,
            tied_lm_head: model.tied_lm_head,
            tensor_count: nodes.len(),
            total_weight_bytes,
            skipped: model.skipped.clone(),
            warnings: model.warnings.clone(),
        },
        parameters: ParametersReport {
            rank: args.rank,
            oversample: args.oversample,
            power_iters: args.power_iters,
            seed: args.seed,
            full_spectrum_max_dim: args.full_spectrum_max_dim,
            edge_mode: args.edges.as_str(),
            min_alignment: args.min_alignment,
            edge_weight: args.edge_weight.as_str().into(),
            vector: settings.vector.as_ref().map(|p| p.display().to_string()),
            kurtosis_threshold: args.kurtosis_threshold,
            color_scale: settings.color_scale.as_str().into(),
            palette: settings.palette.as_str().into(),
            max_nodes: args.max_nodes,
            formats: settings
                .formats
                .iter()
                .map(|f| f.as_str().to_owned())
                .collect(),
        },
        global: GlobalReport {
            nodes: nodes.len(),
            edges: edges.len(),
            structural_edges,
            layers: model.layers,
            hidden: model.hidden,
            outlier_channels: outliers,
            kurtosis_median: median(&kurtosis),
            kurtosis_max: kurtosis.last().copied().unwrap_or(0.0),
            mean_erank_energy_lower: mean_lower,
            mean_erank_energy_upper: mean_upper,
            full_spectrum_nodes,
            non_finite_values: non_finite,
            warnings: warnings.clone(),
        },
        nodes,
        edges,
        baseline_diff: None,
    };
    Ok(report)
}

fn check_cancel(deps: &AppDeps, cancel: &AtomicBool) -> Result<()> {
    if deps.interrupted.load(Ordering::Relaxed) || cancel.load(Ordering::Relaxed) {
        return Err(GraphError::Cancelled);
    }
    Ok(())
}

fn unique_weight_bytes<'a>(nodes: impl Iterator<Item = &'a NodeReport>) -> u64 {
    let mut seen = BTreeSet::new();
    let mut total = 0u64;
    for node in nodes {
        let name = node
            .tensor
            .strip_suffix(" (tête liée)")
            .unwrap_or(&node.tensor);
        if seen.insert(name.to_owned()) {
            total += node.weight_bytes;
        }
    }
    total
}

fn median(sorted: &[f64]) -> f64 {
    if sorted.is_empty() {
        0.0
    } else if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    }
}

fn render_files(report: &GraphReport, formats: &[OutputFormat]) -> Vec<OutputFile> {
    let mut files = Vec::new();
    for format in formats {
        match format {
            OutputFormat::Json => files.push(OutputFile {
                name: "graph.json",
                contents: output::render_json(report).unwrap_or_default(),
            }),
            OutputFormat::Dot => files.push(OutputFile {
                name: "graph.dot",
                contents: output::render_dot(report),
            }),
            OutputFormat::Mmd => files.push(OutputFile {
                name: "graph.mmd",
                contents: output::render_mermaid(report),
            }),
            // The first line of a stream: surgeon-gui opens it and follows what is appended.
            OutputFormat::Ndjson => files.push(OutputFile {
                name: "graph.ndjson",
                contents: stream::GraphPatch::snapshot(report)
                    .to_ndjson()
                    .unwrap_or_default(),
            }),
            OutputFormat::Csv => {
                files.push(OutputFile {
                    name: "matrices.csv",
                    contents: output::render_matrices_csv(report),
                });
                files.push(OutputFile {
                    name: "edges.csv",
                    contents: output::render_edges_csv(report),
                });
            }
        }
    }
    files
}

// ------------------------------------------------------------------ edges

fn build_edges(
    nodes: &[(NodeReport, Basis, Vec<f32>)],
    hidden: usize,
    layers: usize,
    mode: EdgeMode,
    edge_weight: EdgeWeight,
    min_alignment: f64,
) -> Vec<EdgeReport> {
    let mut scored: Vec<(usize, usize, metrics::Similarity)> = Vec::new();
    for (writer_index, (writer, write_basis, write_energy)) in nodes.iter().enumerate() {
        if !writer.role.is_writer() {
            continue;
        }
        for (reader_index, (reader, read_basis, read_energy)) in nodes.iter().enumerate() {
            if !reader.role.is_reader() || writer.seq >= reader.seq {
                continue;
            }
            let similarity =
                metrics::similarity(write_basis, read_basis, hidden, write_energy, read_energy);
            if similarity.alignment < min_alignment {
                continue;
            }
            scored.push((writer_index, reader_index, similarity));
        }
    }
    let adjacency: Vec<bool> = scored
        .iter()
        .map(|(writer, reader, _)| is_adjacent(&nodes[*writer].0, &nodes[*reader].0, layers))
        .collect();
    let mut keep = vec![false; scored.len()];
    match mode {
        EdgeMode::Adjacent => {
            for (index, adjacent) in adjacency.iter().enumerate() {
                keep[index] = *adjacent;
            }
        }
        EdgeMode::All => keep.fill(true),
        EdgeMode::TopK(k) => {
            let mut per_writer: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for (index, (writer, _, _)) in scored.iter().enumerate() {
                if !adjacency[index] {
                    per_writer.entry(*writer).or_default().push(index);
                }
            }
            for (index, adjacent) in adjacency.iter().enumerate() {
                if *adjacent {
                    keep[index] = true;
                }
            }
            for indices in per_writer.values_mut() {
                indices.sort_by(|left, right| {
                    let weight = |index: usize| match edge_weight {
                        EdgeWeight::Alignment => scored[index].2.alignment,
                        EdgeWeight::Coupling => scored[index].2.coupling,
                    };
                    weight(*right)
                        .total_cmp(&weight(*left))
                        .then_with(|| scored[*left].0.cmp(&scored[*right].0))
                        .then_with(|| scored[*left].1.cmp(&scored[*right].1))
                });
                for index in indices.iter().take(k) {
                    keep[*index] = true;
                }
            }
        }
    }
    let mut edges = Vec::new();
    for (index, keep) in keep.iter().enumerate() {
        if !keep {
            continue;
        }
        let (writer_index, reader_index, similarity) = &scored[index];
        let writer = &nodes[*writer_index].0;
        let reader = &nodes[*reader_index].0;
        let weight = match edge_weight {
            EdgeWeight::Alignment => similarity.alignment,
            EdgeWeight::Coupling => similarity.coupling,
        };
        edges.push(EdgeReport {
            source: writer.id.clone(),
            target: reader.id.clone(),
            writer_role: writer.role,
            reader_role: reader.role,
            source_seq: writer.seq,
            target_seq: reader.seq,
            layer_gap: layer_gap(writer, reader),
            rank_used: similarity.rank_used,
            alignment: similarity.alignment,
            chance: similarity.chance,
            alignment_norm: similarity.alignment_norm,
            coupling: similarity.coupling,
            cos_theta: similarity.cos_theta,
            weight,
            structural: false,
        });
    }
    // MoE structural edges: gate/up expert e -> the same expert's down_proj, causally
    // ordered (gate/up at 4L+3, down at 4L+4). They carry no residual alignment.
    for (writer, _, _) in nodes {
        if writer.role != GraphRole::MlpOut {
            continue;
        }
        let Some(expert) = writer.expert else {
            continue;
        };
        for (reader, _, _) in nodes {
            if !matches!(reader.role, GraphRole::Gate | GraphRole::Up)
                || reader.layer != writer.layer
                || reader.expert != Some(expert)
            {
                continue;
            }
            edges.push(EdgeReport {
                source: reader.id.clone(),
                target: writer.id.clone(),
                writer_role: writer.role,
                reader_role: reader.role,
                source_seq: reader.seq,
                target_seq: writer.seq,
                layer_gap: 0,
                rank_used: 0,
                alignment: 0.0,
                chance: 0.0,
                alignment_norm: 0.0,
                coupling: 0.0,
                cos_theta: 0.0,
                weight: 0.0,
                structural: true,
            });
        }
    }
    edges.sort_by(|a, b| {
        a.source_seq
            .cmp(&b.source_seq)
            .then_with(|| a.target_seq.cmp(&b.target_seq))
            .then_with(|| a.source.cmp(&b.source))
            .then_with(|| a.target.cmp(&b.target))
    });
    edges
}

fn layer_gap(writer: &NodeReport, reader: &NodeReport) -> i64 {
    match (writer.layer, reader.layer) {
        (Some(w), Some(r)) => r as i64 - w as i64,
        _ => 0,
    }
}

/// Adjacent = same-layer later readers plus every reader of the next layer (and the head),
/// or, from the embedding, every reader of layer 0.
fn is_adjacent(writer: &NodeReport, reader: &NodeReport, layers: usize) -> bool {
    if writer.role == GraphRole::Embed {
        return reader.layer == Some(0);
    }
    if reader.role == GraphRole::LmHead {
        return layers > 0 && writer.layer == Some(layers - 1);
    }
    match (writer.layer, reader.layer) {
        (Some(w), Some(r)) if w == r => reader.seq > writer.seq,
        (Some(w), Some(r)) => r == w + 1,
        _ => false,
    }
}

// ------------------------------------------------------------------ estimate

fn estimate(args: &GraphArgs, settings: &Settings, model: &ModelData) -> CoreResult {
    let nodes: Vec<&TensorRef> = model
        .tensors
        .iter()
        .filter(|tensor| {
            settings.roles.contains(&tensor.role) && settings.layer_selected(tensor.layer)
        })
        .collect();
    if nodes.is_empty() {
        return Err(
            GraphError::Usage("aucune matrice ne survit à --layers/--matrices".into()).into(),
        );
    }
    let passes = 4 + 2 * args.power_iters;
    let mut bytes = 0u64;
    let mut flops = 0.0f64;
    let mut full_spectrum_nodes = 0usize;
    for tensor in &nodes {
        let (m, n) = (tensor.m(), tensor.n());
        let k = (args.rank + args.oversample).min(m).min(n);
        let full = args.full_spectrum_max_dim > 0
            && m.min(n) <= args.full_spectrum_max_dim
            && m.min(n) <= MAX_N
            && m.saturating_mul(n) <= metrics::FULL_SPECTRUM_PRODUCT_CAP;
        let node_passes = passes + usize::from(full);
        bytes = bytes.saturating_add(tensor.weight_bytes.saturating_mul(node_passes as u64));
        flops += (m as f64)
            * (n as f64)
            * (2.0 * k as f64 * (2.0 + 2.0 * args.power_iters as f64) + 4.0);
        if full {
            full_spectrum_nodes += 1;
            flops += (m as f64) * (n as f64) * 2.0;
        }
    }
    let bench = model::micro_bench_seconds();
    let bench_flops = 3.0 * 2.0 * 128.0f64.powi(3);
    let throughput = bench_flops / bench.max(1e-9);
    let seconds = flops / throughput.max(1.0);
    let value = json!({
        "estimate": true,
        "model": {
            "path": model.path.display().to_string(),
            "sha256": model.sha256,
            "format": model.format.as_str(),
            "architecture": model.arch_id,
            "hidden": model.hidden,
            "layers": model.layers,
        },
        "parameters": {
            "rank": args.rank,
            "oversample": args.oversample,
            "power_iters": args.power_iters,
            "seed": args.seed,
            "edge_mode": args.edges.as_str(),
            "full_spectrum_max_dim": args.full_spectrum_max_dim,
        },
        "nodes": nodes.len(),
        "passes_per_matrix": passes,
        "full_spectrum_nodes": full_spectrum_nodes,
        "bytes_read": bytes,
        "flops": flops,
        "micro_bench_seconds": bench,
        "estimated_seconds": seconds,
        "notes": [
            "worst case, no cache: model hashing and output writing excluded",
            "the SVD dominates: rank + oversample columns, power iterations",
        ],
    });
    let human = format!(
        "surgeon graph --estimate · {} matrices · {} passes · {:.3} Gio lus · ~{:.1} s",
        nodes.len(),
        passes,
        bytes as f64 / (1u64 << 30) as f64,
        seconds
    );
    Ok(Outcome::ok(value, human))
}

// ------------------------------------------------------------------ watchdog

struct Watchdog {
    stop: Option<std::sync::mpsc::Sender<()>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Watchdog {
    fn start(seconds: Option<f64>, flag: Arc<AtomicBool>) -> Self {
        let Some(seconds) = seconds else {
            return Self {
                stop: None,
                handle: None,
            };
        };
        let (stop, receiver) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            if receiver
                .recv_timeout(Duration::from_secs_f64(seconds))
                .is_err()
            {
                flag.store(true, Ordering::SeqCst);
            }
        });
        Self {
            stop: Some(stop),
            handle: Some(handle),
        }
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

// ------------------------------------------------------------------ schema

/// JSON Schema of `graph.json` (`surgeon schema --kind graph`).
pub fn schema() -> serde_json::Value {
    output::schema()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_mode_parses_its_three_forms() {
        assert_eq!(EdgeMode::from_str("adjacent"), Ok(EdgeMode::Adjacent));
        assert_eq!(EdgeMode::from_str("all"), Ok(EdgeMode::All));
        assert_eq!(EdgeMode::from_str("topk:7"), Ok(EdgeMode::TopK(7)));
        assert!(EdgeMode::from_str("topk:0").is_err());
        assert!(EdgeMode::from_str("nearest").is_err());
    }

    #[test]
    fn causal_order_handles_parallel_blocks() {
        assert_eq!(causal_seq(GraphRole::Embed, None, 2, false), 0);
        assert_eq!(causal_seq(GraphRole::Q, Some(1), 2, false), 5);
        assert_eq!(causal_seq(GraphRole::AttnOut, Some(1), 2, false), 6);
        assert_eq!(causal_seq(GraphRole::Gate, Some(1), 2, false), 7);
        assert_eq!(causal_seq(GraphRole::MlpOut, Some(1), 2, false), 8);
        assert_eq!(causal_seq(GraphRole::Gate, Some(1), 2, true), 5);
        assert_eq!(causal_seq(GraphRole::MlpOut, Some(1), 2, true), 6);
        assert_eq!(causal_seq(GraphRole::LmHead, None, 2, false), 9);
    }

    #[test]
    fn node_ids_include_experts_and_roles() {
        assert_eq!(node_id(GraphRole::Q, Some(2), None), "L2.q");
        assert_eq!(
            node_id(GraphRole::MlpOut, Some(2), Some(3)),
            "L2.mlp_out.e3"
        );
        assert_eq!(node_id(GraphRole::Embed, None, None), "embed");
        assert_eq!(node_id(GraphRole::LmHead, None, None), "lm_head");
    }

    #[test]
    fn settings_validate_bad_ranks_and_cache_flags() {
        let mut args = GraphArgs {
            model: PathBuf::from("m"),
            out_dir: None,
            format: vec![OutputFormat::Json],
            layers: None,
            matrices: Vec::new(),
            rank: 32,
            oversample: 10,
            power_iters: 3,
            seed: 0,
            full_spectrum_max_dim: 2048,
            edges: EdgeMode::Adjacent,
            min_alignment: 0.0,
            edge_weight: EdgeWeight::Alignment,
            vector: None,
            kurtosis_threshold: 10.0,
            color_scale: ColorScaleArg::Quantile,
            palette: PaletteArg::Diverging,
            max_nodes: None,
            cache_dir: None,
            no_cache: false,
            resume: false,
            estimate: false,
            jobs: 0,
            overwrite: false,
            baseline: None,
            fast_hash: false,
        };
        assert!(args.check().is_ok());
        args.rank = 0;
        assert!(args.check().is_err());
        args.rank = 32;
        args.resume = true;
        assert!(args.check().is_err());
        args.resume = false;
        args.cache_dir = Some(PathBuf::from("c"));
        args.no_cache = true;
        assert!(args.check().is_err());
    }
}
