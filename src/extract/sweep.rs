use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use serde::Serialize;
use serde_json::Value;

use super::{
    capture_sets, check_direction, companion_path, encode_tensors, file_metadata, load_config,
    merge_metadata, method_metadata, prompt_warnings, separation_along, validate_method,
    validate_request, write_pair_atomic, Captured, Companion, ExtractError, ExtractEvent,
    ExtractObserver, ExtractRequest, Result, SafetensorsDir, Separation, Session, TensorSource,
    COMPANION_SCHEMA, TENSOR_NAME,
};
use crate::hardware::HardwareProbe;
use crate::patch::LayerSelection;
use crate::plugin::{ActivationSet, Diagnostics, ExtractContext, ExtractionMethod};

#[derive(Debug, Clone)]
pub struct SweepRequest {
    pub extract: ExtractRequest,
    pub layers: LayerSelection,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LayerSeparation {
    pub layer: usize,
    pub raw_diff_norm: f64,
    pub auc: f64,
    pub margin: f64,
    pub d_prime: Option<f64>,
    pub threshold_accuracy: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SweepReport {
    pub out_path: PathBuf,
    pub stats_path: PathBuf,
    pub method: &'static str,
    pub site: &'static str,
    pub pooling: &'static str,
    pub hidden_size: usize,
    pub n_pos: usize,
    pub n_neg: usize,
    pub dtype_used: String,
    pub best_layer: usize,
    pub layers: Vec<LayerSeparation>,
    pub warnings: Vec<String>,
    pub duration_ms: u128,
}

pub fn resolve_layers(selection: &LayerSelection, num_layers: usize) -> Result<Vec<usize>> {
    let layers: Vec<usize> = match selection {
        LayerSelection::All => (0..num_layers).collect(),
        LayerSelection::Range {
            start,
            end_inclusive,
        } => (*start..=*end_inclusive).collect(),
        LayerSelection::List(list) => {
            let mut list = list.clone();
            list.sort_unstable();
            list.dedup();
            list
        }
    };
    match layers.last() {
        None => Err(ExtractError::InvalidArgument(
            "aucune couche sélectionnée".into(),
        )),
        Some(&last) if last >= num_layers => Err(ExtractError::InvalidArgument(format!(
            "couche {last} hors bornes : le modèle a num_hidden_layers = {num_layers} (couches 0 à {})",
            num_layers.saturating_sub(1)
        ))),
        Some(_) => Ok(layers),
    }
}

pub fn auc(pos: &[f64], neg: &[f64]) -> f64 {
    let wins: f64 = pos
        .iter()
        .flat_map(|p| neg.iter().map(move |n| (p, n)))
        .map(|(p, n)| match p.partial_cmp(n) {
            Some(std::cmp::Ordering::Greater) => 1.0,
            Some(std::cmp::Ordering::Equal) => 0.5,
            _ => 0.0,
        })
        .sum();
    wins / (pos.len() * neg.len()).max(1) as f64
}

pub fn margin(pos: &[f64], neg: &[f64]) -> f64 {
    let lowest_pos = pos.iter().copied().fold(f64::INFINITY, f64::min);
    let highest_neg = neg.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    lowest_pos - highest_neg
}

pub fn best_layer_index(scores: &[LayerSeparation]) -> Option<usize> {
    let key = |s: &LayerSeparation| (s.auc, s.d_prime.unwrap_or(f64::INFINITY));
    (0..scores.len()).reduce(|best, i| match key(&scores[i]) > key(&scores[best]) {
        true => i,
        false => best,
    })
}

struct LayerFit {
    score: LayerSeparation,
    separation: Separation,
    metadata: BTreeMap<String, String>,
    diagnostics: Option<Diagnostics>,
}

fn fit_layers(
    ctx: &ExtractContext,
    layers: &[usize],
    pos: &[Vec<Vec<f64>>],
    neg: &[Vec<Vec<f64>>],
    method: &dyn ExtractionMethod,
    cancel: &AtomicBool,
) -> Result<(Vec<LayerFit>, Vec<String>)> {
    let mut fits = Vec::with_capacity(layers.len());
    let mut warnings: Vec<String> = Vec::new();
    for ((&layer, pos), neg) in layers.iter().zip(pos).zip(neg) {
        if cancel.load(Ordering::SeqCst) {
            return Err(ExtractError::Cancelled);
        }
        let ctx = ExtractContext { layer, ..*ctx };
        let out = method.extract(&ctx, &ActivationSet { pos, neg }, cancel)?;
        if !out.sidecars.is_empty() {
            return Err(ExtractError::InvalidArgument(format!(
                "{} écrit des fichiers annexes : un balayage de couches garde une seule \
                 direction par couche (rank 1, sans leace)",
                method.id()
            )));
        }
        check_direction(&out.direction, ctx.hidden_size)?;
        let separation = separation_along(pos, neg, &out.direction)?;
        for warning in out.warnings {
            if !warnings.contains(&warning) {
                warnings.push(warning);
            }
        }
        let (p, n) = (&separation.pos_projections, &separation.neg_projections);
        let stats = &separation.stats;
        fits.push(LayerFit {
            score: LayerSeparation {
                layer,
                raw_diff_norm: stats.raw_diff_norm,
                auc: auc(p, n),
                margin: margin(p, n),
                d_prime: stats.d_prime,
                threshold_accuracy: stats.threshold_accuracy,
            },
            separation,
            metadata: out.metadata,
            diagnostics: out.diagnostics,
        });
    }
    Ok((fits, warnings))
}

pub fn run_sweep(
    req: &SweepRequest,
    method: &dyn ExtractionMethod,
    params: &Value,
    probe: &dyn HardwareProbe,
    observer: &mut dyn ExtractObserver,
    cancel: &AtomicBool,
) -> Result<SweepReport> {
    validate_request(&req.extract)?;
    validate_method(&req.extract, method, params)?;
    let source = SafetensorsDir::open(&req.extract.model_dir)?;
    sweep(req, method, params, &source, probe, observer, cancel)
}

fn sweep(
    req: &SweepRequest,
    method: &dyn ExtractionMethod,
    params: &Value,
    source: &dyn TensorSource,
    probe: &dyn HardwareProbe,
    observer: &mut dyn ExtractObserver,
    cancel: &AtomicBool,
) -> Result<SweepReport> {
    let started = Instant::now();
    let base = &req.extract;
    let cfg = load_config(&base.model_dir)?;
    let layers = resolve_layers(&req.layers, cfg.num_hidden_layers)?;
    let Captured { pos, neg, session } =
        capture_sets(base, &cfg, &layers, source, probe, observer)?;
    let Session {
        format,
        dtype_used,
        mut warnings,
        truncated,
        ..
    } = session;
    prompt_warnings(&mut warnings, base, truncated);
    if params
        .get("holdout_frac")
        .and_then(Value::as_f64)
        .is_some_and(|f| f > 0.0)
    {
        warnings.push(
            "holdout_frac ignoré : le balayage mesure chaque couche sur tous les prompts".into(),
        );
    }
    let ctx = ExtractContext {
        hidden_size: cfg.hidden_size,
        layer: 0,
        site: base.site,
        pooling: base.pooling,
        params,
    };
    let (fits, method_warnings) = fit_layers(&ctx, &layers, &pos, &neg, method, cancel)?;
    warnings.extend(method_warnings);
    let scores: Vec<LayerSeparation> = fits.iter().map(|f| f.score.clone()).collect();
    let best = best_layer_index(&scores)
        .ok_or_else(|| ExtractError::InvalidArgument("aucune couche sélectionnée".into()))?;
    let best_fit = &fits[best];
    let best_layer = best_fit.score.layer;
    if observer.on_event(ExtractEvent::Writing).is_break() {
        return Err(ExtractError::Cancelled);
    }

    let best_request = ExtractRequest {
        layer: best_layer,
        ..base.clone()
    };
    let (n_pos, n_neg) = (base.pos_prompts.len(), base.neg_prompts.len());
    let stats = &best_fit.separation.stats;
    let mut metadata = file_metadata(
        &best_request,
        format.template,
        &cfg,
        &dtype_used,
        n_pos,
        n_neg,
        stats,
    );
    if let Some(system) = &format.system {
        metadata.insert("chat_system".into(), system.clone());
    }
    merge_metadata(
        &mut metadata,
        &method_metadata(method.id(), &best_fit.metadata)?,
    )?;
    let scores_json =
        serde_json::to_string(&scores).map_err(|e| ExtractError::Compute(e.to_string()))?;
    metadata.insert("layer_scores".into(), scores_json);

    let names: Vec<String> = layers.iter().map(|l| format!("layers.{l}")).collect();
    let mut tensors: Vec<(&str, &[f32])> = vec![(TENSOR_NAME, &best_fit.separation.direction)];
    tensors.extend(
        names
            .iter()
            .zip(&fits)
            .map(|(name, fit)| (name.as_str(), fit.separation.direction.as_slice())),
    );
    let main_bytes = encode_tensors(&tensors, &metadata)?;
    let companion = Companion {
        schema: COMPANION_SCHEMA,
        pos_projections: &best_fit.separation.pos_projections,
        neg_projections: &best_fit.separation.neg_projections,
        threshold: stats.threshold,
        d_prime: stats.d_prime,
        threshold_accuracy: stats.threshold_accuracy,
        cos_mean_pos_neg: stats.cosine_means,
        raw_diff_norm: stats.raw_diff_norm,
        layer: best_layer,
        site: base.site.as_str(),
        pooling: base.pooling.as_str(),
        hidden_size: cfg.hidden_size,
        method_diagnostics: best_fit.diagnostics.as_ref(),
        layer_scores: Some(&scores),
    };
    let mut companion_bytes =
        serde_json::to_vec_pretty(&companion).map_err(|e| ExtractError::Compute(e.to_string()))?;
    companion_bytes.push(b'\n');
    write_pair_atomic(&base.out, &main_bytes, &companion_bytes, base.overwrite)?;

    Ok(SweepReport {
        out_path: base.out.clone(),
        stats_path: companion_path(&base.out),
        method: method.id(),
        site: base.site.as_str(),
        pooling: base.pooling.as_str(),
        hidden_size: cfg.hidden_size,
        n_pos,
        n_neg,
        dtype_used,
        best_layer,
        layers: scores,
        warnings,
        duration_ms: started.elapsed().as_millis(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::{Method, Pooling, Site};
    use crate::plugin::Builtin;

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        }
    }

    const DIM: usize = 6;
    const PAIRS: usize = 12;

    type LayerRows = Vec<Vec<Vec<f64>>>;

    fn paired_layers(strengths: &[f64]) -> (LayerRows, LayerRows) {
        let mut rng = Lcg(7);
        let mut pos = Vec::new();
        let mut neg = Vec::new();
        for &strength in strengths {
            let (mut p, mut n) = (Vec::new(), Vec::new());
            for _ in 0..PAIRS {
                let base: Vec<f64> = (0..DIM).map(|_| rng.next()).collect();
                let jitter: Vec<f64> = (0..DIM).map(|_| 0.05 * rng.next()).collect();
                let mut shifted: Vec<f64> = base.iter().zip(&jitter).map(|(b, j)| b + j).collect();
                shifted[2] += strength;
                p.push(shifted);
                n.push(base);
            }
            pos.push(p);
            neg.push(n);
        }
        (pos, neg)
    }

    fn sweep_scores(method: Method) -> Vec<LayerSeparation> {
        let (pos, neg) = paired_layers(&[0.0, 0.4, 3.0]);
        let params = Value::Null;
        let ctx = ExtractContext {
            hidden_size: DIM,
            layer: 0,
            site: Site::ResidPost,
            pooling: Pooling::Last,
            params: &params,
        };
        let never = AtomicBool::new(false);
        let (fits, _) = fit_layers(&ctx, &[3, 4, 5], &pos, &neg, &Builtin(method), &never).unwrap();
        fits.into_iter().map(|f| f.score).collect()
    }

    #[test]
    fn mean_diff_and_pca_both_find_the_separable_layer() {
        for method in [Method::MeanDiff, Method::Svd] {
            let scores = sweep_scores(method);
            assert_eq!(
                scores.iter().map(|s| s.layer).collect::<Vec<_>>(),
                [3, 4, 5]
            );
            let strong = &scores[2];
            assert_eq!(strong.auc, 1.0, "{method:?}");
            assert!(strong.margin > 0.0, "{method:?}: {}", strong.margin);
            assert!(
                scores[0].auc < 0.9,
                "{method:?}: noise only {}",
                scores[0].auc
            );
            assert!(scores[2].raw_diff_norm > scores[1].raw_diff_norm);
            assert_eq!(scores[best_layer_index(&scores).unwrap()].layer, 5);
        }
    }

    #[test]
    fn auc_and_margin_on_known_projections() {
        assert_eq!(auc(&[3.0, 4.0], &[1.0, 2.0]), 1.0);
        assert_eq!(auc(&[1.0, 2.0], &[3.0, 4.0]), 0.0);
        assert_eq!(auc(&[1.0], &[1.0]), 0.5);
        assert_eq!(auc(&[2.0, 0.0], &[1.0]), 0.5);
        assert_eq!(margin(&[3.0, 4.0], &[1.0, 2.0]), 1.0);
        assert_eq!(margin(&[1.0, 4.0], &[2.0]), -1.0);
    }

    #[test]
    fn ties_on_auc_go_to_the_larger_d_prime_then_the_earliest_layer() {
        let score = |layer, auc, d_prime| LayerSeparation {
            layer,
            raw_diff_norm: 1.0,
            auc,
            margin: 0.0,
            d_prime,
            threshold_accuracy: 1.0,
        };
        let scores = [
            score(0, 0.9, Some(5.0)),
            score(1, 1.0, Some(2.0)),
            score(2, 1.0, Some(3.0)),
            score(3, 1.0, Some(3.0)),
        ];
        assert_eq!(best_layer_index(&scores), Some(2));
        let perfect = [score(0, 1.0, Some(9.0)), score(1, 1.0, None)];
        assert_eq!(best_layer_index(&perfect), Some(1));
        assert_eq!(best_layer_index(&[]), None);
    }

    #[test]
    fn layer_selections_resolve_against_the_model_depth() {
        assert_eq!(resolve_layers(&LayerSelection::All, 3).unwrap(), [0, 1, 2]);
        let range = LayerSelection::Range {
            start: 1,
            end_inclusive: 2,
        };
        assert_eq!(resolve_layers(&range, 3).unwrap(), [1, 2]);
        let list = LayerSelection::List(vec![2, 0, 2]);
        assert_eq!(resolve_layers(&list, 3).unwrap(), [0, 2]);
        assert!(resolve_layers(&LayerSelection::List(vec![3]), 3).is_err());
        assert!(resolve_layers(&LayerSelection::List(vec![]), 3).is_err());
    }
}
