//! The built-in direction methods `mean_diff | svd | leace` (`crate::plugin::Registry`
//! exposes them through the extension API). `mean_diff` keeps the prompt 8 outputs byte
//! for byte.
//!
//! Hold-out (`--holdout-frac f`): sample `i` of each class is held out when
//! `⌊(i+1)·f⌋ > ⌊i·f⌋` (evenly spread and deterministic; `f = 0.5` holds out the odd
//! indices). The written direction always uses every prompt; the hold-out diagnostic refits
//! on the remaining samples and measures on the held-out ones. Fewer than 10 samples in a
//! class disables it with a warning.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::error::{ExtractError, Result};
use super::leace::{self, LeaceDiagnostics, LeaceFit, LeaceSubspace};
use super::subspace::{self, SubspaceOptions};
use super::svd::{svd_direction, svd_subspace, SvdDiagnostics};

pub const MIN_HOLDOUT_PER_CLASS: usize = 10;

/// The built-in methods; their descriptions live in `plugin::builtin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Method {
    #[default]
    MeanDiff,
    Svd,
    Leace,
}

impl Method {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MeanDiff => "mean_diff",
            Self::Svd => "svd",
            Self::Leace => "leace",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        [Self::MeanDiff, Self::Svd, Self::Leace]
            .into_iter()
            .find(|m| m.as_str() == id)
    }
}

/// Options of the direction methods, next to `ExtractRequest` (kept field for field as
/// in prompt 8, since callers build it literally).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MethodOptions {
    pub method: Method,
    pub svd_center: bool,
    pub leace_shrinkage: f64,
    pub leace_rcond: f64,
    pub holdout_frac: f64,
}

pub const DEFAULT_LEACE_SHRINKAGE: f64 = 0.1;
pub const DEFAULT_LEACE_RCOND: f64 = 1e-10;

impl Default for MethodOptions {
    fn default() -> Self {
        Self {
            method: Method::MeanDiff,
            svd_center: false,
            leace_shrinkage: DEFAULT_LEACE_SHRINKAGE,
            leace_rcond: DEFAULT_LEACE_RCOND,
            holdout_frac: 0.0,
        }
    }
}

impl MethodOptions {
    pub fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(ExtractError::InvalidArgument(m.to_owned()));
        if !(0.0..=1.0).contains(&self.leace_shrinkage) {
            return bad("--leace-shrinkage doit être dans [0, 1] (0 : pseudo-inverse)");
        }
        if !(self.leace_rcond > 0.0 && self.leace_rcond < 1.0) {
            return bad("--leace-rcond doit être dans ]0, 1[");
        }
        if !(0.0..1.0).contains(&self.holdout_frac) {
            return bad("--holdout-frac doit être dans [0, 1[");
        }
        Ok(())
    }

    /// The four options as the `params` object of the extension API (`method` is chosen
    /// by identifier, apart).
    pub fn params(&self) -> Value {
        json!({
            "svd_center": self.svd_center,
            "leace_shrinkage": self.leace_shrinkage,
            "leace_rcond": self.leace_rcond,
            "holdout_frac": self.holdout_frac,
        })
    }

    /// Reverse of [`MethodOptions::params`]: `null` or a missing key keeps the default,
    /// an unknown key or a wrong type is refused.
    pub fn from_params(method: Method, params: &Value) -> Result<Self> {
        #[derive(Deserialize)]
        #[serde(default, deny_unknown_fields)]
        struct Params {
            svd_center: bool,
            leace_shrinkage: f64,
            leace_rcond: f64,
            holdout_frac: f64,
            // Read by `SubspaceOptions::from_params`: accepted here, not kept.
            rank: Option<Value>,
            labels: Option<Value>,
        }
        impl Default for Params {
            fn default() -> Self {
                let d = MethodOptions::default();
                Self {
                    svd_center: d.svd_center,
                    leace_shrinkage: d.leace_shrinkage,
                    leace_rcond: d.leace_rcond,
                    holdout_frac: d.holdout_frac,
                    rank: None,
                    labels: None,
                }
            }
        }
        let p = match params {
            Value::Null => Params::default(),
            other => Params::deserialize(other)
                .map_err(|e| ExtractError::InvalidArgument(format!("paramètres : {e}")))?,
        };
        Ok(Self {
            method,
            svd_center: p.svd_center,
            leace_shrinkage: p.leace_shrinkage,
            leace_rcond: p.leace_rcond,
            holdout_frac: p.holdout_frac,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum MethodDiagnostics {
    Svd(SvdDiagnostics),
    Leace(LeaceDiagnostics),
}

/// What a method produced: the unit direction (f64) and its extras.
#[derive(Debug, Clone, PartialEq)]
pub struct MethodOutput {
    pub direction: Vec<f64>,
    pub diagnostics: Option<MethodDiagnostics>,
    pub leace: Option<LeaceFit>,
    /// The rank-k eraser, when one was asked for (`direction` is its first direction).
    pub subspace: Option<SubspaceFit>,
    pub warnings: Vec<String>,
}

/// `k` write and read directions, one per row: `a` and `b` of `project_out`.
#[derive(Debug, Clone, PartialEq)]
pub struct SubspaceFit {
    pub a: Vec<Vec<f64>>,
    pub b: Vec<Vec<f64>>,
    /// `leace` with labels: its mean, whitened directions and covariance diagnostics.
    pub leace: Option<LeaceSubspace>,
}

/// `opts.params()` plus the rank-k keys, as a front-end passes them to a method. Without
/// `rank` nor `labels` this is `opts.params()`, unchanged.
pub fn params_with_subspace(
    opts: &MethodOptions,
    rank: Option<usize>,
    labels: Option<Value>,
) -> Value {
    let mut params = opts.params();
    if let Some(rank) = rank {
        params["rank"] = json!(rank);
    }
    if let Some(labels) = labels {
        params["labels"] = labels;
    }
    params
}

pub fn holdout_mask(n: usize, frac: f64) -> Vec<bool> {
    (0..n)
        .map(|i| ((i + 1) as f64 * frac).floor() > (i as f64 * frac).floor())
        .collect()
}

fn split(rows: &[Vec<f64>], mask: &[bool]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let (mut train, mut held) = (Vec::new(), Vec::new());
    for (r, &h) in rows.iter().zip(mask) {
        match h {
            true => held.push(r.clone()),
            false => train.push(r.clone()),
        }
    }
    (train, held)
}

fn unit(v: Vec<f64>) -> Result<Vec<f64>> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    match norm > super::MIN_DIFF_NORM && norm.is_finite() {
        true => Ok(v.into_iter().map(|x| x / norm).collect()),
        false => Err(ExtractError::DegenerateDirection { norm }),
    }
}

/// The direction of `opts.method` for `svd` and `leace` (`mean_diff` stays in
/// `stats::separation_with_projections`, unchanged).
pub fn compute(pos: &[Vec<f64>], neg: &[Vec<f64>], opts: &MethodOptions) -> Result<MethodOutput> {
    opts.validate()?;
    let mut warnings = Vec::new();
    let holdout = match opts.holdout_frac > 0.0 {
        false => None,
        true if pos.len() < MIN_HOLDOUT_PER_CLASS || neg.len() < MIN_HOLDOUT_PER_CLASS => {
            warnings.push(format!(
                "--holdout-frac ignoré : au moins {MIN_HOLDOUT_PER_CLASS} prompts par classe requis"
            ));
            None
        }
        true => Some((
            split(pos, &holdout_mask(pos.len(), opts.holdout_frac)),
            split(neg, &holdout_mask(neg.len(), opts.holdout_frac)),
        )),
    };
    match opts.method {
        Method::MeanDiff => Err(ExtractError::InvalidArgument(
            "mean_diff passe par le chemin de prompt 8".into(),
        )),
        Method::Svd => {
            let (v, diagnostics, more) = svd_direction(pos, neg, opts.svd_center)?;
            warnings.extend(more);
            Ok(MethodOutput {
                direction: unit(v)?,
                diagnostics: Some(MethodDiagnostics::Svd(diagnostics)),
                leace: None,
                subspace: None,
                warnings,
            })
        }
        Method::Leace => {
            let fit = leace::fit(pos, neg, opts.leace_shrinkage, opts.leace_rcond)?;
            let (before, after, gap) = leace::erased_stats(&fit, pos, neg);
            let holdout_cross_cov_ratio = match &holdout {
                None => None,
                Some(((pos_train, pos_held), (neg_train, neg_held))) => {
                    let train =
                        leace::fit(pos_train, neg_train, opts.leace_shrinkage, opts.leace_rcond)?;
                    let (hb, ha, _) = leace::erased_stats(&train, pos_held, neg_held);
                    (hb > 0.0).then_some(ha / hb)
                }
            };
            let d = fit.mean.len();
            if pos.len() + neg.len() <= d {
                warnings.push(format!(
                    "n = {} ≤ d = {d} : la covariance croisée s'annule en échantillon, mais un \
                     classifieur linéaire peut encore séparer ; seul holdout_cross_cov_ratio \
                     dit si l'effacement généralise",
                    pos.len() + neg.len()
                ));
            }
            let diagnostics = LeaceDiagnostics {
                shrinkage: fit.shrinkage,
                rcond: opts.leace_rcond,
                rank: fit.rank,
                kappa: fit.kappa,
                b_dot_a: fit.b_dot_a,
                cross_cov_before: before,
                cross_cov_after: after,
                class_mean_gap_after: gap,
                in_sample_cross_cov_ratio: match before > 0.0 {
                    true => after / before,
                    false => 0.0,
                },
                holdout_cross_cov_ratio,
                n_pos: pos.len(),
                n_neg: neg.len(),
                subspace: None,
            };
            Ok(MethodOutput {
                direction: unit(fit.a.clone())?,
                diagnostics: Some(MethodDiagnostics::Leace(diagnostics)),
                leace: Some(fit),
                subspace: None,
                warnings,
            })
        }
    }
}

/// What a method cannot do with these options, whatever the activations: checked before
/// the forward pass.
pub fn check_subspace(method: Method, sub: &SubspaceOptions) -> Result<()> {
    let bad = |m: &str| Err(ExtractError::InvalidArgument(m.to_owned()));
    let rank = sub.rank.unwrap_or(1);
    match (method, sub.labels.is_some()) {
        (Method::MeanDiff, _) if !sub.is_default() => {
            bad("mean_diff reste de rang 1 : --rank et --labels-file demandent svd ou leace")
        }
        (Method::Svd, true) => bad("--labels-file ne s'applique qu'à --method leace"),
        (Method::Leace, false) if rank > 1 => bad(
            "leace sans --labels-file efface un concept binaire, une seule direction : \
             --rank > 1 demande --labels-file",
        ),
        _ => Ok(()),
    }
}

/// [`compute`] with the rank-k options: `svd` keeps its first `rank` singular vectors,
/// `leace` with labels erases every label at once. Without them, [`compute`] unchanged.
pub fn compute_subspace(
    pos: &[Vec<f64>],
    neg: &[Vec<f64>],
    opts: &MethodOptions,
    sub: &SubspaceOptions,
) -> Result<MethodOutput> {
    check_subspace(opts.method, sub)?;
    match (&sub.labels, sub.rank.unwrap_or(1)) {
        (Some(labels), _) => leace_with_labels(pos, neg, opts, labels, sub.rank),
        (None, 1) => compute(pos, neg, opts),
        (None, k) => {
            let mut out = compute(pos, neg, opts)?;
            let (v, sigma) = svd_subspace(pos, neg, opts.svd_center, k)?;
            if v.len() < k {
                out.warnings.push(format!(
                    "rang demandé {k}, rang numérique des différences {} : {} direction(s) écrite(s)",
                    v.len(),
                    v.len()
                ));
            }
            out.warnings
                .extend(subspace::rank_warning(v.len(), pos.len(), "paires"));
            if let Some(MethodDiagnostics::Svd(diagnostics)) = &mut out.diagnostics {
                diagnostics.subspace = Some(subspace::diagnostics(&v, &v, &sigma, pos, neg, None));
            }
            out.subspace = Some(SubspaceFit {
                a: v.clone(),
                b: v,
                leace: None,
            });
            Ok(out)
        }
    }
}

/// Mean of `rows` (non-empty).
fn mean_of(rows: &[Vec<f64>]) -> Vec<f64> {
    let mut mean = vec![0.0; rows.first().map_or(0, Vec::len)];
    for row in rows {
        mean.iter_mut().zip(row).for_each(|(m, v)| *m += v);
    }
    mean.iter_mut().for_each(|m| *m /= rows.len().max(1) as f64);
    mean
}

fn leace_with_labels(
    pos: &[Vec<f64>],
    neg: &[Vec<f64>],
    opts: &MethodOptions,
    labels: &[Vec<f64>],
    max_rank: Option<usize>,
) -> Result<MethodOutput> {
    opts.validate()?;
    let n = pos.len() + neg.len();
    if labels.len() != n {
        return Err(ExtractError::InvalidArgument(format!(
            "labels : {} lignes pour {n} prompts (les positifs puis les négatifs, dans l'ordre \
             des fichiers)",
            labels.len()
        )));
    }
    fn refs(rows: &[Vec<f64>]) -> Vec<&[f64]> {
        rows.iter().map(Vec::as_slice).collect()
    }
    let all: Vec<Vec<f64>> = pos.iter().chain(neg).cloned().collect();
    let (x, z) = (refs(&all), refs(labels));
    let (shrinkage, rcond) = (opts.leace_shrinkage, opts.leace_rcond);
    let fit = leace::fit_multi(&x, &z, shrinkage, rcond, max_rank)?;

    let erased: Vec<Vec<f64>> = all.iter().map(|row| fit.erase(row)).collect();
    let before = leace::frobenius(&leace::cross_cov(&x, &z));
    let after = leace::frobenius(&leace::cross_cov(&refs(&erased), &z));
    let gap: Vec<f64> = mean_of(&erased[..pos.len()])
        .iter()
        .zip(mean_of(&erased[pos.len()..]))
        .map(|(p, q)| p - q)
        .collect();
    let mut warnings = Vec::new();

    // Hold-out: the eraser refitted without the held rows, measured on them.
    let mut mask = holdout_mask(pos.len(), opts.holdout_frac);
    mask.extend(holdout_mask(neg.len(), opts.holdout_frac));
    let held_out = match opts.holdout_frac > 0.0 {
        false => None,
        true if pos.len() < MIN_HOLDOUT_PER_CLASS || neg.len() < MIN_HOLDOUT_PER_CLASS => {
            warnings.push(format!(
                "--holdout-frac ignoré : au moins {MIN_HOLDOUT_PER_CLASS} prompts par classe requis"
            ));
            None
        }
        true => {
            let pick = |rows: &[&[f64]], held: bool| -> Vec<Vec<f64>> {
                let kept = rows.iter().zip(&mask).filter(|(_, h)| **h == held);
                kept.map(|(row, _)| row.to_vec()).collect()
            };
            let (x_train, z_train) = (pick(&x, false), pick(&z, false));
            let (x_held, z_held) = (pick(&x, true), pick(&z, true));
            let train =
                leace::fit_multi(&refs(&x_train), &refs(&z_train), shrinkage, rcond, max_rank)?;
            let held_erased: Vec<Vec<f64>> = x_held.iter().map(|row| train.erase(row)).collect();
            let cov_before = leace::cross_cov(&refs(&x_held), &refs(&z_held));
            let cov_after = leace::cross_cov(&refs(&held_erased), &refs(&z_held));
            // Per component: what b_t still reads of the held-out cross covariance.
            let read = |b: &[f64], cov: &[Vec<f64>]| -> f64 {
                let along = cov.iter().map(|c| super::svd::dot(b, c).powi(2));
                along.sum::<f64>().sqrt()
            };
            let residual: Vec<f64> = train
                .b
                .iter()
                .map(|b| match read(b, &cov_before) {
                    base if base > 0.0 => read(b, &cov_after) / base,
                    _ => 0.0,
                })
                .collect();
            let total = leace::frobenius(&cov_before);
            let ratio = (total > 0.0).then(|| leace::frobenius(&cov_after) / total);
            Some((ratio, residual))
        }
    };

    let d = fit.mean.len();
    if n <= d {
        warnings.push(format!(
            "n = {n} ≤ d = {d} : la covariance croisée s'annule en échantillon, mais un \
             classifieur linéaire peut encore séparer ; seul holdout_cross_cov_ratio \
             dit si l'effacement généralise"
        ));
    }
    let rank = fit.a.len();
    if let Some(asked) = max_rank.filter(|k| *k > rank) {
        warnings.push(format!(
            "rang demandé {asked}, rang de la covariance croisée {rank} : {rank} direction(s) écrite(s)"
        ));
    }
    warnings.extend(subspace::rank_warning(rank, n, "prompts"));
    let trace: f64 = fit
        .b
        .iter()
        .zip(&fit.a)
        .map(|(b, a)| super::svd::dot(b, a))
        .sum();
    let (holdout_ratio, residual) = match held_out {
        Some((ratio, residual)) => (ratio, Some(residual)),
        None => (None, None),
    };
    let diagnostics = LeaceDiagnostics {
        shrinkage: fit.shrinkage,
        rcond,
        rank: fit.cov_rank,
        kappa: fit.kappa,
        // Mean of the diagonal of BᵀA: 1 by construction, like bᵀa of one direction.
        b_dot_a: trace / rank as f64,
        cross_cov_before: before,
        cross_cov_after: after,
        class_mean_gap_after: super::svd::dot(&gap, &gap).sqrt(),
        in_sample_cross_cov_ratio: match before > 0.0 {
            true => after / before,
            false => 0.0,
        },
        holdout_cross_cov_ratio: holdout_ratio,
        n_pos: pos.len(),
        n_neg: neg.len(),
        subspace: Some(subspace::diagnostics(
            &fit.a, &fit.b, &fit.sigma, pos, neg, residual,
        )),
    };
    Ok(MethodOutput {
        direction: unit(fit.a[0].clone())?,
        diagnostics: Some(MethodDiagnostics::Leace(diagnostics)),
        leace: None,
        subspace: Some(SubspaceFit {
            a: fit.a.clone(),
            b: fit.b.clone(),
            leace: Some(fit),
        }),
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holdout_mask_is_even_and_deterministic() {
        let m = holdout_mask(10, 0.5);
        assert_eq!(m.iter().filter(|h| **h).count(), 5);
        assert_eq!(
            m,
            [false, true, false, true, false, true, false, true, false, true]
        );
        assert_eq!(holdout_mask(10, 0.2).iter().filter(|h| **h).count(), 2);
        assert!(holdout_mask(7, 0.0).iter().all(|h| !h));
    }

    #[test]
    fn options_are_validated() {
        let ok = MethodOptions::default();
        assert!(ok.validate().is_ok());
        for bad in [
            MethodOptions {
                leace_shrinkage: -0.1,
                ..ok
            },
            MethodOptions {
                leace_shrinkage: 1.5,
                ..ok
            },
            MethodOptions {
                leace_rcond: 0.0,
                ..ok
            },
            MethodOptions {
                holdout_frac: 1.0,
                ..ok
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?}");
        }
    }

    #[test]
    fn params_round_trip_and_refuse_unknown_keys() {
        let opts = MethodOptions {
            method: Method::Leace,
            svd_center: true,
            leace_shrinkage: 0.3,
            leace_rcond: 1e-8,
            holdout_frac: 0.25,
        };
        assert_eq!(
            MethodOptions::from_params(Method::Leace, &opts.params()).unwrap(),
            opts
        );
        let svd = MethodOptions::from_params(Method::Svd, &Value::Null).unwrap();
        assert_eq!(
            svd,
            MethodOptions {
                method: Method::Svd,
                ..MethodOptions::default()
            }
        );
        for bad in [
            json!({"svd_centre": true}),
            json!({"svd_center": 1}),
            json!(3),
        ] {
            assert!(
                MethodOptions::from_params(Method::Svd, &bad).is_err(),
                "{bad}"
            );
        }
        for m in [Method::MeanDiff, Method::Svd, Method::Leace] {
            assert_eq!(Method::from_id(m.as_str()), Some(m));
        }
        assert_eq!(Method::from_id("pca"), None);
    }

    #[test]
    fn small_classes_disable_the_holdout_with_a_warning() {
        let pos: Vec<Vec<f64>> = (0..4)
            .map(|i| vec![1.0 + i as f64 * 0.1, 0.5, 0.0])
            .collect();
        let neg: Vec<Vec<f64>> = (0..4)
            .map(|i| vec![0.0, 0.5 - i as f64 * 0.1, 1.0])
            .collect();
        let opts = MethodOptions {
            method: Method::Leace,
            holdout_frac: 0.5,
            ..MethodOptions::default()
        };
        let out = compute(&pos, &neg, &opts).unwrap();
        assert!(out
            .warnings
            .iter()
            .any(|w| w.contains("--holdout-frac ignoré")));
        let Some(MethodDiagnostics::Leace(d)) = out.diagnostics else {
            panic!()
        };
        assert_eq!(d.holdout_cross_cov_ratio, None);
    }

    // ------------------------------------------------------------------ rank k

    /// `n` pairs of dimension 12: three planted axes of decreasing strength plus noise.
    fn planted(n: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut rng = crate::linalg::Lcg(17);
        let (mut pos, mut neg) = (Vec::new(), Vec::new());
        for i in 0..n {
            let base: Vec<f64> = (0..12).map(|_| rng.next()).collect();
            let mut with = base.clone();
            for (t, strength) in [4.0, 2.0, 1.0].iter().enumerate() {
                let sign = if (i >> t) & 1 == 0 { 1.0 } else { -0.5 };
                with[3 * t] += strength * sign + 0.01 * rng.next();
            }
            pos.push(with);
            neg.push(base);
        }
        (pos, neg)
    }

    fn sub(rank: Option<usize>, labels: Option<Vec<Vec<f64>>>) -> SubspaceOptions {
        SubspaceOptions { rank, labels }
    }

    #[test]
    fn svd_rank_k_adds_the_subspace_and_keeps_the_rank_one_output() {
        let (pos, neg) = planted(16);
        let opts = MethodOptions {
            method: Method::Svd,
            ..MethodOptions::default()
        };
        let one = compute(&pos, &neg, &opts).unwrap();
        // No option, or rank 1: `compute`, unchanged.
        assert_eq!(
            compute_subspace(&pos, &neg, &opts, &sub(None, None)).unwrap(),
            one
        );
        assert_eq!(
            compute_subspace(&pos, &neg, &opts, &sub(Some(1), None)).unwrap(),
            one
        );

        let three = compute_subspace(&pos, &neg, &opts, &sub(Some(3), None)).unwrap();
        assert_eq!(three.direction, one.direction, "the first direction");
        let fit = three.subspace.as_ref().unwrap();
        assert_eq!(
            (fit.a.len(), fit.a == fit.b, fit.leace.is_none()),
            (3, true, true)
        );
        let Some(MethodDiagnostics::Svd(d)) = &three.diagnostics else {
            panic!("svd diagnostics expected");
        };
        let s = d.subspace.as_ref().unwrap();
        assert_eq!((s.rank, s.sigma_t.len(), s.d_prime.len()), (3, 3, 3));
        assert!(s.gram_bta_error < 1e-6 && s.holdout_cross_cov_residual.is_none());
        assert!(s.sigma_t.windows(2).all(|w| w[0] >= w[1]));
        let cumulative = &s.explained_variance_cumulative;
        assert!(cumulative.windows(2).all(|w| w[0] < w[1]) && cumulative[2] <= 1.0);
        assert!(cumulative[2] > 0.9, "three planted axes carry the energy");
        // The rank-1 diagnostics do not move, and say nothing of a subspace.
        let Some(MethodDiagnostics::Svd(alone)) = &one.diagnostics else {
            panic!("svd diagnostics expected");
        };
        assert!(alone.subspace.is_none() && alone.sigma == d.sigma);
        assert!(serde_json::to_value(alone)
            .unwrap()
            .get("subspace")
            .is_none());

        // 4 k ≥ n: said. More directions than the deltas have: said too.
        let few = compute_subspace(&pos[..8], &neg[..8], &opts, &sub(Some(3), None)).unwrap();
        assert!(
            few.warnings.iter().any(|w| w.contains("n/4")),
            "{:?}",
            few.warnings
        );
        assert!(!three.warnings.iter().any(|w| w.contains("n/4")));
        let all = compute_subspace(&pos[..4], &neg[..4], &opts, &sub(Some(9), None)).unwrap();
        assert_eq!(
            all.subspace.unwrap().a.len(),
            3,
            "the deltas live on three axes"
        );
        assert!(all.warnings.iter().any(|w| w.contains("rang demandé 9")));
    }

    #[test]
    fn leace_with_labels_erases_every_label_and_reports_each_component() {
        let (pos, neg) = planted(20);
        let n = pos.len() + neg.len();
        // Two continuous labels: the planted signs of the first two axes.
        let labels: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let positive = i < pos.len();
                let pair = i % pos.len();
                (0..2)
                    .map(|t| match (positive, (pair >> t) & 1 == 0) {
                        (false, _) => 0.0,
                        (true, true) => 1.0,
                        (true, false) => -0.5,
                    })
                    .collect()
            })
            .collect();
        let opts = MethodOptions {
            method: Method::Leace,
            holdout_frac: 0.5,
            ..MethodOptions::default()
        };
        let out = compute_subspace(&pos, &neg, &opts, &sub(None, Some(labels.clone()))).unwrap();
        let fit = out.subspace.as_ref().unwrap();
        assert_eq!((fit.a.len(), fit.b.len()), (2, 2));
        assert!(fit.leace.is_some() && out.leace.is_none());
        let norm: f64 = out.direction.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-12);
        let Some(MethodDiagnostics::Leace(d)) = &out.diagnostics else {
            panic!("leace diagnostics expected");
        };
        assert!(d.cross_cov_before > 0.05);
        assert!(
            d.in_sample_cross_cov_ratio < 1e-9,
            "{}",
            d.in_sample_cross_cov_ratio
        );
        assert!((d.b_dot_a - 1.0).abs() < 1e-9);
        let s = d.subspace.as_ref().unwrap();
        assert_eq!(s.rank, 2);
        assert!(s.gram_bta_error < 1e-5, "{}", s.gram_bta_error);
        let residual = s.holdout_cross_cov_residual.as_ref().unwrap();
        assert_eq!(residual.len(), 2);
        assert!(residual.iter().all(|r| r.is_finite() && *r >= 0.0));
        assert!(d.holdout_cross_cov_ratio.is_some());
        // Deterministic.
        let again = compute_subspace(&pos, &neg, &opts, &sub(None, Some(labels.clone()))).unwrap();
        assert_eq!(out, again);

        // One direction at most on request; a wrong label count is refused.
        let one = compute_subspace(&pos, &neg, &opts, &sub(Some(1), Some(labels.clone()))).unwrap();
        assert_eq!(one.subspace.unwrap().a.len(), 1);
        let err = compute_subspace(&pos, &neg, &opts, &sub(None, Some(labels[..5].to_vec())))
            .unwrap_err();
        assert!(
            err.to_string().contains("5 lignes pour 40 prompts"),
            "{err}"
        );
    }

    #[test]
    fn options_a_method_cannot_honor_are_refused_before_any_computation() {
        let labels = Some(vec![vec![1.0], vec![0.0]]);
        for (method, options, why) in [
            (
                Method::MeanDiff,
                sub(Some(2), None),
                "mean_diff reste de rang 1",
            ),
            (
                Method::MeanDiff,
                sub(None, labels.clone()),
                "mean_diff reste de rang 1",
            ),
            (Method::Svd, sub(Some(2), labels.clone()), "--method leace"),
            (Method::Leace, sub(Some(2), None), "--labels-file"),
        ] {
            let err = check_subspace(method, &options).unwrap_err().to_string();
            assert!(err.contains(why), "{method:?}: {err}");
        }
        for (method, options) in [
            (Method::MeanDiff, sub(Some(1), None)),
            (Method::Svd, sub(Some(64), None)),
            (Method::Leace, sub(Some(1), None)),
            (Method::Leace, sub(Some(3), labels.clone())),
            (Method::Leace, sub(None, labels)),
        ] {
            assert!(check_subspace(method, &options).is_ok(), "{method:?}");
        }
    }

    #[test]
    fn the_rank_k_keys_travel_in_the_params_and_leave_them_unchanged_when_absent() {
        let opts = MethodOptions {
            method: Method::Svd,
            svd_center: true,
            ..MethodOptions::default()
        };
        assert_eq!(params_with_subspace(&opts, None, None), opts.params());
        let params = params_with_subspace(&opts, Some(4), Some(json!([0, 1])));
        assert_eq!(
            (&params["rank"], &params["labels"]),
            (&json!(4), &json!([0, 1]))
        );
        // The frozen options still read their four keys from it.
        assert_eq!(
            MethodOptions::from_params(Method::Svd, &params).unwrap(),
            opts
        );
        let read = SubspaceOptions::from_params(&params).unwrap();
        assert_eq!(read.rank, Some(4));
        assert_eq!(read.labels.unwrap(), [[1.0, 0.0], [0.0, 1.0]]);
    }
}
