//! `--method mean_diff | svd | leace`: which direction is written from the pooled
//! activations. `mean_diff` keeps the prompt 8 path and outputs byte for byte.
//!
//! Hold-out (`--holdout-frac f`): sample `i` of each class is held out when
//! `⌊(i+1)·f⌋ > ⌊i·f⌋` (evenly spread and deterministic; `f = 0.5` holds out the odd
//! indices). The written direction always uses every prompt; the hold-out diagnostic refits
//! on the remaining samples and measures on the held-out ones. Fewer than 10 samples in a
//! class disables it with a warning.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::error::{ExtractError, Result};
use super::leace::{self, LeaceDiagnostics, LeaceFit};
use super::svd::{svd_direction, SvdDiagnostics};

pub const MIN_HOLDOUT_PER_CLASS: usize = 10;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema, clap::ValueEnum,
)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    /// normalize(μ₊ − μ₋).
    #[default]
    #[value(name = "mean_diff")]
    MeanDiff,
    /// Premier vecteur singulier droit des différences appariées.
    Svd,
    /// Effaceur affine LEACE ; direction = a/‖a‖ (parallèle à μ₊ − μ₋).
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
    pub warnings: Vec<String>,
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
            };
            Ok(MethodOutput {
                direction: unit(fit.a.clone())?,
                diagnostics: Some(MethodDiagnostics::Leace(diagnostics)),
                leace: Some(fit),
                warnings,
            })
        }
    }
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
}
