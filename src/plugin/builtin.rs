//! `mean_diff`, `svd` and `leace` behind the extension API. They call the functions they
//! always called, in the same order, so their files stay byte for byte what they were.

use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;

use serde_json::{json, Value};

use super::{
    ActivationSet, Diagnostics, ExtractContext, ExtractionMethod, ExtractionOutput, MethodInfo,
    ParamInfo, ParamKind, PluginError, Sidecar,
};
use crate::extract::{
    compute_method, mean_diff_direction, Method, MethodOptions, DEFAULT_LEACE_RCOND,
    DEFAULT_LEACE_SHRINKAGE,
};

/// One built-in method. They share the four options of [`MethodOptions`], all checked
/// whatever the method, as the CLI always did.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Builtin(pub(crate) Method);

impl Builtin {
    pub(super) fn all() -> [Self; 3] {
        [Method::MeanDiff, Method::Svd, Method::Leace].map(Self)
    }

    fn options(self, params: &Value) -> Result<MethodOptions, PluginError> {
        Ok(MethodOptions::from_params(self.0, params)?)
    }
}

fn param(name: &'static str, kind: ParamKind, default: Value, summary: &'static str) -> ParamInfo {
    ParamInfo {
        name,
        kind,
        default,
        summary,
    }
}

impl ExtractionMethod for Builtin {
    fn id(&self) -> &'static str {
        self.0.as_str()
    }

    fn describe(&self) -> MethodInfo {
        let holdout = param(
            "holdout_frac",
            ParamKind::Number,
            json!(0.0),
            "fraction hors-échantillon (par index) des diagnostics, 10 prompts par classe au moins",
        );
        match self.0 {
            Method::MeanDiff => MethodInfo {
                title: "Différence des moyennes",
                summary: "normalize(μ₊ − μ₋).",
                params: vec![],
                requires_paired: false,
                sidecars: vec![],
            },
            Method::Svd => MethodInfo {
                title: "SVD des différences appariées",
                summary: "Premier vecteur singulier droit des différences appariées.",
                params: vec![param(
                    "svd_center",
                    ParamKind::Boolean,
                    json!(false),
                    "centre les différences appariées (paires d'ordre arbitraire)",
                )],
                requires_paired: true,
                sidecars: vec![],
            },
            Method::Leace => MethodInfo {
                title: "Effaceur LEACE",
                summary: "Effaceur affine LEACE ; direction = a/‖a‖ (parallèle à μ₊ − μ₋).",
                params: vec![
                    param(
                        "leace_shrinkage",
                        ParamKind::Number,
                        json!(DEFAULT_LEACE_SHRINKAGE),
                        "rétrécissement λ ∈ [0, 1] de Σ vers tr(Σ)/d·I (0 : pseudo-inverse)",
                    ),
                    param(
                        "leace_rcond",
                        ParamKind::Number,
                        json!(DEFAULT_LEACE_RCOND),
                        "avec λ = 0 : seuil relatif des valeurs propres conservées",
                    ),
                    holdout,
                ],
                requires_paired: false,
                sidecars: vec!["leace"],
            },
        }
    }

    fn validate(&self, params: &Value) -> Result<(), PluginError> {
        Ok(self.options(params)?.validate()?)
    }

    fn extract(
        &self,
        ctx: &ExtractContext,
        acts: &ActivationSet,
        _cancel: &AtomicBool,
    ) -> Result<ExtractionOutput, PluginError> {
        if self.0 == Method::MeanDiff {
            let direction = mean_diff_direction(acts.pos, acts.neg)?;
            return Ok(ExtractionOutput::new(direction));
        }
        let opts = self.options(ctx.params)?;
        let out = compute_method(acts.pos, acts.neg, &opts)?;
        let mut metadata =
            BTreeMap::from([("holdout_frac".to_owned(), opts.holdout_frac.to_string())]);
        match self.0 {
            Method::Svd => {
                metadata.insert("svd_center".into(), opts.svd_center.to_string());
            }
            _ => {
                metadata.insert("shrinkage".into(), opts.leace_shrinkage.to_string());
                metadata.insert("rcond".into(), opts.leace_rcond.to_string());
            }
        }
        let as_f32 = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        let sidecars: Vec<Sidecar> = out
            .leace
            .iter()
            .map(|fit| Sidecar {
                suffix: "leace",
                tensors: [
                    ("a", &fit.a),
                    ("b", &fit.b),
                    ("mean", &fit.mean),
                    ("u_whitened", &fit.u_whitened),
                ]
                .map(|(name, v)| (name.to_owned(), as_f32(v)))
                .to_vec(),
                metadata: BTreeMap::from([
                    ("rank".to_owned(), fit.rank.to_string()),
                    ("kappa".to_owned(), fit.kappa.to_string()),
                    ("bTa".to_owned(), fit.b_dot_a.to_string()),
                ]),
            })
            .collect();
        Ok(ExtractionOutput {
            direction: out.direction,
            diagnostics: out.diagnostics.map(Diagnostics::Builtin),
            metadata,
            sidecars,
            warnings: out.warnings,
        })
    }
}
