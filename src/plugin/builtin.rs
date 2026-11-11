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
    check_subspace, compute_method, mean_diff_direction, Method, MethodOptions, SubspaceFit,
    SubspaceOptions, DEFAULT_LEACE_RCOND, DEFAULT_LEACE_SHRINKAGE,
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

    /// `rank` and `labels`, refused when the method cannot honor them.
    fn subspace(self, params: &Value) -> Result<SubspaceOptions, PluginError> {
        let sub = SubspaceOptions::from_params(params)?;
        check_subspace(self.0, &sub)?;
        Ok(sub)
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
                params: vec![
                    param(
                        "svd_center",
                        ParamKind::Boolean,
                        json!(false),
                        "centre les différences appariées (paires d'ordre arbitraire)",
                    ),
                    param(
                        "rank",
                        ParamKind::Integer,
                        json!(1),
                        "nombre de vecteurs singuliers gardés ; au-delà de 1, écrit a = b = V_k dans <out>.subspace.safetensors",
                    ),
                ],
                requires_paired: true,
                sidecars: vec!["subspace"],
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
                    param(
                        "labels",
                        ParamKind::Array,
                        Value::Null,
                        "un label par prompt, positifs puis négatifs : indice de classe entier ou tableau de m nombres ; absent : concept binaire positif / négatif",
                    ),
                    param(
                        "rank",
                        ParamKind::Integer,
                        Value::Null,
                        "avec labels : nombre maximal de directions ; absent : toutes celles de la covariance croisée",
                    ),
                ],
                requires_paired: false,
                sidecars: vec!["leace"],
            },
        }
    }

    fn validate(&self, params: &Value) -> Result<(), PluginError> {
        self.subspace(params)?;
        Ok(self.options(params)?.validate()?)
    }

    fn extract(
        &self,
        ctx: &ExtractContext,
        acts: &ActivationSet,
        _cancel: &AtomicBool,
    ) -> Result<ExtractionOutput, PluginError> {
        let sub = self.subspace(ctx.params)?;
        if self.0 == Method::MeanDiff {
            let direction = mean_diff_direction(acts.pos, acts.neg)?;
            return Ok(ExtractionOutput::new(direction));
        }
        let opts = self.options(ctx.params)?;
        let out = compute_method(acts.pos, acts.neg, &opts, &sub)?;
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
        let mut sidecar_rows = BTreeMap::new();
        let subspace = out.subspace.as_ref().map(|fit| {
            let (sidecar, rows) = subspace_sidecar(fit);
            sidecar_rows = rows;
            sidecar
        });
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
            .chain(subspace)
            .collect();
        Ok(ExtractionOutput {
            direction: out.direction,
            diagnostics: out.diagnostics.map(Diagnostics::Builtin),
            metadata,
            sidecars,
            sidecar_rows,
            warnings: out.warnings,
        })
    }
}

/// The file of a rank-k eraser and the rows of its 2D tensors: `<out>.leace.safetensors`
/// for `leace` with labels (`a`, `b`, `mean`, `u_whitened`), `<out>.subspace.safetensors`
/// for `svd` (`a = b = V_k`). One direction keeps the historical 1D tensors.
fn subspace_sidecar(fit: &SubspaceFit) -> (Sidecar, BTreeMap<(String, String), usize>) {
    let flat =
        |rows: &[Vec<f64>]| -> Vec<f32> { rows.iter().flatten().map(|&x| x as f32).collect() };
    let k = fit.a.len();
    let suffix = match fit.leace {
        Some(_) => "leace",
        None => "subspace",
    };
    let mut tensors = vec![
        ("a".to_owned(), flat(&fit.a)),
        ("b".to_owned(), flat(&fit.b)),
    ];
    let mut ranked = vec!["a", "b"];
    let mut metadata = BTreeMap::from([("subspace_rank".to_owned(), k.to_string())]);
    if let Some(leace) = &fit.leace {
        tensors.push(("mean".to_owned(), flat(std::slice::from_ref(&leace.mean))));
        tensors.push(("u_whitened".to_owned(), flat(&leace.u_whitened)));
        ranked.push("u_whitened");
        metadata.insert("rank".to_owned(), leace.cov_rank.to_string());
        metadata.insert("kappa".to_owned(), leace.kappa.to_string());
    }
    let rows = match k {
        1 => BTreeMap::new(),
        _ => ranked
            .into_iter()
            .map(|name| ((suffix.to_owned(), name.to_owned()), k))
            .collect(),
    };
    let sidecar = Sidecar {
        suffix,
        tensors,
        metadata,
    };
    (sidecar, rows)
}
