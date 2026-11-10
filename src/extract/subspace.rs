//! Rank-k outputs of `svd` and `leace`: the options read from the method parameters
//! (`rank`, `labels`) and the per-component diagnostics written next to the directions.
//!
//! `MethodOptions` is built field by field by its callers, so these options travel beside
//! it, in the same `params` object: `rank` (an integer) and `labels` (one entry per prompt,
//! positives first; an integer class index, or an array of `m` numbers).

use schemars::JsonSchema;
use serde::Serialize;
use serde_json::Value;

use super::error::{ExtractError, Result};
use super::stats::separation_along;
use crate::engine::{SubspaceProjection, DEFAULT_MAX_RANK};

/// Directions and labels asked of a method.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SubspaceOptions {
    /// Directions to write; `None`: one for `svd`, every label direction for `leace`.
    pub rank: Option<usize>,
    /// `leace`: one row of `m` values per prompt, positives first.
    pub labels: Option<Vec<Vec<f64>>>,
}

fn invalid(message: String) -> ExtractError {
    ExtractError::InvalidArgument(message)
}

impl SubspaceOptions {
    /// Reads `rank` and `labels` of a `params` object; absent or `null` keys keep the
    /// defaults, anything else must be well formed.
    pub fn from_params(params: &Value) -> Result<Self> {
        let rank = match params.get("rank") {
            None | Some(Value::Null) => None,
            Some(value) => {
                let rank = value
                    .as_u64()
                    .and_then(|r| usize::try_from(r).ok())
                    .filter(|r| (1..=DEFAULT_MAX_RANK).contains(r));
                Some(rank.ok_or_else(|| {
                    invalid(format!(
                        "rank doit être un entier dans [1, {DEFAULT_MAX_RANK}], reçu {value}"
                    ))
                })?)
            }
        };
        let labels = match params.get("labels") {
            None | Some(Value::Null) => None,
            Some(value) => Some(parse_labels(value)?),
        };
        Ok(Self { rank, labels })
    }

    /// More than the historical single direction of a binary concept.
    pub fn is_default(&self) -> bool {
        self.rank.is_none_or(|r| r == 1) && self.labels.is_none()
    }
}

/// One row of `m` finite numbers per entry. Integer entries are class indices, expanded
/// to one-hot rows over the classes present (in increasing order).
pub fn parse_labels(value: &Value) -> Result<Vec<Vec<f64>>> {
    let entries = value
        .as_array()
        .filter(|a| !a.is_empty())
        .ok_or_else(|| invalid("labels : un tableau non vide est attendu".into()))?;
    if entries.iter().all(Value::is_number) {
        let classes: Vec<i64> = entries
            .iter()
            .map(|e| {
                e.as_i64().ok_or_else(|| {
                    invalid(format!(
                        "labels : {e} n'est pas un indice de classe entier (pour une valeur \
                         continue, écrire une ligne par prompt : [[v], ...])"
                    ))
                })
            })
            .collect::<Result<_>>()?;
        let mut distinct = classes.clone();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() < 2 {
            return Err(invalid("labels : une seule classe, rien à effacer".into()));
        }
        return Ok(classes
            .iter()
            .map(|class| {
                distinct
                    .iter()
                    .map(|c| f64::from(u8::from(c == class)))
                    .collect()
            })
            .collect());
    }
    let rows: Vec<Vec<f64>> = entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let row: Option<Vec<f64>> = entry.as_array().and_then(|values| {
                values
                    .iter()
                    .map(|v| v.as_f64().filter(|x| x.is_finite()))
                    .collect()
            });
            row.filter(|r| !r.is_empty()).ok_or_else(|| {
                invalid(format!(
                    "labels[{i}] : un tableau non vide de nombres finis est attendu"
                ))
            })
        })
        .collect::<Result<_>>()?;
    let m = rows[0].len();
    if let Some(i) = rows.iter().position(|r| r.len() != m) {
        return Err(invalid(format!(
            "labels[{i}] : {} valeurs, {m} attendues",
            rows[i].len()
        )));
    }
    Ok(rows)
}

/// A labels file: a JSON array (the content is checked by [`parse_labels`]).
pub fn load_labels(path: &std::path::Path) -> std::result::Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let value: Value =
        serde_json::from_str(text.trim_start_matches('\u{feff}')).map_err(|e| e.to_string())?;
    match value.is_array() {
        true => Ok(value),
        false => Err("un tableau JSON est attendu".into()),
    }
}

/// What a rank-k extraction says about each component, in `method_diagnostics.subspace`.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct SubspaceDiagnostics {
    /// Directions écrites : lignes des tenseurs a et b, de forme [rank, hidden].
    pub rank: usize,
    /// Valeur singulière de chaque composante (svd : de ΔX ; leace : de Σ_λ^{-1/2} Σ_XZ).
    pub sigma_t: Vec<f64>,
    /// Part de Σσ² portée par les t premières composantes.
    pub explained_variance_cumulative: Vec<f64>,
    /// d′ des projections des prompts positifs et négatifs lues par b_t ; null si indéfini.
    pub d_prime: Vec<Option<f64>>,
    /// ‖BᵀA − I‖ des tenseurs écrits (f32) : 0 pour un projecteur.
    #[serde(rename = "gram_BtA_error")]
    pub gram_bta_error: f64,
    /// leace : part de la covariance croisée lue par b_t qui reste hors échantillon, par
    /// composante de l'effaceur ajusté sans la part retenue ; null sans --holdout-frac.
    pub holdout_cross_cov_residual: Option<Vec<f64>>,
}

fn flat_f32(rows: &[Vec<f64>]) -> Vec<f32> {
    rows.iter().flatten().map(|&x| x as f32).collect()
}

/// The diagnostics of `a` and `b` (one direction per row); `sigma` holds every singular
/// value, the kept ones first.
pub fn diagnostics(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    sigma: &[f64],
    pos: &[Vec<f64>],
    neg: &[Vec<f64>],
    holdout_cross_cov_residual: Option<Vec<f64>>,
) -> SubspaceDiagnostics {
    let rank = b.len();
    let energy: f64 = sigma.iter().map(|s| s * s).sum();
    let mut carried = 0.0;
    let explained_variance_cumulative = sigma
        .iter()
        .take(rank)
        .map(|s| {
            carried += s * s;
            match energy > 0.0 {
                true => carried / energy,
                false => 0.0,
            }
        })
        .collect();
    let d_prime = b
        .iter()
        .map(|direction| {
            let norm = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
            let unit: Vec<f64> = direction.iter().map(|x| x / norm).collect();
            separation_along(pos, neg, &unit)
                .ok()
                .and_then(|s| s.stats.d_prime)
                .filter(|d| d.is_finite())
        })
        .collect();
    let (flat_a, flat_b) = (flat_f32(a), flat_f32(b));
    SubspaceDiagnostics {
        rank,
        sigma_t: sigma.iter().take(rank).copied().collect(),
        explained_variance_cumulative,
        d_prime,
        gram_bta_error: SubspaceProjection {
            a: &flat_a,
            b: &flat_b,
            alpha: 1.0,
            k: rank,
        }
        .gram_error(),
        holdout_cross_cov_residual,
    }
}

/// `k ≥ n / 4`: more directions than the examples can support.
pub fn rank_warning(k: usize, n: usize, examples: &str) -> Option<String> {
    (k > 1 && 4 * k >= n).then(|| {
        format!(
            "rang {k} ≥ n/4 avec n = {n} {examples} : le sous-espace est estimé sur trop peu \
             d'exemples"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rank_and_labels_are_read_from_the_params_and_checked() {
        let none = SubspaceOptions::from_params(&Value::Null).unwrap();
        assert!(none.is_default() && none == SubspaceOptions::default());
        let absent = SubspaceOptions::from_params(&json!({"svd_center": true})).unwrap();
        assert!(absent.is_default());
        let four = SubspaceOptions::from_params(&json!({"rank": 4, "labels": null})).unwrap();
        assert_eq!((four.rank, four.is_default()), (Some(4), false));
        assert!(SubspaceOptions::from_params(&json!({"rank": 1}))
            .unwrap()
            .is_default());
        for bad in [json!(0), json!(65), json!(-1), json!(2.5), json!("2")] {
            let err = SubspaceOptions::from_params(&json!({ "rank": bad })).unwrap_err();
            assert!(err.to_string().contains("[1, 64]"), "{err}");
        }
    }

    #[test]
    fn labels_are_class_indices_or_rows_of_numbers() {
        // Class indices: one-hot over the classes present, in increasing order.
        let one_hot = parse_labels(&json!([2, 0, 2, 7])).unwrap();
        assert_eq!(
            one_hot,
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ]
        );
        // Continuous labels: one row per prompt.
        let rows = parse_labels(&json!([[0.5, -1.0], [2, 3]])).unwrap();
        assert_eq!(rows, [[0.5, -1.0], [2.0, 3.0]]);
        let with = SubspaceOptions::from_params(&json!({"labels": [[1.0], [0.0]]})).unwrap();
        assert!(!with.is_default() && with.labels.unwrap().len() == 2);
        for (bad, why) in [
            (json!([]), "non vide"),
            (json!("x"), "non vide"),
            (json!([1, 1, 1]), "une seule classe"),
            (json!([0.5, 1.5]), "indice de classe"),
            (json!([[1.0], [1.0, 2.0]]), "1 attendues"),
            (json!([[1.0], "a"]), "labels[1]"),
            (json!([[1.0], []]), "labels[1]"),
            (json!([[1.0], [null]]), "labels[1]"),
        ] {
            let err = parse_labels(&bad).unwrap_err().to_string();
            assert!(err.contains(why), "{bad}: {err}");
        }
    }

    #[test]
    fn diagnostics_follow_the_components() {
        // Two orthonormal directions; the classes differ along the first one only.
        let v = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let pos = vec![vec![2.0, 0.1, 0.0], vec![2.2, -0.1, 0.3]];
        let neg = vec![vec![-2.0, 0.1, 0.1], vec![-2.2, -0.1, 0.0]];
        let d = diagnostics(&v, &v, &[3.0, 4.0, 0.0], &pos, &neg, None);
        assert_eq!((d.rank, d.sigma_t.as_slice()), (2, &[3.0, 4.0][..]));
        assert_eq!(d.explained_variance_cumulative, [9.0 / 25.0, 1.0]);
        assert!(d.gram_bta_error < 1e-12);
        assert!(d.d_prime[0].unwrap() > 10.0 && d.d_prime[1].unwrap().abs() < 1e-9);
        let json = serde_json::to_value(&d).unwrap();
        assert!(
            json.get("gram_BtA_error").is_some() && json["holdout_cross_cov_residual"].is_null()
        );

        assert!(rank_warning(4, 16, "paires")
            .unwrap()
            .contains("n = 16 paires"));
        assert_eq!(rank_warning(4, 17, "paires"), None);
        assert_eq!(rank_warning(1, 2, "paires"), None);
    }
}
