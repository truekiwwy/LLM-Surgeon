//! `surgeon compose`: one direction made of several extracted vectors, ready for a patch.
//!
//! ```text
//! inputs   ablate v_1..v_n (weights α_i), keep k_1..k_m (protected directions)
//! checks   same hidden size, finite values, ‖v‖ > 1e-12, for every vector
//! standard d = Σ α_i v_i
//! orthogonal_preserve
//!          Q = modified Gram-Schmidt of k_1..k_m (twice, f64): an orthonormal basis
//!          v_i⊥ = v_i − Σ_j ⟨v_i, u_j⟩ u_j (twice), refused when ‖v_i⊥‖ / ‖v_i‖ < 1e-6
//!          d = Σ α_i v_i⊥, so ⟨d, k_j⟩ = 0: removing d leaves the kept directions alone
//! output   direction = d, a = b = d / ‖d‖ (f32, for subtract or project_out), and the
//!          recipe: sources with their SHA-256 and α, mode, cosines before and after
//! ```
//!
//! Everything is computed in f64 and converted to f32 only when written. A kept direction
//! (nearly) collinear with the previous ones, or a vector to ablate (nearly) inside the kept
//! subspace, is an error that names it: normalizing what is left would turn rounding noise
//! into a direction. Orthogonality in the residual stream is linear: it does not rule out
//! a nonlinear interference, which only a bench of both abilities measures.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Below this L2 norm a vector is zero.
pub const ZERO_NORM: f64 = 1e-12;
/// Below this ratio of its norm, what is left of a vector after projection is noise.
pub const COLLINEAR_RATIO: f64 = 1e-6;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ComposeError {
    #[error("argument invalide : {0}")]
    InvalidArgument(String),
    #[error("{} : {message}", .path.display())]
    Input { path: PathBuf, message: String },
    #[error(
        "{} (couche {}) : quasi colinéaire {what} (‖v⊥‖/‖v‖ = {ratio:.3e} < {COLLINEAR_RATIO:e}) : \
         ce qui reste après projection n'est que du bruit numérique",
        .path.display(),
        .layer.map_or("inconnue".to_owned(), |l| l.to_string())
    )]
    Collinear {
        path: PathBuf,
        layer: Option<usize>,
        what: &'static str,
        ratio: f64,
    },
    #[error("écriture de {} : {message}", .path.display())]
    Io { path: PathBuf, message: String },
}

impl ComposeError {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::InvalidArgument(_) | Self::Input { .. } => "invalid_input",
            Self::Collinear { .. } => "collinear_vectors",
            Self::Io { .. } => "compose_failed",
        }
    }
}

pub type Result<T> = std::result::Result<T, ComposeError>;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema, clap::ValueEnum,
)]
#[serde(rename_all = "snake_case")]
pub enum ComposeMode {
    /// Somme pondérée directe des vecteurs, sans orthogonalisation.
    Standard,
    /// Vecteurs à retirer projetés hors du sous-espace des directions préservées.
    #[default]
    #[value(name = "orthogonal-preserve", alias = "orthogonal_preserve")]
    OrthogonalPreserve,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SourceRole {
    /// Direction composée puis retirée.
    Ablate,
    /// Direction préservée.
    Keep,
}

/// A vector read for composition.
#[derive(Debug, Clone, PartialEq)]
pub struct ComposeVector {
    pub path: PathBuf,
    pub layer: Option<usize>,
    pub sha256: String,
    /// Weight in the sum (vectors to ablate); unused for a kept direction.
    pub alpha: f64,
    pub values: Vec<f64>,
    /// The header of a `surgeon extract` file (empty for a JSON array).
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SourceVector {
    pub role: SourceRole,
    pub path: PathBuf,
    /// sha256 du fichier.
    pub sha256: String,
    /// Poids dans la somme (vecteurs retirés) ; 0 pour une direction préservée.
    pub alpha: f64,
    /// Couche d'extraction (métadonnée layer), si connue.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<usize>,
    /// ‖v‖₂.
    pub norm: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AblateDiagnostics {
    pub path: PathBuf,
    /// cos(v, k_j) avant projection, dans l'ordre des directions préservées.
    pub cos_before: Vec<f64>,
    /// cos(v⊥, k_j) après projection (≈ 0 en mode orthogonal_preserve).
    pub cos_after: Vec<f64>,
    /// ‖v⊥‖ / ‖v‖ : part de la norme conservée par la projection (1 en mode standard).
    pub norm_ratio: f64,
}

/// What a composed patch was made of: written into the recipe (`composition` of the
/// patch file).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Composition {
    pub mode: ComposeMode,
    pub sources: Vec<SourceVector>,
    /// Un bloc par vecteur retiré.
    pub diagnostics: Vec<AblateDiagnostics>,
    /// cos(direction composée, k_j).
    pub result_cos_keep: Vec<f64>,
    /// ‖direction composée‖₂, avant normalisation.
    pub result_norm: f64,
    /// Fichier de la direction composée et son sha256.
    pub output: PathBuf,
    pub output_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Composed {
    /// d, unnormalized, f64.
    pub direction: Vec<f64>,
    pub composition: Composition,
    pub warnings: Vec<String>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let n = norm(a) * norm(b);
    match n > 0.0 {
        true => (dot(a, b) / n).clamp(-1.0, 1.0),
        false => 0.0,
    }
}

/// `v` minus its components on the orthonormal `basis`, done twice ("twice is enough",
/// Giraud et al.): one pass leaves rounding errors of order ε·κ along the basis.
fn project_out(v: &[f64], basis: &[Vec<f64>]) -> Vec<f64> {
    let mut out = v.to_vec();
    for _ in 0..2 {
        for u in basis {
            let c = dot(&out, u);
            out.iter_mut().zip(u).for_each(|(x, ui)| *x -= c * ui);
        }
    }
    out
}

/// Same size as the first vector, finite, not zero.
fn check(v: &ComposeVector, width: usize) -> Result<()> {
    let bad = |message: String| {
        Err(ComposeError::Input {
            path: v.path.clone(),
            message,
        })
    };
    if v.values.len() != width {
        return bad(format!(
            "dimension {} au lieu de {width} (hidden_size des autres vecteurs)",
            v.values.len()
        ));
    }
    if let Some(i) = v.values.iter().position(|x| !x.is_finite()) {
        return bad(format!("valeur non finie à l'indice {i} ({})", v.values[i]));
    }
    if norm(&v.values) <= ZERO_NORM {
        return bad(format!("vecteur nul (‖v‖ ≤ {ZERO_NORM:e})"));
    }
    if !v.alpha.is_finite() {
        return bad(format!("poids {} non fini", v.alpha));
    }
    Ok(())
}

/// The orthonormal basis of the kept directions, in their order.
pub fn orthonormal_basis(keep: &[ComposeVector]) -> Result<Vec<Vec<f64>>> {
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(keep.len());
    for k in keep {
        let before = norm(&k.values);
        let rest = project_out(&k.values, &basis);
        let ratio = norm(&rest) / before;
        if ratio < COLLINEAR_RATIO {
            return Err(ComposeError::Collinear {
                path: k.path.clone(),
                layer: k.layer,
                what: "aux directions préservées précédentes",
                ratio,
            });
        }
        let n = norm(&rest);
        basis.push(rest.iter().map(|x| x / n).collect());
    }
    Ok(basis)
}

/// The composed direction of `ablate` (weighted by their α), kept orthogonal to `keep` in
/// `OrthogonalPreserve` mode, with the diagnostics of the recipe (`output` and its sha256
/// are the caller's to fill once written).
pub fn compose(
    ablate: &[ComposeVector],
    keep: &[ComposeVector],
    mode: ComposeMode,
) -> Result<Composed> {
    let first = ablate.first().ok_or_else(|| {
        ComposeError::InvalidArgument("au moins un vecteur à retirer (--ablate)".into())
    })?;
    let width = first.values.len();
    for v in ablate.iter().chain(keep) {
        check(v, width)?;
    }
    let basis = match mode {
        ComposeMode::OrthogonalPreserve => orthonormal_basis(keep)?,
        ComposeMode::Standard => Vec::new(),
    };
    let mut direction = vec![0.0; width];
    let mut diagnostics = Vec::with_capacity(ablate.len());
    for v in ablate {
        let projected = project_out(&v.values, &basis);
        let ratio = norm(&projected) / norm(&v.values);
        if ratio < COLLINEAR_RATIO {
            return Err(ComposeError::Collinear {
                path: v.path.clone(),
                layer: v.layer,
                what: "au sous-espace des directions préservées",
                ratio,
            });
        }
        direction
            .iter_mut()
            .zip(&projected)
            .for_each(|(d, p)| *d += v.alpha * p);
        diagnostics.push(AblateDiagnostics {
            path: v.path.clone(),
            cos_before: keep.iter().map(|k| cosine(&v.values, &k.values)).collect(),
            cos_after: keep.iter().map(|k| cosine(&projected, &k.values)).collect(),
            norm_ratio: ratio,
        });
    }
    let result_norm = norm(&direction);
    if result_norm <= ZERO_NORM {
        return Err(ComposeError::InvalidArgument(format!(
            "direction composée nulle (‖d‖ ≤ {ZERO_NORM:e}) : les poids s'annulent"
        )));
    }
    let mut warnings = Vec::new();
    let layers: std::collections::BTreeSet<usize> =
        ablate.iter().chain(keep).filter_map(|v| v.layer).collect();
    if layers.len() > 1 {
        warnings.push(format!(
            "vecteurs extraits à des couches différentes {layers:?} : leurs bases résiduelles \
             ne coïncident pas exactement"
        ));
    }
    if mode == ComposeMode::Standard && !keep.is_empty() {
        warnings.push(
            "mode standard : les directions préservées ne servent qu'aux diagnostics (cos_before)"
                .into(),
        );
    }
    for d in &diagnostics {
        if d.norm_ratio < 0.5 {
            warnings.push(format!(
                "{} : la projection n'en garde que {:.0} % (‖v⊥‖/‖v‖ = {:.3})",
                d.path.display(),
                d.norm_ratio * 100.0,
                d.norm_ratio
            ));
        }
    }
    warnings.push(
        "orthogonalité linéaire seulement : mesurer les capacités préservées (bench eval) et \
         le refus (bench refusal) après application"
            .into(),
    );
    let source = |v: &ComposeVector, role: SourceRole| SourceVector {
        role,
        path: v.path.clone(),
        sha256: v.sha256.clone(),
        alpha: match role {
            SourceRole::Ablate => v.alpha,
            SourceRole::Keep => 0.0,
        },
        layer: v.layer,
        norm: norm(&v.values),
    };
    let sources = ablate
        .iter()
        .map(|v| source(v, SourceRole::Ablate))
        .chain(keep.iter().map(|v| source(v, SourceRole::Keep)))
        .collect();
    Ok(Composed {
        composition: Composition {
            mode,
            sources,
            diagnostics,
            result_cos_keep: keep.iter().map(|k| cosine(&direction, &k.values)).collect(),
            result_norm,
            output: PathBuf::new(),
            output_sha256: String::new(),
        },
        direction,
        warnings,
    })
}

/// A vector file: the `direction` tensor of a `.safetensors` (with its `layer` metadata) or
/// a `.json` array, and the file's sha256.
pub fn read_vector(path: &Path, alpha: f64) -> Result<ComposeVector> {
    let bad = |message: String| ComposeError::Input {
        path: path.to_path_buf(),
        message,
    };
    let safetensors = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
    let tensor = safetensors.then_some(crate::extract::TENSOR_NAME);
    let values =
        crate::engine::load_direction_vector(path, tensor).map_err(|e| bad(e.to_string()))?;
    let metadata = match safetensors {
        true => crate::engine::ReadOnlyModel::open(path)
            .map_err(|e| bad(e.to_string()))?
            .metadata(),
        false => BTreeMap::new(),
    };
    Ok(ComposeVector {
        path: path.to_path_buf(),
        layer: metadata.get("layer").and_then(|l| l.parse().ok()),
        sha256: crate::bench::sha256_file(path).map_err(|e| bad(e.to_string()))?,
        alpha,
        values: values.iter().map(|&x| f64::from(x)).collect(),
        metadata,
    })
}

/// `path` or `path:alpha` (the last `:` followed by a number; a Windows drive letter is
/// not one).
pub fn parse_weighted(spec: &str) -> std::result::Result<(PathBuf, f64), String> {
    match spec.rsplit_once(':') {
        Some((path, alpha)) if !path.is_empty() && alpha.parse::<f64>().is_ok() => {
            let alpha: f64 = alpha.parse().map_err(|_| format!("« {spec} »"))?;
            match alpha.is_finite() {
                true => Ok((PathBuf::from(path), alpha)),
                false => Err(format!("« {spec} » : poids non fini")),
            }
        }
        _ => Ok((PathBuf::from(spec), 1.0)),
    }
}

/// The bytes of the composed file: `direction` (d as is), `a` and `b` (d / ‖d‖, a
/// projector for `project_out`), and the extraction metadata of the first vector to
/// ablate, so that a probe or autotune reads the same layer, site, pooling and wrapping.
pub fn encode_composed(composed: &Composed, first: &ComposeVector) -> Result<Vec<u8>> {
    let d: Vec<f32> = composed.direction.iter().map(|&x| x as f32).collect();
    let n = composed.composition.result_norm;
    let unit: Vec<f32> = composed.direction.iter().map(|&x| (x / n) as f32).collect();
    let mut metadata: BTreeMap<String, String> = first
        .metadata
        .iter()
        .filter(|(k, _)| {
            matches!(
                k.as_str(),
                "layer"
                    | "site"
                    | "pooling"
                    | "chat_template"
                    | "chat_system"
                    | "model"
                    | "model_type"
            )
        })
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    metadata.insert(
        "compose_mode".into(),
        serde_json::to_value(composed.composition.mode)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_default(),
    );
    metadata.insert(
        "compose_sources".into(),
        composed.composition.sources.len().to_string(),
    );
    metadata.insert("surgeon_version".into(), env!("CARGO_PKG_VERSION").into());
    crate::extract::encode_tensors(&[("direction", &d), ("a", &unit), ("b", &unit)], &metadata)
        .map_err(|e| ComposeError::InvalidArgument(e.to_string()))
}

fn outcome_error(e: ComposeError) -> crate::app::Outcome {
    let code = match e {
        ComposeError::Io { .. } => crate::app::exit::GENERAL,
        _ => crate::app::exit::INVALID_INPUT,
    };
    crate::app::Outcome::fail(code, e.to_string()).with_kind(e.kind())
}

/// `surgeon compose` and the MCP tool `surgeon_compose`: read, compose, write the composed
/// direction (and, with `patch_out`, a `project_out` patch carrying the recipe), each file
/// atomically; nothing is written when a check fails.
pub fn core_compose(args: &crate::cli::ComposeArgs) -> crate::app::CoreResult {
    let file = |path: &Path, ext: &str| -> Result<()> {
        let bad = |message: String| ComposeError::Input {
            path: path.to_path_buf(),
            message,
        };
        if !path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case(ext))
        {
            return Err(bad(format!("l'extension .{ext} est requise")));
        }
        if path.exists() && !args.overwrite {
            return Err(bad("existe déjà (--overwrite pour le remplacer)".into()));
        }
        crate::extract::ensure_parent(path).map_err(|e| bad(e.to_string()))
    };
    let prepare = || -> Result<(Composed, Vec<ComposeVector>)> {
        file(&args.out, "safetensors")?;
        if let Some(patch) = &args.patch_out {
            file(patch, "json")?;
        }
        if !(0.0..=1.0).contains(&args.alpha) {
            return Err(ComposeError::InvalidArgument(
                "--alpha d'un project_out : dans [0, 1] (1 = effacement complet)".into(),
            ));
        }
        let ablate = args
            .ablate
            .iter()
            .map(|spec| {
                let (path, alpha) = parse_weighted(spec).map_err(ComposeError::InvalidArgument)?;
                read_vector(&path, alpha)
            })
            .collect::<Result<Vec<_>>>()?;
        let keep = args
            .keep
            .iter()
            .map(|path| read_vector(path, 0.0))
            .collect::<Result<Vec<_>>>()?;
        Ok((compose(&ablate, &keep, args.mode)?, ablate))
    };
    let (mut composed, ablate) = prepare().map_err(outcome_error)?;
    let bytes = encode_composed(&composed, &ablate[0]).map_err(outcome_error)?;
    let out = std::path::absolute(&args.out).unwrap_or_else(|_| args.out.clone());
    composed.composition.output = out.clone();
    composed.composition.output_sha256 = crate::bench::sha256_hex(&bytes);
    let patch = match &args.patch_out {
        None => None,
        Some(_) => {
            let layers = crate::patch::LayersField::Spec(args.layers.clone());
            let projections = args
                .projections
                .iter()
                .map(|p| serde_json::from_value(serde_json::Value::String(p.clone())))
                .collect::<std::result::Result<Vec<crate::patch::Projection>, _>>()
                .map_err(|e| {
                    outcome_error(ComposeError::InvalidArgument(format!(
                        "--projections : {e}"
                    )))
                })?;
            Some(crate::patch::PatchFile {
                schema_version: 1,
                description: Some(format!(
                    "surgeon compose ({:?}) : {} vecteur(s) retiré(s), {} direction(s) préservée(s)",
                    composed.composition.mode,
                    ablate.len(),
                    args.keep.len()
                )),
                backup: true,
                operations: vec![crate::patch::Operation {
                    id: "compose".into(),
                    layers,
                    projections,
                    operation: crate::patch::OperationKind::ProjectOut,
                    vector_file: out.clone(),
                    vector_tensor: Some("a".into()),
                    vector_tensor_b: Some("b".into()),
                    rank: None,
                    allow_non_projector: false,
                    alpha: args.alpha,
                    axis: crate::patch::Axis::default(),
                }],
                composition: Some(composed.composition.clone()),
            })
        }
    };
    let io = |path: &Path, e: crate::extract::ExtractError| {
        outcome_error(ComposeError::Io {
            path: path.to_path_buf(),
            message: e.to_string(),
        })
    };
    crate::extract::write_atomic(&args.out, &bytes, args.overwrite)
        .map_err(|e| io(&args.out, e))?;
    if let (Some(path), Some(patch)) = (&args.patch_out, &patch) {
        // Checked like any patch, against the file just written.
        let base = path.parent().map(Path::to_path_buf).unwrap_or_default();
        crate::patch::validate_patch(patch, &base, &crate::patch::ValidationOptions::default())
            .map_err(|e| {
                outcome_error(ComposeError::InvalidArgument(format!(
                    "patch produit invalide : {e}"
                )))
            })?;
        let mut json = serde_json::to_vec_pretty(patch)
            .map_err(|e| outcome_error(ComposeError::InvalidArgument(e.to_string())))?;
        json.push(b'\n');
        crate::extract::write_atomic(path, &json, args.overwrite).map_err(|e| io(path, e))?;
    }
    let c = &composed.composition;
    let mut human = vec![format!(
        "[COMPOSE] {:?} · {} vecteur(s) retiré(s) · {} direction(s) préservée(s) · ‖d‖ {:.4}",
        c.mode,
        ablate.len(),
        args.keep.len(),
        c.result_norm
    )];
    for d in &c.diagnostics {
        let worst = d.cos_after.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        human.push(format!(
            "  {} : norme gardée {:.1} %, |cos| max après {:.2e}",
            d.path.display(),
            d.norm_ratio * 100.0,
            worst
        ));
    }
    human.push(format!("[WRITE]   {}", out.display()));
    if let Some(path) = &args.patch_out {
        human.push(format!("[WRITE]   {}", path.display()));
    }
    let data = serde_json::json!({
        "out": out,
        "out_sha256": c.output_sha256,
        "patch_out": args.patch_out,
        "composition": c,
    });
    let mut outcome = crate::app::Outcome::ok(data, human.join("\n"));
    outcome.warnings = composed.warnings;
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::PatchFile;

    fn vector(name: &str, values: &[f64], alpha: f64) -> ComposeVector {
        ComposeVector {
            path: PathBuf::from(name),
            layer: Some(13),
            sha256: format!("sha-{name}"),
            alpha,
            values: values.to_vec(),
            metadata: BTreeMap::new(),
        }
    }

    /// Deterministic pseudo-random values in [-1, 1].
    fn noise(d: usize, seed: u64) -> Vec<f64> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..d)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn two_and_three_dimensions_by_hand() {
        // 2D: v = (1, 1) without its component on k = (1, 0) is (0, 1).
        let out = compose(
            &[vector("v", &[1.0, 1.0], 2.0)],
            &[vector("k", &[3.0, 0.0], 0.0)],
            ComposeMode::OrthogonalPreserve,
        )
        .unwrap();
        assert!(out.direction[0].abs() < 1e-15 && (out.direction[1] - 2.0).abs() < 1e-15);
        let diag = &out.composition.diagnostics[0];
        assert!((diag.cos_before[0] - 0.5f64.sqrt()).abs() < 1e-12);
        assert!(diag.cos_after[0].abs() < 1e-15);
        assert!((diag.norm_ratio - 0.5f64.sqrt()).abs() < 1e-12);
        // Standard: the plain weighted sum, the kept direction untouched.
        let plain = compose(
            &[
                vector("v", &[1.0, 1.0], 2.0),
                vector("w", &[0.0, 1.0], -1.0),
            ],
            &[vector("k", &[3.0, 0.0], 0.0)],
            ComposeMode::Standard,
        )
        .unwrap();
        assert_eq!(plain.direction, [2.0, 1.0]);
        assert_eq!(plain.composition.diagnostics[0].norm_ratio, 1.0);

        // 3D: keep two non-orthogonal directions; what is removed is along their normal.
        let keep = [
            vector("k1", &[1.0, 0.0, 0.0], 0.0),
            vector("k2", &[1.0, 1.0, 0.0], 0.0),
        ];
        let basis = orthonormal_basis(&keep).unwrap();
        assert!(dot(&basis[0], &basis[1]).abs() < 1e-15);
        let out = compose(
            &[vector("v", &[0.3, -2.0, 4.0], 1.0)],
            &keep,
            ComposeMode::OrthogonalPreserve,
        )
        .unwrap();
        assert!(out.direction[0].abs() < 1e-15 && out.direction[1].abs() < 1e-15);
        assert!((out.direction[2] - 4.0).abs() < 1e-15);
        assert!(out
            .composition
            .result_cos_keep
            .iter()
            .all(|c| c.abs() < 1e-15));
    }

    #[test]
    fn exact_orthogonality_in_4096_dimensions() {
        let d = 4096;
        let keep: Vec<ComposeVector> = (0..5)
            .map(|i| vector(&format!("k{i}"), &noise(d, 10 + i), 0.0))
            .collect();
        let ablate: Vec<ComposeVector> = (0..3)
            .map(|i| vector(&format!("v{i}"), &noise(d, 100 + i), 0.5 + i as f64))
            .collect();
        let basis = orthonormal_basis(&keep).unwrap();
        for (i, u) in basis.iter().enumerate() {
            assert!((norm(u) - 1.0).abs() < 1e-12);
            for w in &basis[i + 1..] {
                assert!(dot(u, w).abs() < 1e-6, "{}", dot(u, w));
            }
        }
        let out = compose(&ablate, &keep, ComposeMode::OrthogonalPreserve).unwrap();
        for k in &keep {
            assert!(cosine(&out.direction, &k.values).abs() < 1e-6);
        }
        for diag in &out.composition.diagnostics {
            assert!(diag.cos_after.iter().all(|c| c.abs() < 1e-6), "{diag:?}");
            // Random vectors in 4096 dimensions: almost all of the norm survives.
            assert!(diag.norm_ratio > 0.99 && diag.norm_ratio <= 1.0, "{diag:?}");
        }
        // The sum of projections equals the projection of the sum.
        let sum: Vec<f64> = (0..d)
            .map(|j| ablate.iter().map(|v| v.alpha * v.values[j]).sum())
            .collect();
        let projected = project_out(&sum, &basis);
        let gap: f64 = projected
            .iter()
            .zip(&out.direction)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(gap < 1e-9, "{gap}");
        // The ablated vectors are 3 sources, the kept ones 5.
        let roles: Vec<SourceRole> = out.composition.sources.iter().map(|s| s.role).collect();
        assert_eq!(roles.iter().filter(|r| **r == SourceRole::Keep).count(), 5);
    }

    #[test]
    fn bad_inputs_are_refused_with_their_name() {
        let k = vector("keep.safetensors", &[1.0, 0.0, 0.0], 0.0);
        let refuse = |ablate: ComposeVector, keep: &[ComposeVector]| {
            compose(&[ablate], keep, ComposeMode::OrthogonalPreserve)
                .unwrap_err()
                .to_string()
        };
        let err = refuse(vector("zero.safetensors", &[0.0, 0.0, 0.0], 1.0), &[]);
        assert!(
            err.contains("zero.safetensors") && err.contains("nul"),
            "{err}"
        );
        let err = refuse(vector("nan.safetensors", &[1.0, f64::NAN, 0.0], 1.0), &[]);
        assert!(err.contains("non finie à l'indice 1"), "{err}");
        let err = refuse(vector("inf.json", &[f64::INFINITY, 0.0, 0.0], 1.0), &[]);
        assert!(err.contains("inf.json"), "{err}");
        let err = refuse(
            vector("short.safetensors", &[1.0, 0.0], 1.0),
            std::slice::from_ref(&k),
        );
        assert!(
            err.contains("short.safetensors") || err.contains("keep.safetensors"),
            "{err}"
        );
        assert!(err.contains("dimension"), "{err}");

        // Collinear with a kept direction: an error naming the vector, its layer and ratio.
        let e = compose(
            &[vector("refusal_L14.safetensors", &[2.0, 1e-9, 0.0], 1.0)],
            std::slice::from_ref(&k),
            ComposeMode::OrthogonalPreserve,
        )
        .unwrap_err();
        assert!(matches!(e, ComposeError::Collinear { ratio, .. } if ratio < COLLINEAR_RATIO));
        let text = e.to_string();
        assert!(
            text.contains("refusal_L14.safetensors") && text.contains("couche 13"),
            "{text}"
        );
        assert_eq!(e.kind(), "collinear_vectors");
        // Two kept directions that are the same one.
        let twin = vector("twin.safetensors", &[-3.0, 0.0, 0.0], 0.0);
        let e = orthonormal_basis(&[k.clone(), twin]).unwrap_err();
        assert!(e.to_string().contains("twin.safetensors"), "{e}");
        // Weights that cancel out.
        let v = vector("v", &[0.0, 1.0, 0.0], 1.0);
        let w = vector("w", &[0.0, 1.0, 0.0], -1.0);
        let e = compose(&[v, w], &[], ComposeMode::Standard).unwrap_err();
        assert!(e.to_string().contains("s'annulent"), "{e}");
        assert!(compose(&[], &[k], ComposeMode::Standard).is_err());
    }

    #[test]
    fn the_recipe_round_trips_and_older_patches_still_read() {
        let older = r#"{"schema_version":1,"operations":[{"id":"a","layers":"all",
            "projections":["down_proj"],"operation":"subtract","vector_file":"v.json",
            "alpha":0.5}]}"#;
        let patch = crate::patch::parse_patch(older).unwrap();
        assert_eq!(patch.composition, None, "no field: no recipe");
        let plain = serde_json::to_string(&patch).unwrap();
        assert!(
            !plain.contains("composition"),
            "nothing added to a plain patch"
        );

        let mut out = compose(
            &[vector("refusal_L14.safetensors", &[1.0, 1.0, 0.5], 0.8)],
            &[vector("dir_math_L13.safetensors", &[1.0, 0.0, 0.0], 0.0)],
            ComposeMode::OrthogonalPreserve,
        )
        .unwrap();
        out.composition.output = PathBuf::from("composed.safetensors");
        out.composition.output_sha256 = "0".repeat(64);
        let recipe = PatchFile {
            composition: Some(out.composition),
            ..patch
        };
        let json = serde_json::to_string_pretty(&recipe).unwrap();
        for key in [
            "\"mode\": \"orthogonal_preserve\"",
            "\"sha256\": \"sha-refusal_L14.safetensors\"",
            "\"alpha\": 0.8",
            "\"cos_after\"",
            "\"norm_ratio\"",
        ] {
            assert!(json.contains(key), "{key} in {json}");
        }
        assert_eq!(crate::patch::parse_patch(&json).unwrap(), recipe);
    }

    #[test]
    fn weighted_paths_and_the_composed_file() {
        assert_eq!(
            parse_weighted("refusal.safetensors:0.5"),
            Ok(("refusal.safetensors".into(), 0.5))
        );
        assert_eq!(
            parse_weighted("C:/v/r.safetensors"),
            Ok(("C:/v/r.safetensors".into(), 1.0))
        );
        assert_eq!(
            parse_weighted("C:/v/r.json:-2"),
            Ok(("C:/v/r.json".into(), -2.0))
        );
        assert!(parse_weighted("r.json:inf").is_err());

        let mut v = vector("v", &[3.0, 4.0], 1.0);
        v.metadata.insert("layer".into(), "14".into());
        v.metadata.insert("site".into(), "resid_post".into());
        v.metadata.insert("d_prime".into(), "4.2".into());
        let out = compose(&[v.clone()], &[], ComposeMode::OrthogonalPreserve).unwrap();
        let bytes = encode_composed(&out, &v).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.safetensors");
        std::fs::write(&path, bytes).unwrap();
        let back = read_vector(&path, 1.0).unwrap();
        assert_eq!(back.values, [3.0, 4.0]);
        assert_eq!(back.layer, Some(14));
        assert_eq!(back.metadata["compose_mode"], "orthogonal_preserve");
        assert!(
            !back.metadata.contains_key("d_prime"),
            "the source's stats do not carry over"
        );
        let unit = crate::engine::load_direction_vector(&path, Some("a")).unwrap();
        assert!((unit[0] - 0.6).abs() < 1e-7 && (unit[1] - 0.8).abs() < 1e-7);
    }
}
