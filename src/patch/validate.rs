use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::Serialize;

use super::error::{PatchError, Result, ValidationIssue};
use super::schema::{Axis, LayerSelection, OperationKind, PatchFile, Projection};

pub const SUPPORTED_SCHEMA_VERSION: u32 = 1;
pub const MAX_ABS_ALPHA: f32 = 100.0;
/// Above this |alpha| the patch is accepted with a warning.
pub const WARN_ABS_ALPHA: f32 = 5.0;

/// Build with `ValidationOptions { no_backup: true, ..Default::default() }`.
#[derive(Debug, Clone)]
pub struct ValidationOptions {
    /// Check that every `vector_file` exists (disable for pure schema checks).
    pub check_files: bool,
    /// `--no-backup` was requested: refused when the patch imposes `backup: true`.
    pub no_backup: bool,
}

impl Default for ValidationOptions {
    fn default() -> Self {
        Self {
            check_files: true,
            no_backup: false,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidatedPatch {
    pub description: Option<String>,
    /// Effective backup decision (patch field combined with `--no-backup`).
    pub backup: bool,
    pub operations: Vec<ValidatedOperation>,
    pub warnings: Vec<ValidationIssue>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidatedOperation {
    pub id: String,
    pub layers: LayerSelection,
    pub projections: Vec<Projection>,
    pub kind: OperationKind,
    /// Resolved against the patch directory.
    pub vector_file: PathBuf,
    pub vector_tensor: Option<String>,
    pub alpha: f32,
    pub axis: Axis,
}

/// Checks everything and reports every problem at once, so an agent fixes its patch in one
/// round trip.
pub fn validate_patch(
    patch: &PatchFile,
    base_dir: &Path,
    opts: &ValidationOptions,
) -> Result<ValidatedPatch> {
    let mut issues = Vec::new();
    let mut warnings = Vec::new();
    let mut issue = |path: String, message: String| issues.push(ValidationIssue { path, message });

    if patch.schema_version != SUPPORTED_SCHEMA_VERSION {
        issue(
            "schema_version".into(),
            format!(
                "version {} non supportée (attendu {SUPPORTED_SCHEMA_VERSION})",
                patch.schema_version
            ),
        );
    }
    if patch.operations.is_empty() {
        issue(
            "operations".into(),
            "au moins une opération est requise".into(),
        );
    }
    if patch.backup && opts.no_backup {
        issue(
            "backup".into(),
            "le patch impose backup=true : --no-backup refusé".into(),
        );
    }

    let mut seen_ids: HashMap<&str, usize> = HashMap::new();
    let mut operations = Vec::with_capacity(patch.operations.len());
    for (i, op) in patch.operations.iter().enumerate() {
        let at = |field: &str| format!("operations[{i}].{field}");

        if op.id.trim().is_empty() {
            issue(at("id"), "id vide".into());
        } else if let Some(first) = seen_ids.get(op.id.as_str()) {
            issue(
                at("id"),
                format!("id « {} » déjà utilisé par operations[{first}]", op.id),
            );
        } else {
            seen_ids.insert(&op.id, i);
        }

        if !(op.alpha.is_finite() && op.alpha.abs() <= MAX_ABS_ALPHA) {
            issue(
                at("alpha"),
                format!(
                    "alpha doit être fini et dans [-{MAX_ABS_ALPHA}, {MAX_ABS_ALPHA}], reçu {}",
                    op.alpha
                ),
            );
        } else if op.alpha.abs() > WARN_ABS_ALPHA {
            warnings.push(ValidationIssue {
                path: at("alpha"),
                message: format!(
                    "|alpha| = {} > {WARN_ABS_ALPHA} : modification très forte",
                    op.alpha.abs()
                ),
            });
        }

        // A duplicate would shift the same tensor twice.
        if op.projections.is_empty() {
            issue(
                at("projections"),
                "au moins une projection est requise".into(),
            );
        } else if op.projections.iter().collect::<HashSet<_>>().len() != op.projections.len() {
            issue(at("projections"), "projection dupliquée".into());
        }

        let layers = match op.layers.to_selection() {
            Ok(LayerSelection::List(list)) if list.is_empty() => {
                issue(at("layers"), "liste de couches vide".into());
                None
            }
            Ok(LayerSelection::List(list))
                if list.iter().collect::<HashSet<_>>().len() != list.len() =>
            {
                issue(at("layers"), "couche dupliquée".into());
                None
            }
            Ok(selection) => Some(selection),
            Err(message) => {
                issue(at("layers"), message);
                None
            }
        };

        let ext = op
            .vector_file
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        let vector_file = base_dir.join(&op.vector_file);
        match ext.as_deref() {
            Some("safetensors") => {}
            Some("json") if op.vector_tensor.is_some() => issue(
                at("vector_tensor"),
                "vector_tensor ne s'applique qu'à un fichier .safetensors".into(),
            ),
            Some("json") => {}
            _ => issue(
                at("vector_file"),
                "extension non supportée (attendu .safetensors ou .json)".into(),
            ),
        }
        if opts.check_files && !vector_file.is_file() {
            issue(
                at("vector_file"),
                format!("fichier introuvable : {}", vector_file.display()),
            );
        }

        if let Some(layers) = layers {
            operations.push(ValidatedOperation {
                id: op.id.clone(),
                layers,
                projections: op.projections.clone(),
                kind: op.operation,
                vector_file,
                vector_tensor: op.vector_tensor.clone(),
                alpha: op.alpha,
                axis: op.axis,
            });
        }
    }

    if !issues.is_empty() {
        return Err(PatchError::Invalid(issues));
    }
    Ok(ValidatedPatch {
        description: patch.description.clone(),
        backup: patch.backup && !opts.no_backup,
        operations,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::schema::parse_patch;

    fn op(id: &str, extra: &str) -> String {
        format!(
            r#"{{"id":"{id}","layers":"all","projections":["gate_proj"],"operation":"subtract",
                "vector_file":"v.json","alpha":0.5{extra}}}"#
        )
    }

    fn patch(ops: &[String]) -> PatchFile {
        parse_patch(&format!(
            r#"{{"schema_version":1,"operations":[{}]}}"#,
            ops.join(",")
        ))
        .unwrap()
    }

    fn no_files() -> ValidationOptions {
        ValidationOptions {
            check_files: false,
            ..Default::default()
        }
    }

    fn issues(result: Result<ValidatedPatch>) -> Vec<(String, String)> {
        match result {
            Err(PatchError::Invalid(issues)) => {
                issues.into_iter().map(|i| (i.path, i.message)).collect()
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    #[test]
    fn valid_patch_resolves_paths_and_warns_on_large_alpha() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("v.json"), "[1]").unwrap();
        let p = patch(&[
            op("a", ""),
            op("b", "").replace(r#""alpha":0.5"#, r#""alpha":-7"#),
        ]);
        let v = validate_patch(&p, dir.path(), &ValidationOptions::default()).unwrap();
        assert_eq!(v.operations.len(), 2);
        assert_eq!(v.operations[0].vector_file, dir.path().join("v.json"));
        assert_eq!(v.operations[0].layers, LayerSelection::All);
        assert!(v.backup);
        assert_eq!(v.warnings.len(), 1);
        assert_eq!(v.warnings[0].path, "operations[1].alpha");
    }

    #[test]
    fn reports_every_issue_with_its_json_path() {
        let p = patch(&[
            op("a", ""),
            op("a", "").replace(r#""alpha":0.5"#, r#""alpha":250"#),
            op("c", r#","layers":{"start":9,"end":3}"#).replace(r#""layers":"all","#, ""),
        ]);
        let got = issues(validate_patch(&p, Path::new("."), &no_files()));
        let paths: Vec<&str> = got.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(
            paths,
            [
                "operations[1].id",
                "operations[1].alpha",
                "operations[2].layers"
            ],
            "{got:?}"
        );
        assert!(got[0].1.contains("operations[0]"));
    }

    #[test]
    fn structural_issues() {
        let mut p = patch(&[
            op("a", "").replace(r#"["gate_proj"]"#, "[]"),
            op("b", "").replace(r#""all""#, "[]"),
            op("c", "")
                .replace(r#""all""#, "[1,1]")
                .replace("v.json", "v.bin"),
            op("d", r#","vector_tensor":"x""#),
            op("e", "").replace(r#"["gate_proj"]"#, r#"["up_proj","up_proj"]"#),
        ]);
        p.schema_version = 2;
        let got = issues(validate_patch(&p, Path::new("."), &no_files()));
        let paths: Vec<&str> = got.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(
            paths,
            [
                "schema_version",
                "operations[0].projections",
                "operations[1].layers",
                "operations[2].layers",
                "operations[2].vector_file",
                "operations[3].vector_tensor",
                "operations[4].projections",
            ]
        );
    }

    #[test]
    fn empty_operations_missing_file_and_refused_no_backup() {
        let empty = parse_patch(r#"{"schema_version":1,"operations":[]}"#).unwrap();
        assert_eq!(
            issues(validate_patch(&empty, Path::new("."), &no_files()))[0].0,
            "operations"
        );

        let dir = tempfile::tempdir().unwrap();
        let p = patch(&[op("a", "")]);
        let opts = ValidationOptions {
            no_backup: true,
            ..Default::default()
        };
        let paths: Vec<String> = issues(validate_patch(&p, dir.path(), &opts))
            .into_iter()
            .map(|i| i.0)
            .collect();
        assert_eq!(paths, ["backup", "operations[0].vector_file"]);

        let mut optional = patch(&[op("a", "")]);
        optional.backup = false;
        let v = validate_patch(&optional, dir.path(), &no_files()).unwrap();
        assert!(!v.backup);
    }
}
