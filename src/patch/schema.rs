use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use super::error::{PatchError, Result};

/// `patch.json`, version 1. Unknown fields are rejected so a typo never goes unnoticed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PatchFile {
    pub schema_version: u32,
    #[serde(default)]
    pub description: Option<String>,
    /// Defaults to `true`: backing up is the safe choice when the agent forgets the field.
    #[serde(default = "default_true")]
    pub backup: bool,
    pub operations: Vec<Operation>,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Operation {
    pub id: String,
    pub layers: LayersField,
    pub projections: Vec<Projection>,
    pub operation: OperationKind,
    /// Relative paths are resolved against the directory of the patch file.
    pub vector_file: PathBuf,
    #[serde(default)]
    pub vector_tensor: Option<String>,
    pub alpha: f32,
    #[serde(default)]
    pub axis: Axis,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OperationKind {
    Subtract,
    Add,
}

/// Declaration order is the canonical order (gate, up, down).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Projection {
    #[serde(rename = "gate_proj")]
    Gate,
    #[serde(rename = "up_proj")]
    Up,
    #[serde(rename = "down_proj")]
    Down,
}

impl FromStr for Projection {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "gate" | "gate_proj" => Ok(Self::Gate),
            "up" | "up_proj" => Ok(Self::Up),
            "down" | "down_proj" => Ok(Self::Down),
            other => Err(format!(
                "projection inconnue « {other} » (gate, up ou down)"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Axis {
    Rows,
    /// Default per the patch contract. Note: for a residual-stream direction on Llama-style
    /// `[out, in]` weights, gate/up expect `rows` and down expects `cols`.
    #[default]
    Cols,
}

/// `"all"` or a CLI-style spec (`"10-20"`, `"3,5,7"`), an explicit list, or `{start, end}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LayersField {
    Spec(String),
    List(Vec<usize>),
    Range(LayerRange),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerRange {
    pub start: usize,
    /// Inclusive.
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerSelection {
    All,
    List(Vec<usize>),
    Range { start: usize, end_inclusive: usize },
}

impl FromStr for LayerSelection {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        let s = s.trim();
        let num = |part: &str| {
            let part = part.trim();
            part.parse::<usize>()
                .map_err(|_| format!("numéro de couche invalide « {part} »"))
        };
        if s.is_empty() {
            return Err("sélection de couches vide".to_owned());
        }
        if s.eq_ignore_ascii_case("all") {
            return Ok(Self::All);
        }
        if let Some((start, end)) = s.split_once('-') {
            let (start, end_inclusive) = (num(start)?, num(end)?);
            if start > end_inclusive {
                return Err(format!("plage inversée {start}-{end_inclusive}"));
            }
            return Ok(Self::Range {
                start,
                end_inclusive,
            });
        }
        s.split(',')
            .map(num)
            .collect::<std::result::Result<_, _>>()
            .map(Self::List)
    }
}

impl LayersField {
    /// Structural conversion; emptiness and duplicates are checked by validation.
    pub fn to_selection(&self) -> std::result::Result<LayerSelection, String> {
        match self {
            LayersField::Spec(spec) => spec.parse(),
            LayersField::List(list) => Ok(LayerSelection::List(list.clone())),
            LayersField::Range(LayerRange { start, end }) if start > end => {
                Err(format!("plage inversée {start}-{end}"))
            }
            LayersField::Range(LayerRange { start, end }) => Ok(LayerSelection::Range {
                start: *start,
                end_inclusive: *end,
            }),
        }
    }
}

pub fn parse_patch(json: &str) -> Result<PatchFile> {
    serde_json::from_str(json).map_err(|e| PatchError::InvalidJson {
        message: e.to_string(),
        line: e.line(),
        column: e.column(),
    })
}

pub fn load_patch(path: &Path) -> Result<PatchFile> {
    let json = std::fs::read_to_string(path).map_err(|e| PatchError::PatchUnreadable {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;
    parse_patch(&json)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: &str = r#"{
        "schema_version": 1,
        "description": "remove refusal",
        "backup": true,
        "operations": [
            {"id": "op-1", "layers": "all", "projections": ["down_proj", "gate_proj"],
             "operation": "subtract", "vector_file": "directions/refusal.safetensors",
             "vector_tensor": "layer_14", "alpha": 0.35, "axis": "rows"},
            {"id": "op-2", "layers": [10, 11, 12], "projections": ["up_proj"],
             "operation": "add", "vector_file": "v.json", "alpha": -1},
            {"id": "op-3", "layers": {"start": 10, "end": 20}, "projections": ["down_proj"],
             "operation": "subtract", "vector_file": "v.json", "alpha": 2.5}
        ]
    }"#;

    #[test]
    fn full_patch_deserializes_with_defaults() {
        let p = parse_patch(FULL).unwrap();
        assert_eq!(p.schema_version, 1);
        assert_eq!(p.operations.len(), 3);
        let [a, b, c] = &p.operations[..] else {
            panic!()
        };
        assert_eq!(a.layers, LayersField::Spec("all".into()));
        assert_eq!(a.projections, [Projection::Down, Projection::Gate]);
        assert_eq!(a.axis, Axis::Rows);
        assert_eq!(a.vector_tensor.as_deref(), Some("layer_14"));
        assert_eq!(b.layers, LayersField::List(vec![10, 11, 12]));
        assert_eq!(b.operation, OperationKind::Add);
        assert_eq!(b.axis, Axis::Cols, "axis defaults to cols");
        assert_eq!(
            c.layers,
            LayersField::Range(LayerRange { start: 10, end: 20 })
        );

        let minimal = parse_patch(r#"{"schema_version":1,"operations":[]}"#).unwrap();
        assert!(minimal.backup, "backup defaults to true");
    }

    #[test]
    fn schema_violations_are_json_errors() {
        let unknown_top = r#"{"schema_version":1,"operations":[],"bakup":false}"#;
        let unknown_op = FULL.replace(r#""alpha": 0.35"#, r#""alpha": 0.35, "alhpa": 1"#);
        let bad_alpha = FULL.replace("0.35", r#""0.35""#);
        let bad_range = FULL.replace(r#""end": 20"#, r#""end": 20, "step": 2"#);
        for json in [
            unknown_top,
            &unknown_op,
            &bad_alpha,
            &bad_range,
            "{\"schema_version\": 1,",
            "",
        ] {
            assert!(
                matches!(parse_patch(json), Err(PatchError::InvalidJson { .. })),
                "{json}"
            );
        }
    }

    #[test]
    fn layer_selection_from_str() {
        use LayerSelection::*;
        assert_eq!("all".parse(), Ok(All));
        assert_eq!(" ALL ".parse(), Ok(All));
        assert_eq!("3,5,7".parse(), Ok(List(vec![3, 5, 7])));
        assert_eq!(
            "10-20".parse(),
            Ok(Range {
                start: 10,
                end_inclusive: 20
            })
        );
        assert_eq!(
            " 4 - 6 ".parse(),
            Ok(Range {
                start: 4,
                end_inclusive: 6
            })
        );
        assert_eq!("7".parse(), Ok(List(vec![7])));
        for bad in ["20-10", "a", "", "  ", "1,,2", "1-", "-3", "1-3,5"] {
            assert!(bad.parse::<LayerSelection>().is_err(), "{bad:?}");
        }
        assert!("20-10"
            .parse::<LayerSelection>()
            .unwrap_err()
            .contains("inversée"));
    }

    #[test]
    fn projection_from_str() {
        assert_eq!("gate".parse(), Ok(Projection::Gate));
        assert_eq!("Down_Proj".parse(), Ok(Projection::Down));
        assert!("q_proj".parse::<Projection>().is_err());
    }
}
