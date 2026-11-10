use std::path::{Path, PathBuf};
use std::str::FromStr;

use schemars::gen::SchemaGenerator;
use schemars::schema::{InstanceType, Schema, SchemaObject};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::error::{PatchError, Result};

// The JSON Schema derived here mirrors serde exactly (deny_unknown_fields, defaults, enum
// spellings) and adds the structural checks of `validate_patch` that a schema can express
// (non-empty lists, unique items, alpha bounds, layer spec syntax), so an agent fails
// before calling the tool. Descriptions are in French: an AI agent reads them.

/// `patch.json`, version 1. Unknown fields are rejected so a typo never goes unnoticed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(
    description = "Patch de chirurgie de poids (patch.json, version 1) : liste d'opérations qui modifient des matrices d'un modèle SafeTensors (MLP, o_proj, embed_tokens ; W ∓= alpha·v ou project_out)."
)]
pub struct PatchFile {
    #[schemars(schema_with = "schema_version_schema")]
    pub schema_version: u32,
    #[serde(default)]
    #[schemars(description = "Description libre du patch, destinée à l'humain qui le relit.")]
    pub description: Option<String>,
    /// Defaults to `true`: backing up is the safe choice when the agent forgets the field.
    #[serde(default = "default_true")]
    #[schemars(
        description = "Sauvegarde vérifiée (SHA-256) des fichiers touchés avant écriture. Défaut true ; le serveur MCP l'impose sauf --allow-no-backup."
    )]
    pub backup: bool,
    #[schemars(
        length(min = 1),
        description = "Opérations appliquées dans l'ordre ; au moins une. Toute erreur annule l'ensemble (rollback)."
    )]
    pub operations: Vec<Operation>,
}

fn default_true() -> bool {
    true
}

fn schema_version_schema(_: &mut SchemaGenerator) -> Schema {
    let mut schema = SchemaObject {
        instance_type: Some(InstanceType::Integer.into()),
        const_value: Some(1.into()),
        ..Default::default()
    };
    schema.metadata().description = Some("Version du format de patch : toujours 1.".into());
    schema.into()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(
    description = "Une opération : un vecteur direction appliqué à des matrices du modèle (projections MLP, o_proj, embed_tokens)."
)]
pub struct Operation {
    #[schemars(
        length(min = 1),
        description = "Identifiant unique de l'opération, repris dans le rapport."
    )]
    pub id: String,
    pub layers: LayersField,
    #[schemars(schema_with = "projections_schema")]
    pub projections: Vec<Projection>,
    pub operation: OperationKind,
    /// Relative paths are resolved against the directory of the patch file.
    #[schemars(
        description = "Fichier du vecteur direction (.safetensors ou .json). Chemin relatif au dossier du patch (CLI) ou à base_dir (MCP)."
    )]
    pub vector_file: PathBuf,
    #[serde(default)]
    #[schemars(
        description = "Nom du tenseur à lire dans un .safetensors qui en contient plusieurs ; interdit avec un .json."
    )]
    pub vector_tensor: Option<String>,
    /// `project_out` only: the read direction `b` (`vector_tensor` is then `a`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(
        description = "project_out uniquement (obligatoire) : tenseur b du fichier .leace.safetensors (vector_tensor désigne alors a)."
    )]
    pub vector_tensor_b: Option<String>,
    #[schemars(
        range(min = -100.0, max = 100.0),
        description = "Intensité du décalage W ∓= alpha·v, dans [-100, 100] ; |alpha| > 5 déclenche un avertissement. project_out : force dans [0, 1] (1 = effacement complet)."
    )]
    pub alpha: f32,
    #[serde(default)]
    #[schemars(
        description = "Axe de diffusion du vecteur. Pour des poids [out, in] (Llama) et une direction du flux résiduel : rows pour gate_proj/up_proj/embed_tokens, cols pour down_proj/o_proj. Défaut cols."
    )]
    pub axis: Axis,
}

fn projections_schema(gen: &mut SchemaGenerator) -> Schema {
    let mut schema = SchemaObject {
        instance_type: Some(InstanceType::Array.into()),
        ..Default::default()
    };
    let array = schema.array();
    array.items = Some(gen.subschema_for::<Projection>().into());
    array.min_items = Some(1);
    array.unique_items = Some(true);
    schema.metadata().description =
        Some("Matrices visées, sans doublon (un doublon décalerait deux fois).".into());
    schema.into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
#[schemars(
    description = "subtract retire alpha·v des poids, add l'ajoute ; project_out retire la direction lue par b des écrivains du flux résiduel : W ← W − alpha·a(bᵀW) pour down_proj et o_proj, W ← W − alpha·(W b)aᵀ pour embed_tokens (a = b = v unitaire : ablation directionnelle ; a et b d'un .leace.safetensors : effaceur LEACE)."
)]
pub enum OperationKind {
    Subtract,
    Add,
    // A plain comment: a doc comment on a variant would turn the schema enum into a oneOf.
    #[serde(rename = "project_out")]
    ProjectOut,
}

/// Declaration order is the canonical order (gate, up, down, then the residual writers
/// outside the MLP).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[schemars(
    description = "Matrice visée : gate_proj, up_proj, down_proj (MLP de chaque couche), o_proj (sortie de l'attention de chaque couche) ou embed_tokens (plongements d'entrée, hors couches : sélectionné quel que soit layers)."
)]
pub enum Projection {
    #[serde(rename = "gate_proj")]
    Gate,
    #[serde(rename = "up_proj")]
    Up,
    #[serde(rename = "down_proj")]
    Down,
    #[serde(rename = "o_proj")]
    AttnOut,
    #[serde(rename = "embed_tokens")]
    EmbedTokens,
}

impl Projection {
    /// The matrices of the MLP, the only ones a GGUF patch or `autotune` can target.
    pub fn is_mlp(self) -> bool {
        matches!(self, Self::Gate | Self::Up | Self::Down)
    }
}

impl FromStr for Projection {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "gate" | "gate_proj" => Ok(Self::Gate),
            "up" | "up_proj" => Ok(Self::Up),
            "down" | "down_proj" => Ok(Self::Down),
            "o" | "o_proj" => Ok(Self::AttnOut),
            "embed" | "embed_tokens" => Ok(Self::EmbedTokens),
            other => Err(format!(
                "projection inconnue « {other} » (gate, up, down, o_proj ou embed_tokens)"
            )),
        }
    }
}

// Cols is the default per the patch contract. For a residual-stream direction on
// Llama-style `[out, in]` weights, gate/up and the embeddings (`[vocab, hidden]`) expect
// `rows`, down and o_proj expect `cols`. (A plain
// comment: a variant doc comment would turn the schema enum into a oneOf.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
#[schemars(description = "rows : v de longueur = colonnes ; cols : v de longueur = lignes.")]
pub enum Axis {
    Rows,
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

/// ECMA-262 (no inline flags): "all" in any case, "10-20" or "3,5,7", spaces tolerated,
/// exactly what `LayerSelection::from_str` accepts.
pub const LAYER_SPEC_PATTERN: &str = r"^\s*([Aa][Ll][Ll]|\d+\s*-\s*\d+|\d+(\s*,\s*\d+)*)\s*$";

/// Written by hand: the untagged enum needs the spec-string pattern and the non-empty,
/// duplicate-free list that `validate_patch` enforces.
impl JsonSchema for LayersField {
    fn schema_name() -> String {
        "LayersField".into()
    }

    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        let index = serde_json::json!({ "type": "integer", "minimum": 0 });
        serde_json::from_value(serde_json::json!({
            "description": "Couches visées : \"all\", une plage \"10-20\" ou une liste \"3,5,7\" (chaîne), un tableau d'indices [10, 11, 12], ou un objet {\"start\": 10, \"end\": 20} (bornes incluses).",
            "anyOf": [
                {
                    "type": "string",
                    "pattern": LAYER_SPEC_PATTERN,
                    "description": "\"all\", \"10-20\" ou \"3,5,7\"."
                },
                {
                    "type": "array",
                    "items": index,
                    "minItems": 1,
                    "uniqueItems": true,
                    "description": "Indices de couches, sans doublon."
                },
                {
                    "type": "object",
                    "properties": { "start": index, "end": index },
                    "required": ["start", "end"],
                    "additionalProperties": false,
                    "description": "Plage inclusive, start <= end."
                }
            ]
        }))
        // The literal above always deserializes; a permissive schema is the safe fallback.
        .unwrap_or(Schema::Bool(true))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LayerRange {
    pub start: usize,
    /// Inclusive.
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
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
        assert_eq!("o_proj".parse(), Ok(Projection::AttnOut));
        assert_eq!("embed_tokens".parse(), Ok(Projection::EmbedTokens));
        assert!("q_proj".parse::<Projection>().is_err());
        assert!(Projection::Down.is_mlp() && !Projection::AttnOut.is_mlp());
        let patch = parse_patch(
            r#"{"schema_version":1,"operations":[{"id":"w","layers":"all",
                "projections":["down_proj","o_proj","embed_tokens"],"operation":"project_out",
                "vector_file":"v.safetensors","vector_tensor":"direction",
                "vector_tensor_b":"direction","alpha":0.6}]}"#,
        )
        .unwrap();
        assert_eq!(
            patch.operations[0].projections,
            [
                Projection::Down,
                Projection::AttnOut,
                Projection::EmbedTokens
            ]
        );
    }
}
