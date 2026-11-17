//! Launch forms, generated from the engine's tool definitions (`surgeon schema --kind
//! tools`, the `tools/list` of the MCP server): a field per property, the same names,
//! required set and choices. A form can never offer an option the engine does not have.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

#[derive(Debug, Clone, PartialEq)]
pub enum FieldKind {
    Text,
    Integer,
    Number,
    Bool,
    Choice(Vec<String>),
    /// Anything else (arrays, objects, unions): JSON, or a bare string.
    Json,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: String,
    pub kind: FieldKind,
    pub required: bool,
    pub description: String,
    pub default: Option<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Form {
    pub tool: String,
    pub title: String,
    pub description: String,
    pub fields: Vec<Field>,
}

/// The settings `[SCALPING]` shows besides the required fields, per surgery tool: the ones
/// that change the outcome most. `[ADVANCE]` shows every field.
const KEY_FIELDS: [(&str, &[&str]); 5] = [
    ("surgeon_apply", &["dry_run", "gguf"]),
    ("surgeon_extract", &["method", "layer", "site"]),
    (
        "surgeon_autotune",
        &["objective", "target_layer_range", "alpha_max"],
    ),
    ("surgeon_diff", &["rank", "include"]),
    ("surgeon_lora", &["model", "lora", "out"]),
];

impl Form {
    /// The field belongs to the guided form: required, or one of the tool's key settings.
    pub fn essential(&self, field: &Field) -> bool {
        field.required
            || KEY_FIELDS
                .iter()
                .any(|(tool, keys)| *tool == self.tool && keys.contains(&field.name.as_str()))
    }
}

/// One form per engine tool, in the engine's order.
pub fn forms() -> Vec<Form> {
    weight_surgeon::mcp::tool_definitions()
        .as_array()
        .map(|tools| tools.iter().map(form_of).collect())
        .unwrap_or_default()
}

pub fn form_of(tool: &Value) -> Form {
    let schema = &tool["inputSchema"];
    let required: Vec<&str> = schema["required"]
        .as_array()
        .map(|r| r.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let empty = Map::new();
    let mut fields: Vec<Field> = schema["properties"]
        .as_object()
        .unwrap_or(&empty)
        .iter()
        .map(|(name, prop)| Field {
            name: name.clone(),
            kind: kind_of(prop),
            required: required.contains(&name.as_str()),
            description: prop["description"].as_str().unwrap_or_default().to_owned(),
            default: prop.get("default").filter(|d| !d.is_null()).cloned(),
        })
        .collect();
    // Required fields first, then by name.
    fields.sort_by(|a, b| b.required.cmp(&a.required).then(a.name.cmp(&b.name)));
    Form {
        tool: tool["name"].as_str().unwrap_or_default().to_owned(),
        title: tool["annotations"]["title"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
        description: tool["description"].as_str().unwrap_or_default().to_owned(),
        fields,
    }
}

/// The string values of an enum, through `oneOf` / `anyOf` branches and `null`.
pub fn choices(schema: &Value) -> Option<Vec<String>> {
    if let Some(values) = schema["enum"].as_array() {
        return values
            .iter()
            .filter(|v| !v.is_null())
            .map(|v| v.as_str().map(String::from))
            .collect();
    }
    let branches = schema["oneOf"]
        .as_array()
        .or_else(|| schema["anyOf"].as_array())?;
    let mut out = Vec::new();
    for branch in branches.iter().filter(|b| b["type"] != "null") {
        out.extend(choices(branch)?);
    }
    (!out.is_empty()).then_some(out)
}

fn kind_of(schema: &Value) -> FieldKind {
    if let Some(values) = choices(schema) {
        return FieldKind::Choice(values);
    }
    // `T | null`, written as a type list or as `anyOf: [T, {"type": "null"}]`.
    if let Some(branches) = schema["anyOf"].as_array() {
        let real: Vec<&Value> = branches.iter().filter(|b| b["type"] != "null").collect();
        if let [only] = real.as_slice() {
            return kind_of(only);
        }
        return FieldKind::Json;
    }
    let types: Vec<&str> = match &schema["type"] {
        Value::String(t) => vec![t.as_str()],
        Value::Array(ts) => ts.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    };
    match types
        .into_iter()
        .filter(|t| *t != "null")
        .collect::<Vec<_>>()[..]
    {
        ["string"] => FieldKind::Text,
        ["integer"] => FieldKind::Integer,
        ["number"] => FieldKind::Number,
        ["boolean"] => FieldKind::Bool,
        _ => FieldKind::Json,
    }
}

impl Form {
    /// The tool arguments from the values typed in (a missing or empty value is left out).
    pub fn args(&self, values: &BTreeMap<String, String>) -> Result<Value, String> {
        let mut out = Map::new();
        for field in &self.fields {
            let raw = values.get(&field.name).map(|v| v.trim()).unwrap_or("");
            if raw.is_empty() {
                if field.required {
                    return Err(format!("{} : requis", field.name));
                }
                continue;
            }
            let bad = |what: &str| format!("{} : {what} attendu, « {raw} » reçu", field.name);
            let value = match &field.kind {
                FieldKind::Text => Value::from(raw),
                FieldKind::Integer => Value::from(raw.parse::<i64>().map_err(|_| bad("entier"))?),
                FieldKind::Number => {
                    let n = raw.parse::<f64>().map_err(|_| bad("nombre"))?;
                    serde_json::Number::from_f64(n)
                        .map(Value::Number)
                        .ok_or_else(|| bad("nombre fini"))?
                }
                FieldKind::Bool => Value::from(raw.parse::<bool>().map_err(|_| bad("true/false"))?),
                FieldKind::Choice(options) => match options.iter().any(|o| o == raw) {
                    true => Value::from(raw),
                    false => return Err(bad(&options.join(" | "))),
                },
                FieldKind::Json => serde_json::from_str(raw).unwrap_or_else(|_| Value::from(raw)),
            };
            out.insert(field.name.clone(), value);
        }
        Ok(Value::Object(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn every_form_is_its_engine_schema() {
        let tools = weight_surgeon::mcp::tool_definitions();
        let tools = tools.as_array().unwrap();
        let forms = forms();
        assert_eq!(forms.len(), tools.len());
        for (form, tool) in forms.iter().zip(tools) {
            let schema = &tool["inputSchema"];
            assert_eq!(form.tool, tool["name"]);
            let names: BTreeSet<&str> = form.fields.iter().map(|f| f.name.as_str()).collect();
            let props: BTreeSet<&str> = schema["properties"]
                .as_object()
                .map(|p| p.keys().map(String::as_str).collect())
                .unwrap_or_default();
            assert_eq!(names, props, "{}", form.tool);
            let required: BTreeSet<&str> = form
                .fields
                .iter()
                .filter(|f| f.required)
                .map(|f| f.name.as_str())
                .collect();
            let expected: BTreeSet<&str> = schema["required"]
                .as_array()
                .map(|r| r.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            assert_eq!(required, expected, "{}", form.tool);
            for field in &form.fields {
                let prop = &schema["properties"][&field.name];
                if let FieldKind::Choice(options) = &field.kind {
                    assert_eq!(
                        Some(options.clone()),
                        choices(prop),
                        "{}.{}",
                        form.tool,
                        field.name
                    );
                }
            }
        }
    }

    #[test]
    fn known_fields_get_their_widget() {
        let forms = forms();
        let field = |tool: &str, name: &str| {
            forms
                .iter()
                .find(|f| f.tool == tool)
                .and_then(|f| f.fields.iter().find(|x| x.name == name))
                .unwrap_or_else(|| panic!("{tool}.{name}"))
                .kind
                .clone()
        };
        assert_eq!(field("surgeon_check", "model"), FieldKind::Text);
        assert_eq!(field("surgeon_apply", "dry_run"), FieldKind::Bool);
        assert_eq!(field("surgeon_extract", "layer"), FieldKind::Integer);
        assert_eq!(field("surgeon_inspect", "layers"), FieldKind::Json);
        let FieldKind::Choice(modes) = field("surgeon_bench", "mode") else {
            panic!("mode")
        };
        assert!(modes.contains(&"refusal".to_owned()));
        let FieldKind::Choice(methods) = field("surgeon_extract", "method") else {
            panic!("method")
        };
        assert_eq!(methods, ["mean_diff", "svd", "leace"]);
    }

    #[test]
    fn the_guided_form_keeps_required_and_key_fields_of_existing_names() {
        let forms = forms();
        for (tool, keys) in KEY_FIELDS {
            let form = forms.iter().find(|f| f.tool == tool).expect(tool);
            for key in keys {
                assert!(
                    form.fields.iter().any(|f| f.name == *key),
                    "{tool}.{key} is not in the schema"
                );
            }
            let essential: Vec<&str> = form
                .fields
                .iter()
                .filter(|f| form.essential(f))
                .map(|f| f.name.as_str())
                .collect();
            assert!(essential.len() < form.fields.len(), "{tool}: {essential:?}");
            assert!(form
                .fields
                .iter()
                .filter(|f| f.required)
                .all(|f| essential.contains(&f.name.as_str())));
        }
    }

    #[test]
    fn values_become_typed_arguments() {
        let form = forms()
            .into_iter()
            .find(|f| f.tool == "surgeon_inspect")
            .unwrap();
        let mut values = BTreeMap::new();
        assert!(form.args(&values).unwrap_err().contains("model"));
        values.insert("model".into(), "m".into());
        values.insert("layers".into(), "1-3".into());
        values.insert("projections".into(), r#"["down_proj"]"#.into());
        values.insert("max_tensors".into(), "12".into());
        values.insert("show_stats".into(), "true".into());
        let args = form.args(&values).unwrap();
        assert_eq!(
            args,
            serde_json::json!({"model": "m", "layers": "1-3", "projections": ["down_proj"],
                "max_tensors": 12, "show_stats": true})
        );
        values.insert("max_tensors".into(), "douze".into());
        assert!(form.args(&values).unwrap_err().contains("entier"));
    }
}
