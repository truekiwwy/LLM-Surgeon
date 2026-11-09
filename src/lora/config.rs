//! `adapter_config.json` reading (refusals: DoRA, fan_in_fan_out, rank/alpha patterns) and
//! writing (the one `lora extract` puts next to the adapter).

use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::surgery::error::{Result, SurgeryError};

pub const CONFIG_FILE: &str = "adapter_config.json";

/// The fields of a PEFT `adapter_config.json` that decide how an adapter is merged; the
/// others are ignored.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct AdapterConfig {
    pub r: Option<usize>,
    pub lora_alpha: Option<f64>,
    pub use_rslora: bool,
    pub use_dora: bool,
    pub fan_in_fan_out: bool,
    pub rank_pattern: Option<Value>,
    pub alpha_pattern: Option<Value>,
    pub modules_to_save: Option<Value>,
}

impl AdapterConfig {
    pub fn load(path: &Path) -> Result<Self> {
        // A missing or unreadable config is a usage error (exit 2), like an invalid one.
        let text = std::fs::read_to_string(path)
            .map_err(|e| SurgeryError::Usage(format!("{} : {e}", path.display())))?;
        serde_json::from_str(&text).map_err(|e| {
            SurgeryError::Usage(format!(
                "{} : adapter_config.json invalide ({e})",
                path.display()
            ))
        })
    }

    /// Explicit refusals, all listed at once: nothing silently merged with a wrong formula.
    pub fn check_supported(&self) -> Result<()> {
        let non_empty = |v: &Option<Value>| match v {
            Some(Value::Object(m)) => !m.is_empty(),
            Some(Value::Null) | None => false,
            Some(_) => true,
        };
        let mut refused = Vec::new();
        if self.use_dora {
            refused.push("use_dora : DoRA n'est pas supporté");
        }
        if self.fan_in_fan_out {
            refused.push("fan_in_fan_out : true n'est pas supporté (poids transposés)");
        }
        if non_empty(&self.rank_pattern) {
            refused.push("rank_pattern non vide : rangs par module non supportés");
        }
        if non_empty(&self.alpha_pattern) {
            refused.push("alpha_pattern non vide : alpha par module non supporté");
        }
        match refused.is_empty() {
            true => Ok(()),
            false => Err(SurgeryError::Usage(format!(
                "adapter non supporté :\n  - {}",
                refused.join("\n  - ")
            ))),
        }
    }

    /// `modules_to_save` weights are fully trained modules stored next to the LoRA keys;
    /// the merge ignores them.
    pub fn has_modules_to_save(&self) -> bool {
        match &self.modules_to_save {
            Some(Value::Array(a)) => !a.is_empty(),
            Some(Value::Null) | None => false,
            Some(_) => true,
        }
    }
}

/// The config to read: `explicit` (must exist) or the neighbour of the adapter, `None` when
/// there is none.
pub fn locate(lora: &Path, explicit: Option<&Path>) -> Option<PathBuf> {
    match explicit {
        Some(p) => Some(p.to_path_buf()),
        None => {
            let p = lora.with_file_name(CONFIG_FILE);
            p.is_file().then_some(p)
        }
    }
}

/// Body of the `adapter_config.json` written by `lora extract` (sorted keys, so the file is
/// deterministic). `base_model_name_or_path` is a directory name, never an absolute path.
pub fn render(r: usize, target_modules: &[String], base_name: &str) -> String {
    let config = json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": r,
        "lora_alpha": r,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_rslora": false,
        "use_dora": false,
        "fan_in_fan_out": false,
        "target_modules": target_modules,
        "base_model_name_or_path": base_name,
    });
    let mut text = serde_json::to_string_pretty(&config).unwrap_or_default();
    text.push('\n');
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rendered_config_parses_back() {
        let text = render(4, &["q_proj".into(), "v_proj".into()], "base");
        let cfg: AdapterConfig = serde_json::from_str(&text).unwrap();
        assert_eq!((cfg.r, cfg.lora_alpha), (Some(4), Some(4.0)));
        cfg.check_supported().unwrap();
        assert!(!cfg.has_modules_to_save());
    }

    #[test]
    fn refusals_are_listed_together() {
        let cfg: AdapterConfig = serde_json::from_str(
            r#"{"use_dora": true, "fan_in_fan_out": true, "rank_pattern": {"q": 2}, "alpha_pattern": {}}"#,
        )
        .unwrap();
        let msg = cfg.check_supported().unwrap_err().to_string();
        assert!(
            msg.contains("use_dora")
                && msg.contains("fan_in_fan_out")
                && msg.contains("rank_pattern")
        );
        assert!(!msg.contains("alpha_pattern"));
    }
}
