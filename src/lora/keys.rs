//! PEFT key mapping `base_model.model.<path>.lora_{A,B}[.default].weight` <-> `<path>.weight`.
//!
//! Merge side: [`pair_keys`] groups the adapter keys per target tensor and reports every key
//! that has no counterpart. Extract side: [`adapter_key`] builds the PEFT name of a factor.

use std::collections::BTreeMap;

const PREFIX: &str = "base_model.model.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    A,
    B,
}

impl Side {
    fn tag(self) -> &'static str {
        match self {
            Side::A => "lora_A",
            Side::B => "lora_B",
        }
    }
}

/// Both factors of one target tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyPair {
    /// Name of the weight in the model (`<path>.weight`).
    pub target: String,
    pub a: String,
    pub b: String,
}

/// `(target tensor, side)` of a PEFT factor key, `None` for any other key.
pub fn parse_key(key: &str) -> Option<(String, Side)> {
    let key = key.strip_prefix(PREFIX).unwrap_or(key);
    for side in [Side::A, Side::B] {
        let marker = format!(".{}.", side.tag());
        let Some((path, rest)) = key.split_once(&marker) else {
            continue;
        };
        // "weight" or "<adapter name>.weight" (PEFT writes `default` in some layouts).
        let named = rest
            .strip_suffix(".weight")
            .is_some_and(|name| !name.contains('.'));
        if !path.is_empty() && (rest == "weight" || named) {
            return Some((format!("{path}.weight"), side));
        }
    }
    None
}

/// `lora_embedding_A/B` keys: the embedding LoRA layout, not supported.
pub fn is_embedding_key(key: &str) -> bool {
    key.contains("lora_embedding_")
}

/// DoRA adapters carry a magnitude vector next to A and B.
pub fn is_dora_key(key: &str) -> bool {
    key.contains("lora_magnitude_vector")
}

/// PEFT name of the factor of `target` (`<path>.weight`), `None` without the `.weight` suffix.
pub fn adapter_key(target: &str, side: Side) -> Option<String> {
    let path = target.strip_suffix(".weight")?;
    Some(format!("{PREFIX}{path}.{}.weight", side.tag()))
}

/// Groups the keys per target, sorted by target name. Every orphan (unknown key, missing
/// counterpart, two keys for the same factor) is listed in the error, not only the first.
pub fn pair_keys<'a>(keys: impl IntoIterator<Item = &'a str>) -> Result<Vec<KeyPair>, Vec<String>> {
    let mut slots: BTreeMap<String, (Option<String>, Option<String>)> = BTreeMap::new();
    let mut problems = Vec::new();
    for key in keys {
        let Some((target, side)) = parse_key(key) else {
            problems.push(format!("{key} : clé non reconnue (ni lora_A ni lora_B)"));
            continue;
        };
        let slot = slots.entry(target).or_default();
        let place = match side {
            Side::A => &mut slot.0,
            Side::B => &mut slot.1,
        };
        match place {
            Some(first) => problems.push(format!("{key} : doublon de {first}")),
            None => *place = Some(key.to_owned()),
        }
    }
    let mut pairs = Vec::new();
    for (target, slot) in slots {
        match slot {
            (Some(a), Some(b)) => pairs.push(KeyPair { target, a, b }),
            (Some(a), None) => problems.push(format!("{a} : lora_B absent")),
            (None, Some(b)) => problems.push(format!("{b} : lora_A absent")),
            (None, None) => {}
        }
    }
    problems.sort();
    match problems.is_empty() {
        true => Ok(pairs),
        false => Err(problems),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_both_layouts_and_rejects_the_rest() {
        let t = "model.layers.0.self_attn.q_proj.weight".to_owned();
        for key in [
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
            "model.layers.0.self_attn.q_proj.lora_A.weight",
        ] {
            assert_eq!(parse_key(key), Some((t.clone(), Side::A)), "{key}");
        }
        assert_eq!(
            parse_key("x.lora_B.weight"),
            Some(("x.weight".into(), Side::B))
        );
        assert_eq!(parse_key("x.lora_A.a.b.weight"), None);
        assert_eq!(parse_key("x.lora_A.bias"), None);
        assert_eq!(parse_key("x.weight"), None);
        assert_eq!(parse_key(".lora_A.weight"), None);
    }

    #[test]
    fn adapter_key_round_trips() {
        let key = adapter_key("a.b.weight", Side::B).unwrap();
        assert_eq!(key, "base_model.model.a.b.lora_B.weight");
        assert_eq!(parse_key(&key), Some(("a.b.weight".into(), Side::B)));
        assert_eq!(adapter_key("a.bias", Side::A), None);
    }

    #[test]
    fn orphans_are_all_listed() {
        let keys = [
            "p.lora_A.weight",
            "q.lora_B.weight",
            "r.lora_A.weight",
            "r.lora_A.default.weight",
            "junk",
        ];
        let problems = pair_keys(keys).unwrap_err();
        assert_eq!(problems.len(), 5, "{problems:?}");
        let ok = pair_keys(["z.lora_B.weight", "z.lora_A.weight"]).unwrap();
        assert_eq!(ok[0].target, "z.weight");
        assert_eq!(ok[0].a, "z.lora_A.weight");
    }
}
