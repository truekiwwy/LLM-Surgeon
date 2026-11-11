//! The role taxonomy, shared with LLM-Surgeon's `arch` module: which tensors
//! write into the residual stream (embed, attention out, MLP out — dense,
//! fused-expert or shared-expert) and which only read from it (gate, up,
//! router). Patterns match the END of a tensor name, `{layer}` and `{expert}`
//! capture a decimal segment, exactly like `src/arch` describes them.

use crate::model::TensorEntry;
pub use crate::model::MoeKind;
use crate::model::Format;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Role {
    Embed,
    AttnOut,
    MlpGate,
    MlpUp,
    MlpOut,
    Router,
    Other,
}

impl Role {
    pub const ALL: [Role; 7] = [
        Role::Embed,
        Role::AttnOut,
        Role::MlpGate,
        Role::MlpUp,
        Role::MlpOut,
        Role::Router,
        Role::Other,
    ];

    /// Short label shown in the UI and accepted by the `set_filter` IPC command.
    pub fn label(self) -> &'static str {
        match self {
            Role::Embed => "embed",
            Role::AttnOut => "o_proj",
            Role::MlpGate => "gate",
            Role::MlpUp => "up",
            Role::MlpOut => "down",
            Role::Router => "router",
            Role::Other => "other",
        }
    }

    /// Parses a filter word (`embed`, `gate`, `up`, `down`, `o_proj`, `router`).
    pub fn from_filter(word: &str) -> Option<Role> {
        Role::ALL.iter().copied().find(|r| r.label() == word)
    }

    /// Writers feed the residual stream: they are the ones `project_out` touches.
    pub fn is_writer(self) -> bool {
        matches!(self, Role::Embed | Role::AttnOut | Role::MlpOut)
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Classification of one tensor name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Classified {
    pub role: Role,
    pub layer: Option<u32>,
    pub expert: Option<u32>,
    pub moe: MoeKind,
}

/// Patterns per role: `HF` names (SafeTensors) and `GGUF` names.
/// `{layer}` / `{expert}` capture a decimal segment; every pattern matches a
/// suffix of the dot-separated name, so any `model.` / `transformer.` prefix is
/// accepted, like LLM-Surgeon's `arch` specs.
struct Patterns {
    embed: &'static [&'static str],
    router: &'static [&'static str],
    attn_out: &'static [&'static str],
    mlp_gate: &'static [&'static str],
    mlp_up: &'static [&'static str],
    fused_gate: &'static [&'static str],
    fused_up: &'static [&'static str],
    shared_gate: &'static [&'static str],
    shared_up: &'static [&'static str],
    mlp_out: &'static [&'static str],
    fused_out: &'static [&'static str],
    shared_out: &'static [&'static str],
}

const HF: Patterns = Patterns {
    embed: &["embed_tokens.weight", "token_embd.weight", "wte.weight", "word_embeddings.weight"],
    router: &["layers.{layer}.mlp.gate.weight", "block_sparse_moe.gate.weight"],
    attn_out: &[
        "layers.{layer}.self_attn.o_proj.weight",
        "layers.{layer}.attention.wo.weight",
        "h.{layer}.self_attention.dense.weight",
    ],
    mlp_gate: &[
        "layers.{layer}.mlp.gate_proj.weight",
        "layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
    ],
    mlp_up: &[
        "layers.{layer}.mlp.up_proj.weight",
        "layers.{layer}.mlp.experts.{expert}.up_proj.weight",
    ],
    fused_gate: &["layers.{layer}.mlp.experts.gate_proj", "layers.{layer}.mlp.experts.gate"],
    fused_up: &["layers.{layer}.mlp.experts.up_proj", "layers.{layer}.mlp.experts.up"],
    shared_gate: &["layers.{layer}.mlp.shared_experts.gate_proj.weight"],
    shared_up: &["layers.{layer}.mlp.shared_experts.up_proj.weight"],
    mlp_out: &[
        "layers.{layer}.mlp.down_proj.weight",
        "layers.{layer}.mlp.experts.{expert}.down_proj.weight",
        "h.{layer}.mlp.dense_4h_to_h.weight",
    ],
    fused_out: &[
        "layers.{layer}.mlp.experts.down_proj.weight",
        "layers.{layer}.mlp.experts.down_proj",
        "layers.{layer}.mlp.experts.down",
    ],
    shared_out: &["layers.{layer}.mlp.shared_experts.down_proj.weight"],
};

const GGUF: Patterns = Patterns {
    embed: &["token_embd.weight", "output.weight"],
    router: &["blk.{layer}.ffn_gate_inp.weight"],
    attn_out: &["blk.{layer}.attn_output.weight"],
    mlp_gate: &["blk.{layer}.ffn_gate.weight"],
    mlp_up: &["blk.{layer}.ffn_up.weight"],
    fused_gate: &["blk.{layer}.ffn_gate_exps.weight", "blk.{layer}.ffn_gate_shexp.weight"],
    fused_up: &["blk.{layer}.ffn_up_exps.weight", "blk.{layer}.ffn_up_shexp.weight"],
    shared_gate: &[],
    shared_up: &[],
    mlp_out: &["blk.{layer}.ffn_down.weight"],
    fused_out: &["blk.{layer}.ffn_down_exps.weight", "blk.{layer}.ffn_down_shexp.weight"],
    shared_out: &[],
};

/// Classifies a tensor name for one on-disk format.
pub fn classify(name: &str, format: Format) -> Classified {
    let patterns = match format {
        Format::SafeTensors => &HF,
        Format::Gguf => &GGUF,
    };
    let (role, moe) = pick(patterns, name);
    let (layer, expert) = captures(name);
    Classified { role, layer, expert, moe }
}

fn pick(p: &Patterns, name: &str) -> (Role, MoeKind) {
    let order: [(Role, &[&str], MoeKind); 13] = [
        (Role::Embed, p.embed, MoeKind::NotMoe),
        (Role::Router, p.router, MoeKind::Router),
        (Role::AttnOut, p.attn_out, MoeKind::NotMoe),
        (Role::MlpOut, p.shared_out, MoeKind::Shared),
        (Role::MlpGate, p.shared_gate, MoeKind::Shared),
        (Role::MlpUp, p.shared_up, MoeKind::Shared),
        (Role::MlpOut, p.fused_out, MoeKind::Fused),
        (Role::MlpGate, p.fused_gate, MoeKind::Fused),
        (Role::MlpUp, p.fused_up, MoeKind::Fused),
        (Role::MlpOut, p.mlp_out, MoeKind::Dense),
        (Role::MlpGate, p.mlp_gate, MoeKind::Dense),
        (Role::MlpUp, p.mlp_up, MoeKind::Dense),
        (Role::Other, &[], MoeKind::NotMoe),
    ];
    for (role, patterns, moe) in order {
        if patterns.iter().any(|pat| match_suffix(pat, name).is_some()) {
            return (role, moe);
        }
    }
    (Role::Other, MoeKind::NotMoe)
}

/// Matches a `a.b.{layer}.c` pattern against the end of `name`; returns the
/// captured `(layer, expert)` when the pattern wins.
pub fn match_suffix(pattern: &str, name: &str) -> Option<(Option<u32>, Option<u32>)> {
    let pat: Vec<&str> = pattern.split('.').collect();
    let segs: Vec<&str> = name.split('.').collect();
    if pat.len() > segs.len() {
        return None;
    }
    let mut layer = None;
    let mut expert = None;
    for (p, s) in pat.iter().rev().zip(segs.iter().rev()) {
        match *p {
            "{layer}" => {
                let value = s.parse::<u32>().ok()?;
                layer = Some(value);
            }
            "{expert}" => {
                let value = s.parse::<u32>().ok()?;
                expert = Some(value);
            }
            other if other.eq_ignore_ascii_case(s) => {}
            _ => return None,
        }
    }
    Some((layer, expert))
}

/// `(layer, expert)` read directly from the name segments, whatever the role.
fn captures(name: &str) -> (Option<u32>, Option<u32>) {
    let segs: Vec<&str> = name.split('.').collect();
    let after = |key: &str| -> Option<u32> {
        segs.iter()
            .zip(segs.iter().skip(1))
            .find(|(s, _)| *s == &key)
            .and_then(|(_, n)| n.parse().ok())
    };
    let layer = ["layers", "blk", "h", "block"].iter().find_map(|k| after(k));
    let expert = after("experts");
    (layer, expert)
}

/// Whether an ablation by weights (`project_out` on every writer) is exact for
/// this architecture family: pre-norm dense or MoE transformers erase exactly,
/// architectures that apply a learned norm between a sublayer and the residual
/// addition (Gemma 2/3 post-norms) only approximately, unknown families report
/// unsupported. Mirrors the topology table of LLM-Surgeon's `arch` module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exactness {
    Exact,
    Approximate,
    Unsupported,
}

impl Exactness {
    pub fn of_family(family: Option<&str>) -> Exactness {
        match family {
            Some("llama" | "mistral" | "qwen2" | "qwen3" | "qwen3_moe" | "mixtral" | "llama4") => {
                Exactness::Exact
            }
            Some("gemma" | "gemma2" | "gemma3") => Exactness::Approximate,
            _ => Exactness::Unsupported,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Exactness::Exact => "Exact",
            Exactness::Approximate => "Approximate",
            Exactness::Unsupported => "Unsupported",
        }
    }

    /// The exactness of ablating one entry: only writers carry the badge.
    pub fn for_entry(family: Option<&str>, entry: &TensorEntry) -> Option<Exactness> {
        if !entry.role.is_writer() {
            return None;
        }
        Some(Exactness::of_family(family))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelIndex;

    #[test]
    fn hf_names_classify_by_role() {
        for (name, role, layer) in [
            ("model.embed_tokens.weight", Role::Embed, None),
            ("model.layers.3.self_attn.o_proj.weight", Role::AttnOut, Some(3)),
            ("model.layers.3.mlp.gate_proj.weight", Role::MlpGate, Some(3)),
            ("model.layers.3.mlp.up_proj.weight", Role::MlpUp, Some(3)),
            ("model.layers.3.mlp.down_proj.weight", Role::MlpOut, Some(3)),
            ("model.layers.3.mlp.experts.7.down_proj.weight", Role::MlpOut, Some(3)),
            ("model.layers.3.mlp.shared_experts.down_proj.weight", Role::MlpOut, Some(3)),
            ("model.layers.3.mlp.gate.weight", Role::Router, Some(3)),
            ("model.layers.3.self_attn.q_proj.weight", Role::Other, Some(3)),
            ("model.norm.weight", Role::Other, None),
        ] {
            let c = classify(name, Format::SafeTensors);
            assert_eq!(c.role, role, "{name}");
            assert_eq!(c.layer, layer, "{name}");
        }
        let expert = classify("model.layers.3.mlp.experts.7.gate_proj.weight", Format::SafeTensors);
        assert_eq!(expert.expert, Some(7));
        assert_eq!(expert.moe, MoeKind::Dense);
        let fused = classify("model.layers.3.mlp.experts.down_proj.weight", Format::SafeTensors);
        assert_eq!(fused.role, Role::MlpOut);
        assert_eq!(fused.moe, MoeKind::Fused);
        let shared = classify("model.layers.3.mlp.shared_experts.gate_proj.weight", Format::SafeTensors);
        assert_eq!(shared.role, Role::MlpGate);
        assert_eq!(shared.moe, MoeKind::Shared);
    }

    #[test]
    fn gguf_names_classify_by_role() {
        for (name, role) in [
            ("token_embd.weight", Role::Embed),
            ("blk.14.attn_output.weight", Role::AttnOut),
            ("blk.14.ffn_gate.weight", Role::MlpGate),
            ("blk.14.ffn_up.weight", Role::MlpUp),
            ("blk.14.ffn_down.weight", Role::MlpOut),
            ("blk.14.ffn_gate_inp.weight", Role::Router),
            ("blk.14.ffn_down_exps.weight", Role::MlpOut),
            ("blk.14.ffn_down_shexp.weight", Role::MlpOut),
            ("blk.14.attn_q.weight", Role::Other),
        ] {
            let c = classify(name, Format::Gguf);
            assert_eq!(c.role, role, "{name}");
            if name.starts_with("blk.") {
                assert_eq!(c.layer, Some(14), "{name}");
            }
        }
        let fused = classify("blk.14.ffn_down_exps.weight", Format::Gguf);
        assert_eq!(fused.moe, MoeKind::Fused);
    }

    #[test]
    fn prefix_free_and_digit_gated() {
        assert!(match_suffix("layers.{layer}.mlp.down_proj.weight", "model.layers.9.mlp.down_proj.weight").is_some());
        // A non-decimal segment never captures {layer}.
        assert!(match_suffix("layers.{layer}.mlp.down_proj.weight", "x.layers.foo.mlp.down_proj.weight").is_none());
        // A shorter name does not match a longer pattern.
        assert!(match_suffix("layers.{layer}.mlp.down_proj.weight", "mlp.down_proj.weight").is_none());
    }

    #[test]
    fn filters_and_exactness() {
        assert_eq!(Role::from_filter("gate"), Some(Role::MlpGate));
        assert_eq!(Role::from_filter("o_proj"), Some(Role::AttnOut));
        assert_eq!(Role::from_filter("nope"), None);
        assert_eq!(Exactness::of_family(Some("qwen3_moe")), Exactness::Exact);
        assert_eq!(Exactness::of_family(Some("gemma3")), Exactness::Approximate);
        assert_eq!(Exactness::of_family(Some("phi3")), Exactness::Unsupported);
        assert_eq!(Exactness::of_family(None), Exactness::Unsupported);
    }

    #[test]
    fn the_index_keeps_roles_sorted_and_findable() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("m.safetensors");
        let tensors = vec![
            ("model.layers.0.mlp.down_proj.weight".to_string(), vec![2u64, 2], super::super::dtype::DType::F32, 16u64, path.clone(), 0u64),
            ("model.layers.0.mlp.up_proj.weight".to_string(), vec![2u64, 2], super::super::dtype::DType::F32, 16u64, path.clone(), 16u64),
            ("model.embed_tokens.weight".to_string(), vec![2u64, 2], super::super::dtype::DType::F32, 16u64, path.clone(), 32u64),
        ];
        let index = ModelIndex::finish(dir.path(), Format::SafeTensors, None, tensors);
        assert_eq!(index.tensors[0].name, "model.embed_tokens.weight");
        assert_eq!(index.roles_present(), vec![Role::Embed, Role::MlpUp, Role::MlpOut]);
        assert_eq!(index.first_of_layer(0), Some(1));
        assert_eq!(index.index_of("model.layers.0.mlp.up_proj.weight"), Some(2));
        assert_eq!(index.num_layers, 1);
    }
}
