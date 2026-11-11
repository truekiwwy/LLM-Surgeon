//! Target selection by architecture: every tensor whose name an [`ArchSpec`] gives a role,
//! filtered by layer. The historical functions (`find_mlp_tensors`, `find_attn_out_tensors`,
//! `find_embed_tokens`) are the default spec ([`arch::legacy`], the Llama/Mistral/Qwen
//! naming), with the same results as before (`tests/arch.rs` keeps the name fixtures).

use serde::Serialize;

use super::index::{ModelIndex, TensorInfo};
use crate::arch::{self, ArchSpec, Format, Role, TensorRole};

/// Declaration order is the sort order of `find_mlp_tensors`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum MlpProj {
    Gate,
    Up,
    Down,
}

impl MlpProj {
    fn role(self) -> Role {
        match self {
            Self::Gate => Role::MlpGate,
            Self::Up => Role::MlpUp,
            Self::Down => Role::MlpOut,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum LayerSpec {
    All,
    List(Vec<usize>),
    Range { start: usize, end_inclusive: usize },
}

impl LayerSpec {
    pub fn contains(&self, layer: usize) -> bool {
        match self {
            LayerSpec::All => true,
            LayerSpec::List(layers) => layers.contains(&layer),
            LayerSpec::Range {
                start,
                end_inclusive,
            } => (*start..=*end_inclusive).contains(&layer),
        }
    }
}

/// The `{i}` in any `...layers.{i}....` tensor name.
pub fn parse_layer_index(name: &str) -> Option<usize> {
    let mut parts = name.split('.');
    while let Some(part) = parts.next() {
        if part == "layers" {
            return parts.next()?.parse().ok();
        }
    }
    None
}

/// Order inside a layer: the MLP (gate, up, out) before the attention output, as the
/// selection has always been listed; the embeddings have no layer and come first.
fn rank(role: Role) -> u8 {
    match role {
        Role::Embed => 0,
        Role::MlpGate => 1,
        Role::MlpUp => 2,
        Role::MlpOut => 3,
        Role::AttnOut => 4,
        Role::Router => 5,
    }
}

/// Tensors of `index` that `spec` gives one of `roles`, in the selected layers (a tensor
/// without layer, the embeddings, is always selected), sorted by (layer, role, expert,
/// name).
pub fn find_by_roles<'a>(
    index: &'a ModelIndex,
    spec: &ArchSpec,
    roles: &[Role],
    layers: &LayerSpec,
) -> Vec<(&'a TensorInfo, TensorRole)> {
    let mut found: Vec<(&TensorInfo, TensorRole)> = index
        .tensors
        .iter()
        .filter_map(|t| {
            let role = spec.role_of(&t.name, Format::SafeTensors)?;
            let wanted =
                roles.contains(&role.role) && role.layer.is_none_or(|l| layers.contains(l));
            wanted.then_some((t, role))
        })
        .collect();
    found.sort_by(|(a, ra), (b, rb)| {
        (ra.layer, rank(ra.role), ra.expert, &a.name).cmp(&(
            rb.layer,
            rank(rb.role),
            rb.expert,
            &b.name,
        ))
    });
    found
}

/// `[prefix.]layers.{i}.mlp.{gate|up|down}_proj.weight`, the Llama/Mistral/Qwen naming,
/// whatever the prefix (`model.`, `model.language_model.`...). MoE experts
/// (`mlp.experts.{j}.gate_proj`) deliberately do not match.
#[cfg(test)]
fn parse_mlp(name: &str) -> Option<(usize, MlpProj)> {
    let role = arch::legacy().role_of(name, Format::SafeTensors)?;
    let proj = [MlpProj::Gate, MlpProj::Up, MlpProj::Down]
        .into_iter()
        .find(|p| p.role() == role.role)?;
    Some((role.layer?, proj))
}

/// MLP weights matching the layers and projections, sorted by (layer, projection, name).
pub fn find_mlp_tensors<'a>(
    index: &'a ModelIndex,
    layers: &LayerSpec,
    projs: &[MlpProj],
) -> Vec<&'a TensorInfo> {
    let roles: Vec<Role> = projs.iter().map(|p| p.role()).collect();
    find_by_roles(index, arch::legacy(), &roles, layers)
        .into_iter()
        .map(|(t, _)| t)
        .collect()
}

/// `[prefix.]layers.{i}.self_attn.o_proj.weight`: the attention matrix that writes into the
/// residual stream, sorted by (layer, name).
pub fn find_attn_out_tensors<'a>(index: &'a ModelIndex, layers: &LayerSpec) -> Vec<&'a TensorInfo> {
    find_by_roles(index, arch::legacy(), &[Role::AttnOut], layers)
        .into_iter()
        .map(|(t, _)| t)
        .collect()
}

/// `[prefix.]embed_tokens.weight`: the input embeddings, `[vocab, hidden]`.
pub fn is_embed_tokens(name: &str) -> bool {
    arch::legacy()
        .role_of(name, Format::SafeTensors)
        .is_some_and(|r| r.role == Role::Embed)
}

pub fn find_embed_tokens(index: &ModelIndex) -> Option<&TensorInfo> {
    index.tensors.iter().find(|t| is_embed_tokens(&t.name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::index::DType;

    fn index() -> ModelIndex {
        let mut names = vec![
            "model.embed_tokens.weight".to_owned(),
            "lm_head.weight".to_owned(),
        ];
        // Reverse order on purpose: the result must not depend on file order.
        for layer in (0..4).rev() {
            for proj in ["down_proj", "up_proj", "gate_proj"] {
                names.push(format!("model.layers.{layer}.mlp.{proj}.weight"));
            }
            names.push(format!("model.layers.{layer}.self_attn.q_proj.weight"));
            names.push(format!("model.layers.{layer}.self_attn.o_proj.weight"));
            names.push(format!(
                "model.layers.{layer}.mlp.experts.0.gate_proj.weight"
            ));
        }
        names.push("model.language_model.layers.9.mlp.up_proj.weight".to_owned());
        ModelIndex {
            header_len: 0,
            tensors: names
                .into_iter()
                .map(|name| TensorInfo {
                    name,
                    dtype: DType::BF16,
                    shape: vec![1, 1],
                    byte_start: 0,
                    byte_end: 2,
                })
                .collect(),
        }
    }

    fn names(found: Vec<&TensorInfo>) -> Vec<String> {
        found.into_iter().map(|t| t.name.clone()).collect()
    }

    #[test]
    fn selects_by_layer_spec_and_projection_in_deterministic_order() {
        let idx = index();
        let all = [MlpProj::Gate, MlpProj::Up, MlpProj::Down];

        assert_eq!(
            names(find_mlp_tensors(
                &idx,
                &LayerSpec::List(vec![2, 0]),
                &[MlpProj::Down, MlpProj::Gate]
            )),
            [
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.down_proj.weight",
                "model.layers.2.mlp.gate_proj.weight",
                "model.layers.2.mlp.down_proj.weight",
            ]
        );
        assert_eq!(
            names(find_mlp_tensors(
                &idx,
                &LayerSpec::Range {
                    start: 1,
                    end_inclusive: 2
                },
                &[MlpProj::Up]
            )),
            [
                "model.layers.1.mlp.up_proj.weight",
                "model.layers.2.mlp.up_proj.weight"
            ]
        );
        let every = names(find_mlp_tensors(&idx, &LayerSpec::All, &all));
        assert_eq!(every.len(), 13); // 4 layers x 3 + the prefixed layer 9
        assert_eq!(every[0], "model.layers.0.mlp.gate_proj.weight");
        assert_eq!(
            every[12],
            "model.language_model.layers.9.mlp.up_proj.weight"
        );
        assert!(find_mlp_tensors(&idx, &LayerSpec::All, &[]).is_empty());
        assert!(find_mlp_tensors(&idx, &LayerSpec::List(vec![]), &all).is_empty());
    }

    #[test]
    fn residual_writers_outside_the_mlp() {
        let idx = index();
        assert_eq!(
            names(find_attn_out_tensors(&idx, &LayerSpec::List(vec![3, 1]))),
            [
                "model.layers.1.self_attn.o_proj.weight",
                "model.layers.3.self_attn.o_proj.weight"
            ]
        );
        assert_eq!(find_attn_out_tensors(&idx, &LayerSpec::All).len(), 4);
        assert_eq!(
            find_embed_tokens(&idx).map(|t| t.name.as_str()),
            Some("model.embed_tokens.weight")
        );
        assert!(is_embed_tokens("embed_tokens.weight"));
        assert!(!is_embed_tokens("model.embed_tokens.bias"));
        assert!(!is_embed_tokens("lm_head.weight"));
        assert!(!is_embed_tokens("model.not_embed_tokens.weight"));
    }

    #[test]
    fn layer_index_parsing() {
        assert_eq!(
            parse_layer_index("model.layers.12.self_attn.o_proj.weight"),
            Some(12)
        );
        assert_eq!(
            parse_layer_index("model.language_model.layers.3.mlp.up_proj.weight"),
            Some(3)
        );
        assert_eq!(parse_layer_index("model.norm.weight"), None);
        assert_eq!(parse_layer_index("model.layers.x.mlp"), None);
        assert_eq!(
            parse_mlp("model.layers.1.mlp.experts.0.gate_proj.weight"),
            None
        );
        assert_eq!(parse_mlp("model.layers.1.mlp.gate_proj.bias"), None);
    }
}
