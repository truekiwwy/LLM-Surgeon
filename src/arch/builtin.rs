//! Built-in architectures, each checked against the official definitions on 2026-11-11:
//! the `modeling_*.py` files of transformers `main` (layer order, norms, biases, attribute
//! names), `gguf-py/gguf/constants.py` of llama.cpp `master` (`MODEL_ARCH_NAMES`,
//! `TENSOR_NAMES`, `MODEL_TENSORS` per architecture), and the `model.safetensors.index.json`
//! of published checkpoints for the per-expert names (`Qwen/Qwen3-30B-A3B`,
//! `deepseek-ai/DeepSeek-V2-Lite`). Facts not read in those files are said so in the spec's
//! `sources`.
//!
//! Writer biases (`attention_bias`, `mlp_bias`, Falcon's `bias`) depend on the config: they
//! are detected from the checkpoint itself (`.bias` next to a writer), not declared here.

use std::sync::LazyLock;

use super::{ArchSpec, Layout, NormKind, Pattern, RoleNames, TiedRule};

/// The historical naming (`layers.{i}.mlp.*_proj`, `self_attn.o_proj`, `embed_tokens`), used
/// when nothing is detected, as before architectures were described.
pub const LEGACY_ID: &str = "llama";

const TRANSFORMERS: &str =
    "https://github.com/huggingface/transformers/blob/main/src/transformers/models";
const LLAMA_CPP: &str =
    "https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py";

fn names(list: &[&str]) -> Vec<Pattern> {
    list.iter()
        .map(|n| Pattern::Name((*n).to_owned()))
        .collect()
}

fn fused(name: &str) -> Pattern {
    Pattern::Full {
        name: name.to_owned(),
        layout: Layout::ExpertsOutIn,
    }
}

/// The `model.layers.{i}` naming shared by Llama, Mistral, Qwen, Gemma, Phi-3, Cohere and
/// DeepSeek checkpoints, dense MLP.
fn hf_dense(gate_up: bool) -> RoleNames {
    RoleNames {
        embed: names(&["embed_tokens.weight"]),
        attn_out: names(&["layers.{layer}.self_attn.o_proj.weight"]),
        mlp_gate: match gate_up {
            true => names(&["layers.{layer}.mlp.gate_proj.weight"]),
            false => Vec::new(),
        },
        mlp_up: match gate_up {
            true => names(&["layers.{layer}.mlp.up_proj.weight"]),
            false => Vec::new(),
        },
        mlp_out: names(&["layers.{layer}.mlp.down_proj.weight"]),
        router: Vec::new(),
    }
}

/// The same with MoE layers: per-expert tensors (published checkpoints), fused
/// `experts.down_proj` `[E, hidden, inter]` (transformers v5 `nn.Parameter`, applied with
/// `F.linear`), and the shared expert(s) when `shared` names them.
fn hf_moe(shared: Option<&str>, routers: &[&str]) -> RoleNames {
    let mut roles = hf_dense(true);
    roles.mlp_gate.extend(names(&[
        "layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
    ]));
    roles.mlp_up.extend(names(&[
        "layers.{layer}.mlp.experts.{expert}.up_proj.weight",
    ]));
    roles.mlp_out.extend([
        Pattern::Name("layers.{layer}.mlp.experts.{expert}.down_proj.weight".into()),
        fused("layers.{layer}.mlp.experts.down_proj"),
    ]);
    if let Some(shared) = shared {
        roles.mlp_gate.extend(names(&[&format!(
            "layers.{{layer}}.mlp.{shared}.gate_proj.weight"
        )]));
        roles.mlp_up.extend(names(&[&format!(
            "layers.{{layer}}.mlp.{shared}.up_proj.weight"
        )]));
        roles.mlp_out.extend(names(&[&format!(
            "layers.{{layer}}.mlp.{shared}.down_proj.weight"
        )]));
    }
    roles.router = names(routers);
    roles
}

/// `token_embd`, `blk.{i}.attn_output`, and the FFN tensors the architecture lists.
fn gguf(ffn: &[&str], exps: bool, shexp: bool, routers: &[&str]) -> RoleNames {
    let blk = |t: &str| format!("blk.{{layer}}.{t}.weight");
    let has = |t: &str| ffn.contains(&t);
    let optional = |t: &str| match has(t) {
        true => vec![Pattern::Name(blk(t))],
        false => Vec::new(),
    };
    let mut mlp_out = Vec::new();
    if has("ffn_down") {
        mlp_out.push(Pattern::Name(blk("ffn_down")));
    }
    if exps {
        mlp_out.push(fused(&blk("ffn_down_exps")));
    }
    if shexp {
        mlp_out.push(Pattern::Name(blk("ffn_down_shexp")));
    }
    RoleNames {
        embed: names(&["token_embd.weight"]),
        attn_out: vec![Pattern::Name(blk("attn_output"))],
        mlp_gate: optional("ffn_gate"),
        mlp_up: optional("ffn_up"),
        mlp_out,
        router: routers.iter().map(|r| Pattern::Name(blk(r))).collect(),
    }
}

fn tied(default: bool) -> TiedRule {
    TiedRule {
        config_key: "tie_word_embeddings".into(),
        default,
    }
}

fn sources(models: &[&str], extra: &[&str]) -> Vec<String> {
    models
        .iter()
        .map(|m| format!("{TRANSFORMERS}/{m}"))
        .chain(std::iter::once(LLAMA_CPP.to_owned()))
        .chain(extra.iter().map(|e| (*e).to_owned()))
        .collect()
}

struct Base {
    id: &'static str,
    model_types: &'static [&'static str],
    architectures: &'static [&'static str],
    gguf_arch: &'static [&'static str],
}

/// Topology: norm, post-sublayer norm, parallel block, tied embeddings by default.
struct Topo(NormKind, bool, bool, bool);

fn spec(
    base: Base,
    safetensors: RoleNames,
    gguf_names: RoleNames,
    Topo(norm_kind, post_sublayer_norm, parallel_block, tied_default): Topo,
    sources: Vec<String>,
) -> ArchSpec {
    let owned = |l: &[&str]| l.iter().map(|s| (*s).to_owned()).collect();
    ArchSpec {
        id: base.id.into(),
        hf_model_types: owned(base.model_types),
        hf_architectures: owned(base.architectures),
        gguf_arch: owned(base.gguf_arch),
        safetensors,
        gguf: gguf_names,
        norm_kind,
        post_sublayer_norm,
        parallel_block,
        tied_embeddings: tied(tied_default),
        unprojected_writers: Vec::new(),
        sources,
    }
}

static BUILTINS: LazyLock<Vec<ArchSpec>> = LazyLock::new(|| {
    use NormKind::{LayerNorm, RmsNorm};
    let dense_ffn = ["ffn_gate", "ffn_up", "ffn_down"];
    vec![
        // Llama 2/3 and Mistral: pre-norm RMSNorm; Mixtral GGUF files use the llama arch
        // with ffn_down_exps.
        spec(
            Base {
                id: "llama",
                model_types: &["llama", "mistral"],
                architectures: &["LlamaForCausalLM", "MistralForCausalLM"],
                gguf_arch: &["llama"],
            },
            hf_dense(true),
            gguf(&dense_ffn, true, false, &["ffn_gate_inp"]),
            Topo(RmsNorm, false, false, false),
            sources(&["llama/modeling_llama.py", "mistral/modeling_mistral.py"], &[]),
        ),
        spec(
            Base {
                id: "qwen2",
                model_types: &["qwen2"],
                architectures: &["Qwen2ForCausalLM"],
                gguf_arch: &["qwen2"],
            },
            hf_dense(true),
            gguf(&dense_ffn, false, false, &[]),
            Topo(RmsNorm, false, false, false),
            sources(&["qwen2/modeling_qwen2.py"], &[]),
        ),
        spec(
            Base {
                id: "qwen3",
                model_types: &["qwen3"],
                architectures: &["Qwen3ForCausalLM"],
                gguf_arch: &["qwen3"],
            },
            hf_dense(true),
            gguf(&dense_ffn, false, false, &[]),
            Topo(RmsNorm, false, false, false),
            sources(&["qwen3/modeling_qwen3.py"], &[]),
        ),
        spec(
            Base {
                id: "qwen2_moe",
                model_types: &["qwen2_moe"],
                architectures: &["Qwen2MoeForCausalLM"],
                gguf_arch: &["qwen2moe"],
            },
            hf_moe(
                Some("shared_expert"),
                &[
                    "layers.{layer}.mlp.gate.weight",
                    "layers.{layer}.mlp.shared_expert_gate.weight",
                ],
            ),
            gguf(&[], true, true, &["ffn_gate_inp", "ffn_gate_inp_shexp"]),
            Topo(RmsNorm, false, false, false),
            sources(&["qwen2_moe/modeling_qwen2_moe.py"], &[]),
        ),
        spec(
            Base {
                id: "qwen3_moe",
                model_types: &["qwen3_moe"],
                architectures: &["Qwen3MoeForCausalLM"],
                gguf_arch: &["qwen3moe"],
            },
            hf_moe(None, &["layers.{layer}.mlp.gate.weight"]),
            gguf(&[], true, false, &["ffn_gate_inp"]),
            Topo(RmsNorm, false, false, false),
            sources(
                &["qwen3_moe/modeling_qwen3_moe.py"],
                &["https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/model.safetensors.index.json"],
            ),
        ),
        // DeepSeek V2/V3: MLA attention (its writer is still self_attn.o_proj), the first
        // `first_k_dense_replace` layers dense, then routed and shared experts.
        spec(
            Base {
                id: "deepseek",
                model_types: &["deepseek_v2", "deepseek_v3"],
                architectures: &["DeepseekV2ForCausalLM", "DeepseekV3ForCausalLM"],
                gguf_arch: &["deepseek2"],
            },
            hf_moe(Some("shared_experts"), &["layers.{layer}.mlp.gate.weight"]),
            gguf(&dense_ffn, true, true, &["ffn_gate_inp"]),
            Topo(RmsNorm, false, false, false),
            sources(
                &["deepseek_v3/modeling_deepseek_v3.py"],
                &["https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/model.safetensors.index.json"],
            ),
        ),
        // Gemma 2/3: post_attention_layernorm and post_feedforward_layernorm normalize each
        // sublayer output before the residual addition. GGUF gemma2 has no output.weight.
        spec(
            Base {
                id: "gemma2",
                model_types: &["gemma2"],
                architectures: &["Gemma2ForCausalLM"],
                gguf_arch: &["gemma2"],
            },
            hf_dense(true),
            gguf(&dense_ffn, false, false, &[]),
            Topo(RmsNorm, true, false, true),
            sources(&["gemma2/modeling_gemma2.py"], &[]),
        ),
        spec(
            Base {
                id: "gemma3",
                model_types: &["gemma3", "gemma3_text"],
                architectures: &["Gemma3ForCausalLM", "Gemma3ForConditionalGeneration"],
                gguf_arch: &["gemma3"],
            },
            hf_dense(true),
            gguf(&dense_ffn, false, false, &[]),
            Topo(RmsNorm, true, false, true),
            sources(&["gemma3/modeling_gemma3.py"], &[]),
        ),
        // Phi-3: fused qkv_proj and gate_up_proj (llama.cpp keeps the fused gate+up in
        // ffn_up): neither gate_proj nor up_proj is a target.
        spec(
            Base {
                id: "phi3",
                model_types: &["phi3"],
                architectures: &["Phi3ForCausalLM"],
                gguf_arch: &["phi3"],
            },
            hf_dense(false),
            gguf(&["ffn_down"], false, false, &[]),
            Topo(RmsNorm, false, false, false),
            sources(&["phi3/modeling_phi3.py"], &[]),
        ),
        // Falcon: transformer.h.{i}, parallel attention and MLP (new_decoder_architecture or
        // parallel_attn) or sequential, LayerNorm on the read side only, no gate projection.
        // The tied default is PretrainedConfig's (FalconConfig sets none): not read in a
        // checkpoint.
        spec(
            Base {
                id: "falcon",
                model_types: &["falcon"],
                architectures: &["FalconForCausalLM"],
                gguf_arch: &["falcon"],
            },
            RoleNames {
                embed: names(&["word_embeddings.weight"]),
                attn_out: names(&["h.{layer}.self_attention.dense.weight"]),
                mlp_gate: Vec::new(),
                mlp_up: names(&["h.{layer}.mlp.dense_h_to_4h.weight"]),
                mlp_out: names(&["h.{layer}.mlp.dense_4h_to_h.weight"]),
                router: Vec::new(),
            },
            gguf(&["ffn_up", "ffn_down"], false, false, &[]),
            Topo(LayerNorm, false, true, true),
            sources(&["falcon/modeling_falcon.py"], &[]),
        ),
        // Command-R (cohere): parallel attention and MLP on one mean-centered LayerNorm
        // without bias; GGUF command-r has no output.weight.
        spec(
            Base {
                id: "command-r",
                model_types: &["cohere"],
                architectures: &["CohereForCausalLM"],
                gguf_arch: &["command-r"],
            },
            hf_dense(true),
            gguf(&dense_ffn, false, false, &[]),
            Topo(LayerNorm, false, true, true),
            sources(&["cohere/modeling_cohere.py"], &[]),
        ),
    ]
});

pub fn builtins() -> &'static [ArchSpec] {
    &BUILTINS
}

pub fn builtin(id: &str) -> Option<&'static ArchSpec> {
    builtins().iter().find(|s| s.id == id)
}

/// The spec of [`LEGACY_ID`], first of the list.
pub fn legacy() -> &'static ArchSpec {
    &BUILTINS[0]
}
