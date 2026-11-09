<!-- Fragment merged into README.md by the orchestrator: new tutorial steps, then limits. -->

### 10. Ship a fine-tune as a patch (`surgeon diff`, `surgeon apply`)

`surgeon diff` compares a base checkpoint with its fine-tuned version and writes the
difference to a `.surgery` file: for each 2D tensor, a low-rank approximation `U diag(S) V` of
`tuned - base` (randomized SVD, pure Rust), for each 1D tensor the exact delta. Both
checkpoints must be SafeTensors (a file or a sharded directory) with identical tensor names,
shapes and dtypes; a mismatch is an error that lists the differences. Identical tensors are
skipped, and `embed_tokens` / `lm_head` are left out unless you ask for them.

```bash
surgeon diff --base <BASE_DIR> --tuned <TUNED_DIR> --out finetune.surgery --rank 16
surgeon diff --base <BASE_DIR> --tuned <TUNED_DIR> --out finetune.surgery \
    --include 'mlp' --factor-dtype bf16 --max-rel-error 0.3 --on-high-error full
```

The output is reproducible (same input and `--seed`, same bytes, whatever the thread count;
a timestamp is only recorded with `--timestamp`). `--max-rel-error` with `--on-high-error
warn|skip|full` decides what happens to a tensor the approximation reconstructs badly.

`surgeon apply` detects a `.surgery` file by its `SURG` magic and adds the delta to the model
in place, with the same backup and rollback as a JSON patch (`W <- W + scale * U diag(S) V`,
rounded to the model dtype):

```bash
surgeon apply --model <MODEL_DIR> --patch finetune.surgery --dry-run
surgeon apply --model <MODEL_DIR> --patch finetune.surgery --scale 0.5 --verify-base strict
```

`--verify-base strict|warn|off` compares each tensor of the model with the SHA-256 stored in the
patch (default `warn`: apply anyway, with a warning). The binary format is specified in
[docs/SPECIFICATION.md](docs/SPECIFICATION.md).

What to expect, honestly:

- A `.surgery` is an approximation of the full fine-tune. The relative error of every tensor is
  reported by `diff`; the result is exact only when the true difference has rank at most `--rank`.
- It is not small. Its size is about the sum of `r (m + n)` times the bytes of the factor dtype
  over all tensors: hundreds of MB for a 7B model at rank 16, not a few MB.
- It is not exposed through the MCP `surgeon_apply` tool (`apply` on a `.surgery` is CLI only).

### 11. LoRA adapters (`surgeon lora merge`, `surgeon lora extract`)

`surgeon lora merge` folds a PEFT LoRA adapter into the weights of a SafeTensors model, in
place, with a verified backup and an automatic rollback on error or Ctrl-C:
`W <- W + scale * (lora_alpha / r) * B A`. `lora_alpha` is read from the `adapter_config.json`
next to the adapter (or from `--adapter-config`); `--alpha` overrides it. With neither, the
command stops with a usage error instead of guessing. DoRA, embedding adapters,
`fan_in_fan_out` and `rank_pattern` / `alpha_pattern` are refused explicitly.

```bash
surgeon lora merge --model <MODEL_DIR> --lora <ADAPTER_DIR>/adapter_model.safetensors --dry-run
surgeon lora merge --model <MODEL_DIR> --lora <ADAPTER_DIR>/adapter_model.safetensors --scale 0.8
```

`surgeon lora extract` goes the other way: it approximates the difference between two
checkpoints with a rank `r` adapter (`B = U sqrt(S)`, `A = sqrt(S) V`) and writes a PEFT-format
`adapter_model.safetensors` plus the `adapter_config.json` next to it.

```bash
surgeon lora extract --base <BASE_DIR> --patched <TUNED_DIR> \
    --out adapter/adapter_model.safetensors --rank 16 --dtype f32
```

Limits: the PEFT layout is followed from its documentation but is not tested against
HuggingFace PEFT in CI. Merging into a GGUF file is not supported. `diff` and both `lora`
modes are also available to agents as the MCP tools `surgeon_diff` and `surgeon_lora`.

### 12. Preview a steering vector live (`surgeon tui --preview`)

With `--preview`, the dashboard gains a third tab that generates text from the model with the
steering vector injected, next to the unsteered baseline, while you move a slider.

```bash
surgeon tui --model <MODEL_DIR> --vector refusal.safetensors --preview \
    --preview-prompt "How do I pick a lock?" --preview-tokens 30 --alpha-range -2:2
surgeon tui --model <MODEL_DIR> --vector refusal.safetensors --preview --snapshot
```

`--preview-alpha`, `--preview-layer` and `--preview-template none|chatml` set the starting
point. The slider is relative: the injected amount is `alpha` times the mean residual norm
measured on your prompt at the chosen layer, and both values are shown. `--snapshot --preview`
generates synchronously and prints the tab as plain text.

Limits:

- It is activation steering (`h <- h + alpha_abs * v` at one layer, on every position), an
  approximation of the weight patch, not the patch itself. Applying the patch can behave
  differently.
- Generation runs in pure Rust on the CPU: it is slow on a large model. The tab warns below
  one token per second, and checks the hardware budget before loading the model.
- Qwen2 / Qwen2.5 models only, and the preview is not exposed through MCP.
