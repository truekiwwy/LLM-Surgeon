//! From command line options to a [`PreviewState`] and, when the machine can hold the
//! model, the loader that the worker thread (or `--snapshot`) runs.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::worker::{load_session, SnapshotProbe};
use super::{Clock, Loader, PreviewConfig, PreviewState, DEFAULT_PROMPT};
use crate::cli::TuiArgs;
use crate::hardware::{evaluate_budget, BudgetPolicy, HardwareProbe, Verdict};
use crate::tui::app::CompanionState;
use crate::tui::{TuiError, VectorView};

pub struct Launch {
    pub state: PreviewState,
    /// `None` when the model cannot be loaded (the state says why).
    pub loader: Option<Loader>,
}

/// Injection layer: `--preview-layer`, else the layer the vector was extracted at.
fn layer_of(args: &TuiArgs, vector: &VectorView) -> Result<usize, TuiError> {
    let from_vector = match &vector.companion {
        CompanionState::Loaded(c) => Some(c.layer),
        _ => vector.metadata.get("layer").and_then(|l| l.parse().ok()),
    };
    args.preview.preview_layer.or(from_vector).ok_or_else(|| {
        TuiError::InvalidArgument(
            "couche d'injection inconnue (ni fichier compagnon ni métadonnée du vecteur) : \
                 utiliser --preview-layer"
                .into(),
        )
    })
}

/// The Qwen2 loader needs a directory (config, tokenizer, weights).
fn model_dir(path: &Path) -> PathBuf {
    match path.is_file() {
        true => path.parent().unwrap_or(path).to_path_buf(),
        false => path.to_path_buf(),
    }
}

/// The probe answers the worker thread will see, or the message to show in the tab.
fn check_hardware(model: &Path, probe: &dyn HardwareProbe) -> Result<SnapshotProbe, String> {
    let unreadable =
        |e: &dyn std::fmt::Display| format!("vérification matérielle impossible : {e}");
    let budget =
        evaluate_budget(model, probe, &BudgetPolicy::default()).map_err(|e| unreadable(&e))?;
    if let Verdict::Insufficient { reason } = budget.verdict {
        return Err(format!("ressources insuffisantes : {reason}"));
    }
    Ok(SnapshotProbe {
        ram: probe.system_ram().map_err(|e| unreadable(&e))?,
        gpus: probe.gpus().map_err(|e| unreadable(&e))?,
    })
}

/// Checks the options, then the hardware budget before anything is loaded: an
/// insufficient machine yields a state that says so, never an error for the other tabs.
pub fn prepare(
    args: &TuiArgs,
    vector: &VectorView,
    probe: &dyn HardwareProbe,
    clock: Arc<dyn Clock>,
) -> Result<Launch, TuiError> {
    let options = &args.preview;
    let (lo, hi) = options.alpha_range;
    if options.preview_tokens == 0 {
        return Err(TuiError::InvalidArgument(
            "--preview-tokens doit être au moins 1".into(),
        ));
    }
    if !(lo..=hi).contains(&options.preview_alpha) {
        return Err(TuiError::InvalidArgument(format!(
            "--preview-alpha {} hors de --alpha-range {lo}:{hi}",
            options.preview_alpha
        )));
    }
    let layer = layer_of(args, vector)?;
    let model = args
        .model
        .as_deref()
        .map(model_dir)
        .ok_or_else(|| TuiError::InvalidArgument("--preview nécessite --model".into()))?;
    let mut state = PreviewState::new(
        PreviewConfig {
            prompt: options
                .preview_prompt
                .clone()
                .unwrap_or_else(|| DEFAULT_PROMPT.to_owned()),
            alpha: options.preview_alpha,
            alpha_range: options.alpha_range,
            tokens: options.preview_tokens,
            layer,
            template: options.preview_template.chat_template(),
        },
        clock,
    );
    let snapshot = match check_hardware(&model, probe) {
        Ok(snapshot) => snapshot,
        Err(message) => {
            state.unavailable(message);
            return Ok(Launch {
                state,
                loader: None,
            });
        }
    };
    let direction = vector.values.clone();
    let tokens = options.preview_tokens;
    let loader: Loader =
        Box::new(move || load_session(&model, &snapshot, direction, layer, tokens));
    Ok(Launch {
        state,
        loader: Some(loader),
    })
}
