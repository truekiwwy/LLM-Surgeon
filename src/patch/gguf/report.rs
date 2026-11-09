//! Human text and JSON of `apply --gguf` and `inspect` on a GGUF file (kept out of
//! `app.rs`, which only wraps them in an `Outcome`).

use std::path::Path;

use serde_json::{json, Value};

use super::backend::GgufRun;
use super::quant::Requant;
use super::{mapping, parse};
use crate::kernel;
use crate::patch::{LayerSelection, Projection};
use crate::ui::{self, Ui};

/// `86.0 MiB`, `14.96 GiB`: the binary units of `inspect`.
fn size(bytes: u64) -> String {
    let b = bytes as f64;
    match bytes {
        0..=1023 => format!("{bytes} B"),
        1024..=1_048_575 => format!("{:.1} KiB", b / 1024.0),
        1_048_576..=1_073_741_823 => format!("{:.1} MiB", b / 1_048_576.0),
        _ => format!("{:.2} GiB", b / 1_073_741_824.0),
    }
}

/// `[GGUF]` summary, per-tensor fidelity table, `[VERIFY]` line.
pub fn apply_text(ui: &Ui, model: &Path, run: &GgufRun, dry_run: bool, no_verify: bool) -> String {
    let opt = |v: Option<f64>| v.map_or("—".to_owned(), |x| format!("{x:.3}"));
    let rows: Vec<Vec<String>> = run
        .tensors
        .iter()
        .map(|t| {
            vec![
                ui::accent(ui, &t.name),
                ui::metric(ui, &t.ggml_type),
                ui::metric(ui, opt(t.realized_over_intended)),
                ui::metric(ui, opt(t.correlation_intended_realized)),
                ui::metric(ui, format!("{:.1}%", 100.0 * t.unchanged_fraction)),
                ui::metric(ui, t.saturated_elements),
                ui::metric(
                    ui,
                    t.nmse_vs_exact.map_or("—".into(), |x| format!("{x:.2e}")),
                ),
            ]
        })
        .collect();
    let requant = match run.requant {
        Requant::KeepScales => "keep-scales",
        Requant::Recompute => "recompute",
    };
    let mut text = format!(
        "{} {} · GGUF v{} · {} · requant {} · {} tenseurs\n",
        ui::badge(ui, "GGUF"),
        ui::path(ui, model),
        run.version,
        run.architecture.as_deref().unwrap_or("?"),
        ui::metric(ui, requant),
        ui::metric(ui, run.tensors.len()),
    );
    let headers = [
        "tenseur",
        "type",
        "réalisé/voulu",
        "corr",
        "inchangés",
        "saturés",
        "nmse",
    ];
    text.push_str(&ui::table(ui, &headers, &rows));
    let verify = match (dry_run, no_verify) {
        (true, _) => "dry-run : rien n'a été écrit".to_owned(),
        (false, true) => "relecture désactivée (--no-verify) ; en-tête inchangé".to_owned(),
        (false, false) => format!(
            "relecture OK : octets, échelles et valeurs finies ; en-tête inchangé (sha256 {}…)",
            run.header_sha256.get(..12).unwrap_or_default()
        ),
    };
    text.push_str(&format!(
        "{} {}\n",
        ui::badge(ui, "VERIFY"),
        ui::text(ui, &verify)
    ));
    text
}

/// `inspect` on a GGUF: MLP tensors with their quantization type (no statistics).
/// Returns `(data, human)`; `Err` is the message of an unreadable file.
pub fn inspect(
    model: &Path,
    layers: &LayerSelection,
    projections: &[Projection],
    show_stats: bool,
    ui: &Ui,
) -> Result<(Value, String), String> {
    let fail = |e: &dyn std::fmt::Display| format!("{} : {e}", model.display());
    let file = std::fs::File::open(model).map_err(|e| fail(&e))?;
    let map = kernel::map_read(&file).map_err(|e| fail(&e))?;
    let parsed = parse::parse(&map).map_err(|e| fail(&e))?;
    let mut rows = Vec::new();
    let mut tensors = Vec::new();
    for (layer, projection, t) in mapping::mlp_tensors(&parsed, layers, projections) {
        // [rows, cols] like a SafeTensors [out, in] shape.
        let shape: Vec<u64> = t.ne.iter().rev().copied().collect();
        let patchable = mapping::layout(t).is_ok();
        rows.push(vec![
            ui::accent(ui, &layer.to_string()),
            ui::metric(ui, format!("{projection:?}").to_lowercase()),
            ui::metric(ui, t.ggml_type.name()),
            ui::metric(ui, format!("{shape:?}")),
            ui::metric(ui, t.size.map_or("—".into(), size)),
            ui::dim(ui, if patchable { "oui" } else { "non" }),
        ]);
        tensors.push(json!({
            "name": t.name,
            "layer": layer,
            "projection": projection,
            "ggml_type": t.ggml_type.name(),
            "shape": shape,
            "bytes": t.size,
            "patchable": patchable,
        }));
    }
    let mut banner = Vec::new();
    let _ = ui::print_banner(ui, &mut banner);
    let mut human = String::from_utf8_lossy(&banner).into_owned();
    human.push_str(&format!(
        "{} {} tenseurs sélectionnés · GGUF v{} · {}\n",
        ui::badge(ui, "INSPECT"),
        ui::metric(ui, rows.len()),
        parsed.version,
        parsed.architecture().unwrap_or("?"),
    ));
    if show_stats {
        human.push_str(&ui::dim(ui, "statistiques non disponibles pour un GGUF\n"));
    }
    let headers = ["couche", "proj", "type", "shape", "taille", "patchable"];
    human.push_str(&ui::table(ui, &headers, &rows));
    let data = json!({
        "format": "gguf",
        "version": parsed.version,
        "architecture": parsed.architecture(),
        "alignment": parsed.alignment,
        "data_start": parsed.data_start,
        "tensor_count": parsed.tensors.len(),
        "tensors": tensors,
    });
    Ok((data, human))
}
