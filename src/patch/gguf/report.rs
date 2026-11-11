//! Human text and JSON of `apply --gguf` and `inspect` on a GGUF file (kept out of
//! `app.rs`, which only wraps them in an `Outcome`).

use std::path::Path;

use serde_json::{json, Value};

use super::backend::GgufRun;
use super::quant::Requant;
use super::{mapping, parse};
use crate::arch::{self, ArchSpec, Format};
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
                ui::metric(ui, opt(t.residual_fraction)),
                ui::metric(ui, t.refine_passes.map_or("—".into(), |n| n.to_string())),
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
        "résiduel",
        "passes",
    ];
    text.push_str(&ui::table(ui, &headers, &rows));
    let p = &run.preflight;
    text.push_str(&format!(
        "{} {} tenseurs, {} ciblés, {} à traiter{}{}
",
        ui::badge(ui, "PRÉ-VOL"),
        ui::metric(ui, p.tensors),
        ui::metric(ui, size(p.target_bytes)),
        ui::metric(ui, size(p.work_bytes)),
        p.estimated_seconds.map_or(String::new(), |s| format!(
            ", ~{} estimées",
            ui::metric(ui, format!("{s:.1} s"))
        )),
        p.backup_bytes.map_or(String::new(), |b| format!(
            ", backup {}",
            ui::metric(ui, size(b))
        )),
    ));
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

/// The `proj` column of `inspect`.
fn short_name(projection: Projection) -> &'static str {
    match projection {
        Projection::Gate => "gate",
        Projection::Up => "up",
        Projection::Down => "down",
        Projection::AttnOut => "o_proj",
        Projection::EmbedTokens => "embed_tokens",
    }
}

/// `inspect` on a GGUF: patch targets (the MLP unless `projections` says otherwise) with
/// their quantization type, orientation and whether they can be patched (no statistics).
/// Returns `(data, human)`; `Err` is the message of an unreadable file.
pub fn inspect(
    model: &Path,
    layers: &LayerSelection,
    projections: &[Projection],
    show_stats: bool,
    arch_spec: Option<&ArchSpec>,
    ui: &Ui,
) -> Result<(Value, String), String> {
    let fail = |e: &dyn std::fmt::Display| format!("{} : {e}", model.display());
    let file = std::fs::File::open(model).map_err(|e| fail(&e))?;
    let map = kernel::map_read(&file).map_err(|e| fail(&e))?;
    let parsed = parse::parse(&map).map_err(|e| fail(&e))?;
    // The MLP by default, as the SafeTensors inspect; o_proj and embed_tokens on request.
    let projections = match projections.is_empty() {
        true => &[Projection::Gate, Projection::Up, Projection::Down][..],
        false => projections,
    };
    let tied = mapping::tied_embeddings(&parsed);
    let resolution = match arch_spec {
        Some(spec) => arch::Resolution {
            spec: spec.clone(),
            source: arch::Source::ArchSpecFile,
            declared: parsed.architecture().map(str::to_owned),
        },
        None => arch::detect_gguf(parsed.architecture()),
    };
    let report = arch::Report::new(
        &resolution,
        parsed.tensors.iter().map(|t| t.name.as_str()),
        Format::Gguf,
        Some(tied),
    );
    let mut rows = Vec::new();
    let mut tensors = Vec::new();
    for (role, projection, t) in mapping::targets_by(&resolution.spec, &parsed, layers, projections)
    {
        // [rows, cols] like a SafeTensors [out, in] shape.
        let shape: Vec<u64> = t.ne.iter().rev().copied().collect();
        let orientation = mapping::Orientation::of(role.role);
        let layout = mapping::layout_as(t, role.layout);
        let reason = match &layout {
            Err(e) => Some(e.to_string()),
            Ok(_) if tied && role.role == arch::Role::Embed => {
                Some("lié à la tête de sortie : --allow-tied-embedding".to_owned())
            }
            Ok(_) => None,
        };
        let patchable = layout.is_ok();
        let layer = role.layer;
        rows.push(vec![
            ui::accent(ui, &layer.map_or("—".to_owned(), |l| l.to_string())),
            ui::metric(ui, short_name(projection)),
            ui::metric(ui, t.ggml_type.name()),
            ui::metric(ui, json!(orientation).as_str().unwrap_or_default()),
            ui::metric(ui, format!("{shape:?}")),
            ui::metric(ui, t.size.map_or("—".into(), size)),
            ui::dim(
                ui,
                &match (patchable, &reason) {
                    (true, None) => "oui".to_owned(),
                    (true, Some(r)) => format!("oui ({r})"),
                    (false, r) => format!("non ({})", r.as_deref().unwrap_or_default()),
                },
            ),
        ]);
        tensors.push(json!({
            "name": t.name,
            "layer": layer,
            "projection": projection,
            "ggml_type": t.ggml_type.name(),
            "orientation": orientation,
            "shape": shape,
            "bytes": t.size,
            "patchable": patchable,
            "patchable_reason": reason,
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
    human.push_str(&format!(
        "{} {}\n",
        ui::badge(ui, "ARCH"),
        ui::text(ui, &report.summary())
    ));
    if show_stats {
        human.push_str(&ui::dim(ui, "statistiques non disponibles pour un GGUF\n"));
    }
    let headers = [
        "couche",
        "proj",
        "type",
        "orientation",
        "shape",
        "taille",
        "patchable",
    ];
    human.push_str(&ui::table(ui, &headers, &rows));
    let data = json!({
        "format": "gguf",
        "version": parsed.version,
        "architecture": parsed.architecture(),
        "alignment": parsed.alignment,
        "data_start": parsed.data_start,
        "tensor_count": parsed.tensors.len(),
        "tied_embeddings": tied,
        "tensors": tensors,
        "arch": report,
    });
    Ok((data, human))
}
