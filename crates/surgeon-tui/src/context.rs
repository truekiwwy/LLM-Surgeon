//! The AI-agent context export: a clean Markdown snippet describing the tensor
//! currently selected (identity, topology role, metrics, ablation exactness),
//! written to a file an agent CLI can consume, plus the launcher that runs the
//! configured agent command with that file.

use std::path::PathBuf;
use std::process::Command;

use crate::model::metrics::Stats;
use crate::model::roles::Exactness;
use crate::model::TensorEntry;

/// Everything the snippet needs about the current selection.
#[derive(Debug, Clone)]
pub struct ContextInfo {
    pub entry: TensorEntry,
    pub family: Option<String>,
    pub format: String,
    pub stats: Option<Stats>,
}

impl ContextInfo {
    /// Builds the Markdown snippet. No ANSI, no AI mentions: just the facts.
    pub fn markdown(&self) -> String {
        let e = &self.entry;
        let mut out = String::new();
        out.push_str("## Tenseur : `");
        out.push_str(&e.name);
        out.push_str("`\n\n");
        out.push_str("| Propriété | Valeur |\n|---|---|\n");
        out.push_str(&format!("| Format | {} |\n", self.format));
        out.push_str(&format!("| Forme | {} |\n", e.shape_text()));
        out.push_str(&format!("| DType | {} |\n", e.dtype.name()));
        if let Some(bpw) = e.dtype.bits_per_weight() {
            out.push_str(&format!("| Bits/weight | {bpw:.2} |\n"));
        }
        out.push_str(&format!("| Rôle | {} |\n", e.role));
        if let Some(layer) = e.layer {
            out.push_str(&format!("| Couche | {layer} |\n"));
        }
        if let Some(expert) = e.expert {
            out.push_str(&format!("| Expert | {expert} |\n"));
        }
        out.push_str(&format!("| Taille | {} o |\n", e.byte_len));
        if let Some(family) = &self.family {
            out.push_str(&format!("| Famille | {family} |\n"));
        }
        if let Some(exactness) = Exactness::for_entry(self.family.as_deref(), e) {
            out.push_str(&format!(
                "| Exactitude d'ablation | {} |\n",
                exactness.label()
            ));
        }
        if let Some(stats) = &self.stats {
            out.push_str("\n### Métriques statiques\n\n");
            if stats.available {
                out.push_str(&format!(
                    "- Norme L2 : {:.4}{}",
                    stats.l2,
                    if stats.sampled < stats.numel {
                        format!(" (échantillon {}/{})", stats.sampled, stats.numel)
                    } else {
                        String::new()
                    }
                ));
                out.push('\n');
                out.push_str(&format!("- Min / Max : {:.6} / {:.6}\n", stats.min, stats.max));
                out.push_str(&format!("- Moyenne / |moyenne| : {:.6} / {:.6}\n", stats.mean, stats.abs_mean));
                out.push_str(&format!("- Kurtosis (excès) : {:.4}\n", stats.kurtosis));
                out.push_str(&format!(
                    "- Rang effectif approx. : {:.2} / {}\n",
                    stats.r_eff,
                    crate::model::metrics::BASIS
                ));
            } else {
                out.push_str("- Dtype non décodé : métriques indisponibles.\n");
            }
        } else {
            out.push_str("\n### Métriques statiques\n\n- Métriques indisponibles (pas encore calculées).\n");
        }
        out
    }

    /// Writes the snippet to a fresh file in the system temp directory and
    /// returns its path.
    pub fn write_temp(&self) -> Result<PathBuf, std::io::Error> {
        let name = self.entry.name.replace(['/', '\\', ':'], "_");
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_millis() as u64 + d.as_secs() % 1_000_000)
            .unwrap_or(0);
        let mut path = std::env::temp_dir();
        path.push(format!("surgeon-context-{stamp}-{name}.md"));
        // Keep the file name bounded; long tensor names stay readable.
        if path.file_name().map(|n| n.len() > 180).unwrap_or(false) {
            path.set_file_name(format!("surgeon-context-{stamp}.md"));
        }
        std::fs::write(&path, self.markdown())?;
        Ok(path)
    }
}

/// Splits a command template on whitespace, honoring double and single quotes.
pub fn split_command(template: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut quote = '\0';
    for ch in template.chars() {
        if quote != '\0' {
            if ch == quote {
                quote = '\0';
            } else {
                current.push(ch);
            }
        } else if ch == '"' || ch == '\'' {
            quote = ch;
        } else if ch.is_whitespace() {
            if !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
        } else {
            current.push(ch);
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

/// Builds the argv for the agent launch: the template either contains a
/// `{context}` placeholder (replaced by the snippet path) or the path is
/// appended as the last argument.
pub fn agent_argv(template: &str, context_file: &std::path::Path) -> Vec<String> {
    let mut argv = split_command(template);
    let mut replaced = false;
    for part in &mut argv {
        if part.contains("{context}") {
            *part = part.replace("{context}", &context_file.display().to_string());
            replaced = true;
        }
    }
    if !replaced {
        argv.push(context_file.display().to_string());
    }
    argv
}

/// Spawns the agent with the snippet path, waits for it, returns its code.
/// The caller must have released the terminal (alternate screen off).
pub fn launch_agent(argv: &[String]) -> Result<i32, std::io::Error> {
    let Some((program, args)) = argv.split_first() else {
        return Err(std::io::Error::other("commande d'agent vide"));
    };
    let mut command = Command::new(program);
    command.args(args);
    match command.status() {
        Ok(status) => Ok(status.code().unwrap_or(1)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && cfg!(windows) => {
            // .cmd shims (claude.cmd, ...) cannot be spawned directly.
            let mut via_cmd = Command::new("cmd");
            via_cmd.arg("/C").arg(program).args(args);
            Ok(via_cmd.status()?.code().unwrap_or(1))
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::dtype::DType;
    use crate::model::MoeKind;
    use crate::model::Role;

    fn info(stats: Option<Stats>) -> ContextInfo {
        ContextInfo {
            entry: TensorEntry {
                name: "model.layers.14.mlp.down_proj.weight".into(),
                shape: vec![1024, 4096],
                dtype: DType::Q6K,
                byte_len: 2_109_235,
                file: PathBuf::from("m.gguf"),
                offset: 1234,
                role: Role::MlpOut,
                layer: Some(14),
                expert: None,
                moe: MoeKind::Dense,
            },
            family: Some("qwen3".into()),
            format: "gguf".into(),
            stats,
        }
    }

    #[test]
    fn markdown_lists_identity_topology_and_metrics() {
        let mut stats = Stats::default();
        stats.available = true;
        stats.numel = 1024 * 4096;
        stats.sampled = 64 * 256;
        stats.l2 = 42.5;
        stats.min = -0.5;
        stats.max = 0.5;
        stats.mean = 0.001;
        stats.abs_mean = 0.01;
        stats.kurtosis = 1.5;
        stats.r_eff = 8.25;
        let md = info(Some(stats)).markdown();
        assert!(md.contains("`model.layers.14.mlp.down_proj.weight`"));
        assert!(md.contains("Q6_K"));
        assert!(md.contains("[1024, 4096]"));
        assert!(md.contains("| Couche | 14 |"));
        assert!(md.contains("Exact"));
        assert!(md.contains("42.5000"));
        assert!(md.contains("échantillon 16384/4194304"));
        assert!(md.contains("8.25"));
        assert!(!md.contains("échantillon") || md.contains("16384"));
    }

    #[test]
    fn markdown_reports_unavailable_metrics() {
        let md = info(None).markdown();
        assert!(md.contains("indisponible"));
    }

    #[test]
    fn snippet_lands_in_a_real_temp_file() {
        let path = info(None).write_temp().expect("write");
        let content = std::fs::read_to_string(&path).expect("read back");
        assert!(content.contains("## Tenseur"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn command_splitting_and_placeholder() {
        assert_eq!(
            split_command(r#"claude --model opus "hello world""#),
            vec!["claude", "--model", "opus", "hello world"]
        );
        let argv = agent_argv("aider --message-file {context}", std::path::Path::new("/tmp/c.md"));
        assert_eq!(argv, vec!["aider", "--message-file", "/tmp/c.md"]);
        // Without a placeholder the path is appended last.
        let argv = agent_argv("codex", std::path::Path::new("/tmp/c.md"));
        assert_eq!(argv, vec!["codex", "/tmp/c.md"]);
    }
}
