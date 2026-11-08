//! Reading `llama-perplexity` output: `Final estimate: PPL = 7.1234 +/- 0.04567` (last
//! match wins; ANSI sequences stripped; a decimal comma is accepted) and the progress
//! tokens `[12]6.9876,` it prints chunk after chunk. The `regex` crate runs in linear
//! time, so no input can trigger catastrophic backtracking.

use std::sync::OnceLock;

use regex::Regex;

use crate::ui::strip_ansi;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PplEstimate {
    pub ppl: f64,
    /// The `±` llama.cpp prints, when present.
    pub err: Option<f64>,
}

fn final_re() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)final\s+estimate\s*:\s*ppl\s*=\s*([+-]?(?:nan|inf(?:inity)?|[0-9]+(?:[.,][0-9]+)?(?:e[+-]?[0-9]+)?))(?:\s*(?:\+/-|±)\s*([0-9]+(?:[.,][0-9]+)?(?:e[+-]?[0-9]+)?))?",
        )
        .ok()
    })
    .as_ref()
}

fn progress_re() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\[(\d+)\]\s*(?:[0-9]+(?:[.,][0-9]+)?|nan|inf|-nan),").ok())
        .as_ref()
}

fn number(text: &str) -> Option<f64> {
    text.replace(',', ".").parse::<f64>().ok()
}

/// Last `Final estimate`. `Err(reason)` when absent, not finite or not positive.
pub fn parse_final(output: &str) -> Result<PplEstimate, String> {
    let clean = strip_ansi(output);
    let re = final_re().ok_or_else(|| "expression régulière interne invalide".to_owned())?;
    let caps = re
        .captures_iter(&clean)
        .last()
        .ok_or_else(|| "« Final estimate: PPL = ... » absent (sortie tronquée ?)".to_owned())?;
    let raw = caps.get(1).map_or("", |m| m.as_str());
    let ppl = number(raw).ok_or_else(|| format!("valeur « {raw} » illisible"))?;
    if !ppl.is_finite() || ppl <= 0.0 {
        return Err(format!("perplexité {raw} non finie ou non positive"));
    }
    let err = caps
        .get(2)
        .and_then(|m| number(m.as_str()))
        .filter(|e| e.is_finite() && *e >= 0.0);
    Ok(PplEstimate { ppl, err })
}

/// Highest chunk index announced in `line` (`[12]6.9876,`), if any.
pub fn progress(line: &str) -> Option<usize> {
    progress_re()?
        .captures_iter(&strip_ansi(line))
        .filter_map(|c| c.get(1)?.as_str().parse().ok())
        .max()
}

/// Total announced by `perplexity: calculating perplexity over 655 chunks, ...`.
pub fn total_chunks(line: &str) -> Option<usize> {
    let rest = line.split("perplexity over ").nth(1)?;
    rest.split_whitespace().next()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regexes_compile() {
        assert!(final_re().is_some() && progress_re().is_some());
    }

    #[test]
    fn llama_cpp_output_formats() {
        let modern = "perplexity: calculating perplexity over 4 chunks, n_ctx=512\n\
                      [1]5.1234,[2]6.2345,[3]6.9,[4]7.1,\n\
                      Final estimate: PPL = 7.1234 +/- 0.04567\n";
        assert_eq!(
            parse_final(modern),
            Ok(PplEstimate {
                ppl: 7.1234,
                err: Some(0.04567)
            })
        );
        assert_eq!(total_chunks(modern.lines().next().unwrap()), Some(4));
        assert_eq!(progress("[1]5.1234,[2]6.2345,[3]6.9,"), Some(3));
        assert_eq!(progress("no progress here"), None);

        let old = "Final estimate: PPL = 9.8765";
        assert_eq!(parse_final(old).unwrap().err, None);

        let ansi = "\x1b[1mFinal estimate: PPL = 7.5000 +/- 0.01\x1b[0m";
        assert_eq!(parse_final(ansi).unwrap().ppl, 7.5);

        let twice = "Final estimate: PPL = 8.0 +/- 0.1\nFinal estimate: PPL = 7.0 +/- 0.2";
        assert_eq!(parse_final(twice).unwrap().ppl, 7.0, "last match wins");

        let comma = "Final estimate: PPL = 7,1234 +/- 0,0456";
        assert_eq!(
            parse_final(comma),
            Ok(PplEstimate {
                ppl: 7.1234,
                err: Some(0.0456)
            })
        );
    }

    #[test]
    fn truncated_or_degenerate_outputs_are_refused() {
        assert!(parse_final("[1]5.1,[2]6.2,").is_err(), "no final line");
        assert!(parse_final("").is_err());
        assert!(parse_final("Final estimate: PPL = nan +/- nan").is_err());
        assert!(parse_final("Final estimate: PPL = inf").is_err());
        assert!(parse_final("Final estimate: PPL = 0.0000 +/- 0.0").is_err());
        assert!(parse_final("Final estimate: PPL = -3.2").is_err());
    }
}
