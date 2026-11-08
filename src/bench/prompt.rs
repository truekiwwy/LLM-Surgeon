//! Prompt template of `bench eval`: a text with the single variable `{question}`.

use std::path::Path;

use super::error::{BenchError, Result};
use super::sha256_hex;

/// Asks for step-by-step reasoning then a final `#### <answer>` line, the format the
/// first extraction rule reads.
pub const DEFAULT_TEMPLATE: &str = "Solve the following problem step by step. \
At the end, write the final answer alone on the last line in the form \"#### <answer>\".\n\n\
Problem: {question}\n\nSolution:";

const VARIABLE: &str = "{question}";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Template(String);

impl Default for Template {
    fn default() -> Self {
        Self(DEFAULT_TEMPLATE.to_owned())
    }
}

impl Template {
    pub fn new(text: &str) -> Result<Self> {
        match text.contains(VARIABLE) {
            true => Ok(Self(text.to_owned())),
            false => Err(BenchError::InvalidArgument(format!(
                "gabarit de prompt sans la variable {VARIABLE}"
            ))),
        }
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| BenchError::input(path, e))?;
        Self::new(text.trim_start_matches('\u{feff}'))
    }

    /// CLI form: `@path` reads a file, anything else is the template itself.
    pub fn from_cli(spec: Option<&str>) -> Result<Self> {
        match spec {
            None => Ok(Self::default()),
            Some(s) => match s.strip_prefix('@') {
                Some(path) => Self::from_file(Path::new(path)),
                None => Self::new(s),
            },
        }
    }

    pub fn render(&self, question: &str) -> String {
        self.0.replace(VARIABLE, question)
    }

    pub fn sha256(&self) -> String {
        sha256_hex(self.0.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variable_is_required_and_replaced_everywhere() {
        let t = Template::new("Q: {question}\nAgain: {question}").unwrap();
        assert_eq!(t.render("2+2?"), "Q: 2+2?\nAgain: 2+2?");
        assert!(Template::new("no variable").is_err());
        assert!(Template::default().render("x").contains("Problem: x"));
        assert_eq!(Template::from_cli(None).unwrap(), Template::default());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.txt");
        std::fs::write(&path, "<<{question}>>").unwrap();
        let spec = format!("@{}", path.display());
        assert_eq!(
            Template::from_cli(Some(&spec)).unwrap().render("a"),
            "<<a>>"
        );
        assert_ne!(t.sha256(), Template::default().sha256());
    }
}
