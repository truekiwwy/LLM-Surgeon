//! Final answer of a model output. Rules in priority order, the first one that finds a
//! non-empty candidate wins:
//!
//! 1. `hashes`: `#### X` (exactly four `#`), last marker whose rest of line holds a number;
//!    a digit-free marker (`#### N` echoed from the prompt) is skipped, see [`hashes`];
//! 2. `boxed`: `\boxed{X}`, last balanced occurrence (nested and escaped braces handled
//!    by a small parser);
//! 3. `bold`: last `**X**` segment holding at least one ASCII digit;
//! 4. `phrase`: last "the answer is X" / "final answer: X" / "réponse finale : X"
//!    (case-insensitive); a leading number is kept alone, otherwise the rest of the line;
//! 5. `last_number`: the last number of the text.
//!
//! The regexes run on the `regex` crate (linear time, no backtracking).

use std::sync::OnceLock;

use regex::Regex;
use schemars::JsonSchema;
use serde::Serialize;

use super::score::{normalize, parse_number};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Rule {
    Hashes,
    Boxed,
    Bold,
    Phrase,
    LastNumber,
}

impl Rule {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hashes => "hashes",
            Self::Boxed => "boxed",
            Self::Bold => "bold",
            Self::Phrase => "phrase",
            Self::LastNumber => "last_number",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Extracted {
    pub answer: String,
    pub rule: Rule,
}

fn cached(cell: &'static OnceLock<Option<Regex>>, pattern: &str) -> Option<&'static Regex> {
    cell.get_or_init(|| Regex::new(pattern).ok()).as_ref()
}

fn bold_re() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    cached(&RE, r"\*\*([^*\n]+)\*\*")
}

fn phrase_re() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    cached(
        &RE,
        r"(?i)(?:the\s+(?:final\s+)?answer\s+is|final\s+answer\s*(?:is)?\s*:|réponse\s+finale\s*(?:est)?\s*:|la\s+réponse\s+est)\s*:?\s*([^\n]*)",
    )
}

/// Thousands groups first (`1,234.5`), else a plain or decimal-comma number with an
/// optional exponent (`1,8e1`).
const NUMBER: &str = r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?";

pub(crate) fn number_re() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    cached(&RE, NUMBER)
}

/// A number, or a plain fraction `a/b` kept whole.
fn number_or_fraction_re() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    RE.get_or_init(|| Regex::new(&format!("(?:{NUMBER})(?:/(?:{NUMBER}))?")).ok())
        .as_ref()
}

fn hash_run_re() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    cached(&RE, "#+")
}

/// Content of the brace group opening at `open` (index of `{`), escaped braces skipped.
fn balanced(text: &str, open: usize) -> Option<&str> {
    let mut depth = 0usize;
    let mut chars = text[open..].char_indices();
    while let Some((i, c)) = chars.next() {
        match c {
            '\\' => {
                chars.next();
            }
            '{' => depth += 1,
            '}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(&text[open + 1..open + i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Content of the last balanced `\boxed{...}`.
pub fn last_boxed(text: &str) -> Option<String> {
    text.rmatch_indices("\\boxed{")
        .find_map(|(at, m)| balanced(text, at + m.len() - 1))
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}

/// Rule `hashes`. A marker is a run of exactly four `#` anywhere in a line (`#####` opens a
/// Markdown heading, not an answer); its candidate is the rest of that line. Markers are
/// read from the last one back and the first candidate holding a number wins:
///
/// - a candidate the scorer reads whole as one number (`18`, `1,234`, `\frac{1}{2}`,
///   `160 minutes`, `60%`) is returned as written;
/// - otherwise its last number or `a/b` fraction, sign and thousands groups kept
///   (`#### Final Line: 3000` -> `3000`, `#### Total = -1,250.50 euros` -> `-1,250.50`);
/// - a candidate without digits (`#### N` or `#### <number>` echoed from the prompt,
///   `####` alone) is skipped, so an earlier marker, then the next rules, get their turn.
fn hashes(text: &str) -> Option<String> {
    let ends: Vec<usize> = hash_run_re()?
        .find_iter(text)
        .filter(|m| m.len() == 4)
        .map(|m| m.end())
        .collect();
    ends.into_iter().rev().find_map(|end| {
        let candidate = text[end..].lines().next().unwrap_or("").trim();
        if parse_number(&normalize(candidate)).is_some() {
            return Some(candidate.to_owned());
        }
        number_or_fraction_re()?
            .find_iter(candidate)
            .last()
            .map(|m| m.as_str().to_owned())
    })
}

fn bold(text: &str) -> Option<String> {
    bold_re()?
        .captures_iter(text)
        .filter_map(|c| Some(c.get(1)?.as_str().trim().to_owned()))
        // Skips digit-free headings (`**Verification:**`) so later rules still get a chance.
        .filter(|s| s.chars().any(|c| c.is_ascii_digit()))
        .last()
}

fn phrase(text: &str) -> Option<String> {
    let rest = phrase_re()?
        .captures_iter(text)
        .filter_map(|c| Some(c.get(1)?.as_str().trim().to_owned()))
        .filter(|s| !s.is_empty())
        .last()?;
    let stripped = rest.trim_start_matches(['$', '\\', ' ']);
    let leading = number_re()?
        .find(stripped)
        .filter(|m| m.start() == 0)
        .map(|m| m.as_str().to_owned());
    Some(leading.unwrap_or_else(|| rest.trim_end_matches('.').trim().to_owned()))
}

fn last_number(text: &str) -> Option<String> {
    number_re()?
        .find_iter(text)
        .last()
        .map(|m| m.as_str().to_owned())
}

type Finder = fn(&str) -> Option<String>;

/// Printed by llama.cpp (`[end of text]`) or left in the stream when a stop token is not
/// filtered; never part of an answer.
const END_MARKERS: [&str; 6] = [
    "[end of text]",
    "</s>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|end_of_text|>",
    "<|eot_id|>",
];

/// `output` without trailing whitespace and end-of-generation markers. Only the tail is
/// touched, so a marker quoted inside the reasoning stays.
fn strip_end_markers(output: &str) -> &str {
    let mut text = output.trim_end();
    while let Some(rest) = END_MARKERS
        .iter()
        .find_map(|marker| text.strip_suffix(marker))
    {
        text = rest.trim_end();
    }
    text
}

/// Final answer of `output`, end-of-generation markers removed first: the first rule, in
/// the priority order of the module docs (`hashes`, `boxed`, `bold`, `phrase`,
/// `last_number`), that finds a candidate. A rule that finds nothing usable (only
/// digit-free `####` markers, a `**heading**` without digit) hands over to the next one.
pub fn extract(output: &str) -> Option<Extracted> {
    let output = strip_end_markers(output);
    let rules: [(Rule, Finder); 5] = [
        (Rule::Hashes, hashes),
        (Rule::Boxed, last_boxed),
        (Rule::Bold, bold),
        (Rule::Phrase, phrase),
        (Rule::LastNumber, last_number),
    ];
    rules.iter().find_map(|(rule, find)| {
        find(output).map(|answer| Extracted {
            answer,
            rule: *rule,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn got(output: &str) -> Option<(String, Rule)> {
        extract(output).map(|e| (e.answer, e.rule))
    }

    fn is(output: &str, answer: &str, rule: Rule) {
        assert_eq!(got(output), Some((answer.to_owned(), rule)), "{output:?}");
    }

    #[test]
    fn extraction_table() {
        is("Adding up: 10 + 8 = 18.\n#### 18", "18", Rule::Hashes);
        is("first #### 3\nthen #### 18\n####", "18", Rule::Hashes);
        is(
            "So $x = \\boxed{\\frac{1}{2}}$.",
            "\\frac{1}{2}",
            Rule::Boxed,
        );
        is(
            "\\boxed{\\left(1,2\\right)}",
            "\\left(1,2\\right)",
            Rule::Boxed,
        );
        is(
            "\\boxed{\\frac{\\sqrt{\\frac{a}{b}}}{2}}",
            "\\frac{\\sqrt{\\frac{a}{b}}}{2}",
            Rule::Boxed,
        );
        is(
            "\\boxed{1} then \\boxed{\\{1,2\\}}",
            "\\{1,2\\}",
            Rule::Boxed,
        );
        is("unbalanced \\boxed{3} and \\boxed{4", "3", Rule::Boxed);
        is("The result is **42**.", "42", Rule::Bold);
        is("**Step 1** ... **Step 2** gives **7**", "7", Rule::Bold);
        is("The answer is $1,234.50.", "1,234.50", Rule::Phrase);
        is("Final answer: 12 apples", "12", Rule::Phrase);
        is("Donc la réponse finale : 12", "12", Rule::Phrase);
        is("the answer is blue.", "blue", Rule::Phrase);
        is("I got 3, then 5, finally 9 cookies", "9", Rule::LastNumber);
        is(
            "The temperature drops to -4 degrees",
            "-4",
            Rule::LastNumber,
        );
        is("Avogadro is about 6.02e23", "6.02e23", Rule::LastNumber);
        is("It grew by 50%", "50", Rule::LastNumber);
        is("He pays 18 dollars", "18", Rule::LastNumber);
        assert_eq!(got(""), None);
        assert_eq!(got("no digits at all"), None);
    }

    #[test]
    fn end_markers_are_not_part_of_the_answer() {
        is("Total.\n#### 18 [end of text]", "18", Rule::Hashes);
        is(
            "#### 160 minutes [end of text]\n",
            "160 minutes",
            Rule::Hashes,
        );
        is("#### 60% </s>", "60%", Rule::Hashes);
        is("The answer is 42.<|im_end|>\n", "42", Rule::Phrase);
        is("**7** <|im_end|> [end of text]", "7", Rule::Bold);
        // Mid-text markers are left alone.
        is("a [end of text] b 5", "5", Rule::LastNumber);
        assert_eq!(got("[end of text]"), None);
    }

    #[test]
    fn bold_without_digit_is_skipped() {
        is("**12** apples.\n**Verification:** done", "12", Rule::Bold);
        is("**Note:** 5 + 7 = 12", "12", Rule::LastNumber);
        is("**Verification:** the answer is 9.", "9", Rule::Phrase);
        assert_eq!(got("**Note:** none"), None);
    }

    #[test]
    fn priority_between_rules() {
        is("**5** but #### 6", "6", Rule::Hashes);
        is("the answer is 3, **4**, \\boxed{5}", "5", Rule::Boxed);
        is("the answer is 3 but **4**", "4", Rule::Bold);
        is("The answer is 8. Checking: 2 * 4 = 8, 9", "8", Rule::Phrase);
    }

    #[test]
    fn digit_free_hash_markers_are_skipped() {
        // Prompt echoed after the real answer: an earlier marker answers.
        is("200 - 138 = 62.\n\n#### 62\n\n#### N", "62", Rule::Hashes);
        is(
            "#### 42\n#### <number>\n#### [answer]\n####",
            "42",
            Rule::Hashes,
        );
        is("#### 42\n#### N\"", "42", Rule::Hashes);
        // Only an echo: the next rules answer.
        is("End with #### N.\nSo \\boxed{42}", "42", Rule::Boxed);
        is("Write \"#### N\".\nShe keeps 42", "42", Rule::LastNumber);
        assert_eq!(got("#### N"), None);
        assert_eq!(got("####"), None);
        is("12 apples\n####", "12", Rule::LastNumber);
    }

    #[test]
    fn text_around_the_number_of_a_hash_marker() {
        is(
            "#### Final Line: 3000\n\nSo, the final numeric answer is 3000 grams.",
            "3000",
            Rule::Hashes,
        );
        is("#### Answer: $45", "45", Rule::Hashes);
        is("#### Answer: $1,250.50", "1,250.50", Rule::Hashes);
        is("#### Total = -1,250.50 euros", "-1,250.50", Rule::Hashes);
        is("#### 3000\"", "3000", Rule::Hashes);
        is("#### 5 + 3 = 8", "8", Rule::Hashes);
        is("#### Answer: 3/4 cup", "3/4", Rule::Hashes);
        is(
            "In the form \"#### N\", this is written as 150.",
            "150",
            Rule::Hashes,
        );
        // A candidate the scorer reads whole stays as written.
        is("#### 18 hours.", "18 hours.", Rule::Hashes);
        is("#### -7", "-7", Rule::Hashes);
        is("#### +7", "+7", Rule::Hashes);
        is("#### 1,234", "1,234", Rule::Hashes);
        is("#### \\frac{1}{2}", "\\frac{1}{2}", Rule::Hashes);
    }

    #[test]
    fn last_numeric_hash_marker_wins_and_headings_are_not_markers() {
        is("#### 12\nRechecking.\n#### 15", "15", Rule::Hashes);
        is("#### 62 then #### N", "62", Rule::Hashes);
        is("#### 5 #### 6", "6", Rule::Hashes);
        is("##### Step 2: 5 + 3 = 8\n\\boxed{9}", "9", Rule::Boxed);
        is("#### 7\n##### Check: 3 + 4", "7", Rule::Hashes);
        is("###### 4\n#### 7", "7", Rule::Hashes);
        // Standard GSM8K reference layout.
        is(
            "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n#### 72",
            "72",
            Rule::Hashes,
        );
    }
}
