//! Streaming detector of degenerate generation loops.
//!
//! Every character is examined exactly once, with bounded state: O(1) work per character
//! for periodicity (one run counter per candidate period), O(window) memory for the
//! n-gram and marker windows. No regex, so no backtracking. Chunk boundaries are
//! irrelevant because all state is carried across `push` calls.

use std::collections::{HashMap, VecDeque};

use serde::Serialize;

use super::truncate_chars;

const MIN_PERIOD: usize = 3;
const MAX_PERIOD: usize = 40;
/// A motif must appear this many times in a row (the motif plus 5 copies).
const MIN_PERIOD_REPEATS: usize = 6;
/// Caps the word buffer when the stream contains no whitespace.
const MAX_WORD_CHARS: usize = 64;
const MAX_EVIDENCE_CHARS: usize = 200;

/// Every `max_*` threshold is exclusive: a loop is reported when the count goes above it.
#[derive(Debug, Clone, Serialize)]
pub struct LoopDetectorConfig {
    /// Marker occurrences allowed inside the sliding window.
    pub max_marker_hits: usize,
    /// Case-insensitive, matched on word boundaries.
    pub markers: Vec<String>,
    /// Words per n-gram; 0 disables the n-gram check.
    pub ngram_size: usize,
    /// Occurrences of one n-gram allowed inside the sliding window.
    pub max_ngram_repeats: usize,
    /// Identical consecutive non-empty lines allowed (blank lines are ignored).
    pub max_identical_lines: usize,
    pub window_chars: usize,
    /// Statistical checks stay silent below this many characters. Markers are exempt:
    /// four "Wait" in a row are conclusive even in a 30-character output.
    pub min_chars_before_check: usize,
}

impl Default for LoopDetectorConfig {
    fn default() -> Self {
        Self {
            max_marker_hits: 3,
            markers: ["Wait", "Hmm", "Actually", "Let me re-check", "But wait"]
                .map(String::from)
                .to_vec(),
            ngram_size: 4,
            max_ngram_repeats: 4,
            max_identical_lines: 3,
            window_chars: 4000,
            min_chars_before_check: 80,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum LoopVerdict {
    Ok,
    Loop {
        reason: LoopReason,
        evidence: String,
        /// Characters consumed when the loop was confirmed.
        at_char: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LoopReason {
    MarkerRepetition,
    NgramRepetition,
    IdenticalLines,
    CharPeriodicity,
}

#[derive(Debug, Clone)]
pub struct LoopDetector {
    cfg: LoopDetectorConfig,
    /// (original text, lowercased chars), longest first so "But wait" wins over "Wait".
    markers: Vec<(String, Vec<char>)>,
    /// Last lowercased characters, long enough for every marker and period.
    recent: VecDeque<char>,
    recent_cap: usize,
    chars_seen: usize,
    /// Positions of marker hits still inside the window.
    marker_hits: VecDeque<usize>,
    /// `period_runs[p]`: consecutive characters equal to the one `p` positions earlier.
    period_runs: [usize; MAX_PERIOD + 1],
    word: String,
    word_start: usize,
    last_words: VecDeque<(String, usize)>,
    ngrams: VecDeque<(String, usize)>,
    ngram_counts: HashMap<String, usize>,
    line: String,
    last_line: String,
    line_streak: usize,
    /// Sticky: once a loop is found, `push` keeps returning it until `reset`.
    verdict: LoopVerdict,
}

impl LoopDetector {
    pub fn new(cfg: LoopDetectorConfig) -> Self {
        let mut markers: Vec<(String, Vec<char>)> = cfg
            .markers
            .iter()
            .filter(|m| !m.trim().is_empty())
            .map(|m| (m.clone(), m.trim().to_lowercase().chars().collect()))
            .collect();
        markers.sort_by_key(|(_, chars)| std::cmp::Reverse(chars.len()));
        let longest = markers.first().map_or(0, |(_, c)| c.len());
        let recent_cap = MAX_PERIOD.max(longest + 1);
        Self {
            cfg,
            markers,
            recent: VecDeque::with_capacity(recent_cap + 1),
            recent_cap,
            chars_seen: 0,
            marker_hits: VecDeque::new(),
            period_runs: [0; MAX_PERIOD + 1],
            word: String::new(),
            word_start: 0,
            last_words: VecDeque::new(),
            ngrams: VecDeque::new(),
            ngram_counts: HashMap::new(),
            line: String::new(),
            last_line: String::new(),
            line_streak: 0,
            verdict: LoopVerdict::Ok,
        }
    }

    pub fn push(&mut self, chunk: &str) -> LoopVerdict {
        if self.verdict == LoopVerdict::Ok {
            for c in chunk.chars() {
                if let Some((reason, evidence)) = self.step(c) {
                    self.verdict = LoopVerdict::Loop {
                        reason,
                        evidence: truncate_chars(&evidence, MAX_EVIDENCE_CHARS).to_owned(),
                        at_char: self.chars_seen,
                    };
                    break;
                }
            }
        }
        self.verdict.clone()
    }

    pub fn reset(&mut self) {
        *self = Self::new(std::mem::take(&mut self.cfg));
    }

    fn step(&mut self, c: char) -> Option<(LoopReason, String)> {
        let lc = c.to_lowercase().next().unwrap_or(c);
        let marker = self.check_marker(lc);
        let period = self.check_period(lc);
        self.chars_seen += 1;
        self.recent.push_back(lc);
        if self.recent.len() > self.recent_cap {
            self.recent.pop_front();
        }
        let ngram = self.feed_word(lc);
        let line = self.feed_line(c);

        if marker.is_some() {
            return marker;
        }
        if self.chars_seen < self.cfg.min_chars_before_check {
            return None;
        }
        ngram.or(line).or(period)
    }

    /// Called before `lc` is appended: `lc` is the character right after a candidate marker.
    fn check_marker(&mut self, lc: char) -> Option<(LoopReason, String)> {
        let len = self.recent.len();
        let found = self.markers.iter().find(|(_, m)| {
            let Some(start) = len.checked_sub(m.len()) else {
                return false;
            };
            let boundary_before = match (m.first(), start) {
                (Some(first), _) if !first.is_alphanumeric() => true,
                (_, 0) => self.chars_seen == m.len(),
                _ => !self.recent[start - 1].is_alphanumeric(),
            };
            let boundary_after =
                m.last().is_some_and(|l| !l.is_alphanumeric()) || !lc.is_alphanumeric();
            boundary_before && boundary_after && self.recent.range(start..).eq(m.iter())
        })?;
        let name = found.0.clone();

        self.marker_hits.push_back(self.chars_seen);
        let oldest = self.chars_seen.saturating_sub(self.cfg.window_chars);
        while self.marker_hits.front().is_some_and(|&p| p < oldest) {
            self.marker_hits.pop_front();
        }
        (self.marker_hits.len() > self.cfg.max_marker_hits).then(|| {
            (
                LoopReason::MarkerRepetition,
                format!(
                    "{} marqueurs dans les {} derniers caractères (dernier : « {name} »)",
                    self.marker_hits.len(),
                    self.cfg.window_chars
                ),
            )
        })
    }

    /// Called before `lc` is appended to `recent`.
    fn check_period(&mut self, lc: char) -> Option<(LoopReason, String)> {
        let len = self.recent.len();
        let mut found = None;
        for p in MIN_PERIOD..=MAX_PERIOD {
            let repeats = len >= p && self.recent[len - p] == lc;
            self.period_runs[p] = if repeats { self.period_runs[p] + 1 } else { 0 };
            if found.is_none() && self.period_runs[p] >= (MIN_PERIOD_REPEATS - 1) * p {
                let motif: String = self
                    .recent
                    .range(len - p + 1..)
                    .copied()
                    .chain([lc])
                    .collect();
                // Separators such as "|---" or runs of spaces are layout, not a loop.
                if motif.chars().any(char::is_alphanumeric) {
                    found = Some((
                        LoopReason::CharPeriodicity,
                        format!("motif « {motif} » répété {MIN_PERIOD_REPEATS} fois de suite"),
                    ));
                }
            }
        }
        found
    }

    fn feed_word(&mut self, lc: char) -> Option<(LoopReason, String)> {
        if !lc.is_whitespace() {
            if self.word.is_empty() {
                self.word_start = self.chars_seen - 1;
            }
            if self.word.chars().count() < MAX_WORD_CHARS {
                self.word.push(lc);
            }
            return None;
        }
        let raw = std::mem::take(&mut self.word);
        let word = raw.trim_matches(|c: char| !c.is_alphanumeric());
        let n = self.cfg.ngram_size;
        if word.is_empty() || n == 0 {
            return None;
        }
        self.last_words
            .push_back((word.to_owned(), self.word_start));
        if self.last_words.len() > n {
            self.last_words.pop_front();
        }
        if self.last_words.len() < n {
            return None;
        }

        let start = self.last_words.front().map_or(0, |w| w.1);
        let key = self
            .last_words
            .iter()
            .map(|w| w.0.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        *self.ngram_counts.entry(key.clone()).or_insert(0) += 1;
        self.ngrams.push_back((key.clone(), start));

        let oldest = self.chars_seen.saturating_sub(self.cfg.window_chars);
        while let Some((old, _)) = self.ngrams.front().filter(|(_, s)| *s < oldest).cloned() {
            self.ngrams.pop_front();
            if let Some(c) = self.ngram_counts.get_mut(&old) {
                *c -= 1;
                if *c == 0 {
                    self.ngram_counts.remove(&old);
                }
            }
        }
        let live = self.ngram_counts.get(&key).copied().unwrap_or(0);
        (live > self.cfg.max_ngram_repeats).then(|| {
            (
                LoopReason::NgramRepetition,
                format!("« {key} » répété {live} fois dans la fenêtre"),
            )
        })
    }

    fn feed_line(&mut self, c: char) -> Option<(LoopReason, String)> {
        if c != '\n' {
            if self.line.len() < self.cfg.window_chars {
                self.line.push(c);
            }
            return None;
        }
        let raw = std::mem::take(&mut self.line);
        let line = raw.trim();
        if line.is_empty() {
            return None;
        }
        if line == self.last_line {
            self.line_streak += 1;
        } else {
            self.last_line = line.to_owned();
            self.line_streak = 1;
        }
        (self.line_streak > self.cfg.max_identical_lines).then(|| {
            (
                LoopReason::IdenticalLines,
                format!(
                    "ligne « {line} » répétée {} fois de suite",
                    self.line_streak
                ),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEALTHY_FR: &str = "La photosynthèse est le processus par lequel les plantes \
        vertes transforment l'énergie lumineuse en énergie chimique. Dans les chloroplastes, \
        la chlorophylle capte les photons et déclenche une chaîne de réactions qui scinde les \
        molécules d'eau, libère de l'oxygène et produit des transporteurs d'énergie. Ces \
        derniers alimentent ensuite le cycle de Calvin, au cours duquel le dioxyde de carbone \
        atmosphérique est fixé puis réduit en sucres simples. Il faut parfois wait quelques \
        secondes avant que la feuille éclairée ne montre une activité mesurable, mais le \
        phénomène reste remarquablement efficace. Les sucres formés servent à la croissance, \
        à la respiration cellulaire et à la constitution des réserves, notamment sous forme \
        d'amidon. Le rendement global dépend de la température, de l'intensité lumineuse, de \
        la disponibilité en eau et de la concentration en dioxyde de carbone. Certaines \
        espèces tropicales ont développé des voies métaboliques alternatives, dites C4 ou \
        CAM, qui limitent les pertes par photorespiration lorsque le climat devient chaud et \
        sec. Comprendre ces mécanismes aide les agronomes à sélectionner des variétés plus \
        résistantes et à mieux gérer l'irrigation des cultures.";

    fn feed(d: &mut LoopDetector, chunks: &[&str]) -> LoopVerdict {
        chunks
            .iter()
            .map(|c| d.push(c))
            .last()
            .unwrap_or(LoopVerdict::Ok)
    }

    fn reason(v: &LoopVerdict) -> Option<LoopReason> {
        match v {
            LoopVerdict::Loop { reason, .. } => Some(*reason),
            LoopVerdict::Ok => None,
        }
    }

    #[test]
    fn healthy_french_text_is_not_a_loop_whatever_the_chunking() {
        assert!(HEALTHY_FR.chars().count() >= 1000);
        assert_eq!(HEALTHY_FR.to_lowercase().matches("wait").count(), 1);
        let mut d = LoopDetector::new(LoopDetectorConfig::default());
        assert_eq!(d.push(HEALTHY_FR), LoopVerdict::Ok);
        let chars: Vec<char> = HEALTHY_FR.chars().collect();
        for size in [1, 7, 33] {
            d.reset();
            for piece in chars.chunks(size) {
                let piece: String = piece.iter().collect();
                assert_eq!(d.push(&piece), LoopVerdict::Ok, "chunk size {size}");
            }
        }
    }

    #[test]
    fn wait_repetition_is_a_marker_loop_case_insensitive() {
        let mut d = LoopDetector::new(LoopDetectorConfig::default());
        let v = d.push("Wait... Wait... Wait... Wait...");
        let LoopVerdict::Loop {
            reason, at_char, ..
        } = &v
        else {
            panic!("{v:?}")
        };
        assert_eq!(*reason, LoopReason::MarkerRepetition);
        assert!(*at_char <= 31);
        assert_eq!(d.push("tout va bien"), v, "verdict is sticky");

        d.reset();
        assert_eq!(
            reason_of(&mut d, "WAIT, wait! wAiT? But wait."),
            Some(LoopReason::MarkerRepetition)
        );
        d.reset();
        assert_eq!(d.push("Waiting awaits Waitress unwait"), LoopVerdict::Ok);
        d.reset();
        assert_eq!(
            d.push("But wait. But wait. But wait."),
            LoopVerdict::Ok,
            "\"But wait\" counts once, not as a second \"Wait\""
        );
    }

    fn reason_of(d: &mut LoopDetector, text: &str) -> Option<LoopReason> {
        reason(&d.push(text))
    }

    #[test]
    fn pattern_split_across_chunks_is_detected() {
        let mut d = LoopDetector::new(LoopDetectorConfig::default());
        let v = feed(
            &mut d,
            &["Wa", "it... Wai", "t...", " W", "ait.", "..", " Wai", "t."],
        );
        assert_eq!(reason(&v), Some(LoopReason::MarkerRepetition));
    }

    #[test]
    fn repeated_ngram_with_varying_tail() {
        let mut d = LoopDetector::new(LoopDetectorConfig::default());
        let text: String = (0..6)
            .map(|i| {
                format!(
                    "Je pense que la réponse vaut {} ou peut-être {}. ",
                    i * 7,
                    i + 2
                )
            })
            .collect();
        assert_eq!(reason_of(&mut d, &text), Some(LoopReason::NgramRepetition));
    }

    #[test]
    fn identical_lines() {
        let mut d = LoopDetector::new(LoopDetectorConfig::default());
        let line = "Je dois vérifier le calcul encore une fois.\n";
        assert_eq!(d.push(&line.repeat(3)), LoopVerdict::Ok);
        assert_eq!(
            d.push("\n  \n"),
            LoopVerdict::Ok,
            "blank lines do not break the streak"
        );
        assert_eq!(reason_of(&mut d, line), Some(LoopReason::IdenticalLines));
    }

    #[test]
    fn char_periodicity_ignores_layout_separators() {
        let mut d = LoopDetector::new(LoopDetectorConfig::default());
        let prefix = "Voici le résultat du calcul demandé, présenté de manière claire :\n";
        assert_eq!(d.push(prefix), LoopVerdict::Ok);
        assert_eq!(d.push(&"|---".repeat(30)), LoopVerdict::Ok);
        assert_eq!(d.push(&" ".repeat(200)), LoopVerdict::Ok);
        assert_eq!(
            reason_of(&mut d, &"xyz.".repeat(6)),
            Some(LoopReason::CharPeriodicity)
        );
    }

    #[test]
    fn statistical_checks_wait_for_min_chars() {
        let mut d = LoopDetector::new(LoopDetectorConfig::default());
        assert_eq!(d.push(&"abc".repeat(20)), LoopVerdict::Ok, "60 chars < 80");
        assert_eq!(
            reason_of(&mut d, &"abc".repeat(7)),
            Some(LoopReason::CharPeriodicity)
        );
    }

    #[test]
    fn sliding_window_forgets_old_markers() {
        let filler = |n: usize| -> String { (0..n).map(|i| format!("mot{i} ")).collect() };
        let text: String = (0..5).map(|_| format!("Wait. {}", filler(50))).collect();

        let small = LoopDetectorConfig {
            window_chars: 200,
            ..Default::default()
        };
        assert_eq!(LoopDetector::new(small).push(&text), LoopVerdict::Ok);
        assert_eq!(
            reason(&LoopDetector::new(LoopDetectorConfig::default()).push(&text)),
            Some(LoopReason::MarkerRepetition)
        );
    }
}
