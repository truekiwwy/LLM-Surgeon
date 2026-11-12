//! Streaming detector of degenerate generation loops.
//!
//! Every character is examined exactly once, with bounded state: O(1) work per character
//! for periodicity (one run counter per candidate period, in characters and in words),
//! O(window) memory for the marker window. No regex, no allocation per character or word.
//! Chunk boundaries are irrelevant because all state is carried across `push` calls.
//!
//! Repetition only counts when it is contiguous at the end of the stream: a step-by-step
//! answer reuses stock phrases ("the total number of") far apart, a loop repeats the same
//! block back to back. Calibration: outputs/reports/loop-calibration.md.

use std::collections::VecDeque;

use serde::Serialize;

use super::truncate_chars;

const MIN_PERIOD: usize = 3;
const MAX_PERIOD: usize = 40;
/// A motif must appear this many times in a row. Healthy GSM8K solutions top out at 4
/// (the zeros of "10000"), 2^8 written out as "2 * 2 * ... * 2" has 8, the shortest real
/// loop of the calibration set has 10.
const MIN_PERIOD_REPEATS: usize = 9;
/// A motif without letters or digits ("!", "|---") must repeat over this many characters:
/// rules and table separators are layout, a screen of "!!!!" is a loop.
const MIN_SYMBOL_RUN: usize = 160;
/// Longest block of words checked for back-to-back repetition (a long sentence or two).
const MAX_WORD_PERIOD: usize = 64;
/// A block of words must appear this many times in a row...
const MIN_WORD_COPIES: usize = 3;
/// ...and the copies after the first must span this many words, so a short block such as
/// "2 *" in "2 * 2 * 2 * 2" needs many more copies than a repeated sentence.
const MIN_REPEATED_WORDS: usize = 24;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;
const MAX_EVIDENCE_CHARS: usize = 200;

/// Every `max_*` threshold is exclusive: a loop is reported when the count goes above it.
#[derive(Debug, Clone, Serialize)]
pub struct LoopDetectorConfig {
    /// Marker occurrences allowed inside the sliding window.
    pub max_marker_hits: usize,
    /// Case-insensitive, matched on word boundaries and followed by punctuation or a line
    /// break, so that "wait time" or "actually ran" do not count.
    pub markers: Vec<String>,
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
    /// A block of words repeated back to back at the end of the stream.
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
    /// FNV-1a hash of the word being read (lowercased, punctuation kept); None between words.
    word: Option<u64>,
    /// Ring of the last word hashes, indexed by `words_seen % MAX_WORD_PERIOD`.
    words: [u64; MAX_WORD_PERIOD],
    words_seen: usize,
    /// `word_runs[p]`: consecutive words equal to the one `p` words earlier.
    word_runs: [usize; MAX_WORD_PERIOD + 1],
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
            word: None,
            words: [0; MAX_WORD_PERIOD],
            words_seen: 0,
            word_runs: [0; MAX_WORD_PERIOD + 1],
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
        // Periodicity sees every run of whitespace as one space: "1, 1,  1, 1," loops too.
        let pc = if lc.is_whitespace() { ' ' } else { lc };
        let mut period = None;
        if !(pc == ' ' && self.recent.back() == Some(&' ')) {
            period = self.check_period(pc);
            self.recent.push_back(pc);
            if self.recent.len() > self.recent_cap {
                self.recent.pop_front();
            }
        }
        self.chars_seen += 1;
        let ngram = self.feed_word(lc);
        let line = self.feed_line(c);

        if marker.is_some() {
            return marker;
        }
        if self.chars_seen < self.cfg.min_chars_before_check {
            return None;
        }
        // A repeated line also completes a repeated word block: the line is the clearer evidence.
        line.or(ngram).or(period)
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
            // A hesitation is punctuated ("Wait,", "Wait..."); "wait time" is a noun.
            let boundary_after = m.last().is_some_and(|l| !l.is_alphanumeric())
                || !(lc.is_alphanumeric() || lc == ' ' || lc == '\t');
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
            let run = self.period_runs[p];
            if found.is_some() || run < (MIN_PERIOD_REPEATS - 1) * p {
                continue;
            }
            let motif: String = self
                .recent
                .range(len - p + 1..)
                .copied()
                .chain([lc])
                .collect();
            // Separators such as "|---" are layout up to a point. Runs of spaces never get
            // here: `step` collapses them.
            if motif.chars().any(char::is_alphanumeric) || run >= MIN_SYMBOL_RUN {
                found = Some((
                    LoopReason::CharPeriodicity,
                    format!("motif « {motif} » répété {} fois de suite", run / p + 1),
                ));
            }
        }
        found
    }

    /// Same run counters as `check_period`, over word hashes instead of characters.
    fn feed_word(&mut self, lc: char) -> Option<(LoopReason, String)> {
        if !lc.is_whitespace() {
            let h = self.word.unwrap_or(FNV_OFFSET) ^ u64::from(u32::from(lc));
            self.word = Some(h.wrapping_mul(FNV_PRIME));
            return None;
        }
        let h = self.word.take()?;
        let mut found = None;
        for p in 1..=MAX_WORD_PERIOD {
            let repeats =
                self.words_seen >= p && self.words[(self.words_seen - p) % MAX_WORD_PERIOD] == h;
            self.word_runs[p] = if repeats { self.word_runs[p] + 1 } else { 0 };
            let run = self.word_runs[p];
            if found.is_none() && run >= MIN_REPEATED_WORDS && run >= (MIN_WORD_COPIES - 1) * p {
                let tail: String = self.recent.iter().collect();
                found = Some((
                    LoopReason::NgramRepetition,
                    format!(
                        "bloc de {p} mots répété {} fois de suite (fin : « {} »)",
                        run / p + 1,
                        tail.trim()
                    ),
                ));
            }
        }
        self.words[self.words_seen % MAX_WORD_PERIOD] = h;
        self.words_seen += 1;
        found
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
    fn a_template_with_changing_numbers_is_not_a_word_block_loop() {
        // Same shape as "Day 1: 2 pages. Day 2: 4 pages." in a real solution: only an
        // exact back-to-back block counts, so a hesitation loop whose numbers change is
        // left to the token budget.
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
        assert_eq!(d.push(&text), LoopVerdict::Ok);
    }

    /// Former false positive (docs/validation/qwen2.5-0.5b-validation.md, ÉCART-3): a
    /// finished, correct step-by-step answer that reuses one stock phrase five times was
    /// cut by the windowed n-gram count. Fixed by the back-to-back word-block rule,
    /// calibrated on independent data (outputs/reports/loop-calibration.md).
    #[test]
    fn a_step_by_step_answer_reusing_a_stock_phrase_is_not_a_loop() {
        let answer = "To solve the problem, we need to follow these steps:

            1. Calculate the total number of pear trees.
            2. Determine the total number of boxes from the trees and the boxes per tree.

            **Step 1: Calculate the total number of pear trees.**

            The grower has 7 rows of 11 pear trees. Therefore, the total number of pear             trees is 7 x 11 = 77 trees.

            **Step 2: Determine the total number of boxes.**

            Each tree gives 2 boxes of pears, so the grower gets 77 x 2 = 154 boxes.

            #### 154
";
        let mut d = LoopDetector::new(LoopDetectorConfig::default());
        assert_eq!(d.push(answer), LoopVerdict::Ok);
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
            reason_of(&mut d, &"xyz.".repeat(MIN_PERIOD_REPEATS)),
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
        // Different words in each paragraph, so that only the markers repeat.
        let filler = |k: usize| -> String { (0..50).map(|i| format!("mot{k}x{i} ")).collect() };
        let text: String = (0..5).map(|k| format!("Wait. {}", filler(k))).collect();

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

    /// Structured but healthy solutions: every one repeats words, lines or shapes, never
    /// one block back to back.
    #[test]
    fn multi_step_math_with_repetitive_structure_is_not_a_loop() {
        let steps: String = (1..=6)
            .map(|i| {
                format!(
                    "**Step {i}: Calculate the total number of apples in box {i}.**\n\n\
                     The total number of apples in box {i} is {i} x 12 = {} apples.\n\n",
                    i * 12
                )
            })
            .collect();
        let doubling: String = (1..=8)
            .map(|d| format!("Day {d}: {} pages\n", 1 << d))
            .collect();
        let list: String = ["apples", "pears", "plums", "figs", "limes", "kiwis"]
            .iter()
            .enumerate()
            .map(|(i, f)| {
                format!(
                    "{}. The cost of the {f} is {} * 2 = {} dollars.\n",
                    i + 1,
                    i + 3,
                    2 * (i + 3)
                )
            })
            .collect();
        let solutions = [
            format!("{steps}Adding the boxes: 12 + 24 + 36 + 48 + 60 + 72 = 252.\n\n#### 252"),
            format!("Let's list the pages read each day.\n\n{doubling}\n#### 256"),
            "The number of grains doubles on every square of the board, starting from one \
             grain. On the ninth square there are 2^8 grains, so 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2 \
             = 256 grains.\n\n#### 256"
                .into(),
            format!("Here is the cost of each fruit:\n\n{list}\nTotal = 6 + 8 + 10 + 12 + 14 + 16 = 66\nTotal = 66 dollars\n\n#### 66"),
            "| Item | Price | Count | Cost |\n|------|-------|-------|------|\n| pen | 2 | 3 | 6 |\n\
             | cup | 5 | 2 | 10 |\n| hat | 7 | 1 | 7 |\n|------|-------|-------|------|\n\n\
             ---\n\nTotal: 6 + 10 + 7 = 23\n\n#### 23"
                .into(),
            "x = 3\ny = 3\nz = 3\n\nSo x = y = z = 3 and x + y + z = 3 + 3 + 3 = 9.\n\n#### 9".into(),
            "1. **First dose wait time:** 20 minutes.\n2. **Second dose wait time:** half of \
             the first wait time, 10 minutes.\n3. **Total wait time:** the first wait time \
             plus the second wait time, 20 + 10 = 30 minutes.\n\n#### 30"
                .into(),
        ];
        let mut d = LoopDetector::new(LoopDetectorConfig::default());
        for text in &solutions {
            d.reset();
            assert_eq!(d.push(text), LoopVerdict::Ok, "{text}");
        }
    }

    /// Degenerate loops of 1-2, 4-8 and 12-20 tokens after a healthy start: each is
    /// caught, by the expected rule, within a few copies of the motif.
    #[test]
    fn short_medium_and_long_loops_are_cut_early() {
        // Longer than `min_chars_before_check`, so the count starts with the loop.
        let prefix = "First, one notebook costs 12 / 4 = 3 dollars, and Mary buys 4 of them for her class.\n\n";
        let cases = [
            (" the", LoopReason::CharPeriodicity, 13),
            ("!", LoopReason::CharPeriodicity, 164),
            ("<br>", LoopReason::CharPeriodicity, 13),
            ("The answer is 3. ", LoopReason::NgramRepetition, 7),
            ("Total = 3 + 3 = 6\n", LoopReason::IdenticalLines, 4),
            (
                "So the total cost of the notebooks is 3 dollars per notebook times 4 notebooks. ",
                LoopReason::NgramRepetition,
                3,
            ),
            (
                "Therefore, 3 x 4 = 12 and 12 / 4 = 3, so each notebook costs 3 dollars, which is the answer we need. ",
                LoopReason::NgramRepetition,
                3,
            ),
        ];
        let mut d = LoopDetector::new(LoopDetectorConfig::default());
        for (motif, expected, max_copies) in cases {
            d.reset();
            let text = format!("{prefix}{}", motif.repeat(300));
            let v = d.push(&text);
            let LoopVerdict::Loop {
                reason, at_char, ..
            } = v
            else {
                panic!("missed loop on {motif:?}")
            };
            assert_eq!(reason, expected, "{motif:?}");
            let copies = (at_char - prefix.chars().count()).div_ceil(motif.chars().count());
            assert!(copies <= max_copies, "{motif:?} cut after {copies} copies");
        }
    }

    #[test]
    fn a_word_block_loop_is_found_whatever_the_chunking() {
        let text = format!(
            "We start from 5 apples. {}",
            "Then Mary gives two apples to John and John gives them back to Mary. ".repeat(4)
        );
        let chars: Vec<char> = text.chars().collect();
        let mut d = LoopDetector::new(LoopDetectorConfig::default());
        let whole = d.push(&text);
        assert_eq!(reason(&whole), Some(LoopReason::NgramRepetition));
        for size in [1, 2, 5] {
            d.reset();
            let v = chars
                .chunks(size)
                .map(|piece| d.push(&piece.iter().collect::<String>()))
                .last()
                .unwrap();
            assert_eq!(v, whole, "chunk size {size}");
        }
    }
}
