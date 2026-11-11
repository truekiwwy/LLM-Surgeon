//! `surgeon bench eval`: greedy generation per question, answer extraction, exact match.
//!
//! Normalization (both sides): `\left`/`\right`, `\!`, `\,`, `\;`, `$`, `%`, `^\circ`
//! removed, `\dfrac`/`\tfrac` -> `\frac`, `\text{X}` -> `X`, whitespace and a final `.`
//! removed, a trailing unit word dropped after a number (`18dollars` -> `18`). When both
//! sides read as numbers (`18`, `18.0`, `1,8e1`, `1,000`, `1/2`, `\frac{1}{2}`) they are
//! compared numerically with a relative tolerance of 1e-9 (so `0.1+0.2` style rounding
//! never fails a match, while `12` vs `120` always does); otherwise the normalized
//! strings must be equal. A thousands group is a comma followed by exactly 3 digits;
//! a single other comma is a decimal comma.
//!
//! Pass@1 is the exact-match rate of one greedy sample per question. The 95 % interval is
//! Wilson's score interval, which stays inside [0, 1] and is honest on small samples.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use regex::Regex;

use super::dataset::{self, DatasetFormat, Item};
use super::error::{BenchError, Result};
use super::extract_answer::{extract, number_re};
use super::generation::{by_prompt, GenOptions};
use super::prompt::Template;
use super::report::{
    BenchEvalReport, BenchRun, EvalBaseline, EvalScore, EvalTiming, ItemResult, PairedComparison,
    Verdict,
};
use super::sha256_file;
use crate::eval::{
    ensure_gguf, truncate_chars, CommandRunner, EvalError, LoopDetector, LoopDetectorConfig,
    LoopVerdict, TextGenerator, LLAMA_DIR_REQUIRED,
};

/// Fixed seed of the greedy decoding (greedy ignores it, set anyway for reproducibility).
pub const SEED: u32 = 42;
const EXCERPT_CHARS: usize = 200;
const SMALL_SAMPLE: usize = 100;
const Z_95: f64 = 1.959_963_984_540_054;
const REL_TOLERANCE: f64 = 1e-9;

fn unit_re() -> Option<&'static Regex> {
    static RE: OnceLock<Option<Regex>> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^([-+]?[0-9][0-9.,]*(?:[eE][-+]?[0-9]+)?)[A-Za-z]+$").ok())
        .as_ref()
}

/// `\text{X}` -> `X`, every occurrence.
fn unwrap_text(mut s: String) -> String {
    while let Some(at) = s.find("\\text{") {
        let open = at + "\\text".len();
        let mut depth = 0usize;
        let mut close = None;
        for (i, c) in s[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth = depth.saturating_sub(1);
                    if depth == 0 {
                        close = Some(open + i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(close) = close else { break };
        let inner = s[open + 1..close].to_owned();
        s.replace_range(at..=close, &inner);
    }
    s
}

pub fn normalize(answer: &str) -> String {
    let mut s = answer.trim().to_owned();
    for (from, to) in [
        ("\\left", ""),
        ("\\right", ""),
        ("\\!", ""),
        ("\\,", ""),
        ("\\;", ""),
        ("\\dfrac", "\\frac"),
        ("\\tfrac", "\\frac"),
        ("\\$", ""),
        ("$", ""),
        ("\\%", ""),
        ("%", ""),
        ("^{\\circ}", ""),
        ("^\\circ", ""),
    ] {
        s = s.replace(from, to);
    }
    s = unwrap_text(s);
    s.retain(|c| !c.is_whitespace());
    while s.ends_with('.') {
        s.pop();
    }
    if let Some(number) = unit_re()
        .and_then(|re| re.captures(&s))
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_owned())
        .filter(|n| plain_number(n).is_some())
    {
        s = number;
    }
    s
}

fn plain_number(s: &str) -> Option<f64> {
    let full = number_re()?.find(s).filter(|m| m.range() == (0..s.len()))?;
    let text = full.as_str();
    let commas = text.matches(',').count();
    let digits = match (commas, text.contains('.')) {
        (0, _) => text.to_owned(),
        // `number_re` only lets a dot coexist with commas in a thousands group.
        (_, true) => text.replace(',', ""),
        (1, false)
            if text
                .split(',')
                .nth(1)
                .is_some_and(|d| d.len() == 3 && !d.contains(['e', 'E'])) =>
        {
            text.replace(',', "")
        }
        (1, false) => text.replace(',', "."),
        _ => text.replace(',', ""),
    };
    digits.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// `\frac{a}{b}` or `\frac12`, optionally negated.
fn latex_fraction(s: &str) -> Option<f64> {
    let (sign, rest) = match s.strip_prefix('-') {
        Some(r) => (-1.0, r),
        None => (1.0, s),
    };
    let body = rest.strip_prefix("\\frac")?;
    let (num, den) = match body.strip_prefix('{') {
        Some(b) => {
            let (num, after) = b.split_once('}')?;
            let den = after.strip_prefix('{')?.strip_suffix('}')?;
            (num, den)
        }
        None if body.len() == 2 && body.is_ascii() => body.split_at(1),
        None => return None,
    };
    let (n, d) = (plain_number(num)?, plain_number(den)?);
    (d != 0.0).then(|| sign * n / d)
}

/// The value of a normalized answer, when it is a number or a fraction.
pub fn parse_number(s: &str) -> Option<f64> {
    let s = s.strip_prefix('+').unwrap_or(s);
    if let Some(v) = latex_fraction(s) {
        return Some(v);
    }
    if let Some((n, d)) = s.split_once('/') {
        let (n, d) = (plain_number(n)?, plain_number(d)?);
        return (d != 0.0).then(|| n / d);
    }
    plain_number(s)
}

fn close(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= REL_TOLERANCE * a.abs().max(b.abs())
}

pub fn answers_match(extracted: &str, gold: &str) -> bool {
    let (a, b) = (normalize(extracted), normalize(gold));
    match (parse_number(&a), parse_number(&b)) {
        (Some(x), Some(y)) => close(x, y),
        _ => a == b,
    }
}

/// Wilson score interval at 95 % for `correct` successes out of `n` (n >= 1).
pub fn wilson(correct: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let (k, n) = (correct as f64, n as f64);
    let p = k / n;
    let z2 = Z_95 * Z_95;
    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let half = Z_95 * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denom;
    // At p = 0 (p = 1) the lower (upper) bound is exactly 0 (1); rounding would miss it.
    let low = if correct == 0 {
        0.0
    } else {
        (center - half).max(0.0)
    };
    let high = if k == n {
        1.0
    } else {
        (center + half).min(1.0)
    };
    (low, high)
}

pub fn score(items: &[ItemResult]) -> EvalScore {
    let n = items.len();
    let correct = items.iter().filter(|i| i.correct).count();
    let (ci95_low, ci95_high) = wilson(correct, n);
    let count = |reason: &str| items.iter().filter(|i| i.reason == reason).count();
    EvalScore {
        n,
        correct,
        accuracy: if n == 0 {
            0.0
        } else {
            correct as f64 / n as f64
        },
        ci95_low,
        ci95_high,
        loops: items.iter().filter(|i| i.loop_detected).count(),
        no_answer: count("no_answer"),
        timeouts: count("timeout"),
    }
}

/// Scores one generated output. A loop is always wrong, whatever was extracted.
pub fn judge(item: &Item, output: &str, loop_detected: bool) -> ItemResult {
    let extracted = extract(output);
    let correct = !loop_detected
        && extracted
            .as_ref()
            .is_some_and(|e| answers_match(&e.answer, &item.gold));
    let reason = match (&extracted, loop_detected, correct) {
        (_, true, _) => "loop",
        (None, false, _) => "no_answer",
        (Some(_), false, true) => "correct",
        (Some(_), false, false) => "wrong",
    };
    ItemResult {
        id: item.id.clone(),
        gold: normalize(&item.gold),
        rule: extracted.as_ref().map(|e| e.rule.as_str().to_owned()),
        extracted: extracted.map(|e| e.answer),
        correct,
        reason: reason.to_owned(),
        loop_detected,
        output_excerpt: truncate_chars(output, EXCERPT_CHARS).to_owned(),
        sample: None,
    }
}

/// Every question in order. A timeout fails that question only; any other generator
/// error (process crash, missing binary) stops the suite.
///
/// Every (question, sample) goes through [`TextGenerator::generate_many`] (batched by the
/// native backend, concurrent with llama-server), each with its own loop detector fed by
/// its own stream, so a loop is cut exactly as in the sequential run.
pub fn run_suite(
    generator: &mut dyn TextGenerator,
    items: &[Item],
    template: &Template,
    max_new_tokens: u32,
    generation: &GenOptions,
    cancel: &AtomicBool,
    on_question: &mut dyn FnMut(usize, usize),
) -> Result<Vec<ItemResult>> {
    let samples = generation.samples;
    let jobs = generation.jobs(items.iter().map(|item| template.render(&item.question)));
    let total = jobs.len();
    let mut detectors: Vec<LoopDetector> = (0..total)
        .map(|_| LoopDetector::new(LoopDetectorConfig::default()))
        .collect();
    let mut looped = vec![false; total];
    let mut done = 0;
    let outcomes = generator.generate_many(
        &jobs,
        max_new_tokens,
        cancel,
        &mut |j, chunk| match detectors[j].push(chunk) {
            LoopVerdict::Loop { .. } => {
                looped[j] = true;
                std::ops::ControlFlow::Break(())
            }
            LoopVerdict::Ok => std::ops::ControlFlow::Continue(()),
        },
        &mut |_| {
            done += 1;
            on_question(done, total);
        },
    )?;
    if cancel.load(Ordering::SeqCst) {
        return Err(BenchError::Cancelled);
    }
    let mut results = Vec::with_capacity(total);
    for (j, outcome) in outcomes.into_iter().enumerate() {
        let item = &items[j / samples];
        let mut result = match outcome {
            Ok(out) => judge(item, &out.text, looped[j]),
            Err(EvalError::Timeout { .. }) => ItemResult {
                id: item.id.clone(),
                gold: normalize(&item.gold),
                extracted: None,
                rule: None,
                correct: false,
                reason: "timeout".into(),
                loop_detected: false,
                output_excerpt: String::new(),
                sample: None,
            },
            Err(e) => return Err(e.into()),
        };
        result.sample = (samples > 1).then_some(j % samples);
        results.push(result);
    }
    if results.len() < total {
        return Err(BenchError::Cancelled);
    }
    Ok(results)
}

pub fn paired(main: &[ItemResult], baseline: &[ItemResult]) -> PairedComparison {
    let mut p = PairedComparison {
        both_correct: 0,
        both_wrong: 0,
        lost: Vec::new(),
        gained: Vec::new(),
        delta_accuracy: score(main).accuracy - score(baseline).accuracy,
    };
    for (m, b) in main.iter().zip(baseline) {
        match (b.correct, m.correct) {
            (true, true) => p.both_correct += 1,
            (false, false) => p.both_wrong += 1,
            (true, false) => p.lost.push(m.id.clone()),
            (false, true) => p.gained.push(m.id.clone()),
        }
    }
    p
}

/// What generates the answers of `bench eval`.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
    clap::ValueEnum,
)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// llama.cpp (`llama-completion` / `llama-cli`) sur un GGUF, converti au besoin.
    #[default]
    Llama,
    /// Génération en Rust pur sur CPU, sur un dossier SafeTensors (forward natif).
    Native,
    /// llama.cpp `llama-completion` / `llama-cli`, un processus par génération (comme llama).
    #[serde(rename = "llama-completion")]
    LlamaCompletion,
    /// llama.cpp `llama-server` : un serveur local, requêtes concurrentes, GPU selon le build.
    #[serde(rename = "llama-server")]
    LlamaServer,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Backend::Llama => "llama",
            Backend::Native => "native",
            Backend::LlamaCompletion => "llama-completion",
            Backend::LlamaServer => "llama-server",
        }
    }

    /// Runs llama.cpp on a GGUF (a SafeTensors model is converted first).
    pub fn is_llama(self) -> bool {
        self != Backend::Native
    }
}

#[derive(Debug, Clone)]
pub struct EvalRequest {
    pub model: PathBuf,
    pub dataset: PathBuf,
    pub backend: Backend,
    pub baseline_model: Option<PathBuf>,
    pub limit: usize,
    pub max_new_tokens: u32,
    pub template: Template,
    pub format: DatasetFormat,
    pub llama_cpp_dir: Option<PathBuf>,
    /// Per question.
    pub timeout: Option<Duration>,
    pub min_accuracy: Option<f64>,
    pub generation: GenOptions,
}

/// Builds a generator for a model: a GGUF with [`Backend::Llama`] (the app's
/// `generator_factory`), the SafeTensors directory itself with [`Backend::Native`].
pub type MakeGenerator<'a> = dyn FnMut(&Path) -> Result<Box<dyn TextGenerator>> + 'a;

fn gguf_error(e: EvalError) -> BenchError {
    match e {
        EvalError::InvalidConfig(m) if m == LLAMA_DIR_REQUIRED => BenchError::InvalidArgument(m),
        other => BenchError::Conversion(other.to_string()),
    }
}

/// Baseline first (when given), then the evaluated model, on the same questions.
pub fn run(
    req: &EvalRequest,
    runner: &dyn CommandRunner,
    make_generator: &mut MakeGenerator<'_>,
    cancel: &AtomicBool,
    on_question: &mut dyn FnMut(usize, usize),
) -> Result<BenchRun<BenchEvalReport>> {
    if req.max_new_tokens == 0 {
        return Err(BenchError::InvalidArgument(
            "--max-new-tokens doit être >= 1".into(),
        ));
    }
    if req
        .min_accuracy
        .is_some_and(|m| !(m.is_finite() && (0.0..=1.0).contains(&m)))
    {
        return Err(BenchError::InvalidArgument(
            "--min-accuracy doit être entre 0 et 1".into(),
        ));
    }
    for model in std::iter::once(&req.model).chain(&req.baseline_model) {
        if !model.exists() {
            return Err(BenchError::input(model, "modèle introuvable"));
        }
    }
    let items = dataset::load(&req.dataset, req.format, req.limit)?;
    let dataset_sha256 = sha256_file(&req.dataset)?;
    let mut evaluate = |model: &Path, tag: &str| -> Result<(Vec<ItemResult>, u64)> {
        let started = Instant::now();
        // Kept until the suite ends: a converted GGUF is a temporary file.
        let gguf = match req.backend.is_llama() {
            true => Some(
                ensure_gguf(model, req.llama_cpp_dir.as_deref(), tag, runner)
                    .map_err(gguf_error)?,
            ),
            false => None,
        };
        let mut generator = make_generator(gguf.as_ref().map_or(model, |g| g.path()))?;
        generator.configure(&req.generation.decoding(req.timeout));
        let results = run_suite(
            generator.as_mut(),
            &items,
            &req.template,
            req.max_new_tokens,
            &req.generation,
            cancel,
            on_question,
        )?;
        Ok((results, started.elapsed().as_millis() as u64))
    };
    let baseline = match &req.baseline_model {
        Some(model) => Some((model.clone(), evaluate(model, "bench-eval-base")?)),
        None => None,
    };
    let (results, total_ms) = evaluate(&req.model, "bench-eval")?;
    let main = score(&results);

    let mut warnings = Vec::new();
    let mut threshold_warnings = Vec::new();
    if main.n < SMALL_SAMPLE {
        warnings.push(format!(
            "échantillon réduit (n = {}) : l'intervalle de confiance est large (± {:.1} points)",
            main.n,
            (main.ci95_high - main.ci95_low) / 2.0 * 100.0
        ));
    }
    if let Some(min) = req.min_accuracy {
        if main.accuracy < min {
            threshold_warnings.push(format!(
                "exactitude {:.1} % < minimum {:.1} %",
                main.accuracy * 100.0,
                min * 100.0
            ));
        }
    }
    warnings.extend(threshold_warnings.iter().cloned());
    // Several samples per question are not independent pairs: per-question rates instead.
    let samples = req.generation.samples;
    let paired_cmp = baseline
        .as_ref()
        .filter(|_| samples == 1)
        .map(|(_, (base, _))| paired(&results, base));
    let hits = |items: &[ItemResult]| by_prompt(items.iter().map(|i| i.correct), samples);
    let generation = (!req.generation.is_historical(req.backend == Backend::Llama)).then(|| {
        req.generation.report(
            req.backend.label(),
            Some(&hits(&results)),
            baseline.as_ref().map(|(_, (b, _))| hits(b)).as_deref(),
        )
    });
    let report = BenchEvalReport {
        model: req.model.clone(),
        dataset: req.dataset.clone(),
        dataset_sha256,
        format: req.format.as_str().into(),
        prompt_template_sha256: req.template.sha256(),
        max_new_tokens: req.max_new_tokens,
        seed: SEED,
        score: main,
        items: results,
        timing_ms: EvalTiming {
            total: total_ms,
            baseline: baseline.as_ref().map(|(_, (_, ms))| *ms),
        },
        baseline: baseline.map(|(model, (items, _))| EvalBaseline {
            model,
            score: score(&items),
            items,
        }),
        paired: paired_cmp,
        min_accuracy: req.min_accuracy,
        verdict: match (req.min_accuracy, threshold_warnings.is_empty()) {
            (None, _) => Verdict::None,
            (Some(_), true) => Verdict::Ok,
            (Some(_), false) => Verdict::Warn,
        },
        generation,
    };
    Ok(BenchRun {
        report,
        warnings,
        threshold_warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_and_string_comparison() {
        for (a, b) in [
            ("18", "18.0"),
            ("18", "1,8e1"),
            ("1,000", "1000"),
            ("1/2", "0.5"),
            ("\\frac{1}{2}", "0.5"),
            ("\\dfrac{1}{2}", "1/2"),
            ("\\frac12", "0.5"),
            ("-\\frac{3}{4}", "-0.75"),
            ("$1,234.50", "1234.5"),
            ("0.30000000000000004", "0.3"),
            ("18 dollars", "18"),
            ("50\\%", "50"),
            ("90^\\circ", "90"),
            ("\\text{(B)}", "(B)"),
            ("\\left(1,2\\right)", "(1, 2)"),
            ("5.", "5"),
        ] {
            assert!(answers_match(a, b), "{a} = {b}");
        }
        for (a, b) in [
            ("12", "120"),
            ("1/3", "0.33"),
            ("(1,2)", "(2,1)"),
            ("blue", "red"),
        ] {
            assert!(!answers_match(a, b), "{a} != {b}");
        }
        assert_eq!(parse_number("1,8e1"), Some(18.0));
        assert_eq!(parse_number("1,5"), Some(1.5));
        assert_eq!(parse_number("12,345,678"), Some(12_345_678.0));
        assert_eq!(parse_number("1/0"), None);
        assert_eq!(parse_number("inf"), None);
    }

    use crate::eval::{CommandOutput, GenerationOutcome};
    use std::ops::ControlFlow;

    /// Replies from a script, in question order, streamed in small chunks.
    struct Scripted {
        replies: Vec<std::result::Result<&'static str, EvalError>>,
        next: usize,
        configured: bool,
    }

    impl Scripted {
        fn new(replies: Vec<std::result::Result<&'static str, EvalError>>) -> Self {
            Self {
                replies,
                next: 0,
                configured: false,
            }
        }
    }

    impl TextGenerator for Scripted {
        fn generate(
            &mut self,
            _prompt: &str,
            _max_new_tokens: u32,
            on_chunk: &mut dyn FnMut(&str) -> ControlFlow<()>,
        ) -> crate::eval::Result<GenerationOutcome> {
            assert!(self.configured, "configure() is called before generate()");
            let reply = self.replies[self.next].clone();
            self.next += 1;
            let text = reply?;
            let mut sent = String::new();
            for chunk in text.split_inclusive(' ') {
                sent.push_str(chunk);
                if on_chunk(chunk).is_break() {
                    break;
                }
            }
            Ok(GenerationOutcome {
                text: sent,
                stopped_early: false,
                duration_ms: 0,
            })
        }

        fn configure(&mut self, decoding: &crate::eval::Decoding) -> bool {
            assert_eq!(decoding.seed, SEED);
            self.configured = true;
            true
        }
    }

    struct NoRunner;
    impl CommandRunner for NoRunner {
        fn run(
            &self,
            _: &Path,
            _: &[String],
            _: Option<&Path>,
        ) -> crate::eval::Result<CommandOutput> {
            panic!("a .gguf model needs no conversion")
        }
    }

    const LOOP: &str = "Wait, let me see. Wait, again. Wait, hmm. Wait, Wait, Wait, Wait.";

    fn fixture() -> (tempfile::TempDir, EvalRequest) {
        let dir = tempfile::tempdir().unwrap();
        for name in ["after.gguf", "before.gguf"] {
            std::fs::write(dir.path().join(name), b"GGUF").unwrap();
        }
        let dataset = dir.path().join("data.json");
        std::fs::write(
            &dataset,
            r#####"[{"id":"q1","question":"2+3?","answer":"2+3=5\n#### 5"},
                {"id":"q2","question":"half of one?","answer":"$\\boxed{\\frac{1}{2}}$"},
                {"id":"q3","question":"10*10?","answer":"#### 100"},
                {"id":"q4","question":"color?","answer":"#### 7"},
                {"id":"q5","question":"slow?","answer":"#### 1"}]"#####,
        )
        .unwrap();
        let req = EvalRequest {
            model: dir.path().join("after.gguf"),
            dataset,
            backend: Backend::Llama,
            baseline_model: None,
            limit: 50,
            max_new_tokens: 64,
            template: Template::default(),
            format: DatasetFormat::Auto,
            llama_cpp_dir: None,
            timeout: None,
            min_accuracy: None,
            generation: Default::default(),
        };
        (dir, req)
    }

    fn after_replies() -> Vec<std::result::Result<&'static str, EvalError>> {
        vec![
            Ok("2 plus 3 is 5.\n#### 5"),
            Ok("So the answer is \\boxed{0.5}"),
            Ok(LOOP),
            Ok("I cannot tell."),
            Err(EvalError::Timeout { after_ms: 10 }),
        ]
    }

    fn evaluate(
        req: &EvalRequest,
        scripts: Vec<Vec<std::result::Result<&'static str, EvalError>>>,
    ) -> Result<BenchRun<BenchEvalReport>> {
        let mut scripts = scripts.into_iter();
        let mut make = |_: &Path| -> Result<Box<dyn TextGenerator>> {
            Ok(Box::new(Scripted::new(scripts.next().unwrap())))
        };
        let never = AtomicBool::new(false);
        let mut progress = Vec::new();
        let run = run(req, &NoRunner, &mut make, &never, &mut |d, t| {
            progress.push((d, t))
        });
        if run.is_ok() {
            assert_eq!(progress.last(), Some(&(5, 5)));
        }
        run
    }

    #[test]
    fn suite_scores_loops_no_answers_and_timeouts() {
        let (_dir, mut req) = fixture();
        req.min_accuracy = Some(0.5);
        let run = evaluate(&req, vec![after_replies()]).unwrap();
        let r = &run.report;
        assert_eq!((r.score.n, r.score.correct), (5, 2));
        assert_eq!(r.score.accuracy, 0.4);
        assert_eq!(
            (r.score.loops, r.score.no_answer, r.score.timeouts),
            (1, 1, 1)
        );
        let reasons: Vec<&str> = r.items.iter().map(|i| i.reason.as_str()).collect();
        assert_eq!(
            reasons,
            ["correct", "correct", "loop", "no_answer", "timeout"]
        );
        assert!(r.items[2].loop_detected && !r.items[2].correct);
        assert_eq!(r.items[1].rule.as_deref(), Some("boxed"));
        assert_eq!(r.verdict, Verdict::Warn);
        assert_eq!(run.threshold_warnings.len(), 1);
        assert!(run
            .warnings
            .iter()
            .any(|w| w.contains("échantillon réduit")));
        let (lo, hi) = wilson(2, 5);
        assert_eq!((r.score.ci95_low, r.score.ci95_high), (lo, hi));
        assert!(r.baseline.is_none() && r.paired.is_none());

        req.min_accuracy = Some(0.4);
        let run = evaluate(&req, vec![after_replies()]).unwrap();
        assert_eq!(run.report.verdict, Verdict::Ok, "equal is not below");
    }

    #[test]
    fn paired_comparison_and_determinism() {
        let (dir, mut req) = fixture();
        req.baseline_model = Some(dir.path().join("before.gguf"));
        let baseline = vec![
            Ok("#### 5"),
            Ok("#### 3"),
            Ok("#### 100"),
            Ok("#### 9"),
            Ok("#### 1"),
        ];
        let run = evaluate(&req, vec![baseline.clone(), after_replies()]).unwrap();
        let p = run.report.paired.as_ref().unwrap();
        assert_eq!((p.both_correct, p.both_wrong), (1, 1));
        assert_eq!(p.lost, ["q3", "q5"]);
        assert_eq!(p.gained, ["q2"]);
        assert!((p.delta_accuracy - (0.4 - 0.6)).abs() < 1e-12);
        assert_eq!(run.report.baseline.as_ref().unwrap().score.correct, 3);

        let json = |run: &BenchRun<BenchEvalReport>| {
            let mut v = serde_json::to_value(&run.report).unwrap();
            v.as_object_mut().unwrap().remove("timing_ms");
            v.to_string()
        };
        let again = evaluate(&req, vec![baseline, after_replies()]).unwrap();
        assert_eq!(json(&run), json(&again));
    }

    #[test]
    fn broken_generator_stops_and_inputs_are_checked() {
        let (_dir, req) = fixture();
        let crash = vec![Err(EvalError::CommandFailed {
            step: "llama-cli".into(),
            code: 1,
            stderr_tail: "model load failed".into(),
        })];
        let err = evaluate(&req, vec![crash]).unwrap_err();
        assert!(matches!(err, BenchError::ProcessFailed { .. }), "{err:?}");
        let mut bad = req.clone();
        bad.min_accuracy = Some(1.5);
        assert!(matches!(
            evaluate(&bad, vec![]),
            Err(BenchError::InvalidArgument(_))
        ));
        let mut missing = req.clone();
        missing.model = missing.model.with_file_name("absent.gguf");
        assert!(matches!(
            evaluate(&missing, vec![]),
            Err(BenchError::Input { .. })
        ));
    }

    #[test]
    fn judge_ignores_end_markers_units_and_percent() {
        for (gold, output) in [
            ("18", "#### 18 [end of text]"),
            ("160", "#### 160 minutes [end of text]"),
            ("60", "#### 60% [end of text]"),
        ] {
            let item = Item {
                id: "x".into(),
                question: "q".into(),
                gold: gold.into(),
            };
            let r = judge(&item, output, false);
            assert!(r.correct, "{output}: {r:?}");
            assert!(!r.extracted.unwrap().contains("end of text"));
        }
    }

    #[test]
    fn scoring_a_thousand_outputs_is_fast() {
        let item = Item {
            id: "x".into(),
            question: "q".into(),
            gold: "1,234".into(),
        };
        let output = "Let us compute step by step. 600 + 634 = 1234. ".repeat(20) + "\n#### 1,234";
        let started = Instant::now();
        let correct = (0..1000)
            .filter(|_| judge(&item, &output, false).correct)
            .count();
        eprintln!(
            "1000 sorties extraites et notées en {:?}",
            started.elapsed()
        );
        assert_eq!(correct, 1000);
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    /// Independent Wilson reference: solve |p − p̂| = z·sqrt(p(1−p)/n) for p.
    fn wilson_reference(k: f64, n: f64) -> (f64, f64) {
        let z = 1.959_963_984_540_054_f64;
        let phat = k / n;
        let a = 1.0 + z * z / n;
        let b = -(2.0 * phat + z * z / n);
        let c = phat * phat;
        let disc = (b * b - 4.0 * a * c).max(0.0).sqrt();
        ((-b - disc) / (2.0 * a), (-b + disc) / (2.0 * a))
    }

    #[test]
    fn wilson_matches_an_independent_reference() {
        let (lo, hi) = wilson(40, 50);
        assert!(
            (lo - 0.6696).abs() < 1e-3 && (hi - 0.8876).abs() < 1e-3,
            "{lo} {hi}"
        );
        for (k, n) in [(40, 50), (0, 7), (7, 7), (1, 1), (0, 1), (13, 100)] {
            let (lo, hi) = wilson(k, n);
            let (rlo, rhi) = wilson_reference(k as f64, n as f64);
            assert!((lo - rlo.max(0.0)).abs() < 1e-12 && (hi - rhi.min(1.0)).abs() < 1e-12);
            assert!((0.0..=1.0).contains(&lo) && lo <= hi && hi <= 1.0);
        }
        assert_eq!(wilson(0, 10).0, 0.0);
        assert_eq!(wilson(10, 10).1, 1.0);
    }
}
