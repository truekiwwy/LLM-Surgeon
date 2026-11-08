use std::ops::ControlFlow;
use std::time::Instant;

use serde::Serialize;

use super::error::{EvalError, Result};
use super::generator::TextGenerator;
use super::loop_detector::{LoopDetector, LoopDetectorConfig, LoopReason, LoopVerdict};
use super::prompts::EvalPrompt;
use super::truncate_chars;

const EXCERPT_CHARS: usize = 500;

#[derive(Debug, Clone, Serialize)]
pub struct EvalConfig {
    pub max_new_tokens: u32,
    pub loop_cfg: LoopDetectorConfig,
    /// Stop at the first non-passing prompt.
    pub fail_fast: bool,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            loop_cfg: LoopDetectorConfig::default(),
            fail_fast: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PromptStatus {
    Pass,
    LoopDetected { reason: LoopReason },
    ExpectationFailed { details: String },
    GeneratorError { message: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct PromptResult {
    pub id: String,
    #[serde(flatten)]
    pub status: PromptStatus,
    /// First 500 characters, cut on a character boundary.
    pub output_excerpt: String,
    pub chars_generated: usize,
    pub duration_ms: u128,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvalReport {
    pub results: Vec<PromptResult>,
    pub passed: usize,
    pub failed: usize,
    pub verdict: EvalVerdict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalVerdict {
    /// Every prompt passed.
    Healthy,
    /// Some failures, loops on at most 30% of the prompts.
    Degraded,
    /// Loops on more than 30% of the prompts.
    Broken,
}

/// Generator errors are recorded per prompt and never stop the run (unless `fail_fast`).
/// With `fail_fast`, the verdict covers only the prompts that ran.
pub fn run_eval(
    generator: &mut dyn TextGenerator,
    prompts: &[EvalPrompt],
    cfg: &EvalConfig,
) -> Result<EvalReport> {
    // An empty run would be "Healthy" and could export an untested model.
    if prompts.is_empty() {
        return Err(EvalError::InvalidConfig("aucun prompt à évaluer".into()));
    }
    let mut detector = LoopDetector::new(cfg.loop_cfg.clone());
    let mut results = Vec::with_capacity(prompts.len());

    for prompt in prompts {
        detector.reset();
        let started = Instant::now();
        let mut streamed = String::new();
        let mut loop_reason = None;
        let outcome = generator.generate(&prompt.text, cfg.max_new_tokens, &mut |chunk| {
            streamed.push_str(chunk);
            match detector.push(chunk) {
                LoopVerdict::Ok => ControlFlow::Continue(()),
                LoopVerdict::Loop { reason, .. } => {
                    loop_reason = Some(reason);
                    ControlFlow::Break(())
                }
            }
        });

        let (text, status) = match (loop_reason, outcome) {
            (Some(reason), outcome) => (
                outcome.map_or(streamed, |o| o.text),
                PromptStatus::LoopDetected { reason },
            ),
            (None, Err(e)) => (
                streamed,
                PromptStatus::GeneratorError {
                    message: e.to_string(),
                },
            ),
            (None, Ok(o)) => {
                let status = check_expectations(prompt, &o.text);
                (o.text, status)
            }
        };
        let failed = status != PromptStatus::Pass;
        results.push(PromptResult {
            id: prompt.id.clone(),
            status,
            output_excerpt: truncate_chars(&text, EXCERPT_CHARS).to_owned(),
            chars_generated: text.chars().count(),
            duration_ms: started.elapsed().as_millis(),
        });
        if failed && cfg.fail_fast {
            break;
        }
    }

    let passed = results
        .iter()
        .filter(|r| r.status == PromptStatus::Pass)
        .count();
    let loops = results
        .iter()
        .filter(|r| matches!(r.status, PromptStatus::LoopDetected { .. }))
        .count();
    let failed = results.len() - passed;
    let verdict = match () {
        _ if failed == 0 => EvalVerdict::Healthy,
        // Integer form of loops / total > 30%.
        _ if loops * 10 > results.len() * 3 => EvalVerdict::Broken,
        _ => EvalVerdict::Degraded,
    };
    Ok(EvalReport {
        results,
        passed,
        failed,
        verdict,
    })
}

fn check_expectations(prompt: &EvalPrompt, text: &str) -> PromptStatus {
    let haystack = text.to_lowercase();
    let mut problems = Vec::new();
    for needle in prompt.expect_contains.iter().flatten() {
        if !haystack.contains(&needle.to_lowercase()) {
            problems.push(format!("attendu « {needle} » absent"));
        }
    }
    for needle in prompt.expect_not_contains.iter().flatten() {
        if haystack.contains(&needle.to_lowercase()) {
            problems.push(format!("interdit « {needle} » présent"));
        }
    }
    if problems.is_empty() {
        PromptStatus::Pass
    } else {
        PromptStatus::ExpectationFailed {
            details: problems.join(" ; "),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::generator::GenerationOutcome;
    use std::collections::VecDeque;

    enum Behavior {
        Say(&'static str),
        Endless(&'static str),
        Fail,
    }

    const GUARD_CHUNKS: usize = 10_000;

    #[derive(Default)]
    struct MockGenerator {
        script: VecDeque<Behavior>,
        last_chunks: usize,
        last_stopped_early: bool,
    }

    impl MockGenerator {
        fn new(script: impl IntoIterator<Item = Behavior>) -> Self {
            Self {
                script: script.into_iter().collect(),
                ..Default::default()
            }
        }
    }

    impl TextGenerator for MockGenerator {
        fn generate(
            &mut self,
            _prompt: &str,
            _max_new_tokens: u32,
            on_chunk: &mut dyn FnMut(&str) -> ControlFlow<()>,
        ) -> Result<GenerationOutcome> {
            let (chunk, limit) = match self.script.pop_front() {
                Some(Behavior::Say(s)) => (s, 1),
                Some(Behavior::Endless(s)) => (s, GUARD_CHUNKS),
                Some(Behavior::Fail) | None => return Err(EvalError::Generator("boom".into())),
            };
            let mut text = String::new();
            self.last_stopped_early = false;
            self.last_chunks = 0;
            while self.last_chunks < limit {
                self.last_chunks += 1;
                text.push_str(chunk);
                if on_chunk(chunk).is_break() {
                    self.last_stopped_early = true;
                    break;
                }
            }
            Ok(GenerationOutcome {
                text,
                stopped_early: self.last_stopped_early,
                duration_ms: 0,
            })
        }
    }

    fn prompt(id: &str) -> EvalPrompt {
        EvalPrompt {
            id: id.into(),
            text: "Quelle est la capitale de la France ?".into(),
            expect_contains: None,
            expect_not_contains: None,
        }
    }

    fn prompts(n: usize) -> Vec<EvalPrompt> {
        (0..n).map(|i| prompt(&format!("p{i}"))).collect()
    }

    const HEALTHY: &str = "La capitale de la France est Paris.";

    #[test]
    fn endless_wait_stops_early_with_loop_detected() {
        let mut g = MockGenerator::new([Behavior::Endless("Wait... ")]);
        let report = run_eval(&mut g, &prompts(1), &EvalConfig::default()).unwrap();
        assert_eq!(
            report.results[0].status,
            PromptStatus::LoopDetected {
                reason: LoopReason::MarkerRepetition
            }
        );
        assert!(g.last_stopped_early);
        assert!(g.last_chunks < 50, "{} chunks", g.last_chunks);
        assert_eq!(report.verdict, EvalVerdict::Broken);
    }

    #[test]
    fn pass_and_expectations() {
        let mut ps = prompts(3);
        ps[1].expect_contains = Some(vec!["paris".into(), "Lyon".into()]);
        ps[2].expect_not_contains = Some(vec!["CAPITALE".into()]);
        ps[0].expect_contains = Some(vec!["PARIS".into()]);
        let mut g = MockGenerator::new([
            Behavior::Say(HEALTHY),
            Behavior::Say(HEALTHY),
            Behavior::Say(HEALTHY),
        ]);
        let r = run_eval(&mut g, &ps, &EvalConfig::default()).unwrap();
        assert_eq!(r.results[0].status, PromptStatus::Pass);
        assert_eq!(
            r.results[1].status,
            PromptStatus::ExpectationFailed {
                details: "attendu « Lyon » absent".into()
            }
        );
        assert_eq!(
            r.results[2].status,
            PromptStatus::ExpectationFailed {
                details: "interdit « CAPITALE » présent".into()
            }
        );
        assert_eq!(r.results[0].output_excerpt, HEALTHY);
        assert_eq!(r.results[0].chars_generated, HEALTHY.chars().count());
        assert_eq!((r.passed, r.failed), (1, 2));
    }

    #[test]
    fn generator_error_is_recorded_and_run_continues_unless_fail_fast() {
        let script = || [Behavior::Fail, Behavior::Say(HEALTHY)];
        let r = run_eval(
            &mut MockGenerator::new(script()),
            &prompts(2),
            &EvalConfig::default(),
        )
        .unwrap();
        assert!(
            matches!(&r.results[0].status, PromptStatus::GeneratorError { message } if message.contains("boom"))
        );
        assert_eq!(r.results[1].status, PromptStatus::Pass);

        let cfg = EvalConfig {
            fail_fast: true,
            ..Default::default()
        };
        let r = run_eval(&mut MockGenerator::new(script()), &prompts(2), &cfg).unwrap();
        assert_eq!(r.results.len(), 1);
        assert_eq!(r.verdict, EvalVerdict::Degraded);
    }

    fn verdict(loops: usize, errors: usize) -> EvalVerdict {
        let script = (0..10).map(|i| match i {
            _ if i < loops => Behavior::Endless("Wait... "),
            _ if i < loops + errors => Behavior::Fail,
            _ => Behavior::Say(HEALTHY),
        });
        run_eval(
            &mut MockGenerator::new(script),
            &prompts(10),
            &EvalConfig::default(),
        )
        .unwrap()
        .verdict
    }

    #[test]
    fn verdict_thresholds() {
        assert_eq!(verdict(0, 0), EvalVerdict::Healthy);
        assert_eq!(verdict(0, 1), EvalVerdict::Degraded);
        assert_eq!(verdict(1, 0), EvalVerdict::Degraded);
        assert_eq!(
            verdict(3, 0),
            EvalVerdict::Degraded,
            "30% is not more than 30%"
        );
        assert_eq!(verdict(4, 0), EvalVerdict::Broken);
    }

    #[test]
    fn empty_prompt_list_is_refused() {
        let err = run_eval(&mut MockGenerator::default(), &[], &EvalConfig::default()).unwrap_err();
        assert!(matches!(err, EvalError::InvalidConfig(_)));
    }

    #[test]
    fn excerpt_is_cut_on_char_boundary() {
        let long: &'static str = "é🦀a".repeat(200).leak();
        assert_eq!(long.chars().count(), 600);
        let cfg = EvalConfig {
            loop_cfg: LoopDetectorConfig {
                min_chars_before_check: usize::MAX,
                ..Default::default()
            },
            ..Default::default()
        };
        let r = run_eval(
            &mut MockGenerator::new([Behavior::Say(long)]),
            &prompts(1),
            &cfg,
        )
        .unwrap();
        let excerpt = &r.results[0].output_excerpt;
        assert_eq!(excerpt.chars().count(), 500);
        assert!(long.starts_with(excerpt.as_str()));
        assert_eq!(r.results[0].chars_generated, 600);
    }

    #[test]
    fn report_json_is_stable() {
        let r = run_eval(
            &mut MockGenerator::new([Behavior::Endless("Wait... ")]),
            &prompts(1),
            &EvalConfig::default(),
        )
        .unwrap();
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["verdict"], "broken");
        assert_eq!(json["results"][0]["status"], "loop_detected");
        assert_eq!(json["results"][0]["reason"], "marker_repetition");
        assert_eq!(json["results"][0]["id"], "p0");
    }
}
