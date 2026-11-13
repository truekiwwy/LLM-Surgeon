//! A scripted provider: each call plays the next step. It records every request so that a
//! test can check what would have left the machine. Used by the tests and `--self-test`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use serde_json::Value;

use super::provider::{Block, Delta, LlmProvider, ProviderError, Request, StopReason, Turn, Usage};

type Reply = Box<dyn Fn(&Request) -> Result<Turn, ProviderError> + Send>;

pub enum Step {
    Reply(Reply),
    /// Waits until the call is cancelled (a hung provider).
    Hang,
}

#[derive(Default)]
pub struct Mock {
    steps: Mutex<VecDeque<Step>>,
    pub requests: Mutex<Vec<Request>>,
    pub models: Vec<String>,
}

impl Mock {
    pub fn new(steps: Vec<Step>) -> Self {
        Self {
            steps: Mutex::new(steps.into()),
            requests: Mutex::default(),
            models: vec!["mock-1".into()],
        }
    }

    pub fn requests(&self) -> Vec<Request> {
        self.requests.lock().map(|r| r.clone()).unwrap_or_default()
    }
}

/// A turn that says `text`.
pub fn say(text: &str) -> Step {
    let text = text.to_owned();
    Step::Reply(Box::new(move |_| {
        Ok(Turn {
            blocks: vec![Block::Text { text: text.clone() }],
            usage: Usage {
                input_tokens: 100,
                output_tokens: 20,
            },
            stop: StopReason::EndTurn,
        })
    }))
}

/// A turn with these tool calls `(id, name, input)`, made in parallel.
pub fn call(calls: Vec<(&str, &str, Value)>) -> Step {
    let blocks: Vec<Block> = calls
        .into_iter()
        .map(|(id, name, input)| Block::ToolUse {
            id: id.into(),
            name: name.into(),
            input,
        })
        .collect();
    Step::Reply(Box::new(move |_| {
        Ok(Turn {
            blocks: blocks.clone(),
            usage: Usage {
                input_tokens: 100,
                output_tokens: 30,
            },
            stop: StopReason::ToolUse,
        })
    }))
}

pub fn fail(err: ProviderError) -> Step {
    Step::Reply(Box::new(move |_| Err(err.clone())))
}

impl LlmProvider for Mock {
    fn list_models(&self) -> Result<Vec<String>, ProviderError> {
        Ok(self.models.clone())
    }

    fn stream(
        &self,
        req: &Request,
        cancel: &AtomicBool,
        on: &mut dyn FnMut(Delta),
    ) -> Result<Turn, ProviderError> {
        if let Ok(mut r) = self.requests.lock() {
            r.push(req.clone());
        }
        let step = self.steps.lock().ok().and_then(|mut s| s.pop_front());
        match step {
            None => Err(ProviderError::Protocol("script épuisé".into())),
            Some(Step::Hang) => {
                while !cancel.load(Ordering::SeqCst) {
                    std::thread::sleep(Duration::from_millis(2));
                }
                Err(ProviderError::Cancelled)
            }
            Some(Step::Reply(f)) => {
                let turn = f(req)?;
                for b in &turn.blocks {
                    if let Block::Text { text } = b {
                        on(Delta::Text(text.clone()));
                    }
                }
                Ok(turn)
            }
        }
    }
}
