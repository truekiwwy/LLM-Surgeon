//! "Live Steering Preview" tab (frozen names: `PreviewState`, `update`, `view`; body by the
//! `tui` agent, wave 2). Pure `update`, `view` never touches the worker.

use ratatui::layout::Rect;
use ratatui::Frame;

use crate::tui::Theme;

#[derive(Debug, Default)]
pub struct PreviewState {}

#[derive(Debug, Clone, PartialEq)]
pub enum PreviewMsg {
    Tick,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PreviewEffect {}

pub fn update(_state: &mut PreviewState, _msg: PreviewMsg) -> Vec<PreviewEffect> {
    Vec::new()
}

pub fn view(_frame: &mut Frame, _area: Rect, _state: &PreviewState, _theme: &Theme) {}
