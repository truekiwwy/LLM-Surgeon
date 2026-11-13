//! The components every view is built from, and the only entry points of style: a view
//! never names a color, a margin, a radius or a frame (`tests/ui_lint.rs`). Borders are one
//! physical pixel on rectangles snapped to the pixel grid, at any `pixels_per_point`.

use eframe::egui::emath::GuiRounding as _;
use eframe::egui::{
    self, Align, Align2, Color32, FontFamily, FontId, Id, Label, Layout, Painter, Pos2, Rangef,
    Rect, Response, RichText, Sense, TextEdit, Ui, UiBuilder, Vec2, WidgetInfo, WidgetType,
};

use super::icons::{self, Icon, State};
use super::theme::{self, TOKENS as T};

// ------------------------------------------------------------------ layout dump

/// Named rectangles of the frame, for `--dump-layout` and the layout tests.
#[derive(Clone, Default)]
pub struct LayoutDump {
    pub enabled: bool,
    pub regions: Vec<(String, Rect)>,
}

fn dump_id() -> Id {
    Id::new("ui-layout-dump")
}

/// Starts a frame: forgets the regions of the previous one.
pub fn begin_dump(ctx: &egui::Context, enabled: bool) {
    ctx.data_mut(|d| {
        let dump = d.get_temp_mut_or_default::<LayoutDump>(dump_id());
        dump.enabled = enabled;
        dump.regions.clear();
    });
}

pub fn record(ctx: &egui::Context, name: &str, rect: Rect) {
    ctx.data_mut(|d| {
        let dump = d.get_temp_mut_or_default::<LayoutDump>(dump_id());
        if dump.enabled {
            dump.regions.push((name.to_owned(), rect));
        }
    });
}

pub fn take_dump(ctx: &egui::Context) -> Vec<(String, Rect)> {
    ctx.data_mut(|d| {
        std::mem::take(&mut d.get_temp_mut_or_default::<LayoutDump>(dump_id()).regions)
    })
}

// ------------------------------------------------------------------ pixels

/// One physical pixel, in points.
pub fn px(painter: &Painter) -> f32 {
    1.0 / painter.pixels_per_point()
}

pub fn snap(painter: &Painter, rect: Rect) -> Rect {
    rect.round_to_pixels(painter.pixels_per_point())
}

/// A one-pixel border inside `rect`.
pub fn border(painter: &Painter, rect: Rect, color: Color32) {
    let (r, w) = (snap(painter, rect), px(painter));
    let edges = [
        Rect::from_min_max(r.min, Pos2::new(r.max.x, r.min.y + w)),
        Rect::from_min_max(Pos2::new(r.min.x, r.max.y - w), r.max),
        Rect::from_min_max(r.min, Pos2::new(r.min.x + w, r.max.y)),
        Rect::from_min_max(Pos2::new(r.max.x - w, r.min.y), r.max),
    ];
    for edge in edges {
        painter.rect_filled(edge, 0.0, color);
    }
}

/// A one-pixel horizontal line whose top edge is at `y`.
pub fn hline(painter: &Painter, x: impl Into<Rangef>, y: f32, color: Color32) {
    let x = x.into();
    let r = snap(
        painter,
        Rect::from_min_max(Pos2::new(x.min, y), Pos2::new(x.max, y + px(painter))),
    );
    painter.rect_filled(r, 0.0, color);
}

/// A one-pixel vertical line whose left edge is at `x`.
pub fn vline(painter: &Painter, x: f32, y: impl Into<Rangef>, color: Color32) {
    let y = y.into();
    let r = snap(
        painter,
        Rect::from_min_max(Pos2::new(x, y.min), Pos2::new(x + px(painter), y.max)),
    );
    painter.rect_filled(r, 0.0, color);
}

/// The keyboard focus ring: one pixel of accent inside `rect`.
pub fn focus_ring(painter: &Painter, rect: Rect) {
    border(painter, rect, T.color.accent);
}

// ------------------------------------------------------------------ text

pub fn font_body() -> FontId {
    FontId::new(T.text.body, FontFamily::Proportional)
}

pub fn font_meta() -> FontId {
    FontId::new(T.text.meta, FontFamily::Proportional)
}

pub fn font_mono() -> FontId {
    FontId::new(T.text.mono, FontFamily::Monospace)
}

pub fn font_title() -> FontId {
    FontId::new(T.text.body, theme::medium())
}

/// Tabs of the title bar.
pub fn font_tab() -> FontId {
    FontId::new(T.text.title, FontFamily::Proportional)
}

pub fn font_brand() -> FontId {
    FontId::new(T.text.brand, theme::medium())
}

fn rich(text: impl Into<String>, font: FontId, color: Color32) -> RichText {
    RichText::new(text).font(font).color(color)
}

/// Running text, wrapped.
pub fn label(ui: &mut Ui, text: impl Into<String>) -> Response {
    ui.add(Label::new(rich(text, font_body(), T.color.text)).wrap())
}

/// Secondary running text, wrapped.
pub fn secondary(ui: &mut Ui, text: impl Into<String>) -> Response {
    ui.add(Label::new(rich(text, font_body(), T.color.text_secondary)).wrap())
}

/// One line of medium-weight text.
pub fn strong(ui: &mut Ui, text: impl Into<String>) -> Response {
    ui.add(Label::new(rich(text, font_title(), T.color.text)).truncate())
}

/// One line of small secondary text.
pub fn meta(ui: &mut Ui, text: impl Into<String>) -> Response {
    ui.add(Label::new(rich(text, font_meta(), T.color.text_secondary)).truncate())
}

/// One line of tabular text: tensors, shapes, norms, hashes.
pub fn mono(ui: &mut Ui, text: impl Into<String>) -> Response {
    ui.add(Label::new(rich(text, font_mono(), T.color.text)).truncate())
}

pub fn mono_secondary(ui: &mut Ui, text: impl Into<String>) -> Response {
    ui.add(Label::new(rich(text, font_mono(), T.color.text_secondary)).truncate())
}

/// Tabular text over several lines (logs, JSON parameters).
pub fn mono_wrapped(ui: &mut Ui, text: impl Into<String>, color: Color32) -> Response {
    ui.add(Label::new(rich(text, font_mono(), color)).wrap())
}

/// A sub-heading inside a panel: closer to its own content than to what is above.
pub fn section(ui: &mut Ui, title: &str) {
    // Opening a panel, the panel's own padding is the space above.
    if ui.cursor().top() > ui.max_rect().top() + 1.0 {
        ui.add_space(T.space.m);
    }
    meta(ui, title);
    ui.add_space(T.space.xs);
}

/// A full-width separator inside a panel.
pub fn separator(ui: &mut Ui) {
    ui.add_space(T.space.s);
    let (rect, _) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), px(ui.painter())),
        Sense::hover(),
    );
    hline(ui.painter(), rect.x_range(), rect.top(), T.color.separator);
    ui.add_space(T.space.s);
}

/// A state: a drawn dot and its word, never the color alone.
pub fn status(ui: &mut Ui, state: State, text: impl Into<String>) -> Response {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = T.space.s;
        let (rect, _) = ui.allocate_exact_size(Vec2::new(T.size.dot, T.size.row), Sense::hover());
        icons::dot(ui.painter(), rect.center(), state);
        ui.add(Label::new(rich(text, font_body(), T.color.text)).truncate())
    })
    .inner
}

/// What an empty panel says: what is missing, then what to do next.
pub fn empty(ui: &mut Ui, title: &str, hint: &str) {
    ui.add_space(T.space.s);
    label(ui, title);
    secondary(ui, hint);
}

/// A message line: an error names its cause, a success what was done.
pub fn message(ui: &mut Ui, text: &str, error: bool) {
    let state = if error { State::Error } else { State::Ok };
    ui.horizontal_top(|ui| {
        let (rect, _) = ui.allocate_exact_size(Vec2::new(T.size.dot, T.size.row), Sense::hover());
        icons::dot(ui.painter(), rect.center(), state);
        ui.add(Label::new(rich(text, font_body(), T.color.text)).wrap());
    });
}

/// A keyboard shortcut, as a small keycap.
pub fn keycap(ui: &mut Ui, text: &str) {
    let galley = ui
        .painter()
        .layout_no_wrap(text.to_owned(), font_meta(), T.color.text_secondary);
    let size = Vec2::new(galley.size().x + 2.0 * T.space.xs, T.size.row - T.space.s);
    let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
    border(ui.painter(), rect, T.color.border);
    let pos = rect.center() - galley.size() / 2.0;
    ui.painter().galley(pos, galley, T.color.text_secondary);
}

// ------------------------------------------------------------------ containers

/// A child `Ui` on `rect`, clipped to it.
pub fn region<R>(ui: &mut Ui, rect: Rect, salt: &str, body: impl FnOnce(&mut Ui) -> R) -> R {
    let mut child = ui.new_child(
        UiBuilder::new()
            .max_rect(rect)
            .layout(Layout::top_down(Align::Min))
            .id_salt(salt),
    );
    child.set_clip_rect(rect.intersect(ui.clip_rect()));
    body(&mut child)
}

/// A panel: one surface, one border, an optional header. Panels are never nested.
pub struct Panel<'a> {
    name: &'static str,
    title: Option<&'a str>,
    flush: bool,
    fill: Color32,
}

impl<'a> Panel<'a> {
    /// `name` identifies the panel in `--dump-layout`.
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            title: None,
            flush: false,
            fill: T.color.surface,
        }
    }

    pub fn title(mut self, title: &'a str) -> Self {
        self.title = Some(title);
        self
    }

    /// No inner padding: the content (a canvas, a table, a list) reaches the border.
    pub fn flush(mut self) -> Self {
        self.flush = true;
        self
    }

    /// The darker canvas ground, for 3D and plots.
    pub fn canvas(mut self) -> Self {
        self.fill = T.color.canvas;
        self.flush = true;
        self
    }

    pub fn show<R>(self, ui: &mut Ui, rect: Rect, body: impl FnOnce(&mut Ui) -> R) -> R {
        self.show_with(ui, rect, &mut (), |_, _| {}, |ui, _| body(ui))
    }

    /// `actions` are laid out right to left in the header; both closures get `state`.
    pub fn show_with<S, R>(
        self,
        ui: &mut Ui,
        rect: Rect,
        state: &mut S,
        actions: impl FnOnce(&mut Ui, &mut S),
        body: impl FnOnce(&mut Ui, &mut S) -> R,
    ) -> R {
        let painter = ui.painter_at(rect);
        let rect = snap(&painter, rect);
        painter.rect_filled(rect, 0.0, self.fill);
        border(&painter, rect, T.color.border);
        record(ui.ctx(), self.name, rect);
        let mut inner = rect.shrink(px(&painter));
        if let Some(title) = self.title {
            let header = Rect::from_min_size(inner.min, Vec2::new(inner.width(), T.size.header));
            painter.rect_filled(header, 0.0, T.color.surface);
            painter.text(
                Pos2::new(header.left() + T.space.m, header.center().y),
                Align2::LEFT_CENTER,
                title.to_uppercase(),
                font_title(),
                T.color.text,
            );
            hline(
                &painter,
                header.x_range(),
                header.bottom(),
                T.color.separator,
            );
            let bar = header.shrink2(Vec2::new(T.space.s, 0.0));
            let mut child = ui.new_child(
                UiBuilder::new()
                    .max_rect(bar)
                    .layout(Layout::right_to_left(Align::Center))
                    .id_salt((self.name, "header")),
            );
            actions(&mut child, state);
            inner.min.y = header.bottom() + px(&painter);
        }
        let content = match self.flush {
            true => inner,
            false => inner.shrink(T.space.m),
        };
        let mut child = ui.new_child(
            UiBuilder::new()
                .max_rect(content)
                .layout(Layout::top_down(Align::Min))
                .id_salt(self.name),
        );
        child.set_clip_rect(inner.intersect(ui.clip_rect()));
        body(&mut child, state)
    }
}

/// The scrim over the whole window, on the foreground layer, with one line in its middle.
pub fn drop_hint(ctx: &egui::Context, text: &str) {
    let layer = egui::LayerId::new(egui::Order::Foreground, Id::new("drop-hint"));
    let rect = ctx.content_rect();
    let painter = ctx.layer_painter(layer);
    painter.rect_filled(rect, 0.0, T.color.scrim);
    painter.text(
        rect.center(),
        Align2::CENTER_CENTER,
        text,
        font_brand(),
        T.color.text,
    );
}

/// The frame of floating surfaces (palette, menus): surface, one border, no shadow.
pub fn popup_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(T.color.surface)
        .stroke(egui::Stroke::new(1.0, T.color.border))
        .corner_radius(T.radius)
        .inner_margin(T.space.s)
}

/// A sunken block inside a panel (the wizard's cards): canvas ground, one border.
pub fn inset_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(T.color.canvas)
        .stroke(egui::Stroke::new(1.0, T.color.border))
        .corner_radius(T.radius)
        .inner_margin(T.space.m)
}

// ------------------------------------------------------------------ rows

/// Hover, selection and focus of a row, painted behind its content.
pub fn row_background(ui: &Ui, rect: Rect, response: &Response, selected: bool) {
    let painter = ui.painter();
    let fill = match (selected, response.hovered()) {
        (true, _) => Some(T.color.selection),
        (false, true) => Some(T.color.surface_hover),
        (false, false) => None,
    };
    if let Some(fill) = fill {
        painter.rect_filled(snap(painter, rect), 0.0, fill);
    }
    if selected {
        // The one accent mark of the list.
        let w = 2.0 * px(painter);
        let mark = Rect::from_min_size(rect.min, Vec2::new(w, rect.height()));
        painter.rect_filled(snap(painter, mark), 0.0, T.color.accent);
    }
    if response.has_focus() {
        focus_ring(painter, rect);
    }
}

/// A clickable full-width row of the standard height; `body` lays its content out.
pub fn row(ui: &mut Ui, selected: bool, body: impl FnOnce(&mut Ui)) -> Response {
    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), T.size.row), Sense::click());
    row_background(ui, rect, &response, selected);
    let inner = rect.shrink2(Vec2::new(T.space.s, 0.0));
    let mut child = ui.new_child(
        UiBuilder::new()
            .max_rect(inner)
            .layout(Layout::left_to_right(Align::Center)),
    );
    child.set_clip_rect(rect.intersect(ui.clip_rect()));
    body(&mut child);
    response
}

/// A key and its value on one row: the key in secondary text, the value tabular, right.
pub fn kv(ui: &mut Ui, key: &str, value: &str) {
    let (rect, _) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), T.size.row), Sense::hover());
    let painter = ui.painter_at(rect);
    let value_galley = painter.layout_no_wrap(value.to_owned(), font_mono(), T.color.text);
    let value_x = rect.right() - value_galley.size().x;
    painter.galley(
        Pos2::new(value_x, rect.center().y - value_galley.size().y / 2.0),
        value_galley,
        T.color.text,
    );
    let key_clip = Rect::from_min_max(rect.min, Pos2::new(value_x - T.space.s, rect.max.y));
    painter.with_clip_rect(key_clip).text(
        Pos2::new(rect.left(), rect.center().y),
        Align2::LEFT_CENTER,
        key,
        font_body(),
        T.color.text_secondary,
    );
}

// ------------------------------------------------------------------ buttons

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ButtonKind {
    /// The one primary action of a screen.
    Primary,
    Secondary,
    /// Text only, for low-weight actions.
    Ghost,
}

fn button_at(ui: &mut Ui, text: &str, kind: ButtonKind, width: Option<f32>) -> Response {
    let font = match kind {
        ButtonKind::Primary => font_title(),
        _ => font_body(),
    };
    let galley = ui
        .painter()
        .layout_no_wrap(text.to_owned(), font, T.color.text);
    let height = match kind {
        ButtonKind::Primary => T.size.cta,
        _ => T.size.control,
    };
    let pad = match kind {
        ButtonKind::Ghost => T.space.s,
        _ => T.space.m,
    };
    let size = Vec2::new(width.unwrap_or(galley.size().x + 2.0 * pad), height);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    let enabled = response.enabled();
    response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, enabled, text));
    let painter = ui.painter();
    let rect = snap(painter, rect);
    let (hover, down) = (response.hovered(), response.is_pointer_button_down_on());
    let c = &T.color;
    let (fill, outline, ink) = match (kind, enabled) {
        (_, false) => (
            (kind != ButtonKind::Ghost).then_some(c.surface_hover),
            None,
            c.text_disabled,
        ),
        (ButtonKind::Primary, true) => (
            Some(if hover && !down {
                c.action_hover
            } else {
                c.action
            }),
            None,
            c.on_action,
        ),
        (ButtonKind::Secondary, true) => (
            Some(match (down, hover) {
                (true, _) => c.selection,
                (false, true) => c.surface_hover,
                (false, false) => c.surface,
            }),
            Some(c.border),
            c.text,
        ),
        (ButtonKind::Ghost, true) => (
            (hover || down).then_some(c.surface_hover),
            None,
            if hover { c.text } else { c.text_secondary },
        ),
    };
    if let Some(fill) = fill {
        painter.rect_filled(rect, T.radius, fill);
    }
    if let Some(outline) = outline {
        border(painter, rect, outline);
    }
    if response.has_focus() {
        focus_ring(painter, rect);
    }
    let pos = rect.center() - galley.size() / 2.0;
    painter.galley_with_override_text_color(pos, galley, ink);
    response
}

pub fn button(ui: &mut Ui, text: &str, kind: ButtonKind) -> Response {
    button_at(ui, text, kind, None)
}

/// A button that takes the available width (the primary action of a form).
pub fn button_wide(ui: &mut Ui, text: &str, kind: ButtonKind) -> Response {
    let width = ui.available_width();
    button_at(ui, text, kind, Some(width))
}

/// A button that can be disabled; `why` explains the disabled state on hover.
pub fn button_if(ui: &mut Ui, enabled: bool, text: &str, kind: ButtonKind, why: &str) -> Response {
    ui.add_enabled_ui(enabled, |ui| button_at(ui, text, kind, None))
        .inner
        .on_disabled_hover_text(why)
}

/// A square icon button with its tooltip (the icon alone never carries the meaning).
pub fn icon_button(ui: &mut Ui, icon: Icon, tooltip: &str) -> Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(T.size.control), Sense::click());
    let enabled = response.enabled();
    response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, enabled, tooltip));
    let painter = ui.painter();
    if response.hovered() && enabled {
        painter.rect_filled(snap(painter, rect), T.radius, T.color.surface_hover);
    }
    if response.has_focus() {
        focus_ring(painter, rect);
    }
    let ink = match (enabled, response.hovered()) {
        (false, _) => T.color.text_disabled,
        (true, true) => T.color.text,
        (true, false) => T.color.text_secondary,
    };
    icons::paint(painter, rect, icon, ink);
    response.on_hover_text(tooltip)
}

// ------------------------------------------------------------------ choices

/// One option of a segmented control.
pub struct Segment<'a, V> {
    pub value: V,
    pub label: &'a str,
    pub tooltip: &'a str,
}

/// Indices in the order to add them so that they read left to right: a header or a bar
/// laid out right to left would otherwise show the options backwards.
fn reading_order(ui: &Ui, n: usize) -> Vec<usize> {
    match ui.layout().main_dir() == egui::Direction::RightToLeft {
        true => (0..n).rev().collect(),
        false => (0..n).collect(),
    }
}

/// A segmented control: the active segment carries the accent underline.
pub fn segmented<V: PartialEq + Copy>(
    ui: &mut Ui,
    value: &mut V,
    segments: &[Segment<'_, V>],
    font: FontId,
) -> bool {
    let mut changed = false;
    let order = reading_order(ui, segments.len());
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        for segment in order.into_iter().map(|k| &segments[k]) {
            let active = *value == segment.value;
            let galley =
                ui.painter()
                    .layout_no_wrap(segment.label.to_owned(), font.clone(), T.color.text);
            let size = Vec2::new(galley.size().x + 2.0 * T.space.m, T.size.control);
            let (rect, response) = ui.allocate_exact_size(size, Sense::click());
            response.widget_info(|| {
                WidgetInfo::selected(WidgetType::RadioButton, true, active, segment.label)
            });
            let painter = ui.painter();
            let rect = snap(painter, rect);
            let fill = match (active, response.hovered()) {
                (true, _) => T.color.selection,
                (false, true) => T.color.surface_hover,
                (false, false) => T.color.surface,
            };
            painter.rect_filled(rect, 0.0, fill);
            border(painter, rect, T.color.border);
            if active {
                let h = 2.0 * px(painter);
                let line = Rect::from_min_max(Pos2::new(rect.left(), rect.bottom() - h), rect.max);
                painter.rect_filled(line, 0.0, T.color.accent);
            }
            if response.has_focus() {
                focus_ring(painter, rect);
            }
            let ink = if active {
                T.color.text
            } else {
                T.color.text_secondary
            };
            let pos = rect.center() - galley.size() / 2.0;
            painter.galley_with_override_text_color(pos, galley, ink);
            if response.on_hover_text(segment.tooltip).clicked() && !active {
                *value = segment.value;
                changed = true;
            }
        }
    });
    changed
}

/// A small segmented choice of short labels (the language of the status bar).
pub fn segmented_small<V: PartialEq + Copy>(
    ui: &mut Ui,
    value: &mut V,
    options: &[(V, &str)],
    tooltip: &str,
) -> bool {
    let mut changed = false;
    let order = reading_order(ui, options.len());
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        for (option, text) in order.into_iter().map(|k| &options[k]) {
            let active = *value == *option;
            let galley = ui
                .painter()
                .layout_no_wrap((*text).to_owned(), font_meta(), T.color.text);
            let size = Vec2::new(galley.size().x + 2.0 * T.space.s, T.size.row - T.space.s);
            let (rect, response) = ui.allocate_exact_size(size, Sense::click());
            response
                .widget_info(|| WidgetInfo::selected(WidgetType::RadioButton, true, active, *text));
            let painter = ui.painter();
            let rect = snap(painter, rect);
            if active {
                painter.rect_filled(rect, 0.0, T.color.selection);
            } else if response.hovered() {
                painter.rect_filled(rect, 0.0, T.color.surface_hover);
            }
            if response.has_focus() {
                focus_ring(painter, rect);
            }
            let ink = if active {
                T.color.text
            } else {
                T.color.text_secondary
            };
            painter.galley_with_override_text_color(
                rect.center() - galley.size() / 2.0,
                galley,
                ink,
            );
            if response.on_hover_text(tooltip).clicked() && !active {
                *value = *option;
                changed = true;
            }
        }
    });
    changed
}

/// Small secondary text that acts (opens a dialog): underlined on hover.
pub fn link(ui: &mut Ui, text: &str, tooltip: &str) -> Response {
    let galley = ui
        .painter()
        .layout_no_wrap(text.to_owned(), font_meta(), T.color.text_secondary);
    let (rect, response) = ui.allocate_exact_size(galley.size(), Sense::click());
    response.widget_info(|| WidgetInfo::labeled(WidgetType::Link, true, text));
    let ink = if response.hovered() {
        T.color.text
    } else {
        T.color.text_secondary
    };
    let painter = ui.painter();
    if response.hovered() {
        hline(painter, rect.x_range(), rect.bottom(), ink);
    }
    if response.has_focus() {
        focus_ring(painter, rect.expand(T.space.xs / 2.0));
    }
    painter.galley_with_override_text_color(rect.min, galley, ink);
    response.on_hover_text(tooltip)
}

/// A window button of the title bar; `danger`: the close button fills with the error color
/// on hover, as the system does.
pub fn window_button(ui: &mut Ui, rect: Rect, icon: Icon, tooltip: &str, danger: bool) -> Response {
    let response = ui.interact(rect, ui.id().with(("window", tooltip)), Sense::click());
    response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, true, tooltip));
    let painter = ui.painter();
    let ink = match (response.hovered(), danger) {
        (true, true) => {
            painter.rect_filled(snap(painter, rect), 0.0, T.color.error);
            T.color.on_action
        }
        (true, false) => {
            painter.rect_filled(snap(painter, rect), 0.0, T.color.surface_hover);
            T.color.text
        }
        (false, _) => T.color.text_secondary,
    };
    if response.has_focus() {
        focus_ring(painter, rect.shrink(T.space.xs));
    }
    icons::paint(painter, rect, icon, ink);
    response.on_hover_text(tooltip)
}

/// A tab of the title bar, painted in `rect`: the active one carries the accent underline.
pub fn tab(ui: &mut Ui, rect: Rect, text: &str, active: bool) -> Response {
    let response = ui.interact(rect, ui.id().with(("tab", text)), Sense::click());
    response.widget_info(|| WidgetInfo::selected(WidgetType::RadioButton, true, active, text));
    let painter = ui.painter();
    let ink = match (active, response.hovered()) {
        (true, _) | (false, true) => T.color.text,
        (false, false) => T.color.text_secondary,
    };
    painter.text(rect.center(), Align2::CENTER_CENTER, text, font_tab(), ink);
    if active {
        let h = 2.0 * px(painter);
        let line = Rect::from_min_max(Pos2::new(rect.left(), rect.bottom() - h), rect.max);
        painter.rect_filled(snap(painter, line), 0.0, T.color.accent);
    }
    if response.has_focus() {
        focus_ring(painter, rect.shrink(T.space.xs));
    }
    response
}

/// A drop-down; `body` adds the options with [`option`].
pub fn select<R>(
    ui: &mut Ui,
    id: impl std::hash::Hash + std::fmt::Debug,
    current: &str,
    width: f32,
    body: impl FnOnce(&mut Ui) -> R,
) -> Option<R> {
    egui::ComboBox::from_id_salt(id)
        .selected_text(rich(current, font_body(), T.color.text))
        .width(width)
        .truncate()
        .show_ui(ui, body)
        .inner
}

/// An option of a [`select`]; `true` when it was just picked.
pub fn option<V: PartialEq>(ui: &mut Ui, value: &mut V, choice: V, text: &str) -> bool {
    ui.selectable_value(value, choice, rich(text, font_body(), T.color.text))
        .clicked()
}

pub fn checkbox(ui: &mut Ui, value: &mut bool, text: &str) -> Response {
    ui.checkbox(value, rich(text, font_body(), T.color.text))
}

// ------------------------------------------------------------------ fields

fn field_frame(ui: &mut Ui, size: Vec2) -> Rect {
    let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
    let painter = ui.painter();
    let rect = snap(painter, rect);
    painter.rect_filled(rect, T.radius, T.color.canvas);
    border(painter, rect, T.color.border);
    rect
}

fn edit(ui: &mut Ui, rect: Rect, edit: TextEdit<'_>) -> Response {
    let inner = rect.shrink2(Vec2::new(T.space.s, 0.0));
    let response = ui.place(
        inner,
        edit.frame(egui::Frame::NONE)
            .margin(egui::Margin::ZERO)
            .desired_width(inner.width())
            .vertical_align(Align::Center),
    );
    if response.has_focus() {
        focus_ring(ui.painter(), rect);
    }
    response
}

/// A one-line text field of `width`.
pub fn field(ui: &mut Ui, text: &mut String, hint: &str, width: f32) -> Response {
    let rect = field_frame(ui, Vec2::new(width, T.size.control));
    edit(
        ui,
        rect,
        TextEdit::singleline(text).font(font_body()).hint_text(rich(
            hint,
            font_body(),
            T.color.text_disabled,
        )),
    )
}

/// A one-line tabular field (paths, numbers, JSON).
pub fn field_mono(ui: &mut Ui, text: &mut String, hint: &str, width: f32) -> Response {
    let rect = field_frame(ui, Vec2::new(width, T.size.control));
    edit(
        ui,
        rect,
        TextEdit::singleline(text).font(font_mono()).hint_text(rich(
            hint,
            font_mono(),
            T.color.text_disabled,
        )),
    )
}

/// A masked one-line field: its content is never painted.
pub fn field_secret(ui: &mut Ui, text: &mut String, hint: &str, width: f32) -> Response {
    let rect = field_frame(ui, Vec2::new(width, T.size.control));
    edit(
        ui,
        rect,
        TextEdit::singleline(text)
            .password(true)
            .font(font_mono())
            .hint_text(rich(hint, font_body(), T.color.text_disabled)),
    )
}

/// A multi-line field of `rows` lines, full width.
pub fn field_multiline(ui: &mut Ui, text: &mut String, hint: &str, rows: usize) -> Response {
    let height = rows as f32 * T.size.row;
    let rect = field_frame(ui, Vec2::new(ui.available_width(), height));
    let inner = rect.shrink2(Vec2::new(T.space.s, T.space.xs));
    let response = ui.place(
        inner,
        TextEdit::multiline(text)
            .font(font_body())
            .hint_text(rich(hint, font_body(), T.color.text_disabled))
            .frame(egui::Frame::NONE)
            .margin(egui::Margin::ZERO)
            .desired_width(inner.width())
            .desired_rows(rows),
    );
    if response.has_focus() {
        focus_ring(ui.painter(), rect);
    }
    response
}

/// The search field of a list: an icon, the query, full width.
pub fn search(ui: &mut Ui, id: Id, text: &mut String, hint: &str) -> Response {
    let rect = field_frame(ui, Vec2::new(ui.available_width(), T.size.control));
    let icon = Rect::from_min_size(rect.min, Vec2::splat(T.size.control));
    icons::paint(ui.painter(), icon, Icon::Search, T.color.text_secondary);
    let inner = Rect::from_min_max(
        Pos2::new(icon.right(), rect.top()),
        Pos2::new(rect.right() - T.space.s, rect.bottom()),
    );
    let response = ui.place(
        inner,
        TextEdit::singleline(text)
            .id(id)
            .font(font_mono())
            .hint_text(rich(hint, font_body(), T.color.text_disabled))
            .frame(egui::Frame::NONE)
            .margin(egui::Margin::ZERO)
            .desired_width(inner.width())
            .vertical_align(Align::Center),
    );
    if response.has_focus() {
        focus_ring(ui.painter(), rect);
    }
    response
}

// ------------------------------------------------------------------ progress

/// A thin progress bar; `over` marks a value past its budget.
pub fn progress(ui: &mut Ui, fraction: f32, over: bool) {
    let (rect, _) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), T.space.s), Sense::hover());
    let painter = ui.painter();
    let track = snap(
        painter,
        Rect::from_center_size(rect.center(), Vec2::new(rect.width(), T.space.xs)),
    );
    painter.rect_filled(track, 0.0, T.color.separator);
    let mut fill = track;
    fill.set_width(track.width() * fraction.clamp(0.0, 1.0));
    let color = if over { T.color.error } else { T.color.data[4] };
    painter.rect_filled(snap(painter, fill), 0.0, color);
}

/// A cursor over `0..=last` on a one-pixel track (replay, stream states). Arrow keys move
/// it by one when it has the focus.
pub fn slider(ui: &mut Ui, value: &mut usize, last: usize) -> Response {
    let width = ui
        .available_width()
        .clamp(T.size.row * 4.0, T.size.row * 12.0);
    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(width, T.size.control), Sense::click_and_drag());
    response.widget_info(|| WidgetInfo::slider(response.enabled(), *value as f64, ""));
    if let Some(p) = response.interact_pointer_pos() {
        let f = ((p.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
        *value = (f * last as f32).round() as usize;
    }
    if response.has_focus() {
        let (left, right) = ui.input(|i| {
            (
                i.key_pressed(egui::Key::ArrowLeft),
                i.key_pressed(egui::Key::ArrowRight),
            )
        });
        if left {
            *value = value.saturating_sub(1);
        }
        if right {
            *value = (*value + 1).min(last);
        }
    }
    let painter = ui.painter();
    let ink = if response.enabled() {
        T.color.text_secondary
    } else {
        T.color.text_disabled
    };
    hline(painter, rect.x_range(), rect.center().y, T.color.border);
    let f = if last == 0 {
        1.0
    } else {
        *value as f32 / last as f32
    };
    let x = rect.left() + f * rect.width();
    hline(painter, rect.left()..=x, rect.center().y, ink);
    let knob = Rect::from_center_size(
        Pos2::new(x, rect.center().y),
        Vec2::new(T.space.xs, T.space.m),
    );
    painter.rect_filled(
        snap(painter, knob),
        0.0,
        if response.enabled() {
            T.color.text
        } else {
            ink
        },
    );
    if response.has_focus() {
        focus_ring(painter, rect);
    }
    response
}

// ------------------------------------------------------------------ table

#[derive(Debug, Clone, Copy)]
pub enum Width {
    Fixed(f32),
    /// A share of what the fixed columns leave.
    Flex(f32),
}

pub struct Column<'a> {
    pub title: &'a str,
    pub width: Width,
    /// Numbers align right.
    pub right: bool,
}

pub struct Cell<'a> {
    pub text: &'a str,
    pub state: Option<State>,
    pub dim: bool,
}

impl<'a> Cell<'a> {
    pub fn new(text: &'a str) -> Self {
        Self {
            text,
            state: None,
            dim: false,
        }
    }

    pub fn dim(text: &'a str) -> Self {
        Self {
            text,
            state: None,
            dim: true,
        }
    }

    /// A state cell: the dot, then its word.
    pub fn state(state: State, text: &'a str) -> Self {
        Self {
            text,
            state: Some(state),
            dim: false,
        }
    }
}

/// Aligned columns of tabular text: every row shares the header's column edges.
pub struct Table<'a> {
    pub columns: &'a [Column<'a>],
}

impl Table<'_> {
    fn spans(&self, rect: Rect) -> Vec<Rangef> {
        let gap = T.space.m;
        let inner = rect.width() - 2.0 * T.space.s - gap * (self.columns.len() as f32 - 1.0);
        let fixed: f32 = self
            .columns
            .iter()
            .map(|c| match c.width {
                Width::Fixed(w) => w,
                Width::Flex(_) => 0.0,
            })
            .sum();
        let shares: f32 = self
            .columns
            .iter()
            .map(|c| match c.width {
                Width::Flex(s) => s,
                Width::Fixed(_) => 0.0,
            })
            .sum();
        let mut x = rect.left() + T.space.s;
        self.columns
            .iter()
            .map(|c| {
                let w = match c.width {
                    Width::Fixed(w) => w,
                    Width::Flex(s) => ((inner - fixed).max(0.0) * s / shares.max(1.0)).floor(),
                };
                let span = Rangef::new(x, x + w);
                x += w + gap;
                span
            })
            .collect()
    }

    pub fn header(&self, ui: &mut Ui) {
        let (rect, _) =
            ui.allocate_exact_size(Vec2::new(ui.available_width(), T.size.row), Sense::hover());
        let painter = ui.painter_at(rect);
        for (column, span) in self.columns.iter().zip(self.spans(rect)) {
            let (x, align) = match column.right {
                true => (span.max, Align2::RIGHT_CENTER),
                false => (span.min, Align2::LEFT_CENTER),
            };
            painter.text(
                Pos2::new(x, rect.center().y),
                align,
                column.title,
                font_meta(),
                T.color.text_secondary,
            );
        }
        hline(
            &painter,
            rect.x_range(),
            rect.bottom() - px(&painter),
            T.color.separator,
        );
    }

    pub fn row(&self, ui: &mut Ui, selected: bool, cells: &[Cell<'_>]) -> Response {
        let (rect, response) =
            ui.allocate_exact_size(Vec2::new(ui.available_width(), T.size.row), Sense::click());
        row_background(ui, rect, &response, selected);
        for ((column, span), cell) in self.columns.iter().zip(self.spans(rect)).zip(cells) {
            let clip = Rect::from_x_y_ranges(span, rect.y_range());
            let painter = ui.painter_at(clip.intersect(ui.clip_rect()));
            let ink = if cell.dim {
                T.color.text_secondary
            } else {
                T.color.text
            };
            let mut left = span.min;
            if let Some(state) = cell.state {
                icons::dot(
                    &painter,
                    Pos2::new(left + T.size.dot / 2.0, rect.center().y),
                    state,
                );
                left += T.size.dot + T.space.s;
            }
            let (x, align) = match column.right {
                true => (span.max, Align2::RIGHT_CENTER),
                false => (left, Align2::LEFT_CENTER),
            };
            painter.text(
                Pos2::new(x, rect.center().y),
                align,
                cell.text,
                font_mono(),
                ink,
            );
        }
        response
    }
}
