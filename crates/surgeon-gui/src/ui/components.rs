//! The components every view is built from, and the only entry points of style: a view
//! never names a color, a margin, a radius or a frame (`tests/ui_lint.rs`). Outlines are one
//! physical pixel on rectangles snapped to the pixel grid, at any `pixels_per_point`; controls
//! are rounded by `radius`, cards and dialogs by `radius_card`.

use eframe::egui::emath::GuiRounding as _;
use eframe::egui::{
    self, Align, Align2, Color32, FontFamily, FontId, Id, Label, Layout, Painter, Pos2, Rangef,
    Rect, Response, RichText, Sense, Stroke, StrokeKind, TextEdit, Ui, UiBuilder, Vec2, WidgetInfo,
    WidgetType,
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

/// A one-pixel rounded outline inside `rect`.
pub fn outline(painter: &Painter, rect: Rect, radius: u8, color: Color32) {
    painter.rect_stroke(
        snap(painter, rect),
        radius,
        Stroke::new(px(painter), color),
        StrokeKind::Inside,
    );
}

/// The keyboard focus ring: one pixel of accent inside `rect`.
pub fn focus_ring(painter: &Painter, rect: Rect) {
    outline(painter, rect, T.radius, T.color.accent);
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

/// Medium body: emphasis inside a panel, the primary action.
pub fn font_strong() -> FontId {
    FontId::new(T.text.body, theme::medium())
}

/// Panel titles.
pub fn font_title() -> FontId {
    FontId::new(T.text.title, theme::medium())
}

/// The question that opens a page.
pub fn font_heading() -> FontId {
    FontId::new(T.text.heading, theme::bold())
}

/// The lead under a heading.
pub fn font_lead() -> FontId {
    FontId::new(T.text.title, FontFamily::Proportional)
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

fn fit(
    painter: &Painter,
    text: &str,
    font: FontId,
    color: Color32,
    width: f32,
) -> std::sync::Arc<egui::Galley> {
    let mut job = egui::text::LayoutJob::simple_singleline(text.to_owned(), font, color);
    job.wrap = egui::text::TextWrapping::truncate_at_width(width.max(0.0));
    painter.layout_job(job)
}

/// Running text, wrapped.
pub fn label(ui: &mut Ui, text: impl Into<String>) -> Response {
    ui.add(Label::new(rich(text, font_body(), T.color.text)).wrap())
}

/// One line of running text, cut with an ellipsis where the room ends.
pub fn line(ui: &mut Ui, text: impl Into<String>) -> Response {
    ui.add(Label::new(rich(text, font_body(), T.color.text)).truncate())
}

/// Secondary running text, wrapped.
pub fn secondary(ui: &mut Ui, text: impl Into<String>) -> Response {
    ui.add(Label::new(rich(text, font_body(), T.color.text_secondary)).wrap())
}

/// One line of medium-weight text.
pub fn strong(ui: &mut Ui, text: impl Into<String>) -> Response {
    ui.add(Label::new(rich(text, font_strong(), T.color.text)).truncate())
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

/// The question that opens a page, then the sentence that says what to do.
pub fn heading(ui: &mut Ui, title: &str, lead: &str) {
    ui.add(Label::new(rich(title, font_heading(), T.color.text)).wrap());
    ui.add_space(T.space.s);
    ui.add(Label::new(rich(lead, font_lead(), T.color.text_secondary)).wrap());
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

/// A line of tabular secondary text behind an icon of the set, in the icon column.
pub fn icon_line(ui: &mut Ui, icon: Icon, text: impl Into<String>) -> Response {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = T.space.s;
        let (rect, _) = ui.allocate_exact_size(Vec2::new(T.size.icon, T.size.row), Sense::hover());
        icons::paint(ui.painter(), rect, icon, T.color.text_disabled);
        mono_secondary(ui, text)
    })
    .inner
}

/// A state: a drawn dot and its word, never the color alone.
pub fn status(ui: &mut Ui, state: State, text: impl Into<String>) -> Response {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = T.space.s;
        // The dot sits in the same 16 column as the icons, so states and tool lines align.
        let (rect, _) = ui.allocate_exact_size(Vec2::new(T.size.icon, T.size.row), Sense::hover());
        icons::dot(ui.painter(), rect.center(), state);
        ui.add(Label::new(rich(text, font_body(), T.color.text)).truncate())
    })
    .inner
}

/// What an empty panel says, centered in the room left: a ringed badge with `icon`, what is
/// missing, then what to do next (two lines at most).
pub fn empty(ui: &mut Ui, icon: Icon, title: &str, hint: &str) {
    const BADGE: f32 = 36.0;
    let room = ui.available_rect_before_wrap();
    let width = room.width().clamp(0.0, T.size.left - 2.0 * T.space.l);
    let centered = |text: &str, font: FontId, color: Color32, rows: usize| {
        let mut job = egui::text::LayoutJob::simple(text.to_owned(), font, color, width);
        job.wrap.max_rows = rows;
        job.halign = Align::Center;
        ui.painter().layout_job(job)
    };
    let title_g = centered(title, font_strong(), T.color.text, 1);
    let hint_g = centered(hint, font_meta(), T.color.text_secondary, 2);
    let height = BADGE + T.space.m + title_g.size().y + T.space.xs + hint_g.size().y;
    // Inside a scroll area the room has no bottom: the block then sits at the top.
    let top = match room.height().is_finite() && room.height() > height {
        true => room.center().y - height / 2.0,
        false => room.top() + T.space.l,
    };
    let (rect, response) = ui.allocate_exact_size(
        Vec2::new(room.width(), top - room.top() + height),
        Sense::hover(),
    );
    response.widget_info(|| WidgetInfo::labeled(WidgetType::Label, true, title));
    let painter = ui.painter();
    let x = rect.center().x;
    let badge = Rect::from_center_size(Pos2::new(x, top + BADGE / 2.0), Vec2::splat(BADGE));
    painter.circle_filled(badge.center(), BADGE / 2.0, T.color.surface);
    painter.circle_stroke(
        badge.center(),
        BADGE / 2.0 - px(painter) / 2.0,
        Stroke::new(px(painter), T.color.border),
    );
    icons::paint(painter, badge, icon, T.color.text_secondary);
    let title_y = badge.bottom() + T.space.m;
    let hint_y = title_y + title_g.size().y + T.space.xs;
    // A centered galley is laid out around x = 0.
    painter.galley(Pos2::new(x, title_y), title_g, T.color.text);
    painter.galley(Pos2::new(x, hint_y), hint_g, T.color.text_secondary);
}

/// A message line: an error names its cause, a success what was done.
pub fn message(ui: &mut Ui, text: &str, error: bool) {
    let state = if error { State::Error } else { State::Ok };
    ui.horizontal_top(|ui| {
        let (rect, _) = ui.allocate_exact_size(Vec2::new(T.size.icon, T.size.row), Sense::hover());
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

/// A zone of the window: one fill, an optional header. The shell draws the lines between
/// zones; panels are never nested.
pub struct Panel<'a> {
    name: &'static str,
    title: Option<&'a str>,
    flush: bool,
    fill: Option<Color32>,
    divided: bool,
}

impl<'a> Panel<'a> {
    /// `name` identifies the panel in `--dump-layout`.
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            title: None,
            flush: false,
            fill: Some(T.color.window),
            divided: false,
        }
    }

    pub fn title(mut self, title: &'a str) -> Self {
        self.title = Some(title);
        self
    }

    pub fn tabs(mut self) -> Self {
        self.title = Some("");
        self
    }

    /// No inner padding: the content (a canvas, a table, a list) reaches the border.
    pub fn flush(mut self) -> Self {
        self.flush = true;
        self
    }

    /// The ground of the center of the window.
    pub fn center(mut self) -> Self {
        self.fill = Some(T.color.canvas);
        self
    }

    /// The 3D ground, content to the edges.
    pub fn canvas(mut self) -> Self {
        self.fill = Some(T.color.canvas);
        self.flush = true;
        self
    }

    /// No fill of its own: a zone inside a dialog.
    pub fn bare(mut self) -> Self {
        self.fill = None;
        self
    }

    /// A line on its top edge: a panel stacked under another.
    pub fn divided(mut self) -> Self {
        self.divided = true;
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
        if let Some(fill) = self.fill {
            painter.rect_filled(rect, 0.0, fill);
        }
        if self.divided {
            hline(&painter, rect.x_range(), rect.top(), T.color.separator);
        }
        record(ui.ctx(), self.name, rect);
        let mut inner = rect;
        if let Some(title) = self.title {
            let header = Rect::from_min_size(inner.min, Vec2::new(inner.width(), T.size.header));
            painter.text(
                Pos2::new(header.left() + T.space.l, header.center().y),
                Align2::LEFT_CENTER,
                title,
                font_title(),
                T.color.text,
            );
            let bar = header.shrink2(Vec2::new(T.space.m, 0.0));
            let mut child = ui.new_child(
                UiBuilder::new()
                    .max_rect(bar)
                    .layout(Layout::right_to_left(Align::Center))
                    .id_salt((self.name, "header")),
            );
            actions(&mut child, state);
            inner.min.y = header.bottom();
        }
        let content = match (self.flush, self.title.is_some()) {
            (true, _) => inner,
            // Under a header, the content starts right below it.
            (false, true) => {
                let mut r = inner.shrink(T.space.l);
                r.min.y = inner.min.y;
                r
            }
            (false, false) => inner.shrink(T.space.l),
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

/// The frame of floating surfaces (palette, menus, dialogs): surface, one border, no shadow.
pub fn popup_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(T.color.surface)
        .stroke(egui::Stroke::new(1.0, T.color.border))
        .corner_radius(T.radius_card)
        .inner_margin(T.space.s)
}

/// A card: a raised block of a page (the new-session form, a wizard step).
pub fn card_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(T.color.surface)
        .stroke(egui::Stroke::new(1.0, T.color.border))
        .corner_radius(T.radius_card)
        .inner_margin(T.space.xl)
}

/// A block inside a card or the thread (an approval, a tool result): one step lighter.
pub fn inset_frame() -> egui::Frame {
    egui::Frame::new()
        .fill(T.color.surface_hover)
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
        painter.rect_filled(snap(painter, rect), T.radius, fill);
    }
    if response.has_focus() {
        focus_ring(painter, rect);
    }
}

/// A clickable full-width row of the standard height; `body` lays its content out, `label`
/// names it for screen readers.
pub fn row(ui: &mut Ui, selected: bool, label: &str, body: impl FnOnce(&mut Ui)) -> Response {
    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), T.size.row), Sense::click());
    response
        .widget_info(|| WidgetInfo::selected(WidgetType::SelectableLabel, true, selected, label));
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
    let value_galley = fit(
        &painter,
        value,
        font_mono(),
        T.color.text,
        rect.width() * 0.65,
    );
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
        ButtonKind::Primary => font_strong(),
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
    let (fill, edge, ink) = match (kind, enabled) {
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
    if let Some(edge) = edge {
        outline(painter, rect, T.radius, edge);
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

/// The rounded track of a segmented control holding options of `widths` and `height`;
/// returns the rectangle of each option, left to right.
fn track(ui: &mut Ui, widths: &[f32], height: f32) -> Vec<Rect> {
    let pad = T.space.xs - 1.0;
    let size = Vec2::new(widths.iter().sum::<f32>() + 2.0 * pad, height);
    let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(snap(painter, rect), T.radius + 2, T.color.surface_hover);
    let mut x = rect.left() + pad;
    widths
        .iter()
        .map(|w| {
            let r = Rect::from_min_size(
                Pos2::new(x, rect.top() + pad),
                Vec2::new(*w, height - 2.0 * pad),
            );
            x += w;
            r
        })
        .collect()
}

/// One option of a track: the active one is raised, the others read secondary.
fn track_option(
    ui: &mut Ui,
    rect: Rect,
    text: &str,
    font: FontId,
    active: bool,
    tooltip: &str,
) -> Response {
    let response = ui.interact(rect, ui.id().with(("segment", text)), Sense::click());
    response.widget_info(|| WidgetInfo::selected(WidgetType::RadioButton, true, active, text));
    let painter = ui.painter();
    let rect = snap(painter, rect);
    if active {
        painter.rect_filled(rect, T.radius, T.color.surface);
        outline(painter, rect, T.radius, T.color.border);
    }
    if response.has_focus() {
        focus_ring(painter, rect);
    }
    let ink = match (active, response.hovered()) {
        (true, _) | (false, true) => T.color.text,
        (false, false) => T.color.text_secondary,
    };
    painter.text(rect.center(), Align2::CENTER_CENTER, text, font, ink);
    response.on_hover_text(tooltip)
}

/// Options laid out on a track; `true` when one was just picked.
fn choice<V: PartialEq + Copy>(
    ui: &mut Ui,
    value: &mut V,
    options: &[(V, &str, &str)],
    font: FontId,
    pad: f32,
    height: f32,
) -> bool {
    let widths: Vec<f32> = options
        .iter()
        .map(|(_, text, _)| {
            let galley =
                ui.painter()
                    .layout_no_wrap((*text).to_owned(), font.clone(), T.color.text);
            galley.size().x + 2.0 * pad
        })
        .collect();
    let mut changed = false;
    for ((option, text, tooltip), rect) in options.iter().zip(track(ui, &widths, height)) {
        let active = *value == *option;
        if track_option(ui, rect, text, font.clone(), active, tooltip).clicked() && !active {
            *value = *option;
            changed = true;
        }
    }
    changed
}

/// A segmented control: a rounded track whose active option is raised.
pub fn segmented<V: PartialEq + Copy>(
    ui: &mut Ui,
    value: &mut V,
    segments: &[Segment<'_, V>],
    font: FontId,
) -> bool {
    let options: Vec<(V, &str, &str)> = reading_order(ui, segments.len())
        .into_iter()
        .map(|k| (segments[k].value, segments[k].label, segments[k].tooltip))
        .collect();
    choice(
        ui,
        value,
        &options,
        font,
        T.space.l,
        T.size.control + T.space.xs,
    )
}

/// A small segmented choice of short labels (a language).
pub fn segmented_small<V: PartialEq + Copy>(
    ui: &mut Ui,
    value: &mut V,
    options: &[(V, &str)],
    tooltip: &str,
) -> bool {
    let options: Vec<(V, &str, &str)> = reading_order(ui, options.len())
        .into_iter()
        .map(|k| (options[k].0, options[k].1, tooltip))
        .collect();
    choice(
        ui,
        value,
        &options,
        font_meta(),
        T.space.s,
        T.size.row - T.space.xs,
    )
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
    let size = Vec2::new(width + 2.0 * ui.spacing().button_padding.x, T.size.control);
    ui.allocate_ui(size, |ui| {
        egui::ComboBox::from_id_salt(id)
            .selected_text(rich(current, font_body(), T.color.text))
            .width(width)
            .truncate()
            .icon(|ui, rect, _, _| {
                icons::paint(
                    ui.painter(),
                    rect,
                    Icon::ChevronDown,
                    T.color.text_secondary,
                )
            })
            .show_ui(ui, body)
            .inner
    })
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
    painter.rect_filled(rect, T.radius, T.color.surface);
    outline(painter, rect, T.radius, T.color.border);
    rect
}

fn edit(ui: &mut Ui, rect: Rect, edit: TextEdit<'_>) -> Response {
    let inner = rect.shrink2(Vec2::new(T.space.m, 0.0));
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
        // The hint is a label, not data: it reads in the interface face.
        TextEdit::singleline(text).font(font_mono()).hint_text(rich(
            hint,
            font_body(),
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
    record(ui.ctx(), "multiline", rect);
    let inner = rect.shrink2(Vec2::new(T.space.m, T.space.s));
    let edit = TextEdit::multiline(text)
        .font(font_body())
        .hint_text(rich(hint, font_body(), T.color.text_disabled))
        .frame(egui::Frame::NONE)
        .margin(egui::Margin::ZERO)
        .desired_rows(rows);
    let response = scrolled(ui, inner, ("multiline", hint), edit);
    if response.has_focus() {
        focus_ring(ui.painter(), rect);
    }
    response
}

/// A multi-line edit held in `rect`: past its height the text scrolls inside it, and never
/// draws over what comes below.
fn scrolled(
    ui: &mut Ui,
    rect: Rect,
    salt: impl std::hash::Hash + std::fmt::Debug,
    edit: TextEdit<'_>,
) -> Response {
    // A child that allocates nothing: the frame already holds the space in the parent.
    let mut child = ui.new_child(
        UiBuilder::new()
            .max_rect(rect)
            .layout(Layout::top_down(Align::Min)),
    );
    egui::ScrollArea::vertical()
        .id_salt(salt)
        .max_height(rect.height())
        .auto_shrink(false)
        .show(&mut child, |ui| {
            let width = ui.available_width();
            ui.add(edit.desired_width(width))
        })
        .inner
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
    painter.rect_filled(track, T.radius, T.color.surface_hover);
    let mut fill = track;
    fill.set_width(track.width() * fraction.clamp(0.0, 1.0));
    let color = if over { T.color.error } else { T.color.data[4] };
    painter.rect_filled(snap(painter, fill), T.radius, color);
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
            let galley = fit(
                &painter,
                column.title,
                font_meta(),
                T.color.text_secondary,
                span.span(),
            );
            let x = match column.right {
                true => span.max - galley.size().x,
                false => span.min,
            };
            let pos = Pos2::new(x, rect.center().y - galley.size().y / 2.0);
            painter.galley(pos, galley, T.color.text_secondary);
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
        let label: Vec<&str> = cells.iter().map(|c| c.text).collect();
        response.widget_info(|| {
            WidgetInfo::selected(
                WidgetType::SelectableLabel,
                true,
                selected,
                label.join(" · "),
            )
        });
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
            let galley = fit(&painter, cell.text, font_mono(), ink, span.max - left);
            let x = match column.right {
                true => span.max - galley.size().x,
                false => left,
            };
            let pos = Pos2::new(x, rect.center().y - galley.size().y / 2.0);
            painter.galley(pos, galley, ink);
        }
        response
    }
}

// ------------------------------------------------------------------ page parts

/// A choice of a page drawn as a tile: an icon and its label. The picked one is filled with
/// the selection, outlined with the accent and checked.
pub fn tile(ui: &mut Ui, icon: Icon, text: &str, selected: bool, width: f32) -> Response {
    let size = Vec2::new(width, T.size.cta + T.space.l);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    response.widget_info(|| WidgetInfo::selected(WidgetType::RadioButton, true, selected, text));
    let painter = ui.painter();
    let rect = snap(painter, rect);
    let (fill, edge, ink) = match (selected, response.hovered()) {
        (true, _) => (T.color.selection, T.color.accent, T.color.accent),
        (false, true) => (T.color.surface_hover, T.color.border, T.color.text),
        (false, false) => (T.color.surface, T.color.border, T.color.text_secondary),
    };
    painter.rect_filled(rect, T.radius, fill);
    outline(painter, rect, T.radius, edge);
    let glyph = Rect::from_center_size(
        Pos2::new(rect.left() + T.space.l + T.size.icon / 2.0, rect.center().y),
        Vec2::splat(T.size.icon),
    );
    icons::paint(painter, glyph, icon, ink);
    let check = (T.size.icon + T.space.s) * f32::from(u8::from(selected));
    let left = glyph.right() + T.space.m;
    let wrap = (rect.right() - T.space.l - check - left).max(T.size.icon);
    let galley = painter.layout(text.to_owned(), font_strong(), T.color.text, wrap);
    let pos = Pos2::new(left, rect.center().y - galley.size().y / 2.0);
    painter.galley(pos, galley, T.color.text);
    if selected {
        let mark = Rect::from_center_size(
            Pos2::new(
                rect.right() - T.space.l - T.size.icon / 2.0,
                rect.center().y,
            ),
            Vec2::splat(T.size.icon),
        );
        icons::paint(painter, mark, Icon::Check, T.color.accent);
    }
    if response.has_focus() {
        focus_ring(painter, rect.expand(2.0));
    }
    response
}

/// A suggestion the user can take: a small outlined label.
pub fn chip(ui: &mut Ui, text: &str) -> Response {
    let galley = ui
        .painter()
        .layout_no_wrap(text.to_owned(), font_body(), T.color.text);
    let size = Vec2::new(galley.size().x + 2.0 * T.space.m, T.size.control);
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, true, text));
    let painter = ui.painter();
    let rect = snap(painter, rect);
    let (fill, ink) = match response.hovered() {
        true => (T.color.surface_hover, T.color.text),
        false => (T.color.surface, T.color.text_secondary),
    };
    painter.rect_filled(rect, T.radius, fill);
    outline(painter, rect, T.radius, T.color.border);
    if response.has_focus() {
        focus_ring(painter, rect);
    }
    painter.galley_with_override_text_color(rect.center() - galley.size() / 2.0, galley, ink);
    response
}

/// A numbered step of a form: the number in a ring, then its title.
pub fn step(ui: &mut Ui, number: usize, title: &str) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = T.space.m;
        let (rect, _) = ui.allocate_exact_size(Vec2::splat(T.size.control), Sense::hover());
        let painter = ui.painter();
        let radius = T.size.control / 2.0 - 2.0;
        painter.circle_filled(rect.center(), radius, T.color.surface_hover);
        painter.circle_stroke(
            rect.center(),
            radius,
            Stroke::new(px(painter), T.color.border),
        );
        painter.text(
            rect.center(),
            Align2::CENTER_CENTER,
            number.to_string(),
            font_body(),
            T.color.text,
        );
        ui.add(Label::new(rich(title, font_brand(), T.color.text)).truncate());
    });
}

/// A labeled on/off switch in an outlined pill; `true` when it was just flipped.
pub fn switch(ui: &mut Ui, icon: Icon, text: &str, on: &mut bool, tooltip: &str) -> bool {
    let galley = ui
        .painter()
        .layout_no_wrap(text.to_owned(), font_body(), T.color.text);
    let knob = Vec2::new(T.size.control - T.space.xs, T.space.l);
    let pad = T.space.m;
    let width = pad + T.size.icon + T.space.s + galley.size().x + T.space.m + knob.x + pad;
    let (rect, response) = ui.allocate_exact_size(Vec2::new(width, T.size.control), Sense::click());
    response.widget_info(|| WidgetInfo::selected(WidgetType::Checkbox, true, *on, text));
    let flipped = response.clicked();
    if flipped {
        *on = !*on;
    }
    let painter = ui.painter();
    let rect = snap(painter, rect);
    if response.hovered() {
        painter.rect_filled(rect, T.radius, T.color.surface_hover);
    }
    outline(painter, rect, T.radius, T.color.border);
    let glyph = Rect::from_min_size(
        Pos2::new(rect.left() + pad, rect.center().y - T.size.icon / 2.0),
        Vec2::splat(T.size.icon),
    );
    let mark = if *on {
        T.color.ok
    } else {
        T.color.text_secondary
    };
    icons::paint(painter, glyph, icon, mark);
    let text_pos = Pos2::new(
        glyph.right() + T.space.s,
        rect.center().y - galley.size().y / 2.0,
    );
    painter.galley(text_pos, galley, T.color.text);
    let track = Rect::from_min_size(
        Pos2::new(rect.right() - pad - knob.x, rect.center().y - knob.y / 2.0),
        knob,
    );
    let radius = knob.y / 2.0;
    if *on {
        painter.rect_filled(track, radius, T.color.accent);
    } else {
        painter.rect_filled(track, radius, T.color.surface_hover);
        outline(painter, track, radius as u8, T.color.border);
    }
    let x = match *on {
        true => track.right() - radius,
        false => track.left() + radius,
    };
    painter.circle_filled(Pos2::new(x, track.center().y), radius - 3.0, T.color.text);
    if response.has_focus() {
        focus_ring(painter, rect);
    }
    response.on_hover_text(tooltip);
    flipped
}

/// An entry of the activity bar in `rect`; `badge` counts what waits there.
pub fn activity(
    ui: &mut Ui,
    rect: Rect,
    icon: Icon,
    active: bool,
    tooltip: &str,
    badge: usize,
) -> Response {
    let response = ui.interact(rect, ui.id().with(("activity", tooltip)), Sense::click());
    response.widget_info(|| WidgetInfo::selected(WidgetType::RadioButton, true, active, tooltip));
    let painter = ui.painter();
    let cell = snap(
        painter,
        Rect::from_center_size(rect.center(), Vec2::splat(T.size.cta)),
    );
    let ink = match (active, response.hovered()) {
        (true, _) => {
            painter.rect_filled(cell, T.radius, T.color.selection);
            T.color.accent
        }
        (false, true) => {
            painter.rect_filled(cell, T.radius, T.color.surface_hover);
            T.color.text
        }
        (false, false) => T.color.text_secondary,
    };
    icons::paint(painter, cell, icon, ink);
    if badge > 0 {
        let center = Pos2::new(cell.right() - T.space.s, cell.top() + T.space.s);
        painter.circle_filled(center, T.space.s, T.color.warn);
        painter.text(
            center,
            Align2::CENTER_CENTER,
            badge.min(9).to_string(),
            font_meta(),
            T.color.on_action,
        );
    }
    if response.has_focus() {
        focus_ring(painter, cell);
    }
    response.on_hover_text(tooltip)
}

/// One segment of a [`capsule`]: its text (cut at `max` wide), the state dot after it, and
/// `on` for a segment that toggles (`None`: it only reports).
pub struct Seg<'a> {
    pub text: &'a str,
    pub state: State,
    pub tip: &'a str,
    pub max: f32,
    pub on: Option<bool>,
}

/// Segments side by side in one outlined capsule of control height, a hairline between
/// them: `[project · copy Ok ● | Local ●]`. Returns each segment's response.
pub fn capsule(ui: &mut Ui, segs: &[Seg<'_>]) -> Vec<Response> {
    let pad = T.space.m;
    let galleys: Vec<_> = segs
        .iter()
        .map(|s| {
            let mut job = egui::text::LayoutJob::simple_singleline(
                s.text.to_owned(),
                font_body(),
                T.color.text,
            );
            job.wrap = egui::text::TextWrapping::truncate_at_width(s.max);
            ui.painter().layout_job(job)
        })
        .collect();
    let widths: Vec<f32> = galleys
        .iter()
        .map(|g| (pad + g.size().x + T.space.s + T.size.dot + pad).round())
        .collect();
    let (rect, _) = ui.allocate_exact_size(
        Vec2::new(widths.iter().sum(), T.size.control),
        Sense::hover(),
    );
    let rect = snap(ui.painter(), rect);
    let last = segs.len().saturating_sub(1);
    let mut x = rect.left();
    let mut out = Vec::with_capacity(segs.len());
    for (k, ((seg, galley), w)) in segs.iter().zip(galleys).zip(widths).enumerate() {
        let cell = Rect::from_min_max(Pos2::new(x, rect.top()), Pos2::new(x + w, rect.bottom()));
        x += w;
        let sense = match seg.on {
            Some(_) => Sense::click(),
            None => Sense::hover(),
        };
        let response = ui.interact(cell, ui.id().with(("capsule", k)), sense);
        response.widget_info(|| match seg.on {
            Some(on) => WidgetInfo::selected(WidgetType::Checkbox, true, on, seg.text),
            None => WidgetInfo::labeled(WidgetType::Label, true, seg.text),
        });
        let painter = ui.painter();
        if seg.on.is_some() && response.hovered() {
            let r = T.radius;
            let corners = egui::CornerRadius {
                nw: if k == 0 { r } else { 0 },
                sw: if k == 0 { r } else { 0 },
                ne: if k == last { r } else { 0 },
                se: if k == last { r } else { 0 },
            };
            painter.rect_filled(cell, corners, T.color.surface_hover);
        }
        if k > 0 {
            vline(painter, cell.left(), cell.y_range(), T.color.border);
        }
        let text_pos = Pos2::new(cell.left() + pad, cell.center().y - galley.size().y / 2.0);
        let dot_x = text_pos.x + galley.size().x + T.space.s + T.size.dot / 2.0;
        painter.galley(text_pos, galley, T.color.text);
        icons::dot(painter, Pos2::new(dot_x, cell.center().y), seg.state);
        if response.has_focus() {
            focus_ring(painter, cell);
        }
        out.push(response.on_hover_text(seg.tip));
    }
    outline(ui.painter(), rect, T.radius, T.color.border);
    out
}

pub fn spinner(ui: &mut Ui) -> Response {
    ui.add(
        egui::Spinner::new()
            .size(T.size.icon - T.space.xs)
            .color(T.color.text_secondary),
    )
}

pub fn busy(ui: &mut Ui, text: impl Into<String>) -> Response {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = T.space.s;
        let rtl = ui.layout().prefer_right_to_left();
        if !rtl {
            spinner(ui);
        }
        let label = ui.add(Label::new(rich(text, font_body(), T.color.text_secondary)));
        if rtl {
            spinner(ui);
        }
        label
    })
    .inner
}

pub struct ToolCard<'a> {
    pub title: &'a str,
    pub tool: &'a str,
    pub state: State,
    pub badge: String,
    pub elapsed: Option<String>,
    pub summary: &'a str,
    pub sections: Vec<(&'a str, String)>,
    pub copy: &'a str,
    pub copied: &'a str,
}

pub fn tool_card(ui: &mut Ui, id: Id, card: &ToolCard<'_>) {
    let open_id = id.with("open");
    let open = ui.data(|d| d.get_temp::<bool>(open_id)).unwrap_or(false);
    let response = row(ui, false, card.title, |ui| {
        ui.spacing_mut().item_spacing.x = T.space.s;
        let (glyph, _) = ui.allocate_exact_size(Vec2::splat(T.size.icon), Sense::hover());
        let chevron = match open {
            true => Icon::ChevronDown,
            false => Icon::ChevronRight,
        };
        icons::paint(ui.painter(), glyph, chevron, T.color.text_secondary);
        match card.state {
            State::Active => {
                spinner(ui);
            }
            state => {
                let (dot, _) = ui.allocate_exact_size(Vec2::splat(T.size.dot), Sense::hover());
                icons::dot(ui.painter(), dot.center(), state);
            }
        }
        ui.add(Label::new(rich(card.title, font_strong(), T.color.text)).selectable(false));
        if card.tool != card.title {
            ui.add(
                Label::new(rich(card.tool, font_mono(), T.color.text_secondary)).selectable(false),
            );
        }
        let ink = match card.state {
            State::Ok | State::Error | State::Warn => icons::state_color(card.state),
            _ => T.color.text_secondary,
        };
        ui.add(Label::new(rich(&card.badge, font_meta(), ink)).selectable(false));
        if let Some(elapsed) = &card.elapsed {
            ui.add(
                Label::new(rich(elapsed, font_meta(), T.color.text_secondary)).selectable(false),
            );
        }
        ui.add(
            Label::new(rich(card.summary, font_meta(), T.color.text_secondary))
                .truncate()
                .selectable(false),
        );
    });
    let response = response.on_hover_text(card.summary);
    if response.clicked() {
        ui.data_mut(|d| d.insert_temp(open_id, !open));
    }
    if !open {
        return;
    }
    ui.horizontal_top(|ui| {
        ui.add_space(T.space.s + T.size.icon);
        inset_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            for (k, (label, body)) in card.sections.iter().enumerate() {
                if k > 0 {
                    separator(ui);
                }
                let copied_id = id.with(("copied", k));
                let now = ui.input(|i| i.time);
                let copied = ui
                    .data(|d| d.get_temp::<f64>(copied_id))
                    .is_some_and(|at| now - at < 1.5);
                ui.horizontal(|ui| {
                    meta(ui, *label);
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if copied {
                            status(ui, State::Ok, card.copied);
                            ui.ctx()
                                .request_repaint_after(std::time::Duration::from_millis(500));
                        } else if button(ui, card.copy, ButtonKind::Ghost).clicked() {
                            ui.ctx().copy_text(body.clone());
                            ui.data_mut(|d| d.insert_temp(copied_id, now));
                        }
                    });
                });
                mono_wrapped(ui, body.as_str(), T.color.text_secondary);
            }
        });
    });
    ui.add_space(T.space.xs);
}

pub fn banner(ui: &mut Ui, state: State, text: &str, detail: &str) {
    inset_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = T.space.s;
            let (dot, _) =
                ui.allocate_exact_size(Vec2::new(T.size.dot, T.size.row), Sense::hover());
            icons::dot(ui.painter(), dot.center(), state);
            ui.add(Label::new(rich(text, font_strong(), T.color.text)));
            if !detail.is_empty() {
                ui.add(Label::new(rich(detail, font_meta(), T.color.text_secondary)).wrap());
            }
        });
    });
}

/// The frame of the [`composer`]: three rows of text and the nested send button.
const COMPOSER_FRAME: f32 = 3.0 * T.size.row + T.space.l;
/// The line under the frame: the last message and the keys hint.
const COMPOSER_NOTE: f32 = T.size.row - T.space.s;

/// The height [`composer`] takes, its note line included.
pub fn composer_height() -> f32 {
    COMPOSER_FRAME + T.space.xs + COMPOSER_NOTE
}

/// The agent's message field: one rounded card holding the text and the send button in its
/// bottom-right corner, then a line of its own under it with `note` (the last message, if
/// any) on the left and the keys hint on the right, so neither ever covers the other. `true`
/// when the user sends while `ready` (Enter or the button; Shift+Enter is a new line).
pub fn composer(
    ui: &mut Ui,
    text: &mut String,
    hint: &str,
    keys: &str,
    note: Option<(&str, bool)>,
    ready: bool,
    send_tip: &str,
) -> bool {
    // One allocation for the frame and its note line: the parts are placed in it, so no
    // child scope can move the parent's cursor between them.
    let size = Vec2::new(ui.available_width(), composer_height());
    let (whole, _) = ui.allocate_exact_size(size, Sense::hover());
    let painter = ui.painter();
    let rect = snap(
        painter,
        Rect::from_min_size(whole.min, Vec2::new(whole.width(), COMPOSER_FRAME)),
    );
    let line = Rect::from_min_max(
        Pos2::new(whole.left(), whole.bottom() - COMPOSER_NOTE),
        whole.max,
    );
    painter.rect_filled(rect, T.radius_card, T.color.surface);
    let id = ui.id().with("composer");
    let focused = ui.memory(|m| m.has_focus(id));
    let edge = if focused {
        T.color.accent
    } else {
        T.color.border
    };
    outline(painter, rect, T.radius_card, edge);
    // Taken before the field sees it: Enter sends, Shift+Enter stays a new line.
    let enter = focused
        && ui.input_mut(|i| {
            !i.modifiers.shift && i.consume_key(egui::Modifiers::NONE, egui::Key::Enter)
        });
    let send = Rect::from_min_size(
        rect.max - Vec2::splat(T.space.s + T.size.control),
        Vec2::splat(T.size.control),
    );
    // The text stops short of the button's column: a long line wraps instead of running under it.
    let field = Rect::from_min_max(
        rect.min + Vec2::new(T.space.l, T.space.m),
        Pos2::new(send.left() - T.space.s, rect.bottom() - T.space.m),
    );
    let edit = TextEdit::multiline(text)
        .id(id)
        .font(font_body())
        .hint_text(rich(hint, font_body(), T.color.text_disabled))
        .frame(egui::Frame::NONE)
        .margin(egui::Margin::ZERO)
        .desired_rows(3);
    scrolled(ui, field, id.with("scroll"), edit);
    let sense = match ready {
        true => Sense::click(),
        false => Sense::hover(),
    };
    let response = ui.interact(send, id.with("send"), sense);
    response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, ready, send_tip));
    let painter = ui.painter();
    let (fill, ink) = match (ready, response.hovered()) {
        (true, true) => (T.color.action_hover, T.color.on_action),
        (true, false) => (T.color.action, T.color.on_action),
        (false, _) => (T.color.surface_hover, T.color.text_disabled),
    };
    painter.rect_filled(snap(painter, send), T.radius, fill);
    icons::paint(painter, send, Icon::Send, ink);
    if response.has_focus() {
        focus_ring(painter, send);
    }
    let clicked = response.on_hover_text(send_tip).clicked();
    let mut row = ui.new_child(
        UiBuilder::new()
            .max_rect(line.shrink2(Vec2::new(T.space.xs, 0.0)))
            .layout(Layout::right_to_left(Align::Center)),
    );
    row.spacing_mut().item_spacing.x = T.space.l;
    // Right to left: the keys keep their room, the note is cut where it would meet them.
    row.add(Label::new(rich(keys, font_meta(), T.color.text_disabled)).truncate());
    if let Some((text, error)) = note {
        row.with_layout(Layout::left_to_right(Align::Center), |ui| {
            ui.spacing_mut().item_spacing.x = T.space.s;
            let state = if error { State::Error } else { State::Ok };
            let (dot, _) =
                ui.allocate_exact_size(Vec2::new(T.space.s, COMPOSER_NOTE), Sense::hover());
            icons::dot(ui.painter(), dot.center(), state);
            ui.add(Label::new(rich(text, font_meta(), T.color.text_secondary)).truncate());
        });
    }
    ready && (clicked || enter)
}

/// A message of the user in the thread: a raised bubble on the right, its text read from
/// the left.
pub fn bubble(ui: &mut Ui, text: &str) {
    let max = (ui.available_width() * 0.8).max(T.size.left);
    ui.with_layout(Layout::right_to_left(Align::Min), |ui| {
        egui::Frame::new()
            .fill(T.color.surface_hover)
            .corner_radius(T.radius_card)
            .inner_margin(egui::Margin::symmetric(T.space.l as i8, T.space.m as i8))
            .show(ui, |ui| {
                ui.set_max_width(max);
                // The bubble's own layout is left to right, or the wrapped lines align right.
                ui.with_layout(Layout::top_down(Align::Min), |ui| {
                    super::markdown::show(ui, text);
                });
            });
    });
}
