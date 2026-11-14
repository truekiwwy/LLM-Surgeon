//! The first-run wizard, a page over the whole body of the window in four steps: welcome
//! (language, the Atelier's mode, local only), AI providers and agents (what the scan found),
//! workspace (the one thing the work needs) and optional components (llama.cpp, the sample
//! model). Every confirmed choice is saved at once; closing the app midway reopens it at the
//! first step with those choices. Opened from System on one step, it is that panel alone.

use eframe::egui::{self, Pos2, Rect, ScrollArea, Ui};

use super::components::{self as c, ButtonKind, Segment};
use super::i18n::{t, tf};
use super::icons::State;
use super::theme::TOKENS as T;
use super::views::system::{machine_facts, provider_detail, provider_list, Show};
use crate::app::{Mode, SurgeonApp, Tab};
use crate::desktop::setup::Pick;
use crate::install;
use crate::releases::{Target, Variant};
use crate::sample_model::QWEN;
use crate::settings::{Lang, Settings};
use crate::studio::view::Studio;

/// The widest the page grows, on the 8 grid.
const PAGE: f32 = 960.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Step {
    Welcome,
    Providers,
    Workspace,
    Components,
}

impl Step {
    pub const ALL: [Step; 4] = [
        Step::Welcome,
        Step::Providers,
        Step::Workspace,
        Step::Components,
    ];

    pub fn index(self) -> usize {
        Self::ALL.iter().position(|s| *s == self).unwrap_or(0)
    }

    /// `wizard:2` of `--screen`: the steps count from 1.
    pub fn from_number(n: usize) -> Option<Self> {
        Self::ALL.get(n.checked_sub(1)?).copied()
    }

    fn title(self) -> &'static str {
        match self {
            Step::Welcome => t("wiz.step.welcome"),
            Step::Providers => t("wiz.step.providers"),
            Step::Workspace => t("wiz.step.workspace"),
            Step::Components => t("wiz.step.components"),
        }
    }
}

pub struct Wizard {
    pub step: Step,
    /// Opened from System on one step: that panel alone, closed by Close.
    pub single: bool,
    /// Step 2 shows the CLI runtimes, else the servers and cloud providers.
    pub agents_tab: bool,
    pub workspace: String,
    pub workspace_error: Option<String>,
    pub pick: Pick,
    pub consent: bool,
}

impl Wizard {
    pub fn new(settings: &Settings) -> Self {
        Self {
            step: Step::Welcome,
            single: false,
            agents_tab: false,
            workspace: settings
                .workspace
                .clone()
                .unwrap_or_else(crate::paths::default_workspace)
                .display()
                .to_string(),
            workspace_error: None,
            pick: Pick::default(),
            consent: false,
        }
    }

    /// One step alone, with `pick` ticked (a repair from System).
    pub fn at(settings: &Settings, step: Step, pick: Pick) -> Self {
        Self {
            step,
            single: true,
            pick,
            ..Self::new(settings)
        }
    }
}

/// What the footer asked for.
enum Go {
    Back,
    Next,
    Skip,
    Close,
}

/// The page, over `body` (the three zones of the window).
pub fn show(ui: &mut Ui, app: &mut SurgeonApp, body: Rect) {
    let Some(mut w) = app.desktop.wizard.take() else {
        return;
    };
    let width = body.width().min(PAGE);
    let page = Rect::from_min_max(
        Pos2::new(body.center().x - width / 2.0, body.top()),
        Pos2::new(body.center().x + width / 2.0, body.bottom()),
    );
    let title = match w.single {
        true => w.step.title().to_owned(),
        false => tf(
            "wiz.title",
            &[
                ("n", &format!("{:02}", w.step.index() + 1)),
                ("total", &format!("{:02}", Step::ALL.len())),
                ("title", &w.step.title()),
            ],
        ),
    };
    let mut go = None;
    c::Panel::new("wizard").title(&title).show(ui, page, |ui| {
        let inner = ui.max_rect();
        let footer = Rect::from_min_max(
            Pos2::new(inner.left(), inner.bottom() - T.size.cta),
            inner.max,
        );
        let content = Rect::from_min_max(
            inner.min,
            Pos2::new(inner.right(), footer.top() - T.space.m),
        );
        c::region(ui, content, "wizard-content", |ui| {
            if !w.single {
                let done = (w.step.index() + 1) as f32 / Step::ALL.len() as f32;
                c::progress(ui, done, false);
                ui.add_space(T.space.m);
            }
            match w.step {
                Step::Welcome => welcome(ui, app),
                Step::Providers => providers(ui, app, &mut w),
                Step::Workspace => workspace(ui, app, &mut w),
                Step::Components => components(ui, app, &mut w),
            }
        });
        c::region(ui, footer, "wizard-footer", |ui| {
            go = footer_bar(ui, app, &w);
        });
    });
    app.desktop.wizard = Some(w);
    match go {
        Some(Go::Back) => step_by(app, -1),
        Some(Go::Next) => step_by(app, 1),
        Some(Go::Skip) => skip(app, &crate::paths::default_workspace()),
        Some(Go::Close) => close(app),
        None => {}
    }
}

fn step_by(app: &mut SurgeonApp, delta: isize) {
    if let Some(w) = app.desktop.wizard.as_mut() {
        let k = w.step.index().saturating_add_signed(delta);
        w.step = Step::ALL[k.min(Step::ALL.len() - 1)];
    }
}

/// The end of the walk, or of a single panel: back to the work.
fn close(app: &mut SurgeonApp) {
    let walked = app.desktop.wizard.as_ref().is_some_and(|w| !w.single);
    app.desktop.close_wizard();
    if walked {
        app.tab = Tab::Atelier;
    }
}

/// `Passer la configuration`: the `default` workspace when there is none, the guided mode,
/// and straight to the Atelier.
fn skip(app: &mut SurgeonApp, default: &std::path::Path) {
    if app.desktop.settings.workspace.is_none() {
        match app.desktop.use_workspace(default) {
            Ok(()) => app.studio = Studio::new(default.to_path_buf()),
            Err(e) => app.desktop.notify(e),
        }
    }
    app.set_mode(Mode::Scalping);
    close(app);
}

fn footer_bar(ui: &mut Ui, app: &SurgeonApp, w: &Wizard) -> Option<Go> {
    let mut go = None;
    ui.horizontal_centered(|ui| {
        match (w.single, w.step) {
            (true, _) => {}
            (false, Step::Welcome) => {
                if c::link(ui, t("wiz.skip"), t("wiz.skip.tip")).clicked() {
                    go = Some(Go::Skip);
                }
            }
            (false, _) => {
                if c::button(ui, t("wiz.back"), ButtonKind::Ghost).clicked() {
                    go = Some(Go::Back);
                }
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let has_default = app.agents.cfg.default_connection.is_some();
            let ready_ws = app.desktop.settings.workspace.is_some();
            let (text, enabled, next) = match (w.single, w.step) {
                (true, _) => (t("wiz.close"), true, Go::Close),
                (false, Step::Components) => (t("wiz.finish"), true, Go::Close),
                (false, Step::Providers) if !has_default => (t("wiz.later"), true, Go::Next),
                (false, Step::Workspace) => (t("wiz.next"), ready_ws, Go::Next),
                (false, _) => (t("wiz.next"), true, Go::Next),
            };
            if c::button_if(
                ui,
                enabled,
                text,
                ButtonKind::Primary,
                t("wiz.workspace.why"),
            )
            .clicked()
            {
                go = Some(next);
            }
        });
    });
    go
}

// ------------------------------------------------------------------ 01 welcome

fn welcome(ui: &mut Ui, app: &mut SurgeonApp) {
    c::label(ui, t("wiz.welcome"));
    c::section(ui, t("wiz.lang"));
    let mut lang = app.desktop.lang();
    c::segmented(
        ui,
        &mut lang,
        &[
            Segment {
                value: Lang::Fr,
                label: "Français",
                tooltip: "Français",
            },
            Segment {
                value: Lang::En,
                label: "English",
                tooltip: "English",
            },
        ],
        c::font_body(),
    );
    if lang != app.desktop.lang() {
        app.desktop.set_lang(lang);
    }
    c::section(ui, t("wiz.mode"));
    let mut mode = app.mode;
    c::segmented(
        ui,
        &mut mode,
        &[
            Segment {
                value: Mode::Scalping,
                label: t("mode.scalping"),
                tooltip: t("mode.scalping.tip"),
            },
            Segment {
                value: Mode::Advance,
                label: t("mode.advance"),
                tooltip: t("mode.advance.tip"),
            },
        ],
        c::font_mono(),
    );
    if mode != app.mode {
        app.set_mode(mode);
    }
    ui.add_space(T.space.xs);
    c::secondary(ui, t("wiz.mode.scalping"));
    c::secondary(ui, t("wiz.mode.advance"));
    c::section(ui, t("wiz.network"));
    let mut local = app.desktop.settings.local_only;
    if c::checkbox(ui, &mut local, t("policy.local_only")).changed() {
        app.set_local_only(local);
    }
    c::secondary(ui, t("policy.local_only.tip"));
}

// ------------------------------------------------------------------ 02 providers and agents

fn providers(ui: &mut Ui, app: &mut SurgeonApp, w: &mut Wizard) {
    c::label(ui, t("wiz.providers.explain"));
    ui.add_space(T.space.s);
    ui.horizontal(|ui| {
        c::segmented(
            ui,
            &mut w.agents_tab,
            &[
                Segment {
                    value: false,
                    label: t("wiz.tab.providers"),
                    tooltip: t("wiz.tab.providers.tip"),
                },
                Segment {
                    value: true,
                    label: t("wiz.tab.agents"),
                    tooltip: t("wiz.tab.agents.tip"),
                },
            ],
            c::font_body(),
        );
        ui.add_space(T.space.l);
        let default = app
            .agents
            .cfg
            .default_connection
            .as_deref()
            .and_then(|id| app.agents.cfg.connection(id))
            .map(|c| c.label.clone());
        match default {
            Some(label) => c::status(ui, State::Ok, tf("wiz.default", &[("label", &label)])),
            None => c::status(ui, State::Idle, t("wiz.default.none")),
        };
    });
    ui.add_space(T.space.s);
    let rest = ui.available_rect_before_wrap();
    let split = (rest.width() * 0.58).floor();
    let left = Rect::from_min_max(rest.min, Pos2::new(rest.left() + split, rest.bottom()));
    let right = Rect::from_min_max(Pos2::new(left.right() + T.space.l, rest.top()), rest.max);
    let show = match w.agents_tab {
        true => Show::Agents,
        false => Show::Providers,
    };
    c::region(ui, left, "wizard-list", |ui| {
        ScrollArea::vertical()
            .id_salt("wizard-list")
            .auto_shrink(false)
            .show(ui, |ui| provider_list(ui, app, show));
    });
    c::region(ui, right, "wizard-detail", |ui| {
        ScrollArea::vertical()
            .id_salt("wizard-detail")
            .auto_shrink(false)
            .show(ui, |ui| provider_detail(ui, app));
    });
}

// ------------------------------------------------------------------ 03 workspace

fn workspace(ui: &mut Ui, app: &mut SurgeonApp, w: &mut Wizard) {
    c::label(ui, t("wiz.workspace.explain"));
    ui.add_space(T.space.s);
    let width = ui.available_width();
    c::field_mono(ui, &mut w.workspace, t("wiz.workspace.hint"), width);
    ui.add_space(T.space.s);
    if c::button(ui, t("wiz.workspace.use"), ButtonKind::Secondary).clicked() {
        let dir = std::path::PathBuf::from(w.workspace.trim());
        w.workspace_error = match app.desktop.use_workspace(&dir) {
            Ok(()) => {
                app.studio = Studio::new(dir);
                None
            }
            Err(e) => Some(e),
        };
    }
    match (&w.workspace_error, &app.desktop.settings.workspace) {
        (Some(e), _) => c::message(ui, e, true),
        (None, Some(dir)) => {
            let path = dir.display().to_string();
            c::status(ui, State::Ok, tf("wiz.workspace.ready", &[("path", &path)]));
        }
        (None, None) => {
            c::status(ui, State::Error, t("wiz.workspace.why"));
        }
    }
    c::section(ui, t("wiz.machine"));
    machine_facts(ui, app);
}

// ------------------------------------------------------------------ 04 components

fn components(ui: &mut Ui, app: &mut SurgeonApp, w: &mut Wizard) {
    let ctx = ui.ctx().clone();
    let local = app.desktop.settings.local_only;
    let workspace = app
        .desktop
        .settings
        .workspace
        .clone()
        .unwrap_or_else(|| w.workspace.trim().into());
    ScrollArea::vertical()
        .id_salt("wizard-components")
        .auto_shrink(false)
        .show(ui, |ui| {
            c::label(ui, t("wiz.components.explain"));
            if local {
                manual(ui, app, &workspace);
                return;
            }
            engine(ui, app, w, &ctx);
            sample(ui, w, &workspace);
            let env = app.desktop.env().clone();
            let setup = &mut app.desktop.setup;
            if w.pick.engine || w.pick.sample {
                ui.add_space(T.space.m);
                c::checkbox(ui, &mut w.consent, t("wiz.consent"));
                ui.add_space(T.space.s);
                let allowed = setup.download_allowed(w.pick, w.consent, local);
                if c::button_if(
                    ui,
                    allowed,
                    t("wiz.install"),
                    ButtonKind::Secondary,
                    t("wiz.install.why"),
                )
                .clicked()
                {
                    setup.install(&ctx, &env, &workspace, w.pick, w.consent, local);
                }
            }
            let setup = &app.desktop.setup;
            for p in setup.progress() {
                ui.add_space(T.space.s);
                c::mono(ui, p.label.as_str());
                c::progress(ui, p.fraction().unwrap_or(0.0), false);
            }
            if setup.installing() {
                c::secondary(ui, t("wiz.background"));
                if c::button(ui, t("action.cancel"), ButtonKind::Ghost).clicked() {
                    setup.cancel();
                }
            }
            outcomes(ui, app);
        });
}

fn engine(ui: &mut Ui, app: &mut SurgeonApp, w: &mut Wizard, ctx: &egui::Context) {
    c::section(ui, "llama.cpp");
    c::checkbox(ui, &mut w.pick.engine, t("wiz.engine"));
    c::secondary(ui, t("wiz.engine.explain"));
    if !w.pick.engine {
        return;
    }
    let setup = &mut app.desktop.setup;
    if !Target::current().supported() {
        c::message(ui, t("wiz.engine.unsupported"), true);
        return;
    }
    let mut variant = setup.variant;
    ui.add_space(T.space.xs);
    c::segmented(
        ui,
        &mut variant,
        &[
            Segment {
                value: Variant::Vulkan,
                label: t("wiz.variant.vulkan"),
                tooltip: t("wiz.variant.vulkan.tip"),
            },
            Segment {
                value: Variant::Cpu,
                label: t("wiz.variant.cpu"),
                tooltip: t("wiz.variant.cpu.tip"),
            },
            Segment {
                value: Variant::Hip,
                label: t("wiz.variant.hip"),
                tooltip: t("wiz.variant.hip.tip"),
            },
        ],
        c::font_body(),
    );
    setup.set_variant(variant);
    let no_vulkan = app
        .desktop
        .system
        .as_ref()
        .is_some_and(|s| s.vulkan == Some(false));
    let setup = &mut app.desktop.setup;
    if no_vulkan && setup.variant == Variant::Vulkan {
        c::status(ui, State::Warn, t("wiz.variant.no_vulkan"));
    }
    // Ticking the box asks the releases API once: the plan carries URL, size and hash.
    if setup.plan.is_none() && !setup.fetching_plan() {
        setup.fetch_plan(ctx, false);
    }
    ui.add_space(T.space.s);
    let desktop = &app.desktop;
    match &desktop.setup.plan {
        None => {
            c::status(ui, State::Active, t("wiz.plan.fetching"));
        }
        Some(Err(e)) => c::message(ui, e, true),
        Some(Ok(plan)) => {
            let pinned = desktop.settings.pins.get(&plan.name);
            c::inset_frame().show(ui, |ui| {
                c::strong(ui, format!("llama.cpp {} · {}", plan.tag, plan.variant));
                c::mono_wrapped(ui, plan.url.as_str(), T.color.text_secondary);
                let mib = plan.size as f64 / (1 << 20) as f64;
                c::kv(ui, t("wiz.size"), &format!("{mib:.1} MiB"));
                match (&plan.sha256, pinned) {
                    (Some(h), _) => {
                        c::mono_wrapped(ui, format!("SHA-256 {h}"), T.color.text_secondary)
                    }
                    (None, Some(p)) => c::mono_wrapped(
                        ui,
                        tf("wiz.hash.pinned", &[("sha", p)]),
                        T.color.text_secondary,
                    ),
                    (None, None) => c::status(ui, State::Warn, t("wiz.hash.none")),
                };
            });
            if desktop.setup.force {
                c::status(ui, State::Warn, t("wiz.engine.force"));
            }
        }
    }
}

fn sample(ui: &mut Ui, w: &mut Wizard, workspace: &std::path::Path) {
    c::section(ui, t("wiz.sample.title"));
    c::checkbox(ui, &mut w.pick.sample, QWEN.id);
    c::secondary(ui, t("wiz.sample.explain"));
    if QWEN.is_installed(workspace) {
        c::status(ui, State::Ok, t("wiz.sample.installed"));
    }
    if !w.pick.sample {
        return;
    }
    c::inset_frame().show(ui, |ui| {
        let mib = QWEN.total_size() as f64 / (1 << 20) as f64;
        c::strong(ui, QWEN.repo);
        c::kv(ui, t("wiz.size"), &format!("{mib:.0} MiB"));
        c::kv(ui, t("wiz.license"), QWEN.license);
        c::mono_wrapped(
            ui,
            format!("huggingface.co @ {}", &QWEN.revision[..12]),
            T.color.text_secondary,
        );
        for f in QWEN.files {
            c::mono_wrapped(
                ui,
                format!("{}  {} B  {}", f.name, f.size, f.sha256),
                T.color.text_secondary,
            );
        }
    });
}

/// Local only: what to download by hand, and where to put it.
fn manual(ui: &mut Ui, app: &SurgeonApp, workspace: &std::path::Path) {
    c::status(ui, State::Idle, t("wiz.manual"));
    c::section(ui, "llama.cpp");
    c::secondary(ui, t("wiz.manual.engine"));
    let root = install::llama_root(&app.desktop.env().components);
    c::mono_wrapped(ui, root.display().to_string(), T.color.text);
    c::section(ui, t("wiz.sample.title"));
    c::secondary(ui, t("wiz.manual.sample"));
    c::mono_wrapped(ui, QWEN.dir(workspace).display().to_string(), T.color.text);
}

/// What the installs ended with.
fn outcomes(ui: &mut Ui, app: &SurgeonApp) {
    let setup = &app.desktop.setup;
    match &setup.engine {
        Some(Ok(done)) => {
            ui.add_space(T.space.s);
            let line = tf(
                "wiz.engine.done",
                &[("tag", &done.tag), ("variant", &done.variant)],
            );
            c::status(ui, State::Ok, line);
            c::mono_wrapped(ui, done.hash_line.as_str(), T.color.text_secondary);
            c::mono_wrapped(
                ui,
                tf("wiz.smoke", &[("line", &done.smoke)]),
                T.color.text_secondary,
            );
            if let Some(e) = &done.register_error {
                c::status(
                    ui,
                    State::Warn,
                    tf("wiz.engine.unregistered", &[("error", e)]),
                );
            }
        }
        Some(Err(e)) => c::message(ui, e, true),
        None => {}
    }
    match &setup.sample {
        Some(Ok(())) => {
            c::status(ui, State::Ok, t("wiz.sample.done"));
        }
        Some(Err(e)) => c::message(ui, e, true),
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::Env;
    use crate::desktop::Desktop;

    fn app(dir: &std::path::Path) -> SurgeonApp {
        let env = Env {
            components: dir.join("data").join("components"),
            config: None,
            settings: dir.join("data").join("settings.json"),
        };
        let desktop = Desktop::detached(None, Settings::default(), env);
        let ctx = egui::Context::default();
        SurgeonApp::with_shell(&ctx, None, desktop, dir.join("projects"))
    }

    #[test]
    fn skipping_creates_the_default_workspace_and_ends_the_first_run_for_good() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = app(dir.path());
        app.desktop.open_wizard();
        app.set_mode(Mode::Advance);
        app.tab = Tab::System;
        let ws = dir.path().join("ws");
        skip(&mut app, &ws);
        assert!(ws.is_dir());
        assert_eq!(
            app.desktop.settings.workspace.as_deref(),
            Some(ws.as_path())
        );
        assert!(app.desktop.wizard.is_none());
        assert_eq!((app.mode, app.tab), (Mode::Scalping, Tab::Atelier));
        let saved = Settings::load(&dir.path().join("data").join("settings.json")).unwrap();
        assert!(saved.wizard_done, "it does not open by itself again");
    }

    #[test]
    fn a_repair_panel_closes_without_ending_the_first_run() {
        let dir = tempfile::tempdir().unwrap();
        let mut app = app(dir.path());
        app.desktop.open_fix(crate::diag::Fix::SampleModel, true);
        let w = app.desktop.wizard.as_ref().unwrap();
        assert!(w.single && w.pick.sample && app.desktop.setup.force);
        app.tab = Tab::System;
        close(&mut app);
        assert!(!app.desktop.settings.wizard_done);
        assert_eq!(app.tab, Tab::System, "back where the repair started");
        assert!(!app.desktop.setup.force);
    }

    #[test]
    fn steps_count_from_one_and_a_repair_is_one_step_alone() {
        assert_eq!(Step::from_number(1), Some(Step::Welcome));
        assert_eq!(Step::from_number(4), Some(Step::Components));
        assert_eq!(Step::from_number(0), None);
        assert_eq!(Step::from_number(5), None);
        let pick = Pick {
            engine: true,
            sample: false,
        };
        let w = Wizard::at(&Settings::default(), Step::Components, pick);
        assert!(w.single && w.pick.engine && !w.consent);
        assert!(!Wizard::new(&Settings::default()).single);
    }
}
