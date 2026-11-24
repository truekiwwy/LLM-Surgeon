//! The engine as the window runs it: the dependencies of the CLI, and `surgeon-gui mcp …`.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use weight_surgeon::app::{self, AppDeps};
use weight_surgeon::eval::SystemCommandRunner;
use weight_surgeon::hardware::SystemProbe;
use weight_surgeon::ui::{ColorMode, UiConfig};

/// The dependencies of the CLI without its Ctrl-C handler: the window keeps the default
/// terminal behavior, and the job's own flag is the cancellation.
pub(crate) fn deps(interrupted: Arc<AtomicBool>) -> AppDeps {
    AppDeps {
        probe: Box::new(SystemProbe::detect()),
        runner: Box::new(SystemCommandRunner),
        generator_factory: Box::new(app::real_generator),
        interrupted,
        ui: UiConfig {
            out: ColorMode::Off,
            err: ColorMode::Off,
            err_is_tty: false,
        },
    }
}

/// `surgeon-gui mcp …`: the engine's MCP server on this process's stdio, exactly as
/// `surgeon mcp …` (same library, same version as the window). The CLI runtimes start it, so
/// no separate `surgeon` binary is needed. `args` are the process arguments.
pub fn serve_mcp(args: impl Iterator<Item = std::ffi::OsString>) -> i32 {
    let argv = std::iter::once("surgeon".into()).chain(args.skip(1));
    let deps = deps(Arc::new(AtomicBool::new(false)));
    let (mut out, mut err) = (std::io::stdout(), std::io::stderr());
    match app::parse_cli(argv, &mut out, &mut err) {
        Ok(cli) => app::run_with(cli, &deps, &mut out, &mut err),
        Err(code) => code,
    }
}
