//! Thin entry point: every behavior lives in `weight_surgeon::app` and is tested there.

use std::io::{IsTerminal, Write};

use weight_surgeon::app;
use weight_surgeon::ui::Ui;

fn main() {
    // always_ansi passes every byte through untouched and turns on VT processing on
    // Windows consoles; whether colors are emitted at all is decided by Ui::detect.
    // Unlocked handles: a panic on a rayon thread must still reach stderr.
    let mut out = anstream::AutoStream::always_ansi(std::io::stdout());
    let mut err = anstream::AutoStream::always_ansi(std::io::stderr());
    app::install_panic_hook();
    let json = std::env::args_os().skip(1).any(|a| a == "--json");
    let banner = !json && std::io::stdout().is_terminal();
    let code = match app::parse_cli_with(
        std::env::args_os(),
        &Ui::detect(json),
        banner,
        &mut out,
        &mut err,
    ) {
        Ok(cli) => app::run_with(cli, &app::real_deps(), &mut out, &mut err),
        Err(code) => code,
    };
    let _ = out.flush();
    std::process::exit(code);
}
