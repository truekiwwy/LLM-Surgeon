//! The few OS facts the app needs, read through the OS's own tools (`reg`, PowerShell) so
//! that the crate keeps `#![deny(unsafe_code)]`. Parsers are pure and tested; only
//! [`command`] and the callers touch the system.

use std::process::Command;

/// A child process that does not flash a console window when the GUI has none.
pub fn command(program: impl AsRef<std::ffi::OsStr>) -> Command {
    #[cfg_attr(not(windows), allow(unused_mut))]
    let mut cmd = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// Runs `cmd`, killing it after `timeout`; the exit code (`None` when killed or signalled)
/// and stdout then stderr, lossily decoded.
pub fn run_capture(
    mut cmd: Command,
    timeout: std::time::Duration,
) -> std::io::Result<(Option<i32>, String)> {
    use std::io::Read;
    use std::process::Stdio;
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    // Drained on threads: a full pipe would otherwise block the child before the deadline.
    let drain = |pipe: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut text = Vec::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.read_to_end(&mut text);
            }
            text
        })
    };
    let out = drain(
        child
            .stdout
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
    );
    let err = drain(
        child
            .stderr
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
    );
    let deadline = std::time::Instant::now() + timeout;
    let code = loop {
        if let Some(status) = child.try_wait()? {
            break status.code();
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    };
    let mut text = String::from_utf8_lossy(&out.join().unwrap_or_default()).into_owned();
    text.push_str(&String::from_utf8_lossy(&err.join().unwrap_or_default()));
    Ok((code, text))
}

/// `NAME    REG_SZ    value` from `reg query` output.
pub fn parse_reg_value(output: &str, name: &str) -> Option<String> {
    output.lines().find_map(|line| {
        let mut parts = line.split_whitespace();
        if parts.next()? != name {
            return None;
        }
        parts.next()?; // the REG_* type
        Some(parts.collect::<Vec<_>>().join(" "))
    })
}

#[cfg(windows)]
fn reg_query(key: &str, name: &str) -> Option<String> {
    let out = command("reg")
        .args(["query", key, "/v", name])
        .output()
        .ok()?;
    parse_reg_value(&String::from_utf8_lossy(&out.stdout), name)
}

/// The user's Windows locale (`fr-FR`).
pub fn user_locale() -> Option<String> {
    #[cfg(windows)]
    return reg_query(r"HKCU\Control Panel\International", "LocaleName");
    #[cfg(not(windows))]
    None
}

/// `host:port`, or `http=h:p;https=h2:p2;...` (the Windows manual-proxy syntax), as a proxy
/// URL for HTTPS traffic.
pub fn parse_proxy_server(value: &str) -> Option<String> {
    let value = value.trim();
    let pick = if value.contains('=') {
        let pairs: Vec<(&str, &str)> = value.split(';').filter_map(|p| p.split_once('=')).collect();
        let by = |scheme: &str| pairs.iter().find(|(s, _)| *s == scheme).map(|(_, a)| *a);
        by("https").or_else(|| by("http"))?
    } else {
        value
    };
    let pick = pick.trim();
    if pick.is_empty() {
        return None;
    }
    Some(match pick.contains("://") {
        true => pick.to_owned(),
        false => format!("http://{pick}"),
    })
}

/// The Windows system proxy (Internet Settings, manual proxy only: PAC scripts are not
/// evaluated). Environment variables (`HTTPS_PROXY`...) are handled by the HTTP client.
pub fn system_proxy() -> Option<String> {
    #[cfg(windows)]
    {
        const KEY: &str = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings";
        let enabled = reg_query(KEY, "ProxyEnable")?;
        if u32::from_str_radix(enabled.trim_start_matches("0x"), 16).ok()? == 0 {
            return None;
        }
        parse_proxy_server(&reg_query(KEY, "ProxyServer")?)
    }
    #[cfg(not(windows))]
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reg_output_is_parsed() {
        let out = "\r\nHKEY_CURRENT_USER\\Control Panel\\International\r\n    LocaleName    REG_SZ    fr-FR\r\n";
        assert_eq!(parse_reg_value(out, "LocaleName").as_deref(), Some("fr-FR"));
        assert_eq!(parse_reg_value(out, "Nope"), None);
    }

    #[test]
    fn proxy_syntaxes() {
        assert_eq!(
            parse_proxy_server("proxy:8080").as_deref(),
            Some("http://proxy:8080")
        );
        assert_eq!(
            parse_proxy_server("http=a:1;https=b:2").as_deref(),
            Some("http://b:2")
        );
        assert_eq!(
            parse_proxy_server("http=a:1").as_deref(),
            Some("http://a:1")
        );
        assert_eq!(parse_proxy_server("socks=c:3"), None);
        assert_eq!(parse_proxy_server(""), None);
    }
}
