//! Child processes that must not outlive whoever started them, with what they started in turn:
//! a CLI runtime and the `mcp` servers it launched, the hipfire daemon and its GPU process, a
//! pipeline worker and its inference server. Shared by `surgeon` and `surgeon-gui`.

use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// How long `taskkill` / `kill` may take before it is given up on.
const KILL_TIMEOUT: Duration = Duration::from_secs(5);

/// Kills `child` and every process it started, then reaps it. On Windows the tree is the
/// parent chain (`taskkill /T`); on Unix it is the child's process group, so start the child
/// with `process_group(0)`. Only a live child is looked up by id: the id of a dead one may
/// already belong to another process. The child itself is killed whatever happens; the error
/// says the rest of the tree may have survived.
pub fn kill_tree(child: &mut Child) -> Result<(), String> {
    let mut tree = Ok(());
    if matches!(child.try_wait(), Ok(None)) {
        let pid = child.id().to_string();
        let killer = match cfg!(windows) {
            true => quiet("taskkill", &["/PID", &pid, "/T", "/F"]),
            false => quiet("kill", &["-KILL", "--", &format!("-{pid}")]),
        };
        tree = run(killer, KILL_TIMEOUT);
    }
    let _ = child.kill();
    let _ = child.wait();
    tree
}

pub fn decode_windows_output(bytes: &[u8]) -> String {
    let utf16 = |body: &[u8]| {
        let units: Vec<u16> = body
            .chunks_exact(2)
            .map(|p| u16::from_le_bytes([p[0], p[1]]))
            .collect();
        String::from_utf16_lossy(&units)
    };
    let zero_high = bytes.iter().skip(1).step_by(2).filter(|b| **b == 0).count();
    let text = match bytes.strip_prefix(&[0xFF, 0xFE]) {
        Some(body) => utf16(body),
        None if bytes.len() >= 2 && zero_high * 4 >= bytes.len() => utf16(bytes),
        None => String::from_utf8_lossy(bytes).into_owned(),
    };
    text.replace(['\0', '\u{feff}'], "").trim().to_owned()
}

pub fn run_capture(mut cmd: Command, timeout: Duration) -> std::io::Result<(Option<i32>, String)> {
    use std::io::Read;
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let drain = |pipe: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.read_to_end(&mut bytes);
            }
            bytes
        })
    };
    let out = drain(child.stdout.take().map(|p| Box::new(p) as _));
    let err = drain(child.stderr.take().map(|p| Box::new(p) as _));
    let deadline = Instant::now() + timeout;
    let code = loop {
        if let Some(status) = child.try_wait()? {
            break status.code();
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(25));
    };
    let text = [out, err]
        .map(|t| decode_windows_output(&t.join().unwrap_or_default()))
        .into_iter()
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    Ok((code, text))
}

/// `program args` with no console window (a GUI has none) and no output.
pub(crate) fn quiet(program: &str, args: &[&str]) -> Command {
    let mut cmd = Command::new(program);
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

/// Runs `cmd` to its end, killed past `timeout`. Its exit code is not an error: `taskkill`
/// reports the members of the tree that ended on their own.
fn run(mut cmd: Command, timeout: Duration) -> Result<(), String> {
    let mut killer = cmd.spawn().map_err(|e| format!("{cmd:?} : {e}"))?;
    let deadline = Instant::now() + timeout;
    loop {
        match killer.try_wait() {
            Ok(Some(_)) => return Ok(()),
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(20)),
            Ok(None) => {
                let _ = killer.kill();
                let _ = killer.wait();
                return Err(format!(
                    "{cmd:?} : pas de réponse en {} s",
                    timeout.as_secs()
                ));
            }
            Err(e) => return Err(format!("{cmd:?} : {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use sysinfo::{Pid, ProcessStatus, ProcessesToUpdate, System};

    use super::*;

    /// A shell that keeps a child of its own, in a process group of its own on Unix.
    fn parent_of_a_sleeper() -> Child {
        let (shell, flag, script) = match cfg!(windows) {
            true => ("cmd", "/C", "ping -n 60 127.0.0.1 >NUL"),
            false => ("sh", "-c", "sleep 60; true"),
        };
        let mut cmd = quiet(shell, &[flag, script]);
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        cmd.spawn().unwrap()
    }

    fn alive(pid: u32) -> bool {
        let pid = Pid::from_u32(pid);
        let mut system = System::new().unwrap();
        system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        system
            .process(pid)
            .is_some_and(|p| p.status() != ProcessStatus::Zombie)
    }

    fn child_of(parent: u32) -> Option<u32> {
        let mut system = System::new().unwrap();
        system.refresh_processes(ProcessesToUpdate::All, true);
        let parent = Pid::from_u32(parent);
        system
            .processes()
            .iter()
            .find(|(_, p)| p.parent() == Some(parent))
            .map(|(pid, _)| pid.as_u32())
    }

    fn eventually(mut done: impl FnMut() -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if done() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        false
    }

    fn utf16le(text: &str) -> Vec<u8> {
        text.encode_utf16().flat_map(u16::to_le_bytes).collect()
    }

    #[test]
    fn windows_output_is_read_as_utf8_or_utf16le() {
        let message = "L'opération a échoué : HCS_E_SERVICE_NOT_AVAILABLE";
        assert_eq!(
            decode_windows_output(format!("  {message}\r\n").as_bytes()),
            message
        );
        let bom = [&[0xFF, 0xFE][..], &utf16le(&format!("{message}\r\n"))].concat();
        assert_eq!(decode_windows_output(&bom), message);
        assert_eq!(decode_windows_output(&utf16le(message)), message);
        assert_eq!(decode_windows_output(b"a\0b\0"), "ab");
        assert_eq!(decode_windows_output(b"\xEF\xBB\xBFok\n"), "ok");
        assert_eq!(decode_windows_output(b""), "");
    }

    #[test]
    fn the_whole_tree_goes_and_a_dead_child_is_only_reaped() {
        let mut parent = parent_of_a_sleeper();
        let id = parent.id();
        let mut grandchild = None;
        assert!(
            eventually(|| {
                grandchild = child_of(id);
                grandchild.is_some()
            }),
            "the shell started its child"
        );
        kill_tree(&mut parent).unwrap();
        assert!(!alive(id));
        let grandchild = grandchild.unwrap();
        assert!(eventually(|| !alive(grandchild)), "no orphan left");
        // Already reaped: nothing is looked up by its id any more.
        assert_eq!(kill_tree(&mut parent), Ok(()));
    }
}
