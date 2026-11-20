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

/// `program args` with no console window (a GUI has none) and no output.
fn quiet(program: &str, args: &[&str]) -> Command {
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
