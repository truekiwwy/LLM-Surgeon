//! The hipfire daemon that lives with the window: started when the app opens on a machine that
//! calls for it and nothing answers yet, stopped when it closes so that the GPU memory comes
//! back. A daemon the app did not start is used as is and never touched.
//!
//! `hipfire serve` is the HTTP front and owns a separate GPU process (hipfire's docs/SERVE.md,
//! read 2026-11-18): stopping it means stopping the whole process tree.

use std::fs::File;
use std::io;
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::task::Task;
use crate::{logging, paths, platform};

const READY_TIMEOUT: Duration = Duration::from_secs(5);
const READY_POLL: Duration = Duration::from_millis(200);
const KILL_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Default)]
pub struct HipfireDaemon {
    /// Off for captures, tests and the demo workspace: only the real window starts a daemon.
    pub enabled: bool,
    /// The process this app started; `None` when the daemon, if any, is someone else's.
    child: Arc<Mutex<Option<Child>>>,
    waiting: Option<Task<bool>>,
    /// What was last started: a daemon that dies is not restarted in a loop.
    tried: Option<(PathBuf, u16)>,
}

impl HipfireDaemon {
    /// Starts `bin` on `port` without waiting for it; `wake` runs when the wait ends. Does
    /// nothing when these settings were already tried, and replaces a daemon of other settings.
    pub fn start(&mut self, bin: &Path, port: u16, wake: impl Fn() + Send + 'static) {
        let wanted = (bin.to_path_buf(), port);
        if !self.enabled || self.tried.as_ref() == Some(&wanted) {
            return;
        }
        self.stop();
        self.tried = Some(wanted);
        if let Err(e) = self.launch(serve_command(bin, port), port, wake) {
            logging::warn(&format!("hipfire serve ({}) : {e}", bin.display()));
        }
    }

    fn launch(
        &mut self,
        mut cmd: Command,
        port: u16,
        wake: impl Fn() + Send + 'static,
    ) -> io::Result<()> {
        *lock(&self.child) = Some(cmd.spawn()?);
        let child = Arc::clone(&self.child);
        self.waiting = Some(Task::spawn(wake, move |_| wait_ready(&child, port)));
        Ok(())
    }

    /// `true` once, when the daemon started here opened its port: time to probe again.
    pub fn poll(&mut self) -> bool {
        let Some(ready) = self.waiting.as_ref().and_then(Task::poll) else {
            return false;
        };
        self.waiting = None;
        if !ready {
            logging::warn("hipfire serve : le port ne répond pas, moteur de repli conservé");
        }
        ready
    }

    /// Stops the daemon this app started, if any, and frees its GPU memory.
    pub fn stop(&mut self) {
        self.waiting = None;
        self.tried = None;
        let child = lock(&self.child).take();
        if let Some(mut child) = child {
            kill_tree(&mut child);
        }
    }
}

impl Drop for HipfireDaemon {
    fn drop(&mut self) {
        self.stop();
    }
}

fn lock(child: &Mutex<Option<Child>>) -> MutexGuard<'_, Option<Child>> {
    child.lock().unwrap_or_else(|e| e.into_inner())
}

/// Bound to `127.0.0.1` whatever the host of `hipfire_url`: the engine connects there. No
/// pre-warm: the GPU memory stays free until a request names its model.
fn serve_args(port: u16) -> [String; 3] {
    [
        "serve".into(),
        format!("127.0.0.1:{port}"),
        "--no-prewarm".into(),
    ]
}

/// No console, no pipe left unread, and on Unix a process group of its own to stop as one.
fn background(cmd: &mut Command, out: Stdio, err: Stdio) {
    cmd.stdin(Stdio::null()).stdout(out).stderr(err);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
}

/// `hipfire.log`, rewritten at each start; nowhere when it cannot be opened.
fn outputs() -> (Stdio, Stdio) {
    let dir = paths::logs_dir();
    let opened = std::fs::create_dir_all(&dir)
        .and_then(|()| File::create(dir.join("hipfire.log")))
        .and_then(|out| Ok((out.try_clone()?, out)));
    match opened {
        Ok((out, err)) => (out.into(), err.into()),
        Err(_) => (Stdio::null(), Stdio::null()),
    }
}

fn serve_command(bin: &Path, port: u16) -> Command {
    let mut cmd = platform::command(bin);
    cmd.args(serve_args(port));
    let (out, err) = outputs();
    background(&mut cmd, out, err);
    cmd
}

/// Whether `port` opens while the child lives. A child that exits is forgotten.
fn wait_ready(child: &Mutex<Option<Child>>, port: u16) -> bool {
    let address = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let deadline = Instant::now() + READY_TIMEOUT;
    while Instant::now() < deadline {
        if TcpStream::connect_timeout(&address, READY_POLL).is_ok() {
            return true;
        }
        {
            let mut slot = lock(child);
            match slot.as_mut().map(Child::try_wait) {
                Some(Ok(None)) => {}
                Some(_) => {
                    *slot = None;
                    return false;
                }
                None => return false,
            }
        }
        std::thread::sleep(READY_POLL);
    }
    false
}

/// The child and what it spawned. Only a live child is looked up by id: the id of a dead one
/// may already belong to another process.
fn kill_tree(child: &mut Child) {
    if matches!(child.try_wait(), Ok(None)) {
        let pid = child.id().to_string();
        let (program, args) = match cfg!(windows) {
            true => (
                "taskkill",
                vec!["/PID".to_owned(), pid, "/T".into(), "/F".into()],
            ),
            false => (
                "kill",
                vec!["-KILL".to_owned(), "--".into(), format!("-{pid}")],
            ),
        };
        let mut killer = platform::command(program);
        killer.args(args);
        if let Err(e) = platform::run_capture(killer, KILL_TIMEOUT) {
            logging::warn(&format!("arrêt de hipfire : {e}"));
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use sysinfo::{Pid, ProcessStatus, ProcessesToUpdate, System};
    use weight_surgeon::backend::hipfire::loopback_port;

    use super::*;

    /// A shell that keeps a child of its own, like `hipfire serve` and its GPU process.
    fn sleeper() -> Command {
        let (shell, flag, script) = match cfg!(windows) {
            true => ("cmd", "/C", "ping -n 60 127.0.0.1 >NUL"),
            false => ("sh", "-c", "sleep 60; true"),
        };
        let mut cmd = platform::command(shell);
        cmd.args([flag, script]);
        background(&mut cmd, Stdio::null(), Stdio::null());
        cmd
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

    fn pid_of(daemon: &HipfireDaemon) -> u32 {
        lock(&daemon.child).as_ref().unwrap().id()
    }

    #[test]
    fn the_port_of_the_url_reaches_a_loopback_bind_without_pre_warm() {
        for url in ["http://127.0.0.1:4321", "http://localhost:4321/"] {
            assert_eq!(
                serve_args(loopback_port(url).unwrap()),
                ["serve", "127.0.0.1:4321", "--no-prewarm"],
                "{url}"
            );
        }
        assert!(loopback_port("http://10.0.0.2:4321").is_err());
    }

    #[test]
    fn a_started_daemon_is_seen_on_its_port_and_stops_with_its_children() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let mut daemon = HipfireDaemon::default();
        daemon.launch(sleeper(), port, || {}).unwrap();
        assert!(eventually(|| daemon.poll()), "the port answers");
        assert!(!daemon.poll(), "reported once");

        let parent = pid_of(&daemon);
        let mut gpu_process = None;
        assert!(eventually(|| {
            gpu_process = child_of(parent);
            gpu_process.is_some()
        }));
        daemon.stop();
        assert!(lock(&daemon.child).is_none());
        assert!(eventually(|| !alive(parent)), "the daemon");
        assert!(
            eventually(|| !alive(gpu_process.unwrap())),
            "and what it spawned"
        );
    }

    #[test]
    fn a_child_that_exits_is_forgotten_and_ends_the_wait() {
        let closed = TcpListener::bind(("127.0.0.1", 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let (shell, flag) = if cfg!(windows) {
            ("cmd", "/C")
        } else {
            ("sh", "-c")
        };
        let mut cmd = platform::command(shell);
        cmd.args([flag, "exit 3"]);
        background(&mut cmd, Stdio::null(), Stdio::null());
        let child = Mutex::new(Some(cmd.spawn().unwrap()));
        let started = Instant::now();
        assert!(!wait_ready(&child, closed));
        assert!(started.elapsed() < READY_TIMEOUT, "no need to wait it out");
        assert!(lock(&child).is_none());
    }

    #[test]
    fn a_binary_that_cannot_start_leaves_nothing_behind() {
        let mut daemon = HipfireDaemon::default();
        let missing = platform::command("surely-not-a-hipfire-binary");
        assert!(daemon.launch(missing, 1, || {}).is_err());
        assert!(lock(&daemon.child).is_none() && daemon.waiting.is_none());
    }

    #[test]
    fn nothing_starts_when_disabled_or_already_tried() {
        let bin = Path::new("surely-not-a-hipfire-binary");
        let mut daemon = HipfireDaemon::default();
        daemon.start(bin, 1, || {});
        assert!(daemon.tried.is_none(), "a capture or a test starts nothing");

        daemon.enabled = true;
        daemon.launch(sleeper(), 1, || {}).unwrap();
        let running = pid_of(&daemon);
        daemon.tried = Some((bin.to_path_buf(), 1));
        daemon.start(bin, 1, || {});
        assert_eq!(pid_of(&daemon), running, "left alone");
        drop(daemon);
        assert!(eventually(|| !alive(running)), "dropping stops it");
    }

    #[test]
    fn a_process_the_app_did_not_start_is_left_alone() {
        let mut foreign = sleeper().spawn().unwrap();
        let mut daemon = HipfireDaemon::default();
        daemon.stop();
        drop(daemon);
        assert!(matches!(foreign.try_wait(), Ok(None)));
        kill_tree(&mut foreign);
    }
}
