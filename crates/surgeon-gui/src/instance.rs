//! Single instance: a second launch hands its arguments to the first one and exits.
//!
//! The first instance listens on a loopback TCP port and writes `{port, token}` to a file in
//! the data folder. A second one reads it and sends the paths with the token; a leftover
//! file whose port nobody answers is stale and replaced. The token keeps another local
//! program from making the app open paths. Two instances started in the same millisecond
//! can both become primary: the cost is a second window, never data loss.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use serde::{Deserialize, Serialize};

const TIMEOUT: Duration = Duration::from_millis(1500);
/// A message is a few paths; anything bigger is not ours.
const MAX_MESSAGE: u64 = 64 * 1024;

#[derive(Serialize, Deserialize)]
struct Lock {
    port: u16,
    token: String,
}

#[derive(Serialize, Deserialize)]
struct Message {
    token: String,
    paths: Vec<PathBuf>,
}

/// Paths other launches asked this instance to open.
pub struct Inbox {
    rx: Receiver<Vec<PathBuf>>,
}

impl Inbox {
    pub fn drain(&self) -> Vec<PathBuf> {
        self.rx.try_iter().flatten().collect()
    }
}

pub enum Role {
    Primary(Inbox),
    /// The first instance took the paths; this process should exit.
    Forwarded,
    /// Neither role could be taken (no loopback, unwritable data folder): run alone.
    Alone,
}

fn token() -> String {
    // RandomState is seeded by the OS: unpredictable without a rand crate.
    let mut out = String::new();
    for _ in 0..2 {
        out.push_str(&format!(
            "{:016x}",
            RandomState::new().build_hasher().finish()
        ));
    }
    out
}

fn try_forward(lock_file: &Path, paths: &[PathBuf]) -> bool {
    let Ok(bytes) = std::fs::read(lock_file) else {
        return false;
    };
    let Ok(lock) = serde_json::from_slice::<Lock>(&bytes) else {
        return false;
    };
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, lock.port));
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, TIMEOUT) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(TIMEOUT));
    let message = Message {
        token: lock.token,
        paths: paths.to_vec(),
    };
    let Ok(mut line) = serde_json::to_vec(&message) else {
        return false;
    };
    line.push(b'\n');
    if stream.write_all(&line).is_err() {
        return false;
    }
    let mut reply = String::new();
    BufReader::new(stream).read_line(&mut reply).is_ok() && reply.trim() == "ok"
}

fn serve(listener: TcpListener, token: String, tx: mpsc::Sender<Vec<PathBuf>>) {
    for stream in listener.incoming().flatten() {
        let _ = stream.set_read_timeout(Some(TIMEOUT));
        let mut line = String::new();
        let mut reader = BufReader::new((&stream).take(MAX_MESSAGE));
        if reader.read_line(&mut line).is_err() {
            continue;
        }
        let Ok(message) = serde_json::from_str::<Message>(&line) else {
            continue;
        };
        if message.token != token {
            continue;
        }
        let _ = tx.send(message.paths);
        let _ = (&stream).write_all(b"ok\n");
    }
}

/// Becomes the primary instance, or forwards `paths` to the running one.
pub fn acquire(lock_file: &Path, paths: &[PathBuf]) -> Role {
    if try_forward(lock_file, paths) {
        return Role::Forwarded;
    }
    let Ok(listener) = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)) else {
        return Role::Alone;
    };
    let Ok(port) = listener.local_addr().map(|a| a.port()) else {
        return Role::Alone;
    };
    let token = token();
    let lock = serde_json::to_vec(&Lock {
        port,
        token: token.clone(),
    })
    .unwrap_or_default();
    let written = lock_file
        .parent()
        .is_none_or(|d| std::fs::create_dir_all(d).is_ok())
        && std::fs::write(lock_file, lock).is_ok();
    if !written {
        return Role::Alone;
    }
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || serve(listener, token, tx));
    Role::Primary(Inbox { rx })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn wait_for(inbox: &Inbox) -> Vec<PathBuf> {
        let start = Instant::now();
        loop {
            let got = inbox.drain();
            if !got.is_empty() || start.elapsed() > Duration::from_secs(5) {
                return got;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn the_second_launch_hands_its_arguments_to_the_first() {
        let dir = tempfile::tempdir().unwrap();
        let lock = dir.path().join("instance.json");
        let Role::Primary(inbox) = acquire(&lock, &[]) else {
            panic!("the first launch is the primary");
        };
        let paths = vec![PathBuf::from("C:/données avec espaces/graph.json")];
        assert!(matches!(acquire(&lock, &paths), Role::Forwarded));
        assert_eq!(wait_for(&inbox), paths);
    }

    #[test]
    fn a_stale_lock_is_replaced_and_a_bad_token_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let lock = dir.path().join("instance.json");
        let dead = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = dead.local_addr().unwrap().port();
        drop(dead);
        std::fs::write(&lock, format!("{{\"port\":{port},\"token\":\"x\"}}")).unwrap();
        let Role::Primary(inbox) = acquire(&lock, &[]) else {
            panic!("nobody answers on the stale port");
        };
        // A forged message with the wrong token is dropped without reply.
        let current: Lock = serde_json::from_slice(&std::fs::read(&lock).unwrap()).unwrap();
        let mut stream = TcpStream::connect((Ipv4Addr::LOCALHOST, current.port)).unwrap();
        stream
            .write_all(b"{\"token\":\"wrong\",\"paths\":[\"/evil\"]}\n")
            .unwrap();
        stream
            .set_read_timeout(Some(Duration::from_millis(400)))
            .unwrap();
        let mut reply = String::new();
        assert!(BufReader::new(stream).read_line(&mut reply).is_err() || reply.is_empty());
        assert!(inbox.drain().is_empty());
    }
}
