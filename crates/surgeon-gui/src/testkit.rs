//! Test support: a fake GitHub releases server on loopback (release list API and archive
//! download with `Range`, with injectable faults) and an in-memory zip builder.

use std::io::{Cursor, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use sha2::{Digest, Sha256};
use zip::write::SimpleFileOptions;

pub fn archive(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    for (name, data) in entries {
        // Stored, so that a big payload stays big on the wire (faults are cut by byte count).
        let options =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        writer.start_file(*name, options).unwrap();
        writer.write_all(data).unwrap();
    }
    writer.finish().unwrap().into_inner()
}

#[derive(Default)]
struct State {
    name: String,
    bytes: Vec<u8>,
    publish_digest: bool,
    corrupt_digest: bool,
    cut_first: Option<usize>,
    archive_hits: usize,
    saw_range: bool,
    bytes_sent: usize,
}

pub struct FakeReleases {
    port: u16,
    state: Arc<Mutex<State>>,
    stop: Arc<AtomicBool>,
}

impl FakeReleases {
    /// Serves one release `b7` carrying `name` (after an empty `v0.9.0` like the real one).
    pub fn serve(name: &str, bytes: Vec<u8>, publish_digest: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let state = Arc::new(Mutex::new(State {
            name: name.into(),
            bytes,
            publish_digest,
            ..Default::default()
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let (s, st) = (state.clone(), stop.clone());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                if st.load(Ordering::Relaxed) {
                    return;
                }
                if let Ok(stream) = stream {
                    handle(stream, port, &s);
                }
            }
        });
        Self { port, state, stop }
    }

    pub fn api_url(&self) -> String {
        format!("http://127.0.0.1:{}/releases", self.port)
    }

    pub fn corrupt_digest(&self) {
        self.state.lock().unwrap().corrupt_digest = true;
    }

    pub fn cut_first_transfer_after(&self, bytes: usize) {
        self.state.lock().unwrap().cut_first = Some(bytes);
    }

    pub fn archive_hits(&self) -> usize {
        self.state.lock().unwrap().archive_hits
    }

    pub fn saw_range_request(&self) -> bool {
        self.state.lock().unwrap().saw_range
    }

    pub fn bytes_sent(&self) -> usize {
        self.state.lock().unwrap().bytes_sent
    }

    pub fn archive_len(&self) -> usize {
        self.state.lock().unwrap().bytes.len()
    }
}

impl Drop for FakeReleases {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = TcpStream::connect(("127.0.0.1", self.port));
    }
}

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn handle(mut stream: TcpStream, port: u16, state: &Mutex<State>) {
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        match stream.read(&mut byte) {
            Ok(1) => head.push(byte[0]),
            _ => return,
        }
    }
    let head = String::from_utf8_lossy(&head).into_owned();
    let path = head.split_whitespace().nth(1).unwrap_or("/").to_owned();
    let range_from: Option<usize> = head
        .lines()
        .find_map(|l| {
            l.to_ascii_lowercase()
                .strip_prefix("range: bytes=")
                .map(str::to_owned)
        })
        .and_then(|r| r.trim_end_matches('-').parse().ok());
    let mut st = state.lock().unwrap();
    if path.starts_with("/releases") {
        let digest = match (st.publish_digest, st.corrupt_digest) {
            (false, _) => String::new(),
            (true, true) => format!(",\"digest\":\"sha256:{}\"", "0".repeat(64)),
            (true, false) => format!(",\"digest\":\"sha256:{}\"", hex(&st.bytes)),
        };
        let body = format!(
            "[{{\"tag_name\":\"v0.9.0\",\"assets\":[{{\"name\":\"nightly-tag.txt\",\"size\":7,\"browser_download_url\":\"http://127.0.0.1:{port}/a/nightly-tag.txt\"}}]}},\
             {{\"tag_name\":\"b7\",\"assets\":[{{\"name\":\"{n}\",\"size\":{len},\"browser_download_url\":\"http://127.0.0.1:{port}/a/{n}\"{digest}}}]}}]",
            n = st.name,
            len = st.bytes.len()
        );
        let _ = write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
    } else if path == format!("/a/{}", st.name) {
        st.archive_hits += 1;
        let total = st.bytes.len();
        let start = range_from.unwrap_or(0).min(total);
        if range_from.is_some() {
            st.saw_range = true;
        }
        let body = st.bytes[start..].to_vec();
        let (status, extra) = match range_from {
            Some(_) => (
                "206 Partial Content",
                format!("Content-Range: bytes {start}-{}/{total}\r\n", total - 1),
            ),
            None => ("200 OK", String::new()),
        };
        let _ = write!(
            stream,
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{extra}Connection: close\r\n\r\n",
            body.len()
        );
        let cut = st.cut_first.take().filter(|_| range_from.is_none());
        let sent = cut.map_or(body.len(), |n| n.min(body.len()));
        let _ = stream.write_all(&body[..sent]);
        st.bytes_sent += sent;
    } else {
        let _ = write!(
            stream,
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
    }
    let _ = stream.flush();
    let _ = stream.shutdown(Shutdown::Both);
}
