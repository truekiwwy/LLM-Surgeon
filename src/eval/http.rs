//! Minimal HTTP/1.1 client for a local `llama-server`: `std::net::TcpStream` only, loopback
//! only, no TLS, no proxy. One request per connection (`Connection: close`); responses
//! with `Content-Length`, `Transfer-Encoding: chunked` or read until close; server-sent
//! events read line by line. Every read has a timeout; dropping the stream (closing the
//! connection) is how a generation is cancelled, llama-server stopping the slot.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpStream};
use std::time::Duration;

/// What can go wrong talking to the server.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HttpError {
    #[error("connexion à {addr} impossible : {message}")]
    Connect { addr: SocketAddr, message: String },
    #[error("délai dépassé en attendant le serveur")]
    Timeout,
    #[error("réponse HTTP invalide : {0}")]
    Protocol(String),
    #[error("le serveur a répondu {status} : {body}")]
    Status { status: u16, body: String },
    #[error("E/S : {0}")]
    Io(String),
}

pub type Result<T> = std::result::Result<T, HttpError>;

fn io(e: std::io::Error) -> HttpError {
    match e.kind() {
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => HttpError::Timeout,
        _ => HttpError::Io(e.to_string()),
    }
}

/// The body of a response, read as it comes.
pub struct Body {
    reader: BufReader<TcpStream>,
    /// `Some(n)`: bytes left of the current chunk (chunked) or of the body (length).
    left: Option<usize>,
    chunked: bool,
    done: bool,
}

impl Body {
    /// The next bytes of the body, `None` at its end.
    pub fn next_bytes(&mut self) -> Result<Option<Vec<u8>>> {
        if self.done {
            return Ok(None);
        }
        if self.chunked && self.left == Some(0) {
            let mut size = String::new();
            self.reader.read_line(&mut size).map_err(io)?;
            let size = size.trim();
            let size = size.split(';').next().unwrap_or_default();
            let n = usize::from_str_radix(size, 16)
                .map_err(|_| HttpError::Protocol(format!("taille de bloc « {size} »")))?;
            if n == 0 {
                self.done = true;
                return Ok(None);
            }
            self.left = Some(n);
        }
        let want = self.left.map_or(8192, |n| n.min(8192));
        let mut buf = vec![0u8; want];
        let got = self.reader.read(&mut buf).map_err(io)?;
        if got == 0 {
            self.done = true;
            return match self.left {
                Some(n) if n > 0 => Err(HttpError::Protocol(
                    "connexion fermée en cours de corps".into(),
                )),
                _ => Ok(None),
            };
        }
        buf.truncate(got);
        if let Some(n) = self.left.as_mut() {
            *n -= got;
            if *n == 0 {
                match self.chunked {
                    true => {
                        // The CRLF that ends a chunk.
                        let mut crlf = [0u8; 2];
                        self.reader.read_exact(&mut crlf).map_err(io)?;
                    }
                    false => self.done = true,
                }
            }
        }
        Ok(Some(buf))
    }

    /// The whole body as text.
    pub fn text(mut self) -> Result<String> {
        let mut all = Vec::new();
        while let Some(bytes) = self.next_bytes()? {
            all.extend(bytes);
        }
        String::from_utf8(all).map_err(|e| HttpError::Protocol(e.to_string()))
    }

    /// Server-sent events: the payload of each `data:` line, in order, until the body ends
    /// or `on_event` returns false (the connection is then dropped by the caller).
    pub fn events(mut self, on_event: &mut dyn FnMut(&str) -> Result<bool>) -> Result<()> {
        let mut pending = Vec::new();
        while let Some(bytes) = self.next_bytes()? {
            pending.extend(bytes);
            while let Some(at) = pending.iter().position(|b| *b == b'\n') {
                let line: Vec<u8> = pending.drain(..=at).collect();
                let line = String::from_utf8_lossy(&line);
                let line = line.trim_end_matches(['\r', '\n']);
                if let Some(data) = line.strip_prefix("data:") {
                    if !on_event(data.trim_start())? {
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }
}

/// One request to `127.0.0.1:port`: status and the body to read. The status line and
/// headers are read here; a status of 400 or more is an error with the body.
pub fn request(
    port: u16,
    method: &str,
    path: &str,
    api_key: Option<&str>,
    json: Option<&str>,
    timeout: Duration,
) -> Result<(u16, Body)> {
    let addr = SocketAddr::new(IpAddr::from([127, 0, 0, 1]), port);
    let stream = TcpStream::connect_timeout(&addr, timeout).map_err(|e| HttpError::Connect {
        addr,
        message: e.to_string(),
    })?;
    stream.set_read_timeout(Some(timeout)).map_err(io)?;
    stream.set_write_timeout(Some(timeout)).map_err(io)?;
    let mut head =
        format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n");
    if let Some(key) = api_key {
        head.push_str(&format!("Authorization: Bearer {key}\r\n"));
    }
    let body = json.unwrap_or("");
    if json.is_some() {
        head.push_str(&format!(
            "Content-Type: application/json\r\nContent-Length: {}\r\n",
            body.len()
        ));
    }
    head.push_str("\r\n");
    let mut writer = stream.try_clone().map_err(io)?;
    writer.write_all(head.as_bytes()).map_err(io)?;
    writer.write_all(body.as_bytes()).map_err(io)?;
    writer.flush().map_err(io)?;

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line).map_err(io)?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| HttpError::Protocol(format!("ligne d'état « {} »", status_line.trim())))?;
    let (mut length, mut chunked) = (None, false);
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).map_err(io)? == 0 {
            return Err(HttpError::Protocol("en-têtes tronqués".into()));
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        let Some((name, value)) = line.split_once(':') else {
            return Err(HttpError::Protocol(format!("en-tête « {line} »")));
        };
        match name.trim().to_ascii_lowercase().as_str() {
            "content-length" => {
                length = Some(value.trim().parse().map_err(|_| {
                    HttpError::Protocol(format!("Content-Length « {} »", value.trim()))
                })?)
            }
            "transfer-encoding" => chunked = value.to_ascii_lowercase().contains("chunked"),
            _ => {}
        }
    }
    let body = Body {
        reader,
        left: match chunked {
            true => Some(0),
            false => length,
        },
        chunked,
        done: !chunked && length == Some(0),
    };
    if status >= 400 {
        let text = body.text().unwrap_or_default();
        return Err(HttpError::Status {
            status,
            body: crate::eval::truncate_chars(&text, 500).to_owned(),
        });
    }
    Ok((status, body))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    /// A one-connection server: reads the request head and body, then writes `pieces`
    /// with a pause between them, then closes. Returns the port and the request it got.
    fn serve(pieces: Vec<&'static str>, pause: Duration) -> (u16, thread::JoinHandle<String>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut head = String::new();
            let mut length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if let Some(v) = line.strip_prefix("Content-Length: ") {
                    length = v.trim().parse().unwrap();
                }
                head.push_str(&line);
                if line == "\r\n" {
                    break;
                }
            }
            let mut body = vec![0u8; length];
            reader.read_exact(&mut body).unwrap();
            let mut writer = stream;
            for piece in pieces {
                if writer.write_all(piece.as_bytes()).is_err() {
                    break;
                }
                let _ = writer.flush();
                thread::sleep(pause);
            }
            head + &String::from_utf8(body).unwrap()
        });
        (port, handle)
    }

    const SECOND: Duration = Duration::from_secs(1);

    #[test]
    fn content_length_body_and_the_request_sent() {
        let (port, server) = serve(
            vec![
                "HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\nhello",
                " world",
            ],
            Duration::from_millis(20),
        );
        let (status, body) =
            request(port, "POST", "/completion", Some("k3y"), Some("{}"), SECOND).unwrap();
        assert_eq!(
            (status, body.text().unwrap().as_str()),
            (200, "hello world")
        );
        let sent = server.join().unwrap();
        assert!(sent.starts_with("POST /completion HTTP/1.1\r\n"), "{sent}");
        for header in [
            format!("Host: 127.0.0.1:{port}\r\n"),
            "Connection: close\r\n".into(),
            "Authorization: Bearer k3y\r\n".into(),
            "Content-Length: 2\r\n".into(),
        ] {
            assert!(sent.contains(&header), "{header} in {sent}");
        }
        assert!(sent.ends_with("\r\n\r\n{}"));
    }

    #[test]
    fn chunked_sse_split_anywhere() {
        // Chunk sizes in hex with an extension; an event and even a line cut across chunks.
        let (port, _server) = serve(
            vec![
                "HTTP/1.1 200 OK\r\ntransfer-encoding: Chunked\r\n\r\n",
                "c;ext=1\r\ndata: {\"a\":1\r\n",
                "6\r\n}\n\ndat\r\n",
                "16\r\na: {\"b\":2}\n\n: comment\n\r\n",
                "f\r\ndata: {\"c\":3}\n\n\r\n",
                // An unfinished last line is not an event (SSE), and is dropped.
                "8\r\ndata: 4\r\r\n",
                "0\r\n\r\n",
            ],
            Duration::from_millis(5),
        );
        let (_, body) = request(port, "GET", "/", None, None, SECOND).unwrap();
        let mut events = Vec::new();
        body.events(&mut |data| {
            events.push(data.to_owned());
            Ok(true)
        })
        .unwrap();
        assert_eq!(events, [r#"{"a":1}"#, r#"{"b":2}"#, r#"{"c":3}"#]);
    }

    #[test]
    fn events_stop_when_asked_and_bodies_end_on_close() {
        let (port, _server) = serve(
            vec!["HTTP/1.1 200 OK\r\n\r\ndata: 1\n\ndata: 2\n\ndata: 3\n\n"],
            Duration::ZERO,
        );
        let (_, body) = request(port, "GET", "/", None, None, SECOND).unwrap();
        let mut seen = Vec::new();
        body.events(&mut |data| {
            seen.push(data.to_owned());
            Ok(seen.len() < 2)
        })
        .unwrap();
        assert_eq!(seen, ["1", "2"]);
        // No length, no chunks: the body is what comes until the server closes.
        let (port, _server) = serve(vec!["HTTP/1.1 200 OK\r\n\r\nuntil close"], Duration::ZERO);
        let (_, body) = request(port, "GET", "/", None, None, SECOND).unwrap();
        assert_eq!(body.text().unwrap(), "until close");
    }

    #[test]
    fn errors_are_typed() {
        let (port, _server) = serve(
            vec!["HTTP/1.1 401 Unauthorized\r\nContent-Length: 15\r\n\r\nInvalid API Key"],
            Duration::ZERO,
        );
        assert_eq!(
            request(port, "GET", "/props", Some("wrong"), None, SECOND).err(),
            Some(HttpError::Status {
                status: 401,
                body: "Invalid API Key".into()
            })
        );
        let (port, _server) = serve(vec!["SSH-2.0-OpenSSH\r\n\r\n"], Duration::ZERO);
        assert!(matches!(
            request(port, "GET", "/", None, None, SECOND),
            Err(HttpError::Protocol(_))
        ));
        // A chunk cut by the server dying: a clean close, or a reset (Windows under load).
        let (port, _server) = serve(
            vec!["HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n10\r\nonly six"],
            Duration::ZERO,
        );
        let (_, body) = request(port, "GET", "/", None, None, SECOND).unwrap();
        assert!(matches!(
            body.text(),
            Err(HttpError::Protocol(_) | HttpError::Io(_))
        ));
        // Nothing listens there any more.
        let closed = TcpListener::bind(("127.0.0.1", 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        assert!(matches!(
            request(closed, "GET", "/", None, None, SECOND),
            Err(HttpError::Connect { .. })
        ));
    }

    #[test]
    fn a_silent_server_times_out() {
        let (port, _server) = serve(
            vec!["HTTP/1.1 200 OK\r\n\r\n", "late"],
            Duration::from_millis(1500),
        );
        let (_, body) = request(port, "GET", "/", None, None, Duration::from_millis(200)).unwrap();
        assert_eq!(body.text(), Err(HttpError::Timeout));
    }
}
