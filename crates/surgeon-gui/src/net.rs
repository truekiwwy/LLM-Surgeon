//! The only network code of the desktop app. Nothing is fetched without the caller having
//! shown the user the URL, the size and the hash first; this module enforces what a caller
//! can still get wrong: a domain allow-list on every hop, HTTPS only, resumable downloads,
//! cancellation, and a size cap on API answers. No telemetry, no cookies, no credentials.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use sha2::{Digest, Sha256};

/// Hosts the app may talk to. A leading dot allows the domain and its subdomains
/// (`release-assets.githubusercontent.com`, Hugging Face's `cas-bridge.xethub.hf.co`).
pub const ALLOWED_HOSTS: [&str; 6] = [
    "api.github.com",
    "github.com",
    ".githubusercontent.com",
    "huggingface.co",
    ".huggingface.co",
    ".hf.co",
];

const MAX_REDIRECTS: usize = 5;
/// API answers (release lists, model metadata) are far below this.
pub const MAX_API_BYTES: u64 = 16 << 20;

#[derive(Debug)]
pub enum NetError {
    /// The URL is not HTTPS, or its host is not on the allow-list.
    Blocked(String),
    /// DNS, connection, TLS, proxy or timeout: the machine is offline or filtered.
    Unreachable(String),
    Http(u16),
    /// The connection dropped; `bytes` are kept on disk for a resume.
    Interrupted(u64),
    Cancelled,
    Io(String),
    TooLarge,
}

impl std::fmt::Display for NetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Blocked(url) => write!(
                f,
                "adresse refusée (HTTPS et domaines connus uniquement) : {url}"
            ),
            Self::Unreachable(e) => write!(f, "réseau injoignable : {e}"),
            Self::Http(code) => write!(f, "le serveur a répondu HTTP {code}"),
            Self::Interrupted(n) => write!(
                f,
                "connexion interrompue après {n} octets (reprise possible)"
            ),
            Self::Cancelled => f.write_str("téléchargement annulé"),
            Self::Io(e) => f.write_str(e),
            Self::TooLarge => f.write_str("réponse trop volumineuse"),
        }
    }
}

impl std::error::Error for NetError {}

/// Which URLs may be requested.
#[derive(Debug, Clone)]
pub struct Policy {
    hosts: Vec<String>,
    /// Tests only: plain `http://` to `127.0.0.1` (the fake release server).
    loopback_http: bool,
}

impl Policy {
    pub fn production() -> Self {
        Self {
            hosts: ALLOWED_HOSTS.iter().map(|h| (*h).to_owned()).collect(),
            loopback_http: false,
        }
    }

    /// The fake server of the tests: loopback over plain HTTP, nothing else.
    pub fn loopback() -> Self {
        Self {
            hosts: vec!["127.0.0.1".into()],
            loopback_http: true,
        }
    }

    pub fn allows(&self, url: &str) -> bool {
        let Some((scheme, host)) = scheme_and_host(url) else {
            return false;
        };
        let scheme_ok = scheme == "https" || (self.loopback_http && scheme == "http");
        scheme_ok
            && self
                .hosts
                .iter()
                .any(|allowed| match allowed.strip_prefix('.') {
                    Some(domain) => host == domain || host.ends_with(allowed.as_str()),
                    None => host == *allowed,
                })
    }
}

/// `https://user@host:443/path` -> (`https`, `host`); credentials in the URL are refused.
fn scheme_and_host(url: &str) -> Option<(&str, &str)> {
    let (scheme, rest) = url.split_once("://")?;
    let authority = rest.split(['/', '?', '#']).next()?;
    if authority.contains('@') {
        return None;
    }
    let host = authority.rsplit_once(':').map_or(authority, |(h, port)| {
        match port.bytes().all(|b| b.is_ascii_digit()) {
            true => h,
            false => authority,
        }
    });
    (!host.is_empty()).then_some((scheme, host))
}

pub struct Fetcher {
    agent: ureq::Agent,
    policy: Policy,
}

/// An open response: status, announced length and the body to read.
pub struct Opened {
    pub status: u16,
    pub content_length: Option<u64>,
    pub content_range_start: Option<u64>,
    pub body: Box<dyn Read + Send>,
}

impl Fetcher {
    pub fn new(policy: Policy) -> Self {
        let mut config = ureq::Agent::config_builder()
            .max_redirects(0) // followed by hand: every hop is checked against the policy
            .http_status_as_error(false)
            .timeout_connect(Some(Duration::from_secs(15)))
            .timeout_recv_response(Some(Duration::from_secs(30)))
            .timeout_recv_body(Some(Duration::from_secs(60)))
            .user_agent(concat!("surgeon-gui/", env!("CARGO_PKG_VERSION")));
        // Environment proxies are the client's default; the Windows system proxy fills in.
        if ureq::Proxy::try_from_env().is_none() {
            if let Some(proxy) =
                crate::platform::system_proxy().and_then(|p| ureq::Proxy::new(&p).ok())
            {
                config = config.proxy(Some(proxy));
            }
        }
        #[cfg(windows)]
        {
            config = config.tls_config(
                ureq::tls::TlsConfig::builder()
                    .provider(ureq::tls::TlsProvider::NativeTls)
                    .build(),
            );
        }
        Self {
            agent: config.build().into(),
            policy,
        }
    }

    pub fn policy(&self) -> &Policy {
        &self.policy
    }

    /// GET with manual redirects; `from` asks for `Range: bytes=from-`.
    pub fn open(&self, url: &str, from: Option<u64>) -> Result<Opened, NetError> {
        let mut current = url.to_owned();
        for _ in 0..=MAX_REDIRECTS {
            if !self.policy.allows(&current) {
                return Err(NetError::Blocked(current));
            }
            let mut request = self.agent.get(&current);
            if let Some(from) = from.filter(|&f| f > 0) {
                request = request.header("Range", format!("bytes={from}-"));
            }
            let response = request
                .call()
                .map_err(|e| NetError::Unreachable(e.to_string()))?;
            let status = response.status().as_u16();
            if matches!(status, 301 | 302 | 303 | 307 | 308) {
                let location = response
                    .headers()
                    .get("location")
                    .and_then(|v| v.to_str().ok())
                    .ok_or(NetError::Http(status))?;
                current = resolve(&current, location);
                continue;
            }
            let header = |name: &str| response.headers().get(name).and_then(|v| v.to_str().ok());
            let content_length = header("content-length").and_then(|v| v.parse().ok());
            let content_range_start = header("content-range")
                .and_then(|v| v.strip_prefix("bytes "))
                .and_then(|v| v.split(['-', '/']).next())
                .and_then(|v| v.parse().ok());
            return Ok(Opened {
                status,
                content_length,
                content_range_start,
                body: Box::new(response.into_body().into_reader()),
            });
        }
        Err(NetError::Http(310))
    }

    /// A small document (an API answer), refused above `limit` bytes.
    pub fn get_bytes(&self, url: &str, limit: u64) -> Result<Vec<u8>, NetError> {
        let opened = self.open(url, None)?;
        if opened.status != 200 {
            return Err(NetError::Http(opened.status));
        }
        let mut out = Vec::new();
        opened
            .body
            .take(limit + 1)
            .read_to_end(&mut out)
            .map_err(|e| NetError::Unreachable(e.to_string()))?;
        match out.len() as u64 > limit {
            true => Err(NetError::TooLarge),
            false => Ok(out),
        }
    }

    /// Downloads `url` into `part`, resuming what is already there. `expected` (the size the
    /// release announces) lets a complete leftover skip the network and detects a short
    /// transfer. Returns the SHA-256 of the whole file.
    pub fn download(
        &self,
        url: &str,
        part: &Path,
        expected: Option<u64>,
        cancel: &AtomicBool,
        mut progress: impl FnMut(u64, Option<u64>),
    ) -> Result<String, NetError> {
        let io = |e: io::Error| NetError::Io(format!("{} : {e}", part.display()));
        if let Some(dir) = part.parent() {
            fs::create_dir_all(dir).map_err(io)?;
        }
        let mut have = fs::metadata(part).map(|m| m.len()).unwrap_or(0);
        if expected.is_some_and(|e| have > e) {
            fs::remove_file(part).map_err(io)?;
            have = 0;
        }
        if expected != Some(have) || have == 0 {
            have = self.fetch_into(url, part, have, expected, cancel, &mut progress)?;
        }
        if expected.is_some_and(|e| have != e) {
            return Err(NetError::Interrupted(have));
        }
        progress(have, Some(have));
        hash_file(part).map_err(io)
    }

    fn fetch_into(
        &self,
        url: &str,
        part: &Path,
        mut have: u64,
        expected: Option<u64>,
        cancel: &AtomicBool,
        progress: &mut impl FnMut(u64, Option<u64>),
    ) -> Result<u64, NetError> {
        let io = |e: io::Error| NetError::Io(format!("{} : {e}", part.display()));
        let mut opened = self.open(url, Some(have))?;
        if opened.status == 416 {
            // The leftover does not fit the file on the server: start over.
            fs::remove_file(part).ok();
            have = 0;
            opened = self.open(url, None)?;
        }
        let append = match opened.status {
            206 if opened.content_range_start == Some(have) => true,
            200 | 206 => false, // the server ignored the range (or answered another one)
            code => return Err(NetError::Http(code)),
        };
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(part)
            .map_err(io)?;
        if !append {
            have = 0;
        }
        let total = expected.or_else(|| opened.content_length.map(|l| l + have));
        let mut buf = vec![0u8; 256 * 1024];
        loop {
            if cancel.load(Ordering::Relaxed) {
                file.flush().map_err(io)?;
                return Err(NetError::Cancelled);
            }
            match opened.body.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    file.write_all(&buf[..n]).map_err(io)?;
                    have += n as u64;
                    progress(have, total);
                }
                Err(_) => {
                    file.flush().map_err(io)?;
                    return Err(NetError::Interrupted(have));
                }
            }
        }
        file.flush().map_err(io)?;
        Ok(have)
    }
}

/// `Location` can be absolute or a path on the same host.
fn resolve(base: &str, location: &str) -> String {
    if location.contains("://") {
        return location.to_owned();
    }
    let origin_end = base.find("://").map_or(0, |i| i + 3);
    let origin_end = base[origin_end..]
        .find('/')
        .map_or(base.len(), |i| origin_end + i);
    format!("{}{}", &base[..origin_end], location)
}

/// SHA-256 of a file, lowercase hex.
pub fn hash_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 256 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_allow_list_is_strict() {
        let p = Policy::production();
        for ok in [
            "https://api.github.com/repos/ggml-org/llama.cpp/releases",
            "https://github.com/ggml-org/llama.cpp/releases/download/b1/x.zip",
            "https://release-assets.githubusercontent.com/x",
            "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json",
            "https://cas-bridge.xethub.hf.co/x",
        ] {
            assert!(p.allows(ok), "{ok}");
        }
        for bad in [
            "http://github.com/x",
            "https://evilgithub.com/x",
            "https://github.com.evil.example/x",
            "https://evilhf.co/x",
            "https://user@github.com/x",
            "https://127.0.0.1/x",
            "ftp://github.com/x",
            "github.com/x",
        ] {
            assert!(!p.allows(bad), "{bad}");
        }
        assert!(Policy::loopback().allows("http://127.0.0.1:8123/a"));
        assert!(!Policy::loopback().allows("http://example.com/a"));
    }

    #[test]
    fn relative_redirects_keep_the_origin() {
        assert_eq!(
            resolve("https://huggingface.co/a/b?x=1", "/api/c"),
            "https://huggingface.co/api/c"
        );
        assert_eq!(resolve("https://a.b", "https://c.d/e"), "https://c.d/e");
        assert_eq!(resolve("https://a.b", "/x"), "https://a.b/x");
    }
}
