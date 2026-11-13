//! HTTP for the providers: an endpoint the user configured, HTTPS (plain HTTP only to this
//! machine: Ollama, LM Studio, `llama-server`), the same `ureq` client and TLS stack as
//! `net.rs`, and no header ever logged or put in an error message (keys travel in headers).

use std::io::{BufReader, Read};
use std::time::Duration;

use super::provider::ProviderError;
use super::vault::Secret;

/// What an endpoint may be, decided before any socket is opened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reach {
    /// Any HTTPS host, or plain HTTP to loopback.
    Network,
    /// Local-only mode: loopback hosts only.
    LocalOnly,
}

pub struct Endpoint {
    base: String,
    agent: ureq::Agent,
}

/// A response whose status is already checked (2xx).
pub struct Body {
    pub reader: BufReader<Box<dyn Read + Send>>,
}

impl Endpoint {
    pub fn new(base: &str, reach: Reach, timeouts: Timeouts) -> Result<Self, ProviderError> {
        let base = base.trim().trim_end_matches('/').to_owned();
        let (scheme, host) = scheme_and_host(&base)
            .ok_or_else(|| ProviderError::Blocked(format!("adresse invalide : {base}")))?;
        let loopback = is_loopback(host);
        if reach == Reach::LocalOnly && !loopback {
            return Err(ProviderError::Blocked(format!(
                "mode local seulement : {host} n'est pas cette machine"
            )));
        }
        if !(scheme == "https" || (scheme == "http" && loopback)) {
            return Err(ProviderError::Blocked(format!(
                "HTTPS obligatoire hors de cette machine : {base}"
            )));
        }
        let mut config = ureq::Agent::config_builder()
            .max_redirects(0)
            .http_status_as_error(false)
            .timeout_connect(Some(timeouts.connect))
            .timeout_recv_response(Some(timeouts.first_byte))
            .timeout_recv_body(Some(timeouts.body))
            .user_agent(concat!("surgeon-gui/", env!("CARGO_PKG_VERSION")));
        // A proxy never sees loopback traffic; remote traffic uses the system's proxy.
        config = match loopback {
            true => config.proxy(None),
            false => match ureq::Proxy::try_from_env()
                .or_else(|| crate::platform::system_proxy().and_then(|p| ureq::Proxy::new(&p).ok()))
            {
                Some(p) => config.proxy(Some(p)),
                None => config,
            },
        };
        #[cfg(windows)]
        {
            config = config.tls_config(
                ureq::tls::TlsConfig::builder()
                    .provider(ureq::tls::TlsProvider::NativeTls)
                    .build(),
            );
        }
        Ok(Self {
            base,
            agent: config.build().into(),
        })
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    pub fn post_json(
        &self,
        path: &str,
        headers: &[(&str, HeaderValue)],
        body: &serde_json::Value,
    ) -> Result<Body, ProviderError> {
        let mut req = self
            .agent
            .post(format!("{}{path}", self.base))
            .header("content-type", "application/json");
        for (name, value) in headers {
            req = req.header(*name, value.expose());
        }
        let bytes = serde_json::to_vec(body).map_err(|e| ProviderError::Protocol(e.to_string()))?;
        checked(req.send(&bytes[..]))
    }

    pub fn get(&self, path: &str, headers: &[(&str, HeaderValue)]) -> Result<Body, ProviderError> {
        let mut req = self.agent.get(format!("{}{path}", self.base));
        for (name, value) in headers {
            req = req.header(*name, value.expose());
        }
        checked(req.call())
    }
}

/// A header value: plain, or a secret that is only exposed to the HTTP client.
pub enum HeaderValue<'a> {
    Plain(&'a str),
    Secret(&'a Secret, &'static str),
}

impl HeaderValue<'_> {
    fn expose(&self) -> String {
        match self {
            Self::Plain(v) => (*v).to_owned(),
            Self::Secret(s, prefix) => format!("{prefix}{}", s.expose()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Timeouts {
    pub connect: Duration,
    /// Until the response headers (a local model may load first).
    pub first_byte: Duration,
    /// The whole streamed body.
    pub body: Duration,
}

impl Default for Timeouts {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(15),
            first_byte: Duration::from_secs(180),
            body: Duration::from_secs(900),
        }
    }
}

fn checked(
    result: Result<ureq::http::Response<ureq::Body>, ureq::Error>,
) -> Result<Body, ProviderError> {
    let response = result.map_err(|e| match e {
        ureq::Error::Timeout(_) => ProviderError::Timeout,
        // The message names the host or the I/O failure, never a header.
        other => ProviderError::Network(other.to_string()),
    })?;
    let status = response.status().as_u16();
    let retry_after = response
        .headers()
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s >= 0.0)
        .map(Duration::from_secs_f64);
    let reader: Box<dyn Read + Send> = Box::new(response.into_body().into_reader());
    if !(200..300).contains(&status) {
        let mut text = String::new();
        let _ = reader.take(64 << 10).read_to_string(&mut text);
        return Err(ProviderError::from_status(status, &text, retry_after));
    }
    Ok(Body {
        reader: BufReader::new(reader),
    })
}

/// Reads a JSON answer of at most 16 MiB.
pub fn read_json(body: Body) -> Result<serde_json::Value, ProviderError> {
    let mut text = String::new();
    body.reader
        .take(16 << 20)
        .read_to_string(&mut text)
        .map_err(|e| ProviderError::Network(e.to_string()))?;
    serde_json::from_str(&text).map_err(|e| ProviderError::Protocol(e.to_string()))
}

/// `https://host:port/path` -> (`https`, `host`); credentials in the URL are refused.
fn scheme_and_host(url: &str) -> Option<(&str, &str)> {
    let (scheme, rest) = url.split_once("://")?;
    let authority = rest.split(['/', '?', '#']).next()?;
    if authority.contains('@') {
        return None;
    }
    let host = match authority.strip_prefix('[') {
        Some(v6) => v6.split(']').next()?,
        None => authority.rsplit_once(':').map_or(authority, |(h, _)| h),
    };
    (!host.is_empty()).then_some((scheme, host))
}

pub fn is_loopback(host: &str) -> bool {
    let host = host.to_ascii_lowercase();
    host == "localhost"
        || host == "::1"
        || host
            .parse::<std::net::Ipv4Addr>()
            .is_ok_and(|ip| ip.is_loopback())
}

/// Whether a base URL stays on this machine (for the consent screen and local-only mode).
pub fn is_local_url(url: &str) -> bool {
    scheme_and_host(url.trim()).is_some_and(|(_, h)| is_loopback(h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_are_https_or_loopback() {
        let t = Timeouts::default();
        assert!(Endpoint::new("https://api.anthropic.com", Reach::Network, t).is_ok());
        assert!(Endpoint::new("http://127.0.0.1:11434/v1", Reach::Network, t).is_ok());
        assert!(Endpoint::new("http://localhost:1234/v1", Reach::LocalOnly, t).is_ok());
        assert!(Endpoint::new("http://[::1]:8080/v1", Reach::LocalOnly, t).is_ok());
        for (url, reach) in [
            ("http://example.com/v1", Reach::Network),
            ("https://api.openai.com/v1", Reach::LocalOnly),
            ("https://user:pw@api.openai.com/v1", Reach::Network),
            ("ftp://127.0.0.1", Reach::Network),
            ("nonsense", Reach::Network),
        ] {
            assert!(
                matches!(Endpoint::new(url, reach, t), Err(ProviderError::Blocked(_))),
                "{url}"
            );
        }
        assert!(is_local_url("http://127.0.0.1:8080"));
        assert!(!is_local_url("https://openrouter.ai/api/v1"));
    }
}
