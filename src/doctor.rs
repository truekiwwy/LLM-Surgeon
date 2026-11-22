//! `surgeon doctor` and the MCP tool `surgeon_doctor`: whether this machine is ready for a
//! campaign, in a few seconds and without loading a model.
//!
//! - memory: total and free RAM, the largest weights that run in f32 and in bf16 without the
//!   slow fallback (the RAM rule of `extract::dtype_decision`), and with `--model` that
//!   model's exact need and the dtype it gets;
//! - CPU: physical cores, the threads and (with `--model`) the batch of native generation;
//! - reference cache (`ref_bench`): directory, entries, size, unreadable and obsolete ones,
//!   purged by `--clean-cache` (CLI only);
//! - hipfire: when configured or answering, its URL, `/health`, `/metrics`, and the GPU memory
//!   of its last load, read from the daemon's log since hipfire 0.4 exposes it nowhere else.
//!   `0 MB total` there is a daemon that lost its GPU: it needs a restart.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Serialize;
use serde_json::{json, Value};

use crate::app::{CoreResult, Outcome};
use crate::backend::{hipfire, BackendConfig};
use crate::bench::ref_cache;
use crate::hardware::{BudgetPolicy, HardwareProbe};

/// How long a hipfire endpoint has to answer.
const HTTP_TIMEOUT: Duration = Duration::from_secs(2);
/// The WSL distribution and log file `start-hipfire.ps1` uses.
const HIPFIRE_DISTRO: &str = "Ubuntu-24.04";
const HIPFIRE_LOG: &str = "hipfire-serve.log";

#[derive(Debug, Clone, Default)]
pub struct DoctorRequest {
    pub model: Option<PathBuf>,
    pub clean_cache: bool,
    /// Default: the configuration's `hipfire_url`, else the daemon's documented port.
    pub hipfire_url: Option<String>,
    /// Default: the log `start-hipfire.ps1` writes.
    pub hipfire_log: Option<PathBuf>,
    /// Default: [`ref_cache::default_dir`].
    pub cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    Ok,
    Warn,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Check {
    pub name: &'static str,
    pub level: Level,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelFit {
    pub path: PathBuf,
    /// RAM needed with the margin, in each dtype.
    pub f32_required_bytes: u64,
    pub bf16_required_bytes: u64,
    /// What `--dtype auto` loads: f32, bf16 (the slow fallback), or none.
    pub dtype: String,
    /// Sequences native generation decodes together, from the RAM left after the weights.
    pub batch_size: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MemoryReport {
    pub total_bytes: u64,
    pub available_bytes: u64,
    /// Largest weights (bytes) that load in f32, and in bf16, with the margin; the
    /// activations of a long prompt come on top.
    pub max_f32_weights_bytes: u64,
    pub max_bf16_weights_bytes: u64,
    /// The same as parameters (4 and 2 bytes each).
    pub max_f32_params: u64,
    pub max_bf16_params: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelFit>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CpuReport {
    pub physical_cores: Option<usize>,
    /// Threads of the native generation pool.
    pub threads: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CacheReport {
    /// null: no cache (none configured, or the test harness).
    pub dir: Option<PathBuf>,
    pub entries: usize,
    pub bytes: u64,
    pub corrupt: usize,
    pub obsolete: usize,
    /// Files removed by `--clean-cache`.
    pub removed: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GpuMemory {
    pub gpu: String,
    pub free_mb: u64,
    pub total_mb: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HipfireReport {
    pub url: String,
    /// Named by the configuration (backend, hipfire_url or hipfire_bin).
    pub configured: bool,
    /// `/health` answered.
    pub up: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<u64>,
    /// `hipfire_*` counters of `/metrics` (uptime_seconds, requests_failed_total...).
    pub metrics: HashMap<String, f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log: Option<PathBuf>,
    /// Of the last load the log reports.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_memory: Option<GpuMemory>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DoctorReport {
    /// No check failed.
    pub ready: bool,
    pub checks: Vec<Check>,
    pub memory: MemoryReport,
    pub cpu: CpuReport,
    pub cache: CacheReport,
    /// null: hipfire neither configured nor answering.
    pub hipfire: Option<HipfireReport>,
}

const GIB: f64 = (1u64 << 30) as f64;

fn gib(bytes: u64) -> String {
    format!("{:.1} Gio", bytes as f64 / GIB)
}

/// The largest weights `required_ram` lets load into `available` bytes.
pub fn max_weights(available: u64, policy: &BudgetPolicy) -> u64 {
    let factor = policy.ram_overhead_factor.max(1.0);
    (available.saturating_sub(policy.ram_safety_margin_bytes) as f64 / factor) as u64
}

fn memory(probe: &dyn HardwareProbe, model: Option<&Path>) -> (MemoryReport, Check) {
    let policy = BudgetPolicy::default();
    let ram = probe.system_ram().unwrap_or(crate::hardware::MemoryInfo {
        total_bytes: 0,
        available_bytes: 0,
    });
    let weights = max_weights(ram.available_bytes, &policy);
    let mut report = MemoryReport {
        total_bytes: ram.total_bytes,
        available_bytes: ram.available_bytes,
        max_f32_weights_bytes: weights,
        max_bf16_weights_bytes: weights,
        max_f32_params: weights / 4,
        max_bf16_params: weights / 2,
        model: None,
    };
    let billions = |p: u64| format!("{:.1} G", p as f64 / 1e9);
    let mut check = Check {
        name: "memory",
        level: Level::Ok,
        message: format!(
            "{} libres sur {} : f32 jusqu'à ~{} paramètres, bf16 ~{} (activations en plus)",
            gib(ram.available_bytes),
            gib(ram.total_bytes),
            billions(report.max_f32_params),
            billions(report.max_bf16_params)
        ),
    };
    if let Some(path) = model {
        match model_fit(path, ram.available_bytes, &policy) {
            Ok(fit) => {
                check.level = match fit.dtype.as_str() {
                    "f32" => Level::Ok,
                    "bf16" => Level::Warn,
                    _ => Level::Fail,
                };
                check.message = format!(
                    "{} ; {} demande {} en f32, {} en bf16 (marge comprise) -> {}",
                    check.message,
                    path.display(),
                    gib(fit.f32_required_bytes),
                    gib(fit.bf16_required_bytes),
                    match fit.dtype.as_str() {
                        "f32" => "f32".to_owned(),
                        "bf16" => "bf16, repli lent (environ 2,5 fois)".into(),
                        _ => "ne se charge pas".into(),
                    }
                );
                report.model = Some(fit);
            }
            Err(e) => {
                check.level = Level::Fail;
                check.message = format!("{} : {e}", path.display());
            }
        }
    }
    (report, check)
}

fn model_fit(path: &Path, available: u64, policy: &BudgetPolicy) -> Result<ModelFit, String> {
    use crate::extract::{
        check_tensors, load_config, plan_memory, required_ram, required_tensors_full,
        SafetensorsDir,
    };
    let dir = crate::extract::model_folder(path);
    let cfg = load_config(dir).map_err(|e| e.to_string())?;
    let source = SafetensorsDir::open(dir).map_err(|e| e.to_string())?;
    let totals = check_tensors(&source, &required_tensors_full(&cfg)).map_err(|e| e.to_string())?;
    let plan = plan_memory(&cfg, totals, crate::bench::native::PLANNED_CONTEXT);
    let (f32_need, bf16_need) = (
        required_ram(plan.f32_bytes, policy),
        required_ram(plan.bf16_bytes, policy),
    );
    let (dtype, used) = match () {
        _ if f32_need <= available => ("f32", plan.f32_bytes),
        _ if bf16_need <= available => ("bf16", plan.bf16_bytes),
        _ => ("none", 0),
    };
    let batch_size = (dtype != "none").then(|| {
        let per_sequence = crate::bench::native::kv_bytes_per_sequence(&cfg);
        crate::bench::generation::auto_batching(
            available.saturating_sub(used),
            per_sequence,
            sysinfo::System::physical_core_count().ok(),
        )
        .0
    });
    Ok(ModelFit {
        path: path.to_path_buf(),
        f32_required_bytes: f32_need,
        bf16_required_bytes: bf16_need,
        dtype: dtype.into(),
        batch_size,
    })
}

fn cpu(model: Option<&ModelFit>) -> (CpuReport, Check) {
    let cores = sysinfo::System::physical_core_count().ok();
    let threads = cores.unwrap_or_else(rayon::current_num_threads);
    let batch = model.and_then(|m| m.batch_size).map_or_else(
        || "lots selon le modèle (--model)".to_owned(),
        |b| format!("lots de {b} séquences au plus"),
    );
    let check = Check {
        name: "cpu",
        level: Level::Ok,
        message: format!(
            "{} cœurs physiques : génération native sur {threads} threads, {batch}",
            cores.map_or("?".to_owned(), |c| c.to_string())
        ),
    };
    (
        CpuReport {
            physical_cores: cores,
            threads,
        },
        check,
    )
}

fn cache(dir: Option<PathBuf>, clean: bool) -> (CacheReport, Check) {
    let Some(dir) = dir else {
        let report = CacheReport {
            dir: None,
            entries: 0,
            bytes: 0,
            corrupt: 0,
            obsolete: 0,
            removed: 0,
        };
        let check = Check {
            name: "cache",
            level: Level::Ok,
            message: "cache des références désactivé : les références sont regénérées".into(),
        };
        return (report, check);
    };
    let mut scan = ref_cache::scan(&dir);
    let removed = match clean {
        true => ref_cache::purge(&scan),
        false => 0,
    };
    if removed > 0 {
        scan = ref_cache::scan(&dir);
    }
    let stale = scan.corrupt.len() + scan.obsolete.len();
    let check = Check {
        name: "cache",
        level: if stale > 0 { Level::Warn } else { Level::Ok },
        message: format!(
            "{} : {} entrées, {:.1} Mio, {} illisibles, {} obsolètes{}",
            dir.display(),
            scan.entries,
            scan.bytes as f64 / (1u64 << 20) as f64,
            scan.corrupt.len(),
            scan.obsolete.len(),
            match (stale, removed) {
                (_, r) if r > 0 => format!(" ({r} fichiers supprimés)"),
                (0, _) => String::new(),
                _ => " : --clean-cache les supprime".into(),
            }
        ),
    };
    let report = CacheReport {
        dir: Some(dir),
        entries: scan.entries,
        bytes: scan.bytes,
        corrupt: scan.corrupt.len(),
        obsolete: scan.obsolete.len(),
        removed,
    };
    (report, check)
}

/// The `hipfire_*` samples of a Prometheus text, histograms and labelled series aside.
pub fn parse_metrics(text: &str) -> HashMap<String, f64> {
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let (name, value) = l.split_once(' ')?;
            let name = name.strip_prefix("hipfire_")?;
            if name.contains('{') || name.ends_with("_bucket") {
                return None;
            }
            Some((name.to_owned(), value.trim().parse().ok()?))
        })
        .collect()
}

/// The last `GPU: <arch> (<free> MB free / <total> MB total)` of a daemon log.
pub fn parse_gpu_memory(log: &str) -> Option<GpuMemory> {
    log.lines().rev().find_map(|line| {
        let rest = &line[line.find("GPU: ")? + 5..];
        let (gpu, rest) = rest.split_once(" (")?;
        let (free, rest) = rest.split_once(" MB free / ")?;
        let total = rest.split_once(" MB total)")?.0;
        Some(GpuMemory {
            gpu: gpu.trim().to_owned(),
            free_mb: free.trim().parse().ok()?,
            total_mb: total.trim().parse().ok()?,
        })
    })
}

/// Where `start-hipfire.ps1` writes the daemon's log, seen from here.
fn default_log() -> Option<PathBuf> {
    let path = match cfg!(windows) {
        true => {
            let user = std::env::var("USERNAME").ok()?;
            PathBuf::from(format!(
                r"\\wsl.localhost\{HIPFIRE_DISTRO}\home\{user}\{HIPFIRE_LOG}"
            ))
        }
        false => PathBuf::from(std::env::var_os("HOME")?).join(HIPFIRE_LOG),
    };
    path.is_file().then_some(path)
}

fn get(port: u16, path: &str) -> Option<String> {
    let (_, body) = crate::eval::http::request(port, "GET", path, None, None, HTTP_TIMEOUT).ok()?;
    body.text().ok()
}

fn hipfire_check(req: &DoctorRequest, user: &BackendConfig) -> Option<(HipfireReport, Check)> {
    let configured = req.hipfire_url.is_some()
        || user.hipfire_url.is_some()
        || user.hipfire_bin.is_some()
        || user.backend == Some(crate::bench::score::Backend::Hipfire);
    let url = req
        .hipfire_url
        .clone()
        .or_else(|| user.hipfire_url.clone())
        .unwrap_or_else(|| format!("http://127.0.0.1:{}", hipfire::DEFAULT_PORT));
    let fail = |message: String| Check {
        name: "hipfire",
        level: Level::Fail,
        message,
    };
    let port = match hipfire::loopback_port(&url) {
        Ok(port) => port,
        Err(e) if configured => {
            return Some((empty_hipfire(url, configured), fail(e)));
        }
        Err(_) => return None,
    };
    let health = get(port, "/health").and_then(|t| serde_json::from_str::<Value>(&t).ok());
    if health.is_none() && !configured {
        return None;
    }
    let mut report = empty_hipfire(url.clone(), configured);
    let restart = "arrête le démon dans WSL puis relance .\\start-hipfire.ps1 (il ne relance pas \
                   un démon qui répond encore)";
    let Some(health) = health else {
        let check = fail(format!(
            "{url} ne répond pas : lance .\\start-hipfire.ps1 (ou corrige hipfire_url)"
        ));
        return Some((report, check));
    };
    report.up = true;
    report.status = health["status"].as_str().map(String::from);
    report.pid = health["pid"].as_u64();
    report.metrics = get(port, "/metrics")
        .map(|t| parse_metrics(&t))
        .unwrap_or_default();
    report.log = req.hipfire_log.clone().or_else(default_log);
    report.gpu_memory = report
        .log
        .as_ref()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| parse_gpu_memory(&t));
    let uptime = report
        .metrics
        .get("uptime_seconds")
        .map_or(String::new(), |s| {
            format!(", en route depuis {:.1} h", s / 3600.0)
        });
    let failed = report.metrics.get("requests_failed_total").copied();
    let up = format!(
        "{url} répond ({}{uptime}{})",
        report.status.as_deref().unwrap_or("?"),
        failed.map_or(String::new(), |f| format!(", {f} requêtes en échec"))
    );
    let check = match &report.gpu_memory {
        Some(g) if g.total_mb == 0 => fail(format!(
            "{up} ; GPU {} : {} Mo libres sur {} Mo au dernier chargement : démon figé, sans \
             GPU, tout chargement échoue ; {restart}",
            g.gpu, g.free_mb, g.total_mb
        )),
        Some(g) if g.free_mb == 0 => Check {
            name: "hipfire",
            level: Level::Warn,
            message: format!(
                "{up} ; GPU {} : mémoire pleine au dernier chargement (0 Mo libres sur {} Mo)",
                g.gpu, g.total_mb
            ),
        },
        Some(g) => Check {
            name: "hipfire",
            level: Level::Ok,
            message: format!(
                "{up} ; GPU {} : {} Mo libres sur {} Mo au dernier chargement",
                g.gpu, g.free_mb, g.total_mb
            ),
        },
        None => Check {
            name: "hipfire",
            level: Level::Warn,
            message: format!(
                "{up} ; mémoire GPU inconnue : hipfire ne l'écrit que dans son log au \
                 chargement d'un modèle ({}, --hipfire-log pour un autre)",
                report
                    .log
                    .as_ref()
                    .map_or("log introuvable".to_owned(), |p| format!(
                        "aucune ligne GPU dans {}",
                        p.display()
                    ))
            ),
        },
    };
    Some((report, check))
}

fn empty_hipfire(url: String, configured: bool) -> HipfireReport {
    HipfireReport {
        url,
        configured,
        up: false,
        status: None,
        pid: None,
        metrics: HashMap::new(),
        log: None,
        gpu_memory: None,
    }
}

/// Every check, the report and its verdict.
pub fn run(req: &DoctorRequest, probe: &dyn HardwareProbe, user: &BackendConfig) -> DoctorReport {
    let (memory, memory_check) = memory(probe, req.model.as_deref());
    let (cpu, cpu_check) = cpu(memory.model.as_ref());
    let dir = req.cache_dir.clone().or_else(ref_cache::default_dir);
    let (cache, cache_check) = cache(dir, req.clean_cache);
    let hipfire = hipfire_check(req, user);
    let mut checks = vec![memory_check, cpu_check, cache_check];
    let hipfire = hipfire.map(|(report, check)| {
        checks.push(check);
        report
    });
    DoctorReport {
        ready: checks.iter().all(|c| c.level != Level::Fail),
        checks,
        memory,
        cpu,
        cache,
        hipfire,
    }
}

/// The CLI and MCP core: the report in `data`, the checks that are not ok as warnings.
pub fn core_doctor(
    req: &DoctorRequest,
    probe: &dyn HardwareProbe,
    ui: &crate::ui::Ui,
) -> CoreResult {
    let user = crate::backend::load_user_config().unwrap_or_default();
    let report = run(req, probe, &user);
    let badge = |name: &str| match name {
        "memory" => "RAM",
        "cpu" => "CPU",
        "cache" => "CACHE",
        _ => "HIPFIRE",
    };
    let mark = |level: Level| match level {
        Level::Ok => "ok",
        Level::Warn => "attention",
        Level::Fail => "échec",
    };
    let mut lines = vec![format!(
        "{} {}",
        crate::ui::badge(ui, "DOCTOR"),
        crate::ui::metric(
            ui,
            match report.ready {
                true => "prêt",
                false => "pas prêt",
            }
        )
    )];
    lines.extend(report.checks.iter().map(|c| {
        format!(
            "{} {} · {}",
            crate::ui::badge(ui, badge(c.name)),
            crate::ui::metric(ui, mark(c.level)),
            c.message
        )
    }));
    let mut outcome = Outcome::ok(json!(report), lines.join("\n") + "\n");
    outcome.warnings = report
        .checks
        .iter()
        .filter(|c| c.level != Level::Ok)
        .map(|c| c.message.clone())
        .collect();
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    use super::*;
    use crate::hardware::{GpuInfo, MemoryInfo};

    struct Ram(u64);

    impl HardwareProbe for Ram {
        fn system_ram(&self) -> crate::hardware::Result<MemoryInfo> {
            Ok(MemoryInfo {
                total_bytes: 64 << 30,
                available_bytes: self.0,
            })
        }

        fn gpus(&self) -> crate::hardware::Result<Vec<GpuInfo>> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn memory_says_how_large_a_model_runs_in_f32_and_bf16() {
        let policy = BudgetPolicy::default();
        // 23.15 GiB free: 2 GiB of margin, then 1.15 of overhead leave 18.4 GiB of weights.
        let free = (2u64 << 30) + (23u64 << 30);
        let weights = max_weights(free, &policy);
        assert_eq!(weights, ((23u64 << 30) as f64 / 1.15) as u64);
        let (report, check) = memory(&Ram(free), None);
        assert_eq!(report.max_f32_params, weights / 4);
        assert_eq!(report.max_bf16_params, weights / 2);
        assert_eq!(check.level, Level::Ok);
        assert_eq!(max_weights(1 << 30, &policy), 0, "less than the margin");
        // A model that does not exist is a failed check, not a crash.
        let (_, check) = memory(&Ram(free), Some(Path::new("/nowhere/model")));
        assert_eq!(check.level, Level::Fail);
    }

    #[test]
    fn the_cpu_check_gives_the_native_threads() {
        let (report, check) = cpu(None);
        assert!(report.threads >= 1);
        assert!(check.message.contains("--model"), "{}", check.message);
    }

    #[test]
    fn the_cache_check_counts_and_cleans_what_cannot_be_read_again() {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("ref_bench");
        let store = ref_cache::RefCache::new(&cache_dir);
        store
            .put("k1", "refusal", Path::new("m"), 3, &vec![1, 2])
            .unwrap();
        store
            .put("k2", "eval", Path::new("m"), 3, &vec![3])
            .unwrap();
        std::fs::write(cache_dir.join("broken.json"), b"{\"schema\": 1, \"ke").unwrap();
        std::fs::write(cache_dir.join("k3.json.tmp"), b"half").unwrap();
        let old = json!({"schema": 0, "key": "k4", "kind": "eval", "model": "m",
                         "created_at": "", "timing_ms": 1, "items": []});
        std::fs::write(cache_dir.join("k4.json"), old.to_string()).unwrap();
        // A key that is not its name: a lookup would never find it.
        std::fs::copy(cache_dir.join("k1.json"), cache_dir.join("k5.json")).unwrap();

        let (report, check) = cache(Some(cache_dir.clone()), false);
        assert_eq!(
            (report.entries, report.corrupt, report.obsolete),
            (2, 2, 2),
            "{report:?}"
        );
        assert!(report.bytes > 0);
        assert_eq!(check.level, Level::Warn);
        assert!(check.message.contains("--clean-cache"));
        let (report, check) = cache(Some(cache_dir.clone()), true);
        assert_eq!(report.removed, 4);
        assert_eq!((report.entries, report.corrupt, report.obsolete), (2, 0, 0));
        assert_eq!(check.level, Level::Ok);
        assert!(cache_dir.join("k1.json").is_file() && cache_dir.join("k2.json").is_file());
        let (report, check) = cache(None, true);
        assert_eq!((report.dir, check.level), (None, Level::Ok));
    }

    /// A daemon that answers `/health` and `/metrics` like hipfire 0.4.6, as many times as asked.
    fn fake_hipfire() -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut stream = stream;
                let mut buf = [0u8; 1024];
                let n = stream.read(&mut buf).unwrap_or(0);
                let head = String::from_utf8_lossy(&buf[..n]).into_owned();
                let body = match head.split_whitespace().nth(1) {
                    Some("/health") => {
                        json!({"status": "ok", "model": null, "pid": 606}).to_string()
                    }
                    Some("/metrics") => "# HELP hipfire_uptime_seconds Uptime.\n\
                        hipfire_uptime_seconds 162265\nhipfire_requests_total 3\n\
                        hipfire_requests_failed_total 623\n\
                        hipfire_ttft_milliseconds_bucket{le=\"10\"} 0\n"
                        .to_owned(),
                    _ => String::new(),
                };
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        port
    }

    fn hipfire_with(log: &str, port: u16) -> (HipfireReport, Check) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hipfire-serve.log");
        std::fs::write(&path, log).unwrap();
        let req = DoctorRequest {
            hipfire_url: Some(format!("http://127.0.0.1:{port}")),
            hipfire_log: Some(path),
            ..DoctorRequest::default()
        };
        hipfire_check(&req, &BackendConfig::default()).unwrap()
    }

    #[test]
    fn a_healthy_daemon_and_one_frozen_at_zero_megabytes() {
        let port = fake_hipfire();
        let healthy = "qwen2: loading token_embd...\n\
            load ok. GPU: gfx1100 (15234 MB free / 16368 MB total); model loaded\n";
        let (report, check) = hipfire_with(healthy, port);
        assert!(report.up && report.configured);
        assert_eq!(report.pid, Some(606));
        assert_eq!(report.metrics["uptime_seconds"], 162265.0);
        assert_eq!(report.metrics["requests_failed_total"], 623.0);
        assert!(!report.metrics.contains_key("ttft_milliseconds_bucket"));
        assert_eq!(
            report.gpu_memory,
            Some(GpuMemory {
                gpu: "gfx1100".into(),
                free_mb: 15234,
                total_mb: 16368
            })
        );
        assert_eq!(check.level, Level::Ok, "{}", check.message);

        // The line hipfire 0.4.6 wrote on 2026-11-20 after 1 d 16 h: the last one counts.
        let frozen = format!(
            "{healthy}[hipfire] chatcmpl-606-297: streaming completion failed: daemon error: \
             [gpu retryable=false] load failed: qwen2: load_weights failed: HipError {{ code: 2, \
             message: \"hipMalloc: out of memory\" }}. GPU: gfx1100 (0 MB free / 0 MB total); \
             no model loaded\n"
        );
        let (report, check) = hipfire_with(&frozen, port);
        assert_eq!(report.gpu_memory.unwrap().total_mb, 0);
        assert_eq!(check.level, Level::Fail);
        assert!(
            check.message.contains("start-hipfire.ps1"),
            "{}",
            check.message
        );
        assert!(check.message.contains("démon figé"), "{}", check.message);

        // No GPU line: unknown, said as such.
        let (_, check) = hipfire_with("qwen2: loading token_embd...\n", port);
        assert_eq!(check.level, Level::Warn);
        assert!(check.message.contains("mémoire GPU inconnue"));
    }

    #[test]
    fn a_configured_daemon_that_does_not_answer_fails_an_absent_one_is_skipped() {
        // A port nothing listens on.
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let req = DoctorRequest {
            hipfire_url: Some(format!("http://127.0.0.1:{port}")),
            ..DoctorRequest::default()
        };
        let (report, check) = hipfire_check(&req, &BackendConfig::default()).unwrap();
        assert!(!report.up);
        assert_eq!(check.level, Level::Fail);
        assert!(check.message.contains("ne répond pas"));
        let remote = DoctorRequest {
            hipfire_url: Some("http://10.0.0.2:11435".into()),
            ..DoctorRequest::default()
        };
        let (_, check) = hipfire_check(&remote, &BackendConfig::default()).unwrap();
        assert!(check.message.contains("boucle locale"));
        // Neither configured nor answering on a free port: no hipfire section.
        let user = BackendConfig {
            hipfire_url: None,
            ..BackendConfig::default()
        };
        let unconfigured = DoctorRequest::default();
        if get(hipfire::DEFAULT_PORT, "/health").is_none() {
            assert!(hipfire_check(&unconfigured, &user).is_none());
        }
    }

    #[test]
    fn metrics_and_gpu_lines_are_parsed_strictly() {
        let m = parse_metrics("hipfire_engine_up 1\nother_metric 3\nhipfire_x{a=\"b\"} 2\n");
        assert_eq!(m.len(), 1);
        assert_eq!(m["engine_up"], 1.0);
        assert_eq!(
            parse_gpu_memory("GPU: gfx1100 (x MB free / 0 MB total)"),
            None
        );
        assert_eq!(parse_gpu_memory("no gpu here"), None);
    }
}
