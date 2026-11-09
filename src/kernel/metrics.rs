//! I/O measurements of an apply: page faults (process-wide counters), bytes, time and the
//! kernel level used. Reported in `data.io` of the `apply` JSON envelope.

use std::time::Instant;

use serde::Serialize;

use super::dispatch::KernelChoice;
use super::mmap_io::{IoConfig, IoCounters};

/// Page faults of this process. Unix (`getrusage`) splits minor / major; Windows
/// (`GetProcessMemoryInfo`) only counts the total (soft + hard).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct PageFaults {
    pub minor: Option<u64>,
    pub major: Option<u64>,
    pub total: u64,
}

impl PageFaults {
    /// Faults between `before` and `self`.
    pub fn since(self, before: PageFaults) -> PageFaults {
        let sub = |a: Option<u64>, b: Option<u64>| Some(a?.saturating_sub(b?));
        PageFaults {
            minor: sub(self.minor, before.minor),
            major: sub(self.major, before.major),
            total: self.total.saturating_sub(before.total),
        }
    }
}

/// Current counters; `None` when the platform call fails or does not exist.
pub fn page_faults() -> Option<PageFaults> {
    #[cfg(unix)]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        // SAFETY: `usage` is a writable, correctly sized `rusage` buffer.
        let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
        if rc != 0 {
            return None;
        }
        // SAFETY: `getrusage` returned 0, so it filled the structure (zeroed beforehand).
        let usage = unsafe { usage.assume_init() };
        let minor = u64::try_from(usage.ru_minflt).ok()?;
        let major = u64::try_from(usage.ru_majflt).ok()?;
        Some(PageFaults {
            minor: Some(minor),
            major: Some(major),
            total: minor + major,
        })
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::ProcessStatus::{
            GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        let size = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>();
        let mut counters = PROCESS_MEMORY_COUNTERS {
            cb: u32::try_from(size).ok()?,
            ..Default::default()
        };
        // SAFETY: `counters` is a writable `PROCESS_MEMORY_COUNTERS` whose `cb` holds its
        // exact size; the pseudo-handle of the current process needs no closing.
        let ok = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
        match ok {
            0 => None,
            _ => Some(PageFaults {
                minor: None,
                major: None,
                total: u64::from(counters.PageFaultCount),
            }),
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        None
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IoReport {
    pub kernel: KernelChoice,
    pub config: IoConfig,
    #[serde(flatten)]
    pub counters: IoCounters,
    pub seconds: f64,
    /// Bytes of targeted tensors processed per second (read + write of each byte).
    pub throughput_mib_s: f64,
    /// `None` when the platform does not report page faults.
    pub page_faults: Option<PageFaults>,
}

/// Started before the first write, finished after the last flush.
#[derive(Debug, Clone, Copy)]
pub struct IoMeter {
    started: Instant,
    faults: Option<PageFaults>,
}

impl IoMeter {
    pub fn start() -> Self {
        Self {
            faults: page_faults(),
            started: Instant::now(),
        }
    }

    pub fn finish(&self, kernel: KernelChoice, config: IoConfig, counters: IoCounters) -> IoReport {
        let seconds = self.started.elapsed().as_secs_f64();
        let page_faults = match (self.faults, page_faults()) {
            (Some(before), Some(after)) => Some(after.since(before)),
            _ => None,
        };
        IoReport {
            kernel,
            config,
            counters,
            seconds,
            throughput_mib_s: match seconds > 0.0 {
                true => counters.bytes as f64 / (1024.0 * 1024.0) / seconds,
                false => 0.0,
            },
            page_faults,
        }
    }
}
