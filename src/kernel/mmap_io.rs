//! Chunked in-place patching of a mapped file, with OS page hints, plus the few system calls
//! the patchers need (mapping, free disk space, page-cache eviction for cold benchmarks).
//!
//! Prefetch is advisory: `madvise` (Unix, through `memmap2`) or `PrefetchVirtualMemory`
//! (Windows). Every failure is counted, never reported as an error. Direct I/O (`O_DIRECT`)
//! is not implemented: it bypasses the page cache that a memory mapping is made of.

use std::fs::File;
use std::io;
use std::ops::Range;
use std::path::Path;

use memmap2::{Mmap, MmapMut, MmapOptions};
use serde::Serialize;

pub const DEFAULT_CHUNK_BYTES: usize = 64 << 20;

/// Chunks requested ahead of the one being computed.
pub const PREFETCH_AHEAD: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct IoConfig {
    /// Target chunk size; rounded to whole rows, at least one row.
    pub chunk_bytes: usize,
    /// `false` (`--no-prefetch`): no hint and no intermediate flush, for comparison.
    pub prefetch: bool,
}

impl Default for IoConfig {
    fn default() -> Self {
        Self {
            chunk_bytes: DEFAULT_CHUNK_BYTES,
            prefetch: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct IoCounters {
    pub chunks: u64,
    pub bytes: u64,
    pub prefetch_requests: u64,
    /// Failed hints (sequential or will-need). Never fatal.
    pub prefetch_failures: u64,
    pub flush_requests: u64,
    /// Failed asynchronous flushes. Never fatal: the final synchronous flush reports errors.
    pub flush_failures: u64,
}

impl IoCounters {
    pub fn merge(&mut self, other: IoCounters) {
        self.chunks += other.chunks;
        self.bytes += other.bytes;
        self.prefetch_requests += other.prefetch_requests;
        self.prefetch_failures += other.prefetch_failures;
        self.flush_requests += other.flush_requests;
        self.flush_failures += other.flush_failures;
    }
}

/// OS hints on a range of a mapping. [`OsHints`] in production; tests inject failures.
pub trait PageHints {
    fn sequential(&mut self, map: &MmapMut, range: Range<usize>) -> io::Result<()>;
    fn will_need(&mut self, map: &MmapMut, range: Range<usize>) -> io::Result<()>;
    fn flush_async(&mut self, map: &MmapMut, range: Range<usize>) -> io::Result<()>;
}

/// `madvise` through `memmap2` on Unix, `PrefetchVirtualMemory` on Windows, nothing
/// elsewhere. `flush_async` is `msync(MS_ASYNC)` / `FlushViewOfFile`.
#[derive(Debug, Default, Clone, Copy)]
pub struct OsHints;

impl PageHints for OsHints {
    fn sequential(&mut self, map: &MmapMut, range: Range<usize>) -> io::Result<()> {
        #[cfg(unix)]
        {
            map.advise_range(memmap2::Advice::Sequential, range.start, range.len())
        }
        // Windows has no access-pattern hint for a view: the will-need prefetch does the work.
        #[cfg(not(unix))]
        {
            let _ = (map, range);
            Ok(())
        }
    }

    fn will_need(&mut self, map: &MmapMut, range: Range<usize>) -> io::Result<()> {
        #[cfg(unix)]
        {
            map.advise_range(memmap2::Advice::WillNeed, range.start, range.len())
        }
        #[cfg(windows)]
        {
            prefetch_windows(map, range)
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = (map, range);
            Ok(())
        }
    }

    fn flush_async(&mut self, map: &MmapMut, range: Range<usize>) -> io::Result<()> {
        map.flush_async_range(range.start, range.len())
    }
}

#[cfg(windows)]
fn prefetch_windows(map: &MmapMut, range: Range<usize>) -> io::Result<()> {
    use windows_sys::Win32::System::Memory::{PrefetchVirtualMemory, WIN32_MEMORY_RANGE_ENTRY};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let bytes = map
        .get(range)
        .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
    if bytes.is_empty() {
        return Ok(());
    }
    let entry = WIN32_MEMORY_RANGE_ENTRY {
        VirtualAddress: bytes.as_ptr().cast_mut().cast(),
        NumberOfBytes: bytes.len(),
    };
    // SAFETY: `entry` describes `bytes`, a sub-slice of the live mapping (bounds checked by
    // `get`) borrowed for the whole call. The call only reads the one-entry array and asks
    // the memory manager to page the range in; it never writes to it.
    let ok = unsafe { PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0) };
    match ok {
        0 => Err(io::Error::last_os_error()),
        _ => Ok(()),
    }
}

/// Why [`ChunkedPatcher::run`] stopped.
#[derive(Debug)]
pub enum ChunkFailure<E> {
    /// The cancellation flag was raised between two chunks.
    Cancelled {
        chunks_done: u64,
    },
    /// `range` outside the mapping or not whole rows.
    InvalidRange,
    Work(E),
}

/// Walks a byte range of a mapping chunk by chunk:
///
/// 1. once: `Sequential` on the whole range, `WillNeed` on the first chunks;
/// 2. before chunk `i`: `WillNeed` on chunk `i + PREFETCH_AHEAD` (chunks `i + 1 ..` were
///    already requested), then `work(chunk, first_row)`;
/// 3. after chunk `i`: asynchronous flush of chunk `i - 1`, which bounds dirty pages.
///
/// Chunks are whole rows, so the bytes written are the same as one unchunked pass whatever
/// the chunk size.
#[derive(Debug, Clone, Copy)]
pub struct ChunkedPatcher {
    pub config: IoConfig,
}

impl ChunkedPatcher {
    pub fn new(config: IoConfig) -> Self {
        Self { config }
    }

    pub fn run<E>(
        &self,
        map: &mut MmapMut,
        range: Range<usize>,
        row_bytes: usize,
        hints: &mut dyn PageHints,
        cancelled: &dyn Fn() -> bool,
        mut work: impl FnMut(&mut [u8], usize) -> Result<(), E>,
    ) -> Result<IoCounters, ChunkFailure<E>> {
        let len = range.end.saturating_sub(range.start);
        if range.start > range.end
            || range.end > map.len()
            || row_bytes == 0
            || !len.is_multiple_of(row_bytes)
        {
            return Err(ChunkFailure::InvalidRange);
        }
        let mut counters = IoCounters::default();
        if len == 0 {
            return Ok(counters);
        }
        let chunk_len = (self.config.chunk_bytes / row_bytes).max(1) * row_bytes;
        let n_chunks = len.div_ceil(chunk_len);
        let chunk = |i: usize| {
            let start = range.start + i * chunk_len;
            start..(start + chunk_len).min(range.end)
        };
        let prefetch = self.config.prefetch;
        let hint = |counters: &mut IoCounters, result: io::Result<()>| {
            counters.prefetch_requests += 1;
            if result.is_err() {
                counters.prefetch_failures += 1;
            }
        };
        if prefetch {
            hint(&mut counters, hints.sequential(map, range.clone()));
            for i in 0..PREFETCH_AHEAD.min(n_chunks) {
                hint(&mut counters, hints.will_need(map, chunk(i)));
            }
        }
        for i in 0..n_chunks {
            if cancelled() {
                return Err(ChunkFailure::Cancelled {
                    chunks_done: counters.chunks,
                });
            }
            if prefetch && i + PREFETCH_AHEAD < n_chunks {
                hint(
                    &mut counters,
                    hints.will_need(map, chunk(i + PREFETCH_AHEAD)),
                );
            }
            let current = chunk(i);
            let first_row = (current.start - range.start) / row_bytes;
            counters.bytes += current.len() as u64;
            work(&mut map[current], first_row).map_err(ChunkFailure::Work)?;
            counters.chunks += 1;
            if prefetch && i > 0 {
                counters.flush_requests += 1;
                if hints.flush_async(map, chunk(i - 1)).is_err() {
                    counters.flush_failures += 1;
                }
            }
        }
        Ok(counters)
    }
}

/// Read-only mapping of a whole file.
pub fn map_read(file: &File) -> io::Result<Mmap> {
    // SAFETY: the mapping is only read. Like every mmap user, we assume no other process
    // truncates the file while it is mapped (that would be undefined behavior).
    unsafe { MmapOptions::new().map(file) }
}

/// Writable shared mapping of a whole file (the file must be open read + write).
pub fn map_write(file: &File) -> io::Result<MmapMut> {
    // SAFETY: same assumption as `map_read`; this process is the only writer, the patch
    // pipeline holds a backup and a crash marker while the mapping lives.
    unsafe { MmapOptions::new().map_mut(file) }
}

/// Bytes available to this user on the volume holding `dir`.
pub fn available_space(dir: &Path) -> io::Result<u64> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;

        let wide: Vec<u16> = dir
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        let mut free = 0u64;
        // SAFETY: `wide` is NUL-terminated and outlives the call; `free` is a valid u64 out
        // pointer; the two other outputs are optional and passed as null.
        let ok = unsafe {
            GetDiskFreeSpaceExW(
                wide.as_ptr(),
                &mut free,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        match ok {
            0 => Err(io::Error::last_os_error()),
            _ => Ok(free),
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;

        let path = std::ffi::CString::new(dir.as_os_str().as_bytes())
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let mut st = std::mem::MaybeUninit::<libc::statvfs>::zeroed();
        // SAFETY: `path` is NUL-terminated and outlives the call; `st` is a writable,
        // correctly sized `statvfs` buffer.
        let rc = unsafe { libc::statvfs(path.as_ptr(), st.as_mut_ptr()) };
        if rc != 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `statvfs` returned 0, so it filled the whole structure (and it was
        // zero-initialised anyway).
        let st = unsafe { st.assume_init() };
        // Field widths differ per platform (u32 on macOS, u64 on Linux).
        #[allow(clippy::unnecessary_cast)]
        let free = st.f_bavail as u64 * st.f_frsize as u64;
        Ok(free)
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = dir;
        Err(io::Error::from(io::ErrorKind::Unsupported))
    }
}

/// Evicts a file from the page cache (Linux `posix_fadvise(DONTNEED)`, no privilege
/// needed), for cold benchmarks. Best effort; `Unsupported` elsewhere.
pub fn drop_file_cache(path: &Path) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;

        let file = File::open(path)?;
        // Dirty pages cannot be evicted: write them back first.
        file.sync_all()?;
        // SAFETY: the descriptor is owned by `file`, open for the whole call; offset 0 and
        // length 0 cover the whole file.
        let rc = unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        match rc {
            0 => Ok(()),
            errno => Err(io::Error::from_raw_os_error(errno)),
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = path;
        Err(io::Error::from(io::ErrorKind::Unsupported))
    }
}
