//! Kernel and I/O benchmark (non-blocking: not part of the test suite).
//!
//! `cargo run --release --example bench_kernels -- [--size-mib 1024] [--file-mib 2048]
//!  [--dir <scratch dir>] [--no-file]`
//!
//! 1. In memory: throughput of every SIMD level available on this CPU x dtype x axis on a
//!    buffer of `--size-mib`, next to the `memcpy` bandwidth of the same machine (both count
//!    the bytes of the buffer, each read and written once).
//! 2. End to end: `WritableModel::shift_tensor` on a SafeTensors file of `--file-mib`, with
//!    and without prefetch, warm (file in the page cache) and, on Linux, cold
//!    (`posix_fadvise(DONTNEED)` first), with the page faults of each run.
//!
//! Prints Markdown tables. Levels the CPU lacks are listed as not run.

use std::path::{Path, PathBuf};
use std::time::Instant;

use weight_surgeon::engine::{BroadcastAxis, ShiftOp, WritableModel};
use weight_surgeon::kernel::{self, Deltas, Elem, IoConfig, KernelConfig, Level};

const COLS: usize = 4096;
const REPEATS: usize = 3;

struct Args {
    size_mib: usize,
    file_mib: usize,
    dir: PathBuf,
    file: bool,
}

fn args() -> Args {
    let mut a = Args {
        size_mib: 1024,
        file_mib: 2048,
        dir: std::env::temp_dir(),
        file: true,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let value = argv.get(i + 1).cloned().unwrap_or_default();
        match argv[i].as_str() {
            "--size-mib" => a.size_mib = value.parse().unwrap_or(a.size_mib),
            "--file-mib" => a.file_mib = value.parse().unwrap_or(a.file_mib),
            "--dir" => a.dir = PathBuf::from(value),
            "--no-file" => {
                a.file = false;
                i -= 1;
            }
            other => panic!("argument inconnu {other}"),
        }
        i += 2;
    }
    a
}

/// Deterministic bf16/f16/f32 weights around N(0, 0.02²)-ish magnitudes.
fn fill(buf: &mut [u8], elem: Elem) {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for cell in buf.chunks_exact_mut(elem.size()) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let x = ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.08;
        match elem {
            Elem::Bf16 => cell.copy_from_slice(&kernel::simd::bf16_store(x)),
            Elem::F16 => cell.copy_from_slice(&kernel::simd::f16_store(x)),
            Elem::F32 => cell.copy_from_slice(&x.to_le_bytes()),
        }
    }
}

fn mib_s(bytes: usize, seconds: f64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0) / seconds
}

fn best_of(mut run: impl FnMut()) -> f64 {
    (0..REPEATS)
        .map(|_| {
            let t = Instant::now();
            run();
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

fn system() {
    let mut sys = sysinfo::System::new().expect("sysinfo");
    sys.refresh_cpu_all();
    sys.refresh_memory();
    let brand = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_owned())
        .unwrap_or_default();
    let cores = sysinfo::System::physical_core_count().unwrap_or(0);
    println!("## Configuration\n");
    println!("| élément | valeur |\n|---|---|");
    println!("| CPU | {brand} ({} arch) |", sysinfo::System::cpu_arch());
    println!(
        "| cœurs physiques / logiques | {cores} / {} |",
        sys.cpus().len()
    );
    println!(
        "| RAM | {:.1} Gio |",
        sys.total_memory() as f64 / (1u64 << 30) as f64
    );
    println!(
        "| OS | {} |",
        sysinfo::System::long_os_version().unwrap_or_default()
    );
    println!("| threads rayon | {} |", rayon::current_num_threads());
    println!("| disque | non détecté par le banc (voir README) |");
    let levels: Vec<String> = Level::ALL
        .iter()
        .map(|l| format!("{l} {}", if l.is_available() { "oui" } else { "non" }))
        .collect();
    println!("| niveaux SIMD disponibles | {} |", levels.join(", "));
    println!("| niveau par défaut | {} |\n", kernel::from_env().level);
}

fn in_memory(size: usize) {
    use rayon::prelude::*;
    let src = vec![0x5Au8; size];
    let mut dst = vec![0u8; size];
    let single = mib_s(size, best_of(|| dst.copy_from_slice(&src)));
    // Same thread count as the kernels: the fair bandwidth reference.
    let memcpy = mib_s(
        size,
        best_of(|| {
            dst.par_chunks_mut(1 << 20)
                .zip(src.par_chunks(1 << 20))
                .for_each(|(d, s)| d.copy_from_slice(s))
        }),
    );
    drop(src);
    drop(dst);
    println!(
        "## Débit en mémoire ({} Mio, colonnes {COLS})\n",
        size >> 20
    );
    println!(
        "memcpy parallèle ({} threads) : {memcpy:.0} Mio/s ; memcpy 1 thread : {single:.0} Mio/s\n",
        rayon::current_num_threads()
    );
    println!("| niveau | dtype | axe | Mio/s | vs memcpy |\n|---|---|---|---:|---:|");
    for level in Level::ALL {
        if !level.is_available() {
            println!("| {level} | — | — | non exécuté (absent de ce CPU) | — |");
            continue;
        }
        for elem in [Elem::Bf16, Elem::F16, Elem::F32] {
            let rows = size / (COLS * elem.size());
            let mut buf = vec![0u8; rows * COLS * elem.size()];
            fill(&mut buf, elem);
            let per_col: Vec<f32> = (0..COLS).map(|c| (c as f32 - 2048.0) * 1e-6).collect();
            let per_row: Vec<f32> = (0..rows).map(|r| (r % 97) as f32 * 1e-5).collect();
            let cfg = KernelConfig::new(level);
            for (axis, deltas) in [
                ("rows", Deltas::PerCol(&per_col)),
                ("cols", Deltas::PerRow(&per_row)),
            ] {
                let t = best_of(|| {
                    kernel::shift_rows(&mut buf, elem, COLS, deltas, &cfg).expect("shift");
                });
                let rate = mib_s(buf.len(), t);
                println!(
                    "| {level} | {elem:?} | {axis} | {rate:.0} | {:.2} |",
                    rate / memcpy
                );
            }
        }
    }
    println!();
}

fn write_model(path: &Path, size: usize) -> usize {
    use std::io::Write;
    let rows = size / (COLS * 2);
    let name = "model.layers.0.mlp.down_proj.weight";
    let mut header = format!(
        "{{\"{name}\":{{\"dtype\":\"BF16\",\"shape\":[{rows},{COLS}],\"data_offsets\":[0,{}]}}}}",
        rows * COLS * 2
    );
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut file = std::io::BufWriter::new(std::fs::File::create(path).expect("create"));
    file.write_all(&(header.len() as u64).to_le_bytes())
        .expect("write");
    file.write_all(header.as_bytes()).expect("write");
    let mut chunk = vec![0u8; 64 << 20];
    fill(&mut chunk, Elem::Bf16);
    let mut left = rows * COLS * 2;
    while left > 0 {
        let n = left.min(chunk.len());
        file.write_all(&chunk[..n]).expect("write");
        left -= n;
    }
    file.flush().expect("flush");
    rows
}

fn end_to_end(dir: &Path, size: usize) {
    let path = dir.join("bench_kernels.safetensors");
    let rows = write_model(&path, size);
    let v: Vec<f32> = (0..rows).map(|r| (r % 31) as f32 * 1e-4).collect();
    let cold_possible = kernel::drop_file_cache(&path).is_ok();
    println!(
        "## Bout en bout : `shift_tensor` sur un fichier de {} Mio (bf16 [{rows}, {COLS}])\n",
        size >> 20
    );
    println!("| run | cache | prefetch | calcul s | flush s | Mio/s (total) | défauts de page (min. / maj. / total) | blocs | échecs prefetch |");
    println!("|---:|---|---|---:|---:|---:|---|---:|---:|");
    let caches: &[bool] = if cold_possible {
        &[false, true]
    } else {
        &[false]
    };
    // Alternated, twice: the first run follows the write of the file.
    let mut run = 0;
    for &cold in caches {
        for prefetch in [true, false, true, false] {
            run += 1;
            if cold {
                kernel::drop_file_cache(&path).expect("fadvise");
            }
            let mut model = WritableModel::open(&path).expect("open");
            model.set_io(
                IoConfig {
                    prefetch,
                    ..IoConfig::default()
                },
                None,
            );
            let before = kernel::page_faults();
            let t = Instant::now();
            model
                .shift_tensor(
                    "model.layers.0.mlp.down_proj.weight",
                    &v,
                    0.5,
                    ShiftOp::Subtract,
                    BroadcastAxis::Cols,
                )
                .expect("shift");
            let compute = t.elapsed().as_secs_f64();
            model.flush().expect("flush");
            let seconds = t.elapsed().as_secs_f64();
            let faults = match (before, kernel::page_faults()) {
                (Some(b), Some(a)) => {
                    let d = a.since(b);
                    let opt = |x: Option<u64>| x.map_or("—".to_owned(), |v| v.to_string());
                    format!("{} / {} / {}", opt(d.minor), opt(d.major), d.total)
                }
                _ => "—".into(),
            };
            let c = model.io_counters();
            println!(
                "| {run} | {} | {} | {compute:.2} | {:.2} | {:.0} | {faults} | {} | {} |",
                if cold { "froid" } else { "chaud" },
                if prefetch { "oui" } else { "non" },
                seconds - compute,
                mib_s(c.bytes as usize, seconds),
                c.chunks,
                c.prefetch_failures,
            );
        }
    }
    if !cold_possible {
        println!(
            "\nCache froid non mesuré : pas d'éviction du cache de pages sans privilèges sur cet OS \
             (posix_fadvise n'existe que sous Linux)."
        );
    }
    let _ = std::fs::remove_file(&path);
}

fn main() {
    let a = args();
    system();
    in_memory(a.size_mib << 20);
    if a.file {
        end_to_end(&a.dir, a.file_mib << 20);
    }
}
