//! P12: the GGUF parser on truncated and corrupted files: a typed error, never a panic, a
//! read past the end, nor an allocation sized by an untrusted count. A counting global
//! allocator (this binary holds this single test) records the largest single allocation.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use weight_surgeon::patch::gguf;

struct Counting;

static LARGEST: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards every call to the system allocator unchanged; only records sizes.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        LARGEST.fetch_max(layout.size(), Ordering::Relaxed);
        // SAFETY: same contract as the caller's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: same contract as the caller's.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LARGEST.fetch_max(new_size, Ordering::Relaxed);
        // SAFETY: same contract as the caller's.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn put_str(b: &mut Vec<u8>, s: &str) {
    b.extend((s.len() as u64).to_le_bytes());
    b.extend(s.as_bytes());
}

/// Small GGUF v3: strings, integers, a string array, an i32 array, a nested array, and
/// three tensors (Q8_0, F16, an unpatchable Q4_0).
fn small_gguf() -> (Vec<u8>, usize) {
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend(3u64.to_le_bytes());
    b.extend(5u64.to_le_bytes());
    put_str(&mut b, "general.architecture");
    b.extend(8u32.to_le_bytes());
    put_str(&mut b, "qwen2");
    put_str(&mut b, "general.alignment");
    b.extend(4u32.to_le_bytes());
    b.extend(32u32.to_le_bytes());
    put_str(&mut b, "tokenizer.ggml.tokens");
    b.extend(9u32.to_le_bytes());
    b.extend(8u32.to_le_bytes());
    b.extend(3u64.to_le_bytes());
    for t in ["a", "bc", "def"] {
        put_str(&mut b, t);
    }
    put_str(&mut b, "ids");
    b.extend(9u32.to_le_bytes());
    b.extend(5u32.to_le_bytes());
    b.extend(2u64.to_le_bytes());
    b.extend(7i32.to_le_bytes());
    b.extend((-7i32).to_le_bytes());
    put_str(&mut b, "nested");
    b.extend(9u32.to_le_bytes());
    b.extend(9u32.to_le_bytes());
    b.extend(1u64.to_le_bytes());
    b.extend(0u32.to_le_bytes());
    b.extend(2u64.to_le_bytes());
    b.extend([1u8, 2]);
    let tensors: [(&str, u32, u64, usize); 3] = [
        ("blk.0.ffn_up.weight", 8, 64, 2 * 34),
        ("blk.0.ffn_down.weight", 1, 16, 32),
        ("blk.0.attn_q.weight", 2, 32, 18),
    ];
    let mut offset = 0u64;
    let mut sizes = Vec::new();
    for (name, ty, ne0, size) in tensors {
        put_str(&mut b, name);
        b.extend(1u32.to_le_bytes());
        b.extend(ne0.to_le_bytes());
        b.extend(ty.to_le_bytes());
        b.extend(offset.to_le_bytes());
        sizes.push((offset as usize, size));
        offset = (offset + size as u64).next_multiple_of(32);
    }
    b.resize(b.len().next_multiple_of(32), 0);
    let data_start = b.len();
    for (off, size) in sizes {
        b.resize(data_start + off, 0);
        b.extend(std::iter::repeat_n(0x11u8, size));
    }
    // The F16 tensor (offset 96, 32 bytes) is the last sized one.
    let sized_end = data_start + 128;
    (b, sized_end)
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

#[test]
fn truncations_and_corruptions_are_errors_without_panic_or_huge_allocation() {
    let (bytes, sized_end) = small_gguf();
    let file = gguf::parse(&bytes).expect("the fixture itself parses");
    assert_eq!(file.tensors.len(), 3);
    assert_eq!(
        file.tensor("blk.0.ffn_down.weight")
            .map(|t| t.start + t.size.unwrap_or(0)),
        Some(sized_end as u64)
    );

    // Every prefix: an error until the last sized tensor fits, then a valid parse.
    for n in 0..bytes.len() {
        match gguf::parse(&bytes[..n]) {
            Ok(f) => {
                assert!(n >= sized_end, "prefix {n} parsed");
                assert!(f
                    .tensors
                    .iter()
                    .all(|t| t.start + t.size.unwrap_or(0) <= n as u64));
            }
            Err(_) => assert!(n < bytes.len()),
        }
    }

    let budget = 16 * bytes.len() + (64 << 10);
    LARGEST.store(0, Ordering::Relaxed);
    let mut rng = Rng(0xF022);
    let giant = [u64::MAX, 1 << 40, 1 << 62, u32::MAX as u64 + 1];
    for i in 0..20_000 {
        let mut bad = bytes.clone();
        match i % 3 {
            // Random bytes anywhere in the header region.
            0 => {
                for _ in 0..1 + rng.next() % 4 {
                    let at = (rng.next() as usize) % file.data_start as usize;
                    bad[at] = rng.next() as u8;
                }
            }
            // A giant u64 at a random 8-byte position (counts, lengths, dims, offsets).
            1 => {
                let at = (rng.next() as usize) % (file.data_start as usize - 8);
                bad[at..at + 8].copy_from_slice(&giant[i % giant.len()].to_le_bytes());
            }
            // A random u32 at a random position (types, n_dims, versions).
            _ => {
                let at = (rng.next() as usize) % (file.data_start as usize - 4);
                bad[at..at + 4].copy_from_slice(&(rng.next() as u32).to_le_bytes());
            }
        }
        if let Ok(f) = gguf::parse(&bad) {
            assert!(f.data_start <= bad.len() as u64 || f.tensors.is_empty());
            for t in &f.tensors {
                assert!(t.start + t.size.unwrap_or(0) <= bad.len() as u64, "{t:?}");
            }
        }
    }
    let largest = LARGEST.load(Ordering::Relaxed);
    assert!(largest <= budget, "largest allocation {largest} > {budget}");
}
