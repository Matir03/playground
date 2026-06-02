//! Reader for Pascal Pons' Connect 4 opening book (`7x6.book`).
//!
//! The book is a fixed-size hash table mapping a position's canonical base-3
//! key (`key3`, mirror-folded) to its exact game-theoretic score. Format
//! (from <https://github.com/PascalPons/connect4>, `OpeningBook.hpp`):
//!
//! ```text
//!   byte 0: board width   (7)
//!   byte 1: board height  (6)
//!   byte 2: max stored position depth (in stones)
//!   byte 3: partial-key size in bytes (1, 2 or 4)
//!   byte 4: value size in bytes (1)
//!   byte 5: log_size; table length = next prime ≥ 2^log_size
//!   then:   <size> partial keys (little-endian), then <size> value bytes
//! ```
//!
//! A stored value `v` decodes to score `v + MIN_SCORE - 1` (MIN_SCORE = -18);
//! `v == 0` means "absent". Lookup index is `key3 % size`; the entry matches
//! only if its stored partial key equals the low bytes of `key3`.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Pons' MIN_SCORE for 7×6: `-(W*H)/2 + 3`.
const MIN_SCORE: i32 = -18;

pub struct OpeningBook {
    depth: u32,
    size: u64,
    partial_key_bytes: usize,
    keys: Vec<u8>,   // size * partial_key_bytes, little-endian per entry
    values: Vec<u8>, // size
}

impl OpeningBook {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let mut r = BufReader::new(File::open(path)?);

        let mut header = [0u8; 6];
        r.read_exact(&mut header)?;
        let [width, height, depth, partial_key_bytes, value_bytes, log_size] = header;

        if width != 7 || height != 6 {
            anyhow::bail!("unexpected board {width}x{height}, want 7x6");
        }
        if value_bytes != 1 {
            anyhow::bail!("unsupported value size {value_bytes}, want 1");
        }
        if !matches!(partial_key_bytes, 1 | 2 | 4) {
            anyhow::bail!("unsupported partial-key size {partial_key_bytes}");
        }

        let size = next_prime(1u64 << log_size);
        let pkb = partial_key_bytes as usize;

        let mut keys = vec![0u8; size as usize * pkb];
        r.read_exact(&mut keys)?;
        let mut values = vec![0u8; size as usize];
        r.read_exact(&mut values)?;

        Ok(Self { depth: depth as u32, size, partial_key_bytes: pkb, keys, values })
    }

    /// Exact score for the side to move, or `None` if the position is past the
    /// book's depth or simply not stored.
    pub fn get(&self, key3: u64, n_moves: u32) -> Option<i32> {
        if n_moves > self.depth {
            return None;
        }
        let idx = (key3 % self.size) as usize;
        let value = self.values[idx];
        if value == 0 {
            return None;
        }
        let off = idx * self.partial_key_bytes;
        let want = key3.to_le_bytes();
        if self.keys[off..off + self.partial_key_bytes] == want[..self.partial_key_bytes] {
            Some(value as i32 + MIN_SCORE - 1)
        } else {
            None
        }
    }
}

/// Smallest prime ≥ `n` (n ≥ 2). Matches the table sizing in the book format.
fn next_prime(n: u64) -> u64 {
    let mut c = n.max(2);
    while !is_prime(c) {
        c += 1;
    }
    c
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut d = 3u64;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}
