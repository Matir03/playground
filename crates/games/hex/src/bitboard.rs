//! A 169-bit board as three `u64` words — go9/amazons use `u128`, but 13×13
//! Hex needs 169 bits, so we spell out the carry between words.
//!
//! Bit `i = r*13 + c`, row-major, same convention as the other games. Only the
//! ops the rules need are here (or/and/andnot, shifts by < 64, popcount, bit
//! iteration); everything is `#[inline]` and branch-free so move generation and
//! the connectivity flood-fill stay in registers.

use crate::game::{N, NN};

pub const WORDS: usize = 3;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Bb(pub [u64; WORDS]);

/// Low `NN` bits set — the valid board area.
pub const BOARD_MASK: Bb = {
    let mut w = [u64::MAX; WORDS];
    w[WORDS - 1] = (1u64 << (NN - 64 * (WORDS - 1))) - 1;
    Bb(w)
};

impl Bb {
    pub const ZERO: Bb = Bb([0; WORDS]);

    #[inline]
    pub const fn bit(i: usize) -> Bb {
        let mut w = [0u64; WORDS];
        w[i / 64] = 1u64 << (i % 64);
        Bb(w)
    }

    #[inline]
    pub const fn cell(r: usize, c: usize) -> Bb {
        Bb::bit(r * N + c)
    }

    #[inline]
    pub fn get(self, i: usize) -> bool {
        (self.0[i / 64] >> (i % 64)) & 1 == 1
    }

    #[inline]
    pub fn set(&mut self, i: usize) {
        self.0[i / 64] |= 1u64 << (i % 64);
    }

    #[inline]
    pub fn clear(&mut self, i: usize) {
        self.0[i / 64] &= !(1u64 << (i % 64));
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        (self.0[0] | self.0[1] | self.0[2]) == 0
    }

    #[inline]
    pub fn any(self) -> bool {
        !self.is_empty()
    }

    #[inline]
    pub fn count(self) -> u32 {
        self.0[0].count_ones() + self.0[1].count_ones() + self.0[2].count_ones()
    }

    #[inline]
    pub fn or(self, o: Bb) -> Bb {
        Bb([self.0[0] | o.0[0], self.0[1] | o.0[1], self.0[2] | o.0[2]])
    }

    #[inline]
    pub fn and(self, o: Bb) -> Bb {
        Bb([self.0[0] & o.0[0], self.0[1] & o.0[1], self.0[2] & o.0[2]])
    }

    /// `self & !o`.
    #[inline]
    pub fn andnot(self, o: Bb) -> Bb {
        Bb([
            self.0[0] & !o.0[0],
            self.0[1] & !o.0[1],
            self.0[2] & !o.0[2],
        ])
    }

    /// Complement, masked to the board.
    #[inline]
    pub fn complement(self) -> Bb {
        BOARD_MASK.andnot(self)
    }

    /// Shift towards higher indices by `k` (0 < k < 64), masked to the board.
    #[inline]
    pub fn shift_up(self, k: u32) -> Bb {
        debug_assert!(k > 0 && k < 64);
        let w = self.0;
        Bb([
            w[0] << k,
            (w[1] << k) | (w[0] >> (64 - k)),
            (w[2] << k) | (w[1] >> (64 - k)),
        ])
        .and(BOARD_MASK)
    }

    /// Shift towards lower indices by `k` (0 < k < 64).
    #[inline]
    pub fn shift_down(self, k: u32) -> Bb {
        debug_assert!(k > 0 && k < 64);
        let w = self.0;
        Bb([
            (w[0] >> k) | (w[1] << (64 - k)),
            (w[1] >> k) | (w[2] << (64 - k)),
            w[2] >> k,
        ])
    }

    /// Indices of the set bits, ascending.
    #[inline]
    pub fn iter(self) -> BbIter {
        BbIter { w: self.0, i: 0 }
    }
}

pub struct BbIter {
    w: [u64; WORDS],
    i: usize,
}

impl Iterator for BbIter {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        while self.i < WORDS {
            let word = self.w[self.i];
            if word != 0 {
                let b = word.trailing_zeros() as usize;
                self.w[self.i] &= word - 1;
                return Some(self.i * 64 + b);
            }
            self.i += 1;
        }
        None
    }
}

impl std::fmt::Debug for Bb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Bb[")?;
        for (n, i) in self.iter().enumerate() {
            if n > 0 {
                write!(f, " ")?;
            }
            write!(f, "{}{}", (b'a' + (i % N) as u8) as char, i / N + 1)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn board_mask_covers_exactly_the_board() {
        assert_eq!(BOARD_MASK.count(), NN as u32);
        assert!(BOARD_MASK.get(NN - 1));
        assert_eq!(BOARD_MASK.0[2] >> (NN - 128), 0, "no bits above the board");
    }

    #[test]
    fn shifts_carry_across_words() {
        // Bit 63 → 64 and back: the interesting boundary between words.
        let b = Bb::bit(63);
        assert_eq!(b.shift_up(1), Bb::bit(64));
        assert_eq!(Bb::bit(64).shift_down(1), b);
        // A shift that would leave the board is masked off.
        assert!(Bb::bit(NN - 1).shift_up(13).is_empty());
    }

    #[test]
    fn iter_yields_set_bits_in_order() {
        let mut b = Bb::ZERO;
        for i in [0usize, 5, 63, 64, 130, 168] {
            b.set(i);
        }
        assert_eq!(b.iter().collect::<Vec<_>>(), vec![0, 5, 63, 64, 130, 168]);
        assert_eq!(b.count(), 6);
    }
}
