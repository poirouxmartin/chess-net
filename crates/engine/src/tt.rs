//! Transposition table, fixed-size power-of-two, depth-preferred replacement.
//! Each entry is exactly 16 bytes: `u64 key + i32 score + u32 mv`. The depth
//! (offset by 128) and bound flag are packed into the free high bits of `mv`
//! (a Move only uses bits 0..21), so a probe touches a single cache line.

use std::mem::size_of;

use crate::move_::Move;

pub const FLAG_EXACT: u8 = 0;
pub const FLAG_LOWER: u8 = 1;
pub const FLAG_UPPER: u8 = 2;

/// Bits 21..29 of `mv`: depth stored as `depth + 128` (0 = -128, never usable).
const DEPTH_SHIFT: u32 = 21;
/// Bits 29..32 of `mv`: bound flag.
const FLAG_SHIFT: u32 = 29;

#[derive(Clone, Copy)]
pub struct TTEntry {
    pub key: u64,
    pub score: i32,
    /// Move (bits 0..21) plus packed depth and flag (bits 21..32).
    pub mv: u32,
}

impl TTEntry {
    fn empty() -> Self {
        TTEntry { key: 0, score: 0, mv: 0 }
    }

    #[inline(always)]
    pub fn depth(&self) -> i32 {
        ((self.mv >> DEPTH_SHIFT) & 0xFF) as i32 - 128
    }

    #[inline(always)]
    pub fn flag(&self) -> u8 {
        ((self.mv >> FLAG_SHIFT) & 0x07) as u8
    }

    #[inline(always)]
    pub fn mv_move(&self) -> Move {
        Move(self.mv & MOVE_MASK)
    }
}

/// Bits 0..21 of `mv` hold the actual move.
const MOVE_MASK: u32 = (1 << 21) - 1;

pub struct TT {
    entries: Box<[TTEntry]>,
    mask: usize,
}

impl TT {
    pub fn new(size_mb: usize) -> Self {
        let n = (size_mb * 1024 * 1024) / size_of::<TTEntry>();
        let n = n.max(1) as u64;
        let mask = (1u64 << (64 - n.leading_zeros() - 1)).max(1) - 1;
        Self {
            entries: vec![TTEntry::empty(); mask as usize + 1].into_boxed_slice(),
            mask: mask as usize,
        }
    }

    #[inline]
    pub fn probe(&self, key: u64) -> Option<&TTEntry> {
        let e = &self.entries[(key as usize) & self.mask];
        if e.key == key {
            Some(e)
        } else {
            None
        }
    }

    #[inline]
    pub fn store(&mut self, key: u64, score: i32, depth: i32, flag: u8, mv: Move) {
        let idx = (key as usize) & self.mask;
        let e = &mut self.entries[idx];
        if e.key == key || depth > e.depth() || e.depth() < 0 {
            e.key = key;
            e.score = score;
            e.mv = mv.0 | (((depth + 128) as u32) << DEPTH_SHIFT) | ((flag as u32) << FLAG_SHIFT);
        }
    }

    pub fn clear(&mut self) {
        for e in self.entries.iter_mut() {
            *e = TTEntry::empty();
        }
    }
}