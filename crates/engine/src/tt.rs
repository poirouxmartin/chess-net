//! Transposition table, fixed-size power-of-two, depth-preferred replacement.

use crate::move_::Move;

pub const FLAG_EXACT: u8 = 0;
pub const FLAG_LOWER: u8 = 1;
pub const FLAG_UPPER: u8 = 2;

#[derive(Clone, Copy)]
pub struct TTEntry {
    pub key: u64,
    pub score: i32,
    pub depth: i8,
    pub flag: u8,
    pub mv: u32,
}

impl TTEntry {
    fn empty() -> Self {
        TTEntry {
            key: 0,
            score: 0,
            depth: -128,
            flag: 0,
            mv: 0,
        }
    }
}

pub struct TT {
    entries: Box<[TTEntry]>,
    mask: usize,
}

impl TT {
    pub fn new(size_mb: usize) -> Self {
        let n = (size_mb * 1024 * 1024) / 16;
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
        if e.key == key || (depth as i8) > e.depth || e.depth < 0 {
            *e = TTEntry {
                key,
                score,
                depth: depth as i8,
                flag,
                mv: mv.0,
            };
        }
    }

    pub fn clear(&mut self) {
        for e in self.entries.iter_mut() {
            *e = TTEntry::empty();
        }
    }
}