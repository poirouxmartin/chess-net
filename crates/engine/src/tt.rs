//! Transposition table, fixed-size power-of-two, depth-preferred replacement.
//! Each entry is exactly 16 bytes: `u64 key + i32 score + u32 mv`. The depth
//! (offset by 128) and bound flag are packed into the free high bits of `mv`
//! (a Move only uses bits 0..21), so a probe touches a single cache line.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::move_::Move;

pub const FLAG_EXACT: u8 = 0;
pub const FLAG_LOWER: u8 = 1;
pub const FLAG_UPPER: u8 = 2;

/// Bits 21..29 of `mv`: depth stored as `depth + 128` (0 = -128, never usable).
const DEPTH_SHIFT: u32 = 21;
/// Bits 29..32 of `mv`: bound flag.
const FLAG_SHIFT: u32 = 29;

/// A single TT entry packed into 16 bytes for lock-free concurrent access.
/// key (64 bit) is stored separately; the mv field packs move + depth + flag.
struct PackedEntry {
    key: AtomicU64,
    mv: AtomicU64,
}

impl PackedEntry {
    fn empty() -> Self {
        PackedEntry { key: AtomicU64::new(0), mv: AtomicU64::new(0) }
    }
}

pub struct TT {
    entries: Box<[PackedEntry]>,
    mask: usize,
}

impl TT {
    pub fn new(size_mb: usize) -> Self {
        let n = (size_mb * 1024 * 1024) / 16; // 16 bytes per entry
        let n = n.max(1) as u64;
        let mask = (1u64 << (64 - n.leading_zeros() - 1)).max(1) - 1;
        Self {
            entries: (0..=mask).map(|_| PackedEntry::empty()).collect::<Vec<_>>().into_boxed_slice(),
            mask: mask as usize,
        }
    }

    #[inline]
    pub fn probe(&self, key: u64) -> Option<(i32, Move, u8, i32)> {
        let idx = (key as usize) & self.mask;
        let e = &self.entries[idx];
        // Acquire pairs with the Release key-store below: a matching key
        // implies the data store (sequenced before it) is visible.
        let k = e.key.load(Ordering::Acquire);
        if k == key {
            let mv = e.mv.load(Ordering::Relaxed);
            // Re-verify: a concurrent store could have swapped the slot
            // between the two loads (torn read). Discard on any change.
            if e.key.load(Ordering::Acquire) != key {
                return None;
            }
            let score = (mv >> 32) as i32;
            let raw_mv = (mv & 0x1F_FFFF) as u32;
            let depth = ((mv >> 21) & 0xFF) as i32 - 128;
            let flag = ((mv >> 29) & 0x07) as u8;
            Some((score, Move(raw_mv), flag, depth))
        } else {
            None
        }
    }

    #[inline]
    pub fn store(&self, key: u64, score: i32, depth: i32, flag: u8, mv: Move) {
        let idx = (key as usize) & self.mask;
        let e = &self.entries[idx];
        let old_mv = e.mv.load(Ordering::Relaxed);
        let old_depth = ((old_mv >> 21) & 0xFF) as i32 - 128;
        let new_mv = (mv.0 as u64)
            | (((depth + 128) as u64) << 21)
            | ((flag as u64) << 29)
            | ((score as u32 as u64) << 32);
        // Always replace on key match, deeper entry, or empty slot.
        if e.key.load(Ordering::Relaxed) == key || depth > old_depth || old_depth < 0 {
            // Data first, key last with Release: readers that Acquire-load
            // this key are guaranteed to see the matching data.
            e.mv.store(new_mv, Ordering::Relaxed);
            e.key.store(key, Ordering::Release);
        }
    }

    pub fn clear(&self) {
        for e in self.entries.iter() {
            e.key.store(0, Ordering::Relaxed);
            e.mv.store(0, Ordering::Relaxed);
        }
    }
}