//! Magic bitboards for sliding piece attack generation.
//! Magics are found at startup with a deterministic PRNG (no external crates).

use std::sync::OnceLock;

use crate::bitboard::{bit, popcount};
use crate::attack::step_sq;

pub const ROOK_BITS: u32 = 12;
pub const BISHOP_BITS: u32 = 9;
const ROOK_DIRS: [i32; 4] = [8, -8, 1, -1];
const BISHOP_DIRS: [i32; 4] = [9, 7, -7, -9];

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

pub struct Magics {
    pub rook_masks: [u64; 64],
    pub rook_magics: [u64; 64],
    pub rook_shifts: [u32; 64],
    pub rook_attacks: Vec<u64>, // 64 * 4096
    pub bishop_masks: [u64; 64],
    pub bishop_magics: [u64; 64],
    pub bishop_shifts: [u32; 64],
    pub bishop_attacks: Vec<u64>, // 64 * 512
}

fn ray(sq: usize, dir: i32) -> u64 {
    let mut b = 0u64;
    let mut cur = sq;
    while let Some(n) = step_sq(cur, dir) {
        b |= bit(n);
        cur = n;
    }
    b
}

fn ray_no_edge(sq: usize, dir: i32) -> u64 {
    let mut b = 0u64;
    let mut cur = sq;
    while let Some(n) = step_sq(cur, dir) {
        if step_sq(n, dir).is_none() {
            break;
        }
        b |= bit(n);
        cur = n;
    }
    b
}

fn sliding_attacks(sq: usize, occ: u64, dirs: &[i32]) -> u64 {
    let mut atk = 0u64;
    for &dir in dirs {
        let mut cur = sq;
        while let Some(n) = step_sq(cur, dir) {
            atk |= bit(n);
            if occ & bit(n) != 0 {
                break;
            }
            cur = n;
        }
    }
    atk
}

fn find_magic(sq: usize, mask: u64, bits: u32) -> u64 {
    let mut rng = SplitMix64::new(
        0x1BAD_5EEDu64.wrapping_add((sq as u64).wrapping_mul(0x9E3779B97F4A7C15)),
    );
    let n = 1usize << bits;
    let mut used = vec![0u64; n];
    loop {
        let magic = rng.next() & rng.next() & rng.next();
        if popcount((mask.wrapping_mul(magic)) & 0xFF00_0000_0000_0000) < 6 {
            continue;
        }
        used.fill(0);
        let mut ok = true;
        let mut sub = mask;
        loop {
            let idx = ((sub.wrapping_mul(magic)) >> (64 - bits)) as usize;
            if used[idx] != 0 && used[idx] != sub {
                ok = false;
                break;
            }
            used[idx] = sub;
            if sub == 0 {
                break;
            }
            sub = (sub - 1) & mask;
        }
        if ok {
            return magic;
        }
    }
}

impl Magics {
    pub fn build() -> Self {
        let mut m = Magics {
            rook_masks: [0; 64],
            rook_magics: [0; 64],
            rook_shifts: [0; 64],
            rook_attacks: vec![0; 64 * 4096],
            bishop_masks: [0; 64],
            bishop_magics: [0; 64],
            bishop_shifts: [0; 64],
            bishop_attacks: vec![0; 64 * 512],
        };
        for sq in 0..64 {
            let rmask = ROOK_DIRS
                .iter()
                .fold(0u64, |acc, &d| acc | ray_no_edge(sq, d));
            m.rook_masks[sq] = rmask;
            m.rook_shifts[sq] = 64 - ROOK_BITS;
            m.rook_magics[sq] = find_magic(sq, rmask, ROOK_BITS);
            let mut sub = rmask;
            loop {
                let idx = ((sub.wrapping_mul(m.rook_magics[sq])) >> m.rook_shifts[sq]) as usize;
                m.rook_attacks[sq * 4096 + idx] = sliding_attacks(sq, sub, &ROOK_DIRS);
                if sub == 0 {
                    break;
                }
                sub = (sub - 1) & rmask;
            }

            let bmask = BISHOP_DIRS
                .iter()
                .fold(0u64, |acc, &d| acc | ray_no_edge(sq, d));
            m.bishop_masks[sq] = bmask;
            m.bishop_shifts[sq] = 64 - BISHOP_BITS;
            m.bishop_magics[sq] = find_magic(sq, bmask, BISHOP_BITS);
            let mut sub = bmask;
            loop {
                let idx = ((sub.wrapping_mul(m.bishop_magics[sq])) >> m.bishop_shifts[sq]) as usize;
                m.bishop_attacks[sq * 512 + idx] = sliding_attacks(sq, sub, &BISHOP_DIRS);
                if sub == 0 {
                    break;
                }
                sub = (sub - 1) & bmask;
            }
        }
        m
    }
}

static SLIDING: OnceLock<Magics> = OnceLock::new();

pub fn init() {
    let _ = SLIDING.get_or_init(Magics::build);
}

#[inline]
pub fn rook_attacks(sq: usize, occ: u64) -> u64 {
    let m = SLIDING.get().expect("magic not initialized");
    let idx = (((occ & m.rook_masks[sq]).wrapping_mul(m.rook_magics[sq])) >> m.rook_shifts[sq])
        as usize;
    m.rook_attacks[sq * 4096 + idx]
}

#[inline]
pub fn bishop_attacks(sq: usize, occ: u64) -> u64 {
    let m = SLIDING.get().expect("magic not initialized");
    let idx = (((occ & m.bishop_masks[sq]).wrapping_mul(m.bishop_magics[sq])) >> m.bishop_shifts[sq])
        as usize;
    m.bishop_attacks[sq * 512 + idx]
}

#[inline]
pub fn queen_attacks(sq: usize, occ: u64) -> u64 {
    rook_attacks(sq, occ) | bishop_attacks(sq, occ)
}

// Keep `ray` used by attack tables; expose for tests.
pub fn full_ray(sq: usize, dir: i32) -> u64 {
    ray(sq, dir)
}