//! Square and bitboard primitives.
//! Square numbering: sq = rank * 8 + file, a1 = 0 .. h8 = 63.

#[inline(always)]
pub const fn bit(sq: usize) -> u64 {
    1u64 << sq
}

#[inline(always)]
pub const fn file_of(sq: usize) -> usize {
    sq & 7
}

#[inline(always)]
pub const fn rank_of(sq: usize) -> usize {
    sq >> 3
}

#[inline(always)]
pub const fn mirror_sq(sq: usize) -> usize {
    sq ^ 56
}

#[inline(always)]
pub fn lsb(b: u64) -> usize {
    b.trailing_zeros() as usize
}

#[inline(always)]
pub fn pop_lsb(b: &mut u64) -> usize {
    let s = b.trailing_zeros() as usize;
    *b &= *b - 1;
    s
}

#[inline(always)]
pub fn popcount(b: u64) -> u32 {
    b.count_ones()
}

#[inline(always)]
pub fn file_bb(file: usize) -> u64 {
    0x0101010101010101 << file
}

#[inline(always)]
pub fn rank_bb(rank: usize) -> u64 {
    0xFF << (rank * 8)
}

#[inline(always)]
pub fn other_color(c: usize) -> usize {
    c ^ 1
}