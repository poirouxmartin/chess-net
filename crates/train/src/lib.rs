//! Self-play data generation and neural training, entirely in Rust.
//!
//! Pipeline:
//! 1. `gen::generate` — self-play games with the loaded net + MCTS (parallel),
//!    records active features, legal moves + MCTS visit counts, and the game
//!    result, written to a binary data file.
//! 2. `train::train` — batch forward/backward (BCE value + policy CE) with
//!    Adam, starting from a loaded CSNN net (or a fresh init).
//! 3. `nn::write_csnn` — export the trained net for the engine.

pub mod data;
pub mod gen;
pub mod net;
pub mod train;

/// Deterministic splitmix64-ish PRNG (xorshift64*). No external deps.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_add(0x9E3779B97F4A7C15))
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in [0, n).
    #[inline]
    pub fn range(&mut self, n: u32) -> u32 {
        (self.next_u64() % n as u64) as u32
    }

    /// Standard normal via Box-Muller.
    pub fn normal(&mut self) -> f32 {
        let u1 = self.f32().max(1e-9);
        let u2 = self.f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).sin()
    }
}