//! chess-net engine library.

pub mod attack;
pub mod bitboard;
pub mod evaluate;
pub mod magic;
pub mod move_;
pub mod movegen;
pub mod perft;
pub mod position;
pub mod search;
pub mod tt;
pub mod uci;

/// Initialize all runtime tables (magics, attacks, zobrist). Call once at startup.
pub fn init() {
    magic::init();
    attack::init();
    position::init();
}