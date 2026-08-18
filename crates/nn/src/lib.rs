//! Neural network inference for the engine.
//!
//! Loads models exported by the Python trainer (see `python/chessnet/export.py`)
//! in the CSNN binary format and evaluates positions.
//!
//! Supported architectures:
//!   feat=1  HalfKP (NNUE-style): 2 * 41024 sparse one-hot features
//!   feat=2  KP768:               768 sparse one-hot features
//! Both are 3-layer MLPs: sparse L0 (l0) -> ReLU -> L1 (l1) -> ReLU -> logit.

use std::sync::OnceLock;

use engine::position::{Position, PAWN, BISHOP, KNIGHT, QUEEN, ROOK};

pub const FEAT_HALFKP: u32 = 1;
pub const FEAT_KP768: u32 = 2;

const HALFKP_PER_KING: usize = 641;
const MAX_ACTIVE: usize = 96;
const MAX_L0: usize = 1024;

pub struct Net {
    pub feat: u32,
    pub l0: usize,
    pub l1: usize,
    pub feat_count: usize,
    pub fw: Vec<f32>, // [feat_count][l0]
    pub fb: Vec<f32>, // l0
    pub w1: Vec<f32>, // [l1][l0]
    pub b1: Vec<f32>, // l1
    pub wo: Vec<f32>, // l1
    pub bo: f32,
}

static LOADED: OnceLock<Net> = OnceLock::new();

/// Load a CSNN model file. Call once per EvalFile; panics if called twice.
pub fn load(path: &str) -> Result<(), String> {
    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    let net = parse(&data)?;
    let _ = LOADED.set(net);
    Ok(())
}

pub fn is_loaded() -> bool {
    LOADED.get().is_some()
}

fn parse(data: &[u8]) -> Result<Net, String> {
    if data.len() < 24 || &data[0..4] != b"CSNN" {
        return Err("bad magic".into());
    }
    let u32_at = |i: usize| -> u32 {
        u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]])
    };
    let version = u32_at(4);
    if version != 1 {
        return Err(format!("unsupported version {version}"));
    }
    let feat = u32_at(8);
    let l0 = u32_at(12) as usize;
    let l1 = u32_at(16) as usize;
    let feat_count = u32_at(20) as usize;
    if l0 > MAX_L0 || l1 > MAX_L0 {
        return Err("hidden layer too large".into());
    }

    let mut pos = 24usize;
    let fw = read_f32s(data, &mut pos, feat_count * l0)?;
    let fb = read_f32s(data, &mut pos, l0)?;
    let w1 = read_f32s(data, &mut pos, l1 * l0)?;
    let b1 = read_f32s(data, &mut pos, l1)?;
    let wo = read_f32s(data, &mut pos, l1)?;
    let bo = read_f32s(data, &mut pos, 1)?[0];

    Ok(Net {
        feat,
        l0,
        l1,
        feat_count,
        fw,
        fb,
        w1,
        b1,
        wo,
        bo,
    })
}

fn read_f32s(data: &[u8], pos: &mut usize, n: usize) -> Result<Vec<f32>, String> {
    if *pos + n * 4 > data.len() {
        return Err("truncated file".into());
    }
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let off = *pos + i * 4;
        v.push(f32::from_le_bytes([
            data[off],
            data[off + 1],
            data[off + 2],
            data[off + 3],
        ]));
    }
    *pos += n * 4;
    Ok(v)
}

/// Encode active feature indices for a position.
fn encode_features(net: &Net, pos: &Position, out: &mut [usize]) -> usize {
    let mut n = 0;
    match net.feat {
        FEAT_HALFKP => {
            for p in 0..2 {
                let k = pos.king_sq[p];
                let base = k * HALFKP_PER_KING;
                for (li, pt) in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN].iter().enumerate() {
                    let mut b = pos.piece_bb(p, *pt);
                    while b != 0 {
                        let sq = engine::bitboard::pop_lsb(&mut b);
                        out[n] = base + li * 64 + sq;
                        n += 1;
                    }
                }
                let them = p ^ 1;
                for (li, pt) in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN].iter().enumerate() {
                    let mut b = pos.piece_bb(them, *pt);
                    while b != 0 {
                        let sq = engine::bitboard::pop_lsb(&mut b);
                        out[n] = base + (5 + li) * 64 + sq;
                        n += 1;
                    }
                }
                out[n] = base + HALFKP_PER_KING - 1;
                n += 1;
            }
        }
        FEAT_KP768 => {
            for c in 0..2 {
                for pt in 0..6 {
                    let mut b = pos.piece_bb(c, pt);
                    while b != 0 {
                        let sq = engine::bitboard::pop_lsb(&mut b);
                        out[n] = c * 6 * 64 + pt * 64 + sq;
                        n += 1;
                    }
                }
            }
        }
        _ => {}
    }
    n
}

/// Evaluate the loaded model from the side-to-move perspective, in centipawns.
pub fn evaluate_loaded(pos: &Position) -> i32 {
    let net = LOADED.get().expect("no model loaded");
    let mut feats = [0usize; MAX_ACTIVE];
    let n = encode_features(net, pos, &mut feats);

    let mut acc = [0f32; MAX_L0];
    acc[..net.l0].copy_from_slice(&net.fb);
    for &f in &feats[..n] {
        let base = f * net.l0;
        for j in 0..net.l0 {
            acc[j] += net.fw[base + j];
        }
    }

    let mut h1 = [0f32; MAX_L0];
    for j in 0..net.l1 {
        let mut s = net.b1[j];
        let row = j * net.l0;
        for i in 0..net.l0 {
            s += net.w1[row + i] * relu(acc[i]);
        }
        h1[j] = relu(s);
    }

    let mut out = net.bo;
    for j in 0..net.l1 {
        out += net.wo[j] * h1[j];
    }
    (out * 400.0).round() as i32
}

#[inline(always)]
fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::position::Position;

    #[test]
    fn kp768_feature_count_matches_piece_count() {
        engine::init();
        let net = Net {
            feat: FEAT_KP768,
            l0: 8,
            l1: 4,
            feat_count: 768,
            fw: vec![0.0; 768 * 8],
            fb: vec![0.0; 8],
            w1: vec![0.0; 4 * 8],
            b1: vec![0.0; 4],
            wo: vec![0.0; 4],
            bo: 0.0,
        };
        let pos = Position::startpos();
        let mut feats = [0usize; MAX_ACTIVE];
        let n = encode_features(&net, &pos, &mut feats);
        assert_eq!(n, 32); // 16 pieces per side * 2 colors
        // No duplicate features in startpos.
        let mut sorted = feats[..n].to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), n);
    }
}