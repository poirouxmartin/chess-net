//! Neural network inference for the engine.
//!
//! Loads models exported by the Python trainer (see `python/chessnet/export.py`)
//! in the CSNN binary format and evaluates positions.
//!
//! Supported architectures:
//!   feat=1  HalfKP (NNUE-style): 2 * 41024 sparse one-hot features
//!   feat=2  KP768:               768 sparse one-hot features
//! Both are 3-layer MLPs: sparse L0 (l0) -> ReLU -> L1 (l1) -> ReLU -> logit.
//! CSNN v1 is value-only; v2 adds a policy head sharing h1: one logit per
//! from*64+to move (4096) that the MCTS can use as move priors.

use std::sync::OnceLock;

use engine::movegen::MoveList;
use engine::position::{Position, PAWN, BISHOP, KNIGHT, QUEEN, ROOK};

pub const FEAT_HALFKP: u32 = 1;
pub const FEAT_KP768: u32 = 2;

const HALFKP_PER_KING: usize = 641;
const MAX_ACTIVE: usize = 96;
const MAX_L0: usize = 1024;
const MAX_POLICY: usize = 1 << 16;
/// Policy output space the engine can consume: one logit per from*64+to move.
pub const POLICY_MOVES: usize = 4096;

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
    /// Policy head (CSNN v2): logits = bp + wp . h1, one per from*64+to move.
    pub policy_size: usize,
    pub wp: Vec<f32>, // [policy_size][l1]
    pub bp: Vec<f32>, // policy_size
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

/// The currently loaded net, if any.
pub fn loaded() -> Option<&'static Net> {
    LOADED.get()
}

/// Feature kind of the currently loaded net, if any.
pub fn loaded_feat() -> Option<u32> {
    LOADED.get().map(|n| n.feat)
}

/// Write `net` to a CSNN file: version 1 (value only) when there is no policy
/// head, version 2 otherwise. Format matches the Python `csnn.py` writer.
pub fn write_csnn(net: &Net, path: &str) -> Result<(), String> {
    let mut out = Vec::with_capacity(24 + (net.feat_count * net.l0 + net.l0 + net.l1 * net.l0
        + net.l1 + net.l1 + 1) * 4);
    out.extend_from_slice(b"CSNN");
    let version: u32 = if net.policy_size == 0 { 1 } else { 2 };
    for v in [version, net.feat, net.l0 as u32, net.l1 as u32, net.feat_count as u32] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for arr in [
        net.fw.as_slice(),
        net.fb.as_slice(),
        net.w1.as_slice(),
        net.b1.as_slice(),
        net.wo.as_slice(),
        std::slice::from_ref(&net.bo),
    ] {
        for f in arr {
            out.extend_from_slice(&f.to_le_bytes());
        }
    }
    if net.policy_size != 0 {
        out.extend_from_slice(&(net.policy_size as u32).to_le_bytes());
        for f in &net.wp {
            out.extend_from_slice(&f.to_le_bytes());
        }
        for f in &net.bp {
            out.extend_from_slice(&f.to_le_bytes());
        }
    }
    std::fs::write(path, &out).map_err(|e| e.to_string())
}

fn parse(data: &[u8]) -> Result<Net, String> {
    if data.len() < 24 || &data[0..4] != b"CSNN" {
        return Err("bad magic".into());
    }
    let u32_at = |i: usize| -> u32 {
        u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]])
    };
    let version = u32_at(4);
    if version != 1 && version != 2 {
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

    let mut policy_size = 0usize;
    let mut wp = Vec::new();
    let mut bp = Vec::new();
    if version == 2 {
        if pos + 4 > data.len() {
            return Err("truncated file".into());
        }
        policy_size = u32_at(pos) as usize;
        pos += 4;
        if policy_size > MAX_POLICY {
            return Err("policy too large".into());
        }
        wp = read_f32s(data, &mut pos, policy_size * l1)?;
        bp = read_f32s(data, &mut pos, policy_size)?;
    }

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
        policy_size,
        wp,
        bp,
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

/// Encode active feature indices for a position (KP768: c*6*64 + pt*64 + sq).
/// Returns the number of active features written to `out`.
pub fn active_features(feat: u32, pos: &Position, out: &mut [usize]) -> usize {
    let mut n = 0;
    match feat {
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

/// Accumulates the sparse L0 features of `pos` and the shared ReLU hidden
/// layer h1 = relu(b1 + w1 . relu(acc)). Both the value head and the policy
/// head consume `h1`.
fn forward_h1(net: &Net, pos: &Position) -> [f32; MAX_L0] {
    let mut feats = [0usize; MAX_ACTIVE];
    let n = active_features(net.feat, pos, &mut feats);

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
    h1
}

/// Evaluate the loaded model from WHITE's perspective, in centipawns.
///
/// The features are side-agnostic (absolute colors), so the net's raw output
/// is the white win prob/logit. The search needs side-to-move scores and must
/// use [`evaluate_loaded_stm`] instead.
pub fn evaluate_loaded(pos: &Position) -> i32 {
    let net = LOADED.get().expect("no model loaded");
    let h1 = forward_h1(net, pos);

    let mut out = net.bo;
    for j in 0..net.l1 {
        out += net.wo[j] * h1[j];
    }
    (out * 400.0).round() as i32
}

/// Combined value (centipawns, side-to-move) and move-policy eval for MCTS.
/// Both heads share the same hidden layer, so a single forward is used.
/// `logits` has one entry per move in `legal` (empty when no policy head).
pub fn evaluate_loaded_combined(pos: &Position, legal: &MoveList) -> (i32, Vec<f32>) {
    let net = LOADED.get().expect("no model loaded");
    let h1 = forward_h1(net, pos);

    let mut out = net.bo;
    for j in 0..net.l1 {
        out += net.wo[j] * h1[j];
    }
    let cp = (out * 400.0).round() as i32;
    let cp = if pos.side == 0 { cp } else { -cp };

    let logits = if net.policy_size == POLICY_MOVES {
        policy_logits_from_h1(net, &h1, legal)
    } else {
        Vec::new()
    };
    (cp, logits)
}

/// Policy logits for an explicit net and hidden layer (testable without the
/// OnceLock). Empty when the net has no policy head.
fn policy_logits_from_h1(net: &Net, h1: &[f32], legal: &MoveList) -> Vec<f32> {
    let mut out = Vec::with_capacity(legal.len);
    for i in 0..legal.len {
        let m = legal.get(i);
        let idx = m.from() * 64 + m.to();
        out.push(policy_logit(net, h1, idx));
    }
    out
}

/// One policy logit: bp[idx] + wp[idx] . h1, AVX2-accelerated.
#[inline]
fn policy_logit(net: &Net, h1: &[f32], idx: usize) -> f32 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: checked the CPU supports avx2+fma above.
        unsafe { policy_logit_avx2(net, h1, idx) }
    } else {
        policy_logit_scalar(net, h1, idx)
    }
}

#[inline]
fn policy_logit_scalar(net: &Net, h1: &[f32], idx: usize) -> f32 {
    let row = idx * net.l1;
    let mut s = net.bp[idx];
    for j in 0..net.l1 {
        s += net.wp[row + j] * h1[j];
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn policy_logit_avx2(net: &Net, h1: &[f32], idx: usize) -> f32 {
    use std::arch::x86_64::*;
    let row = idx * net.l1;
    let mut acc = _mm256_setzero_ps();
    let mut j = 0;
    while j + 8 <= net.l1 {
        let w = _mm256_loadu_ps(net.wp.as_ptr().add(row + j));
        let h = _mm256_loadu_ps(h1.as_ptr().add(j));
        acc = _mm256_fmadd_ps(w, h, acc);
        j += 8;
    }
    let mut s = net.bp[idx];
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum = _mm_add_ps(lo, hi);
    let sum = _mm_hadd_ps(sum, sum);
    let sum = _mm_hadd_ps(sum, sum);
    s += _mm_cvtss_f32(sum);
    for k in j..net.l1 {
        s += net.wp[row + k] * h1[k];
    }
    s
}

/// Evaluate from the side to move's perspective (what the alpha-beta search
/// expects): white's score when it is white to move, negated for black.
pub fn evaluate_loaded_stm(pos: &Position) -> i32 {
    let s = evaluate_loaded(pos);
    if pos.side == 0 {
        s
    } else {
        -s
    }
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
            policy_size: 0,
            wp: Vec::new(),
            bp: Vec::new(),
        };
        let pos = Position::startpos();
        let mut feats = [0usize; MAX_ACTIVE];
        let n = active_features(net.feat, &pos, &mut feats);
        assert_eq!(n, 32); // 16 pieces per side * 2 colors
        // No duplicate features in startpos.
        let mut sorted = feats[..n].to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), n);
    }

    #[test]
    fn parse_v2_with_policy() {
        let l0 = 8usize;
        let l1 = 4usize;
        let feat_count = 768usize;
        let policy_size = 16usize;
        let mut data = Vec::new();
        data.extend_from_slice(b"CSNN");
        for v in [2u32, FEAT_KP768, l0 as u32, l1 as u32, feat_count as u32] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        for n in [feat_count * l0, l0, l1 * l0, l1, l1, 1] {
            for _ in 0..n {
                data.extend_from_slice(&0f32.to_le_bytes());
            }
        }
        data.extend_from_slice(&(policy_size as u32).to_le_bytes());
        for i in 0..policy_size * l1 {
            data.extend_from_slice(&((i as f32) * 0.5).to_le_bytes());
        }
        for i in 0..policy_size {
            data.extend_from_slice(&((i as f32) + 1.0).to_le_bytes());
        }
        let net = parse(&data).unwrap();
        assert_eq!(net.policy_size, policy_size);
        assert_eq!(net.wp.len(), policy_size * l1);
        assert_eq!(net.bp.len(), policy_size);
        assert_eq!(net.bp[3], 4.0);
        assert_eq!(net.wp[1], 0.5);
        assert_eq!(net.bo, 0.0);
    }

    #[test]
    fn policy_eval_matches_bp_with_zero_hidden() {
        engine::init();
        let l0 = 8usize;
        let l1 = 4usize;
        let net = Net {
            feat: FEAT_KP768,
            l0,
            l1,
            feat_count: 768,
            fw: vec![0.0; 768 * l0],
            fb: vec![0.0; l0],
            w1: vec![0.0; l1 * l0],
            b1: vec![0.0; l1],
            wo: vec![0.0; l1],
            bo: 0.0,
            policy_size: POLICY_MOVES,
            wp: vec![0.0; POLICY_MOVES * l1],
            bp: vec![0.0; POLICY_MOVES],
        };
        let pos = Position::startpos();
        let legal = engine::movegen::generate_legal(&pos);
        let mut bp = vec![0.0; POLICY_MOVES];
        let m = legal.get(0);
        bp[m.from() * 64 + m.to()] = 1.5;
        let net = Net { bp, ..net };
        let h1 = forward_h1(&net, &pos);
        let logits = policy_logits_from_h1(&net, &h1, &legal);
        assert_eq!(logits.len(), legal.len);
        assert!((logits[0] - 1.5).abs() < 1e-6);
        assert!(logits[1..].iter().all(|x| x.abs() < 1e-6));
    }

    #[test]
    fn policy_logit_avx2_matches_scalar() {
        let l1 = 32usize;
        let policy_size = POLICY_MOVES;
        let wp: Vec<f32> = (0..policy_size * l1).map(|i| ((i as f32) * 0.37) % 1.0 - 0.5).collect();
        let bp: Vec<f32> = (0..policy_size).map(|i| ((i as f32) * 0.13) % 0.5).collect();
        let h1: Vec<f32> = (0..l1).map(|j| ((j as f32) * 0.7) % 1.0).collect();
        let net = Net {
            feat: FEAT_KP768,
            l0: 8,
            l1,
            feat_count: 768,
            fw: Vec::new(),
            fb: Vec::new(),
            w1: Vec::new(),
            b1: Vec::new(),
            wo: Vec::new(),
            bo: 0.0,
            policy_size,
            wp,
            bp,
        };
        for idx in [0usize, 1, 63, 64, 795, 796, 2048, 4095] {
            let scalar = policy_logit_scalar(&net, &h1, idx);
            let simd = policy_logit(&net, &h1, idx);
            assert!((scalar - simd).abs() < 1e-3, "idx {idx}: scalar {scalar} vs simd {simd}");
        }
    }

    #[test]
    fn mcts_uses_loaded_policy_priors() {
        use engine::mcts::{mcts_root, MctsLimits};
        use std::sync::atomic::AtomicBool;

        engine::init();
        // All-zero net (value 0 everywhere) except bp biased toward e2e4, so the
        // search must be driven by the policy head: e2e4 becomes most-visited.
        let l0 = 8usize;
        let l1 = 4usize;
        let feat_count = 768usize;
        let policy_size = POLICY_MOVES;
        let mut bp = vec![0.0f32; policy_size];
        let e2e4 = 12 * 64 + 28; // e2=12, e4=28
        bp[e2e4] = 5.0;

        let mut data = Vec::new();
        data.extend_from_slice(b"CSNN");
        for v in [2u32, FEAT_KP768, l0 as u32, l1 as u32, feat_count as u32] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        for n in [feat_count * l0, l0, l1 * l0, l1, l1, 1] {
            for _ in 0..n {
                data.extend_from_slice(&0f32.to_le_bytes());
            }
        }
        data.extend_from_slice(&(policy_size as u32).to_le_bytes());
        for _ in 0..policy_size * l1 {
            data.extend_from_slice(&0f32.to_le_bytes());
        }
        for b in &bp {
            data.extend_from_slice(&b.to_le_bytes());
        }
        let path = std::env::temp_dir().join("mcts_policy_test.csnn");
        std::fs::write(&path, &data).unwrap();
        load(&path.to_string_lossy()).unwrap();

        let mut pos = Position::startpos();
        let result = mcts_root(
            &mut pos,
            &MctsLimits { playouts: Some(300), movetime: None, threads: 1 },
            &AtomicBool::new(false),
            evaluate_loaded_combined,
            None,
        );
        assert_eq!(result.best.to_uci(), "e2e4", "policy must steer MCTS to e2e4");
        let _ = std::fs::remove_file(&path);
    }
}