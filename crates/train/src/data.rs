//! Binary dataset format for self-play positions.
//!
//! Layout (all little-endian):
//!   magic  "CHSP" (4 bytes)
//!   u32    version (=1)
//!   u32    count
//!   per sample:
//!     u32    n_active,  u32[n_active]   active feature indices
//!     u32    n_moves,   u32[n_moves]    raw `Move` values (all legal moves)
//!                u32[n_moves]           MCTS visit counts (aligned)
//!     f32    result                     white POV win probability in [0, 1]

use std::io::Read;

use engine::move_::Move;

pub const MAX_ACTIVE: usize = 96;

#[derive(Clone)]
pub struct Sample {
    pub active: Vec<u32>,
    pub moves: Vec<Move>,
    pub visits: Vec<u32>,
    pub result: f32,
}

pub fn write_samples(path: &str, samples: &[Sample]) -> Result<(), String> {
    let mut out = Vec::with_capacity(16 + samples.len() * 400);
    out.extend_from_slice(b"CHSP");
    for v in [1u32, samples.len() as u32] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for s in samples {
        out.extend_from_slice(&(s.active.len() as u32).to_le_bytes());
        for f in &s.active {
            out.extend_from_slice(&f.to_le_bytes());
        }
        out.extend_from_slice(&(s.moves.len() as u32).to_le_bytes());
        for mv in &s.moves {
            out.extend_from_slice(&mv.0.to_le_bytes());
        }
        for v in &s.visits {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&s.result.to_le_bytes());
    }
    std::fs::write(path, &out).map_err(|e| e.to_string())
}

pub fn read_samples(path: &str) -> Result<Vec<Sample>, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let mut r = &bytes[..];
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != b"CHSP" {
        return Err(format!("not a CHSP dataset: {}", path));
    }
    let version = read_u32(&mut r)?;
    if version != 1 {
        return Err(format!("unsupported CHSP version: {}", version));
    }
    let count = read_u32(&mut r)?;
    if count > 10_000_000 {
        return Err(format!("absurd sample count: {}", count));
    }
    let mut samples = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let n_active = read_u32(&mut r)? as usize;
        if n_active > MAX_ACTIVE {
            return Err(format!("bad n_active: {}", n_active));
        }
        let mut active = vec![0u32; n_active];
        for a in active.iter_mut() {
            *a = read_u32(&mut r)?;
        }
        let n_moves = read_u32(&mut r)? as usize;
        if n_moves > 256 {
            return Err(format!("bad n_moves: {}", n_moves));
        }
        let mut moves = Vec::with_capacity(n_moves);
        for _ in 0..n_moves {
            moves.push(Move(read_u32(&mut r)?));
        }
        let mut visits = vec![0u32; n_moves];
        for v in visits.iter_mut() {
            *v = read_u32(&mut r)?;
        }
        let result = read_f32(&mut r)?;
        samples.push(Sample { active, moves, visits, result });
    }
    Ok(samples)
}

fn read_u32(r: &mut &[u8]) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(b))
}

fn read_f32(r: &mut &[u8]) -> Result<f32, String> {
    Ok(f32::from_bits(read_u32(r)?))
}

/// Aggregate visit counts by policy index (from*64+to), collapsing e.g.
/// promotion moves onto a single index, like the Python trainer.
pub fn policy_target(moves: &[Move], visits: &[u32]) -> Vec<(usize, u32)> {
    let mut out: Vec<(usize, u32)> = Vec::new();
    for (mv, v) in moves.iter().zip(visits) {
        let idx = mv.from() * 64 + mv.to();
        match out.iter_mut().find(|(i, _)| *i == idx) {
            Some((_, acc)) => *acc += *v,
            None => out.push((idx, *v)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let samples = vec![
            Sample {
                active: vec![1, 5, 300],
                moves: vec![Move(3), Move(77)],
                visits: vec![12, 4],
                result: 1.0,
            },
            Sample {
                active: vec![0],
                moves: vec![],
                visits: vec![],
                result: 0.0,
            },
        ];
        let dir = std::env::temp_dir().join("chsp_roundtrip.bin");
        let path = dir.to_str().unwrap();
        write_samples(path, &samples).unwrap();
        let got = read_samples(path).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].active, vec![1, 5, 300]);
        assert_eq!(got[0].moves, vec![Move(3), Move(77)]);
        assert_eq!(got[0].visits, vec![12, 4]);
        assert_eq!(got[0].result, 1.0);
        assert!(got[1].moves.is_empty());
    }

    #[test]
    fn target_collapses_promotions() {
        let moves = vec![Move(0), Move(64), Move(0), Move(1)];
        let visits = vec![5, 7, 2, 9];
        let t = policy_target(&moves, &visits);
        assert_eq!(t, vec![(0, 7), (1, 7), (64, 9)]);
    }
}