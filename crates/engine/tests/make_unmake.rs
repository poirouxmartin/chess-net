//! Verify make_move/unmake_move restores the exact position (bitboards, key).

use engine::movegen::generate_legal;
use engine::position::Position;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

fn random_roundtrip(fen: &str, games: usize, plies: usize) {
    engine::init();
    let mut rng = Lcg(0x5EED);
    for _ in 0..games {
        let mut pos = Position::from_fen(fen);
        for p in 0..plies {
            let moves = generate_legal(&pos);
            if moves.len == 0 {
                break;
            }
            let m = moves.get((rng.next() as usize) % moves.len);
            let before = pos.clone();
            let undo = pos.make_move(m);
            pos.unmake_move(undo);
            if pos.key != before.key
                || pos.occ != before.occ
                || pos.to_fen() != before.to_fen()
                || pos.side != before.side
                || pos.castle != before.castle
                || pos.ep != before.ep
                || pos.king_sq != before.king_sq
            {
                panic!(
                    "ply {} move {} failed:\n  after  {}\n  before {}",
                    p,
                    m.to_uci(),
                    pos.to_fen(),
                    before.to_fen()
                );
            }
        }
    }
}

#[test]
fn roundtrip_startpos() {
    random_roundtrip(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        40,
        80,
    );
}

#[test]
fn roundtrip_kiwipete() {
    random_roundtrip(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        40,
        80,
    );
}

#[test]
fn roundtrip_position_4() {
    random_roundtrip(
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        40,
        80,
    );
}

#[test]
fn roundtrip_position_6() {
    random_roundtrip(
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        40,
        80,
    );
}