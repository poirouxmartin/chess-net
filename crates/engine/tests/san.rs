//! SAN notation tests.

use engine::move_::Move;
use engine::movegen::generate_legal;
use engine::position::Position;
use engine::san::to_san;

fn san(fen: &str, uci: &str) -> String {
    engine::init();
    let pos = Position::from_fen(fen);
    let legal = generate_legal(&pos);
    let mut m = Move::null();
    for i in 0..legal.len {
        if legal.get(i).to_uci() == uci {
            m = legal.get(i);
            break;
        }
    }
    assert!(m != Move::null(), "move {uci} not legal in {fen}");
    to_san(&pos, m)
}

#[test]
fn simple_moves() {
    assert_eq!(san("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4"), "e4");
    assert_eq!(san("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "g1f3"), "Nf3");
    assert_eq!(san("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e3"), "e3");
}

#[test]
fn castle() {
    assert_eq!(san("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", "e1g1"), "O-O");
    assert_eq!(san("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", "e1c1"), "O-O-O");
}

#[test]
fn disambiguation() {
    // Knights on c3 and d2 both reach e4.
    let fen = "rnbqkbnr/pppppppp/8/8/8/2N5/3N4/RNBQKB1R w KQkq - 0 1";
    assert_eq!(san(fen, "c3e4"), "Nce4");
    assert_eq!(san(fen, "d2e4"), "Nde4");
}

#[test]
fn pawn_capture() {
    let fen = "rnbqkbnr/ppp1pppp/8/2p1p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1";
    assert_eq!(san(fen, "d4e5"), "dxe5");
    assert_eq!(san(fen, "d4c5"), "dxc5");
}

#[test]
fn promotion_check() {
    assert_eq!(san("7k/P6P/8/8/8/8/8/6K1 w - - 0 1", "a7a8q"), "a8=Q+");
}

#[test]
fn capture_check() {
    // Rook f7-f8 gives check on the 8th rank.
    assert_eq!(san("6k1/5R2/8/8/8/8/8/4K3 w - - 0 1", "f7f8"), "Rf8+");
}