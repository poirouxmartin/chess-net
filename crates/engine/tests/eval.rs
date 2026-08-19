//! Sanity checks for the tapered evaluation: material must survive the
//! phase blend (endgame rooks/queens keep their value), PST must be applied
//! in the right orientation, and mirrored positions must be symmetric.

use engine::evaluate::evaluate;
use engine::position::Position;

#[test]
fn endgame_material_survives_tapering() {
    engine::init();
    // KR vs K: a rook must be worth ~a rook even in the endgame phase.
    let kr = Position::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1");
    assert!(evaluate(&kr) > 400, "KR vs K eval = {}", evaluate(&kr));
    assert!(evaluate(&kr) < 650, "KR vs K eval = {}", evaluate(&kr));

    // KQ vs K: a queen must be worth ~a queen.
    let kq = Position::from_fen("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1");
    assert!(evaluate(&kq) > 800, "KQ vs K eval = {}", evaluate(&kq));

    // Mirrored: black rook on h8 ~= -white rook on h1 (up to the tempo term).
    let wr = Position::from_fen("4k3/8/8/8/8/8/8/4K2R w - - 0 1");
    let br = Position::from_fen("4k2r/8/8/8/8/8/8/4K3 w - - 0 1");
    let diff = evaluate(&wr) - 2 * engine::evaluate::TEMPO + evaluate(&br);
    assert!(diff.abs() < 40, "rook symmetry broken: {diff}");

    // KPP vs K: two pawns ~200cp (not hundreds more from a wrong PST row).
    let kpp = Position::from_fen("4k3/8/8/8/8/8/PP6/4K3 w - - 0 1");
    assert!(evaluate(&kpp) > 150 && evaluate(&kpp) < 350, "KPP vs K eval = {}", evaluate(&kpp));

    // K+P vs K+N: a lone knight is ahead of a lone pawn.
    let kpn = Position::from_fen("4k3/8/8/8/8/n7/4P3/4K3 w - - 0 1");
    assert!(evaluate(&kpn) < 0, "K+P vs K+N eval = {} (should favour black knight)", evaluate(&kpn));
}