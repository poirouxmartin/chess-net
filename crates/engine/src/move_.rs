//! Move encoding: 21 bits packed in a u32.
//! bits 0-5: from, bits 6-11: to, bits 12-14: promotion piece, bits 15-17: flags,
//! bits 18-20: moving piece type (set by movegen, used by perft fast make/unmake).

pub const FLAG_QUIET: u8 = 0;
pub const FLAG_DOUBLE: u8 = 1;
pub const FLAG_CASTLE_KS: u8 = 2;
pub const FLAG_CASTLE_QS: u8 = 3;
pub const FLAG_CAPTURE: u8 = 4;
pub const FLAG_EN_PASSANT: u8 = 5;
pub const FLAG_PROMO: u8 = 6;
pub const FLAG_PROMO_CAPTURE: u8 = 7;

// Promotion piece values (also used as MVV-LVA order later).
pub const PROMO_QUEEN: u8 = 0;
pub const PROMO_ROOK: u8 = 1;
pub const PROMO_BISHOP: u8 = 2;
pub const PROMO_KNIGHT: u8 = 3;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Move(pub u32);

impl Move {
    #[inline(always)]
    pub fn new(from: usize, to: usize, promo: u8, flags: u8) -> Self {
        Move((from as u32) | ((to as u32) << 6) | ((promo as u32) << 12) | ((flags as u32) << 15))
    }

    #[inline(always)]
    pub const fn null() -> Self {
        Move(0)
    }

    #[inline(always)]
    pub fn from(self) -> usize {
        (self.0 & 63) as usize
    }

    #[inline(always)]
    pub fn to(self) -> usize {
        ((self.0 >> 6) & 63) as usize
    }

    #[inline(always)]
    pub fn promo(self) -> u8 {
        ((self.0 >> 12) & 7) as u8
    }

    #[inline(always)]
    pub fn flags(self) -> u8 {
        ((self.0 >> 15) & 7) as u8
    }

    #[inline(always)]
    pub fn with_piece(self, pt: usize) -> Move {
        Move(self.0 | ((pt as u32) << 18))
    }

    #[inline(always)]
    pub fn piece_pt(self) -> usize {
        ((self.0 >> 18) & 7) as usize
    }

    #[inline(always)]
    pub fn is_quiet(self) -> bool {
        self.flags() == FLAG_QUIET || self.flags() == FLAG_DOUBLE
    }

    #[inline(always)]
    pub fn is_capture(self) -> bool {
        matches!(self.flags(), FLAG_CAPTURE | FLAG_EN_PASSANT | FLAG_PROMO_CAPTURE)
    }

    #[inline(always)]
    pub fn is_promotion(self) -> bool {
        matches!(self.flags(), FLAG_PROMO | FLAG_PROMO_CAPTURE)
    }

    #[inline(always)]
    pub fn is_castle(self) -> bool {
        matches!(self.flags(), FLAG_CASTLE_KS | FLAG_CASTLE_QS)
    }

    #[inline(always)]
    pub fn is_en_passant(self) -> bool {
        self.flags() == FLAG_EN_PASSANT
    }

    #[inline(always)]
    pub fn promo_pt(self) -> usize {
        match self.promo() {
            PROMO_QUEEN => crate::position::QUEEN,
            PROMO_ROOK => crate::position::ROOK,
            PROMO_BISHOP => crate::position::BISHOP,
            _ => crate::position::KNIGHT,
        }
    }

    /// UCI representation, e.g. "e2e4", "e7e8q".
    pub fn to_uci(self) -> String {
        let from = sq_to_name(self.from());
        let to = sq_to_name(self.to());
        if self.is_promotion() {
            format!("{from}{to}{}", promo_char(self.promo()))
        } else {
            format!("{from}{to}")
        }
    }
}

pub fn sq_to_name(sq: usize) -> String {
    let file = (b'a' + (sq & 7) as u8) as char;
    let rank = (b'1' + (sq >> 3) as u8) as char;
    format!("{file}{rank}")
}

pub fn parse_sq(name: &str) -> usize {
    let b = name.as_bytes();
    let file = (b[0] - b'a') as usize;
    let rank = (b[1] - b'1') as usize;
    rank * 8 + file
}

pub fn promo_char(p: u8) -> char {
    match p {
        PROMO_QUEEN => 'q',
        PROMO_ROOK => 'r',
        PROMO_BISHOP => 'b',
        _ => 'n',
    }
}