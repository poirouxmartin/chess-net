# chess_network - Project Conventions

## Architecture
- Rust workspace (Cargo) + Python training toolkit
- Crates:
  - `crates/engine/` - Core engine: bitboards, magic movegen, zobrist, FEN, PeSTO eval, alpha-beta/PVS, TT, killers/history, null-move, quiescence, UCI
  - `crates/nn/` - Neural backend: CSNN format, HalfKP/KP768 features, sparse 3-layer MLP inference
  - `crates/engine-uci/` - `chess-net` binary wiring engine + NN
- Python: `python/chessnet/` - Training (PyTorch):
  - `features.py` - HalfKP/KP768 encoders (aligned with Rust)
  - `csnn.py` - CSNN binary format read/write
  - `nnue.py` - NNUE module + supervised training + CSNN export
  - `alpha.py` - MCTS + self-play RL (parallel workers)
  - `gen_data.py` - Dataset generator (random/self-play)
  - `train_nnue.py`, `train_alpha.py`, `export.py` - CLIs

## Commands
```bash
# Rust engine
cargo build --release
cargo test --release
./target/release/chess-net.exe

# UCI: uci, isready, ucinewgame, setoption Hash/EvalFile, position, go, stop, perft, divide, eval
# Load NN: setoption name EvalFile value path/to/net.csnn

# Python training
pip install -r python/requirements.txt

# Supervised
python -m chessnet.gen_data --mode random --games 500 --out data.txt
python -m chessnet.train_nnue --data data.txt --out model.pt --csnn net.csnn

# Self-play RL
python -m chessnet.train_alpha --cycles 5 --games 40 --mcts-iters 100 --mcts-workers 4 --out model.pt --csnn net.csnn
```

## Code Style
- Rust: 2021 edition, `clippy`, `rustfmt`
- Engine: bitboards (u64), magic bitboards, no unsafe in engine crate
- NN: `feat` enum (1=HalfKP 41024, 2=KP768), CSNN binary format
- Python: type hints, PyTorch, CUDA optional

## Conventions
- Perft tests green (startpos, kiwipete, pos 3-6)
- Feature encoders MUST match between Rust and Python
- CSNN format: magic `CSNN`, v1, feat, l0, l1, feat_count, float32 arrays
- Score scaling: `(bo + wo·relu(b1 + w1·relu(acc))) * 400` cp