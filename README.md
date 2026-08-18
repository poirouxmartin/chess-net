# chess-net

High-performance chess engine written in Rust with neural-network evaluation
(two approaches: AlphaZero-style self-play RL and supervised NNUE), plus a
Python training toolkit. UCI-compatible binary.

## Structure

```
crates/
  engine/       Chess engine: bitboards, magic move generation, legal movegen,
                zobrist, FEN, make/unmake, PeSTO evaluation, alpha-beta/PVS
                search with TT, killers/history, null-move, quiescence, UCI.
  nn/           Neural evaluation backend: loads CSNN models, encodes HalfKP
                and KP768 features, sparse 3-layer MLP inference.
  engine-uci/   `chess-net` binary wiring the engine to the NN (EvalFile).
python/
  chessnet/     Training toolkit (PyTorch, CUDA or CPU):
    features.py     HalfKP / KP768 encoders aligned with the Rust engine
    csnn.py         CSNN binary format reader/writer
    nnue.py         NNUE module + supervised training + CSNN export
    alpha.py        MCTS + self-play reinforcement learning (parallel workers)
    gen_data.py     dataset generator (random or net self-play)
    export.py       checkpoint -> CSNN CLI
    train_nnue.py   supervised training CLI
    train_alpha.py  self-play RL training CLI
```

## Build and test

```sh
cargo build --release
cargo test --release
```

Perft suite (startpos, kiwipete, positions 3-6) and make/unmake roundtrip
tests are green.

## Run

```sh
target/release/chess-net.exe
```

UCI commands: `uci`, `isready`, `ucinewgame`, `setoption` (`Hash`,
`EvalFile`), `position`, `go`, `stop`, plus diagnostics `perft <d>`,
`divide <d>`, `moves`, `d`, `eval`.

Load a neural network instead of the built-in PeSTO evaluation:

```
setoption name EvalFile value path/to/net.csnn
```

## Neural evaluation

CSNN format: magic `CSNN`, version 1, then `feat` (1 = HalfKP `2*41024`
features, 2 = KP768 `768` features), `l0`, `l1`, `feat_count`, and float32
arrays `fw[feat_count*l0]`, `fb[l0]`, `w1[l1*l0]`, `b1[l1]`, `wo[l1]`,
`bo`. Score = `(bo + wo·relu(b1 + w1·relu(acc))) * 400` centipawns.

## Training

```sh
pip install -r python/requirements.txt

# Supervised: dataset lines "<fen> <label>"
python -m chessnet.gen_data --mode random --games 500 --out data.txt
python -m chessnet.train_nnue --data data.txt --out model.pt --csnn net.csnn

# Self-play RL (AlphaZero-style), parallel self-play via --mcts-workers
python -m chessnet.train_alpha --cycles 5 --games 40 --mcts-iters 100 \
    --mcts-workers 4 --out model.pt --csnn net.csnn

# Supervised data from an existing net's self-play (MCTS)
python -m chessnet.gen_data --mode net --checkpoint model.pt --games 200 \
    --mcts-iters 80 --workers 4 --out data.txt

# Use HalfKP features instead of KP768 (engine: --feat 1)
python -m chessnet.train_nnue --feat 1 --data data.txt --out model.pt --csnn net.csnn

# Export an existing checkpoint
python -m chessnet.export --checkpoint model.pt --out net.csnn
```
