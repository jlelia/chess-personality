# Chess Personality Indexer

A chess stylometry system that maps player games to historical Grandmasters using a machine-learned latent space.

## Architecture & Design Decisions

### 1. Data Representation & POV Normalization
- **Strict 18-Channel Grid:** Each board state is encoded as a strict $18 \times 8 \times 8$ tensor:
  - Channels 0–5: Player piece types (P, N, B, R, Q, K).
  - Channels 6–11: Opponent piece types (P, N, B, R, Q, K).
  - Channels 12–15: Castling rights (player kingside/queenside, opponent kingside/queenside).
  - Channel 16: Active turn (1 if target's turn, 0 otherwise).
  - Channel 17: En Passant square (1 on the target square, 0 elsewhere).
- **POV Invariance:** If the target player plays Black, the board is vertically flipped (`np.flipud`), and the player/opponent piece and castling channels are swapped. Pawns always advance in the same tensor direction (row 6 to row 0).
- **Sequence Processing:** Game sequences are standardized to the first 64 states ($L=64$). Shorter sequences are zero-padded and handled via key-padding masks; longer sequences are truncated.

### 2. Model Architecture
- **Spatial Front-End (CNN):** Processes each $18 \times 8 \times 8$ state using ResNet blocks and outputs a $128$-dimensional embedding via Global Average Pooling.
- **Sequential Back-End (Transformer):** Aggregates the sequence of $L$ states. A learnable `[CLS]` token is prepended to the sequence, and its final hidden state represents the entire game ($\mathbf{z} \in \mathbb{R}^{128}$).

### 3. Two-Phase Training Objective
- **Phase 1: Joint Embedding Predictive Architecture (JEPA):** Pre-trains the encoder self-supervised by predicting the latent representations of masked moves from surrounding context.
  - *EMA Target:* Target encoder parameters are updated via a cosine schedule (EMA $\beta_t$ from $0.996$ to $1.0$).
  - *Dynamic Masking:* Latent mask window size is dynamically scaled based on game length: $W = \text{clip}(\text{int}(0.15 \times N_{actual}), 5, 10)$.
- **Phase 2: Supervised Contrastive Learning (SupCon):** Fine-tunes the network using 22 Grandmaster identities as labels.
  - The CNN and first 2 Transformer layers are frozen.
  - The `[CLS]` token is projected to the unit sphere using a 2-layer MLP projection head and optimized with SupCon loss ($\tau=0.07$).

---

## Execution Guide

### Prerequisites
Manage dependencies using `uv`:
```bash
# Set up environment and install dependencies
uv sync
```

### 1. Preprocessing
PGN parsing is executed out-of-loop to prevent disk bottlenecks. To parse raw PGNs from `concat_master_games/` and write pre-computed tensors to disk:
```bash
uv run python preprocess_games.py --pgn_dir concat_master_games --out_dir preprocessed_data
```

### 2. Phase 1: JEPA Pre-training
Run self-supervised pre-training:
```bash
uv run python train_phase1.py --data_dir preprocessed_data --save_dir checkpoints --epochs 5 --batch_size 4096
```

### 3. Phase 2: SupCon Fine-tuning
Run contrastive fine-tuning using the checkpoint from Phase 1:
```bash
uv run python train_phase2.py --data_dir preprocessed_data --checkpoint checkpoints/jepa_best.pt --save_dir checkpoints --epochs 5 --batch_size 4096
```

### 4. Running the Local Demo & Inference
Evaluate any user PGN file and match playing style to the 22 Grandmasters:
```bash
uv run python demo.py --pgn test.pgn --player "JamesElia"
```

### 5. Held-out Validation Benchmark
Evaluate the model against 1,675 held-out games (100 games held out for masters with >2,000 games; 25 for masters with ≤2,000 games) to calculate Top-1, Top-3, Top-5 retrieval accuracy and Mean Reciprocal Rank (MRR):
```bash
uv run python demo.py --validate
```

### 6. Running Unit Tests
Validate the pipeline, models, and optimization logic:
```bash
PYTHONPATH=. uv run pytest
```
