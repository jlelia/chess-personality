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
Run contrastive fine-tuning using balanced batch sampling (22 GMs × 12 games) and hybrid SupCon + Cross-Entropy loss:
```bash
uv run python train_phase2.py --data_dir preprocessed_data --checkpoint checkpoints/jepa_best.pt --save_dir checkpoints_v2 --epochs 4 --batch_size 264
```

### 4. Running the Web Application
Launch the local web app with real-time browser interface and REST API:
```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser to:
- Drag & drop single or multi-game `.pgn` files or paste notation directly.
- Inspect the primary Grandmaster match, alignment percentage, and narrative archetype.
- View stylistic radar attributes (*Tactical Sharpness*, *Positional Solidity*, *Dynamic Initiative*, *Endgame Technique*).
- Explore the podium and full 22 Grandmaster cosine alignment rankings.

### 5. Running the CLI Demo
Evaluate any user PGN file directly from the terminal:
```bash
uv run python demo.py --pgn test.pgn --player "JamesElia"
```

### 6. Held-out Validation Benchmarks
The model was evaluated against **1,675 strictly held-out unseen games** (100 games held out for masters with >2,000 games; 25 for masters with ≤2,000 games) across 1-game, 3-game, and 10-game profile aggregations:

| Evaluation Setup | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy | Mean Reciprocal Rank (MRR) |
| :--- | :---: | :---: | :---: | :---: |
| **1-Game (Single Game)** | 14.57% | 34.75% | 51.28% | 0.3202 |
| **3-Game Profile** | 30.64% | 57.09% | 71.45% | 0.4862 |
| **10-Game Profile** | **51.00%** | **80.18%** | **90.55%** | **0.6728** |

*Note: In contrastive stylometry, a player's true stylistic fingerprint stabilizes as multiple games are aggregated. With a 10-game sample, the correct Grandmaster is retrieved in the Top-3 over 80% of the time and in the Top-5 over 90% of the time.*

To re-run the benchmark suite:
```bash
uv run python demo.py --validate
```

### 7. Running Unit Tests
Validate the pipeline, models, and optimization logic:
```bash
PYTHONPATH=. uv run pytest
```

