# Project: Chess Personality Indexer

## 1. Project Objective
Build a web application that takes user-uploaded chess games (PGN format) and maps their stylistic "personality" to one of 22 Grandmasters using a machine-learned latent space.

## 2. Data Engineering & Representation
*   **Sequence Length:** Full games are used to capture stylistic transitions without shortcut learning. Sequences are standardized to a maximum of $L = 64$ moves. Pad shorter games; truncate longer games.
*   **State Encoding:** Convert sequences of FENs into a sequence of NumPy arrays. 
*   **Tensor Shape:** Strictly adhere to PyTorch's $C \times H \times W$ format. Each board state must be an $18 \times 8 \times 8$ tensor. Stack arrays along `axis=0`.
*   **Channels (18 Total):**
    *   12 channels for pieces (1 for each piece type per color, using `chess.SquareSet`).
    *   4 channels for castling rights (broadcast to $8 \times 8$).
    *   1 channel for whose turn it is (broadcast to $8 \times 8$).
    *   1 channel for En Passant square (1 on the legal square, 0 elsewhere).
*   **POV Normalization:** Standardize the "forward" direction based on the target player. If the master/user is playing Black, dynamically flip the board representations vertically so pawns always advance in the same tensor direction.

## 3. Model Architecture
*   **Dimensionality:** Latent vector size for the encoder pipeline is $d = 128$.
*   **Spatial Front-End:** A Convolutional Neural Network (CNN) to process each $18 \times 8 \times 8$ state into a dense 128-dimensional feature vector.
*   **Sequential Back-End:** A Transformer encoder to process the sequence of move vectors.
*   **Game Vectorization:** Prepend a learnable `[CLS]` token to the sequence. The final hidden state of this token becomes the single dense vector $\mathbf{z} \in \mathbb{R}^{128}$ representing the entire game.

## 4. Two-Phase Training Objective
*   **Phase 1: JEPA Pre-training (Self-Supervised)**
    *   Train the encoder to predict the latent representations of missing moves based on surrounding context.
    *   **Masking Strategy:** Mask continuous blocks in the latent sequence (not the input bitboards).
    *   **Masking Window:** Dynamically scaled based on the actual length of the game, strictly bounded by a minimum of 5 moves and a maximum of 10 moves.
*   **Phase 2: SupCon Fine-tuning (Supervised)**
    *   Freeze the CNN and lower Transformer layers. Pass full, unmasked game sequences to extract the `[CLS]` token.
    *   Apply Supervised Contrastive Learning (SupCon) using the 22 master identities as labels to explicitly organize the latent space geometry. The attention mechanism natively filters opponent noise.

## 5. Inference & Evaluation
*   **Metric:** Cosine Similarity against the 22 cluster centroids or via k-Nearest Neighbors (k-NN).
*   **Constraint:** Do NOT attempt to predict player Elo or evaluation metrics. This pipeline is strictly isolated to mapping stylistic personality.
