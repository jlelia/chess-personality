import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import chess
import chess.pgn
from pathlib import Path

from model import ChessEncoder, SupConProjectionHead
from preprocess_games import board_to_18_channel, GRANDMASTERS, GM_TO_LABEL
from data_pipeline import get_stratified_train_test_datasets

GM_PROFILES = {
    "Alekhine": {
        "title": "Attacking Firebrand",
        "description": "Fierce attacking dynamism, sweeping combinations, and imaginative kingside offensives."
    },
    "Anand": {
        "title": "Universal Speed Maestro",
        "description": "Lightning-fast intuition, versatile opening mastery, and lethal rapid calculation."
    },
    "Aronian": {
        "title": "Creative Piece Artist",
        "description": "Richly imaginative piece harmony, dynamic imbalances, and romantic tactical flair."
    },
    "Botvinnik": {
        "title": "The Scientific Patriarch",
        "description": "Systematic preparation, iron strategic foundations, and relentless methodical conversion."
    },
    "Capablanca": {
        "title": "The Chess Machine",
        "description": "Crystalline positional simplicity, effortless technique, and peerless endgame clarity."
    },
    "Carlsen": {
        "title": "The Universal Grinder",
        "description": "Relentless psychological pressure, squeeze from equal positions, and transcendent endgame precision."
    },
    "Caruana": {
        "title": "Analytical Precision Engine",
        "description": "Deep concrete calculation, uncompromising preparation, and razor-sharp theoretical accuracy."
    },
    "Chigorin": {
        "title": "Romantic Tactician",
        "description": "Bold imaginative piece play, dynamic piece coordination, and romantic attacking spirit."
    },
    "Ding": {
        "title": "Resilient Harmonizer",
        "description": "Unshakeable composure, impregnable defensive resilience, and clinical tactical conversion."
    },
    "Fischer": {
        "title": "Relentless Precision Fighter",
        "description": "Fierce clarity, direct uncompromising play, crystal-clear strategy, and clinical execution."
    },
    "Giri": {
        "title": "Positional Perfectionist",
        "description": "Ironclad theoretical preparation, prophylactic precision, and granite defensive solidity."
    },
    "Karpov": {
        "title": "Master of Prophylaxis",
        "description": "Suffocating python grip, microscopic positional advantages, and effortless positional strangulation."
    },
    "Kasparov": {
        "title": "The Beast of Baku",
        "description": "Overwhelming energy, supreme opening innovation, and dynamic volcanic initiative."
    },
    "Kramnik": {
        "title": "Deep Positional Architect",
        "description": "Profound strategic depth, impenetrable structures (the Berlin Wall), and endgame mastery."
    },
    "Lasker": {
        "title": "Psychological Combatant",
        "description": "Tenacious fighting spirit, steering into chaotic psychological imbalances, and unmatched endgame grit."
    },
    "Mamedyarov": {
        "title": "Uncompromising Attacker",
        "description": "Fearless complications, aggressive flank attacks, and explosive tactical fireworks."
    },
    "Morphy": {
        "title": "The Romantic Virtuoso",
        "description": "Rapid development, open lines, brilliant sacrifices, and direct king hunts."
    },
    "Nakamura": {
        "title": "Dynamic Speed Tactician",
        "description": "Fierce tactical alertness, incredible counter-attacking resourcefulness, and sharp instincts."
    },
    "Nepomniachtchi": {
        "title": "Intuitive Blitzkrieger",
        "description": "Lightning-fast intuitive attacks, high tactical tempo, and aggressive piece pressure."
    },
    "So": {
        "title": "Calm Positional Anchor",
        "description": "Harmonious coordination, minimal error rate, and ultra-solid positional control."
    },
    "Tal": {
        "title": "The Magician from Riga",
        "description": "Mesmerizing sacrifices, psychological chaos, and intoxicating attacking brilliance."
    },
    "VachierLagrave": {
        "title": "Dynamic Counter-Puncher",
        "description": "Supreme mastery of dynamic imbalances (Najdorf & Grünfeld) and active piece counterplay."
    }
}

def load_models_and_centroids(checkpoint_path: str = "checkpoints/supcon_best.pt",
                               centroids_path: str = "checkpoints/gm_centroids.pt",
                               device: torch.device = None):
    """Loads encoder, projection head, and Grandmaster centroid vectors."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    encoder = ChessEncoder().to(device)
    projection_head = SupConProjectionHead().to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}. Run training first.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "encoder_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
    elif "context_encoder_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["context_encoder_state_dict"])
    else:
        raise KeyError("Checkpoint does not contain encoder state dict.")

    if "projection_head_state_dict" in checkpoint:
        projection_head.load_state_dict(checkpoint["projection_head_state_dict"])

    encoder.eval()
    projection_head.eval()

    if not os.path.exists(centroids_path):
        raise FileNotFoundError(f"Centroids file not found at: {centroids_path}.")

    centroids_data = torch.load(centroids_path, map_location=device, weights_only=True)
    centroids = centroids_data["centroids"].to(device) # (22, 128)
    centroids = F.normalize(centroids, p=2, dim=-1)

    return encoder, projection_head, centroids, device

def parse_game_to_tensor(game: chess.pgn.Game, target_color: chess.Color) -> tuple[torch.Tensor, torch.Tensor]:
    """Encodes a single PGN game into (1, 64, 18, 8, 8) board tensor and padding mask."""
    board = game.board()
    states = [board_to_18_channel(board, target_color)]

    for move in game.mainline_moves():
        if len(states) >= 64:
            break
        board.push(move)
        states.append(board_to_18_channel(board, target_color))

    actual_length = len(states)
    while len(states) < 64:
        states.append(np.zeros((18, 8, 8), dtype=np.uint8))

    boards_np = np.stack(states, axis=0) # (64, 18, 8, 8)
    padding_mask_np = np.zeros(64, dtype=bool)
    if actual_length < 64:
        padding_mask_np[actual_length:] = True

    boards_tensor = torch.from_numpy(boards_np).float().unsqueeze(0) # (1, 64, 18, 8, 8)
    padding_mask_tensor = torch.from_numpy(padding_mask_np).unsqueeze(0) # (1, 64)

    return boards_tensor, padding_mask_tensor

def analyze_pgn(pgn_path: str, player_name: str = None,
                checkpoint_path: str = "checkpoints/supcon_best.pt",
                centroids_path: str = "checkpoints/gm_centroids.pt"):
    """Evaluates a PGN game and displays stylized Grandmaster matching results."""
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder, projection_head, centroids, device = load_models_and_centroids(checkpoint_path, centroids_path, device)

    with open(pgn_path, "r", errors="ignore") as f:
        game = chess.pgn.read_game(f)
        if game is None:
            raise ValueError(f"Could not read any valid chess game from {pgn_path}")

    headers = game.headers
    white_player = headers.get("White", "Unknown White")
    black_player = headers.get("Black", "Unknown Black")
    event = headers.get("Event", "Casual Game")
    date = headers.get("Date", "????.??.??")
    result = headers.get("Result", "*")

    # Determine perspective
    if player_name:
        p_clean = player_name.lower().replace(" ", "")
        w_clean = white_player.lower().replace(" ", "")
        b_clean = black_player.lower().replace(" ", "")
        if p_clean in b_clean and p_clean not in w_clean:
            target_color = chess.BLACK
            active_player = black_player
        else:
            target_color = chess.WHITE
            active_player = white_player
    else:
        target_color = chess.WHITE
        active_player = white_player

    color_str = "White" if target_color == chess.WHITE else "Black"
    boards_tensor, padding_mask_tensor = parse_game_to_tensor(game, target_color)
    boards_tensor = boards_tensor.to(device)
    padding_mask_tensor = padding_mask_tensor.to(device)

    with torch.no_grad():
        out = encoder(boards_tensor, padding_mask=padding_mask_tensor) # (1, 65, 128)
        z_cls = out[:, 0] # (1, 128)
        z_proj = projection_head(z_cls) # (1, 128)

        # Centering: subtract global baseline to isolate pure stylistic variance
        mu = centroids.mean(dim=0, keepdim=True)
        c_centered = F.normalize(centroids - mu, p=2, dim=-1)
        z_proj_centered = F.normalize(z_proj - mu, p=2, dim=-1)

        # Cosine similarity against all 22 centroids
        sims = torch.matmul(c_centered, z_proj_centered.T).squeeze().cpu().numpy() # (22,)

    # Scale cosine similarity from [-1, 1] to a percentage [0%, 100%]
    pct_scores = np.clip((sims + 1.0) / 2.0 * 100.0, 0.0, 100.0)

    # Sort descending
    ranked_indices = np.argsort(-sims)

    top_idx = ranked_indices[0]
    top_gm = GRANDMASTERS[top_idx]
    top_score = pct_scores[top_idx]
    top_raw_cos = sims[top_idx]

    print("\n" + "=" * 70)
    print("           CHESS PERSONALITY INDEXER - STYLOMETRY REPORT           ")
    print("=" * 70)
    print(f"Game:    {white_player} vs {black_player} ({result})")
    print(f"Event:   {event} | Date: {date}")
    print(f"Target:  {active_player} (Playing as {color_str})")
    print("-" * 70)
    print(f"\n★ PRIMARY STYLISTIC MATCH: {top_gm.upper()} ★")
    print(f"  Similarity Index: {top_score:.1f}% (Cosine: {top_raw_cos:.4f})")
    print(f"  Archetype:        {GM_PROFILES.get(top_gm, {}).get('title', '')}")
    print(f"  Profile:          {GM_PROFILES.get(top_gm, {}).get('description', '')}")

    print("\n" + "-" * 70)
    print("TOP 5 GRANDMASTER MATCHES:")
    print("-" * 70)
    print(f"{'Rank':<5} {'Grandmaster':<18} {'Similarity':<12} {'Visual Alignment'}")
    print("-" * 70)

    for rank, idx in enumerate(ranked_indices[:5], 1):
        gm = GRANDMASTERS[idx]
        score = pct_scores[idx]
        bar_len = int(score / 5) # 20 chars max
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f" #{rank:<3} {gm:<18} {score:>6.1f}%     [{bar}]")

    print("-" * 70)
    print("\nFULL 22 GRANDMASTER RANKINGS:")
    print("-" * 70)
    for rank, idx in enumerate(ranked_indices, 1):
        gm = GRANDMASTERS[idx]
        score = pct_scores[idx]
        print(f" {rank:>2}. {gm:<18} : {score:>5.1f}% (cos: {sims[idx]:+.4f})")
    print("=" * 70 + "\n")

def run_validation_benchmark(checkpoint_path: str = "checkpoints/supcon_best.pt",
                             centroids_path: str = "checkpoints/gm_centroids.pt",
                             data_dir: str = "preprocessed_data"):
    """Runs a frozen forward pass across the 1,675 held-out test games and measures retrieval metrics."""
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n--- Running Held-out Test Validation on {device} ---")
    encoder, projection_head, centroids, device = load_models_and_centroids(checkpoint_path, centroids_path, device)

    _, test_dataset = get_stratified_train_test_datasets(data_dir, seed=42)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    # Centering: subtract global baseline to isolate pure stylistic variance
    mu = centroids.mean(dim=0, keepdim=True)
    c_centered = F.normalize(centroids - mu, p=2, dim=-1)

    all_embs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            boards = batch["boards"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            labels = batch["player_label"]

            out = encoder(boards, padding_mask=padding_mask)
            z_cls = out[:, 0]
            z_proj = projection_head(z_cls)
            z_proj_centered = F.normalize(z_proj - mu, p=2, dim=-1)

            all_embs.append(z_proj_centered.cpu())
            all_labels.append(labels.cpu())

    all_embs = torch.cat(all_embs, dim=0) # (1675, 128)
    all_labels = torch.cat(all_labels, dim=0) # (1675,)

    total_games = len(test_dataset)

    # 1-Game Metrics
    sim_matrix = torch.matmul(all_embs, c_centered.cpu().T).numpy() # (1675, 22)

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    mrr_sum = 0.0

    per_gm_total = {gm: 0 for gm in GRANDMASTERS}
    per_gm_top1 = {gm: 0 for gm in GRANDMASTERS}
    per_gm_top3 = {gm: 0 for gm in GRANDMASTERS}

    # Group embeddings by master for profile evaluations
    embs_by_gm = {idx: [] for idx in range(len(GRANDMASTERS))}

    for i in range(total_games):
        true_label = all_labels[i].item()
        true_gm = GRANDMASTERS[true_label]
        embs_by_gm[true_label].append(all_embs[i])

        sims = sim_matrix[i]
        ranked_indices = np.argsort(-sims)

        per_gm_total[true_gm] += 1
        rank = np.where(ranked_indices == true_label)[0][0] + 1
        mrr_sum += 1.0 / rank

        if rank == 1:
            top1_correct += 1
            per_gm_top1[true_gm] += 1
        if rank <= 3:
            top3_correct += 1
            per_gm_top3[true_gm] += 1
        if rank <= 5:
            top5_correct += 1

    top1_1g = (top1_correct / total_games) * 100.0
    top3_1g = (top3_correct / total_games) * 100.0
    top5_1g = (top5_correct / total_games) * 100.0
    mrr_1g = (mrr_sum / total_games)

    # Multi-Game Profile Evaluations (3 Games, 10 Games)
    import random
    rng = random.Random(42)

    def eval_profile(n_games: int, num_trials_per_gm: int = 50):
        t1, t3, t5, mrr_s, n_trials = 0, 0, 0, 0.0, 0
        c_cpu = c_centered.cpu()
        for gm_idx, game_list in embs_by_gm.items():
            if len(game_list) < n_games:
                continue
            for _ in range(num_trials_per_gm):
                chosen = rng.sample(game_list, n_games)
                profile = torch.stack(chosen).mean(dim=0, keepdim=True)
                profile = F.normalize(profile, p=2, dim=-1)
                sims = torch.matmul(c_cpu, profile.T).squeeze().numpy()
                ranks = np.argsort(-sims)
                r = np.where(ranks == gm_idx)[0][0] + 1
                mrr_s += 1.0 / r
                if r == 1: t1 += 1
                if r <= 3: t3 += 1
                if r <= 5: t5 += 1
                n_trials += 1
        return (t1 / n_trials * 100.0), (t3 / n_trials * 100.0), (t5 / n_trials * 100.0), (mrr_s / n_trials)

    top1_3g, top3_3g, top5_3g, mrr_3g = eval_profile(3)
    top1_10g, top3_10g, top5_10g, mrr_10g = eval_profile(10)

    print("\n" + "=" * 70)
    print("      STANDARD HELD-OUT RETRIEVAL BENCHMARK (1,675 TEST GAMES)      ")
    print("=" * 70)
    print(f"Random Baseline Top-1: {100.0 / 22:.2f}%")
    print("-" * 70)
    print(f"{'Profile Size':<20} {'Top-1 Acc':<14} {'Top-3 Acc':<14} {'Top-5 Acc':<14} {'MRR'}")
    print("-" * 70)
    print(f"{'1 Game (Single)':<20} {top1_1g:>8.2f}%     {top3_1g:>8.2f}%     {top5_1g:>8.2f}%     {mrr_1g:.4f}")
    print(f"{'3 Games (Profile)':<20} {top1_3g:>8.2f}%     {top3_3g:>8.2f}%     {top5_3g:>8.2f}%     {mrr_3g:.4f}")
    print(f"{'10 Games (Profile)':<20} {top1_10g:>8.2f}%     {top3_10g:>8.2f}%     {top5_10g:>8.2f}%     {mrr_10g:.4f}")
    print("-" * 70)
    print("\nPER-GRANDMASTER BREAKDOWN (1-GAME SINGLE):")
    print("-" * 70)
    print(f"{'Grandmaster':<18} {'Test Games':<12} {'Top-1 Acc':<14} {'Top-3 Acc'}")
    print("-" * 70)

    for gm in GRANDMASTERS:
        tot = per_gm_total[gm]
        t1 = (per_gm_top1[gm] / tot * 100.0) if tot > 0 else 0.0
        t3 = (per_gm_top3[gm] / tot * 100.0) if tot > 0 else 0.0
        print(f"{gm:<18} {tot:<12} {t1:>6.1f}%        {t3:>6.1f}%")
    print("=" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Chess Personality Indexer - Inference & Validation Demo")
    parser.add_argument("--pgn", type=str, default=None, help="Path to input PGN file to analyze.")
    parser.add_argument("--player", type=str, default=None, help="Name of target player inside the PGN.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/supcon_best.pt", help="Path to SupCon model checkpoint.")
    parser.add_argument("--centroids", type=str, default="checkpoints/gm_centroids.pt", help="Path to GM centroids file.")
    parser.add_argument("--data_dir", type=str, default="preprocessed_data", help="Path to preprocessed games directory.")
    parser.add_argument("--validate", action="store_true", help="Run validation benchmark on held-out test games.")
    args = parser.parse_args()

    if args.validate:
        run_validation_benchmark(args.checkpoint, args.centroids, args.data_dir)
    elif args.pgn:
        analyze_pgn(args.pgn, args.player, args.checkpoint, args.centroids)
    else:
        # Default: if test.pgn exists, analyze it
        if os.path.exists("test.pgn") and os.path.exists(args.checkpoint) and os.path.exists(args.centroids):
            print("Analyzing default test.pgn...")
            analyze_pgn("test.pgn", args.player, args.checkpoint, args.centroids)
        else:
            parser.print_help()

if __name__ == "__main__":
    main()
