import io
import os
import re
import chess
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from model import ChessEncoder, SupConProjectionHead
from preprocess_games import board_to_18_channel, GRANDMASTERS, GM_TO_LABEL
from demo import GM_PROFILES

class ChessStylometryEngine:
    """In-memory inference engine for Grandmaster stylometry matching."""

    def __init__(self, checkpoint_path: str = "checkpoints_v2/supcon_best.pt",
                 centroids_path: str = "checkpoints_v2/gm_centroids.pt"):
        # Fallback to v1 if v2 does not exist
        if not os.path.exists(checkpoint_path):
            checkpoint_path = "checkpoints/supcon_best.pt"
            centroids_path = "checkpoints/gm_centroids.pt"

        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Loading ChessStylometryEngine on {self.device} from {checkpoint_path}...")

        self.encoder = ChessEncoder().to(self.device)
        self.projection_head = SupConProjectionHead().to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.encoder.load_state_dict(ckpt["encoder_state_dict"])
        self.projection_head.load_state_dict(ckpt["projection_head_state_dict"])
        self.encoder.eval()
        self.projection_head.eval()

        cent_data = torch.load(centroids_path, map_location=self.device, weights_only=True)
        raw_centroids = cent_data["centroids"].to(self.device) # (22, 128)
        self.mu = raw_centroids.mean(dim=0, keepdim=True)
        self.centroids_centered = F.normalize(raw_centroids - self.mu, p=2, dim=-1)

        print("ChessStylometryEngine initialized successfully!")

    def _parse_game_to_tensor(self, game: chess.pgn.Game, target_color: chess.Color) -> tuple[torch.Tensor, torch.Tensor]:
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

        boards_np = np.stack(states, axis=0)
        padding_mask_np = np.zeros(64, dtype=bool)
        if actual_length < 64:
            padding_mask_np[actual_length:] = True

        boards_t = torch.from_numpy(boards_np).float().unsqueeze(0)
        padding_mask_t = torch.from_numpy(padding_mask_np).unsqueeze(0)
        return boards_t, padding_mask_t

    def parse_pgn_text(self, pgn_content: str, player_name: str = None) -> list[dict]:
        """Parses all valid games from PGN text content."""
        pgn_io = io.StringIO(pgn_content)
        games_data = []

        while True:
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                break

            headers = game.headers
            white_player = headers.get("White", "Unknown White")
            black_player = headers.get("Black", "Unknown Black")
            event = headers.get("Event", "Game")
            date = headers.get("Date", "????.??.??")
            result = headers.get("Result", "*")

            # Determine perspective
            target_color = chess.WHITE
            active_player = white_player

            if player_name:
                p_clean = re.sub(r'[^a-zA-Z0-9]', '', player_name).lower()
                w_clean = re.sub(r'[^a-zA-Z0-9]', '', white_player).lower()
                b_clean = re.sub(r'[^a-zA-Z0-9]', '', black_player).lower()

                if p_clean in b_clean and p_clean not in w_clean:
                    target_color = chess.BLACK
                    active_player = black_player
                elif p_clean in w_clean:
                    target_color = chess.WHITE
                    active_player = white_player

            games_data.append({
                "game": game,
                "white": white_player,
                "black": black_player,
                "event": event,
                "date": date,
                "result": result,
                "target_color": target_color,
                "active_player": active_player,
                "ply_count": len(list(game.mainline_moves()))
            })

        return games_data

    def analyze_games(self, pgn_content: str, player_name: str = None) -> dict:
        """Encodes all games in the PGN, computes profile vector, and matches against the 22 GMs."""
        parsed_games = self.parse_pgn_text(pgn_content, player_name)
        if not parsed_games:
            raise ValueError("No valid chess games found in the provided PGN content.")

        game_embeddings = []
        games_summary = []

        with torch.no_grad():
            for g_info in parsed_games:
                game = g_info["game"]
                target_color = g_info["target_color"]
                boards_t, pad_mask_t = self._parse_game_to_tensor(game, target_color)
                boards_t = boards_t.to(self.device)
                pad_mask_t = pad_mask_t.to(self.device)

                out = self.encoder(boards_t, padding_mask=pad_mask_t)
                z_cls = out[:, 0]
                z_proj = self.projection_head(z_cls)
                z_proj_centered = F.normalize(z_proj - self.mu, p=2, dim=-1)

                game_embeddings.append(z_proj_centered.squeeze(0))

                games_summary.append({
                    "white": g_info["white"],
                    "black": g_info["black"],
                    "result": g_info["result"],
                    "event": g_info["event"],
                    "date": g_info["date"],
                    "played_as": "White" if target_color == chess.WHITE else "Black",
                    "plies": g_info["ply_count"]
                })

            # Profile embedding: mean over all games in this PGN, re-normalized to unit sphere
            stacked_embs = torch.stack(game_embeddings, dim=0) # (N, 128)
            profile_emb = stacked_embs.mean(dim=0, keepdim=True) # (1, 128)
            profile_emb = F.normalize(profile_emb, p=2, dim=-1)

            # Cosine similarity against all 22 centered centroids
            sims = torch.matmul(profile_emb, self.centroids_centered.T).squeeze(0).cpu().numpy()

        # Convert cosine values [-1, 1] to intuitive percentage alignment [0%, 100%]
        pct_scores = np.clip((sims + 1.0) / 2.0 * 100.0, 0.0, 100.0)
        ranked_indices = np.argsort(-sims)

        rankings = []
        for rank, idx in enumerate(ranked_indices, 1):
            gm = GRANDMASTERS[idx]
            profile_meta = GM_PROFILES.get(gm, {"title": "Grandmaster", "description": "Legendary chess player."})
            rankings.append({
                "rank": rank,
                "name": gm,
                "title": profile_meta["title"],
                "description": profile_meta["description"],
                "similarity_pct": round(float(pct_scores[idx]), 1),
                "cosine_sim": round(float(sims[idx]), 4)
            })

        top_match = rankings[0]
        runner_ups = rankings[1:5]

        # Calculate stylistic trait attributes from the top alignment profile
        # Derived from projections onto tactical, positional, dynamic, and endgame reference clusters
        tactical_cluster = ["Alekhine", "Kasparov", "Tal", "Morphy", "Mamedyarov", "Chigorin"]
        positional_cluster = ["Capablanca", "Karpov", "Kramnik", "Botvinnik", "Giri", "So"]
        dynamic_cluster = ["Fischer", "Nakamura", "Anand", "Nepomniachtchi", "VachierLagrave"]
        endgame_cluster = ["Carlsen", "Capablanca", "Karpov", "Ding", "Caruana", "Lasker"]

        def cluster_score(cluster_list):
            matching_scores = [r["similarity_pct"] for r in rankings if r["name"] in cluster_list]
            return round(float(np.mean(matching_scores)), 1) if matching_scores else 50.0

        style_radar = {
            "tactical_sharpness": cluster_score(tactical_cluster),
            "positional_solidity": cluster_score(positional_cluster),
            "dynamic_initiative": cluster_score(dynamic_cluster),
            "endgame_technique": cluster_score(endgame_cluster)
        }

        active_player_name = parsed_games[0]["active_player"]
        if player_name:
            active_player_name = player_name

        return {
            "player_name": active_player_name,
            "games_analyzed": len(parsed_games),
            "top_match": top_match,
            "top_5": rankings[:5],
            "all_rankings": rankings,
            "style_radar": style_radar,
            "games": games_summary
        }
