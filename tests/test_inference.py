import pytest
import chess
import torch
from pathlib import Path
from app.inference import ChessStylometryEngine

def test_inference_pgn_parsing():
    dummy_pgn = """[Event "Casual Game"]
[Site "Online"]
[Date "2024.01.01"]
[White "PlayerOne"]
[Black "PlayerTwo"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 1-0
"""
    if Path("checkpoints_v2/supcon_best.pt").exists():
        engine = ChessStylometryEngine()
        parsed = engine.parse_pgn_text(dummy_pgn, player_name="PlayerTwo")
        assert len(parsed) == 1
        assert parsed[0]["target_color"] == chess.BLACK
        assert parsed[0]["active_player"] == "PlayerTwo"

        # Test full analysis
        result = engine.analyze_games(dummy_pgn, player_name="PlayerTwo")
        assert result["games_analyzed"] == 1
        assert len(result["all_rankings"]) == 22
        assert "style_radar" in result
        assert "tactical_sharpness" in result["style_radar"]
