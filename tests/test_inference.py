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
        assert "primary_final_svg" in result
        assert "<svg" in result["primary_final_svg"]
        assert "final_fen" in result["games"][0]

def test_invalid_pgn_rejection():
    if Path("checkpoints_v2/supcon_best.pt").exists():
        engine = ChessStylometryEngine()
        with pytest.raises(ValueError, match="No valid chess moves found"):
            engine.analyze_games("James Elia")

def test_side_selection():
    dummy_pgn = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0"
    if Path("checkpoints_v2/supcon_best.pt").exists():
        engine = ChessStylometryEngine()
        parsed_black = engine.parse_pgn_text(dummy_pgn, side="black")
        assert parsed_black[0]["target_color"] == chess.BLACK

        parsed_white = engine.parse_pgn_text(dummy_pgn, side="white")
        assert parsed_white[0]["target_color"] == chess.WHITE

def test_max_games_cap():
    dummy_game = "[Event \"Test\"]\n\n1. e4 e5 2. Nf3 Nc6 1-0\n\n"
    # Create PGN with 30 games
    multi_pgn = dummy_game * 30
    if Path("checkpoints_v2/supcon_best.pt").exists():
        engine = ChessStylometryEngine()
        result = engine.analyze_games(multi_pgn, side="white")
        assert result["games_analyzed"] == 25
        assert len(result["games"]) == 25
        assert result["truncated_notice"] is not None
        assert "first 25 of 30 games" in result["truncated_notice"]

def test_fetch_online_games_validation():
    from app.main import fetch_online_games
    with pytest.raises(ValueError, match="Please provide a valid username"):
        fetch_online_games("lichess", "   ")

    with pytest.raises(ValueError, match="Unsupported platform"):
        fetch_online_games("unknown_plat", "magnus")

def test_fetch_online_games_upstream_errors(monkeypatch):
    import ssl
    import urllib.error
    from app.main import UpstreamFetchError, fetch_online_games

    # UpstreamFetchError must stay a ValueError so broad handlers keep working.
    assert issubclass(UpstreamFetchError, ValueError)

    # TLS interception (DNS filter / proxy presenting the wrong certificate)
    def fake_tls_error(req, timeout=10):
        raise urllib.error.URLError(
            ssl.SSLCertVerificationError(
                1, "certificate verify failed: Hostname mismatch, certificate is not valid for 'api.chess.com'"
            )
        )

    monkeypatch.setattr("app.main.urllib.request.urlopen", fake_tls_error)
    with pytest.raises(UpstreamFetchError, match="TLS certificate verification failed"):
        fetch_online_games("chesscom", "JamesElia")
    with pytest.raises(UpstreamFetchError, match="DNS filter"):
        fetch_online_games("lichess", "JamesElia")

    # Plain unreachable upstream (connection refused / timeout)
    def fake_unreachable(req, timeout=10):
        raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))

    monkeypatch.setattr("app.main.urllib.request.urlopen", fake_unreachable)
    with pytest.raises(UpstreamFetchError, match="Could not reach Chess.com"):
        fetch_online_games("chesscom", "JamesElia")

def test_fetch_online_games_user_errors_not_misclassified(monkeypatch):
    import urllib.error
    from app.main import fetch_online_games

    # HTTP 404 is a user-input problem (unknown username), not an upstream outage.
    def fake_404(req, timeout=10):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr("app.main.urllib.request.urlopen", fake_404)
    with pytest.raises(ValueError, match="not found"):
        fetch_online_games("chesscom", "NoSuchUserHere")
    with pytest.raises(ValueError, match="not found"):
        fetch_online_games("lichess", "NoSuchUserHere")

