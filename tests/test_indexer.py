import os
import torch
import torch.nn.functional as F
import numpy as np
import pytest
import chess
from pathlib import Path

from preprocess_games import board_to_18_channel, preprocess_single_pgn, GRANDMASTERS
from data_pipeline import ChessDataset, get_train_val_datasets
from model import ChessEncoder, ChessPredictor, SupConProjectionHead
from train_phase1 import get_optimizer_params
from train_phase2 import SupConLoss

def test_pov_flipping():
    # Construct a standard starting chess board
    board = chess.Board()
    
    # Generate 18-channel representation for White perspective
    tensor_w = board_to_18_channel(board, chess.WHITE)
    assert tensor_w.shape == (18, 8, 8)
    
    # Generate 18-channel representation for Black perspective
    tensor_b = board_to_18_channel(board, chess.BLACK)
    assert tensor_b.shape == (18, 8, 8)
    
    # Verify pawn ranks (White pawns on Rank 2 -> row 6; Black pawns on Rank 7 -> flipped to row 6)
    assert (tensor_w[0, 6, :] == 1).all()
    assert (tensor_b[0, 6, :] == 1).all()

def test_pipeline_preprocessing(tmp_path):
    out_dir = tmp_path / "preprocessed"
    
    # Write a dummy PGN for Carlsen
    dummy_pgn = tmp_path / "Carlsen.pgn"
    with open(dummy_pgn, "w") as f:
        f.write('[Event "Dummy Event"]\n[White "Carlsen, Magnus"]\n[Black "Anand, Viswanathan"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 1-0\n')
        
    res = preprocess_single_pgn(dummy_pgn, out_dir)
    assert res["status"] == "success"
    assert res["count"] == 1
    
    saved_file = out_dir / "Carlsen" / "game_000000.pt"
    assert saved_file.exists()
    
    data = torch.load(saved_file, weights_only=True)
    assert "boards" in data
    assert "padding_mask" in data
    assert "player_label" in data
    assert "actual_length" in data
    
    assert data["boards"].shape == (64, 18, 8, 8)
    assert data["padding_mask"].shape == (64,)
    assert data["player_label"].item() == 5  # Carlsen label index
    assert data["actual_length"].item() == 6  # Initial state + 5 plies

def test_model_forward():
    encoder = ChessEncoder()
    predictor = ChessPredictor()
    proj_head = SupConProjectionHead()
    
    B, L = 4, 64
    boards = torch.randn(B, L, 18, 8, 8)
    padding_mask = torch.zeros(B, L, dtype=torch.bool)
    
    # Phase 1: Forward (with masking)
    mask_indices = [torch.tensor([5, 6, 7, 8, 9]) for _ in range(B)]
    context_out = encoder(boards, padding_mask=padding_mask, mask_indices=mask_indices)
    assert context_out.shape == (B, 65, 128)
    
    predictions = predictor(context_out, padding_mask=None)
    assert predictions.shape == (B, 65, 128)
    
    # Phase 2: Forward
    out_full = encoder(boards, padding_mask=padding_mask)
    z_CLS = out_full[:, 0]
    assert z_CLS.shape == (B, 128)
    
    z_proj = proj_head(z_CLS)
    assert z_proj.shape == (B, 128)
    
    # Assert unit-sphere normalization
    norms = torch.norm(z_proj, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms))

def test_weight_decay_partition():
    encoder = ChessEncoder()
    predictor = ChessPredictor()
    
    optim_groups = get_optimizer_params(encoder, predictor, weight_decay=0.01)
    assert len(optim_groups) == 2
    assert optim_groups[0]["weight_decay"] == 0.01
    assert optim_groups[1]["weight_decay"] == 0.0
    
    # Ensure param lists are distinct
    decay_params_set = set(id(p) for p in optim_groups[0]["params"])
    no_decay_params_set = set(id(p) for p in optim_groups[1]["params"])
    assert len(decay_params_set.intersection(no_decay_params_set)) == 0

def test_supcon_loss():
    loss_fn = SupConLoss(temperature=0.07)
    B = 8
    features = torch.randn(B, 128)
    features = F.normalize(features, p=2, dim=-1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    
    loss = loss_fn(features, labels)
    assert loss.item() > 0.0
