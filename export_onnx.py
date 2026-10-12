#!/usr/bin/env python3
"""
ONNX Export Utility for Chess Personality Indexer
Exports trained PyTorch ChessEncoder + SupConProjectionHead to an optimized ONNX model
for high-speed, low-memory inference on Raspberry Pi 5 CPU and edge accelerators.
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn

from model import ChessEncoder, SupConProjectionHead

class CombinedStylometryModel(nn.Module):
    """Wraps ChessEncoder and SupConProjectionHead into a single end-to-end pipeline."""
    def __init__(self, encoder: ChessEncoder, proj_head: SupConProjectionHead):
        super().__init__()
        self.encoder = encoder
        self.proj_head = proj_head

    def forward(self, boards: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            boards: Tensor of shape (B, 64, 18, 8, 8) representing 18-channel board states
            padding_mask: Bool tensor of shape (B, 64) where True indicates padded ply
        Returns:
            Unit-normalized 128-dimensional latent embedding vector of shape (B, 128)
        """
        out = self.encoder(boards, padding_mask=padding_mask)
        cls_rep = out[:, 0, :]
        return self.proj_head(cls_rep)

def export_to_onnx(checkpoint_path: str, output_path: str, opset_version: int = 17):
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        print(f"Error: Checkpoint file not found at {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(ckpt_file, map_location="cpu")

    encoder = ChessEncoder(d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.0)
    proj_head = SupConProjectionHead(input_dim=128, latent_dim=128)

    if "encoder_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        proj_head.load_state_dict(checkpoint["projection_head_state_dict"])
    elif "encoder" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder"])
        proj_head.load_state_dict(checkpoint["proj_head"])
    else:
        raise KeyError(f"Unexpected keys in checkpoint: {list(checkpoint.keys())}")

    model = CombinedStylometryModel(encoder, proj_head)
    model.eval()

    # Dummy inputs for tracing
    dummy_boards = torch.randn(1, 64, 18, 8, 8, dtype=torch.float32)
    dummy_mask = torch.zeros(1, 64, dtype=torch.bool)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting model to ONNX: {output_path} (opset {opset_version})...")
    try:
        import onnx
    except ImportError:
        print("\n[Optional Dependency Notice]")
        print("Exporting to ONNX requires the 'onnx' package.")
        print("To install dependencies and export, run:")
        print("    uv add --dev onnx onnxscript\n")
        return False

    try:
        torch.onnx.export(
            model,
            (dummy_boards, dummy_mask),
            str(out_file),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["boards", "padding_mask"],
            output_names=["embedding"],
            dynamic_axes={
                "boards": {0: "batch_size"},
                "padding_mask": {0: "batch_size"},
                "embedding": {0: "batch_size"}
            },
            dynamo=False
        )
        print(f"Successfully exported ONNX model ({out_file.stat().st_size / (1024 * 1024):.2f} MB) to {out_file}")
        return True
    except Exception as e:
        print(f"\nONNX export failed: {e}")
        print("To install compatible exporter dependencies, run: uv add --dev onnx onnxscript")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Chess Stylometry Model to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_v2/supcon_best.pt", help="Path to supcon_best.pt")
    parser.add_argument("--output", type=str, default="checkpoints_v2/chess_stylometry.onnx", help="Path to export .onnx file")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output, args.opset)
