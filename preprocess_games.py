import os
import re
import sys
import chess.pgn
import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Define the 22 Grandmasters in sorted order
GRANDMASTERS = [
    "Alekhine", "Anand", "Aronian", "Botvinnik", "Capablanca",
    "Carlsen", "Caruana", "Chigorin", "Ding", "Fischer",
    "Giri", "Karpov", "Kasparov", "Kramnik", "Lasker",
    "Mamedyarov", "Morphy", "Nakamura", "Nepomniachtchi", "So",
    "Tal", "VachierLagrave"
]

GM_TO_LABEL = {gm: idx for idx, gm in enumerate(GRANDMASTERS)}

def clean_name(name: str) -> str:
    """Lowercase and strip non-alphanumeric characters."""
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

def board_to_18_channel(board: chess.Board, target_color: chess.Color) -> np.ndarray:
    """Converts a board state to a strict 18-channel 8x8 representation.
    
    Channels:
    - 0-5: Player piece types (P, N, B, R, Q, K)
    - 6-11: Opponent piece types (P, N, B, R, Q, K)
    - 12-13: Player castling rights (kingside, queenside)
    - 14-15: Opponent castling rights (kingside, queenside)
    - 16: Active Turn (1 if target's turn, 0 otherwise)
    - 17: En Passant square (1 at the EP square, 0 elsewhere)
    """
    player_color = target_color
    opponent_color = not target_color
    
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    
    # 12 piece channels
    piece_arrays = []
    for p_type in piece_types:
        squareset = board.pieces(p_type, player_color)
        arr = np.array(squareset.tolist(), dtype=np.uint8).reshape((8, 8))
        arr = np.flipud(arr)
        piece_arrays.append(arr)
        
    for p_type in piece_types:
        squareset = board.pieces(p_type, opponent_color)
        arr = np.array(squareset.tolist(), dtype=np.uint8).reshape((8, 8))
        arr = np.flipud(arr)
        piece_arrays.append(arr)
        
    # Castling rights (4 channels)
    castling_arrays = [
        np.ones((8, 8), dtype=np.uint8) * int(board.has_kingside_castling_rights(player_color)),
        np.ones((8, 8), dtype=np.uint8) * int(board.has_queenside_castling_rights(player_color)),
        np.ones((8, 8), dtype=np.uint8) * int(board.has_kingside_castling_rights(opponent_color)),
        np.ones((8, 8), dtype=np.uint8) * int(board.has_queenside_castling_rights(opponent_color)),
    ]
    
    # Active turn (1 channel)
    turn_val = 1 if board.turn == player_color else 0
    turn_array = np.ones((8, 8), dtype=np.uint8) * turn_val
    
    # En Passant (1 channel)
    ep_array = np.zeros((8, 8), dtype=np.uint8)
    if board.ep_square is not None:
        ep_flat = np.zeros(64, dtype=np.uint8)
        ep_flat[board.ep_square] = 1
        ep_array = ep_flat.reshape((8, 8))
        ep_array = np.flipud(ep_array)
        
    all_arrays = piece_arrays + castling_arrays + [turn_array, ep_array]
    tensor = np.stack(all_arrays, axis=0)
    
    # POV Normalization
    if target_color == chess.BLACK:
        tensor = np.flip(tensor, axis=1) # Flip ranks vertically
        
    return tensor

def preprocess_single_pgn(pgn_path: Path, out_dir: Path) -> dict:
    """Processes a single PGN file, extracts games, and saves them to disk."""
    stem = pgn_path.stem
    label = GM_TO_LABEL.get(stem)
    if label is None:
        return {"player": stem, "status": "ignored", "count": 0}
        
    player_out_dir = out_dir / stem
    player_out_dir.mkdir(exist_ok=True, parents=True)
    
    stem_clean = clean_name(stem)
    
    count = 0
    with open(pgn_path, "r", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
                
            white_name = game.headers.get("White", "")
            black_name = game.headers.get("Black", "")
            
            # Determine target player color
            white_clean = clean_name(white_name)
            black_clean = clean_name(black_name)
            
            if stem_clean in white_clean:
                target_color = chess.WHITE
            elif stem_clean in black_clean:
                target_color = chess.BLACK
            else:
                # Default to White if we can't determine, or skip if needed.
                target_color = chess.WHITE
                
            # Process board sequence
            board = game.board()
            states = []
            
            # Step 0: Initial position
            states.append(board_to_18_channel(board, target_color))
            
            # Subsequence steps
            for move in game.mainline_moves():
                if len(states) >= 64:
                    break
                board.push(move)
                states.append(board_to_18_channel(board, target_color))
                
            actual_length = len(states)
            
            # Pad if shorter than 64
            while len(states) < 64:
                states.append(np.zeros((18, 8, 8), dtype=np.uint8))
                
            boards_np = np.stack(states, axis=0)
            
            # Padding mask: True where we padded (i.e. indices >= actual_length)
            padding_mask_np = np.zeros(64, dtype=bool)
            if actual_length < 64:
                padding_mask_np[actual_length:] = True
                
            # Convert to PyTorch tensors (uint8 for boards, bool for mask)
            boards_tensor = torch.from_numpy(boards_np)
            padding_mask_tensor = torch.from_numpy(padding_mask_np)
            
            # Save the game data
            game_data = {
                "boards": boards_tensor,
                "padding_mask": padding_mask_tensor,
                "player_label": torch.tensor(label, dtype=torch.long),
                "actual_length": torch.tensor(actual_length, dtype=torch.long)
            }
            
            out_file = player_out_dir / f"game_{count:06d}.pt"
            torch.save(game_data, out_file)
            count += 1
            
    return {"player": stem, "status": "success", "count": count}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn_dir", type=str, default="concat_master_games")
    parser.add_argument("--out_dir", type=str, default="preprocessed_data")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    
    pgn_dir = Path(args.pgn_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    pgn_files = list(pgn_dir.glob("*.pgn"))
    print(f"Found {len(pgn_files)} PGN files to preprocess. Using {args.workers} workers.")
    
    total_games = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(preprocess_single_pgn, f, out_dir): f for f in pgn_files}
        for future in as_completed(futures):
            res = future.result()
            print(f"Player: {res['player']:<20} Status: {res['status']:<10} Games parsed: {res['count']}")
            total_games += res['count']
            
    print(f"Preprocessing complete. Total games preprocessed: {total_games}")

if __name__ == "__main__":
    main()
