import os
import random
import torch
from torch.utils.data import Dataset, random_split
from pathlib import Path

class ChessDataset(Dataset):
    def __init__(self, data_source: str | Path | list[Path | str], in_memory: bool = False):
        if isinstance(data_source, (str, Path)):
            self.data_dir = Path(data_source)
            self.file_paths = sorted(list(self.data_dir.glob("**/*.pt")))
        else:
            self.file_paths = [Path(p) for p in data_source]
            self.data_dir = None
            
        self.in_memory = in_memory
        self.cached_data = None
        if self.in_memory and len(self.file_paths) > 0:
            print(f"Pre-caching {len(self.file_paths)} games into RAM for high-throughput training...")
            self.cached_data = []
            for path in self.file_paths:
                d = torch.load(path, weights_only=True)
                self.cached_data.append({
                    "boards": d["boards"],
                    "padding_mask": d["padding_mask"],
                    "player_label": d["player_label"],
                    "actual_length": d["actual_length"]
                })
            print("Caching complete!")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if self.cached_data is not None:
            data = self.cached_data[idx]
            boards = data["boards"].to(dtype=torch.float32)
            return {
                "boards": boards,
                "padding_mask": data["padding_mask"],
                "player_label": data["player_label"],
                "actual_length": data["actual_length"]
            }

        path = self.file_paths[idx]
        data = torch.load(path, weights_only=True)

        # Cast boards to float32 (on disk they are uint8)
        boards = data["boards"].to(dtype=torch.float32)
        padding_mask = data["padding_mask"]
        player_label = data["player_label"]
        actual_length = data["actual_length"]

        return {
            "boards": boards,
            "padding_mask": padding_mask,
            "player_label": player_label,
            "actual_length": actual_length
        }

def get_stratified_train_test_datasets(data_dir: str | Path, seed: int = 42, in_memory: bool = False):
    """Splits master games into stratified train and held-out test sets.
    
    Rule:
    - If master has > 2000 games: hold out 100 games
    - If master has <= 2000 games: hold out 25 games
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"Directory {data_dir} does not exist.")

    rng = random.Random(seed)
    train_paths = []
    test_paths = []

    player_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    for p_dir in player_dirs:
        files = sorted(list(p_dir.glob("*.pt")))
        if len(files) == 0:
            continue
        
        # Deterministic shuffle per master
        shuffled = list(files)
        rng.shuffle(shuffled)
        
        holdout_count = 100 if len(files) > 2000 else 25
        holdout_count = min(holdout_count, len(files)) # Safety check
        
        test_paths.extend(shuffled[:holdout_count])
        train_paths.extend(shuffled[holdout_count:])

    print(f"Stratified Split: {len(train_paths)} training games, {len(test_paths)} held-out test games.")
    train_dataset = ChessDataset(train_paths, in_memory=in_memory)
    test_dataset = ChessDataset(test_paths, in_memory=in_memory)
    return train_dataset, test_dataset

def get_train_val_datasets(data_dir: str | Path, val_split: float = 0.1, seed: int = 42):
    """Splits the preprocessed files into train and validation datasets."""
    dataset = ChessDataset(data_dir)
    if len(dataset) == 0:
        raise ValueError(f"No preprocessed files found in {data_dir}. Run preprocess_games.py first.")

    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=generator)

    return train_dataset, val_dataset

