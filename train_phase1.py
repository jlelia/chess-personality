import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import copy

from data_pipeline import get_stratified_train_test_datasets, get_train_val_datasets
from model import ChessEncoder, ChessPredictor

def get_optimizer_params(encoder, predictor, weight_decay):
    decay = set()
    no_decay = set()
    
    for model_name, model in [("encoder", encoder), ("predictor", predictor)]:
        for pn, p in model.named_parameters():
            fpn = f"{model_name}.{pn}"
            
            # Exclude biases, 1D normalization params, and special token embeddings from decay
            if pn.endswith('bias') or p.ndim < 2 or 'token' in pn or 'embedding' in pn:
                no_decay.add(fpn)
            else:
                decay.add(fpn)
                
    param_dict = {}
    for model_name, model in [("encoder", encoder), ("predictor", predictor)]:
        for pn, p in model.named_parameters():
            fpn = f"{model_name}.{pn}"
            param_dict[fpn] = p
            
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"Intersection: {inter_params}"
    assert len(param_dict.keys() - union_params) == 0, f"Missing: {param_dict.keys() - union_params}"
    
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    return optim_groups

def get_lr(step, total_steps, eta_max, warmup_ratio=0.1):
    warmup_steps = int(warmup_ratio * total_steps)
    if step < warmup_steps:
        return eta_max * (step / max(1, warmup_steps))
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        eta_min = eta_max / 100.0
        return eta_min + 0.5 * (eta_max - eta_min) * (1.0 + math.cos(math.pi * progress))

def get_ema_beta(step, total_steps, beta_start=0.996, beta_end=1.0):
    progress = step / max(1, total_steps)
    return beta_end - (beta_end - beta_start) * 0.5 * (1.0 + math.cos(math.pi * progress))

@torch.no_grad()
def update_target_ema(context_encoder, target_encoder, beta):
    for param_c, param_t in zip(context_encoder.parameters(), target_encoder.parameters()):
        param_t.data.mul_(beta).add_(param_c.data, alpha=1.0 - beta)

def generate_mask_indices(actual_lengths, device):
    B = actual_lengths.shape[0]
    W_batch = torch.clamp((0.15 * actual_lengths).long(), min=5, max=10)
    
    mask_indices = []
    for i in range(B):
        n = actual_lengths[i].item()
        w = W_batch[i].item()
        if n <= w:
            w = max(1, n)
            s = 0
        else:
            s = torch.randint(0, n - w + 1, (1,)).item()
        indices = list(range(s, s + w))
        mask_indices.append(torch.tensor(indices, dtype=torch.long, device=device))
    return mask_indices

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="preprocessed_data")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--in_memory", action="store_true", help="Cache dataset tensors into RAM")
    args = parser.parse_args()
    
    # Device selection (MPS, CUDA, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load dataset using stratified holdout
    train_dataset, val_dataset = get_stratified_train_test_datasets(args.data_dir, seed=42, in_memory=args.in_memory)
    
    # To handle massive batch sizes on hardware with limited memory, we support gradient accumulation
    target_batch_size = args.batch_size
    physical_batch_size = min(256, target_batch_size)
    accum_steps = max(1, target_batch_size // physical_batch_size)
    print(f"Target Batch Size: {target_batch_size}, Physical Batch Size: {physical_batch_size}, Accumulation Steps: {accum_steps}")
    
    train_loader = DataLoader(train_dataset, batch_size=physical_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=physical_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    
    # Initialize models
    context_encoder = ChessEncoder().to(device)
    target_encoder = copy.deepcopy(context_encoder).to(device)
    # Target encoder has no gradients
    for p in target_encoder.parameters():
        p.requires_grad = False
        
    predictor = ChessPredictor().to(device)
    
    # Optimizer
    optim_groups = get_optimizer_params(context_encoder, predictor, args.weight_decay)
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr)
    
    # Mixed precision setup
    # Use generic amp autocast
    autocast_device = "cuda" if device.type == "cuda" else ("cpu" if device.type == "cpu" else "mps")
    # Note: MPS autocast is supported in PyTorch 2.0+, but let's be safe.
    # GradScaler is only used on CUDA
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    
    total_steps = args.epochs * len(train_loader)
    print(f"Total training steps: {total_steps}")
    
    Path(args.save_dir).mkdir(exist_ok=True, parents=True)
    
    global_step = 0
    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        context_encoder.train()
        predictor.train()
        train_loss = 0.0
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            # Dynamic Schedules
            lr = get_lr(global_step, total_steps, args.lr)
            beta = get_ema_beta(global_step, total_steps)
            
            for g in optimizer.param_groups:
                g['lr'] = lr
                
            boards = batch["boards"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            actual_length = batch["actual_length"].to(device)
            
            B = boards.shape[0]
            
            # Generate random masking windows
            mask_indices = generate_mask_indices(actual_length, device)
            
            # Run Target Encoder (No gradients)
            with torch.no_grad():
                # Target encoder processes unmasked sequence
                targets = target_encoder(boards, padding_mask=padding_mask) # (B, 65, 128)
                
            # Run Context Encoder (with masking) and Predictor (with AMP)
            # MPS supports float16 or bfloat16 in autocast depending on versions.
            # Using autocast safely.
            try:
                with torch.amp.autocast(device_type=autocast_device, dtype=torch.bfloat16):
                    # Context encoder processes masked sequence
                    context_out = context_encoder(boards, padding_mask=padding_mask, mask_indices=mask_indices) # (B, 65, 128)
                    
                    # Predictor tries to predict targets from context
                    predictions = predictor(context_out, padding_mask=padding_mask) # (B, 65, 128)
                    
                    # Compute L2 Loss on masked positions only
                    loss = 0.0
                    total_masked = 0
                    for i in range(B):
                        idx = mask_indices[i] + 1 # offset by 1 for [CLS]
                        pred = predictions[i, idx]
                        target = targets[i, idx]
                        loss += F.mse_loss(pred, target, reduction='sum')
                        total_masked += idx.shape[0]
                    loss = (loss / total_masked) / accum_steps
            except Exception as e:
                # Fallback to standard precision if AMP fails on this device
                context_out = context_encoder(boards, padding_mask=padding_mask, mask_indices=mask_indices)
                predictions = predictor(context_out, padding_mask=padding_mask)
                loss = 0.0
                total_masked = 0
                for i in range(B):
                    idx = mask_indices[i] + 1
                    pred = predictions[i, idx]
                    target = targets[i, idx]
                    loss += F.mse_loss(pred, target, reduction='sum')
                    total_masked += idx.shape[0]
                loss = (loss / total_masked) / accum_steps
                
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            train_loss += loss.item() * accum_steps
            
            # Step optimizer after accumulation
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                
                # Update Target Encoder EMA
                update_target_ema(context_encoder, target_encoder, beta)
                
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  [Epoch {epoch:02d}/{args.epochs:02d}] Step {batch_idx+1:03d}/{len(train_loader):03d} | Batch Loss: {loss.item() * accum_steps:.6f} | LR: {lr:.6f} | EMA Beta: {beta:.6f}", flush=True)

            global_step += 1
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Phase
        context_encoder.eval()
        predictor.eval()
        val_loss = 0.0
        total_val_masked = 0
        
        with torch.no_grad():
            for batch in val_loader:
                boards = batch["boards"].to(device)
                padding_mask = batch["padding_mask"].to(device)
                actual_length = batch["actual_length"].to(device)
                
                B = boards.shape[0]
                mask_indices = generate_mask_indices(actual_length, device)
                
                # Target Encoder (unmasked)
                targets = target_encoder(boards, padding_mask=padding_mask)
                
                # Context Encoder (masked)
                context_out = context_encoder(boards, padding_mask=padding_mask, mask_indices=mask_indices)
                
                # Predictor
                predictions = predictor(context_out, padding_mask=padding_mask)
                
                loss_batch = 0.0
                for i in range(B):
                    idx = mask_indices[i] + 1
                    pred = predictions[i, idx]
                    target = targets[i, idx]
                    loss_batch += F.mse_loss(pred, target, reduction='sum')
                    total_val_masked += idx.shape[0]
                val_loss += loss_batch.item()
                
        avg_val_loss = val_loss / max(1, total_val_masked)
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {lr:.6f} | EMA Beta: {beta:.6f}")
        
        # Save checkpoints
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(args.save_dir, "jepa_best.pt")
            torch.save({
                "epoch": epoch,
                "context_encoder_state_dict": context_encoder.state_dict(),
                "target_encoder_state_dict": target_encoder.state_dict(),
                "predictor_state_dict": predictor.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss
            }, checkpoint_path)
            print(f"Saved new best checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()
