import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

from data_pipeline import get_stratified_train_test_datasets, get_train_val_datasets
from model import ChessEncoder, SupConProjectionHead
from preprocess_games import GRANDMASTERS, GM_TO_LABEL

class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: (B, d) - must be L2 normalized
        # labels: (B,)
        device = features.device
        B = features.shape[0]
        
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(B).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # Cosine similarity matrix (scaled by temperature)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Denominator of log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        sum_exp_logits = torch.sum(exp_logits, dim=1, keepdim=True)
        
        # Log probability
        log_prob = logits - torch.log(sum_exp_logits + 1e-8)
        
        # Sum over positive pairs for each anchor i
        sum_mask = torch.sum(mask, dim=1)
        
        valid_indices = sum_mask > 0
        if not torch.any(valid_indices):
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        mean_log_prob_pos = torch.sum(mask * log_prob, dim=1)[valid_indices] / sum_mask[valid_indices]
        
        # Loss is negative mean log probability
        loss = -mean_log_prob_pos.mean()
        return loss

def get_optimizer_params_phase2(encoder, projection_head, weight_decay):
    decay = set()
    no_decay = set()
    
    for model_name, model in [("encoder", encoder), ("proj", projection_head)]:
        for pn, p in model.named_parameters():
            if not p.requires_grad:
                continue
            fpn = f"{model_name}.{pn}"
            
            # Exclude biases, 1D normalization params, and special token embeddings from decay
            if pn.endswith('bias') or p.ndim < 2 or 'token' in pn or 'embedding' in pn:
                no_decay.add(fpn)
            else:
                decay.add(fpn)
                
    param_dict = {}
    for model_name, model in [("encoder", encoder), ("proj", projection_head)]:
        for pn, p in model.named_parameters():
            if not p.requires_grad:
                continue
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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="preprocessed_data")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jepa_best.pt")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--temp", type=float, default=0.07)
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
    
    # Support gradient accumulation
    target_batch_size = args.batch_size
    physical_batch_size = min(256, target_batch_size)
    accum_steps = max(1, target_batch_size // physical_batch_size)
    print(f"Target Batch Size: {target_batch_size}, Physical Batch Size: {physical_batch_size}, Accumulation Steps: {accum_steps}")
    
    train_loader = DataLoader(train_dataset, batch_size=physical_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=physical_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    
    # Initialize encoder
    encoder = ChessEncoder().to(device)
    
    # Load Phase 1 weights if available
    if os.path.exists(args.checkpoint):
        print(f"Loading pre-trained encoder weights from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        encoder.load_state_dict(checkpoint["context_encoder_state_dict"])
    else:
        print(f"Warning: Pre-trained checkpoint '{args.checkpoint}' not found. Training from scratch.")
        
    # Freezing: CNN front-end and first 2 layers of the Transformer encoder
    for p in encoder.cnn.parameters():
        p.requires_grad = False
        
    for layer_idx, layer in enumerate(encoder.transformer.layers):
        if layer_idx < 2:
            for p in layer.parameters():
                p.requires_grad = False
        else:
            for p in layer.parameters():
                p.requires_grad = True
                
    # Initialize projection head
    projection_head = SupConProjectionHead().to(device)
    
    # Optimizer & Loss
    optim_groups = get_optimizer_params_phase2(encoder, projection_head, args.weight_decay)
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr)
    criterion = SupConLoss(temperature=args.temp)
    
    # Mixed precision setup
    autocast_device = "cuda" if device.type == "cuda" else ("cpu" if device.type == "cpu" else "mps")
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    
    total_steps = args.epochs * len(train_loader)
    print(f"Total training steps: {total_steps}")
    
    global_step = 0
    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        projection_head.train()
        train_loss = 0.0
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            # Dynamic LR
            lr = get_lr(global_step, total_steps, args.lr)
            for g in optimizer.param_groups:
                g['lr'] = lr
                
            boards = batch["boards"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            labels = batch["player_label"].to(device)
            
            # Run forward pass (with AMP)
            try:
                with torch.amp.autocast(device_type=autocast_device, dtype=torch.bfloat16):
                    # Pass full, unmasked sequence
                    out = encoder(boards, padding_mask=padding_mask) # (B, 65, 128)
                    
                    # Extract CLS token representation (index 0)
                    z_CLS = out[:, 0] # (B, 128)
                    
                    # Project to unit sphere
                    z_proj = projection_head(z_CLS) # (B, 128)
                    
                    # Compute SupCon loss
                    loss = criterion(z_proj, labels) / accum_steps
            except Exception as e:
                # Fallback to standard precision if AMP fails on this device
                out = encoder(boards, padding_mask=padding_mask)
                z_CLS = out[:, 0]
                z_proj = projection_head(z_CLS)
                loss = criterion(z_proj, labels) / accum_steps
                
            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            train_loss += loss.item() * accum_steps
            
            # Optimizer step after accumulation
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  [Epoch {epoch:02d}/{args.epochs:02d}] Step {batch_idx+1:03d}/{len(train_loader):03d} | SupCon Batch Loss: {loss.item() * accum_steps:.6f} | LR: {lr:.6f}", flush=True)

            global_step += 1
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Phase
        encoder.eval()
        projection_head.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                boards = batch["boards"].to(device)
                padding_mask = batch["padding_mask"].to(device)
                labels = batch["player_label"].to(device)
                
                out = encoder(boards, padding_mask=padding_mask)
                z_CLS = out[:, 0]
                z_proj = projection_head(z_CLS)
                loss_batch = criterion(z_proj, labels)
                val_loss += loss_batch.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {lr:.6f}")
        
        # Save latest epoch checkpoint so training progress is never lost
        latest_path = os.path.join(args.save_dir, "supcon_latest.pt")
        torch.save({
            "epoch": epoch,
            "encoder_state_dict": encoder.state_dict(),
            "projection_head_state_dict": projection_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss
        }, latest_path)

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(args.save_dir, "supcon_best.pt")
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": encoder.state_dict(),
                "projection_head_state_dict": projection_head.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss
            }, checkpoint_path)
            print(f"Saved new best checkpoint to {checkpoint_path}")

    # Load best checkpoint weights before computing centroids
    best_ckpt_path = os.path.join(args.save_dir, "supcon_best.pt")
    if os.path.exists(best_ckpt_path):
        print(f"\nLoading best checkpoint {best_ckpt_path} to compute Grandmaster cluster centroids...")
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        projection_head.load_state_dict(ckpt["projection_head_state_dict"])

    encoder.eval()
    projection_head.eval()

    print("Computing 22 Grandmaster reference centroids from training set...")
    num_classes = len(GRANDMASTERS)
    centroid_sums = torch.zeros((num_classes, 128), device=device)
    centroid_counts = torch.zeros(num_classes, device=device)

    centroid_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in centroid_loader:
            boards = batch["boards"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            labels = batch["player_label"].to(device)

            out = encoder(boards, padding_mask=padding_mask)
            z_cls = out[:, 0]
            z_proj = projection_head(z_cls)
            z_proj = F.normalize(z_proj, p=2, dim=-1)

            for i in range(len(labels)):
                lbl = labels[i].item()
                centroid_sums[lbl] += z_proj[i]
                centroid_counts[lbl] += 1

    valid_mask = centroid_counts > 0
    centroids = torch.zeros_like(centroid_sums)
    centroids[valid_mask] = centroid_sums[valid_mask] / centroid_counts[valid_mask].unsqueeze(1)
    centroids = F.normalize(centroids, p=2, dim=-1)

    centroids_path = os.path.join(args.save_dir, "gm_centroids.pt")
    torch.save({
        "centroids": centroids.cpu(),
        "grandmasters": GRANDMASTERS,
        "gm_to_label": GM_TO_LABEL,
        "counts": centroid_counts.cpu()
    }, centroids_path)
    print(f"Successfully saved 22 Grandmaster reference centroids to {centroids_path}!\n")

if __name__ == "__main__":
    main()
