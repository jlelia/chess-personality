import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessSpatialFrontEnd(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(18, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.res1 = ResBlock(64)
        self.res2 = ResBlock(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.res3 = ResBlock(128)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        # Input shape: (B * L, 18, 8, 8)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.res3(x)
        x = self.pool(x) # (B * L, 128, 1, 1)
        x = torch.flatten(x, 1) # (B * L, 128)
        return x

class ChessEncoder(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 8, num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.0):
        super().__init__()
        self.cnn = ChessSpatialFrontEnd()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, 65, d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model)) # For masking during JEPA
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            norm_first=True,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Initialize parameters
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        
    def forward(self, boards, padding_mask=None, mask_indices=None):
        # boards: (B, L, 18, 8, 8)
        # padding_mask: (B, L)
        # mask_indices: (B, W) or None
        
        B, L, C, H, W = boards.shape
        boards_flat = boards.view(B * L, C, H, W)
        features_flat = self.cnn(boards_flat) # (B * L, 128)
        features = features_flat.view(B, L, 128) # (B, L, 128)
        
        # Apply latent masking if mask_indices is provided
        if mask_indices is not None:
            # mask_indices is shape (B, W) containing move indices (0 to 63)
            # Create a copy to avoid in-place modification error in autograd if needed
            features = features.clone()
            for i in range(B):
                features[i, mask_indices[i]] = self.mask_token
                
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, 128)
        x = torch.cat((cls_tokens, features), dim=1) # (B, L+1, 128)
        
        # Add Positional Embedding
        x = x + self.pos_embedding
        
        # Create key padding mask for Transformer
        # Transformer key padding mask shape: (B, L+1)
        if padding_mask is not None:
            cls_pad_mask = torch.zeros((B, 1), dtype=torch.bool, device=boards.device)
            transformer_pad_mask = torch.cat((cls_pad_mask, padding_mask), dim=1) # (B, L+1)
        else:
            transformer_pad_mask = None
            
        out = self.transformer(x, src_key_padding_mask=transformer_pad_mask)
        return out

class ChessPredictor(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 512, dropout: float = 0.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            norm_first=True,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x, padding_mask=None):
        # x: (B, L+1, 128)
        # padding_mask: (B, L+1) or (B, L)
        if padding_mask is not None:
            if padding_mask.shape[1] == x.shape[1] - 1:
                cls_pad_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device)
                padding_mask = torch.cat((cls_pad_mask, padding_mask), dim=1)
        return self.transformer(x, src_key_padding_mask=padding_mask)

class SupConProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 128, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, latent_dim)
        )
        
    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, p=2, dim=-1)
