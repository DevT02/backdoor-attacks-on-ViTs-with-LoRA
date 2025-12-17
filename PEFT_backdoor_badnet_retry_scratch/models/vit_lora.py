import torch
import torch.nn as nn
import torch.nn.functional as F
from .LoRA import LoRAConfig, LoRALinear

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them"""
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (batch, channels, H, W)
        x = self.projection(x)  # (batch, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)  # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, embed_dim)
        return x


class MultiHeadAttentionLoRA(nn.Module):
    """Multi-head attention with LoRA applied"""
    def __init__(self, embed_dim, num_heads, config: LoRAConfig = None, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Use LoRA if config provided; otherwise regular Linear
        if config is not None:
            self.qkv = LoRALinear(embed_dim, embed_dim * 3, config=config)
            self.out = LoRALinear(embed_dim, embed_dim, config=config)
        else:
            self.qkv = nn.Linear(embed_dim, embed_dim * 3)
            self.out = nn.Linear(embed_dim, embed_dim)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape
        
        # QKV projection
        qkv = self.qkv(x)  # (batch, num_patches, 3*embed_dim)
        qkv = qkv.reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, num_patches, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Combine heads
        out = (attn @ v).transpose(1, 2)  # (batch, num_patches, num_heads, head_dim)
        out = out.reshape(batch_size, num_patches, embed_dim)
        out = self.out(out)
        
        return out


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, config: LoRAConfig = None, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionLoRA(embed_dim, num_heads, config, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden = int(embed_dim * mlp_ratio)
        if config is not None:
            self.mlp = nn.Sequential(
                LoRALinear(embed_dim, mlp_hidden, config=config),
                nn.GELU(),
                nn.Dropout(dropout),
                LoRALinear(mlp_hidden, embed_dim, config=config),
                nn.Dropout(dropout)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, embed_dim),
                nn.Dropout(dropout)
            )
        
    def forward(self, x):
        # Attention block with residual
        x = x + self.attn(self.norm1(x))
        # MLP block with residual
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerLoRA(nn.Module):
    """Vision Transformer with LoRA"""
    def __init__(self, img_size=28, patch_size=4, in_channels=1, num_classes=10, 
                 embed_dim=64, depth=4, num_heads=4, mlp_ratio=4, config: LoRAConfig = None, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Learnable class token and position embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, config, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        if config is not None:
            self.head = LoRALinear(embed_dim, num_classes, config=config)
        else:
            self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches+1, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Classification (use class token)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        return F.softmax(logits, dim=-1)


def load_vit_model(dataname, config: LoRAConfig = None):
    """Factory function to create ViT model"""
    if dataname == 'mnist':
        model = VisionTransformerLoRA(
            img_size=28, 
            patch_size=4, 
            in_channels=1, 
            num_classes=10,
            embed_dim=128,  # Increased from 64
            depth=6,        # Increased from 4
            num_heads=4,
            config=config,
            dropout=0.1
        )
    elif dataname == 'cifar10':
        model = VisionTransformerLoRA(
            img_size=32, 
            patch_size=4, 
            in_channels=3, 
            num_classes=10,
            embed_dim=256,  # Increased from 128
            depth=8,        # Increased from 6
            num_heads=8,    # Increased from 4
            config=config,
            dropout=0.1
        )
    else:
        raise ValueError(f"Unknown dataset: {dataname}")
    
    # Initialize parameters properly
    model.apply(_init_vit_weights)
    return model


def _init_vit_weights(module):
    """Initialize ViT weights properly"""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

