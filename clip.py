"""
CLIP Training Script
This script implements the CLIP (Contrastive Language-Image Pre-training) model
and training pipeline using PyTorch.
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def tokenizer(text: str, encode: bool = True, mask: torch.Tensor = None,
             max_seq_length: int = 77) -> tuple:
    """
    Tokenize and encode/decode text for CLIP model.
    
    Args:
        text (str): Input text to tokenize
        encode (bool): Whether to encode or decode
        mask (torch.Tensor): Mask for decoding
        max_seq_length (int): Maximum sequence length
        
    Returns:
        tuple: (tokens, mask) if encoding, (text, None) if decoding
    """
    if encode:
        out = chr(2) + text + chr(3)
        
        if len(out) > max_seq_length:
            out = out[:max_seq_length-1] + chr(3)
        
        out = out + "".join([chr(0) for _ in range(max_seq_length - len(out))])
        out = torch.IntTensor(list(out.encode("utf-8")))
        
        n_actual_tokens = min(len(text) + 2, max_seq_length)
        mask = torch.zeros(max_seq_length, dtype=torch.long)
        mask[:n_actual_tokens] = 1
        
    else:
        if mask is None:
            raise ValueError("Mask is required for decoding")
        
        out = [chr(x) for x in text[1:len(mask.nonzero())]]
        out = "".join(out)
        mask = None

    return out, mask


class PositionalEmbedding(nn.Module):
    """Positional embedding layer for transformer models."""
    
    def __init__(self, width: int, max_seq_length: int):
        super().__init__()
        pe = torch.zeros(max_seq_length, width)
        for pos in range(max_seq_length):
            for i in range(width):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos/(10000 ** (i/width)))
                else:
                    pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/width)))
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe


class AttentionHead(nn.Module):
    """Single attention head implementation."""
    
    def __init__(self, width: int, head_size: int):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(width, head_size)
        self.key = nn.Linear(width, head_size)
        self.value = nn.Linear(width, head_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        attention = query @ key.transpose(-2, -1)
        attention = attention / (self.head_size ** 0.5)
        
        if mask is not None:
            attention_mask = mask.unsqueeze(1).expand(-1, seq_len, -1)
            attention = attention.masked_fill(attention_mask == 0, float('-inf'))
        
        attention = torch.softmax(attention, dim=-1)
        return attention @ value


class MultiHeadAttention(nn.Module):
    """Multi-head attention implementation."""
    
    def __init__(self, width: int, n_heads: int):
        super().__init__()
        self.head_size = width // n_heads
        self.heads = nn.ModuleList([
            AttentionHead(width, self.head_size) for _ in range(n_heads)
        ])
        self.output_projection = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        out = torch.cat([head(x, mask=mask) for head in self.heads], dim=-1)
        return self.output_projection(out)


class TransformerEncoder(nn.Module):
    """Transformer encoder block implementation."""
    
    def __init__(self, width: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(width)
        self.attention = MultiHeadAttention(width, n_heads)
        self.layer_norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * mlp_ratio),
            nn.GELU(),
            nn.Linear(width * mlp_ratio, width)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attention(self.layer_norm1(x), mask=mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class ImageEncoder(nn.Module):
    """Image encoder using Vision Transformer architecture."""
    
    def __init__(self, width: int, img_size: tuple, patch_size: tuple,
                 n_channels: int, n_layers: int, n_heads: int, emb_dim: int):
        super().__init__()
        assert img_size[0] % patch_size[0] == 0
        assert img_size[1] % patch_size[1] == 0
        
        self.n_patches = (img_size[0] * img_size[1]) // (patch_size[0] * patch_size[1])
        self.max_seq_length = self.n_patches + 1
        
        self.patch_embedding = nn.Conv2d(
            n_channels, width, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, width))
        self.positional_embedding = PositionalEmbedding(width, self.max_seq_length)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(width, n_heads) for _ in range(n_layers)
        ])
        self.projection = nn.Parameter(torch.randn(width, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(x.size()[0], -1, -1), x), dim=1)
        x = self.positional_embedding(x)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        x = x[:, 0] @ self.projection
        return x / torch.norm(x, dim=-1, keepdim=True)


class TextEncoder(nn.Module):
    """Text encoder using Transformer architecture."""
    
    def __init__(self, vocab_size: int, width: int, max_seq_length: int,
                 n_heads: int, n_layers: int, emb_dim: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = PositionalEmbedding(width, max_seq_length)
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(width, n_heads) for _ in range(n_layers)
        ])
        self.projection = nn.Parameter(torch.randn(width, emb_dim))

    def forward(self, text: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.token_embedding(text)
        x = self.positional_embedding(x)
        
        for layer in self.transformer_layers:
            x = layer(x, mask=mask)
        
        x = x[:, 0] @ self.projection
        return x / torch.norm(x, dim=-1, keepdim=True)


class CLIP(nn.Module):
    """CLIP model implementation combining image and text encoders."""
    
    def __init__(self, config: dict):
        super().__init__()
        self.image_encoder = ImageEncoder(
            config['width'],
            config['img_size'],
            config['patch_size'],
            config['n_channels'],
            config['vit_layers'],
            config['vit_heads'],
            config['emb_dim']
        )
        self.text_encoder = TextEncoder(
            config['vocab_size'],
            config['text_width'],
            config['max_seq_length'],
            config['text_heads'],
            config['text_layers'],
            config['emb_dim']
        )
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, image: torch.Tensor, text: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            mask = mask.to(self.device)
            
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text, mask=mask)
        
        logits = (image_features @ text_features.transpose(-2, -1)) * torch.exp(self.temperature)
        
        labels = torch.arange(logits.shape[0], device=self.device)
        loss_i = nn.functional.cross_entropy(logits.transpose(-2, -1), labels)
        loss_t = nn.functional.cross_entropy(logits, labels)
        
        return (loss_i + loss_t) / 2


class FlickrDataset(Dataset):
    """Dataset class for Flickr8k dataset."""
    
    def __init__(self, df: pd.DataFrame, image_path: str, transform=None):
        self.df = df
        self.image_path = image_path
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.image_groups = df.groupby('image')
        self.unique_images = list(self.image_groups.groups.keys())

    def __len__(self) -> int:
        return len(self.unique_images)

    def __getitem__(self, idx: int) -> dict:
        image_name = self.unique_images[idx]
        captions = self.image_groups.get_group(image_name)['caption'].tolist()
        
        caption = np.random.choice(captions)
        
        image = Image.open(os.path.join(self.image_path, image_name)).convert('RGB')
        image = self.transform(image)
        
        tokens, mask = tokenizer(caption)
        
        return {
            'image': image,
            'caption': tokens,
            'mask': mask,
            'raw_caption': caption
        }


def train_clip(model: nn.Module, train_loader: DataLoader,
               optimizer: torch.optim.Optimizer, device: torch.device,
               epochs: int = 30) -> None:
    """Train the CLIP model."""
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch in progress_bar:
            images = batch['image'].to(device)
            captions = batch['caption'].to(device)
            masks = batch['mask'].to(device)
            
            loss = model(images, captions, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                model.state_dict(),
                'clip.pt'
            )
            print("Model saved.")


def main():
    """Main training function."""
    # Model configuration
    config = {
        'emb_dim': 256,
        'width': 768,
        'img_size': (224, 224),
        'patch_size': (16, 16),
        'n_channels': 3,
        'vit_layers': 6,
        'vit_heads': 6,
        'vocab_size': 25000,
        'text_width': 512,
        'max_seq_length': 77,
        'text_heads': 4,
        'text_layers': 6,
        'batch_size': 64,
        'learning_rate': 1e-4,
        'epochs': 10
    }
    
    # Load and prepare data
    df = pd.read_csv('captions.csv')
    dataset = FlickrDataset(
        df=df,
        image_path="Images"
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    # Initialize model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIP(config).to(device)
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    
    # Train model
    train_clip(model, train_loader, optimizer, device, epochs=config['epochs'])


if __name__ == "__main__":
    main()
