import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import cv2

from .model import CLIP, VisionTransformer
from .clip import load, tokenize

class InterpretableVisionTransformer(VisionTransformer):
    """Modified VisionTransformer that returns patch embeddings."""
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # Get patch embeddings (excluding CLS token)
        patch_embeddings = x[:, 1:, :]
        
        # Get CLS token embedding
        cls_embedding = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            cls_embedding = cls_embedding @ self.proj
            
        return cls_embedding, patch_embeddings

class InterpretableCLIP(CLIP):
    """Modified CLIP that enables token-patch similarity computation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the visual encoder with our interpretable version
        if isinstance(self.visual, VisionTransformer):
            self.visual = InterpretableVisionTransformer(
                input_resolution=self.visual.input_resolution,
                patch_size=self.visual.conv1.kernel_size[0],
                width=self.visual.conv1.out_channels,
                layers=len(self.visual.transformer.resblocks),
                heads=self.visual.transformer.resblocks[0].attn.num_heads,
                output_dim=self.visual.proj.shape[1]
            )
    
    def encode_text_with_tokens(self, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text and return both pooled embedding and token embeddings."""
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # Get token embeddings before pooling
        token_embeddings = x
        
        # Get pooled embedding
        pooled_embedding = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        
        return pooled_embedding, token_embeddings
    
    def encode_image_with_patches(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image and return both pooled embedding and patch embeddings."""
        return self.visual(image.type(self.dtype))
    
    def get_token_patch_similarity(self, image: torch.Tensor, text: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
        """Compute similarity between each real text token and each real image patch (exclude special tokens and CLS)."""
        print("[DEBUG] Input image shape:", image.shape)
        print("[DEBUG] Input text shape:", text.shape)
        # Get patch embeddings (exclude CLS)
        _, patch_embeddings = self.encode_image_with_patches(image)  # [batch, num_patches, dim]
        # Get token embeddings (all tokens)
        _, token_embeddings = self.encode_text_with_tokens(text)  # [batch, num_tokens, dim]
        print("[DEBUG] Patch embeddings shape (before projection):", patch_embeddings.shape)
        print("[DEBUG] Token embeddings shape (before projection):", token_embeddings.shape)
        # Project embeddings to same dimension
        if hasattr(self.visual, 'proj') and self.visual.proj is not None:
            patch_embeddings_proj = torch.matmul(patch_embeddings, self.visual.proj)
        else:
            patch_embeddings_proj = patch_embeddings
        if hasattr(self, 'text_projection') and self.text_projection is not None:
            token_embeddings_proj = torch.matmul(token_embeddings, self.text_projection)
        else:
            token_embeddings_proj = token_embeddings
        print("[DEBUG] Patch embeddings shape (after projection):", patch_embeddings_proj.shape)
        print("[DEBUG] Token embeddings shape (after projection):", token_embeddings_proj.shape)
        # Normalize embeddings
        patch_embeddings_proj = F.normalize(patch_embeddings_proj, dim=-1)
        token_embeddings_proj = F.normalize(token_embeddings_proj, dim=-1)
        print("[DEBUG] Patch embeddings shape (after normalization):", patch_embeddings_proj.shape)
        print("[DEBUG] Token embeddings shape (after normalization):", token_embeddings_proj.shape)
        # Get token strings using convert_ids_to_tokens if available
        if hasattr(self.tokenizer, 'convert_ids_to_tokens'):
            all_tokens = self.tokenizer.convert_ids_to_tokens([t.cpu().item() for t in text[0]])
        else:
            all_tokens = [self.tokenizer.decode([t.cpu().item()]) for t in text[0]]
        # Remove special tokens (BOS, EOS, padding) by both ID and string
        special_token_strings = {"<|startoftext|>", "<|endoftext|>"}
        special_ids = set()
        if hasattr(self.tokenizer, 'all_special_ids'):
            special_ids = set(self.tokenizer.all_special_ids)
        real_token_indices = [
            i for i, (tok_id, tok_str) in enumerate(zip(text[0], all_tokens))
            if (tok_id.cpu().item() not in special_ids and tok_str.strip() != '!' and tok_str not in special_token_strings)
        ]
        tokens = [all_tokens[i] for i in real_token_indices]
        print("[DEBUG] Real tokens:", tokens)
        # Select only real token embeddings
        token_embeddings_proj = token_embeddings_proj[0, real_token_indices, :]
        # Compute similarity matrix (real tokens x all patches)
        similarity = torch.matmul(token_embeddings_proj, patch_embeddings_proj[0].transpose(0, 1))  # [num_real_tokens, num_patches]
        print("[DEBUG] Similarity matrix shape:", similarity.shape)
        # Patch grid shape assertion
        grid_size = int(np.sqrt(similarity.shape[-1]))
        assert grid_size * grid_size == similarity.shape[-1], f"Expected square patch grid, got {similarity.shape[-1]}"
        return tokens, similarity  # Only real tokens, all real patches
    
    @staticmethod
    def find_token_indices(tokens, query):
        """Return a list of indices where the token contains the query substring (case-insensitive)."""
        return [i for i, t in enumerate(tokens) if query.lower() in t.lower()]

    def visualize_token_patch_similarity(self, image: Image.Image, text: str, token_idx: Optional[int] = None):
        """Visualize token-patch similarities for all tokens or a specific token index."""
        with torch.no_grad():
            image_input = self.preprocess(image).unsqueeze(0)
            text_input = tokenize([text])
            tokens, similarity = self.get_token_patch_similarity(image_input, text_input)
            similarity = similarity.detach().cpu().numpy()
            patch_size = self.visual.conv1.kernel_size[0]
            grid_size = int(np.sqrt(similarity.shape[1]))

            if token_idx is not None:
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(image)
                plt.axis('off')
                plt.title('Input Image')
                plt.subplot(1, 2, 2)
                heatmap = similarity[token_idx].reshape(grid_size, grid_size)
                plt.imshow(heatmap, cmap='viridis')
                plt.colorbar()
                plt.title(f'Similarity for token: {tokens[token_idx]} (idx={token_idx})')
                plt.axis('off')
            else:
                n_tokens = len(tokens)
                n_cols = min(4, n_tokens+1)
                n_rows = ((n_tokens + 1) + n_cols - 1) // n_cols  # +1 for the image
                plt.figure(figsize=(4*n_cols, 4*n_rows))
                plt.subplot(n_rows, n_cols, 1)
                plt.imshow(image)
                plt.axis('off')
                plt.title('Input Image')
                for i, token in enumerate(tokens):
                    plt.subplot(n_rows, n_cols, i + 2)
                    heatmap = similarity[i].reshape(grid_size, grid_size)
                    plt.imshow(heatmap, cmap='viridis')
                    plt.colorbar()
                    plt.title(f'Token: {token}\n(idx={i})')
                    plt.axis('off')
            plt.tight_layout()
            plt.show()

    def visualize_token_patch_overlay(self, image: Image.Image, text: str, token: str = None, token_idx: int = None, alpha: float = 0.5):
        """Overlay the patch similarity heatmap for a selected token on the image."""
        with torch.no_grad():
            image_input = self.preprocess(image).unsqueeze(0)
            text_input = tokenize([text])
            tokens, similarity = self.get_token_patch_similarity(image_input, text_input)
            similarity = similarity.detach().cpu().numpy()
            patch_size = self.visual.conv1.kernel_size[0]
            grid_size = int(np.sqrt(similarity.shape[1]))
            if token_idx is None and token is not None:
                matches = self.find_token_indices(tokens, token)
                if not matches:
                    raise ValueError(f"No token containing '{token}' found. Available tokens: {tokens}")
                token_idx = matches[0]
            if token_idx is None:
                raise ValueError("Token or token_idx must be specified and found in tokens.")
            heatmap = similarity[token_idx].reshape(grid_size, grid_size)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            img_w, img_h = image.size
            heatmap_resized = cv2.resize(heatmap, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
            plt.figure(figsize=(8, 8))
            plt.imshow(image)
            plt.imshow(heatmap_resized, cmap='jet', alpha=alpha)
            plt.axis('off')
            plt.title(f"Token: {tokens[token_idx]} (idx={token_idx})")
            plt.colorbar(label='Similarity')
            plt.show()

    def plot_token_importance(self, image: Image.Image, text: str):
        """Plot a bar chart of average similarity for each token."""
        with torch.no_grad():
            image_input = self.preprocess(image).unsqueeze(0)
            text_input = tokenize([text])
            tokens, similarity = self.get_token_patch_similarity(image_input, text_input)
            similarity = similarity.detach().cpu().numpy()
            avg_sim = similarity[:len(tokens)].mean(axis=1)
            plt.figure(figsize=(max(10, len(tokens)), 4))
            plt.bar(tokens, avg_sim)
            plt.ylabel('Average Patch Similarity')
            plt.xlabel('Token')
            plt.title('Token Importance (Average Similarity)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    def visualize_all_token_patch_overlays(self, image: Image.Image, text: str, alpha: float = 0.5, max_cols: int = 4):
        """Overlay patch similarity heatmaps for all real tokens in a single figure for easy comparison."""
        print("[DEBUG] Visualizing all token patch overlays...")
        with torch.no_grad():
            image_input = self.preprocess(image).unsqueeze(0)
            text_input = tokenize([text])
            tokens, similarity = self.get_token_patch_similarity(image_input, text_input)
            similarity = similarity.detach().cpu().numpy()
            patch_size = self.visual.conv1.kernel_size[0]
            grid_size = int(np.sqrt(similarity.shape[1]))
            img_w, img_h = image.size
            n_tokens = len(tokens)
            n_cols = min(max_cols, n_tokens)
            n_rows = (n_tokens + n_cols - 1) // n_cols
            plt.figure(figsize=(4*n_cols, 4*n_rows))
            for i, token in enumerate(tokens):
                heatmap = similarity[i].reshape(grid_size, grid_size)
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
                heatmap_resized = cv2.resize(heatmap, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
                plt.subplot(n_rows, n_cols, i + 1)
                plt.imshow(image)
                plt.imshow(heatmap_resized, cmap='jet', alpha=alpha)
                plt.axis('off')
                plt.title(repr(token), fontsize=10, y=1.02)
            plt.tight_layout()
            plt.show()
            print("[DEBUG] All token overlays visualized.")

    def plot_token_patch_matrix(self, tokens, similarity):
        # Ensure similarity is a NumPy array
        if hasattr(similarity, 'detach'):
            similarity = similarity.detach().cpu().numpy()
        plt.figure(figsize=(10, max(4, len(tokens) * 0.5)))
        plt.imshow(similarity, aspect='auto', cmap='viridis')
        plt.colorbar(label='Cosine Similarity')
        plt.yticks(np.arange(len(tokens)), [str(t) for t in tokens])
        plt.xlabel('Patch Index')
        plt.ylabel('Token')
        plt.title('Token-Patch Similarity Matrix')
        plt.tight_layout()
        plt.show()

def load_interpretable_clip(name: str = "ViT-B/32", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Load an interpretable version of CLIP."""
    model, preprocess = load(name, device=device)
    
    # Get model parameters from state dict
    state_dict = model.state_dict()
    
    # Get dimensions from state dict
    embed_dim = state_dict["text_projection"].shape[1]
    image_resolution = model.visual.input_resolution
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
    
    interpretable_model = InterpretableCLIP(
        embed_dim=embed_dim,
        image_resolution=image_resolution,
        vision_layers=len(model.visual.transformer.resblocks),
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers
    )
    
    # Copy weights
    interpretable_model.load_state_dict(model.state_dict())
    interpretable_model.preprocess = preprocess
    
    # Get tokenizer from CLIP module
    from .simple_tokenizer import SimpleTokenizer
    interpretable_model.tokenizer = SimpleTokenizer()
    
    return interpretable_model 