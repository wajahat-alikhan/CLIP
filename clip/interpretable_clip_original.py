"""
TCLIP: Extends original CLIP to enable fine-grained token-patch similarity analysis.

This module adds interpretability methods to CLIP models while maintaining perfect compatibility
with the original implementation. It extracts ALL patch and token embeddings (instead of just 
CLS/EOS tokens) and applies the same projection heads and normalization for cosine similarity 
computation between text tokens and image patches.

Core Technical Approach:
- Preserve original CLIP weights and processing pipeline
- Extract all intermediate embeddings from encoder layers  
- Apply same projection heads to all tokens/patches
- Enable fine-grained similarity matrix computation

Author: Based on OpenAI's CLIP with our interpretability method
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List

from .clip import load, tokenize


def load_interpretable_clip(name: str = "ViT-B/32", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load an interpretable version of CLIP that can compute token-patch similarities.
    
    This extends the original CLIP model by adding methods to extract ALL patch embeddings
    and ALL token embeddings (instead of just CLS/EOS tokens), while using the exact same
    weights, projection heads, and processing pipeline for perfect compatibility.
    
    Technical Details:
    - Uses original CLIP weights without modification
    - Applies same projection matrices to all embeddings
    - Uses same L2 normalization for cosine similarity
    - Maintains identical global embeddings as original CLIP
    - Enables [num_tokens x num_patches] fine-grained similarity matrix
    
    Args:
        name: CLIP model name (e.g., "ViT-B/32", "ViT-L/14", "ViT-B/16")
        device: device to load the model on
        
    Returns:
        CLIP model with added interpretability methods:
        - encode_text_with_tokens(): extracts all token embeddings
        - encode_image_with_patches(): extracts all patch embeddings  
        - get_token_patch_similarity(): computes fine-grained similarity matrix
        - preprocess: original CLIP preprocessing function
    """
    # Load standard CLIP model with original weights
    model, preprocess = load(name, device=device)
    
    def encode_text_with_tokens(self, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text and return both pooled embedding and ALL token embeddings.
        Uses identical processing as original CLIP but extracts all intermediate tokens.
        
        Args:
            text: [batch_size, sequence_length] tokenized text
            
        Returns:
            pooled_embedding: [batch_size, embed_dim] - standard CLIP text embedding (EOS token)
            token_embeddings: [batch_size, sequence_length, embed_dim] - ALL token embeddings before projection
        """
        text = text.to(next(self.parameters()).device)
        
        # Identical processing to original CLIP text encoder
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # ALL token embeddings (before pooling)
        token_embeddings = x
        
        # Standard CLIP pooled embedding (EOS token with projection)
        pooled_embedding = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        
        return pooled_embedding, token_embeddings
    
    def encode_image_with_patches(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image and return both pooled embedding and ALL patch embeddings.
        Uses identical processing as original CLIP but extracts all intermediate patches.
        
        Args:
            image: [batch_size, 3, height, width] preprocessed image
            
        Returns:
            pooled_embedding: [batch_size, embed_dim] - standard CLIP image embedding (CLS token)
            patch_embeddings: [batch_size, num_patches, embed_dim] - ALL patch embeddings with projection
        """
        # Ensure input is on correct device
        image = image.to(next(self.parameters()).device)
        
        # Identical processing to original CLIP visual encoder
        x = self.visual.conv1(image.type(self.dtype))  # [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)      # [*, width, grid²]
        x = x.permute(0, 2, 1)                         # [*, grid², width]
        
        # Add CLS token (same as original CLIP)
        cls_token = self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        
        # Transformer blocks (identical to original)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # Apply same post-processing to both CLS and patch tokens
        # CLS token (index 0) - standard CLIP image embedding
        cls_embedding = self.visual.ln_post(x[:, 0, :])
        if self.visual.proj is not None:
            cls_embedding = cls_embedding @ self.visual.proj
            
        # Patch tokens (index 1:) - individual patch embeddings with same processing
        patch_embeddings = self.visual.ln_post(x[:, 1:, :])
        if self.visual.proj is not None:
            patch_embeddings = patch_embeddings @ self.visual.proj
            
        return cls_embedding, patch_embeddings
    
    def get_token_patch_similarity(self, image: torch.Tensor, text: torch.Tensor, debug: bool = False) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute comprehensive cosine similarities between text and image embeddings in CLIP's latent space.
        
        This function computes:
        1. Fine-grained token-patch similarities (existing functionality)
        2. EOS token vs all image patches (global text vs local image)
        3. CLS token vs all text tokens (global image vs local text)
        
        Technical Process:
        1. Extract all patch embeddings and CLS embedding from image encoder
        2. Extract all token embeddings and EOS embedding from text encoder  
        3. Apply same projection heads as original CLIP
        4. L2 normalize all embeddings
        5. Compute three similarity matrices via dot product
        6. Filter out special tokens for meaningful analysis
        
        Args:
            image: [1, 3, height, width] preprocessed image (batch_size=1 only)
            text: [1, sequence_length] tokenized text (batch_size=1 only)
            debug: whether to print debug information
            
        Returns:
            tokens: List of meaningful token strings (excluding special tokens)
            token_patch_similarity: [num_tokens, num_patches] - individual tokens vs patches
            eos_patch_similarity: [1, num_patches] - EOS token vs all patches
            cls_token_similarity: [1, num_tokens] - CLS token vs all tokens
        """
        if debug:
            print(f"Input shapes - Image: {image.shape}, Text: {text.shape}")
        
        # Currently supports batch_size=1 only (can be extended for batches)
        assert image.shape[0] == 1 and text.shape[0] == 1, "Currently only supports batch_size=1"
        
        # Ensure inputs are on correct device
        image = image.to(next(self.parameters()).device)
        text = text.to(next(self.parameters()).device)
        
        # Extract all embeddings using same processing as original CLIP
        cls_embedding, patch_embeddings = self.encode_image_with_patches(image)  # [1, embed_dim], [1, num_patches, embed_dim]
        eos_embedding, token_embeddings = self.encode_text_with_tokens(text)     # [1, embed_dim], [1, seq_len, embed_dim]
        
        if debug:
            print(f"Extracted embeddings - Patches: {patch_embeddings.shape}, Tokens: {token_embeddings.shape}")
            print(f"Global embeddings - CLS: {cls_embedding.shape}, EOS: {eos_embedding.shape}")
        
        # Apply text projection to put tokens in same space as patches (same as original CLIP)
        token_embeddings_proj = torch.matmul(token_embeddings, self.text_projection)
        
        # L2 normalize all embeddings for cosine similarity (identical to original CLIP)
        patch_embeddings_norm = F.normalize(patch_embeddings, dim=-1)
        token_embeddings_norm = F.normalize(token_embeddings_proj, dim=-1)
        cls_embedding_norm = F.normalize(cls_embedding, dim=-1)
        eos_embedding_norm = F.normalize(eos_embedding, dim=-1)
        
        # Decode token strings for interpretability
        from .simple_tokenizer import SimpleTokenizer
        tokenizer = SimpleTokenizer()
        all_tokens = [tokenizer.decode([token_id.cpu().item()]) for token_id in text[0]]
        
        # Filter out special tokens and padding
        sot_token_id, eot_token_id = 49406, 49407  # <|startoftext|>, <|endoftext|>
        
        real_token_indices = []
        tokens_clean = []
        
        for i, (token_id, token_str) in enumerate(zip(text[0], all_tokens)):
            token_id_val = token_id.cpu().item()
            token_str_clean = token_str.strip().replace('</w>', '')
            
            # Keep only meaningful content tokens
            if (token_id_val not in [sot_token_id, eot_token_id, 0] and 
                token_str_clean and 
                token_str_clean not in ['<|startoftext|>', '<|endoftext|>', '!', '.']):
                real_token_indices.append(i)
                tokens_clean.append(token_str_clean)
        
        if len(real_token_indices) == 0:
            raise ValueError("No meaningful tokens found after filtering!")
        
        # Extract meaningful token embeddings
        token_embeddings_real = token_embeddings_norm[0, real_token_indices, :]
        
        # 1. Compute token-patch similarities: [num_real_tokens, num_patches]
        token_patch_similarity = torch.matmul(token_embeddings_real, patch_embeddings_norm[0].transpose(0, 1))
        
        # 2. Compute EOS vs all patches: [1, num_patches] 
        # This shows how the global text meaning aligns with each image region
        eos_patch_similarity = torch.matmul(eos_embedding_norm, patch_embeddings_norm[0].transpose(0, 1))
        
        # 3. Compute CLS vs all meaningful tokens: [1, num_real_tokens]
        # This shows how the global image meaning aligns with each text token
        cls_token_similarity = torch.matmul(cls_embedding_norm, token_embeddings_real.transpose(0, 1))
        
        if debug:
            print(f"Token-Patch similarity: {token_patch_similarity.shape}")
            print(f"EOS-Patch similarity: {eos_patch_similarity.shape}")
            print(f"CLS-Token similarity: {cls_token_similarity.shape}")
            print(f"Token-Patch range: [{token_patch_similarity.min().item():.4f}, {token_patch_similarity.max().item():.4f}]")
            print(f"EOS-Patch range: [{eos_patch_similarity.min().item():.4f}, {eos_patch_similarity.max().item():.4f}]")
            print(f"CLS-Token range: [{cls_token_similarity.min().item():.4f}, {cls_token_similarity.max().item():.4f}]")
        
        return tokens_clean, token_patch_similarity, eos_patch_similarity, cls_token_similarity

    # Add interpretability methods to the existing CLIP model
    import types
    model.encode_text_with_tokens = types.MethodType(encode_text_with_tokens, model)
    model.encode_image_with_patches = types.MethodType(encode_image_with_patches, model)
    model.get_token_patch_similarity = types.MethodType(get_token_patch_similarity, model)
    
    # Attach preprocessing function for convenience
    model.preprocess = preprocess
    
    # Print model info
    grid_size = model.visual.input_resolution // model.visual.conv1.kernel_size[0]
    print(f"Interpretable CLIP ({name}) loaded successfully!")
    print(f"Grid size: {grid_size}×{grid_size} = {grid_size**2} patches")
    print(f"Enables fine-grained analysis of {grid_size**2} image regions × text tokens")
    
    return model


# Utility function for easy text tokenization
def tokenize_text(text: str):
    """Convenience function for tokenizing text for interpretable CLIP."""
    return tokenize([text])


# Example usage:
"""
# Load interpretable CLIP
model = load_interpretable_clip("ViT-B/32")

# Prepare inputs  
image = model.preprocess(PIL_image).unsqueeze(0)
text = tokenize_text("a photo of a cat")

# Get comprehensive similarities
tokens, token_patch_sim, eos_patch_sim, cls_token_sim = model.get_token_patch_similarity(image, text)

# Results:
# tokens: ['a', 'photo', 'of', 'a', 'cat'] 
# token_patch_sim: [5, 49] - each word vs each image patch
# eos_patch_sim: [1, 49] - global text meaning vs each image patch  
# cls_token_sim: [1, 5] - global image meaning vs each word
""" 