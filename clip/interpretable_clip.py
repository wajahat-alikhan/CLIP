import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from .model import CLIP, VisionTransformer
from .clip import load, tokenize

class InterpretableVisionTransformer(VisionTransformer):
    """Modified VisionTransformer that returns patch embeddings."""
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns both CLS token embedding and patch embeddings.
        
        Returns:
            cls_embedding: [batch_size, embed_dim] - CLS token embedding (for image-level tasks)
            patch_embeddings: [batch_size, num_patches, embed_dim] - Individual patch embeddings
        """
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # Apply layer norm and projection to both CLS and patch tokens
        # This ensures they're in the same semantic space as CLIP's contrastive learning
        
        # CLS token (index 0) - standard CLIP image embedding
        cls_embedding = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            cls_embedding = cls_embedding @ self.proj
            
        # Patch tokens (index 1:) - individual patch embeddings with same processing
        patch_embeddings = self.ln_post(x[:, 1:, :])  
        if self.proj is not None:
            patch_embeddings = patch_embeddings @ self.proj
            
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
        """
        Encode text and return both pooled embedding and individual token embeddings.
        
        Args:
            text: [batch_size, sequence_length] tokenized text
            
        Returns:
            pooled_embedding: [batch_size, embed_dim] - standard CLIP text embedding
            token_embeddings: [batch_size, sequence_length, embed_dim] - individual token embeddings
        """
        text = text.to(next(self.parameters()).device)
        
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # Individual token embeddings (before pooling)
        token_embeddings = x
        
        # Standard CLIP pooled embedding (take features from the eot embedding)
        pooled_embedding = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        
        return pooled_embedding, token_embeddings
    
    def encode_image_with_patches(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image and return both pooled embedding and patch embeddings.
        
        Args:
            image: [batch_size, 3, height, width] preprocessed image
            
        Returns:
            pooled_embedding: [batch_size, embed_dim] - standard CLIP image embedding
            patch_embeddings: [batch_size, num_patches, embed_dim] - individual patch embeddings
        """
        image = image.to(next(self.visual.parameters()).device)
        return self.visual(image.type(self.dtype))
    
    def get_token_patch_similarity(self, image: torch.Tensor, text: torch.Tensor, debug: bool = False) -> Tuple[List[str], torch.Tensor]:
        """
        Compute pure cosine similarity between text tokens and image patches in CLIP's latent space.
        
        Args:
            image: [batch_size, 3, height, width] preprocessed image
            text: [batch_size, sequence_length] tokenized text
            debug: whether to print debug information
            
        Returns:
            tokens: List of meaningful token strings (excluding special tokens)
            similarity: [num_tokens, num_patches] cosine similarity matrix
        """
        if debug:
            print(f"Input image shape: {image.shape}")
            print(f"Input text shape: {text.shape}")
        
        # Get embeddings in CLIP's projected space
        _, patch_embeddings = self.encode_image_with_patches(image)  # [batch, num_patches, embed_dim]
        _, token_embeddings = self.encode_text_with_tokens(text)     # [batch, seq_len, embed_dim]
        
        if debug:
            print(f"Patch embeddings shape: {patch_embeddings.shape}")
            print(f"Token embeddings shape: {token_embeddings.shape}")
        
        # Project token embeddings to the same space as patch embeddings
        token_embeddings_proj = torch.matmul(token_embeddings, self.text_projection)
        
        if debug:
            print(f"Token embeddings projected shape: {token_embeddings_proj.shape}")
        
        # L2 normalize both embeddings for cosine similarity
        patch_embeddings_norm = F.normalize(patch_embeddings, dim=-1)
        token_embeddings_norm = F.normalize(token_embeddings_proj, dim=-1)
        
        # Get token strings and filter out special tokens
        if hasattr(self.tokenizer, 'convert_ids_to_tokens'):
            all_tokens = self.tokenizer.convert_ids_to_tokens([t.cpu().item() for t in text[0]])
        else:
            all_tokens = [self.tokenizer.decode([t.cpu().item()]) for t in text[0]]
        
        # Filter out special tokens and padding
        special_token_strings = {"<|startoftext|>", "<|endoftext|>", "!"}
        special_ids = set()
        if hasattr(self.tokenizer, 'all_special_ids'):
            special_ids = set(self.tokenizer.all_special_ids)
        
        real_token_indices = []
        for i, (tok_id, tok_str) in enumerate(zip(text[0], all_tokens)):
            tok_str_stripped = tok_str.strip()
            if (tok_id.cpu().item() not in special_ids and 
                tok_str_stripped and 
                tok_str_stripped != '!' and 
                tok_str_stripped not in special_token_strings):
                real_token_indices.append(i)
        
        # Extract meaningful tokens
        tokens = [all_tokens[i].strip() for i in real_token_indices]
        token_embeddings_real = token_embeddings_norm[0, real_token_indices, :]
        
        if debug:
            print(f"Real tokens: {tokens}")
            print(f"Real token embeddings shape: {token_embeddings_real.shape}")
        
        # Compute cosine similarity matrix: [num_real_tokens, num_patches]
        similarity = torch.matmul(token_embeddings_real, patch_embeddings_norm[0].transpose(0, 1))
        
        if debug:
            print(f"Similarity matrix shape: {similarity.shape}")
            print(f"Similarity range: [{similarity.min().item():.4f}, {similarity.max().item():.4f}]")
        
        return tokens, similarity

    def plot_token_patch_matrix(self, tokens, similarity):
        """
        Plots the token-patch similarity matrix as a heatmap (confusion matrix style).
        This shows the cosine similarity between every image patch and every text token.
        """
        if hasattr(similarity, 'detach'):
            similarity = similarity.detach().cpu().numpy()

        plt.figure(figsize=(12, max(4, len(tokens) * 0.4)))
        
        # Find the maximum absolute value for symmetric color scaling
        vmax = np.max(np.abs(similarity))
        
        plt.imshow(similarity, aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
        
        plt.yticks(np.arange(len(tokens)), [str(t) for t in tokens])
        plt.xlabel("Image Patch Index")
        plt.ylabel("Text Token")
        plt.title("Token-Patch Cosine Similarity Matrix")
        
        cbar = plt.colorbar()
        cbar.set_label("Cosine Similarity")
        
        plt.tight_layout()
        plt.show()

    def visualize_text_impact_on_image(self, image: Image.Image, similarity: torch.Tensor, 
                                       alpha: float = 0.6, gaussian_sigma: float = 3.0, 
                                       colormap: str = 'viridis', use_positive_only: bool = True):
        """
        Creates a smooth, gradient-style saliency heatmap overlaid on the image.
        This produces the type of visualization commonly seen in interpretability research
        with smooth color transitions and natural blending.

        Args:
            image (Image.Image): The original PIL image for visualization.
            similarity (torch.Tensor): The [num_tokens, num_patches] similarity matrix.
            alpha (float): The transparency of the heatmap overlay (0.0 = transparent, 1.0 = opaque).
            gaussian_sigma (float): Standard deviation for Gaussian smoothing (higher = smoother).
            colormap (str): Matplotlib colormap name ('viridis', 'plasma', 'hot', 'jet', 'coolwarm', etc.).
            use_positive_only (bool): If True, only shows positive similarities for cleaner visualization.
        """
        if hasattr(similarity, 'detach'):
            similarity = similarity.detach().cpu().numpy()

        # Average the similarity scores across all tokens for each patch
        patch_impact_scores = np.mean(similarity, axis=0)
        
        # If using positive only, clip negative values to zero
        if use_positive_only:
            patch_impact_scores = np.maximum(patch_impact_scores, 0)
        
        grid_size = int(np.sqrt(patch_impact_scores.shape[0]))
        heatmap = patch_impact_scores.reshape(grid_size, grid_size)

        # Resize the small heatmap to the full image size using cubic interpolation for smoother results
        img_w, img_h = image.size
        heatmap_resized = cv2.resize(heatmap, (img_w, img_h), interpolation=cv2.INTER_CUBIC)

        # Apply Gaussian smoothing for that gradient-like effect
        # Convert sigma to kernel size (rule of thumb: kernel_size = 6*sigma + 1)
        kernel_size = int(6 * gaussian_sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        
        heatmap_smooth = cv2.GaussianBlur(heatmap_resized, (kernel_size, kernel_size), gaussian_sigma)

        # Normalize the heatmap to [0, 1] for better color mapping
        if use_positive_only:
            heatmap_norm = (heatmap_smooth - heatmap_smooth.min()) / (heatmap_smooth.max() - heatmap_smooth.min() + 1e-8)
        else:
            # For symmetric normalization when including negative values
            vmax = np.max(np.abs(heatmap_smooth))
            heatmap_norm = (heatmap_smooth + vmax) / (2 * vmax + 1e-8)

        # Create the visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Display the original image
        ax.imshow(image)
        
        # Create and overlay the smooth heatmap
        if use_positive_only:
            im = ax.imshow(heatmap_norm, cmap=colormap, alpha=alpha, vmin=0, vmax=1)
        else:
            vmax_display = np.max(np.abs(heatmap_smooth))
            im = ax.imshow(heatmap_smooth, cmap=colormap, alpha=alpha, vmin=-vmax_display, vmax=vmax_display)

        # Clean styling for professional appearance
        ax.axis('off')
        
        # Add colorbar with proper labeling
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if use_positive_only:
            cbar.set_label("Text-Image Similarity", fontsize=12)
        else:
            cbar.set_label("Text-Image Cosine Similarity", fontsize=12)
        
        plt.tight_layout()
        plt.show()

    def visualize_gradient_heatmap(self, image: Image.Image, similarity: torch.Tensor, style: str = 'research'):
        """
        Alternative visualization function that creates different styles of gradient heatmaps.
        
        Args:
            image: PIL Image
            similarity: token-patch similarity matrix
            style: 'research' (viridis), 'thermal' (hot), 'attention' (plasma), or 'classic' (jet)
        """
        style_configs = {
            'research': {'colormap': 'viridis', 'alpha': 0.6, 'sigma': 4.0},
            'thermal': {'colormap': 'hot', 'alpha': 0.7, 'sigma': 3.5},
            'attention': {'colormap': 'plasma', 'alpha': 0.6, 'sigma': 4.5},
            'classic': {'colormap': 'jet', 'alpha': 0.5, 'sigma': 3.0}
        }
        
        config = style_configs.get(style, style_configs['research'])
        
        print(f"Creating {style}-style gradient heatmap...")
        self.visualize_text_impact_on_image(
            image, similarity, 
            alpha=config['alpha'], 
            gaussian_sigma=config['sigma'],
            colormap=config['colormap'],
            use_positive_only=True
        )

    def visualize_individual_token_impacts(self, image: Image.Image, tokens: List[str], similarity: torch.Tensor,
                                          alpha: float = 0.6, gaussian_sigma: float = 3.0, 
                                          colormap: str = 'plasma', use_positive_only: bool = True,
                                          max_cols: int = 3):
        """
        Creates separate gradient heatmaps for each individual token's impact on the image.
        This shows how each word in the text prompt affects different regions of the image.
        
        Args:
            image (Image.Image): The original PIL image for visualization.
            tokens (List[str]): List of meaningful tokens from the text.
            similarity (torch.Tensor): The [num_tokens, num_patches] similarity matrix.
            alpha (float): The transparency of the heatmap overlay.
            gaussian_sigma (float): Standard deviation for Gaussian smoothing.
            colormap (str): Matplotlib colormap name.
            use_positive_only (bool): If True, only shows positive similarities.
            max_cols (int): Maximum number of columns in the subplot grid.
        """
        if hasattr(similarity, 'detach'):
            similarity = similarity.detach().cpu().numpy()
        
        num_tokens = len(tokens)
        if num_tokens == 0:
            print("No meaningful tokens to visualize.")
            return
        
        # Calculate grid dimensions
        cols = min(max_cols, num_tokens)
        rows = (num_tokens + cols - 1) // cols  # Ceiling division
        
        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Get image dimensions
        img_w, img_h = image.size
        grid_size = int(np.sqrt(similarity.shape[1]))
        
        # Process each token
        for i, (token, ax) in enumerate(zip(tokens, axes)):
            # Get similarity scores for this specific token
            token_similarities = similarity[i, :]
            
            # Apply positive-only filtering if requested
            if use_positive_only:
                token_similarities = np.maximum(token_similarities, 0)
            
            # Reshape to spatial grid
            heatmap = token_similarities.reshape(grid_size, grid_size)
            
            # Resize and smooth
            heatmap_resized = cv2.resize(heatmap, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
            
            # Apply Gaussian smoothing
            kernel_size = int(6 * gaussian_sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            heatmap_smooth = cv2.GaussianBlur(heatmap_resized, (kernel_size, kernel_size), gaussian_sigma)
            
            # Normalize the heatmap
            if use_positive_only:
                if heatmap_smooth.max() > heatmap_smooth.min():
                    heatmap_norm = (heatmap_smooth - heatmap_smooth.min()) / (heatmap_smooth.max() - heatmap_smooth.min())
                else:
                    heatmap_norm = heatmap_smooth
            else:
                vmax = np.max(np.abs(heatmap_smooth))
                if vmax > 0:
                    heatmap_norm = (heatmap_smooth + vmax) / (2 * vmax)
                else:
                    heatmap_norm = heatmap_smooth
            
            # Display the original image
            ax.imshow(image)
            
            # Overlay the heatmap
            if use_positive_only:
                im = ax.imshow(heatmap_norm, cmap=colormap, alpha=alpha, vmin=0, vmax=1)
            else:
                vmax_display = np.max(np.abs(heatmap_smooth))
                if vmax_display > 0:
                    im = ax.imshow(heatmap_smooth, cmap=colormap, alpha=alpha, vmin=-vmax_display, vmax=vmax_display)
                else:
                    im = ax.imshow(heatmap_smooth, cmap=colormap, alpha=alpha)
            
            # Clean styling
            ax.axis('off')
            ax.set_title(f'Token: "{token}"', fontsize=14, fontweight='bold', pad=10)
            
            # Add individual colorbar for each subplot
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Similarity", fontsize=10)
        
        # Hide unused subplots
        for j in range(num_tokens, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle('Individual Token Impact Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for suptitle
        plt.show()

    def analyze_image_text_pair(self, image: Image.Image, text: str, debug: bool = False, 
                               heatmap_style: str = 'research', show_matrix: bool = True, 
                               show_individual_tokens: bool = False):
        """
        Performs a focused analysis of an image-text pair.

        1. Computes token-patch cosine similarity.
        2. Optionally displays the full similarity matrix (confusion matrix).
        3. Displays the overall text impact as a smooth gradient heatmap overlaid on the image.
        4. Optionally displays individual token impact heatmaps.

        Args:
            image (Image.Image): The original PIL image.
            text (str): The text prompt.
            debug (bool): If True, prints detailed shapes and values.
            heatmap_style (str): Style of heatmap - 'research', 'thermal', 'attention', or 'classic'.
            show_matrix (bool): Whether to show the token-patch similarity matrix.
            show_individual_tokens (bool): Whether to show individual token impact heatmaps.
        """
        # Get the core similarity matrix
        image_input = self.preprocess(image).unsqueeze(0)
        text_input = tokenize([text])
        tokens, similarity = self.get_token_patch_similarity(image_input, text_input, debug=debug)

        step_num = 1
        
        if show_matrix:
            print("="*60)
            print(f"Step {step_num}: Token-Patch Similarity Matrix")
            print("This shows the raw cosine similarity between every token and every image patch.")
            print("="*60)
            self.plot_token_patch_matrix(tokens, similarity)
            step_num += 1
            print("\n" + "="*60)
            print(f"Step {step_num}: Generating Gradient-Style Saliency Heatmap...")
        else:
            print("="*60)
            print(f"Step {step_num}: Generating Gradient-Style Saliency Heatmap...")
        
        print("This creates a smooth visualization showing where the text concepts align with the image.")
        print("="*60)
        self.visualize_gradient_heatmap(image, similarity, style=heatmap_style)
        step_num += 1

        if show_individual_tokens:
            print("\n" + "="*60)
            print(f"Step {step_num}: Individual Token Impact Analysis")
            print("This shows how each individual word affects different regions of the image.")
            print("="*60)
            self.visualize_individual_token_impacts(image, tokens, similarity, colormap='plasma')

        print("\nAnalysis complete.")
        return tokens, similarity

def load_interpretable_clip(name: str = "ViT-B/32", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load an interpretable version of CLIP that can compute token-patch similarities.
    
    Args:
        name: CLIP model name (e.g., "ViT-B/32")
        device: device to load the model on
        
    Returns:
        InterpretableCLIP model with preprocess function and tokenizer attached
    """
    # Load standard CLIP model
    model, preprocess = load(name, device=device)
    
    # Create interpretable version with same architecture
    interpretable_model = InterpretableCLIP(
        embed_dim=model.visual.proj.shape[1],
        image_resolution=model.visual.input_resolution,
        vision_layers=len(model.visual.transformer.resblocks),
        vision_width=model.visual.transformer.width,
        vision_patch_size=model.visual.conv1.kernel_size[0],
        context_length=model.context_length,
        vocab_size=model.vocab_size,
        transformer_width=model.transformer.width,
        transformer_heads=model.transformer.width // 64,
        transformer_layers=len(model.transformer.resblocks)
    )
    
    # Load the trained weights
    interpretable_model.load_state_dict(model.state_dict())
    interpretable_model = interpretable_model.to(device)
    
    # Attach utilities
    interpretable_model.preprocess = preprocess
    from .simple_tokenizer import SimpleTokenizer
    interpretable_model.tokenizer = SimpleTokenizer()
    
    print("Interpretable CLIP loaded successfully!")
    return interpretable_model 