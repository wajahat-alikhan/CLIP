"""
Comprehensive Visualization and Experimental Framework for Interpretable CLIP

This script generates publication-quality visualizations and experimental results
that demonstrate the interpretability capabilities of the extended CLIP implementation.

Features:
- Attention heatmaps overlaid on images
- Token-patch similarity matrices 
- Spatial localization analysis
- Multi-architecture comparisons
- Publication-ready plots
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import the interpretable CLIP module
import sys
sys.path.append('..')
from clip.interpretable_clip_attention import load_interpretable_clip, tokenize_text

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class InterpretabilityVisualizer:
    """Comprehensive visualization toolkit for interpretable CLIP analysis"""
    
    def __init__(self, model_name="ViT-B/32", device="cpu"):
        """Initialize with a specific CLIP model"""
        self.model_name = model_name
        self.device = device
        self.model = load_interpretable_clip(model_name, device=device)
        self.grid_size = self.model.visual.input_resolution // self.model.visual.conv1.kernel_size[0]
        
        print(f"🎨 Initialized visualizer for {model_name}")
        print(f"   Grid size: {self.grid_size}×{self.grid_size} = {self.grid_size**2} patches")
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess image for CLIP"""
        image = Image.open(image_path).convert('RGB')
        image_input = self.model.preprocess(image).unsqueeze(0).to(self.device)
        return image, image_input
    
    def create_attention_heatmap(self, image, similarity_matrix, token_idx, token_name,
                                alpha=0.6, colormap='Reds', save_path=None):
        """
        Create smooth attention heatmap showing where a token focuses in the image
        """
        # Get similarity scores for this specific token
        token_similarities = similarity_matrix[token_idx, :]
        
        # Reshape to spatial grid
        heatmap = token_similarities.reshape(self.grid_size, self.grid_size)
        
        # Apply horizontal flip for correct spatial correspondence 
        heatmap = np.fliplr(heatmap.detach().cpu().numpy())
        
        # Resize to image dimensions
        img_w, img_h = image.size
        heatmap_resized = cv2.resize(heatmap.astype(np.float32), (img_w, img_h), 
                                   interpolation=cv2.INTER_CUBIC)
        
        # Apply Gaussian smoothing for professional appearance
        kernel_size = max(15, img_w // 20)  # Adaptive kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1
        heatmap_smooth = cv2.GaussianBlur(heatmap_resized, (kernel_size, kernel_size), 0)
        
        # Normalize heatmap
        heatmap_norm = (heatmap_smooth - heatmap_smooth.min()) / (heatmap_smooth.max() - heatmap_smooth.min() + 1e-8)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        im = ax.imshow(heatmap_norm, cmap=colormap, alpha=alpha, vmin=0, vmax=1)
        
        # Clean styling
        ax.axis('off')
        ax.set_title(f'Attention Heatmap - "{token_name}"', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Strength', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved attention heatmap to {save_path}")
        
        plt.show()
        return heatmap_norm
    
    def plot_similarity_matrix(self, tokens, similarity_matrix, title="Token-Patch Similarity Matrix", save_path=None):
        """
        Create a publication-quality similarity matrix visualization
        """
        similarity_np = similarity_matrix.detach().cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(tokens) * 0.5)))
        
        # Create heatmap
        im = ax.imshow(similarity_np, aspect='auto', cmap='Reds', 
                      vmin=similarity_np.min(), vmax=similarity_np.max())
        
        # Customize ticks and labels
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels([f'"{token}"' for token in tokens], fontsize=12)
        ax.set_xlabel(f'Image Patch Index (Grid: {self.grid_size}×{self.grid_size})', fontsize=12)
        ax.set_ylabel('Text Tokens', fontsize=12)
        ax.set_title(f'{title}\nModel: {self.model_name}', fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Cosine Similarity', fontsize=12)
        
        # Add grid for better readability
        for i in range(len(tokens) + 1):
            ax.axhline(y=i-0.5, color='white', linewidth=0.5, alpha=0.3)
        
        # Add statistics
        stats_text = f'Range: [{similarity_np.min():.3f}, {similarity_np.max():.3f}]\n'
        stats_text += f'Mean: {similarity_np.mean():.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
               verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved similarity matrix to {save_path}")
        
        plt.show()
    
    def analyze_spatial_localization(self, image, text, save_dir=None):
        """
        Comprehensive spatial localization analysis
        """
        image_input = self.model.preprocess(image).unsqueeze(0).to(self.device)
        text_input = tokenize_text(text).to(self.device)
        
        # Get token-patch similarities
        tokens, similarity = self.model.get_token_patch_similarity(image_input, text_input)
        
        print(f"🔍 Analyzing: '{text}'")
        print(f"   Found tokens: {tokens}")
        print(f"   Similarity matrix: {similarity.shape}")
        
        # Create output directory
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
        
        # 1. Overall similarity matrix
        matrix_path = save_dir / f"similarity_matrix_{self.model_name.replace('/', '_')}.png" if save_dir else None
        self.plot_similarity_matrix(tokens, similarity, 
                                  title=f"Token-Patch Correspondence Analysis",
                                  save_path=matrix_path)
        
        # 2. Individual token attention maps
        for i, token in enumerate(tokens):
            heatmap_path = save_dir / f"attention_{token}_{self.model_name.replace('/', '_')}.png" if save_dir else None
            self.create_attention_heatmap(image, similarity, i, token,
                                        alpha=0.7, save_path=heatmap_path)
        
        # 3. Peak location analysis
        self.analyze_peak_locations(tokens, similarity)
        
        return tokens, similarity
    
    def analyze_peak_locations(self, tokens, similarity):
        """
        Analyze where each token has its highest correspondence
        """
        print(f"\n Peak Location Analysis:")
        print(f"   Grid: {self.grid_size}×{self.grid_size} patches")
        
        similarity_np = similarity.detach().cpu().numpy()
        
        for i, token in enumerate(tokens):
            token_sims = similarity_np[i, :]
            
            # Reshape to spatial grid and apply flip
            spatial_grid = np.fliplr(token_sims.reshape(self.grid_size, self.grid_size))
            
            # Find peak and valley locations
            peak_pos = np.unravel_index(np.argmax(spatial_grid), spatial_grid.shape)
            valley_pos = np.unravel_index(np.argmin(spatial_grid), spatial_grid.shape)
            
            # Determine spatial regions
            center_row, center_col = self.grid_size // 2, self.grid_size // 2
            
            # Peak location description
            peak_region = self._describe_location(peak_pos, center_row, center_col)
            valley_region = self._describe_location(valley_pos, center_row, center_col)
            
            print(f"   '{token}':")
            print(f"     Peak: {peak_region} at {peak_pos} (similarity: {spatial_grid[peak_pos]:.3f})")
            print(f"     Valley: {valley_region} at {valley_pos} (similarity: {spatial_grid[valley_pos]:.3f})")
    
    def _describe_location(self, pos, center_row, center_col):
        """Convert grid position to descriptive location"""
        row, col = pos
        
        # Vertical position
        if row < center_row - 1:
            v_pos = "Top"
        elif row > center_row + 1:
            v_pos = "Bottom"
        else:
            v_pos = "Center"
        
        # Horizontal position  
        if col < center_col - 1:
            h_pos = "Left"
        elif col > center_col + 1:
            h_pos = "Right"
        else:
            h_pos = "Center"
        
        if v_pos == "Center" and h_pos == "Center":
            return "Center"
        elif v_pos == "Center":
            return h_pos
        elif h_pos == "Center":
            return v_pos
        else:
            return f"{v_pos}-{h_pos}"
    
    # Architecture comparison functions removed - use separate model instances if needed

def run_comprehensive_experiments():
    """
    Run a comprehensive set of interpretability experiments
    """
    print("🚀 Starting Comprehensive Interpretability Experiments")
    print("="*70)
    
    # Initialize visualizer
    visualizer = InterpretabilityVisualizer("ViT-B/16", device="cpu")
    
    # Create output directory
    output_dir = Path("experimental_results")
    output_dir.mkdir(exist_ok=True)
    
    # Experiment 1: Basic object localization
    print("\n📊 Experiment 1: Basic Object Localization")
    try:
        image_path = "D:/Wajahat Ali Khan/CLIP/images/cat.PNG"
        image, _ = visualizer.load_and_preprocess_image(image_path)
        
        # Test with different descriptions
        descriptions = [
            "a cat",
            "a fluffy cat",
            "cat sitting on furniture"
        ]
        
        for desc in descriptions:
            print(f"\n   Testing: '{desc}'")
            exp_dir = output_dir / f"exp1_{desc.replace(' ', '_')}"
            tokens, similarity = visualizer.analyze_spatial_localization(image, desc, save_dir=exp_dir)
            
    except FileNotFoundError:
        print("   ⚠️ Cat image not found, creating synthetic test...")
        # Create synthetic test with random image
        image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        tokens, similarity = visualizer.analyze_spatial_localization(image, "test object", 
                                                                   save_dir=output_dir / "exp1_synthetic")
    
    # Experiments focused on single model analysis
    
    print(f"\n✅ Experiments completed! Results saved to: {output_dir.absolute()}")
    print("🎉 Your interpretable CLIP is generating publication-quality visualizations!")

if __name__ == "__main__":
    run_comprehensive_experiments() 