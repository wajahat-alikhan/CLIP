#Global Similarity

import torch
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clip.interpretable_clip import load_interpretable_clip
from clip.clip import tokenize

def main():
    # Load model
    model = load_interpretable_clip("ViT-B/32")
    
    # Load and preprocess image
    image_path = "D:/Wajahat Ali Khan/CLIP/cat.png"
    try:
        image = Image.open(image_path)
        print(f"Successfully loaded image from: {image_path}")
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        print("Please update the image path in the script.")
        return
    
    # Example text (change as needed)
    text = "A photo of a cat"
    
    # Compute token-patch similarity
    print("Computing token-patch similarity...")
    tokens, similarity = model.get_token_patch_similarity(
        model.preprocess(image).unsqueeze(0),
        tokenize([text])
    )
    
    # Calculate global min/max for consistent scaling across all tokens
    sim_np = similarity.detach().cpu().numpy()
    global_min_raw = sim_np.min()
    global_max_raw = sim_np.max()
    global_min_abs = 0  # Absolute values always start from 0
    global_max_abs = np.abs(sim_np).max()
    
    print(f"Global similarity range: {global_min_raw:.4f} to {global_max_raw:.4f}")
    print(f"Global absolute similarity range: {global_min_abs:.4f} to {global_max_abs:.4f}")
    
    # Create three-panel visualization for each token using consistent scaling
    print("\nCreating three-panel visualizations for each token...")
    
    for i, token in enumerate(tokens):
        print(f"\nCreating visualization for token '{token}'...")
        
        # Extract this token's similarity and create smooth heatmaps
        token_sim = similarity[i].detach().cpu().numpy()
        patch_size = int(np.sqrt(len(token_sim)))
        token_sim_2d = token_sim.reshape(patch_size, patch_size)
        
        # Create smooth heatmaps with proper interpolation
        from scipy.ndimage import zoom
        
        # Calculate scale factor to resize to image dimensions
        target_size = max(image.height, image.width)
        scale_factor = target_size / patch_size
        
        # Create smooth interpolated heatmaps
        raw_heatmap = zoom(token_sim_2d, scale_factor, order=1)
        abs_heatmap = zoom(np.abs(token_sim_2d), scale_factor, order=1)
        
        # Crop or pad to exact image dimensions if needed
        h, w = image.height, image.width
        if raw_heatmap.shape[0] > h or raw_heatmap.shape[1] > w:
            # Crop to image size
            raw_heatmap_resized = raw_heatmap[:h, :w]
            abs_heatmap_resized = abs_heatmap[:h, :w]
        else:
            # Use as is
            raw_heatmap_resized = raw_heatmap
            abs_heatmap_resized = abs_heatmap
        
        # Create three-panel figure for this token
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axs[0].imshow(image)
        axs[0].set_title("Original Image")
        axs[0].axis('off')
        
        # Raw similarity overlay for this token with FIXED scale
        axs[1].imshow(image)
        im1 = axs[1].imshow(raw_heatmap_resized, cmap='coolwarm', alpha=0.5, 
                           vmin=global_min_raw, vmax=global_max_raw)
        axs[1].set_title(f"Raw Similarity for Token: '{token}'\n(Red=Positive, Blue=Negative)")
        axs[1].axis('off')
        
        # Add colorbar for raw similarity with FIXED scale
        divider1 = make_axes_locatable(axs[1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        cbar1 = plt.colorbar(im1, cax=cax1)
        cbar1.set_label('Similarity', rotation=270, labelpad=15)
        
        # Absolute similarity overlay for this token with FIXED scale
        axs[2].imshow(image)
        im2 = axs[2].imshow(abs_heatmap_resized, cmap='hot', alpha=0.5,
                           vmin=global_min_abs, vmax=global_max_abs)
        axs[2].set_title(f"Absolute Similarity for Token: '{token}'\n(Magnitude Only)")
        axs[2].axis('off')
        
        # Add colorbar for absolute similarity with FIXED scale
        divider2 = make_axes_locatable(axs[2])
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        cbar2 = plt.colorbar(im2, cax=cax2)
        cbar2.set_label('|Similarity|', rotation=270, labelpad=15)
        
        # Print this token's actual similarity range for debugging
        token_min = token_sim.min()
        token_max = token_sim.max()
        token_abs_max = np.abs(token_sim).max()
        print(f"  Token '{token}' similarity range: {token_min:.4f} to {token_max:.4f}")
        print(f"  Token '{token}' absolute max: {token_abs_max:.4f}")
        
        # Adjust layout
        plt.tight_layout()
        plt.suptitle(f"Token-Patch Similarity Overlays for '{token}'", fontsize=16, y=1.02)
        plt.show()
    
    print("\nFinished!")

if __name__ == "__main__":
    main() 