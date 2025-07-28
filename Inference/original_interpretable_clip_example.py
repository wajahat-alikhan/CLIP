#DO NOT CHANGE OR MODIFY THIS

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from clip.interpretable_clip_attention import load_interpretable_clip, tokenize_text

def plot_similarity_matrix(similarity_np, tokens, grid_size, text, save_path=None):
    """Plot confusion matrix showing token-patch similarities."""
    plt.figure(figsize=(12, 8))
    
    # Create the heatmap
    im = plt.imshow(similarity_np, cmap='viridis', aspect='auto')
    
    # Set labels
    plt.xlabel(f'Image Patches ({grid_size}×{grid_size} grid)', fontsize=12)
    plt.ylabel('Text Tokens', fontsize=12)
    plt.title(f'Token-Patch Similarity Matrix\nText: "{text}"', fontsize=14, pad=20)
    
    # Set y-axis labels to token names
    plt.yticks(range(len(tokens)), tokens, fontsize=10)
    
    # Set x-axis labels (show every 5th patch)
    patch_labels = [f'P{i}' for i in range(similarity_np.shape[1])]
    step = max(1, len(patch_labels) // 10)  # Show ~10 labels max
    plt.xticks(range(0, len(patch_labels), step), 
              [patch_labels[i] for i in range(0, len(patch_labels), step)], 
              rotation=45, fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}_similarity_matrix.png", dpi=300, bbox_inches='tight')
        print(f"Similarity matrix saved to {save_path}_similarity_matrix.png")
    
    plt.show()

def plot_attention_overlay(similarity_np, tokens, grid_size, image, text, save_path=None):
    """Plot heatmaps overlaid on the original image."""
    # Resize image for overlay
    image_resized = image.resize((224, 224))
    
    # Create subplots for each significant token
    num_tokens = len(tokens)
    cols = min(3, num_tokens)
    rows = (num_tokens + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if num_tokens == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle(f'Token Heatmap Overlays\nText: "{text}"', fontsize=16, y=0.98)
    
    # Custom colormap for heatmap (transparent to red)
    colors = ['white', 'yellow', 'orange', 'red']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('attention', colors, N=n_bins)
    
    for i, token in enumerate(tokens):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Get attention for this token
        attention = similarity_np[i, :]
        
        # Reshape to spatial grid
        attention_grid = attention.reshape(grid_size, grid_size)
        
        # Normalize attention values
        attention_norm = (attention_grid - attention_grid.min()) / (attention_grid.max() - attention_grid.min())
        
        # Display original image
        ax.imshow(image_resized)
        
        # Overlay attention heatmap
        attention_upsampled = np.kron(attention_norm, np.ones((224//grid_size, 224//grid_size)))
        
        # Crop/pad to exactly 224x224 if needed
        if attention_upsampled.shape[0] != 224:
            pad_h = 224 - attention_upsampled.shape[0]
            attention_upsampled = np.pad(attention_upsampled, ((0, pad_h), (0, 0)), mode='edge')
        if attention_upsampled.shape[1] != 224:
            pad_w = 224 - attention_upsampled.shape[1]
            attention_upsampled = np.pad(attention_upsampled, ((0, 0), (0, pad_w)), mode='edge')
        
        # Apply overlay with transparency
        overlay = ax.imshow(attention_upsampled, cmap=cmap, alpha=0.6, vmin=0, vmax=1)
        
        # Find peak attention location
        peak_pos = np.unravel_index(np.argmax(attention_grid), attention_grid.shape)
        peak_value = attention_grid[peak_pos]
        
        #ax.set_title(f"'{token}'\nPeak: {peak_value:.3f} at {peak_pos}", fontsize=12)
        ax.set_title(f"'{token}'", fontsize=12)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_tokens, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}_attention_overlay.png", dpi=300, bbox_inches='tight')
        print(f"Attention overlay saved to {save_path}_attention_overlay.png")
    
    plt.show()

def main():
    # --- Setup ---
    print("INTERPRETABLE CLIP VISUALIZATION")
    
    print("Loading interpretable CLIP model...")
    model = load_interpretable_clip("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Load image and set text prompt
    image_path = r"D:/Wajahat Ali Khan/CLIP/images/dog.png"
    image = Image.open(image_path).convert("RGB")
    text = "an image of a dog"
    
    print(f"\nAnalyzing image: {image_path}")
    print(f"Text prompt: '{text}'")
    
    # --- Get similarity data ---
    print("\nComputing token-patch similarities...")
    image_input = model.preprocess(image).unsqueeze(0)
    text_input = tokenize_text(text)
    tokens, similarity = model.get_token_patch_similarity(image_input, text_input, debug=False)
    
    # Convert to numpy for visualization
    if hasattr(similarity, 'detach'):
        similarity_np = similarity.detach().cpu().numpy()
    else:
        similarity_np = similarity
        
    grid_size = int(np.sqrt(similarity_np.shape[1]))
    
    # Print basic info
    print(f"\nResults:")
    print(f"  Tokens found: {tokens}")
    print(f"  Grid size: {grid_size}×{grid_size} = {similarity_np.shape[1]} patches")
    print(f"  Similarity range: [{similarity_np.min():.4f}, {similarity_np.max():.4f}]")
    
    # Find most relevant tokens
    max_sims = np.max(similarity_np, axis=1)
    sorted_indices = np.argsort(max_sims)[::-1]
    
    print(f"\n  Most relevant tokens:")
    for i in sorted_indices[:3]:
        token = tokens[i]
        peak_sim = max_sims[i]
        peak_patch = np.argmax(similarity_np[i, :])
        patch_row, patch_col = divmod(peak_patch, grid_size)
        print(f"    '{token}': {peak_sim:.4f} at patch ({patch_row}, {patch_col})")
    
    # --- MAIN VISUALIZATIONS ---
    print("GENERATING MAIN VISUALIZATIONS")
    
    # 1. Similarity Matrix (Confusion Matrix)
    print("\n1. Text Token-Image Patch Similarity Matrix:")
    plot_similarity_matrix(similarity_np, tokens, grid_size, text, save_path="results/dog_analysis")
    
    # 2. Token Overlays
    print("\n2. Token Overlays on Original Image:")
    plot_attention_overlay(similarity_np, tokens, grid_size, image, text, save_path="results/dog_analysis")
    

    print("VISUALIZATION COMPLETE!")
   
    print("Both visualizations show how each text token corresponds to different image regions.")
    print("- Similarity Matrix: Shows raw cosine similarity values between tokens and patches")
    print("- Similarity Overlays: Shows similarity heatmaps overlaid on the original image")

if __name__ == "__main__":
    main() 

# DONOT CHANGE OR MODIFY THIS