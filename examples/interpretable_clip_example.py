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
    
    # Visualize patch grid to verify patch ordering
    print("Visualizing patch grid to verify patch ordering...")
    model.visualize_patch_grid(image)
    
    # Print real tokens and their indices
    print("\nComputing token-patch similarity...")
    tokens, similarity = model.get_token_patch_similarity(
        model.preprocess(image).unsqueeze(0),
        tokenize([text]),
        debug=True  # Enable debug output to see similarity ranges
    )
    print("Real tokens and their indices:")
    for i, t in enumerate(tokens):
        print(f"{i}: '{t}'")
    
    # Plot token-patch similarity matrix (confusion matrix style)
    print("\nPlotting token-patch similarity matrix...")
    model.plot_token_patch_matrix(tokens, similarity)
    
    # Visualize each token's highest similarity patches
    for i, token in enumerate(tokens):
        print(f"\nVisualizing patches with highest similarity for token '{token}'...")
        model.visualize_high_similarity_patches(image, text, token=token, top_k=5, debug=True)
    
    # Visualize overall aggregate similarity
    print("\nVisualizing overall patch similarity...")
    model.visualize_overall_patch_heatmap(image, text, agg="mean", debug=True)
    
    print("\nFinished!")

    fig, axs = plt.subplots(1, 3, figsize=(24, 6))

    # Original image
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    sim_np = similarity.detach().cpu().numpy()
    # Raw similarity overlay
    axs[1].imshow(sim_np, cmap='coolwarm', alpha=0.5)
    axs[1].set_title("Raw Similarity Overlay\n(Red=Positive, Blue=Negative)")
    axs[1].axis('off')

    # Absolute similarity overlay
    axs[2].imshow(np.abs(sim_np), cmap='hot', alpha=0.5)
    axs[2].set_title("Absolute Similarity Overlay\n(Magnitude Only)")
    axs[2].axis('off')

    # Adjust layout to push colorbars to the far right
    plt.subplots_adjust(left=0.05, right=0.75, top=0.9, bottom=0.1, wspace=0.2)
    plt.show()

if __name__ == "__main__":
    main() 