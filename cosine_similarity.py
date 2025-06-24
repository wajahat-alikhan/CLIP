"""
Simple test script for pure cosine similarity computation between image patches and text tokens.
No frequency weighting, no visualizations - just the raw similarity matrix.
"""

import torch
from PIL import Image
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clip.interpretable_clip import load_interpretable_clip, tokenize

def main():
    print("=== PURE COSINE SIMILARITY TEST ===")
    
    # Load model
    print("Loading interpretable CLIP model...")
    model = load_interpretable_clip("ViT-B/32", device="cpu")
    preprocess = model.preprocess
    
    # Load image and text
    print("Loading image and text...")
    image = preprocess(Image.open("D:\Wajahat Ali Khan\CLIP\dogcat2.PNG")).unsqueeze(0)
    text_input = tokenize(["A photo of a cat and a dog"]).to("cpu")
    
    print(f"Image shape: {image.shape}")
    print(f"Text tokens: {text_input}")
    
    # Compute pure cosine similarities
    print("\nComputing cosine similarities...")
    with torch.no_grad():
        tokens, similarity = model.get_token_patch_similarity(
            image, text_input, debug=True
        )
    
    # Convert to numpy for easier handling
    similarity_np = similarity.cpu().numpy()
    
    print(f"\n=== RESULTS ===")
    print(f"Tokens: {tokens}")
    print(f"Similarity matrix shape: {similarity_np.shape}")
    print(f"Grid size: {int(np.sqrt(similarity_np.shape[1]))}x{int(np.sqrt(similarity_np.shape[1]))}")
    print(f"Similarity range: [{similarity_np.min():.4f}, {similarity_np.max():.4f}]")
    
    # Print similarity statistics for each token
    print(f"\n=== TOKEN STATISTICS ===")
    for i, token in enumerate(tokens):
        token_sim = similarity_np[i]
        print(f"{token:8s}: min={token_sim.min():.4f}, max={token_sim.max():.4f}, "
              f"mean={token_sim.mean():.4f}, std={token_sim.std():.4f}")
    
    # Print the full similarity matrix
    print(f"\n=== SIMILARITY MATRIX ===")
    print("Tokens x Patches (rows=tokens, cols=patches)")
    print("Patch indices: 0-48 (7x7 grid)")
    print("\nSimilarity values:")
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print(similarity_np)
    
    # Show spatial arrangement
    grid_size = int(np.sqrt(similarity_np.shape[1]))
    print(f"\n=== SPATIAL ARRANGEMENT (for reference) ===")
    print(f"Patches arranged in {grid_size}x{grid_size} grid:")
    for i in range(grid_size):
        row_indices = [i * grid_size + j for j in range(grid_size)]
        print(f"Row {i}: patches {row_indices}")
    
    # Find highest similarity patches for each token
    print(f"\n=== TOP PATCHES PER TOKEN ===")
    for i, token in enumerate(tokens):
        token_sim = similarity_np[i]
        top_indices = np.argsort(token_sim)[-3:][::-1]  # Top 3 patches
        print(f"{token:8s}: ", end="")
        for j, patch_idx in enumerate(top_indices):
            row, col = patch_idx // grid_size, patch_idx % grid_size
            print(f"patch_{patch_idx}({row},{col})={token_sim[patch_idx]:.3f}", end="")
            if j < len(top_indices) - 1:
                print(", ", end="")
        print()

if __name__ == "__main__":
    main() 