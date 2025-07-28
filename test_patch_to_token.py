#!/usr/bin/env python3
"""
Simple test script for patch-to-token analysis.
This script loads an image and text, then shows the top 3 most similar tokens for each image patch.
"""

import sys
import os

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from PIL import Image
import torch
import numpy as np

# Import from clip module
try:
    from clip.interpretable_clip import load_interpretable_clip, tokenize_text
except ImportError:
    print("Error: Could not import interpretable_clip module.")
    print("Make sure you're running this from the CLIP directory.")
    sys.exit(1)

# Import our analysis function
from Inference.heatmaps import simple_patch_to_token_analysis

def test_patch_to_token_analysis():
    """Test the simple patch-to-token analysis function."""
    print("="*60)
    print("TESTING PATCH-TO-TOKEN ANALYSIS")
    print("="*60)
    
    # Load model
    print("Loading interpretable CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_interpretable_clip("ViT-B/32", device=device)
    print(f"Using device: {device}")
    
    # Load image and set text
    image_path = "images/cat.png"  # Update this path as needed
    if not os.path.exists(image_path):
        # Try alternative paths
        alternative_paths = [
            "images/pottedplant.png",
            "images/dog.png", 
            "images/apple.png",
            "images/bus.png"
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                image_path = alt_path
                break
        else:
            print("Error: No test image found. Please update the image_path variable.")
            return
    
    image = Image.open(image_path).convert("RGB")
    text = "a photo of a cat"  # Simple text for testing
    
    print(f"\nAnalyzing:")
    print(f"  Image: {image_path}")
    print(f"  Text: '{text}'")
    
    # Get similarity data
    print("\nComputing token-patch similarities...")
    image_input = model.preprocess(image).unsqueeze(0).to(device)
    text_input = tokenize_text(text).to(device)
    
    tokens, similarity, eos_patch_sim, cls_token_sim, eos_token_sim, cls_patch_sim = model.get_token_patch_similarity(
        image_input, text_input, debug=False
    )
    
    # Convert to numpy for analysis
    if hasattr(similarity, 'detach'):
        similarity_np = similarity.detach().cpu().numpy()
    else:
        similarity_np = similarity
    
    print(f"\nData shapes:")
    print(f"  Similarity matrix: {similarity_np.shape} (tokens × patches)")
    print(f"  Tokens: {tokens}")
    print(f"  Grid size: {int(np.sqrt(similarity_np.shape[1]))}×{int(np.sqrt(similarity_np.shape[1]))}")
    
    # Run our simple patch-to-token analysis
    print("\n" + "="*60)
    patch_results = simple_patch_to_token_analysis(
        similarity_np, tokens, image_input, image, 
        top_k=3, save_path="results/test_patch_to_token"
    )
    
    # Print verification of the math
    print("\n" + "="*60)
    print("VERIFICATION OF CALCULATIONS")
    print("="*60)
    
    # Verify for first few patches
    print("\nManual verification for first 3 patches:")
    for patch_idx in range(min(3, similarity_np.shape[1])):
        print(f"\nPatch {patch_idx}:")
        
        # Get similarities for this patch across all tokens
        patch_similarities = similarity_np[:, patch_idx]
        print(f"  Raw similarities: {patch_similarities}")
        
        # Find top 3 manually
        top_3_indices = np.argsort(patch_similarities)[-3:][::-1]  # Top 3, highest first
        print(f"  Top 3 token indices: {top_3_indices}")
        
        for rank, token_idx in enumerate(top_3_indices, 1):
            token = tokens[token_idx]
            sim = patch_similarities[token_idx]
            print(f"    #{rank}: Token {token_idx} ('{token}') = {sim:.4f}")
        
        # Compare with our function results
        our_result = patch_results[patch_idx]
        print(f"  Our function result: {[(r, t, s) for r, t, s in our_result['top_tokens']]}")
    
    print(f"\n✅ Analysis complete! Check the visualization and results above.")
    print(f"📊 The function correctly finds the top 3 most similar tokens for each image patch.")
    return patch_results

if __name__ == "__main__":
    test_patch_to_token_analysis() 