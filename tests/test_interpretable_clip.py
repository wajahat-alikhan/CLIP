"""
Test script for Interpretable CLIP

This script demonstrates how to use the interpretable CLIP implementation
to analyze fine-grained correspondences between text tokens and image patches.
"""

import torch
from PIL import Image
import requests
from io import BytesIO

# Import the interpretable CLIP module
from clip.interpretable_clip import load_interpretable_clip, tokenize_text

def test_basic_functionality():
    """Test basic interpretable CLIP functionality"""
    print("="*60)
    print("Testing Interpretable CLIP Basic Functionality")
    print("="*60)
    
    # Load interpretable CLIP model
    print("Loading interpretable CLIP model...")
    model = load_interpretable_clip("ViT-B/32") #B/16, B/32, L/14
    
    # Test with a sample image (you can replace with your own)
    print("\nTesting with sample inputs...")
    
    # Create a simple test image (or load your own)
    # For this example, let's assume you have cat.PNG and dog.PNG files
    try:
        image = Image.open("D:/Wajahat Ali Khan/CLIP/images/apple.png")
        print("Loaded image")
    except FileNotFoundError:
        print("image not found, creating a dummy image for testing...")
        # Create a dummy image for testing
        import numpy as np
        dummy_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(dummy_array)
    
    # Prepare inputs
    image_input = model.preprocess(image).unsqueeze(0)
    text_input = tokenize_text("an image of eight apples")
    
    print(f"Image input shape: {image_input.shape}")
    print(f"Text input shape: {text_input.shape}")
    
    # Test core functionality
    print("\n" + "-"*40)
    print("Testing encode_image_with_patches...")
    cls_embedding, patch_embeddings = model.encode_image_with_patches(image_input)
    print(f"CLS embedding shape: {cls_embedding.shape}")
    print(f"Patch embeddings shape: {patch_embeddings.shape}")
    
    print("\nTesting encode_text_with_tokens...")
    pooled_embedding, token_embeddings = model.encode_text_with_tokens(text_input)
    print(f"Pooled embedding shape: {pooled_embedding.shape}")
    print(f"Token embeddings shape: {token_embeddings.shape}")
    
    print("\nTesting get_token_patch_similarity...")
    tokens, similarity = model.get_token_patch_similarity(image_input, text_input, debug=True)
    print(f"Filtered tokens: {tokens}")
    print(f"Similarity matrix shape: {similarity.shape}")
    print(f"Similarity range: [{similarity.min().item():.4f}, {similarity.max().item():.4f}]")
    
    # Verify compatibility with original CLIP
    print("\n" + "-"*40)
    print("Verifying compatibility with original CLIP...")
    
    # Move inputs to model device for original CLIP methods
    image_input_gpu = image_input.to(next(model.parameters()).device)
    text_input_gpu = text_input.to(next(model.parameters()).device)
    
    # Compute global similarity using original method
    original_image_features = model.encode_image(image_input_gpu)
    original_text_features = model.encode_text(text_input_gpu)
    
    # Normalize and compute similarity
    original_image_features = original_image_features / original_image_features.norm(dim=1, keepdim=True)
    original_text_features = original_text_features / original_text_features.norm(dim=1, keepdim=True)
    original_similarity = (original_image_features @ original_text_features.T).item()
    
    # Compare with our CLS embedding
    cls_normalized = cls_embedding / cls_embedding.norm(dim=1, keepdim=True)
    pooled_normalized = pooled_embedding / pooled_embedding.norm(dim=1, keepdim=True)
    our_similarity = (cls_normalized @ pooled_normalized.T).item()
    
    print(f"Original CLIP global similarity: {original_similarity:.6f}")
    print(f"Our implementation global similarity: {our_similarity:.6f}")
    print(f"Difference: {abs(original_similarity - our_similarity):.8f}")
    
    if abs(original_similarity - our_similarity) < 1e-6:
        print("Perfect compatibility verified!")
    else:
        print("Compatibility issue detected!")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("You can now use model.get_token_patch_similarity() for interpretability analysis.")
    print("="*60)

if __name__ == "__main__":
    test_basic_functionality() 