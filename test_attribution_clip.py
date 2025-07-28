#!/usr/bin/env python3
"""
Test script for Attribution-Weighted Interpretable CLIP

This script demonstrates the new principled approach to CLIP interpretability
that weights embeddings by their actual contribution to final representations.

Usage:
    python test_attribution_clip.py
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

# Import our attribution-weighted interpretable CLIP
from clip.interpretable_clip_attention import load_interpretable_clip, tokenize_text

def create_test_image():
    """Create a simple test image with colored squares for testing"""
    # Create a 224x224 image with colored squares
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Red square (top-left)
    img[50:100, 50:100, 0] = 255  
    
    # Green square (top-right)
    img[50:100, 150:200, 1] = 255
    
    # Blue square (bottom-left)
    img[150:200, 50:100, 2] = 255
    
    # White square (bottom-right)
    img[150:200, 150:200, :] = 255
    
    return Image.fromarray(img)

def test_attribution_weighted_clip():
    """
    Test the attribution-weighted CLIP interpretability approach
    """
    print("🚀 Testing Attribution-Weighted Interpretable CLIP")
    print("=" * 60)
    
    # Load the model (this will print model info)
    print("\n📥 Loading model...")
    model = load_interpretable_clip("ViT-B/32", device="cpu")  # Use CPU for testing
    
    # Create test data
    print("\n🖼️  Creating test image...")
    test_image = create_test_image()
    test_text = "red and green squares"
    
    # Preprocess inputs
    image_tensor = model.preprocess(test_image).unsqueeze(0)
    text_tensor = tokenize_text(test_text)
    
    print(f"✓ Image tensor shape: {image_tensor.shape}")
    print(f"✓ Text tensor shape: {text_tensor.shape}")
    print(f"✓ Text: '{test_text}'")
    
    # Test 1: Original approach (baseline)
    print("\n" + "="*60)
    print("TEST 1: ORIGINAL APPROACH (BASELINE)")
    print("="*60)
    
    try:
        tokens_orig, token_patch_sim, eos_patch_sim, cls_token_sim, eos_token_sim, cls_patch_sim = model.get_token_patch_similarity(
            image_tensor, text_tensor, debug=True
        )
        
        print(f"\n✅ Original approach completed successfully!")
        print(f"   Tokens: {tokens_orig}")
        print(f"   Token-patch similarity shape: {token_patch_sim.shape}")
        print(f"   Similarity range: [{token_patch_sim.min():.4f}, {token_patch_sim.max():.4f}]")
        
    except Exception as e:
        print(f"❌ Original approach failed: {e}")
        token_patch_sim = None
    
    # Test 2: Attribution-weighted approach (NEW)
    print("\n" + "="*60)
    print("TEST 2: ATTRIBUTION-WEIGHTED APPROACH (NEW)")
    print("="*60)
    
    try:
        tokens_attr, attributed_similarity, debug_info = model.get_attributed_token_patch_similarity(
            image_tensor, text_tensor, alpha=0.5, debug=True
        )
        
        print(f"\n✅ Attribution-weighted approach completed successfully!")
        print(f"   Tokens: {tokens_attr}")
        print(f"   Attribution-weighted similarity shape: {attributed_similarity.shape}")
        print(f"   Similarity range: [{attributed_similarity.min():.4f}, {attributed_similarity.max():.4f}]")
        
        # Show attribution scores
        print(f"\n📊 Attribution Analysis:")
        print(f"   Text attribution shape: {debug_info['text_attribution'].shape}")
        print(f"   Image attribution shape: {debug_info['image_attribution'].shape}")
        print(f"   Alpha (grad vs attention balance): {debug_info['alpha']}")
        
        # Show top attributed tokens and patches
        text_attr = debug_info['text_attribution'][debug_info['real_token_indices']]
        image_attr = debug_info['image_attribution']
        
        print(f"\n🔍 Top Contributing Elements:")
        top_token_indices = torch.topk(text_attr, min(3, len(text_attr))).indices
        for i, idx in enumerate(top_token_indices):
            token = tokens_attr[idx]
            score = text_attr[idx].item()
            print(f"   Token #{i+1}: '{token}' (attribution: {score:.4f})")
        
        top_patch_indices = torch.topk(image_attr, min(5, len(image_attr))).indices
        grid_size = int(np.sqrt(len(image_attr)))
        print(f"\n   Top {len(top_patch_indices)} image patches (out of {len(image_attr)} total):")
        for i, idx in enumerate(top_patch_indices):
            row, col = idx // grid_size, idx % grid_size
            score = image_attr[idx].item()
            print(f"   Patch #{i+1}: position ({row}, {col}) (attribution: {score:.4f})")
        
    except Exception as e:
        print(f"❌ Attribution-weighted approach failed: {e}")
        import traceback
        traceback.print_exc()
        attributed_similarity = None
    
    # Test 3: Compare approaches
    if token_patch_sim is not None and attributed_similarity is not None:
        print("\n" + "="*60)
        print("TEST 3: COMPARING APPROACHES")
        print("="*60)
        
        print(f"\n📈 Similarity Statistics:")
        print(f"   Original approach:")
        print(f"     Mean: {token_patch_sim.mean():.4f}")
        print(f"     Std:  {token_patch_sim.std():.4f}")
        print(f"     Range: [{token_patch_sim.min():.4f}, {token_patch_sim.max():.4f}]")
        
        print(f"   Attribution-weighted approach:")
        print(f"     Mean: {attributed_similarity.mean():.4f}")
        print(f"     Std:  {attributed_similarity.std():.4f}")
        print(f"     Range: [{attributed_similarity.min():.4f}, {attributed_similarity.max():.4f}]")
        
        # Calculate difference
        similarity_diff = torch.abs(attributed_similarity - token_patch_sim)
        print(f"   Absolute difference:")
        print(f"     Mean: {similarity_diff.mean():.4f}")
        print(f"     Max:  {similarity_diff.max():.4f}")
        
        # Show which approach gives higher similarities for each token
        print(f"\n🔄 Per-token comparison:")
        for i, token in enumerate(tokens_attr):
            orig_max = token_patch_sim[i].max().item()
            attr_max = attributed_similarity[i].max().item()
            winner = "Attribution" if attr_max > orig_max else "Original"
            print(f"   '{token}': Original={orig_max:.3f}, Attribution={attr_max:.3f} → {winner} wins")
    
    # Test 4: Different alpha values
    print("\n" + "="*60) 
    print("TEST 4: TESTING DIFFERENT ALPHA VALUES")
    print("="*60)
    
    alpha_values = [0.0, 0.5, 1.0]  # 0.0 = only attention, 1.0 = only gradients
    alpha_results = {}
    
    for alpha in alpha_values:
        print(f"\n🎛️  Testing alpha = {alpha} ({'only attention' if alpha == 0.0 else 'only gradients' if alpha == 1.0 else 'balanced'})")
        
        try:
            tokens_alpha, similarity_alpha, debug_alpha = model.get_attributed_token_patch_similarity(
                image_tensor, text_tensor, alpha=alpha, debug=False
            )
            
            alpha_results[alpha] = similarity_alpha
            print(f"   ✅ Alpha {alpha}: Mean similarity = {similarity_alpha.mean():.4f}")
            
        except Exception as e:
            print(f"   ❌ Alpha {alpha} failed: {e}")
    
    if len(alpha_results) > 1:
        print(f"\n📊 Alpha comparison:")
        for alpha, sim in alpha_results.items():
            label = "only attention" if alpha == 0.0 else "only gradients" if alpha == 1.0 else "balanced"
            print(f"   Alpha {alpha} ({label}): mean={sim.mean():.4f}, max={sim.max():.4f}")
    
    print("\n" + "="*60)
    print("🎉 TESTING COMPLETE!")
    print("="*60)
    print("\n💡 Key Insights:")
    print("   • Attribution-weighted approach respects CLIP's training objective")
    print("   • Gradients show actual influence on final representations") 
    print("   • Attention shows what the model 'looks at'")
    print("   • Combining both (alpha=0.5) gives balanced interpretability")
    print("   • Use attributed similarities for meaningful token-patch analysis!")

if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Run tests
    test_attribution_weighted_clip() 