"""
Test script to verify that attribution scores are now meaningful and different for each token.
"""

import torch
import sys
import os
from PIL import Image

# Add the current directory to Python path
sys.path.append('.')

from clip.interpretable_clip_attention import load_interpretable_clip, tokenize_text

def test_attribution_scores():
    """Test that attribution scores are now meaningful and different."""
    
    print("🧪 Testing Fixed Attribution Scores")
    print("="*50)
    
    # Load model
    model = load_interpretable_clip("ViT-B/32")
    
    # Load test image
    image_path = r"images\dogcat2.png"
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
        
    image = Image.open(image_path).convert('RGB')
    image_input = model.preprocess(image).unsqueeze(0)
    
    # Test text with clear semantic differences
    text = "a photo of a cat"
    text_input = tokenize_text(text)
    
    print(f"📝 Testing text: '{text}'")
    print(f"🖼️ Testing image: {image_path}")
    
    # Run attribution analysis with debug=True
    try:
        tokens, attributed_similarity, debug_info = model.get_attributed_token_patch_similarity(
            image_input, text_input, alpha=0.5, debug=True
        )
        
        print("\n📊 ATTRIBUTION ANALYSIS RESULTS:")
        print("-" * 40)
        
        # Check text attribution scores
        text_attribution = debug_info['text_attribution']
        real_token_indices = debug_info['real_token_indices']
        
        print(f"✅ Found {len(tokens)} meaningful tokens: {tokens}")
        
        print("\n🔤 TEXT ATTRIBUTION SCORES:")
        for i, token in enumerate(tokens):
            original_idx = real_token_indices[i]
            score = text_attribution[original_idx].item()
            print(f"  '{token}': {score:.4f}")
        
        # Check if scores are different (not uniform)
        real_scores = [text_attribution[idx].item() for idx in real_token_indices]
        score_std = torch.std(torch.tensor(real_scores)).item()
        
        if score_std > 0.01:
            print(f"✅ SUCCESS: Attribution scores have variation (std = {score_std:.4f})")
            print(f"   This means different tokens have different importance!")
        else:
            print(f"❌ PROBLEM: Attribution scores are still too uniform (std = {score_std:.4f})")
        
        # Check image attribution scores  
        image_attribution = debug_info['image_attribution']
        image_std = torch.std(image_attribution).item()
        
        print(f"\n🖼️ IMAGE ATTRIBUTION:")
        print(f"  Patch attribution std: {image_std:.4f}")
        print(f"  Top 3 patches: {torch.topk(image_attribution, 3).indices.tolist()}")
        
        if image_std > 0.01:
            print(f"✅ SUCCESS: Image patches have varied importance")
        else:
            print(f"❌ PROBLEM: Image patches are still too uniform")
        
        # Show similarity results
        print(f"\n📈 SIMILARITY MATRIX:")
        print(f"  Shape: {attributed_similarity.shape}")
        print(f"  Range: [{attributed_similarity.min():.4f}, {attributed_similarity.max():.4f}]")
        
        # Show top similarities for key tokens
        if 'cat' in tokens:
            cat_idx = tokens.index('cat')
            top_patches = torch.topk(attributed_similarity[cat_idx], 3)
            print(f"  'cat' focuses on patches: {top_patches.indices.tolist()} (scores: {[f'{s:.3f}' for s in top_patches.values.tolist()]})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_attribution_scores() 