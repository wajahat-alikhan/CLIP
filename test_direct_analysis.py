"""
Test Direct CLIP Attention Analysis

Simple test of the direct approach that follows the reference code.
"""

import torch
import numpy as np
from clip.direct_attention_analysis import analyze_clip_direct, visualize_direct_results


def test_direct_analysis():
    """Test the direct CLIP analysis."""
    
    print("🎯 TESTING DIRECT CLIP ATTENTION ANALYSIS")
    print("=" * 70)
    
    # Test with cat image and multiple texts
    image_path = "images/cat.PNG"
    texts = ["a dog", "a cat"]
    
    print(f"📸 Image: {image_path}")
    print(f"📝 Texts: {texts}")
    print()
    
    try:
        # Run direct analysis
        results = analyze_clip_direct(
            image_path=image_path,
            texts=texts,
            debug=True
        )
        
        print(f"\n📊 RESULTS SUMMARY:")
        for key, result in results.items():
            text = result['text']
            attention = result['image_attention']
            prob = result['probabilities'][0][result['index']]
            
            print(f"  '{text}':")
            print(f"    Probability: {prob:.4f}")
            print(f"    Attention std: {attention.std():.6f}")
            print(f"    Attention range: [{attention.min():.6f}, {attention.max():.6f}]")
        
        # Visualize results
        visualize_direct_results(results, image_path, save_prefix="direct_cat")
        
        # Check if attention patterns differ
        if len(results) >= 2:
            attn1 = list(results.values())[0]['image_attention']
            attn2 = list(results.values())[1]['image_attention']
            correlation = torch.corrcoef(torch.stack([attn1.flatten(), attn2.flatten()]))[0, 1]
            print(f"\n📊 Attention correlation between texts: {correlation:.6f}")
            
            if correlation < 0.8:
                print(f"  ✅ Different attention patterns detected!")
            else:
                print(f"  ⚠️ Similar attention patterns")
        
        return results
        
    except Exception as e:
        print(f"❌ Direct analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = test_direct_analysis() 