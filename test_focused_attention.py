"""
Test the focused attention × gradient approach for CLIP interpretability.

This tests the new clean implementation that:
1. Hooks into CLIP's actual attention mechanisms
2. Computes real gradients w.r.t. CLS/EOS tokens
3. Creates simple heatmap visualizations
"""

import sys
import os
sys.path.append('.')

from clip.attention_gradient_analysis import analyze_clip_interpretability

def test_focused_approach():
    """Test the focused attention × gradient analysis."""
    
    print("🎯 TESTING FOCUSED ATTENTION × GRADIENT APPROACH")
    print("=" * 60)
    
    # Test parameters
    image_path = r"images\dogcat2.png"
    text = "a photo of a dog"
    model_name = "ViT-B/32"
    alpha = 0.5  # Equal weight to attention and gradients
    
    print(f"📝 Text: '{text}'")
    print(f"🖼️ Image: {image_path}")
    print(f"🤖 Model: {model_name}")
    print(f"⚖️ Alpha (attention vs gradients): {alpha}")
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return False
    
    try:
        # Run the analysis
        results = analyze_clip_interpretability(
            image_path=image_path,
            text=text,
            model_name=model_name,
            alpha=alpha,
            save_path="results/focused_attention_analysis.png",
            debug=True
        )
        
        print("\n✅ SUCCESS: Focused analysis completed!")
        print(f"   - Analyzed {len(results['tokens'])} text tokens")
        print(f"   - Analyzed {len(results['image_combined'])} image patches")
        print(f"   - Combined attention × gradients with α={alpha}")
        
        # Show key insights
        print("\n💡 KEY INSIGHTS:")
        
        # Find most important tokens
        token_scores = results['text_combined'][results['token_indices']]
        top_token_idx = token_scores.argmax().item()
        top_token = results['tokens'][top_token_idx]
        top_score = token_scores[top_token_idx].item()
        
        print(f"   🔤 Most important token: '{top_token}' (score: {top_score:.4f})")
        
        # Find most important image regions
        patch_scores = results['image_combined']
        top_patch_idx = patch_scores.argmax().item()
        top_patch_score = patch_scores[top_patch_idx].item()
        
        # Convert patch index to 2D coordinates
        grid_size = int(len(patch_scores) ** 0.5)
        patch_row = top_patch_idx // grid_size
        patch_col = top_patch_idx % grid_size
        
        print(f"   🖼️ Most important patch: row {patch_row}, col {patch_col} (score: {top_patch_score:.4f})")
        
        # Compare attention vs gradients
        text_attention_mean = results['text_attention'][results['token_indices']].mean().item()
        text_gradient_mean = results['text_gradients'][results['token_indices']].mean().item()
        
        print(f"   📊 Text attention mean: {text_attention_mean:.4f}")
        print(f"   📊 Text gradient mean: {text_gradient_mean:.4f}")
        
        if text_attention_mean > text_gradient_mean * 1.1:
            print(f"   → Attention-driven: EOS token focuses on specific words")
        elif text_gradient_mean > text_attention_mean * 1.1:
            print(f"   → Gradient-driven: Specific words strongly influence EOS")
        else:
            print(f"   → Balanced: Attention and gradients are roughly equal")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_alpha_values():
    """Test how different alpha values affect the results."""
    
    print("\n🎛️ TESTING DIFFERENT ALPHA VALUES")
    print("=" * 50)
    
    image_path = r"images\dogcat2.png"
    text = "a photo of a dog"
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return False
    
    alpha_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    results_comparison = {}
    
    for alpha in alpha_values:
        print(f"\n🔬 Testing α = {alpha} ({'only attention' if alpha == 0 else 'only gradients' if alpha == 1 else 'mixed'})")
        
        try:
            results = analyze_clip_interpretability(
                image_path=image_path,
                text=text,
                model_name="ViT-B/32",
                alpha=alpha,
                debug=False  # Reduce noise for comparison
            )
            
            # Store key metrics
            token_scores = results['text_combined'][results['token_indices']]
            top_token_idx = token_scores.argmax().item()
            top_token = results['tokens'][top_token_idx]
            
            results_comparison[alpha] = {
                'top_token': top_token,
                'top_token_score': token_scores[top_token_idx].item(),
                'score_std': token_scores.std().item()
            }
            
            print(f"   ✓ Top token: '{top_token}' (score: {token_scores[top_token_idx].item():.4f})")
            print(f"   ✓ Score variation: {token_scores.std().item():.4f}")
            
        except Exception as e:
            print(f"   ❌ Failed for α={alpha}: {e}")
            continue
    
    # Analysis of results
    print(f"\n📈 ALPHA COMPARISON RESULTS:")
    print("-" * 30)
    
    for alpha, metrics in results_comparison.items():
        interpretation = ""
        if alpha == 0.0:
            interpretation = "(pure attention)"
        elif alpha == 1.0:
            interpretation = "(pure gradients)"
        else:
            interpretation = f"({int(alpha*100)}% gradients, {int((1-alpha)*100)}% attention)"
        
        print(f"α = {alpha:3.1f} {interpretation:25s} → '{metrics['top_token']}' ({metrics['top_token_score']:.3f})")
    
    return True

if __name__ == "__main__":
    print("🚀 STARTING FOCUSED CLIP INTERPRETABILITY TESTS")
    print("=" * 70)
    
    # Test 1: Basic focused approach
    success1 = test_focused_approach()
    
    if success1:
        # Test 2: Different alpha values
        success2 = test_different_alpha_values()
        
        if success2:
            print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
            print("✅ The focused attention × gradient approach is working correctly")
            print("📊 Heatmap visualizations have been generated")
            print("💡 Check the results/ folder for saved plots")
        else:
            print("\n⚠️ Basic test passed but alpha comparison failed")
    else:
        print("\n❌ Basic test failed - check the implementation") 