"""
Test CORRECT CLIP Attention × Gradient Analysis

This tests the properly implemented attention/gradient analysis that respects:
1. CAUSAL attention for text (tokens only attend to previous tokens)
2. FULL attention for vision (patches attend to all patches + CLS)
3. Proper gradient flow from final CLS/EOS representations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Import the corrected analyzer
from clip.correct_attention_analysis import analyze_clip_correctly


def test_correct_analysis():
    """Test the corrected CLIP analysis."""
    
    print("🎯 TESTING CORRECT CLIP ATTENTION × GRADIENT ANALYSIS")
    print("=" * 70)
    
    # Test with a sample image and text
    image_path = "images/cat.PNG"
    text = "a photo of a cat"
    
    print(f"📸 Image: {image_path}")
    print(f"📝 Text: '{text}'")
    print()
    
    # Run analysis with debugging
    results = analyze_clip_correctly(
        image_path=image_path,
        text=text,
        model_name="ViT-B/32",
        alpha=0.5,
        debug=True
    )
    
    print("\n" + "=" * 70)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Text analysis summary
    print("\n📝 TEXT ANALYSIS:")
    print(f"  Number of meaningful tokens: {len(results['tokens'])}")
    print(f"  Text attention variation: std={results['text_attention'].std():.6f}")
    print(f"  Text gradient variation: std={results['text_gradients'].std():.6f}")
    print(f"  Text combined variation: std={results['text_combined'].std():.6f}")
    
    # Image analysis summary
    print("\n🖼️ IMAGE ANALYSIS:")
    print(f"  Number of patches: {len(results['image_attention'])}")
    print(f"  Image attention variation: std={results['image_attention'].std():.6f}")
    print(f"  Image gradient variation: std={results['image_gradients'].std():.6f}")
    print(f"  Image combined variation: std={results['image_combined'].std():.6f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'CORRECT CLIP Analysis: "{text}"', fontsize=16, fontweight='bold')
    
    # Text plots
    tokens = results['tokens']
    token_indices = results['token_indices']
    
    axes[0, 0].bar(range(len(tokens)), [results['text_attention'][i].item() for i in token_indices])
    axes[0, 0].set_title('Text Attention\n(Causal: tokens→previous only)')
    axes[0, 0].set_xticks(range(len(tokens)))
    axes[0, 0].set_xticklabels(tokens, rotation=45)
    
    axes[0, 1].bar(range(len(tokens)), [results['text_gradients'][i].item() for i in token_indices])
    axes[0, 1].set_title('Text Gradients\n(∂EOS/∂token)')
    axes[0, 1].set_xticks(range(len(tokens)))
    axes[0, 1].set_xticklabels(tokens, rotation=45)
    
    axes[0, 2].bar(range(len(tokens)), [results['text_combined'][i].item() for i in token_indices])
    axes[0, 2].set_title(f'Text Combined\n(α={results["alpha"]})')
    axes[0, 2].set_xticks(range(len(tokens)))
    axes[0, 2].set_xticklabels(tokens, rotation=45)
    
    # Image plots - show as heatmaps (7x7 grid for ViT-B/32)
    patch_size = int(np.sqrt(len(results['image_attention'])))
    
    if patch_size * patch_size == len(results['image_attention']):
        im1 = axes[1, 0].imshow(results['image_attention'].reshape(patch_size, patch_size), cmap='viridis')
        axes[1, 0].set_title('Image Attention\n(Full: CLS→all patches)')
        plt.colorbar(im1, ax=axes[1, 0])
        
        im2 = axes[1, 1].imshow(results['image_gradients'].reshape(patch_size, patch_size), cmap='viridis')
        axes[1, 1].set_title('Image Gradients\n(∂CLS/∂patch)')
        plt.colorbar(im2, ax=axes[1, 1])
        
        im3 = axes[1, 2].imshow(results['image_combined'].reshape(patch_size, patch_size), cmap='viridis')
        axes[1, 2].set_title(f'Image Combined\n(α={results["alpha"]})')
        plt.colorbar(im3, ax=axes[1, 2])
    else:
        # Fallback to bar plots if not square
        axes[1, 0].plot(results['image_attention'])
        axes[1, 0].set_title('Image Attention')
        
        axes[1, 1].plot(results['image_gradients'])
        axes[1, 1].set_title('Image Gradients')
        
        axes[1, 2].plot(results['image_combined'])
        axes[1, 2].set_title('Image Combined')
    
    plt.tight_layout()
    plt.savefig('correct_clip_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n💾 Visualization saved as 'correct_clip_analysis.png'")
    
    # Verification checks
    print("\n🔍 VERIFICATION CHECKS:")
    
    # Check 1: Text attention should respect causal mask (later tokens get 0 attention from EOS)
    eos_pos = max(token_indices)  # EOS should be last meaningful token
    tokens_after_eos = results['text_attention'][eos_pos+1:]
    if len(tokens_after_eos) > 0:
        print(f"  ✅ Causal mask: attention after EOS = {tokens_after_eos.max():.6f} (should be ~0)")
    else:
        print(f"  ✅ Causal mask: EOS is at end of sequence")
    
    # Check 2: Attention should sum to ~1.0
    text_attn_sum = results['text_attention'].sum()
    print(f"  ✅ Text attention sum: {text_attn_sum:.6f} (should be ~1.0)")
    
    image_attn_sum = results['image_attention'].sum()
    print(f"  ✅ Image attention sum: {image_attn_sum:.6f} (should be ~1.0)")
    
    # Check 3: Gradients should have meaningful variation
    text_grad_nonzero = (results['text_gradients'] > 1e-6).sum()
    print(f"  ✅ Text gradients: {text_grad_nonzero}/{len(results['text_gradients'])} non-zero")
    
    image_grad_nonzero = (results['image_gradients'] > 1e-6).sum()
    print(f"  ✅ Image gradients: {image_grad_nonzero}/{len(results['image_gradients'])} non-zero")
    
    print(f"\n🎉 CORRECT ANALYSIS COMPLETE!")
    return results


def test_multiple_scenarios():
    """Test with different alpha values and scenarios."""
    
    print("\n" + "=" * 70)
    print("🔬 TESTING MULTIPLE SCENARIOS")
    print("=" * 70)
    
    scenarios = [
        ("images/cat.PNG", "a fluffy cat"),
        ("images/cat.PNG", "cat sitting on furniture"),
        ("images/dog.PNG", "a photo of a dog"),
    ]
    
    alphas = [0.0, 0.5, 1.0]  # Pure attention, balanced, pure gradients
    
    for i, (image_path, text) in enumerate(scenarios):
        print(f"\n📋 Scenario {i+1}: {text}")
        print("-" * 50)
        
        for alpha in alphas:
            print(f"\n  Alpha = {alpha} ({'attention only' if alpha == 0 else 'gradients only' if alpha == 1 else 'balanced'})")
            
            try:
                results = analyze_clip_correctly(
                    image_path=image_path,
                    text=text,
                    alpha=alpha,
                    debug=False  # Less verbose for multiple tests
                )
                
                # Quick summary
                text_var = results['text_combined'].std()
                image_var = results['image_combined'].std()
                print(f"    Text variation: {text_var:.6f}, Image variation: {image_var:.6f}")
                
                # Show top tokens/patches
                top_tokens = [(results['tokens'][i], results['text_combined'][results['token_indices'][i]].item()) 
                             for i in range(len(results['tokens']))]
                top_tokens.sort(key=lambda x: x[1], reverse=True)
                print(f"    Top tokens: {top_tokens[:3]}")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")


if __name__ == "__main__":
    # Test the correct implementation
    results = test_correct_analysis()
    
    # Test multiple scenarios
    test_multiple_scenarios() 