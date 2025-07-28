#!/usr/bin/env python3
"""
Attribution-Weighted CLIP Interpretability Example

This script demonstrates the NEW attribution-weighted approach to CLIP interpretability
that respects the training objective by weighting embeddings based on their actual 
contribution to final CLS/EOS representations.

Key Features:
- Attention + Gradient attribution analysis
- Comparison with original approach  
- Real image and text examples
- Clear visualizations
- Multiple test scenarios

Author: Based on our attribution-weighted interpretability method
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import sys
import os
import json

# Add the parent directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from clip.interpretable_clip_attention import load_interpretable_clip, tokenize_text

def create_attribution_heatmap_overlay(img, attribution_scores, grid_size=7):
    """Create a heatmap overlay showing attribution scores on the image."""
    # Reshape attribution scores to grid
    attribution_grid = attribution_scores.reshape(grid_size, grid_size)
    
    # Normalize to [0, 1]
    attr_norm = (attribution_grid - attribution_grid.min()) / (attribution_grid.max() - attribution_grid.min() + 1e-8)
    
    # Upsample to image size (224x224)
    patch_size = 224 // grid_size
    attr_upsampled = np.kron(attr_norm, np.ones((patch_size, patch_size)))
    
    # Create colormap overlay
    import matplotlib.cm as cm
    colormap = cm.get_cmap('coolwarm')  # Blue=Low, Red=High
    heatmap = colormap(attr_upsampled)[:, :, :3]  # Remove alpha
    
    # Blend with original image
    overlay = 0.6 * heatmap + 0.4 * np.float32(img)
    overlay = overlay / np.max(overlay)
    
    return overlay, attribution_grid

def run_attribution_analysis(image_path, text_description, alpha=0.5, debug=True):
    """
    Run comprehensive attribution-weighted analysis on image and text.
    
    Args:
        image_path: Path to image file
        text_description: Text description to analyze
        alpha: Balance between gradients (1.0) and attention (0.0)
        debug: Whether to print detailed debug information
        
    Returns:
        Dictionary with all results
    """
    print(f"\n🔍 Attribution Analysis: '{text_description}' on {os.path.basename(image_path)}")
    print("=" * 80)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Device: {device}")
    
    # Load model
    print("📥 Loading attribution-weighted CLIP model...")
    model = load_interpretable_clip("ViT-B/32", device=device)
    
    # Load and preprocess image
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
        
    print(f"🖼️  Loading image: {image_path}")
    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = model.preprocess(image_pil).unsqueeze(0).to(device)
    text_tensor = tokenize_text(text_description).to(device)
    
    # Convert image to numpy for visualization
    image_np = image_tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    
    print(f"✓ Image shape: {image_tensor.shape}")
    print(f"✓ Text: '{text_description}'")
    
    # PART 1: Original approach (baseline)
    print("\n" + "="*50)
    print("📊 PART 1: ORIGINAL APPROACH (BASELINE)")
    print("="*50)
    
    try:
        tokens_orig, token_patch_sim, eos_patch_sim, cls_token_sim, eos_token_sim, cls_patch_sim = model.get_token_patch_similarity(
            image_tensor, text_tensor, debug=debug
        )
        
        print(f"✅ Original approach completed!")
        print(f"  📋 Tokens: {tokens_orig}")
        print(f"  📊 Token-patch similarities shape: {token_patch_sim.shape}")
        print(f"  📈 Similarity range: [{token_patch_sim.min():.4f}, {token_patch_sim.max():.4f}]")
        
        original_results = {
            'tokens': tokens_orig,
            'token_patch_similarity': token_patch_sim,
            'eos_patch_similarity': eos_patch_sim,
            'cls_token_similarity': cls_token_sim,
            'eos_token_similarity': eos_token_sim,
            'cls_patch_similarity': cls_patch_sim
        }
        
    except Exception as e:
        print(f"❌ Original approach failed: {e}")
        original_results = None
    
    # PART 2: Attribution-weighted approach (NEW)
    print("\n" + "="*50)
    print("🚀 PART 2: ATTRIBUTION-WEIGHTED APPROACH (NEW)")
    print("="*50)
    
    try:
        tokens_attr, attributed_similarity, debug_info = model.get_attributed_token_patch_similarity(
            image_tensor, text_tensor, alpha=alpha, debug=debug
        )
        
        print(f"\n✅ Attribution-weighted approach completed!")
        print(f"  📋 Tokens: {tokens_attr}")
        print(f"  📊 Attribution-weighted similarities shape: {attributed_similarity.shape}")
        print(f"  📈 Similarity range: [{attributed_similarity.min():.4f}, {attributed_similarity.max():.4f}]")
        
        # Extract attribution scores
        text_attribution = debug_info['text_attribution']
        image_attribution = debug_info['image_attribution']
        
        print(f"\n🔍 Attribution Analysis:")
        print(f"  📊 Text attribution range: [{text_attribution.min():.4f}, {text_attribution.max():.4f}]")
        print(f"  📊 Image attribution range: [{image_attribution.min():.4f}, {image_attribution.max():.4f}]")
        
        # Show top contributing elements
        real_token_indices = debug_info['real_token_indices']
        text_attr_real = text_attribution[real_token_indices]
        
        print(f"\n🏆 Top Contributing Elements:")
        print(f"  📝 Text tokens (by attribution):")
        top_token_indices = torch.topk(text_attr_real, min(len(tokens_attr), 3)).indices
        for i, idx in enumerate(top_token_indices):
            token = tokens_attr[idx]
            score = text_attr_real[idx].item()
            print(f"    {i+1}. '{token}': {score:.4f}")
        
        print(f"  🖼️  Image patches (by attribution):")
        top_patch_indices = torch.topk(image_attribution, min(5, len(image_attribution))).indices
        grid_size = int(np.sqrt(len(image_attribution)))
        for i, idx in enumerate(top_patch_indices):
            row, col = idx // grid_size, idx % grid_size
            score = image_attribution[idx].item()
            print(f"    {i+1}. Patch ({row}, {col}): {score:.4f}")
        
        attribution_results = {
            'tokens': tokens_attr,
            'attributed_similarity': attributed_similarity,
            'text_attribution': text_attribution,
            'image_attribution': image_attribution,
            'debug_info': debug_info,
            'alpha': alpha
        }
        
    except Exception as e:
        print(f"❌ Attribution-weighted approach failed: {e}")
        import traceback
        traceback.print_exc()
        attribution_results = None
    
    # PART 3: Visualization and Comparison
    print("\n" + "="*50)
    print("🎨 PART 3: VISUALIZATION AND COMPARISON")
    print("="*50)
    
    if original_results and attribution_results:
        create_comparison_visualization(
            image_np, tokens_attr, 
            original_results, attribution_results,
            text_description, image_path
        )
    
    # Compile final results
    results = {
        'image_path': image_path,
        'text_description': text_description,
        'alpha': alpha,
        'original_results': original_results,
        'attribution_results': attribution_results,
        'device': device
    }
    
    return results

def create_comparison_visualization(image_np, tokens, original_results, attribution_results, text_description, image_path):
    """Create comprehensive comparison visualization."""
    
    print("🎨 Creating comparison visualization...")
    
    # Extract data
    token_patch_sim_orig = original_results['token_patch_similarity']
    attributed_similarity = attribution_results['attributed_similarity']
    image_attribution = attribution_results['image_attribution']
    text_attribution = attribution_results['text_attribution']
    real_token_indices = attribution_results['debug_info']['real_token_indices']
    
    num_tokens = len(tokens)
    grid_size = 7  # ViT-B/32
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    
    # Title
    fig.suptitle(f"Attribution-Weighted CLIP Analysis: '{text_description}'\nImage: {os.path.basename(image_path)}", 
                 fontsize=16, fontweight='bold')
    
    # Layout: 3 rows x (num_tokens + 2) columns
    rows, cols = 3, num_tokens + 2
    
    # Row 1: Original approach token-patch similarities
    # Original image
    ax = plt.subplot(rows, cols, 1)
    ax.imshow(image_np)
    ax.set_title("Original Image", fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Original token-patch similarities  
    for i, token in enumerate(tokens):
        ax = plt.subplot(rows, cols, i + 2)
        token_sim_grid = token_patch_sim_orig[i].detach().cpu().numpy().reshape(grid_size, grid_size)
        im = ax.imshow(token_sim_grid, cmap='coolwarm')
        ax.set_title(f"Original: '{token}'\nMax: {token_sim_grid.max():.3f}", fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Image attribution scores
    ax = plt.subplot(rows, cols, num_tokens + 2)
    image_attr_grid = image_attribution.detach().cpu().numpy().reshape(grid_size, grid_size)
    im = ax.imshow(image_attr_grid, cmap='viridis')
    ax.set_title(f"Image Attribution\nMax: {image_attr_grid.max():.3f}", fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Row 2: Attribution-weighted token-patch similarities
    # Original image again
    ax = plt.subplot(rows, cols, cols + 1)
    ax.imshow(image_np)
    ax.set_title("Original Image", fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Attribution-weighted token-patch similarities
    for i, token in enumerate(tokens):
        ax = plt.subplot(rows, cols, cols + i + 2)
        attr_sim_grid = attributed_similarity[i].detach().cpu().numpy().reshape(grid_size, grid_size)
        im = ax.imshow(attr_sim_grid, cmap='coolwarm')
        
        # Get text attribution for this token
        token_attr = text_attribution[real_token_indices[i]].item()
        ax.set_title(f"Attribution: '{token}'\nText Attr: {token_attr:.3f}\nMax: {attr_sim_grid.max():.3f}", fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Text attribution bar chart
    ax = plt.subplot(rows, cols, cols + num_tokens + 2)
    text_attr_values = [text_attribution[real_token_indices[i]].item() for i in range(len(tokens))]
    bars = ax.bar(range(len(tokens)), text_attr_values, color='skyblue', alpha=0.7)
    ax.set_title("Text Attribution\nScores", fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_ylabel("Attribution Score")
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, text_attr_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Row 3: Overlays on original image
    # Original image
    ax = plt.subplot(rows, cols, 2*cols + 1)
    ax.imshow(image_np)
    ax.set_title("Original Image", fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Attribution-weighted overlays
    for i, token in enumerate(tokens):
        ax = plt.subplot(rows, cols, 2*cols + i + 2)
        attr_sim_scores = attributed_similarity[i].detach().cpu().numpy()
        overlay, _ = create_attribution_heatmap_overlay(image_np, attr_sim_scores, grid_size)
        ax.imshow(overlay)
        ax.set_title(f"'{token}' Overlay", fontsize=10)
        ax.axis('off')
    
    # Overall image attribution overlay
    ax = plt.subplot(rows, cols, 2*cols + num_tokens + 2)
    img_attr_scores = image_attribution.detach().cpu().numpy()
    overlay, _ = create_attribution_heatmap_overlay(image_np, img_attr_scores, grid_size)
    ax.imshow(overlay)
    ax.set_title("Image Attribution\nOverlay", fontsize=10)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    text_clean = text_description.replace(" ", "_").replace(",", "").replace(".", "")
    save_name = f"attribution_analysis_{base_name}_{text_clean}.png"
    save_path = os.path.join("results", save_name)
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved as: {save_path}")
    
    plt.show()

def test_multiple_scenarios():
    """Test the attribution method on multiple image-text combinations."""
    
    print("\n🧪 TESTING MULTIPLE SCENARIOS")
    print("=" * 80)
    
    # Define test scenarios
    scenarios = [
        {
            'image': 'D:\Wajahat Ali Khan\CLIP\images\dogcat2.png',
            'texts': ['a photo of a cat']
        }
        
    ]
    
    results_summary = []
    
    for scenario in scenarios:
        image_path = scenario['image']
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"⚠️  Skipping {image_path} - file not found")
            continue
            
        print(f"\n📸 Testing image: {os.path.basename(image_path)}")
        
        for text in scenario['texts']:
            print(f"\n📝 Text: '{text}'")
            
            # Run analysis with different alpha values
            for alpha in [0.0, 0.5, 1.0]:
                alpha_label = "attention-only" if alpha == 0.0 else "balanced" if alpha == 0.5 else "gradient-only"
                print(f"  🎛️  Alpha {alpha} ({alpha_label})")
                
                try:
                    results = run_attribution_analysis(image_path, text, alpha=alpha, debug=False)
                    
                    if results and results['attribution_results']:
                        attr_sim = results['attribution_results']['attributed_similarity']
                        mean_sim = attr_sim.mean().item()
                        max_sim = attr_sim.max().item()
                        
                        results_summary.append({
                            'image': os.path.basename(image_path),
                            'text': text,
                            'alpha': alpha,
                            'alpha_label': alpha_label,
                            'mean_similarity': mean_sim,
                            'max_similarity': max_sim
                        })
                        
                        print(f"    ✅ Mean similarity: {mean_sim:.4f}, Max: {max_sim:.4f}")
                    else:
                        print(f"    ❌ Analysis failed")
                        
                except Exception as e:
                    print(f"    ❌ Error: {e}")
    
    # Save results summary
    if results_summary:
        summary_path = os.path.join("results", "attribution_analysis_summary.json")
        os.makedirs("results", exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
            
        print(f"\n📊 Results summary saved to: {summary_path}")
        
        # Print summary statistics
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"  Total tests run: {len(results_summary)}")
        
        for alpha in [0.0, 0.5, 1.0]:
            alpha_results = [r for r in results_summary if r['alpha'] == alpha]
            if alpha_results:
                mean_similarities = [r['mean_similarity'] for r in alpha_results]
                avg_mean = np.mean(mean_similarities)
                alpha_label = "attention-only" if alpha == 0.0 else "balanced" if alpha == 0.5 else "gradient-only"
                print(f"  Alpha {alpha} ({alpha_label}): Avg mean similarity = {avg_mean:.4f}")

def main():
    """Main function to run attribution analysis examples."""
    
    print("🚀 Attribution-Weighted CLIP Interpretability Examples")
    print("=" * 80)
    print("This script demonstrates the NEW attribution-weighted approach")
    print("that respects CLIP's training objective by weighting embeddings")
    print("based on their actual contribution to CLS/EOS representations.")
    print("=" * 80)
    
    # Quick single example
    print("\n🎯 QUICK EXAMPLE")
    print("-" * 40)
    
    # Test with a cat image and simple text
    image_path = 'D:\Wajahat Ali Khan\CLIP\images\dogcat2.png'
    text = 'cat'
    
    if os.path.exists(image_path):
        results = run_attribution_analysis(image_path, text, alpha=0.5, debug=True)
        print(f"\n✅ Quick example completed!")
    else:
        print(f"⚠️  Cat image not found, trying alternative...")
        # Try alternative images
        alternatives = ['../images/dog.PNG', '../images/apple.png', '../images/bus.PNG']
        for alt_path in alternatives:
            if os.path.exists(alt_path):
                results = run_attribution_analysis(alt_path, 'object in image', alpha=0.5, debug=True)
                break
    
    # Ask user if they want to run comprehensive tests
    try:
        response = input("\n🤔 Would you like to run comprehensive tests on multiple scenarios? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            test_multiple_scenarios()
        else:
            print("\n✅ Analysis complete! Check the results/ folder for visualizations.")
    except (EOFError, KeyboardInterrupt):
        print("\n✅ Analysis complete! Check the results/ folder for visualizations.")

if __name__ == "__main__":
    main() 