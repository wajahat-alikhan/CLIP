"""
Test the coordinate system fix to resolve the similarity inversion issue.
"""

import torch
from PIL import Image
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clip.interpretable_clip_attention import load_interpretable_clip, tokenize

def test_coordinate_fix():
    """Test if coordinate system fixes resolve the inversion issue."""
    print("🔧 TESTING COORDINATE SYSTEM FIX")
    print("="*50)
    
    model = load_interpretable_clip("ViT-L/14", device="cpu")
    preprocess = model.preprocess
    
    # Load image and text
    image = preprocess(Image.open("D:\Wajahat Ali Khan\CLIP\dogcat2.PNG")).unsqueeze(0)
    cat_text = tokenize(["cat"]).to("cpu")
    dog_text = tokenize(["dog"]).to("cpu")
    
    test_tokens = ["cat", "dog"]
    
    for token in test_tokens:
        print(f"\n--- Testing '{token}' token ---")
        text_input = tokenize([token]).to("cpu")
        
        with torch.no_grad():
            tokens, similarity = model.get_token_patch_similarity(image, text_input, debug=False)
            
            if token in tokens:
                token_idx = tokens.index(token)
                sims = similarity[token_idx, :].numpy()
                grid_size = int(np.sqrt(sims.shape[0]))
                
                # Test different coordinate systems
                methods = {
                    "Original (Row-major)": sims.reshape(grid_size, grid_size),
                    "Column-major": sims.reshape(grid_size, grid_size, order='F'),
                    "Transposed": sims.reshape(grid_size, grid_size).T,
                    "Horizontal Flip": np.fliplr(sims.reshape(grid_size, grid_size))
                }
                
                print(f"Testing coordinate system fixes for '{token}':")
                
                expected_side = "RIGHT" if token == "cat" else "LEFT"
                
                for method_name, spatial_grid in methods.items():
                    # Find peak
                    max_idx = np.argmax(spatial_grid)
                    max_row, max_col = np.unravel_index(max_idx, spatial_grid.shape)
                    
                    # Determine side
                    if max_col < grid_size // 2:
                        peak_side = "LEFT"
                    else:
                        peak_side = "RIGHT"
                    
                    # Check if correct
                    is_correct = peak_side == expected_side
                    status = "✅" if is_correct else "❌"
                    
                    print(f"  {method_name:20s}: Peak at ({max_row}, {max_col}) = {peak_side} {status}")
                    
                    if is_correct:
                        print(f"    🎉 {method_name} FIXES THE ISSUE!")
                        
                        # Create visualization with this fix
                        create_fixed_visualization(image.squeeze(0), spatial_grid, token, method_name)

def create_fixed_visualization(image_tensor, spatial_grid, token, method_name):
    """Create visualization with the fixed coordinate system."""
    
    # Convert tensor back to PIL Image for visualization
    # The image tensor is normalized, so we need to denormalize
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    
    # Denormalize
    image_denorm = image_tensor * std + mean
    image_denorm = torch.clamp(image_denorm, 0, 1)
    
    # Convert to PIL
    image_pil = Image.fromarray((image_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_pil)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Similarity heatmap
    im1 = axes[1].imshow(spatial_grid, cmap='viridis', interpolation='nearest')
    axes[1].set_title(f"Fixed Similarities - '{token}' token\n({method_name})")
    axes[1].set_xlabel("Patch Column")
    axes[1].set_ylabel("Patch Row")
    
    # Add peak marker
    max_idx = np.argmax(spatial_grid)
    max_row, max_col = np.unravel_index(max_idx, spatial_grid.shape)
    axes[1].scatter(max_col, max_row, color='red', s=100, marker='x', linewidth=3)
    
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay on original image
    import cv2
    img_width, img_height = image_pil.size
    heatmap_resized = cv2.resize(spatial_grid, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
    
    axes[2].imshow(image_pil)
    im2 = axes[2].imshow(heatmap_norm, cmap='plasma', alpha=0.6, extent=[0, img_width, img_height, 0])
    axes[2].set_title(f"Fixed Overlay - '{token}' token")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(f"FIXED_{method_name.replace(' ', '_')}_{token}_similarities.png", dpi=150, bbox_inches='tight')
    plt.show()

def main():
    print("🔧 COORDINATE SYSTEM FIX TEST")
    print("Testing if coordinate transforms resolve the inversion issue")
    print("="*60)
    
    test_coordinate_fix()
    
    print(f"\n{'='*60}")
    print("🎯 CONCLUSION:")
    print("If any method shows correct peak locations, that's our fix!")
    print("We should implement that coordinate system in the main code.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 