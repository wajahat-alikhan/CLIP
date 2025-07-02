"""
Debug Coordinate System for CLIP Spatial Mapping

Simple test to understand how coordinates map between 1D and 2D
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from clip.interpretable_clip import load_interpretable_clip, tokenize_text

def create_simple_test_image(size=224):
    """Create a very simple test image with clear features"""
    image = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(image)
    
    # Draw a small red square in the top-left corner (first few patches)
    draw.rectangle([0, 0, 32, 32], fill='red')
    
    # Draw a small blue square in the top-right corner
    draw.rectangle([size-32, 0, size, 32], fill='blue')
    
    # Draw a small green square in the bottom-left corner  
    draw.rectangle([0, size-32, 32, size], fill='green')
    
    # Draw a small yellow square in the bottom-right corner
    draw.rectangle([size-32, size-32, size, size], fill='yellow')
    
    return image

def test_coordinate_mapping():
    """Test the basic coordinate mapping"""
    print("🔍 Testing Basic Coordinate Mapping")
    print("="*50)
    
    grid_size = 14  # ViT-B/16
    total_patches = grid_size * grid_size
    
    # Create a test pattern: each patch gets its 1D index as value
    test_pattern_1d = np.arange(total_patches)
    print(f"1D pattern (first 10): {test_pattern_1d[:10]}")
    print(f"1D pattern (last 10): {test_pattern_1d[-10:]}")
    
    # Reshape to 2D (this is what we do in the saliency generator)
    test_pattern_2d = test_pattern_1d.reshape(grid_size, grid_size)
    
    print(f"\n2D pattern (top-left 3x3):")
    print(test_pattern_2d[:3, :3])
    
    print(f"\n2D pattern (top-right 3x3):")
    print(test_pattern_2d[:3, -3:])
    
    print(f"\n2D pattern (bottom-left 3x3):")
    print(test_pattern_2d[-3:, :3])
    
    print(f"\n2D pattern (bottom-right 3x3):")
    print(test_pattern_2d[-3:, -3:])
    
    # Key coordinates
    print(f"\nKey coordinates:")
    print(f"  Top-left (0,0): value = {test_pattern_2d[0,0]}")
    print(f"  Top-right (0,{grid_size-1}): value = {test_pattern_2d[0,grid_size-1]}")
    print(f"  Bottom-left ({grid_size-1},0): value = {test_pattern_2d[grid_size-1,0]}")
    print(f"  Bottom-right ({grid_size-1},{grid_size-1}): value = {test_pattern_2d[grid_size-1,grid_size-1]}")
    
    # Test with horizontal flip (current implementation)
    flipped_2d = np.fliplr(test_pattern_2d)
    
    print(f"\nAfter horizontal flip:")
    print(f"  Top-left (0,0): value = {flipped_2d[0,0]}")
    print(f"  Top-right (0,{grid_size-1}): value = {flipped_2d[0,grid_size-1]}")
    print(f"  Bottom-left ({grid_size-1},0): value = {flipped_2d[grid_size-1,0]}")
    print(f"  Bottom-right ({grid_size-1},{grid_size-1}): value = {flipped_2d[grid_size-1,grid_size-1]}")

def test_with_actual_model():
    """Test with actual CLIP model using simple image"""
    print(f"\n🎯 Testing with Actual CLIP Model")
    print("="*50)
    
    # Load model
    model = load_interpretable_clip("ViT-B/16", device="cpu")
    
    # Create simple test image
    test_image = create_simple_test_image(224)
    test_image.save("debug_test_image.png")
    print("💾 Saved debug test image: debug_test_image.png")
    
    # Test with different prompts
    test_prompts = [
        "red color",
        "blue color", 
        "green color",
        "yellow color"
    ]
    
    # Process image
    image_input = model.preprocess(test_image).unsqueeze(0)
    
    for prompt in test_prompts:
        print(f"\n--- Testing: '{prompt}' ---")
        
        text_input = tokenize_text(prompt)
        tokens, similarity = model.get_token_patch_similarity(image_input, text_input)
        
        # Find the color token
        color_token_idx = None
        target_color = prompt.split()[0]  # "red", "blue", etc.
        
        for i, token in enumerate(tokens):
            if target_color in token.lower():
                color_token_idx = i
                break
        
        if color_token_idx is None:
            print(f"  Color token '{target_color}' not found in: {tokens}")
            continue
        
        # Get similarity for the color token
        token_similarities = similarity[color_token_idx, :].detach().cpu().numpy()
        
        # Find top 5 patches with highest similarity
        top_indices = np.argsort(token_similarities)[-5:][::-1]
        
        print(f"  Token: '{tokens[color_token_idx]}'")
        print(f"  Top 5 patches (1D index, 2D coords, similarity):")
        
        grid_size = 14
        for idx in top_indices:
            row = idx // grid_size
            col = idx % grid_size
            sim_val = token_similarities[idx]
            print(f"    Index {idx:3d} → ({row:2d},{col:2d}) = {sim_val:.4f}")
        
        # Also show what would happen with horizontal flip
        reshaped = token_similarities.reshape(grid_size, grid_size)
        flipped = np.fliplr(reshaped)
        
        max_pos_original = np.unravel_index(np.argmax(reshaped), reshaped.shape)
        max_pos_flipped = np.unravel_index(np.argmax(flipped), flipped.shape)
        
        print(f"  Max position (original): {max_pos_original}")
        print(f"  Max position (flipped): {max_pos_flipped}")

def analyze_patch_layout():
    """Analyze how patches are laid out"""
    print(f"\n📐 Analyzing Patch Layout")
    print("="*40)
    
    image_size = 224
    grid_size = 14
    patch_size = image_size // grid_size
    
    print(f"Image size: {image_size}x{image_size}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Patch size: {patch_size}x{patch_size}")
    
    print(f"\nPatch coordinates:")
    print(f"  Top-left patch covers: (0, 0) to ({patch_size}, {patch_size})")
    print(f"  Top-right patch covers: ({image_size-patch_size}, 0) to ({image_size}, {patch_size})")
    print(f"  Bottom-left patch covers: (0, {image_size-patch_size}) to ({patch_size}, {image_size})")
    print(f"  Bottom-right patch covers: ({image_size-patch_size}, {image_size-patch_size}) to ({image_size}, {image_size})")
    
    # Our test squares
    print(f"\nOur test squares:")
    print(f"  Red square: (0, 0) to (32, 32) → Should be in top-left patches")
    print(f"  Blue square: ({image_size-32}, 0) to ({image_size}, 32) → Should be in top-right patches")
    print(f"  Green square: (0, {image_size-32}) to (32, {image_size}) → Should be in bottom-left patches") 
    print(f"  Yellow square: ({image_size-32}, {image_size-32}) to ({image_size}, {image_size}) → Should be in bottom-right patches")

if __name__ == "__main__":
    print("🐛 Debugging CLIP Coordinate System")
    print("="*60)
    
    # Test basic coordinate mapping
    test_coordinate_mapping()
    
    # Analyze patch layout
    analyze_patch_layout()
    
    # Test with actual model
    test_with_actual_model()
    
    print(f"\n🎯 Debug Complete!")
    print(f"Check 'debug_test_image.png' to see the test image with colored squares.") 