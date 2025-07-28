"""
Verification Script: Spatial Mapping Correctness for CLIP Saliency

This script verifies that the spatial mapping from 1D similarity vectors 
to 2D spatial overlays is correct. It tests the entire pipeline step by step.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from clip.interpretable_clip_attention import load_interpretable_clip, tokenize_text

def create_test_image_with_markers(size=224, grid_size=14):
    """
    Create a test image with clear spatial markers to verify correspondence
    """
    image = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(image)
    
    # Calculate patch size
    patch_size = size // grid_size
    
    # Add colored markers in specific patches for verification
    test_patches = [
        (0, 0, 'red'),      # Top-left corner
        (0, grid_size-1, 'blue'),   # Top-right corner  
        (grid_size-1, 0, 'green'),  # Bottom-left corner
        (grid_size-1, grid_size-1, 'yellow'),  # Bottom-right corner
        (grid_size//2, grid_size//2, 'purple'), # Center
    ]
    
    colors = {
        'red': (255, 0, 0),
        'blue': (0, 0, 255), 
        'green': (0, 255, 0),
        'yellow': (255, 255, 0),
        'purple': (255, 0, 255)
    }
    
    for row, col, color_name in test_patches:
        x1 = col * patch_size
        y1 = row * patch_size
        x2 = x1 + patch_size
        y2 = y1 + patch_size
        
        # Fill the patch with color
        draw.rectangle([x1, y1, x2, y2], fill=colors[color_name])
        
        # Add text label
        try:
            draw.text((x1+5, y1+5), f"({row},{col})", fill='white')
        except:
            pass  # Skip text if font issues
    
    return image, test_patches

def verify_patch_indexing(grid_size=14):
    """
    Verify how patches are indexed in the 1D similarity vector
    """
    print("🔍 Verifying Patch Indexing")
    print("="*40)
    
    # Create mapping from 2D coordinates to 1D index
    index_map = {}
    flat_index = 0
    
    for row in range(grid_size):
        for col in range(grid_size):
            index_map[(row, col)] = flat_index
            flat_index += 1
    
    # Print some key mappings
    print(f"Grid size: {grid_size}×{grid_size} = {grid_size**2} patches")
    print(f"Key patch indices:")
    print(f"  Top-left (0,0): index {index_map[(0,0)]}")
    print(f"  Top-right (0,{grid_size-1}): index {index_map[(0,grid_size-1)]}")
    print(f"  Bottom-left ({grid_size-1},0): index {index_map[(grid_size-1,0)]}")
    print(f"  Bottom-right ({grid_size-1},{grid_size-1}): index {index_map[(grid_size-1,grid_size-1)]}")
    print(f"  Center ({grid_size//2},{grid_size//2}): index {index_map[(grid_size//2,grid_size//2)]}")
    
    return index_map

def test_reshape_operation(grid_size=14):
    """
    Test the reshape operation from 1D to 2D
    """
    print(f"\n🔍 Testing Reshape Operation")
    print("="*40)
    
    # Create a test 1D vector with known pattern
    test_vector = np.arange(grid_size**2)  # [0, 1, 2, ..., 195]
    
    print(f"Original 1D vector (first 10): {test_vector[:10]}")
    print(f"Original 1D vector (last 10): {test_vector[-10:]}")
    
    # Reshape to 2D
    reshaped_2d = test_vector.reshape(grid_size, grid_size)
    
    print(f"\nAfter reshape to {grid_size}×{grid_size}:")
    print(f"Top-left corner (first 3×3):")
    print(reshaped_2d[:3, :3])
    print(f"Bottom-right corner (last 3×3):")
    print(reshaped_2d[-3:, -3:])
    
    # Test with and without horizontal flip
    flipped_2d = np.fliplr(reshaped_2d)
    
    print(f"\nAfter horizontal flip:")
    print(f"Top-left corner (first 3×3):")
    print(flipped_2d[:3, :3])
    print(f"Bottom-right corner (last 3×3):")
    print(flipped_2d[-3:, -3:])
    
    return reshaped_2d, flipped_2d

def verify_spatial_correspondence():
    """
    Complete verification of spatial correspondence using interpretable CLIP
    """
    print(f"\n🎯 Verifying Complete Spatial Correspondence")
    print("="*60)
    
    # Load model
    model = load_interpretable_clip("ViT-B/16", device="cpu")
    grid_size = model.visual.input_resolution // model.visual.conv1.kernel_size[0]
    
    # Create test image with clear markers
    test_image, test_patches = create_test_image_with_markers(224, grid_size)
    
    print(f"Created test image with {len(test_patches)} colored markers")
    for row, col, color in test_patches:
        print(f"  {color} marker at patch ({row}, {col})")
    
    # Save test image for reference
    test_image.save("test_spatial_markers.png")
    print(f"💾 Saved test image: test_spatial_markers.png")
    
    # Test with different text prompts targeting specific colors
    test_prompts = [
        ("red patch", "red"),
        ("blue patch", "blue"), 
        ("green patch", "green"),
        ("yellow patch", "yellow"),
        ("purple patch", "purple")
    ]
    
    # Process image
    image_input = model.preprocess(test_image).unsqueeze(0)
    
    for prompt, target_color in test_prompts:
        print(f"\n--- Testing prompt: '{prompt}' (targeting {target_color}) ---")
        
        text_input = tokenize_text(prompt)
        tokens, similarity = model.get_token_patch_similarity(image_input, text_input)
        
        # Find the token that matches our target color
        target_token_idx = None
        for i, token in enumerate(tokens):
            if target_color in token.lower():
                target_token_idx = i
                break
        
        if target_token_idx is None:
            print(f"⚠️  '{target_color}' token not found in: {tokens}")
            continue
        
        # Get similarity for the target token
        token_similarities = similarity[target_token_idx, :].detach().cpu().numpy()
        
        # Find patch with highest similarity
        max_patch_idx = np.argmax(token_similarities)
        max_similarity = token_similarities[max_patch_idx]
        
        # Convert to 2D coordinates (without flip first)
        max_row = max_patch_idx // grid_size
        max_col = max_patch_idx % grid_size
        
        print(f"Token '{tokens[target_token_idx]}' has highest similarity at:")
        print(f"  1D index: {max_patch_idx}")
        print(f"  2D coordinates: ({max_row}, {max_col})")
        print(f"  Similarity value: {max_similarity:.4f}")
        
        # Check if this matches expected location
        expected_location = None
        for row, col, color in test_patches:
            if color == target_color:
                expected_location = (row, col)
                break
        
        if expected_location:
            expected_row, expected_col = expected_location
            if (max_row, max_col) == (expected_row, expected_col):
                print(f"  ✅ CORRECT: Matches expected location ({expected_row}, {expected_col})")
            else:
                print(f"  ❌ INCORRECT: Expected ({expected_row}, {expected_col}), got ({max_row}, {max_col})")
        
        # Now test with horizontal flip (current implementation)
        reshaped_similarities = token_similarities.reshape(grid_size, grid_size)
        flipped_similarities = np.fliplr(reshaped_similarities)
        
        # Find max in flipped version
        max_pos_flipped = np.unravel_index(np.argmax(flipped_similarities), flipped_similarities.shape)
        print(f"  With horizontal flip: max at {max_pos_flipped}")
        
    return test_image

def analyze_current_implementation():
    """
    Analyze the current saliency generation implementation
    """
    print(f"\n🔍 Analyzing Current Implementation")
    print("="*50)
    
    print("Current pipeline in create_cam_saliency():")
    print("1. similarity_vector (1D) → reshape(grid_size, grid_size)")
    print("2. Apply np.fliplr() for horizontal flip")  
    print("3. Resize to image dimensions with cv2.INTER_CUBIC")
    print("4. Apply Gaussian blur")
    print("5. Overlay on image")
    
    print(f"\n🤔 Question: Is the horizontal flip necessary/correct?")
    print("Possible reasons for flip:")
    print("- Coordinate system mismatch (image vs. array coordinates)")
    print("- Convention differences in patch ordering")
    print("- Correction for specific CLIP implementation details")
    
    print(f"\n💡 Recommendation:")
    print("Run the verification test to determine if flip is needed!")

if __name__ == "__main__":
    print("🧪 Spatial Correspondence Verification for CLIP Saliency")
    print("="*70)
    
    # Step 1: Verify patch indexing logic
    index_map = verify_patch_indexing(14)
    
    # Step 2: Test reshape operations  
    reshaped, flipped = test_reshape_operation(14)
    
    # Step 3: Test with actual model and marked image
    test_image = verify_spatial_correspondence()
    
    # Step 4: Analyze current implementation
    analyze_current_implementation()
    
    print(f"\n🎯 Verification Complete!")
    print(f"Check the console output to see if spatial correspondence is correct.")
    print(f"Look at 'test_spatial_markers.png' to see the test image with markers.") 