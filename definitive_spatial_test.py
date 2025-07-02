"""
Definitive Spatial Correspondence Test

This creates a grid image where each patch shows its expected index number,
allowing us to verify the exact spatial correspondence.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from clip.interpretable_clip import load_interpretable_clip, tokenize_text

def create_numbered_grid_image(size=224, grid_size=14):
    """Create an image with each patch showing its expected index number"""
    image = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(image)
    
    patch_size = size // grid_size
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", size=8)
    except:
        font = ImageFont.load_default()
    
    index = 0
    for row in range(grid_size):
        for col in range(grid_size):
            # Calculate patch boundaries
            x1 = col * patch_size
            y1 = row * patch_size
            x2 = x1 + patch_size
            y2 = y1 + patch_size
            
            # Alternate colors for better visibility
            if (row + col) % 2 == 0:
                color = (240, 240, 240)  # Light gray
                text_color = (0, 0, 0)   # Black text
            else:
                color = (200, 200, 200)  # Darker gray
                text_color = (0, 0, 0)   # Black text
            
            # Fill the patch
            draw.rectangle([x1, y1, x2, y2], fill=color)
            
            # Draw border
            draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 0), width=1)
            
            # Add index number
            text = str(index)
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            text_x = x1 + (patch_size - text_width) // 2
            text_y = y1 + (patch_size - text_height) // 2
            
            draw.text((text_x, text_y), text, fill=text_color, font=font)
            
            index += 1
    
    return image

def test_specific_patches():
    """Test specific patches by their index numbers"""
    print("🎯 Testing Specific Patch Correspondence")
    print("="*50)
    
    # Load model
    model = load_interpretable_clip("ViT-B/16", device="cpu")
    grid_size = 14
    
    # Create numbered grid image
    grid_image = create_numbered_grid_image(224, grid_size)
    grid_image.save("numbered_grid_test.png")
    print("💾 Saved numbered grid: numbered_grid_test.png")
    
    # Process image
    image_input = model.preprocess(grid_image).unsqueeze(0)
    
    # Test specific patch numbers
    test_patches = [0, 13, 182, 195, 98]  # corners + center
    expected_positions = [
        (0, 0),    # Top-left
        (0, 13),   # Top-right  
        (13, 0),   # Bottom-left
        (13, 13),  # Bottom-right
        (7, 0)     # Middle-left (98 = 7*14 + 0)
    ]
    
    for i, patch_idx in enumerate(test_patches):
        expected_row, expected_col = expected_positions[i]
        
        print(f"\n--- Testing patch {patch_idx} (expected at row {expected_row}, col {expected_col}) ---")
        
        # Create text prompt for this patch number
        text_prompt = f"patch number {patch_idx}"
        text_input = tokenize_text(text_prompt)
        
        tokens, similarity = model.get_token_patch_similarity(image_input, text_input)
        
        # Find the number token
        number_token_idx = None
        for j, token in enumerate(tokens):
            if str(patch_idx) in token:
                number_token_idx = j
                break
        
        if number_token_idx is None:
            print(f"  Number token '{patch_idx}' not found in: {tokens}")
            continue
        
        # Get similarity for the number token
        token_similarities = similarity[number_token_idx, :].detach().cpu().numpy()
        
        # Find patch with highest similarity
        max_patch_idx = np.argmax(token_similarities)
        max_similarity = token_similarities[max_patch_idx]
        
        # Convert to 2D coordinates
        actual_row = max_patch_idx // grid_size
        actual_col = max_patch_idx % grid_size
        
        print(f"  Token: '{tokens[number_token_idx]}'")
        print(f"  Expected patch index: {patch_idx} → ({expected_row}, {expected_col})")
        print(f"  Actual max similarity: index {max_patch_idx} → ({actual_row}, {actual_col})")
        print(f"  Similarity value: {max_similarity:.4f}")
        
        # Check if correspondence is correct
        if max_patch_idx == patch_idx:
            print(f"  ✅ PERFECT MATCH!")
        else:
            print(f"  ❌ MISMATCH: Expected {patch_idx}, got {max_patch_idx}")
            
            # Check if the expected patch has high similarity too
            expected_similarity = token_similarities[patch_idx]
            print(f"  Expected patch similarity: {expected_similarity:.4f}")
            
            # Show top 3 patches
            top_3_indices = np.argsort(token_similarities)[-3:][::-1]
            print(f"  Top 3 patches:")
            for k, idx in enumerate(top_3_indices):
                row = idx // grid_size
                col = idx % grid_size
                sim = token_similarities[idx]
                print(f"    {k+1}. Index {idx} → ({row}, {col}) = {sim:.4f}")

def analyze_flip_necessity():
    """Determine if horizontal flip is necessary"""
    print(f"\n🔍 Analyzing Flip Necessity")
    print("="*40)
    
    # Create a simple test: single bright patch in known location
    test_image = Image.new('RGB', (224, 224), color=(128, 128, 128))  # Gray background
    draw = ImageDraw.Draw(test_image)
    
    # Put a bright white patch in a specific location
    # Let's put it at patch (5, 5) which should be 1D index 5*14+5 = 75
    patch_size = 16
    x1 = 5 * patch_size
    y1 = 5 * patch_size
    x2 = x1 + patch_size  
    y2 = y1 + patch_size
    
    draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))  # Bright white
    test_image.save("single_patch_test.png")
    print("💾 Saved single patch test: single_patch_test.png")
    
    # Test with model
    model = load_interpretable_clip("ViT-B/16", device="cpu")
    image_input = model.preprocess(test_image).unsqueeze(0)
    text_input = tokenize_text("white patch")
    
    tokens, similarity = model.get_token_patch_similarity(image_input, text_input)
    
    # Find white token
    white_token_idx = None
    for i, token in enumerate(tokens):
        if 'white' in token.lower():
            white_token_idx = i
            break
    
    if white_token_idx is not None:
        token_similarities = similarity[white_token_idx, :].detach().cpu().numpy()
        max_patch_idx = np.argmax(token_similarities)
        
        expected_idx = 5 * 14 + 5  # Should be 75
        actual_row = max_patch_idx // 14
        actual_col = max_patch_idx % 14
        
        print(f"White patch placed at: patch (5, 5) = index {expected_idx}")
        print(f"Model found max at: index {max_patch_idx} = patch ({actual_row}, {actual_col})")
        
        if max_patch_idx == expected_idx:
            print("✅ NO FLIP NEEDED - Direct correspondence works!")
        else:
            # Check if flipped coordinates work
            flipped_col = 14 - 1 - 5  # 13 - 5 = 8
            flipped_expected = 5 * 14 + flipped_col  # 5*14 + 8 = 78
            
            if max_patch_idx == flipped_expected:
                print("🔄 FLIP NEEDED - Horizontal flip required for correct correspondence")
            else:
                print("❓ COMPLEX ISSUE - Neither direct nor flipped correspondence works")
                
        # Show similarity map
        sim_2d = token_similarities.reshape(14, 14)
        flipped_2d = np.fliplr(sim_2d)
        
        print(f"\nSimilarity analysis:")
        print(f"  Max similarity (original): {np.max(sim_2d):.4f} at {np.unravel_index(np.argmax(sim_2d), sim_2d.shape)}")
        print(f"  Max similarity (flipped): {np.max(flipped_2d):.4f} at {np.unravel_index(np.argmax(flipped_2d), flipped_2d.shape)}")

if __name__ == "__main__":
    print("🧪 Definitive Spatial Correspondence Test")
    print("="*60)
    
    # Test with numbered grid
    test_specific_patches()
    
    # Analyze flip necessity  
    analyze_flip_necessity()
    
    print(f"\n🎯 Definitive Test Complete!")
    print(f"This should reveal the exact spatial correspondence pattern.") 