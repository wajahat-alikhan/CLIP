"""
Investigate CLIP Preprocessing Pipeline

This script examines step-by-step how CLIP processes images to understand
the spatial correspondence issue.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
from clip.interpretable_clip_attention import load_interpretable_clip
import clip

def create_corner_test_image(size=224):
    """Create an image with distinct features in each corner"""
    image = Image.new('RGB', (size, size), color='gray')
    draw = ImageDraw.Draw(image)
    
    # Make each corner a different color with large area
    corner_size = size // 4
    
    # Top-left: RED
    draw.rectangle([0, 0, corner_size, corner_size], fill='red')
    
    # Top-right: BLUE  
    draw.rectangle([size-corner_size, 0, size, corner_size], fill='blue')
    
    # Bottom-left: GREEN
    draw.rectangle([0, size-corner_size, corner_size, size], fill='green')
    
    # Bottom-right: YELLOW
    draw.rectangle([size-corner_size, size-corner_size, size, size], fill='yellow')
    
    # Add center marker
    center = size // 2
    marker_size = 20
    draw.rectangle([center-marker_size//2, center-marker_size//2, 
                   center+marker_size//2, center+marker_size//2], fill='white')
    
    return image

def analyze_clip_preprocessing():
    """Analyze what CLIP's preprocessing does to images"""
    print("🔍 Analyzing CLIP Preprocessing Pipeline")
    print("="*50)
    
    # Create test image
    test_image = create_corner_test_image(224)
    test_image.save("corner_test_original.png")
    print("💾 Saved original: corner_test_original.png")
    
    # Load CLIP model to get the preprocessing
    device = "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)
    
    print(f"\nCLIP preprocessing pipeline:")
    print(f"  {preprocess}")
    
    # Apply preprocessing step by step
    print(f"\nStep-by-step preprocessing:")
    
    # Step 1: Resize
    resize_transform = transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)
    resized = resize_transform(test_image)
    resized.save("corner_test_resized.png")
    print(f"  1. Resized to 224x224")
    
    # Step 2: CenterCrop 
    crop_transform = transforms.CenterCrop(224)
    cropped = crop_transform(resized)
    cropped.save("corner_test_cropped.png")
    print(f"  2. Center cropped to 224x224")
    
    # Step 3: Convert to tensor
    tensor_transform = transforms.ToTensor()
    tensor_img = tensor_transform(cropped)
    print(f"  3. Converted to tensor: {tensor_img.shape}")
    
    # Step 4: Normalize
    normalize_transform = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), 
        (0.26862954, 0.26130258, 0.27577711)
    )
    normalized = normalize_transform(tensor_img)
    print(f"  4. Normalized with ImageNet stats")
    
    # Convert normalized tensor back to viewable image
    # Denormalize for visualization
    denorm_tensor = normalized.clone()
    for t, m, s in zip(denorm_tensor, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]):
        t.mul_(s).add_(m)
    denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
    
    # Convert to PIL and save
    to_pil = transforms.ToPILImage()
    final_image = to_pil(denorm_tensor)
    final_image.save("corner_test_preprocessed.png")
    print(f"  💾 Saved preprocessed: corner_test_preprocessed.png")
    
    return normalized.unsqueeze(0)  # Add batch dimension

def examine_patch_extraction():
    """Examine how patches are extracted from the preprocessed image"""
    print(f"\n🔍 Examining Patch Extraction")
    print("="*40)
    
    # Load interpretable CLIP
    model = load_interpretable_clip("ViT-B/16", device="cpu")
    
    # Create test image
    test_image = create_corner_test_image(224)
    
    # Preprocess  
    image_input = model.preprocess(test_image).unsqueeze(0)
    print(f"Preprocessed image shape: {image_input.shape}")
    
    # Let's manually extract patches like CLIP does
    # CLIP uses conv2d with kernel_size=patch_size, stride=patch_size
    patch_size = 16  # ViT-B/16
    
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Expected grid: {224//patch_size}x{224//patch_size} = {(224//patch_size)**2} patches")
    
    # Extract patches manually using unfold
    image_tensor = image_input[0]  # Remove batch dim
    patches = image_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    print(f"Manual patch extraction shape: {patches.shape}")
    # Should be [3, 14, 14, 16, 16] = [channels, grid_h, grid_w, patch_h, patch_w]
    
    # Visualize first few patches
    print(f"\nVisualizing patch extraction order:")
    
    # Convert patches to viewable format
    patches_reshaped = patches.permute(1, 2, 0, 3, 4)  # [grid_h, grid_w, channels, patch_h, patch_w]
    
    # Save corner patches to verify ordering
    corners_to_check = [
        (0, 0, "top_left"),
        (0, 13, "top_right"), 
        (13, 0, "bottom_left"),
        (13, 13, "bottom_right"),
        (7, 7, "center")
    ]
    
    for row, col, name in corners_to_check:
        patch = patches_reshaped[row, col]  # [3, 16, 16]
        
        # Denormalize patch for visualization
        denorm_patch = patch.clone()
        for t, m, s in zip(denorm_patch, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]):
            t.mul_(s).add_(m)
        denorm_patch = torch.clamp(denorm_patch, 0, 1)
        
        # Convert to PIL
        to_pil = transforms.ToPILImage()
        patch_image = to_pil(denorm_patch)
        
        # Resize for better viewing
        patch_image = patch_image.resize((64, 64), Image.NEAREST)
        patch_image.save(f"patch_{name}_row{row}_col{col}.png")
        
        # Calculate 1D index
        patch_1d_index = row * 14 + col
        print(f"  {name}: row={row}, col={col}, 1D_index={patch_1d_index}")

def test_actual_clip_similarities():
    """Test CLIP similarities on our corner test image"""
    print(f"\n🎯 Testing CLIP Similarities on Corner Image")
    print("="*50)
    
    # Load model
    model = load_interpretable_clip("ViT-B/16", device="cpu")
    
    # Create and process image
    test_image = create_corner_test_image(224)
    image_input = model.preprocess(test_image).unsqueeze(0)
    
    # Test each corner color
    colors = ["red", "blue", "green", "yellow", "white"]
    expected_locations = {
        "red": (0, 0),      # Top-left
        "blue": (0, 13),    # Top-right  
        "green": (13, 0),   # Bottom-left
        "yellow": (13, 13), # Bottom-right
        "white": (7, 7)     # Center
    }
    
    for color in colors:
        print(f"\n--- Testing '{color}' color ---")
        
        text_input = clip.tokenize([f"{color} color"]).to("cpu")
        tokens, similarity = model.get_token_patch_similarity(image_input, text_input)
        
        # Find color token
        color_token_idx = None
        for i, token in enumerate(tokens):
            if color in token.lower():
                color_token_idx = i
                break
        
        if color_token_idx is None:
            print(f"  Token '{color}' not found in: {tokens}")
            continue
        
        # Get similarities
        token_similarities = similarity[color_token_idx, :].detach().cpu().numpy()
        
        # Find top patches
        top_5_indices = np.argsort(token_similarities)[-5:][::-1]
        
        print(f"  Token: '{tokens[color_token_idx]}'")
        print(f"  Expected location: {expected_locations.get(color, 'unknown')}")
        print(f"  Top 5 patches:")
        
        for i, idx in enumerate(top_5_indices):
            row = idx // 14
            col = idx % 14
            sim = token_similarities[idx]
            print(f"    {i+1}. Index {idx:3d} → ({row:2d},{col:2d}) = {sim:.4f}")
        
        # Check if expected location is in top 5
        expected_pos = expected_locations.get(color)
        if expected_pos:
            expected_row, expected_col = expected_pos
            expected_idx = expected_row * 14 + expected_col
            expected_sim = token_similarities[expected_idx]
            
            if expected_idx in top_5_indices:
                print(f"  ✅ Expected location in top 5! Similarity: {expected_sim:.4f}")
            else:
                rank = np.where(np.argsort(token_similarities)[::-1] == expected_idx)[0]
                if len(rank) > 0:
                    print(f"  ⚠️  Expected location rank: {rank[0]+1}, similarity: {expected_sim:.4f}")
                else:
                    print(f"  ❌ Expected location not found, similarity: {expected_sim:.4f}")

if __name__ == "__main__":
    print("🔬 CLIP Preprocessing Investigation")
    print("="*60)
    
    # Step 1: Analyze preprocessing
    preprocessed_tensor = analyze_clip_preprocessing()
    
    # Step 2: Examine patch extraction
    examine_patch_extraction()
    
    # Step 3: Test actual similarities
    test_actual_clip_similarities()
    
    print(f"\n🎯 Investigation Complete!")
    print(f"Check the generated images to see each step of the process.") 