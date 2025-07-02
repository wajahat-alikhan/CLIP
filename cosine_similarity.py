"""
Test different CLIP architectures for better spatial localization.
Compares ViT-B/32, ViT-B/16, and ViT-L/14 for patch-token correspondences.
"""

import torch
from PIL import Image
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clip.interpretable_clip import load_interpretable_clip, tokenize
from clip import load as load_standard_clip

def test_architecture(model_name, prompt="cat"):
    """Test a specific CLIP architecture."""
    print(f"\n{'='*60}")
    print(f"TESTING {model_name}")
    print(f"{'='*60}")
    
    try:
        # Load model
        print(f"Loading {model_name}...")
        model = load_interpretable_clip(model_name, device="cpu")
        preprocess = model.preprocess
        
        # Load image
        image = preprocess(Image.open("D:\Wajahat Ali Khan\CLIP\dogcat2.PNG")).unsqueeze(0)
        text_input = tokenize([prompt]).to("cpu")
        
        print(f"Testing with prompt: '{prompt}'")
        
        with torch.no_grad():
            tokens, similarity = model.get_token_patch_similarity(image, text_input, debug=False)
            
            if prompt in tokens:
                token_idx = tokens.index(prompt)
                sims = similarity[token_idx, :].numpy()
                
                # Calculate grid size
                num_patches = sims.shape[0]
                grid_size = int(np.sqrt(num_patches))
                patch_size = 224 // grid_size  # Assuming 224x224 input
                
                print(f"Architecture details:")
                print(f"  Patch size: {patch_size}×{patch_size} pixels")
                print(f"  Grid size: {grid_size}×{grid_size}")
                print(f"  Total patches: {num_patches}")
                
                print(f"\nSimilarity statistics:")
                print(f"  Min: {sims.min():.6f}")
                print(f"  Max: {sims.max():.6f}")
                print(f"  Mean: {sims.mean():.6f}")
                print(f"  Std: {sims.std():.6f}")
                print(f"  Range: {sims.max() - sims.min():.6f}")
                
                # Reshape to spatial grid
                if grid_size * grid_size == num_patches:
                    spatial_grid = sims.reshape(grid_size, grid_size)
                    
                    # Find peak location
                    max_idx = np.argmax(sims)
                    max_row, max_col = max_idx // grid_size, max_idx % grid_size
                    
                    print(f"\nPeak location:")
                    print(f"  Grid position: ({max_row}, {max_col})")
                    print(f"  Peak value: {sims[max_idx]:.6f}")
                    
                    # Calculate pixel coordinates (approximate center of patch)
                    pixel_row = max_row * patch_size + patch_size // 2
                    pixel_col = max_col * patch_size + patch_size // 2
                    print(f"  Approximate pixel position: ({pixel_row}, {pixel_col})")
                    
                    # Show top 5 patches
                    top_indices = np.argsort(sims)[-5:][::-1]
                    print(f"\nTop 5 patches:")
                    for i, idx in enumerate(top_indices):
                        row, col = idx // grid_size, idx % grid_size
                        pixel_r = row * patch_size + patch_size // 2
                        pixel_c = col * patch_size + patch_size // 2
                        print(f"  {i+1}. Patch {idx} at grid({row},{col}) ≈ pixel({pixel_r},{pixel_c}): {sims[idx]:.6f}")
                    
                    # Create visualization
                    plt.figure(figsize=(8, 6))
                    im = plt.imshow(spatial_grid, cmap='viridis', interpolation='nearest')
                    plt.title(f"{model_name} - '{prompt}' token similarities")
                    plt.xlabel("Patch Column")
                    plt.ylabel("Patch Row")
                    plt.colorbar(im, label="Cosine Similarity")
                    
                    # Add grid lines
                    for x in range(grid_size+1):
                        plt.axhline(x-0.5, color='white', linewidth=0.5, alpha=0.7)
                        plt.axvline(x-0.5, color='white', linewidth=0.5, alpha=0.7)
                    
                    # Mark peak
                    plt.scatter(max_col, max_row, color='red', s=100, marker='x', linewidth=3)
                    
                    plt.tight_layout()
                    plt.savefig(f"{model_name.replace('/', '-')}_{prompt}_similarities.png", dpi=150, bbox_inches='tight')
                    plt.show()
                    
                    return {
                        'model': model_name,
                        'grid_size': grid_size,
                        'patch_size': patch_size,
                        'similarities': sims,
                        'spatial_grid': spatial_grid,
                        'peak_location': (max_row, max_col),
                        'peak_value': sims[max_idx],
                        'variance': sims.std()
                    }
                else:
                    print(f"⚠️ Cannot reshape {num_patches} patches into square grid")
                    return None
            else:
                print(f"⚠️ Token '{prompt}' not found in {tokens}")
                return None
                
    except Exception as e:
        print(f"❌ Error testing {model_name}: {str(e)}")
        return None

def compare_architectures():
    """Compare multiple CLIP architectures."""
    print("🔍 COMPARING CLIP ARCHITECTURES FOR SPATIAL LOCALIZATION")
    print("="*70)
    
    # Test different architectures
    architectures = [
        "ViT-B/32",  # Current - coarse
        "ViT-B/16",  # Finer resolution
        "ViT-L/14"   # Finest resolution
    ]
    
    results = {}
    
    for arch in architectures:
        result = test_architecture(arch, prompt="cat")
        if result:
            results[arch] = result
    
    # Summary comparison
    if results:
        print(f"\n{'='*70}")
        print("ARCHITECTURE COMPARISON SUMMARY")
        print(f"{'='*70}")
        
        print(f"{'Architecture':<12} {'Grid':<8} {'Patch Size':<12} {'Peak Value':<12} {'Variance':<10} {'Peak Location'}")
        print("-" * 70)
        
        for arch, data in results.items():
            grid_str = f"{data['grid_size']}×{data['grid_size']}"
            patch_str = f"{data['patch_size']}×{data['patch_size']}"
            peak_pos = f"({data['peak_location'][0]},{data['peak_location'][1]})"
            
            print(f"{arch:<12} {grid_str:<8} {patch_str:<12} {data['peak_value']:<12.6f} {data['variance']:<10.6f} {peak_pos}")
        
        # Recommend best architecture
        best_arch = max(results.keys(), key=lambda k: results[k]['variance'])
        print(f"\n🏆 BEST ARCHITECTURE FOR LOCALIZATION: {best_arch}")
        print(f"   Reason: Highest variance ({results[best_arch]['variance']:.6f}) indicates better spatial discrimination")

def test_multiple_tokens():
    """Test the best architecture with multiple tokens."""
    print(f"\n{'='*70}")
    print("TESTING MULTIPLE TOKENS WITH BEST ARCHITECTURE")
    print(f"{'='*70}")
    
    # Use ViT-L/14 (should be the best)
    model_name = "ViT-L/14"
    tokens_to_test = ["cat", "dog"]
    
    print(f"Using {model_name} for detailed token analysis...")
    
    results = {}
    for token in tokens_to_test:
        print(f"\n--- Testing '{token}' ---")
        result = test_architecture(model_name, prompt=token)
        if result:
            results[token] = result
    
    # Compare token locations
    if len(results) >= 2:
        print(f"\n🎯 COMPARING TOKEN PEAK LOCATIONS:")
        for token, data in results.items():
            peak_pos = data['peak_location']
            pixel_pos = (peak_pos[0] * data['patch_size'] + data['patch_size']//2,
                        peak_pos[1] * data['patch_size'] + data['patch_size']//2)
            print(f"  {token}: grid({peak_pos[0]},{peak_pos[1]}) ≈ pixel{pixel_pos}")
        
        # Check if they're different
        locations = [data['peak_location'] for data in results.values()]
        if len(set(locations)) == len(locations):
            print("✅ All tokens have DIFFERENT peak locations - good spatial discrimination!")
        else:
            print("⚠️ Some tokens have the same peak locations")

def debug_spatial_mapping():
    """Debug the spatial mapping to identify coordinate system issues."""
    print("🔍 DEBUGGING SPATIAL MAPPING")
    print("="*50)
    
    # Load model
    model = load_interpretable_clip("ViT-L/14", device="cpu")
    preprocess = model.preprocess
    
    # Load image
    original_image = Image.open("D:\Wajahat Ali Khan\CLIP\dogcat2.PNG")
    image = preprocess(original_image).unsqueeze(0)
    
    print(f"Original image size: {original_image.size}")
    print(f"Preprocessed image shape: {image.shape}")
    
    # Test with both cat and dog
    test_tokens = ["cat", "dog"]
    
    results = {}
    
    for token in test_tokens:
        print(f"\n--- Testing '{token}' token ---")
        text_input = tokenize([token]).to("cpu")
        
        with torch.no_grad():
            tokens, similarity = model.get_token_patch_similarity(image, text_input, debug=True)
            
            if token in tokens:
                token_idx = tokens.index(token)
                sims = similarity[token_idx, :].numpy()
                
                # Calculate grid info
                num_patches = sims.shape[0]
                grid_size = int(np.sqrt(num_patches))
                patch_size = 224 // grid_size
                
                print(f"Grid size: {grid_size}×{grid_size}")
                print(f"Patch size: {patch_size}×{patch_size} pixels")
                
                # Reshape to spatial grid
                spatial_grid = sims.reshape(grid_size, grid_size)
                
                # Find peak
                max_idx = np.argmax(sims)
                max_row, max_col = max_idx // grid_size, max_idx % grid_size
                
                print(f"Peak location:")
                print(f"  Flat index: {max_idx}")
                print(f"  Grid position: ({max_row}, {max_col})")
                print(f"  Peak value: {sims[max_idx]:.6f}")
                
                # Calculate pixel coordinates 
                pixel_row = max_row * patch_size + patch_size // 2
                pixel_col = max_col * patch_size + patch_size // 2
                print(f"  Pixel position: ({pixel_row}, {pixel_col})")
                
                # Store results
                results[token] = {
                    'similarities': sims,
                    'spatial_grid': spatial_grid,
                    'peak_grid': (max_row, max_col),
                    'peak_pixel': (pixel_row, pixel_col),
                    'peak_value': sims[max_idx],
                    'grid_size': grid_size,
                    'patch_size': patch_size
                }
    
    # Create comprehensive visualization
    if results:
        create_debug_visualization(original_image, results)
        
    return results

def create_debug_visualization(original_image, results):
    """Create a comprehensive visualization to debug spatial mapping."""
    
    num_tokens = len(results)
    fig, axes = plt.subplots(2, num_tokens + 1, figsize=(5 * (num_tokens + 1), 10))
    
    if num_tokens == 1:
        axes = axes.reshape(2, -1)
    
    # First column: Original image with annotations
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Image\n(Dog LEFT, Cat RIGHT)", fontweight='bold')
    axes[0, 0].axis('off')
    
    # Add manual annotations to show where animals are
    img_width, img_height = original_image.size
    
    # Approximate animal locations (you may need to adjust these)
    dog_bbox = patches.Rectangle((0, img_height*0.1), img_width*0.5, img_height*0.8, 
                                linewidth=3, edgecolor='blue', facecolor='none', alpha=0.7)
    cat_bbox = patches.Rectangle((img_width*0.5, img_height*0.1), img_width*0.5, img_height*0.8,
                                linewidth=3, edgecolor='red', facecolor='none', alpha=0.7)
    
    axes[0, 0].add_patch(dog_bbox)
    axes[0, 0].add_patch(cat_bbox)
    axes[0, 0].text(img_width*0.25, img_height*0.05, 'DOG', ha='center', va='top', 
                   fontsize=12, fontweight='bold', color='blue')
    axes[0, 0].text(img_width*0.75, img_height*0.05, 'CAT', ha='center', va='top',
                   fontsize=12, fontweight='bold', color='red')
    
    # Second row: Grid overlay
    axes[1, 0].imshow(original_image)
    axes[1, 0].set_title("Original + Grid Overlay", fontweight='bold')
    axes[1, 0].axis('off')
    
    # Add grid overlay to show patch boundaries
    if results:
        grid_size = list(results.values())[0]['grid_size']
        patch_size = list(results.values())[0]['patch_size']
        
        # Draw grid lines (approximate for 224x224 -> original size scaling)
        scale_x = img_width / 224
        scale_y = img_height / 224
        
        for i in range(grid_size + 1):
            x = i * patch_size * scale_x
            y = i * patch_size * scale_y
            axes[1, 0].axvline(x, color='white', alpha=0.5, linewidth=1)
            axes[1, 0].axhline(y, color='white', alpha=0.5, linewidth=1)
    
    # Token similarity visualizations
    for col, (token, data) in enumerate(results.items(), 1):
        # Top row: Similarity heatmap
        im = axes[0, col].imshow(data['spatial_grid'], cmap='viridis', interpolation='nearest')
        axes[0, col].set_title(f"'{token}' token similarities")
        axes[0, col].set_xlabel("Patch Column")
        axes[0, col].set_ylabel("Patch Row")
        
        # Add grid lines
        grid_size = data['grid_size']
        for x in range(grid_size + 1):
            axes[0, col].axhline(x-0.5, color='white', linewidth=0.5, alpha=0.7)
            axes[0, col].axvline(x-0.5, color='white', linewidth=0.5, alpha=0.7)
        
        # Mark peak
        peak_row, peak_col = data['peak_grid']
        axes[0, col].scatter(peak_col, peak_row, color='red', s=100, marker='x', linewidth=3)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)
        
        # Bottom row: Overlaid on original image
        axes[1, col].imshow(original_image)
        
        # Resize similarity map to original image size for overlay
        import cv2
        heatmap_resized = cv2.resize(data['spatial_grid'], (img_width, img_height), 
                                   interpolation=cv2.INTER_CUBIC)
        
        # Normalize for better visualization
        heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())
        
        im_overlay = axes[1, col].imshow(heatmap_norm, cmap='plasma', alpha=0.6, 
                                       extent=[0, img_width, img_height, 0])
        axes[1, col].set_title(f"'{token}' overlay on original")
        axes[1, col].axis('off')
        
        # Mark the peak on the overlay
        peak_pixel_col, peak_pixel_row = data['peak_pixel']
        # Scale to original image coordinates
        peak_orig_col = peak_pixel_col * (img_width / 224)
        peak_orig_row = peak_pixel_row * (img_height / 224)
        
        axes[1, col].scatter(peak_orig_col, peak_orig_row, color='red', s=100, marker='x', linewidth=3)
        
        plt.colorbar(im_overlay, ax=axes[1, col], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig("spatial_mapping_debug.png", dpi=150, bbox_inches='tight')
    plt.show()

def test_coordinate_systems():
    """Test different ways of interpreting the coordinate system."""
    print("\n🧪 TESTING COORDINATE SYSTEM INTERPRETATIONS")
    print("="*50)
    
    model = load_interpretable_clip("ViT-L/14", device="cpu")
    preprocess = model.preprocess
    
    image = preprocess(Image.open("D:\Wajahat Ali Khan\CLIP\dogcat2.PNG")).unsqueeze(0)
    text_input = tokenize(["cat"]).to("cpu")
    
    with torch.no_grad():
        tokens, similarity = model.get_token_patch_similarity(image, text_input, debug=False)
        
        if "cat" in tokens:
            cat_idx = tokens.index("cat")
            sims = similarity[cat_idx, :].numpy()
            
            grid_size = int(np.sqrt(sims.shape[0]))
            
            print(f"Testing different reshape interpretations for {grid_size}×{grid_size} grid:")
            
            # Method 1: Row-major (current)
            grid1 = sims.reshape(grid_size, grid_size)
            peak1 = np.unravel_index(np.argmax(grid1), grid1.shape)
            print(f"Method 1 (row-major): Peak at {peak1}")
            
            # Method 2: Column-major  
            grid2 = sims.reshape(grid_size, grid_size, order='F')
            peak2 = np.unravel_index(np.argmax(grid2), grid2.shape)
            print(f"Method 2 (col-major): Peak at {peak2}")
            
            # Method 3: Transpose
            grid3 = sims.reshape(grid_size, grid_size).T
            peak3 = np.unravel_index(np.argmax(grid3), grid3.shape)
            print(f"Method 3 (transposed): Peak at {peak3}")
            
            # Method 4: Flip horizontally
            grid4 = np.fliplr(sims.reshape(grid_size, grid_size))
            peak4 = np.unravel_index(np.argmax(grid4), grid4.shape)
            print(f"Method 4 (h-flipped): Peak at {peak4}")
            
            # Method 5: Flip vertically
            grid5 = np.flipud(sims.reshape(grid_size, grid_size))
            peak5 = np.unravel_index(np.argmax(grid5), grid5.shape)
            print(f"Method 5 (v-flipped): Peak at {peak5}")
            
            # Show which method would put the peak on the RIGHT side (where cat should be)
            print(f"\nExpected: Peak should be on RIGHT side (higher column values)")
            print(f"For {grid_size}×{grid_size} grid, RIGHT side ≈ columns {grid_size//2} to {grid_size-1}")

def debug_token_assignment():
    """Debug the token extraction and similarity assignment process."""
    print("🔍 DEBUGGING TOKEN-SIMILARITY ASSIGNMENT BUG")
    print("="*60)
    
    model = load_interpretable_clip("ViT-L/14", device="cpu")
    preprocess = model.preprocess
    
    # Test each token individually to isolate the issue
    individual_tests = ["cat", "dog"]
    
    for test_token in individual_tests:
        print(f"\n{'='*40}")
        print(f"TESTING INDIVIDUAL TOKEN: '{test_token}'")
        print(f"{'='*40}")
        
        # Load image
        image = preprocess(Image.open("D:\Wajahat Ali Khan\CLIP\dogcat2.PNG")).unsqueeze(0)
        text_input = tokenize([test_token]).to("cpu")
        
        print(f"Input text: '{test_token}'")
        print(f"Tokenized text shape: {text_input.shape}")
        print(f"Token IDs: {text_input[0].tolist()}")
        
        with torch.no_grad():
            # Get raw embeddings
            _, patch_embeddings = model.encode_image_with_patches(image)
            _, token_embeddings = model.encode_text_with_tokens(text_input)
            
            print(f"Patch embeddings shape: {patch_embeddings.shape}")
            print(f"Token embeddings shape: {token_embeddings.shape}")
            
            # Step-by-step token processing (replicate the internal logic)
            print(f"\n--- Token Processing Details ---")
            
            # Project token embeddings
            token_embeddings_proj = torch.matmul(token_embeddings, model.text_projection)
            print(f"Token embeddings after projection shape: {token_embeddings_proj.shape}")
            
            # Get token strings manually
            if hasattr(model, 'tokenizer') and model.tokenizer is not None:
                tokenizer = model.tokenizer
            else:
                from clip.simple_tokenizer import SimpleTokenizer
                tokenizer = SimpleTokenizer()
            
            all_tokens = []
            for token_id in text_input[0]:
                token_str = tokenizer.decode([token_id.cpu().item()])
                all_tokens.append(token_str)
            
            print(f"All decoded tokens: {all_tokens}")
            
            # Apply the same filtering logic
            sot_token_id = 49406  # <|startoftext|>  
            eot_token_id = 49407  # <|endoftext|>
            
            real_token_indices = []
            tokens_clean = []
            
            for i, (token_id, token_str) in enumerate(zip(text_input[0], all_tokens)):
                token_id_val = token_id.cpu().item()
                token_str_clean = token_str.strip().replace('</w>', '')
                
                print(f"  Token {i}: ID={token_id_val}, str='{token_str}', clean='{token_str_clean}'")
                
                # Skip special tokens, empty tokens, and padding
                if (token_id_val != sot_token_id and 
                    token_id_val != eot_token_id and
                    token_id_val != 0 and  # padding
                    token_str_clean and 
                    token_str_clean not in ['<|startoftext|>', '<|endoftext|>', '!', '.']):
                    real_token_indices.append(i)
                    tokens_clean.append(token_str_clean)
                    print(f"    ✅ KEPT: Index {i}, Token '{token_str_clean}'")
                else:
                    print(f"    ❌ FILTERED OUT")
            
            print(f"\nFinal filtered tokens: {tokens_clean}")
            print(f"Final token indices: {real_token_indices}")
            
            if len(tokens_clean) == 1 and tokens_clean[0].strip() == test_token.strip():
                print(f"✅ Token extraction is CORRECT")
            else:
                print(f"⚠️ Token extraction MISMATCH!")
                print(f"   Expected: '{test_token}'")
                print(f"   Got: {tokens_clean}")
            
            # Compute similarities for this single token
            if len(real_token_indices) > 0:
                patch_norm = torch.nn.functional.normalize(patch_embeddings, dim=-1)
                token_norm = torch.nn.functional.normalize(token_embeddings_proj, dim=-1)
                
                token_embeddings_real = token_norm[0, real_token_indices, :]
                print(f"Real token embeddings shape: {token_embeddings_real.shape}")
                
                # Compute similarity for each real token
                for j, (token_idx, token_name) in enumerate(zip(real_token_indices, tokens_clean)):
                    single_token_emb = token_embeddings_real[j:j+1, :]
                    similarity = torch.matmul(single_token_emb, patch_norm[0].transpose(0, 1))
                    
                    print(f"\n--- Similarity for token '{token_name}' (index {token_idx}) ---")
                    print(f"Similarity shape: {similarity.shape}")
                    print(f"Similarity range: [{similarity.min().item():.4f}, {similarity.max().item():.4f}]")
                    
                    # Find peak
                    sims = similarity[0].numpy()
                    max_idx = np.argmax(sims)
                    grid_size = int(np.sqrt(sims.shape[0]))
                    max_row, max_col = max_idx // grid_size, max_idx % grid_size
                    
                    print(f"Peak location: Grid({max_row}, {max_col}), Value: {sims[max_idx]:.4f}")
                    
                    # Interpret location
                    if max_col < grid_size // 2:
                        side = "LEFT (where dog is)"
                    else:
                        side = "RIGHT (where cat is)"
                    
                    print(f"Peak is on: {side}")
                    
                    # Check if this matches expectation
                    if test_token == "cat" and "cat" in side.lower():
                        print(f"✅ CORRECT: Cat token peaks where cat is")
                    elif test_token == "dog" and "dog" in side.lower():
                        print(f"✅ CORRECT: Dog token peaks where dog is")
                    else:
                        print(f"❌ BUG: {test_token} token peaks on wrong side!")

def debug_multi_token_text():
    """Test with multi-token text to see if the issue persists."""
    print(f"\n{'='*60}")
    print("TESTING MULTI-TOKEN TEXT")
    print(f"{'='*60}")
    
    model = load_interpretable_clip("ViT-L/14", device="cpu")
    preprocess = model.preprocess
    
    # Test with phrase containing both words
    test_phrase = "a cat and a dog"
    
    image = preprocess(Image.open("D:\Wajahat Ali Khan\CLIP\dogcat2.PNG")).unsqueeze(0)
    text_input = tokenize([test_phrase]).to("cpu")
    
    print(f"Testing phrase: '{test_phrase}'")
    
    with torch.no_grad():
        tokens, similarity = model.get_token_patch_similarity(image, text_input, debug=True)
        
        print(f"\nExtracted tokens: {tokens}")
        
        # Check each relevant token
        for i, token in enumerate(tokens):
            if token.strip().lower() in ['cat', 'dog']:
                sims = similarity[i, :].numpy()
                max_idx = np.argmax(sims)
                grid_size = int(np.sqrt(sims.shape[0]))
                max_row, max_col = max_idx // grid_size, max_idx % grid_size
                
                if max_col < grid_size // 2:
                    side = "LEFT (dog side)"
                else:
                    side = "RIGHT (cat side)"
                
                print(f"Token '{token}': peaks at Grid({max_row}, {max_col}) = {side}")

def test_inverted_similarities():
    """Test if inverted similarities show correct semantic localization."""
    print("🔄 TESTING INVERTED SIMILARITY HYPOTHESIS")
    print("="*60)
    print("Hypothesis: Lowest similarities = Strongest semantic correspondence")
    print("="*60)
    
    model = load_interpretable_clip("ViT-L/14", device="cpu")
    preprocess = model.preprocess
    
    # Load image
    original_image = Image.open("D:\Wajahat Ali Khan\CLIP\dogcat2.PNG")
    image = preprocess(original_image).unsqueeze(0)
    
    # Test with both cat and dog
    test_tokens = ["cat", "dog"]
    
    for token in test_tokens:
        print(f"\n{'='*40}")
        print(f"TESTING TOKEN: '{token}'")
        print(f"{'='*40}")
        
        text_input = tokenize([token]).to("cpu")
        
        with torch.no_grad():
            tokens, similarity = model.get_token_patch_similarity(image, text_input, debug=False)
            
            if token in tokens:
                token_idx = tokens.index(token)
                sims = similarity[token_idx, :].numpy()
                
                # Calculate statistics
                print(f"Original similarity stats:")
                print(f"  Min: {sims.min():.4f}")
                print(f"  Max: {sims.max():.4f}")
                print(f"  Mean: {sims.mean():.4f}")
                print(f"  Std: {sims.std():.4f}")
                
                # Find peak in original similarities
                grid_size = int(np.sqrt(sims.shape[0]))
                max_idx = np.argmax(sims)
                max_row, max_col = max_idx // grid_size, max_idx % grid_size
                
                # Find minimum in original similarities (inverted peak)
                min_idx = np.argmin(sims)
                min_row, min_col = min_idx // grid_size, min_idx % grid_size
                
                print(f"\nOriginal similarities:")
                print(f"  Peak (max): Grid({max_row}, {max_col}), Value: {sims[max_idx]:.4f}")
                if max_col < grid_size // 2:
                    orig_peak_side = "LEFT (dog side)"
                else:
                    orig_peak_side = "RIGHT (cat side)"
                print(f"  Peak location: {orig_peak_side}")
                
                print(f"\nInverted similarities (looking at minimum):")
                print(f"  Valley (min): Grid({min_row}, {min_col}), Value: {sims[min_idx]:.4f}")
                if min_col < grid_size // 2:
                    inv_peak_side = "LEFT (dog side)"
                else:
                    inv_peak_side = "RIGHT (cat side)"
                print(f"  Valley location: {inv_peak_side}")
                
                # Check if inversion gives correct results
                expected_side = "cat side" if token == "cat" else "dog side"
                
                print(f"\n🎯 ANALYSIS:")
                print(f"  Expected: {token} should focus on {expected_side}")
                
                orig_correct = expected_side in orig_peak_side.lower()
                inv_correct = expected_side in inv_peak_side.lower()
                
                print(f"  Original peak: {'✅ CORRECT' if orig_correct else '❌ WRONG'}")
                print(f"  Inverted peak: {'✅ CORRECT' if inv_correct else '❌ WRONG'}")
                
                if inv_correct and not orig_correct:
                    print(f"  🎉 HYPOTHESIS CONFIRMED: Inversion fixes the localization!")
                elif orig_correct and not inv_correct:
                    print(f"  📊 Original is correct, inversion makes it worse")
                elif inv_correct and orig_correct:
                    print(f"  🤔 Both methods give correct results")
                else:
                    print(f"  😕 Neither method gives correct results")
                
                # Create comparative visualization
                create_comparison_visualization(original_image, sims, token, grid_size)

def create_comparison_visualization(original_image, similarities, token, grid_size):
    """Create side-by-side comparison of normal vs inverted similarities."""
    
    # Create spatial grids
    spatial_grid_normal = similarities.reshape(grid_size, grid_size)
    spatial_grid_inverted = -similarities.reshape(grid_size, grid_size)  # Invert
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Image", fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(original_image)
    axes[1, 0].set_title("Original Image", fontweight='bold') 
    axes[1, 0].axis('off')
    
    # Normal similarities (top row)
    im1 = axes[0, 1].imshow(spatial_grid_normal, cmap='viridis', interpolation='nearest')
    axes[0, 1].set_title(f"Normal Similarities\n'{token}' token")
    axes[0, 1].set_xlabel("Patch Column")
    axes[0, 1].set_ylabel("Patch Row")
    
    # Add peak marker
    max_idx = np.argmax(similarities)
    max_row, max_col = max_idx // grid_size, max_idx % grid_size
    axes[0, 1].scatter(max_col, max_row, color='red', s=100, marker='x', linewidth=3)
    
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Inverted similarities (bottom row)
    im2 = axes[1, 1].imshow(spatial_grid_inverted, cmap='viridis', interpolation='nearest')
    axes[1, 1].set_title(f"INVERTED Similarities\n'{token}' token")
    axes[1, 1].set_xlabel("Patch Column")
    axes[1, 1].set_ylabel("Patch Row")
    
    # Add valley marker (which becomes peak after inversion)
    min_idx = np.argmin(similarities)
    min_row, min_col = min_idx // grid_size, min_idx % grid_size
    axes[1, 1].scatter(min_col, min_row, color='red', s=100, marker='x', linewidth=3)
    
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Overlays on original image
    img_width, img_height = original_image.size
    
    # Normal overlay
    import cv2
    heatmap_normal = cv2.resize(spatial_grid_normal, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
    heatmap_norm = (heatmap_normal - heatmap_normal.min()) / (heatmap_normal.max() - heatmap_normal.min())
    
    axes[0, 2].imshow(original_image)
    im3 = axes[0, 2].imshow(heatmap_norm, cmap='plasma', alpha=0.6, extent=[0, img_width, img_height, 0])
    axes[0, 2].set_title(f"Normal Overlay\n'{token}' token")
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Inverted overlay  
    heatmap_inverted = cv2.resize(spatial_grid_inverted, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
    heatmap_inv_norm = (heatmap_inverted - heatmap_inverted.min()) / (heatmap_inverted.max() - heatmap_inverted.min())
    
    axes[1, 2].imshow(original_image)
    im4 = axes[1, 2].imshow(heatmap_inv_norm, cmap='plasma', alpha=0.6, extent=[0, img_width, img_height, 0])
    axes[1, 2].set_title(f"INVERTED Overlay\n'{token}' token")
    axes[1, 2].axis('off')
    plt.colorbar(im4, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(f"inversion_test_{token}.png", dpi=150, bbox_inches='tight')
    plt.show()

def test_distance_metrics():
    """Test different distance metrics to see if inversion affects them all."""
    print(f"\n{'='*60}")
    print("TESTING DIFFERENT DISTANCE METRICS")
    print(f"{'='*60}")
    
    model = load_interpretable_clip("ViT-L/14", device="cpu")
    preprocess = model.preprocess
    
    image = preprocess(Image.open("D:\Wajahat Ali Khan\CLIP\dogcat2.PNG")).unsqueeze(0)
    text_input = tokenize(["cat"]).to("cpu")
    
    with torch.no_grad():
        # Get embeddings
        _, patch_embeddings = model.encode_image_with_patches(image)
        _, token_embeddings = model.encode_text_with_tokens(text_input)
        token_embeddings_proj = torch.matmul(token_embeddings, model.text_projection)
        
        # Get cat token embedding
        cat_embedding = token_embeddings_proj[0, 1, :]  # Index 1 is cat token
        patch_embeddings_flat = patch_embeddings[0, :, :]  # All patches
        
        print(f"Testing different similarity metrics:")
        
        # 1. Cosine similarity (current method)
        patch_norm = torch.nn.functional.normalize(patch_embeddings_flat, dim=-1)
        cat_norm = torch.nn.functional.normalize(cat_embedding.unsqueeze(0), dim=-1)
        cosine_sim = torch.matmul(cat_norm, patch_norm.transpose(0, 1))[0].numpy()
        
        # 2. Euclidean distance (inverted)
        euclidean_dist = torch.norm(patch_embeddings_flat - cat_embedding.unsqueeze(0), dim=1).numpy()
        euclidean_sim = -euclidean_dist  # Invert so higher = more similar
        
        # 3. Dot product (no normalization)
        dot_product = torch.matmul(cat_embedding.unsqueeze(0), patch_embeddings_flat.transpose(0, 1))[0].numpy()
        
        # Find peaks for each method
        grid_size = int(np.sqrt(cosine_sim.shape[0]))
        
        methods = {
            "Cosine Similarity": cosine_sim,
            "Euclidean Distance (inverted)": euclidean_sim, 
            "Dot Product": dot_product
        }
        
        for method_name, values in methods.items():
            max_idx = np.argmax(values)
            max_row, max_col = max_idx // grid_size, max_idx % grid_size
            
            if max_col < grid_size // 2:
                side = "LEFT (dog side)"
            else:
                side = "RIGHT (cat side)"
                
            print(f"  {method_name}: Peak at Grid({max_row}, {max_col}) = {side}")
            print(f"    Value range: [{values.min():.4f}, {values.max():.4f}]")

def main():
    print("🔄 INVERTED SIMILARITY HYPOTHESIS TEST")
    print("Testing if lowest similarities = strongest semantic correspondence")
    print("="*70)
    
    # Test the main hypothesis
    test_inverted_similarities()
    
    # Test different distance metrics
    test_distance_metrics()
    
    print(f"\n{'='*70}")
    print("🎯 CONCLUSION:")
    print("Look at the visualizations to see if inversion fixes the localization!")
    print("If valleys (dark areas) in normal maps become peaks in inverted maps,")
    print("and those peaks align with the correct animals, then the hypothesis is confirmed!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

"""
Comprehensive investigation to find the mathematical bug causing inverted similarities.
Since CLIP's global similarities work correctly, there must be an implementation error.
"""

def debug_mathematical_operations():
    """Step-by-step debugging of all mathematical operations to find the bug."""
    print("🔍 COMPREHENSIVE MATHEMATICAL DEBUG")
    print("Looking for matrix operations, indexing, or coordinate system bugs")
    print("="*70)
    
    # Load both models for comparison
    interpretable_model = load_interpretable_clip("ViT-L/14", device="cpu")
    standard_model, standard_preprocess = load_standard_clip("ViT-L/14", device="cpu")
    
    # Load image and text
    image = interpretable_model.preprocess(Image.open("D:\Wajahat Ali Khan\CLIP\dogcat2.PNG")).unsqueeze(0)
    cat_text = tokenize(["cat"]).to("cpu")
    
    print("STEP 1: VERIFY GLOBAL SIMILARITIES MATCH")
    print("-" * 50)
    
    with torch.no_grad():
        # Standard CLIP global similarity
        std_img_features = standard_model.encode_image(image)
        std_text_features = standard_model.encode_text(cat_text)
        std_similarity = torch.cosine_similarity(std_img_features, std_text_features, dim=1).item()
        
        # Interpretable CLIP global similarity
        interp_img_features, _ = interpretable_model.encode_image_with_patches(image)
        interp_text_features, _ = interpretable_model.encode_text_with_tokens(cat_text)
        interp_similarity = torch.cosine_similarity(interp_img_features, interp_text_features, dim=1).item()
        
        print(f"Standard CLIP global similarity: {std_similarity:.6f}")
        print(f"Interpretable CLIP global similarity: {interp_similarity:.6f}")
        print(f"Difference: {abs(std_similarity - interp_similarity):.8f}")
        
        if abs(std_similarity - interp_similarity) < 1e-6:
            print("✅ Global similarities match perfectly - base implementation is correct")
        else:
            print("❌ Global similarities don't match - fundamental issue!")
            return
    
    print(f"\nSTEP 2: ANALYZE PATCH EMBEDDING EXTRACTION")
    print("-" * 50)
    
    with torch.no_grad():
        # Get patch embeddings step by step
        _, patch_embeddings = interpretable_model.encode_image_with_patches(image)
        print(f"Patch embeddings shape: {patch_embeddings.shape}")
        print(f"Expected: [1, 256, 768] for ViT-L/14")
        
        # Check if patch embeddings are in the same space as global embedding
        global_emb = interp_img_features[0]  # [768]
        patch_embs = patch_embeddings[0]     # [256, 768]
        
        print(f"Global embedding norm: {torch.norm(global_emb):.4f}")
        print(f"Patch embedding norms: mean={torch.norm(patch_embs, dim=1).mean():.4f}")
        
        # Check if global embedding is related to patch embeddings
        # In ViT, global embedding is typically the CLS token (index 0)
        # Let's see if our patch embeddings exclude or include the CLS token
        print(f"Number of patches: {patch_embs.shape[0]}")
        print(f"Expected for 224x224 with 14x14 patches: 16*16 = 256")
        
        # Check if any patch embedding is similar to global embedding
        patch_similarities_to_global = torch.cosine_similarity(global_emb.unsqueeze(0), patch_embs, dim=1)
        max_sim_to_global = torch.max(patch_similarities_to_global)
        print(f"Max similarity between global and any patch: {max_sim_to_global:.4f}")
        
        if max_sim_to_global < 0.5:
            print("✅ Patch embeddings are separate from global (good)")
        else:
            print("⚠️ Some patch embedding is very similar to global embedding")
    
    print(f"\nSTEP 3: ANALYZE TOKEN EMBEDDING EXTRACTION")
    print("-" * 50)
    
    with torch.no_grad():
        # Get token embeddings step by step
        _, token_embeddings = interpretable_model.encode_text_with_tokens(cat_text)
        print(f"Token embeddings shape: {token_embeddings.shape}")
        print(f"Expected: [1, 77, 768] for CLIP")
        
        # Project to image space
        token_embeddings_proj = torch.matmul(token_embeddings, interpretable_model.text_projection)
        print(f"Projected token embeddings shape: {token_embeddings_proj.shape}")
        
        # Check if projection is correct
        global_text_emb = interp_text_features[0]  # [768]
        
        # Find which token gives us the global embedding
        # Standard CLIP uses the EOS token for global representation
        text_tokens = cat_text[0]  # [77]
        eot_position = text_tokens.argmax().item()  # Position of EOS token
        
        print(f"EOS token position: {eot_position}")
        
        # Check if our projected token at EOS position matches global
        eot_token_emb = token_embeddings_proj[0, eot_position, :]
        global_similarity = torch.cosine_similarity(global_text_emb.unsqueeze(0), eot_token_emb.unsqueeze(0), dim=1).item()
        print(f"Similarity between global text and EOS token: {global_similarity:.6f}")
        
        if global_similarity > 0.99:
            print("✅ Token projection is correct")
        else:
            print("❌ Token projection might be wrong!")
    
    print(f"\nSTEP 4: DEBUG SIMILARITY COMPUTATION")
    print("-" * 50)
    
    with torch.no_grad():
        # Extract the specific embeddings we're comparing
        cat_token_emb = token_embeddings_proj[0, 1, :]  # Cat token (index 1)
        patch_embs = patch_embeddings[0, :, :]           # All patches
        
        print(f"Cat token embedding shape: {cat_token_emb.shape}")
        print(f"Patch embeddings shape: {patch_embs.shape}")
        
        # Method 1: Our current method
        patch_norm = torch.nn.functional.normalize(patch_embs, dim=1)
        cat_norm = torch.nn.functional.normalize(cat_token_emb.unsqueeze(0), dim=1)
        similarities_method1 = torch.matmul(cat_norm, patch_norm.transpose(0, 1))[0]
        
        # Method 2: Alternative order
        similarities_method2 = torch.matmul(patch_norm, cat_norm.transpose(0, 1))[:, 0]
        
        # Method 3: Using torch.cosine_similarity directly
        similarities_method3 = torch.cosine_similarity(cat_token_emb.unsqueeze(0), patch_embs, dim=1)
        
        print(f"Method 1 (our current): range=[{similarities_method1.min():.4f}, {similarities_method1.max():.4f}]")
        print(f"Method 2 (alternative): range=[{similarities_method2.min():.4f}, {similarities_method2.max():.4f}]")
        print(f"Method 3 (torch.cosine): range=[{similarities_method3.min():.4f}, {similarities_method3.max():.4f}]")
        
        # Check if methods give same results
        diff_1_2 = torch.max(torch.abs(similarities_method1 - similarities_method2))
        diff_1_3 = torch.max(torch.abs(similarities_method1 - similarities_method3))
        
        print(f"Difference between method 1 and 2: {diff_1_2:.8f}")
        print(f"Difference between method 1 and 3: {diff_1_3:.8f}")
        
        if diff_1_3 < 1e-6:
            print("✅ Similarity computation is mathematically correct")
        else:
            print("❌ Similarity computation has issues!")
    
    print(f"\nSTEP 5: CHECK SPATIAL COORDINATE SYSTEM")
    print("-" * 50)
    
    with torch.no_grad():
        # Test if patch ordering matches spatial layout
        sims = similarities_method1.numpy()
        grid_size = int(np.sqrt(len(sims)))
        
        print(f"Patch grid size: {grid_size}×{grid_size}")
        
        # Test different reshape methods
        spatial_grid_1 = sims.reshape(grid_size, grid_size)           # Row-major
        spatial_grid_2 = sims.reshape(grid_size, grid_size, order='F') # Column-major
        spatial_grid_3 = sims.reshape(grid_size, grid_size).T        # Transposed
        
        # Find peaks in each method
        peak_1 = np.unravel_index(np.argmax(spatial_grid_1), spatial_grid_1.shape)
        peak_2 = np.unravel_index(np.argmax(spatial_grid_2), spatial_grid_2.shape)
        peak_3 = np.unravel_index(np.argmax(spatial_grid_3), spatial_grid_3.shape)
        
        print(f"Peak locations:")
        print(f"  Row-major reshape: {peak_1} = {'LEFT' if peak_1[1] < grid_size//2 else 'RIGHT'}")
        print(f"  Col-major reshape: {peak_2} = {'LEFT' if peak_2[1] < grid_size//2 else 'RIGHT'}")
        print(f"  Transposed reshape: {peak_3} = {'LEFT' if peak_3[1] < grid_size//2 else 'RIGHT'}")
        
        # Check valleys (minimum values) too
        valley_1 = np.unravel_index(np.argmin(spatial_grid_1), spatial_grid_1.shape)
        valley_2 = np.unravel_index(np.argmin(spatial_grid_2), spatial_grid_2.shape)
        valley_3 = np.unravel_index(np.argmin(spatial_grid_3), spatial_grid_3.shape)
        
        print(f"Valley locations:")
        print(f"  Row-major reshape: {valley_1} = {'LEFT' if valley_1[1] < grid_size//2 else 'RIGHT'}")
        print(f"  Col-major reshape: {valley_2} = {'LEFT' if valley_2[1] < grid_size//2 else 'RIGHT'}")
        print(f"  Transposed reshape: {valley_3} = {'LEFT' if valley_3[1] < grid_size//2 else 'RIGHT'}")
        
        # Expected: cat peaks should be on RIGHT, valleys on LEFT (if our hypothesis is wrong)
        #           OR cat valleys should be on RIGHT (if inversion hypothesis is correct)
        
        print(f"\n🎯 COORDINATE SYSTEM ANALYSIS:")
        print(f"  Cat token should correspond to RIGHT side of image")
        
        peak_correct = [
            "Peak on RIGHT" if peak_1[1] >= grid_size//2 else "Peak on LEFT",
            "Peak on RIGHT" if peak_2[1] >= grid_size//2 else "Peak on LEFT",
            "Peak on RIGHT" if peak_3[1] >= grid_size//2 else "Peak on LEFT"
        ]
        
        valley_correct = [
            "Valley on RIGHT" if valley_1[1] >= grid_size//2 else "Valley on LEFT",
            "Valley on RIGHT" if valley_2[1] >= grid_size//2 else "Valley on LEFT", 
            "Valley on RIGHT" if valley_3[1] >= grid_size//2 else "Valley on LEFT"
        ]
        
        for i, method in enumerate(["Row-major", "Col-major", "Transposed"]):
            print(f"  {method}: {peak_correct[i]}, {valley_correct[i]}")
    
    print(f"\nSTEP 6: VERIFY AGAINST VISION TRANSFORMER IMPLEMENTATION")
    print("-" * 50)
    
    # Check how patches are extracted in the original ViT
    with torch.no_grad():
        # Look at the actual Vision Transformer forward pass
        print("Checking ViT patch extraction order...")
        
        # The issue might be in how we interpret the patch ordering
        # Standard ViT processes patches in row-major order: (0,0), (0,1), ..., (0,15), (1,0), (1,1), ...
        
        # Create a test pattern to verify ordering
        test_similarities = np.arange(256).reshape(16, 16)  # Numbers 0-255 in spatial order
        
        # Flatten in the same way ViT would
        flattened_test = test_similarities.flatten()  # Row-major flattening
        
        # Reshape back - this should recover the original
        recovered_test = flattened_test.reshape(16, 16)
        
        print(f"Test pattern recovery successful: {np.array_equal(test_similarities, recovered_test)}")
        
        # Now apply this to our actual similarities
        print(f"\n🔍 FINAL DIAGNOSIS:")
        if diff_1_3 < 1e-6:
            print("✅ Mathematical operations are correct")
        else:
            print("❌ Found mathematical error in similarity computation")
            
        # The key insight: if our math is correct but results are inverted,
        # then either:
        # 1. CLIP actually does learn inverted relationships (unlikely)
        # 2. We're misunderstanding what the embeddings represent
        # 3. There's a subtle bug in how we extract or interpret embeddings
        
        return {
            'similarities': sims,
            'grid_size': grid_size,
            'mathematical_error': diff_1_3 > 1e-6
        }

def main():
    print("🔧 MATHEMATICAL BUG HUNT")
    print("Investigating every matrix operation to find why we need inverted similarities")
    print("="*80)
    
    results = debug_mathematical_operations()
    
    print(f"\n{'='*80}")
    print("🎯 SUMMARY:")
    print("If mathematical operations are correct but results are inverted,")
    print("then the issue is likely in our interpretation of what embeddings mean,")
    print("not in the computations themselves.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 