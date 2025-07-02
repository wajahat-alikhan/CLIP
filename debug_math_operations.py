"""
Comprehensive investigation to find the mathematical bug causing inverted similarities.
Since CLIP's global similarities work correctly, there must be an implementation error.
"""

import torch
from PIL import Image
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clip.interpretable_clip import load_interpretable_clip, tokenize
from clip import load as load_standard_clip

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
        
        return {
            'similarities': sims,
            'grid_size': grid_size,
            'peaks': [peak_1, peak_2, peak_3],
            'valleys': [valley_1, valley_2, valley_3],
            'mathematical_error': diff_1_3 > 1e-6
        }

def test_coordinate_system_fix():
    """Test if any coordinate system transformation fixes the issue."""
    print(f"\nSTEP 6: TESTING COORDINATE SYSTEM FIXES")
    print("-" * 50)
    
    # Load model and get similarities
    model = load_interpretable_clip("ViT-L/14", device="cpu")
    image = model.preprocess(Image.open("D:\Wajahat Ali Khan\CLIP\dogcat2.PNG")).unsqueeze(0)
    cat_text = tokenize(["cat"]).to("cpu")
    
    with torch.no_grad():
        tokens, similarity = model.get_token_patch_similarity(image, cat_text, debug=False)
        if "cat" in tokens:
            cat_idx = tokens.index("cat")
            sims = similarity[cat_idx, :].numpy()
            grid_size = int(np.sqrt(len(sims)))
            
            transformations = {
                "Original": sims.reshape(grid_size, grid_size),
                "Horizontal Flip": np.fliplr(sims.reshape(grid_size, grid_size)),
                "Vertical Flip": np.flipud(sims.reshape(grid_size, grid_size)),
                "Transpose": sims.reshape(grid_size, grid_size).T,
                "Rotate 90°": np.rot90(sims.reshape(grid_size, grid_size)),
                "Rotate 180°": np.rot90(sims.reshape(grid_size, grid_size), 2),
                "Rotate 270°": np.rot90(sims.reshape(grid_size, grid_size), 3)
            }
            
            print("Testing different coordinate transformations:")
            for name, grid in transformations.items():
                peak = np.unravel_index(np.argmax(grid), grid.shape)
                valley = np.unravel_index(np.argmin(grid), grid.shape)
                
                peak_side = "RIGHT" if peak[1] >= grid_size//2 else "LEFT"
                valley_side = "RIGHT" if valley[1] >= grid_size//2 else "LEFT"
                
                print(f"  {name:15s}: Peak {peak} ({peak_side}), Valley {valley} ({valley_side})")
                
                # Check if this transformation gives correct results
                if peak_side == "RIGHT":
                    print(f"    ✅ {name} gives correct peak location!")
                if valley_side == "RIGHT":
                    print(f"    🔄 {name} gives correct valley location (supports inversion)")

def main():
    print("🔧 MATHEMATICAL BUG HUNT")
    print("Investigating every matrix operation to find why we need inverted similarities")
    print("="*80)
    
    results = debug_mathematical_operations()
    test_coordinate_system_fix()
    
    print(f"\n{'='*80}")
    print("🎯 SUMMARY:")
    if results and not results['mathematical_error']:
        print("✅ Mathematical operations are correct")
        print("❌ Issue is likely in coordinate system interpretation or")
        print("   CLIP actually does learn anti-correlations (less likely)")
    else:
        print("❌ Found mathematical errors that need fixing")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 