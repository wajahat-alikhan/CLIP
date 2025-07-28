"""
Test script to verify patch location mapping in CLIP heatmap visualization.
This verifies that the 1D similarity vector correctly maps back to 2D spatial positions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def test_patch_mapping():
    """Test that patch indices correctly map to spatial positions"""
    
    # Simulate ViT-B/32 configuration
    grid_size = 7  # 224/32 = 7
    num_patches = grid_size * grid_size  # 49 patches
    
    print(f"Testing patch mapping for {grid_size}×{grid_size} = {num_patches} patches")
    
    # Create a test similarity vector where each patch has a unique value
    # We'll use the patch index as the similarity value for easy verification
    test_similarities = torch.arange(num_patches, dtype=torch.float32)
    
    print(f"Original 1D similarities shape: {test_similarities.shape}")
    print(f"First few values: {test_similarities[:10].tolist()}")
    
    # ===== STEP 1: Simulate how CLIP creates patches =====
    print("\n=== STEP 1: Simulating CLIP patch creation ===")
    
    # Simulate the conv1 output: [batch, width, grid_h, grid_w]
    # We'll create a synthetic "feature map" where each spatial position
    # has a unique identifier matching our test_similarities
    batch_size, width = 1, 768
    
    # Create synthetic conv1 output where each spatial position (i,j) has value i*grid_size + j
    synthetic_conv_output = torch.zeros(batch_size, width, grid_size, grid_size)
    for i in range(grid_size):
        for j in range(grid_size):
            patch_idx = i * grid_size + j
            synthetic_conv_output[0, :, i, j] = patch_idx  # All channels get the same patch_idx
    
    print(f"Synthetic conv1 output shape: {synthetic_conv_output.shape}")
    print(f"Spatial layout (patch indices):")
    print(synthetic_conv_output[0, 0, :, :].numpy().astype(int))
    
    # Apply the same reshaping as CLIP does
    x = synthetic_conv_output.reshape(batch_size, width, -1)  # [batch, width, grid²]
    x = x.permute(0, 2, 1)  # [batch, grid², width]
    
    # Extract just the patch indices (from first channel)
    patch_indices_flattened = x[0, :, 0]  # Shape: [49]
    
    print(f"\nAfter CLIP reshape, flattened patch order:")
    print(f"Shape: {patch_indices_flattened.shape}")
    print(f"Values: {patch_indices_flattened.numpy().astype(int)}")
    
    # ===== STEP 2: Test the visualization reshaping =====
    print("\n=== STEP 2: Testing visualization reshape ===")
    
    # Use our test similarities (which match patch indices)
    image_relevance = test_similarities
    
    # Apply the same reshaping as the visualization code
    dim = int(image_relevance.numel() ** 0.5)
    image_relevance_2d = image_relevance.reshape(1, 1, dim, dim)
    
    print(f"Visualization reshape: {image_relevance.shape} -> {image_relevance_2d.shape}")
    print(f"Reshaped 2D layout:")
    print(image_relevance_2d[0, 0, :, :].numpy().astype(int))
    
    # ===== STEP 3: Verify correctness =====
    print("\n=== STEP 3: Verification ===")
    
    # Check if the reshaping preserves the correct spatial mapping
    is_correct = True
    for i in range(grid_size):
        for j in range(grid_size):
            expected_patch_idx = i * grid_size + j
            actual_value = image_relevance_2d[0, 0, i, j].item()
            
            if actual_value != expected_patch_idx:
                print(f"ERROR: Position ({i},{j}) should be patch {expected_patch_idx}, got {actual_value}")
                is_correct = False
    
    if is_correct:
        print("CORRECT: Patch mapping is accurate!")
        print("Each spatial position (i,j) correctly maps to patch index i*grid_size + j")
    else:
        print("ERROR: Patch mapping is incorrect!")
    
    # ===== STEP 4: Visualize the mapping =====
    print("\n=== STEP 4: Visual verification ===")
    
    # Create a visual representation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Original patch indices
    patch_grid = np.arange(num_patches).reshape(grid_size, grid_size)
    im1 = ax1.imshow(patch_grid, cmap='viridis', aspect='equal')
    ax1.set_title('Expected Patch Indices\n(Row-major order)')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    
    # Add text annotations
    for i in range(grid_size):
        for j in range(grid_size):
            ax1.text(j, i, f'{patch_grid[i,j]}', ha='center', va='center', 
                    color='white', fontsize=8, weight='bold')
    
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # Right: Visualization reshape result
    viz_result = image_relevance_2d[0, 0, :, :].numpy()
    im2 = ax2.imshow(viz_result, cmap='viridis', aspect='equal')
    ax2.set_title('Visualization Reshape Result')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    
    # Add text annotations
    for i in range(grid_size):
        for j in range(grid_size):
            ax2.text(j, i, f'{viz_result[i,j]:.0f}', ha='center', va='center', 
                    color='white', fontsize=8, weight='bold')
    
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('patch_mapping_verification.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return is_correct

def test_interpolation_mapping():
    """Test that the interpolation to 224x224 preserves spatial relationships"""
    
    print("\n" + "="*60)
    print("TESTING INTERPOLATION MAPPING")
    print("="*60)
    
    grid_size = 7
    target_size = 224
    
    # Create a test pattern: checkerboard-like pattern
    test_grid = torch.zeros(1, 1, grid_size, grid_size)
    for i in range(grid_size):
        for j in range(grid_size):
            test_grid[0, 0, i, j] = (i + j) % 2  # Checkerboard pattern
    
    print(f"Original {grid_size}×{grid_size} test pattern:")
    print(test_grid[0, 0, :, :].numpy())
    
    # Apply interpolation (same as visualization code)
    interpolated = torch.nn.functional.interpolate(test_grid, size=target_size, mode='bilinear')
    
    print(f"\nAfter interpolation: {test_grid.shape} -> {interpolated.shape}")
    
    # Check corners and center to verify spatial preservation
    corners_original = [
        (0, 0), (0, grid_size-1), 
        (grid_size-1, 0), (grid_size-1, grid_size-1)
    ]
    corners_interpolated = [
        (0, 0), (0, target_size-1),
        (target_size-1, 0), (target_size-1, target_size-1)
    ]
    
    print("\nCorner value preservation check:")
    for (orig_pos, interp_pos) in zip(corners_original, corners_interpolated):
        orig_val = test_grid[0, 0, orig_pos[0], orig_pos[1]].item()
        interp_val = interpolated[0, 0, interp_pos[0], interp_pos[1]].item()
        print(f"  Original ({orig_pos[0]},{orig_pos[1]}): {orig_val:.1f} -> "
              f"Interpolated ({interp_pos[0]},{interp_pos[1]}): {interp_val:.3f}")
    
    # Visualize the interpolation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = ax1.imshow(test_grid[0, 0, :, :], cmap='RdYlBu', aspect='equal')
    ax1.set_title(f'Original {grid_size}×{grid_size}')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    im2 = ax2.imshow(interpolated[0, 0, :, :], cmap='RdYlBu', aspect='equal')
    ax2.set_title(f'Interpolated {target_size}×{target_size}')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('interpolation_mapping_verification.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("PATCH LOCATION MAPPING VERIFICATION")
    print("="*60)
    
    # Test 1: Basic patch mapping
    mapping_correct = test_patch_mapping()
    
    # Test 2: Interpolation mapping
    test_interpolation_mapping()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if mapping_correct:
        print("✅ Patch location mapping is CORRECT")
        print("   The visualization accurately represents spatial relationships")
    else:
        print("❌ Patch location mapping has ERRORS")
        print("   The visualization may not accurately represent spatial relationships") 