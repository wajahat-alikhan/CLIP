#DO NOT CHANGE OR MODIFY THIS

import sys
import os

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt


# Import from clip module
try:
    from clip.interpretable_clip import load_interpretable_clip, tokenize_text
except ImportError:
    # Fallback: try direct import
    import clip.interpretable_clip as interpretable_clip
    load_interpretable_clip = interpretable_clip.load_interpretable_clip
    tokenize_text = interpretable_clip.tokenize_text

def plot_comprehensive_similarity_matrix(similarity_np, eos_patch_sim_np, cls_token_sim_np, eos_token_sim_np, cls_patch_sim_np, tokens, grid_size, text, save_path=None):
    """Plot comprehensive similarity matrices showing all five types of similarities in separate windows."""
    import time
    
    print("Creating separate similarity matrix windows...")
    
    # Calculate global min/max across all similarity matrices for consistent scaling
    all_similarities = [
        similarity_np,
        eos_patch_sim_np,
        cls_token_sim_np,
        eos_token_sim_np,
        cls_patch_sim_np
    ]
    
    global_min = min(sim.min() for sim in all_similarities)
    global_max = max(sim.max() for sim in all_similarities)
    
    print(f"Using consistent colorbar scale: [{global_min:.3f}, {global_max:.3f}] across all matrices")
    
    # 1. Token-Patch Similarity
    plt.figure(figsize=(12, 8))
    im1 = plt.imshow(similarity_np, cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    plt.xlabel(f'Image Patches ({grid_size}×{grid_size} grid)', fontsize=12)
    plt.ylabel('Text Tokens', fontsize=12)
    plt.title(f'Token-Patch Similarities\nText: "{text}"\n(Fine-grained: each token vs each patch)', fontsize=14, pad=20)
    plt.yticks(range(len(tokens)), tokens, fontsize=10)
    
    # Set x-axis labels for patches
    patch_labels = [f'P{i}' for i in range(similarity_np.shape[1])]
    step = max(1, len(patch_labels) // 8)
    plt.xticks(range(0, len(patch_labels), step), 
              [patch_labels[i] for i in range(0, len(patch_labels), step)], rotation=45, fontsize=9)
    
    cbar1 = plt.colorbar(im1, shrink=0.8)
    cbar1.set_label(f'Cosine Similarity [{global_min:.3f}, {global_max:.3f}]', rotation=270, labelpad=20)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}_token_patch_similarity.png", dpi=300, bbox_inches='tight')
        print(f"Token-Patch similarity saved to {save_path}_token_patch_similarity.png")
    
    plt.show(block=False)
    time.sleep(0.5)  # Small delay to ensure window renders
    
    # 2. EOS-Patch Similarity
    plt.figure(figsize=(12, 6))
    im2 = plt.imshow(eos_patch_sim_np, cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    plt.xlabel(f'Image Patches ({grid_size}×{grid_size} grid)', fontsize=12)
    plt.ylabel('EOS Token', fontsize=12)
    plt.title(f'EOS-Patch Similarities\nText: "{text}"\n(Global text vs each patch)', fontsize=14, pad=20)
    plt.yticks([0], ['[EOS]'], fontsize=10)
    plt.xticks(range(0, len(patch_labels), step), 
              [patch_labels[i] for i in range(0, len(patch_labels), step)], rotation=45, fontsize=9)
    
    cbar2 = plt.colorbar(im2, shrink=0.8)
    cbar2.set_label(f'Cosine Similarity [{global_min:.3f}, {global_max:.3f}]', rotation=270, labelpad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_eos_patch_similarity.png", dpi=300, bbox_inches='tight')
        print(f"EOS-Patch similarity saved to {save_path}_eos_patch_similarity.png")
    
    plt.show(block=False)
    time.sleep(0.5)
    
    # 3. CLS-Token Similarity
    plt.figure(figsize=(10, 6))
    im3 = plt.imshow(cls_token_sim_np, cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    plt.xlabel('Text Tokens', fontsize=12)
    plt.ylabel('CLS Token', fontsize=12)
    plt.title(f'CLS-Token Similarities\nText: "{text}"\n(Global image vs each token)', fontsize=14, pad=20)
    plt.yticks([0], ['[CLS]'], fontsize=10)
    plt.xticks(range(len(tokens)), tokens, rotation=45, fontsize=10)
    
    cbar3 = plt.colorbar(im3, shrink=0.8)
    cbar3.set_label(f'Cosine Similarity [{global_min:.3f}, {global_max:.3f}]', rotation=270, labelpad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_cls_token_similarity.png", dpi=300, bbox_inches='tight')
        print(f"CLS-Token similarity saved to {save_path}_cls_token_similarity.png")
    
    plt.show(block=False)
    time.sleep(0.5)
    
    # 4. EOS-Token Similarity
    plt.figure(figsize=(10, 6))
    im4 = plt.imshow(eos_token_sim_np, cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    plt.xlabel('Text Tokens', fontsize=12)
    plt.ylabel('EOS Token', fontsize=12)
    plt.title(f'EOS-Token Similarities\nText: "{text}"\n(Global text vs each token)', fontsize=14, pad=20)
    plt.yticks([0], ['[EOS]'], fontsize=10)
    plt.xticks(range(len(tokens)), tokens, rotation=45, fontsize=10)
    
    cbar4 = plt.colorbar(im4, shrink=0.8)
    cbar4.set_label(f'Cosine Similarity [{global_min:.3f}, {global_max:.3f}]', rotation=270, labelpad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_eos_token_similarity.png", dpi=300, bbox_inches='tight')
        print(f"EOS-Token similarity saved to {save_path}_eos_token_similarity.png")
    
    plt.show(block=False)
    time.sleep(0.5)
    
    # 5. CLS-Patch Similarity
    plt.figure(figsize=(12, 6))
    im5 = plt.imshow(cls_patch_sim_np, cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    plt.xlabel(f'Image Patches ({grid_size}×{grid_size} grid)', fontsize=12)
    plt.ylabel('CLS Token', fontsize=12)
    plt.title(f'CLS-Patch Similarities\nText: "{text}"\n(Global image vs each patch)', fontsize=14, pad=20)
    plt.yticks([0], ['[CLS]'], fontsize=10)
    
    # Set x-axis labels for patches
    patch_labels_cls = [f'P{i}' for i in range(cls_patch_sim_np.shape[1])]
    step_cls = max(1, len(patch_labels_cls) // 8)
    plt.xticks(range(0, len(patch_labels_cls), step_cls), 
              [patch_labels_cls[i] for i in range(0, len(patch_labels_cls), step_cls)], rotation=45, fontsize=9)
    
    cbar5 = plt.colorbar(im5, shrink=0.8)
    cbar5.set_label(f'Cosine Similarity [{global_min:.3f}, {global_max:.3f}]', rotation=270, labelpad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_cls_patch_similarity.png", dpi=300, bbox_inches='tight')
        print(f"CLS-Patch similarity saved to {save_path}_cls_patch_similarity.png")
    
    plt.show(block=False)
    time.sleep(0.5)
    
    # 6. Combined Overview Matrix
    plt.figure(figsize=(14, 10))
    
    # Create combined matrix with EOS and CLS tokens
    # Add EOS row at top and CLS column at left
    combined_matrix = np.zeros((len(tokens) + 1, similarity_np.shape[1] + 1))
    
    # Original token-patch similarities
    combined_matrix[1:, 1:] = similarity_np
    
    # EOS-patch similarities (top row, excluding top-left corner)
    combined_matrix[0, 1:] = eos_patch_sim_np[0]
    
    # CLS-token similarities (left column, excluding top-left corner)  
    combined_matrix[1:, 0] = cls_token_sim_np[0]
    
    # EOS-CLS similarity (top-left corner) - can compute if needed
    combined_matrix[0, 0] = 0.5  # placeholder
    
    im6 = plt.imshow(combined_matrix, cmap='viridis', aspect='auto', vmin=global_min, vmax=global_max)
    plt.xlabel('Patches + [CLS]', fontsize=12)
    plt.ylabel('[EOS] + Tokens', fontsize=12)
    plt.title(f'Combined Similarity Matrix\nText: "{text}"\n(All embeddings)', fontsize=14, pad=20)
    
    # Set labels for combined matrix
    y_labels = ['[EOS]'] + tokens
    x_labels = ['[CLS]'] + [f'P{i}' for i in range(similarity_np.shape[1])]
    
    plt.yticks(range(len(y_labels)), y_labels, fontsize=10)
    
    step_x = max(1, len(x_labels) // 10)
    plt.xticks(range(0, len(x_labels), step_x), 
              [x_labels[i] for i in range(0, len(x_labels), step_x)], rotation=45, fontsize=9)
    
    cbar6 = plt.colorbar(im6, shrink=0.8)
    cbar6.set_label(f'Cosine Similarity [{global_min:.3f}, {global_max:.3f}]', rotation=270, labelpad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_combined_similarity.png", dpi=300, bbox_inches='tight')
        print(f"Combined similarity matrix saved to {save_path}_combined_similarity.png")
    
    plt.show(block=False)
    time.sleep(0.5)
    
    # Print summary statistics to console
    print("\n" + "="*60)
    print("SIMILARITY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Text: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Grid Size: {grid_size}×{grid_size} = {grid_size**2} patches")
    print()
    print("Matrix Shapes:")
    print(f"  • Token-Patch: {similarity_np.shape}")
    print(f"  • EOS-Patch: {eos_patch_sim_np.shape}")
    print(f"  • CLS-Token: {cls_token_sim_np.shape}")
    print(f"  • EOS-Token: {eos_token_sim_np.shape}")
    print(f"  • CLS-Patch: {cls_patch_sim_np.shape}")
    print()
    print("Individual Similarity Ranges:")
    print(f"  • Token-Patch: [{similarity_np.min():.3f}, {similarity_np.max():.3f}]")
    print(f"  • EOS-Patch: [{eos_patch_sim_np.min():.3f}, {eos_patch_sim_np.max():.3f}]")
    print(f"  • CLS-Token: [{cls_token_sim_np.min():.3f}, {cls_token_sim_np.max():.3f}]")
    print(f"  • EOS-Token: [{eos_token_sim_np.min():.3f}, {eos_token_sim_np.max():.3f}]")
    print(f"  • CLS-Patch: [{cls_patch_sim_np.min():.3f}, {cls_patch_sim_np.max():.3f}]")
    print()
    print(f"Consistent Colorbar Scale: [{global_min:.3f}, {global_max:.3f}] (applied to all matrices)")
    print("="*60)
    
    print("\n✅ All 6 similarity matrix windows created successfully!")
    print("🎯 All matrices use the SAME colorbar scale for easy comparison!")
    print("You can now view all matrices simultaneously without closing any windows.")
    
    # Keep all windows open until manually closed
    input("\nPress Enter to close all windows...")

def plot_similarity_matrix(similarity_np, tokens, grid_size, text, save_path=None):
    """Plot confusion matrix showing token-patch similarities (legacy function)."""
    plt.figure(figsize=(12, 8))
    
    # Create the heatmap
    im = plt.imshow(similarity_np, cmap='viridis', aspect='auto')
    
    # Set labels
    plt.xlabel(f'Image Patches ({grid_size}×{grid_size} grid)', fontsize=12)
    plt.ylabel('Text Tokens', fontsize=12)
    plt.title(f'Token-Patch Similarity Matrix\nText: "{text}"', fontsize=14, pad=20)
    
    # Set y-axis labels to token names
    plt.yticks(range(len(tokens)), tokens, fontsize=10)
    
    # Set x-axis labels (show every 5th patch)
    patch_labels = [f'P{i}' for i in range(similarity_np.shape[1])]
    step = max(1, len(patch_labels) // 10)  # Show ~10 labels max
    plt.xticks(range(0, len(patch_labels), step), 
              [patch_labels[i] for i in range(0, len(patch_labels), step)], 
              rotation=45, fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}_similarity_matrix.png", dpi=300, bbox_inches='tight')
        print(f"Similarity matrix saved to {save_path}_similarity_matrix.png")
    
    plt.show()



def main():
    # --- Setup ---
    print("INTERPRETABLE CLIP CONFUSION MATRICES")
    
    print("Loading interpretable CLIP model...")
    model = load_interpretable_clip("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Load image and set text prompt
    image_path = r"D:/Wajahat Ali Khan/CLIP/images/dogcat2.png"
    image = Image.open(image_path).convert("RGB")
    text = "dog"
    
    print(f"\nAnalyzing image: {image_path}")
    print(f"Text prompt: '{text}'")
    
    # --- Get similarity data ---
    print("\nComputing token-patch similarities...")
    image_input = model.preprocess(image).unsqueeze(0)
    text_input = tokenize_text(text)
    tokens, similarity, eos_patch_sim, cls_token_sim, eos_token_sim, cls_patch_sim = model.get_token_patch_similarity(image_input, text_input, debug=False)
    
    # Convert to numpy for visualization
    if hasattr(similarity, 'detach'):
        similarity_np = similarity.detach().cpu().numpy()
        eos_patch_sim_np = eos_patch_sim.detach().cpu().numpy()
        cls_token_sim_np = cls_token_sim.detach().cpu().numpy()
        eos_token_sim_np = eos_token_sim.detach().cpu().numpy()
        cls_patch_sim_np = cls_patch_sim.detach().cpu().numpy()
    else:
        similarity_np = similarity
        eos_patch_sim_np = eos_patch_sim
        cls_token_sim_np = cls_token_sim
        eos_token_sim_np = eos_token_sim
        cls_patch_sim_np = cls_patch_sim
        
    grid_size = int(np.sqrt(similarity_np.shape[1]))
    
    # Print basic info
    print(f"\nResults:")
    print(f"  Tokens found: {tokens}")
    print(f"  Grid size: {grid_size}×{grid_size} = {similarity_np.shape[1]} patches")
    print(f"  Token-Patch similarity range: [{similarity_np.min():.4f}, {similarity_np.max():.4f}]")
    print(f"  EOS-Patch similarity range: [{eos_patch_sim_np.min():.4f}, {eos_patch_sim_np.max():.4f}]")
    print(f"  CLS-Token similarity range: [{cls_token_sim_np.min():.4f}, {cls_token_sim_np.max():.4f}]")
    print(f"  EOS-Token similarity range: [{eos_token_sim_np.min():.4f}, {eos_token_sim_np.max():.4f}]")
    print(f"  CLS-Patch similarity range: [{cls_patch_sim_np.min():.4f}, {cls_patch_sim_np.max():.4f}]")
    
    print(f"\n  Similarity Matrix Shapes:")
    print(f"    Token-Patch: {similarity_np.shape} (each token vs each patch)")
    print(f"    EOS-Patch: {eos_patch_sim_np.shape} (global text vs each patch)")
    print(f"    CLS-Token: {cls_token_sim_np.shape} (global image vs each token)")
    print(f"    EOS-Token: {eos_token_sim_np.shape} (global text vs each token)")
    print(f"    CLS-Patch: {cls_patch_sim_np.shape} (global image vs each patch)")
    
    # Find most relevant tokens
    max_sims = np.max(similarity_np, axis=1)
    sorted_indices = np.argsort(max_sims)[::-1]
    
    print(f"\n  Most relevant tokens:")
    for i in sorted_indices[:3]:
        token = tokens[i]
        peak_sim = max_sims[i]
        peak_patch = np.argmax(similarity_np[i, :])
        patch_row, patch_col = divmod(peak_patch, grid_size)
        print(f"    '{token}': {peak_sim:.4f} at patch ({patch_row}, {patch_col})")
    
    # Analyze EOS-Patch similarities (global text vs patches)
    print(f"\n  EOS token (global text) - Top patches:")
    eos_top_patches = np.argsort(eos_patch_sim_np[0])[::-1][:3]
    for patch_idx in eos_top_patches:
        patch_row, patch_col = divmod(patch_idx, grid_size)
        sim_value = eos_patch_sim_np[0, patch_idx]
        print(f"    Patch ({patch_row}, {patch_col}): {sim_value:.4f}")
    
    # Analyze CLS-Token similarities (global image vs tokens)
    print(f"\n  CLS token (global image) - Top text tokens:")
    cls_top_tokens = np.argsort(cls_token_sim_np[0])[::-1][:3]
    for token_idx in cls_top_tokens:
        if token_idx < len(tokens):
            token = tokens[token_idx]
            sim_value = cls_token_sim_np[0, token_idx]
            print(f"    '{token}': {sim_value:.4f}")
    
    # Analyze EOS-Token similarities (global text vs tokens)
    print(f"\n  EOS token (global text) - Top text tokens:")
    eos_top_tokens = np.argsort(eos_token_sim_np[0])[::-1][:3]
    for token_idx in eos_top_tokens:
        if token_idx < len(tokens):
            token = tokens[token_idx]
            sim_value = eos_token_sim_np[0, token_idx]
            print(f"    '{token}': {sim_value:.4f}")
    
    # Analyze CLS-Patch similarities (global image vs patches)
    print(f"\n  CLS token (global image) - Top patches:")
    cls_top_patches = np.argsort(cls_patch_sim_np[0])[::-1][:3]
    for patch_idx in cls_top_patches:
        patch_row, patch_col = divmod(patch_idx, grid_size)
        sim_value = cls_patch_sim_np[0, patch_idx]
        print(f"    Patch ({patch_row}, {patch_col}): {sim_value:.4f}")
    
    # --- CONFUSION MATRIX VISUALIZATIONS ---
    print("\nGENERATING CONFUSION MATRIX VISUALIZATIONS")
    
    # 1. All Similarity Matrices (Separate Windows)
    print("\n1. Creating Individual Similarity Matrix Windows:")
    plot_comprehensive_similarity_matrix(similarity_np, eos_patch_sim_np, cls_token_sim_np, eos_token_sim_np, cls_patch_sim_np, tokens, grid_size, text, save_path="results/confusion_matrix_analysis")
    
    print("\nCONFUSION MATRIX VISUALIZATION COMPLETE!")
   
    print("Confusion matrix analysis finished:")
    print("- 6 Separate Similarity Matrix Windows: Each matrix in its own window for easy comparison")
    print("- Token-Patch Similarity: Individual text tokens vs image patches (fine-grained)")
    print("- EOS-Patch Similarity: Global text meaning vs image patches (text-to-local)")
    print("- CLS-Token Similarity: Global image meaning vs text tokens (image-to-local)")
    print("- EOS-Token Similarity: Global text meaning vs text tokens (global-to-local text)")
    print("- CLS-Patch Similarity: Global image meaning vs image patches (global-to-local image)")
    print("- Combined Matrix: All embeddings + [EOS] + [CLS] in one comprehensive view")


if __name__ == "__main__":
    main() 

# DONOT CHANGE OR MODIFY THIS