import sys
import os

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

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

def analyze_similarity_distribution():
    """Analyze and visualize the similarity distribution to understand the max patch issue."""
    print("=== SIMILARITY DISTRIBUTION ANALYSIS ===")
    print("This will help you understand why max patches appear in weird places!\n")
    
    # Load model and data
    print("Loading model...")
    model = load_interpretable_clip("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
    
    image_path = r"D:/Wajahat Ali Khan/CLIP/images/dog.png"
    image = Image.open(image_path).convert("RGB")
    text = "a photo of a dog"
    
    print(f"Analyzing: {text}")
    
    # Get similarity data
    image_input = model.preprocess(image).unsqueeze(0)
    text_input = tokenize_text(text)
    tokens, similarity, eos_patch_sim, cls_token_sim, eos_token_sim, cls_patch_sim = model.get_token_patch_similarity(image_input, text_input, debug=False)
    
    # Convert to numpy
    if hasattr(similarity, 'detach'):
        similarity_np = similarity.detach().cpu().numpy()
    else:
        similarity_np = similarity
    
    grid_size = int(np.sqrt(similarity_np.shape[1]))
    print(f"Grid size: {grid_size}x{grid_size} = {similarity_np.shape[1]} patches")
    
    # Analyze each token
    for i, token in enumerate(tokens):
        print(f"\n{'='*50}")
        print(f"ANALYZING TOKEN: '{token}'")
        print(f"{'='*50}")
        
        token_similarities = similarity_np[i, :]
        
        # Basic statistics
        max_sim = np.max(token_similarities)
        min_sim = np.min(token_similarities)
        mean_sim = np.mean(token_similarities)
        std_sim = np.std(token_similarities)
        max_patch_idx = np.argmax(token_similarities)
        
        print(f"Max similarity: {max_sim:.4f} (patch {max_patch_idx})")
        print(f"Min similarity: {min_sim:.4f}")
        print(f"Mean similarity: {mean_sim:.4f}")
        print(f"Std deviation: {std_sim:.4f}")
        
        # Convert patch index to grid coordinates
        max_row = max_patch_idx // grid_size
        max_col = max_patch_idx % grid_size
        print(f"Max patch location: grid[{max_row}, {max_col}]")
        
        # Show top patches
        sorted_indices = np.argsort(token_similarities)[::-1]
        print(f"\nTop 10 patches:")
        for j in range(min(10, len(sorted_indices))):
            idx = sorted_indices[j]
            sim = token_similarities[idx]
            row = idx // grid_size
            col = idx % grid_size
            print(f"  {j+1}. Patch {idx} [grid {row},{col}]: {sim:.4f}")
        
        # Show similarity distribution
        print(f"\nSimilarity distribution:")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(token_similarities, p)
            print(f"  {p}th percentile: {val:.4f}")
        
        # Check if max is an outlier
        q3 = np.percentile(token_similarities, 75)
        q1 = np.percentile(token_similarities, 25)
        iqr = q3 - q1
        outlier_threshold = q3 + 1.5 * iqr
        
        if max_sim > outlier_threshold:
            print(f"\n⚠️  WARNING: Max similarity ({max_sim:.4f}) is an outlier!")
            print(f"   Outlier threshold: {outlier_threshold:.4f}")
            print(f"   This explains why the max patch appears isolated!")
        else:
            print(f"\n✓ Max similarity is not an outlier (threshold: {outlier_threshold:.4f})")
        
        # Create a simple visualization of the similarity map
        similarity_2d = token_similarities.reshape(grid_size, grid_size)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Similarity heatmap
        im1 = ax1.imshow(similarity_2d, cmap='viridis')
        ax1.set_title(f"Token '{token}' - Full Similarity Map")
        ax1.set_xlabel("Grid Column")
        ax1.set_ylabel("Grid Row")
        
        # Mark the max patch
        ax1.scatter([max_col], [max_row], color='red', s=100, marker='x', linewidth=3)
        ax1.text(max_col, max_row-0.5, f'MAX\n{max_sim:.3f}', 
                ha='center', va='bottom', color='red', fontweight='bold', fontsize=8)
        
        plt.colorbar(im1, ax=ax1, label='Similarity')
        
        # Histogram
        ax2.hist(token_similarities, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(max_sim, color='red', linestyle='--', linewidth=2, label=f'Max: {max_sim:.4f}')
        ax2.axvline(mean_sim, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_sim:.4f}')
        ax2.axvline(outlier_threshold, color='orange', linestyle='--', linewidth=2, label=f'Outlier threshold: {outlier_threshold:.4f}')
        ax2.set_xlabel('Similarity Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f"Token '{token}' - Similarity Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Ask user if they want to continue
        if i < len(tokens) - 1:
            response = input(f"\nPress Enter to continue to next token, or 'q' to quit: ")
            if response.lower() == 'q':
                break

if __name__ == "__main__":
    analyze_similarity_distribution() 