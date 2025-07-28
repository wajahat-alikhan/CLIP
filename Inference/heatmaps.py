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
import cv2

# Import scipy for spatial smoothing
try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None
    print("Warning: scipy not available. Spatial smoothing will be disabled.")

# Import from clip module
try:
    from clip.interpretable_clip import load_interpretable_clip, tokenize_text
except ImportError:
    # Fallback: try direct import
    import clip.interpretable_clip as interpretable_clip
    load_interpretable_clip = interpretable_clip.load_interpretable_clip
    tokenize_text = interpretable_clip.tokenize_text

def is_stop_word(token):
    """Check if a token is a stop word that should be filtered from visualization."""
    # Clean the token (remove special characters, make lowercase)
    clean_token = token.lower().replace('<', '').replace('>', '').replace('|', '').replace('</w>', '').strip()
    
    # List of common stop words to filter out (keeping meaningful words like photo, image, picture)
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
        'by', 'from', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 
        'might', 'must', 'can', 'shall', 'this', 'that', 'these', 'those', 'i', 'you', 
        'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 
        'his', 'hers', 'its', 'our', 'their', '<|startoftext|>', '<|endoftext|>', '</w>', 
        '<unk>'
    }
    
    return clean_token in stop_words

def show_image_relevance(image_relevance, image, orig_image, token_name):
    """Create heatmap from mask on image using OpenCV JET colormap"""
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show original image
    axs[0].imshow(orig_image)
    axs[0].axis('off')
    axs[0].set_title('Original Image', fontsize=12)

    # Process relevance map
    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224)
    
    # Move to CPU if on CUDA
    if image_relevance.is_cuda:
        image_relevance = image_relevance.cpu()
    image_relevance = image_relevance.data.numpy()
    
    # Normalize relevance map
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    
    # Process image
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    
    # Create heatmap visualization
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)  # Convert back to RGB for matplotlib
    
    axs[1].imshow(vis)
    axs[1].axis('off')
    axs[1].set_title(f"Token: '{token_name}' Heatmap", fontsize=12)
    
    plt.tight_layout()
    return fig

def plot_top_similarity_heatmaps(similarity_np, tokens, image_tensor, orig_image, text, save_path=None, top_percent=0.1):
    """Plot heatmaps showing the top similarity patches for each token (more meaningful than single max)."""
    print(f"\nGenerating TOP {top_percent*100:.0f}% SIMILARITY heatmaps for {len(tokens)} tokens...")
    
    for i, token in enumerate(tokens):
        # Get all similarities for this token
        token_similarities = similarity_np[i, :]
        
        # Find threshold for top N% of patches
        threshold = np.percentile(token_similarities, (1-top_percent)*100)
        
        # Create mask that preserves original similarity values for top patches
        top_patch_mask = np.where(token_similarities >= threshold, token_similarities, 0)
        
        max_similarity = np.max(token_similarities)
        num_top_patches = np.sum(top_patch_mask > 0)
        
        print(f"Processing TOP similarity for token {i+1}/{len(tokens)}: '{token}' (max: {max_similarity:.4f}, showing top {num_top_patches} patches)")
        
        # Convert to tensor for visualization
        token_relevance = torch.tensor(top_patch_mask)
        
        # Create heatmap visualization
        fig = show_image_relevance(token_relevance, image_tensor, orig_image, f"{token} (Top {top_percent*100:.0f}%)")
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            token_safe = token.replace('<', '').replace('>', '').replace('|', '_')
            save_file = f"{save_path}_token_{token_safe}_top_patches_heatmap.png"
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_file}")
        
        plt.show()
        plt.close(fig)  # Close figure to free memory

def plot_token_heatmaps(similarity_np, tokens, image_tensor, orig_image, text, save_path=None):
    """Plot heatmaps for each token using the new visualization style."""
    print(f"\nGenerating heatmaps for {len(tokens)} tokens...")
    
    for i, token in enumerate(tokens):
        print(f"Processing token {i+1}/{len(tokens)}: '{token}'")
        
        # Get attention for this token
        token_relevance = torch.tensor(similarity_np[i, :])
        
        # Create heatmap visualization
        fig = show_image_relevance(token_relevance, image_tensor, orig_image, token)
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            token_safe = token.replace('<', '').replace('>', '').replace('|', '_')
            save_file = f"{save_path}_token_{token_safe}_heatmap.png"
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_file}")
        
        plt.show()
        plt.close(fig)  # Close figure to free memory

def analyze_patch_to_token_assignments(similarity_np, tokens, top_k=3):
    """Analyze which top K tokens each image patch has highest similarity with."""
    print(f"\n=== BIJECTIVE ANALYSIS: PATCH-TO-TOP-{top_k}-TOKENS ASSIGNMENTS ===")
    
    num_tokens, num_patches = similarity_np.shape
    grid_size = int(np.sqrt(num_patches))
    
    print(f"Grid size: {grid_size}×{grid_size} = {num_patches} patches")
    print(f"Analyzing top {top_k} tokens for each patch...")
    
    # Find top K tokens for each patch
    top_k_token_indices = np.argsort(similarity_np, axis=0)[-top_k:]  # Shape: [top_k, num_patches]
    top_k_similarities = np.sort(similarity_np, axis=0)[-top_k:]      # Shape: [top_k, num_patches]
    
    # Create statistics for each token - how many patches have it in their top K
    token_top_k_counts = {token: 0 for token in tokens}
    token_rank_1_counts = {token: 0 for token in tokens}
    
    for patch_idx in range(num_patches):
        # Get top K tokens for this patch (highest rank first)
        patch_top_k_indices = top_k_token_indices[:, patch_idx][::-1]  # Reverse to get highest first
        patch_top_k_sims = top_k_similarities[:, patch_idx][::-1]
        
        # Count rank 1 (best token for this patch)
        best_token = tokens[patch_top_k_indices[0]]
        token_rank_1_counts[best_token] += 1
        
        # Count all top K appearances
        for token_idx in patch_top_k_indices:
            token = tokens[token_idx]
            token_top_k_counts[token] += 1
    
    print(f"\nToken statistics (how many patches have each token in top {top_k}):")
    for token, count in sorted(token_top_k_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / num_patches) * 100
        rank_1_count = token_rank_1_counts[token]
        rank_1_percentage = (rank_1_count / num_patches) * 100
        print(f"  '{token}': {count}/{num_patches} patches ({percentage:.1f}%) - Rank #1 in {rank_1_count} patches ({rank_1_percentage:.1f}%)")
    
    # Show detailed analysis for first few patches
    print(f"\nDetailed patch-to-top-{top_k}-tokens assignments (first 10 patches):")
    for patch_idx in range(min(10, num_patches)):
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        print(f"  Patch {patch_idx} [grid {row},{col}]:")
        
        # Get top K tokens for this patch
        patch_top_k_indices = top_k_token_indices[:, patch_idx][::-1]  # Reverse to get highest first
        patch_top_k_sims = top_k_similarities[:, patch_idx][::-1]
        
        for rank, (token_idx, similarity) in enumerate(zip(patch_top_k_indices, patch_top_k_sims), 1):
            token = tokens[token_idx]
            print(f"    #{rank}: '{token}' (sim: {similarity:.4f})")
    
    if num_patches > 10:
        print(f"  ... and {num_patches - 10} more patches")
    
    return top_k_token_indices, top_k_similarities, token_top_k_counts, token_rank_1_counts

def visualize_patch_token_assignments(similarity_np, tokens, image_tensor, orig_image, save_path=None, top_k=3):
    """Visualize which top K tokens each patch is most similar to."""
    print(f"\n=== VISUALIZING PATCH-TO-TOP-{top_k}-TOKENS ASSIGNMENTS ===")
    
    # Get patch assignments
    top_k_token_indices = np.argsort(similarity_np, axis=0)[-top_k:]  # Shape: [top_k, num_patches]
    top_k_similarities = np.sort(similarity_np, axis=0)[-top_k:]      # Shape: [top_k, num_patches]
    
    grid_size = int(np.sqrt(similarity_np.shape[1]))
    
    # Create a color map for tokens
    num_tokens = len(tokens)
    colors = plt.cm.Set3(np.linspace(0, 1, num_tokens))  # Use Set3 colormap for distinct colors
    
    # Prepare the image
    image_np = image_tensor[0].permute(1, 2, 0).cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # 1. Original image
    axes[0].imshow(orig_image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # 2. Best token assignment (rank 1)
    axes[1].imshow(image_np)
    patch_height = 224 // grid_size
    patch_width = 224 // grid_size
    
    for patch_idx in range(similarity_np.shape[1]):
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        
        # Get best (rank 1) token for this patch
        best_token_idx = top_k_token_indices[-1, patch_idx]  # Last element is highest
        color = colors[best_token_idx]
        
        # Draw colored overlay
        y_start = row * patch_height
        y_end = (row + 1) * patch_height
        x_start = col * patch_width
        x_end = (col + 1) * patch_width
        
        alpha = 0.4
        overlay = np.zeros((patch_height, patch_width, 4))
        overlay[:, :, :3] = color[:3]  # RGB
        overlay[:, :, 3] = alpha      # Alpha
        
        axes[1].imshow(overlay, extent=[x_start, x_end, y_end, y_start])
    
    axes[1].set_title('Rank #1 Token per Patch\n(Best matching token)', fontsize=14)
    axes[1].axis('off')
    
    # Add grid lines
    for i in range(grid_size + 1):
        axes[1].axhline(y=i * patch_height, color='white', linewidth=0.5, alpha=0.7)
        axes[1].axvline(x=i * patch_width, color='white', linewidth=0.5, alpha=0.7)
    
    # 3. Top K tokens visualization with borders
    axes[2].imshow(image_np)
    
    for patch_idx in range(similarity_np.shape[1]):
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        
        y_start = row * patch_height
        y_end = (row + 1) * patch_height
        x_start = col * patch_width
        x_end = (col + 1) * patch_width
        
        # Draw borders for top K tokens (different thickness for different ranks)
        patch_top_k_indices = top_k_token_indices[:, patch_idx][::-1]  # Reverse to get highest rank first
        
        for rank, token_idx in enumerate(patch_top_k_indices):
            color = colors[token_idx]
            border_width = 3 - rank  # Rank 1 gets width 3, rank 2 gets width 2, rank 3 gets width 1
            
            # Draw border around patch
            from matplotlib.patches import Rectangle
            rect = Rectangle((x_start, y_start), patch_width, patch_height, 
                           linewidth=border_width, edgecolor=color, facecolor='none', alpha=0.8)
            axes[2].add_patch(rect)
    
    axes[2].set_title(f'Top {top_k} Tokens per Patch\n(Colored borders: thick=rank1, thin=rank{top_k})', fontsize=14)
    axes[2].axis('off')
    
    # 4. Legend showing token colors
    axes[3].axis('off')
    legend_elements = []
    for i, token in enumerate(tokens):
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], label=f"'{token}'"))
    
    axes[3].legend(handles=legend_elements, loc='center', fontsize=12, title='Token Colors')
    axes[3].set_title('Token Legend', fontsize=14)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_file = f"{save_path}_patch_to_top{top_k}_token_assignments.png"
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_file}")
    
    plt.show()
    plt.close(fig)

def create_patch_token_similarity_matrix(similarity_np, tokens, save_path=None):
    """Create a detailed similarity matrix visualization showing all patch-token relationships."""
    print(f"\n=== CREATING DETAILED PATCH-TOKEN SIMILARITY MATRIX ===")
    
    num_tokens, num_patches = similarity_np.shape
    grid_size = int(np.sqrt(num_patches))
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create heatmap
    im = ax.imshow(similarity_np, cmap='viridis', aspect='auto')
    
    # Set labels
    ax.set_xlabel('Image Patches', fontsize=12)
    ax.set_ylabel('Text Tokens', fontsize=12)
    ax.set_title('Complete Token-Patch Similarity Matrix\n(Rows=Tokens, Columns=Patches)', fontsize=14)
    
    # Set token labels on y-axis
    ax.set_yticks(range(num_tokens))
    ax.set_yticklabels([f"'{token}'" for token in tokens])
    
    # Set patch labels on x-axis (show every 10th patch to avoid crowding)
    patch_tick_interval = max(1, num_patches // 20)
    patch_ticks = range(0, num_patches, patch_tick_interval)
    ax.set_xticks(patch_ticks)
    patch_labels = []
    for patch_idx in patch_ticks:
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        patch_labels.append(f"{patch_idx}\n[{row},{col}]")
    ax.set_xticklabels(patch_labels, rotation=45, ha='right', fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Similarity Score', fontsize=12)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_file = f"{save_path}_complete_similarity_matrix.png"
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_file}")
    
    plt.show()
    plt.close(fig)

def visualize_patch_similarity_grid(similarity_np, tokens, image_tensor, orig_image, save_path=None):
    """Visualize the top 1 patch for each token with color-coded overlay on image and text."""
    print(f"\n=== TOKEN-TO-PATCH MAPPING VISUALIZATION ===")
    print("Showing top 1 patch for each token with color-coded overlay")
    
    grid_size = int(np.sqrt(similarity_np.shape[1]))
    
    # Initialize grids
    token_assignment_grid = np.full((grid_size, grid_size), -1, dtype=int)  # -1 means unassigned
    similarity_values_grid = np.zeros((grid_size, grid_size))
    
    # Create color map for tokens
    num_tokens = len(tokens)
    colors = plt.cm.Set3(np.linspace(0, 1, num_tokens))
    
    # For each token, find its top 1 patch
    token_patch_info = []  # Store (token, patch_row, patch_col, similarity) for each token
    
    print(f"\nToken assignments (top 1 patch per token):")
    for token_idx, token in enumerate(tokens):
        # Get similarities for this token across all patches
        token_similarities = similarity_np[token_idx, :]
        
        # Find top 1 patch for this token
        best_patch_idx = np.argmax(token_similarities)
        best_similarity = token_similarities[best_patch_idx]
        
        row = best_patch_idx // grid_size
        col = best_patch_idx % grid_size
        
        # Always assign this patch to this token (even if conflict - we'll handle it)
        token_assignment_grid[row, col] = token_idx
        similarity_values_grid[row, col] = best_similarity
        
        token_patch_info.append((token, row, col, best_similarity))
        print(f"  Token '{token}': Patch [{row},{col}] = {best_similarity:.4f}")
    
    # Filter out stop words from visualization
    print(f"\nFiltering stop words from visualization...")
    meaningful_token_patch_info = [(token, row, col, similarity, i) for i, (token, row, col, similarity) in enumerate(token_patch_info) if not is_stop_word(token)]
    print(f"Filtered tokens for display: {len(token_patch_info)} → {len(meaningful_token_patch_info)}")
    print(f"Showing meaningful tokens: {[token for token, _, _, _, _ in meaningful_token_patch_info]}")
    print(f"Filtered out: {[token for token, _, _, _ in token_patch_info if is_stop_word(token)]}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Add main title
    fig.suptitle('TOKEN-TO-PATCH MAPPING VISUALIZATION', fontsize=10, weight='bold', y=0.95)
    
    # 1. Original image with clean colored overlay (no text on top)
    axes = []
    axes.append(plt.subplot2grid((3, 2), (0, 0), fig=fig))
    axes[0].imshow(orig_image)
    
    # Calculate patch dimensions for overlay
    img_height, img_width = orig_image.size[1], orig_image.size[0]
    patch_height = img_height // grid_size
    patch_width = img_width // grid_size
    
    # Overlay patches with their token colors (only for meaningful tokens)
    for token, row, col, similarity, original_token_idx in meaningful_token_patch_info:
        # Calculate patch boundaries in image coordinates
        x_start = col * patch_width
        x_end = (col + 1) * patch_width
        y_start = row * patch_height
        y_end = (row + 1) * patch_height
        
        # Create colored rectangle for this patch (much more visible)
        color = colors[original_token_idx][:3]  # Use original index to maintain color consistency
        from matplotlib.patches import Rectangle
        rect = Rectangle((x_start, y_start), patch_width, patch_height, 
                        linewidth=6, edgecolor=color, facecolor=color, alpha=0.7)
        axes[0].add_patch(rect)
    
    # Add visible grid lines to show patch boundaries clearly
    # Vertical lines
    for i in range(grid_size + 1):
        x_pos = i * patch_width
        axes[0].axvline(x=x_pos, color='white', linewidth=2, alpha=0.8)
    
    # Horizontal lines
    for i in range(grid_size + 1):
        y_pos = i * patch_height
        axes[0].axhline(y=y_pos, color='white', linewidth=2, alpha=0.8)
    
    # Remove title for clean look
    axes[0].axis('off')
    
    # 1.5. Color-coded text caption directly below the image (no headers)
    axes.append(plt.subplot2grid((3, 2), (1, 0), fig=fig))
    axes[1].axis('off')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    
    # Create a natural text caption with colored backgrounds (no "TEXT PROMPT:" header)
    # Use the already filtered meaningful tokens
    
    # Build the complete text string from meaningful tokens only
    full_text = " ".join([token for token, _, _, _, _ in meaningful_token_patch_info])
    
    # Calculate positions for each meaningful token in the natural text flow
    words = [token for token, _, _, _, _ in meaningful_token_patch_info]
    
    # Create the text with colored segments positioned higher (closer to image)
    # Start position for the text
    start_x = 0.5 - len(full_text) * 0.008  # Rough centering based on character count
    
    current_x = start_x
    y_pos = 0.7  # Position higher to be closer to image
    
    for token, row, col, similarity, original_i in meaningful_token_patch_info:
        color = colors[original_i][:3]  # Use original color index to maintain consistency
        
        # Add the token with colored background
        axes[1].text(current_x, y_pos, token, ha='left', va='center', 
                    fontsize=14, weight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8, 
                             edgecolor='none'),
                    transform=axes[1].transAxes)
        
        # Move position for next word (approximate character width + space)
        current_x += len(token) * 0.016 + 0.03  # Adjust multiplier for natural spacing
        
        # Add space between words (except for last word)
        if token != meaningful_token_patch_info[-1][0]:  # If not the last meaningful token
            current_x += 0.01
    
    # 2. Token details with patch coordinates
    axes.append(plt.subplot2grid((3, 2), (0, 1), rowspan=2, fig=fig))
    axes[2].axis('off')
    axes[2].set_xlim(0, 10)
    axes[2].set_ylim(0, num_tokens + 1)
    
    # Display tokens with their details
    axes[2].text(5, num_tokens, 'TOKEN DETAILS', ha='center', va='center', 
                  fontsize=16, weight='bold')
    
    for i, (token, row, col, similarity) in enumerate(token_patch_info):
        color = colors[i][:3]
        y_pos = num_tokens - 1 - i
        
        # Create colored background for token (darker)
        axes[2].text(3, y_pos, f"'{token}'", ha='center', va='center', 
                       fontsize=14, weight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.9, 
                                edgecolor='black', linewidth=2))
        
        # Add patch coordinates and similarity
        axes[2].text(7, y_pos, f'Patch [{row},{col}]\nSimilarity: {similarity:.3f}', ha='center', va='center', 
                       fontsize=11, weight='bold')
    
    axes[2].set_title('Token-Patch Assignments\n(Detailed information)', fontsize=14)
    
    # 3. Grid visualization of assignments
    # Create colored visualization
    token_colors_grid = np.ones((grid_size, grid_size, 3)) * 0.9  # Light gray background
    
    for token_idx, (token, row, col, similarity) in enumerate(token_patch_info):
        token_colors_grid[row, col] = colors[token_idx][:3]
    
    axes.append(plt.subplot2grid((3, 2), (2, 0), fig=fig))
    axes[3].imshow(token_colors_grid)
    axes[3].set_title('Token-Patch Grid Assignment\n(Each token gets 1 patch)', fontsize=14)
    
    # Add token names and similarities in each assigned patch
    for token_idx, (token, row, col, similarity) in enumerate(token_patch_info):
        # Truncate long token names
        display_name = token if len(token) <= 6 else token[:4] + '..'
        axes[3].text(col, row, f'{display_name}\n{similarity:.3f}', 
                       ha='center', va='center', color='black', fontsize=9, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    # Add grid lines
    for i in range(grid_size + 1):
        axes[3].axhline(y=i-0.5, color='black', linewidth=0.5, alpha=0.3)
        axes[3].axvline(x=i-0.5, color='black', linewidth=0.5, alpha=0.3)
    
    axes[3].set_xlabel('Patch Column')
    axes[3].set_ylabel('Patch Row')
    
    # 4. Similarity values heatmap
    masked_similarities = np.ma.masked_where(token_assignment_grid == -1, similarity_values_grid)
    axes.append(plt.subplot2grid((3, 2), (2, 1), fig=fig))
    im = axes[4].imshow(masked_similarities, cmap='hot', interpolation='nearest')
    axes[4].set_title('Similarity Values\n(Higher = stronger token-patch match)', fontsize=14)
    
    # Add similarity values as text
    for token_idx, (token, row, col, similarity) in enumerate(token_patch_info):
        text_color = 'white' if similarity > np.max(similarity_values_grid) * 0.5 else 'black'
        axes[4].text(col, row, f'{similarity:.3f}', 
                       ha='center', va='center', color=text_color, fontsize=10, weight='bold')
    
    plt.colorbar(im, ax=axes[4], label='Similarity Score')
    axes[4].set_xlabel('Patch Column')
    axes[4].set_ylabel('Patch Row')
    
    plt.tight_layout()
    
    # Print enhanced statistics
    print(f"\nDetailed token-patch assignments:")
    sorted_info = sorted(token_patch_info, key=lambda x: x[3], reverse=True)  # Sort by similarity
    
    for rank, (token, row, col, similarity) in enumerate(sorted_info, 1):
        print(f"  #{rank}: '{token}' → Patch [{row},{col}] (similarity: {similarity:.4f})")
    
    print(f"\nSummary statistics:")
    similarities = [info[3] for info in token_patch_info]
    print(f"  All tokens represented: {len(token_patch_info)}/{len(tokens)}")
    print(f"  Similarity range: {min(similarities):.4f} to {max(similarities):.4f}")
    print(f"  Mean similarity: {np.mean(similarities):.4f}")
    
    # Check for conflicts (multiple tokens wanting same patch)
    patch_positions = [(info[1], info[2]) for info in token_patch_info]
    unique_positions = set(patch_positions)
    if len(unique_positions) < len(patch_positions):
        conflicts = len(patch_positions) - len(unique_positions)
        print(f"  ⚠️  Patch conflicts: {conflicts} tokens share patch positions")
    else:
        print(f"  ✓ No conflicts: All tokens have unique patches")
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_file = f"{save_path}_token_patch_overlay.png"
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_file}")
    
    plt.show()
    plt.close(fig)

def visualize_row_wise_token_dominance(similarity_np, tokens, image_tensor, orig_image, save_path=None):
    """Visualize the patch with highest similarity in each row of the image."""
    print(f"\n=== ROW-WISE HIGHEST SIMILARITY PATCH ANALYSIS ===")
    print("Finding the patch with highest similarity in each image row")
    
    grid_size = int(np.sqrt(similarity_np.shape[1]))
    num_tokens = len(tokens)
    
    # Reshape similarity matrix to grid format for easier row processing
    # similarity_np shape: [num_tokens, num_patches]
    # We want: [num_tokens, grid_size, grid_size]
    similarity_grid = similarity_np.reshape(num_tokens, grid_size, grid_size)
    
    # Analyze each row
    row_max_patches = []      # Column position of max patch in each row
    row_max_similarities = [] # Max similarity value in each row
    row_max_tokens = []       # Token corresponding to max similarity in each row
    
    print(f"\nRow-by-row analysis (highest similarity patch per row):")
    for row_idx in range(grid_size):
        # Get all similarities for this row across all tokens and columns
        row_data = similarity_grid[:, row_idx, :]  # Shape: [num_tokens, grid_size]
        
        # Find the maximum similarity in this entire row
        max_similarity = np.max(row_data)
        
        # Find the position (token, column) of this maximum
        max_token_idx, max_col_idx = np.unravel_index(np.argmax(row_data), row_data.shape)
        max_token = tokens[max_token_idx]
        
        row_max_patches.append(max_col_idx)
        row_max_similarities.append(max_similarity)
        row_max_tokens.append(max_token)
        
        print(f"  Row {row_idx}: Patch [{row_idx},{max_col_idx}] → '{max_token}' (similarity: {max_similarity:.4f})")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Original image
    axes[0, 0].imshow(orig_image)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    # 2. Row-wise max patch visualization
    # Create a grid showing only the highest similarity patch in each row
    max_patch_grid = np.zeros((grid_size, grid_size))
    token_grid = np.full((grid_size, grid_size), -1, dtype=int)  # -1 for unassigned
    
    # Create color map for tokens
    colors = plt.cm.Set3(np.linspace(0, 1, num_tokens))
    
    for row_idx in range(grid_size):
        col_idx = row_max_patches[row_idx]
        max_patch_grid[row_idx, col_idx] = row_max_similarities[row_idx]
        token_idx = tokens.index(row_max_tokens[row_idx])
        token_grid[row_idx, col_idx] = token_idx
    
    # Create colored visualization
    row_vis_image = np.ones((grid_size, grid_size, 3)) * 0.9  # Light gray background
    
    for row_idx in range(grid_size):
        col_idx = row_max_patches[row_idx]
        token_idx = token_grid[row_idx, col_idx]
        if token_idx != -1:
            row_vis_image[row_idx, col_idx] = colors[token_idx][:3]
    
    axes[0, 1].imshow(row_vis_image)
    axes[0, 1].set_title('Highest Similarity Patch per Row\n(One patch highlighted per row)', fontsize=14)
    
    # Add text annotations for each highlighted patch
    for row_idx in range(grid_size):
        col_idx = row_max_patches[row_idx]
        token = row_max_tokens[row_idx]
        similarity = row_max_similarities[row_idx]
        
        # Add patch coordinates and similarity
        axes[0, 1].text(col_idx, row_idx, f'{token}\n{similarity:.3f}', 
                       ha='center', va='center', fontsize=8, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Add grid lines
    for i in range(grid_size + 1):
        axes[0, 1].axhline(y=i-0.5, color='black', linewidth=0.5, alpha=0.3)
        axes[0, 1].axvline(x=i-0.5, color='black', linewidth=0.5, alpha=0.3)
    
    axes[0, 1].set_xlabel('Patch Column')
    axes[0, 1].set_ylabel('Patch Row')
    
    # 3. Similarity values heatmap for max patches only
    masked_similarities = np.ma.masked_where(max_patch_grid == 0, max_patch_grid)
    im = axes[1, 0].imshow(masked_similarities, cmap='hot', interpolation='nearest')
    axes[1, 0].set_title('Similarity Values\n(Only highest patch per row)', fontsize=14)
    
    # Add similarity values as text
    for row_idx in range(grid_size):
        col_idx = row_max_patches[row_idx]
        similarity = row_max_similarities[row_idx]
        text_color = 'white' if similarity > np.max(row_max_similarities) * 0.5 else 'black'
        axes[1, 0].text(col_idx, row_idx, f'{similarity:.3f}', 
                       ha='center', va='center', color=text_color, fontsize=10, weight='bold')
    
    plt.colorbar(im, ax=axes[1, 0], label='Similarity Score')
    axes[1, 0].set_xlabel('Patch Column')
    axes[1, 0].set_ylabel('Patch Row')
    
    # 4. Column distribution of max patches
    col_counts = {}
    for col_idx in row_max_patches:
        col_counts[col_idx] = col_counts.get(col_idx, 0) + 1
    
    cols = list(range(grid_size))
    counts = [col_counts.get(col, 0) for col in cols]
    
    bars = axes[1, 1].bar(cols, counts, alpha=0.7)
    axes[1, 1].set_title('Column Distribution of Max Patches\n(How many rows have max in each column)', fontsize=14)
    axes[1, 1].set_xlabel('Patch Column')
    axes[1, 1].set_ylabel('Number of Rows')
    axes[1, 1].set_xticks(cols)
    
    # Add count labels on bars
    for i, count in enumerate(counts):
        if count > 0:
            axes[1, 1].text(i, count + 0.05, str(count), ha='center', va='bottom', fontsize=10, weight='bold')
    
    plt.tight_layout()
    
    # Print summary statistics
    print(f"\nSummary statistics:")
    print(f"  Similarity range: {min(row_max_similarities):.4f} to {max(row_max_similarities):.4f}")
    print(f"  Mean similarity: {np.mean(row_max_similarities):.4f}")
    
    print(f"\nColumn distribution (where max patches appear):")
    for col in range(grid_size):
        count = col_counts.get(col, 0)
        if count > 0:
            percentage = (count / grid_size) * 100
            print(f"  Column {col}: {count}/{grid_size} rows ({percentage:.1f}%)")
    
    print(f"\nToken distribution (which tokens have max patches):")
    token_counts = {}
    for token in row_max_tokens:
        token_counts[token] = token_counts.get(token, 0) + 1
    
    for token, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / grid_size) * 100
        print(f"  '{token}': {count}/{grid_size} rows ({percentage:.1f}%)")
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_file = f"{save_path}_row_wise_max_patches.png"
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_file}")
    
    plt.show()
    plt.close(fig)

def analyze_custom_regions(similarity_np, tokens, image_tensor, orig_image, regions=None, aggregation_method='mean', save_path=None):
    """
    Analyze token-patch similarities for custom defined regions instead of fixed patches.
    
    Args:
        similarity_np: [num_tokens, num_patches] similarity matrix
        tokens: List of token strings
        image_tensor: Preprocessed image tensor
        orig_image: Original PIL image
        regions: List of region definitions. If None, uses predefined semantic regions.
                Format: [{'name': 'region_name', 'bbox': (x1, y1, x2, y2)}, ...]
                Coordinates are in original image space (0-1 normalized)
        aggregation_method: 'mean', 'max', 'weighted_mean', 'top_k_mean'
        save_path: Path to save visualizations
    
    Returns:
        region_token_similarities: Dict mapping region names to token similarities
    """
    print(f"\n=== CUSTOM REGION-BASED SIMILARITY ANALYSIS ===")
    print(f"Aggregation method: {aggregation_method}")
    
    grid_size = int(np.sqrt(similarity_np.shape[1]))
    num_tokens, num_patches = similarity_np.shape
    
    # Define default semantic regions if none provided
    if regions is None:
        regions = [
            {'name': 'top_left', 'bbox': (0.0, 0.0, 0.5, 0.5)},
            {'name': 'top_right', 'bbox': (0.5, 0.0, 1.0, 0.5)},
            {'name': 'bottom_left', 'bbox': (0.0, 0.5, 0.5, 1.0)},
            {'name': 'bottom_right', 'bbox': (0.5, 0.5, 1.0, 1.0)},
            {'name': 'center', 'bbox': (0.25, 0.25, 0.75, 0.75)},
            {'name': 'top_strip', 'bbox': (0.0, 0.0, 1.0, 0.33)},
            {'name': 'middle_strip', 'bbox': (0.0, 0.33, 1.0, 0.67)},
            {'name': 'bottom_strip', 'bbox': (0.0, 0.67, 1.0, 1.0)},
        ]
    
    print(f"Analyzing {len(regions)} custom regions...")
    
    region_token_similarities = {}
    region_patch_mappings = {}
    
    for region in regions:
        region_name = region['name']
        bbox = region['bbox']  # (x1, y1, x2, y2) in normalized coordinates
        
        print(f"\nProcessing region '{region_name}': {bbox}")
        
        # Convert normalized bbox to patch indices
        x1, y1, x2, y2 = bbox
        
        # Convert to grid coordinates
        grid_x1 = int(x1 * grid_size)
        grid_y1 = int(y1 * grid_size)
        grid_x2 = min(int(x2 * grid_size), grid_size - 1)
        grid_y2 = min(int(y2 * grid_size), grid_size - 1)
        
        # Find all patches within this region
        patches_in_region = []
        for row in range(grid_y1, grid_y2 + 1):
            for col in range(grid_x1, grid_x2 + 1):
                patch_idx = row * grid_size + col
                if patch_idx < num_patches:
                    patches_in_region.append(patch_idx)
        
        region_patch_mappings[region_name] = patches_in_region
        print(f"  Patches in region: {len(patches_in_region)} (indices: {patches_in_region})")
        
        if len(patches_in_region) == 0:
            print(f"  Warning: No patches found in region '{region_name}'")
            continue
        
        # Extract similarities for patches in this region
        region_similarities = similarity_np[:, patches_in_region]  # [num_tokens, patches_in_region]
        
        # Aggregate similarities based on method
        if aggregation_method == 'mean':
            aggregated_similarities = np.mean(region_similarities, axis=1)
        elif aggregation_method == 'max':
            aggregated_similarities = np.max(region_similarities, axis=1)
        elif aggregation_method == 'weighted_mean':
            # Weight by similarity values (higher similarities get more weight)
            weights = region_similarities / (np.sum(region_similarities, axis=1, keepdims=True) + 1e-8)
            aggregated_similarities = np.sum(region_similarities * weights, axis=1)
        elif aggregation_method == 'top_k_mean':
            # Take mean of top 50% of patches in the region
            k = max(1, len(patches_in_region) // 2)
            top_k_similarities = np.partition(region_similarities, -k, axis=1)[:, -k:]
            aggregated_similarities = np.mean(top_k_similarities, axis=1)
        else:
            aggregated_similarities = np.mean(region_similarities, axis=1)
        
        region_token_similarities[region_name] = aggregated_similarities
        
        # Find best token for this region
        best_token_idx = np.argmax(aggregated_similarities)
        best_token = tokens[best_token_idx]
        best_similarity = aggregated_similarities[best_token_idx]
        
        print(f"  Best token: '{best_token}' (similarity: {best_similarity:.4f})")
        print(f"  Top 3 tokens:")
        top_3_indices = np.argsort(aggregated_similarities)[-3:][::-1]
        for rank, token_idx in enumerate(top_3_indices, 1):
            token = tokens[token_idx]
            sim = aggregated_similarities[token_idx]
            print(f"    #{rank}: '{token}' ({sim:.4f})")
    
    return region_token_similarities, region_patch_mappings

def visualize_region_token_analysis(similarity_np, tokens, image_tensor, orig_image, regions=None, 
                                  aggregation_method='mean', save_path=None):
    """
    Visualize the region-based token analysis with custom regions overlaid on the image.
    """
    print(f"\n=== VISUALIZING REGION-BASED TOKEN ANALYSIS ===")
    
    # Get region similarities
    region_token_similarities, region_patch_mappings = analyze_custom_regions(
        similarity_np, tokens, image_tensor, orig_image, regions, aggregation_method
    )
    
    # Define default regions if none provided
    if regions is None:
        regions = [
            {'name': 'top_left', 'bbox': (0.0, 0.0, 0.5, 0.5)},
            {'name': 'top_right', 'bbox': (0.5, 0.0, 1.0, 0.5)},
            {'name': 'bottom_left', 'bbox': (0.0, 0.5, 0.5, 1.0)},
            {'name': 'bottom_right', 'bbox': (0.5, 0.5, 1.0, 1.0)},
            {'name': 'center', 'bbox': (0.25, 0.25, 0.75, 0.75)},
        ]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Region-Based Token Analysis (Aggregation: {aggregation_method})', fontsize=16, weight='bold')
    
    # Colors for regions
    region_colors = plt.cm.Set3(np.linspace(0, 1, len(regions)))
    
    # 1. Original image
    axes[0, 0].imshow(orig_image)
    axes[0, 0].set_title('Original Image', fontsize=14)
    axes[0, 0].axis('off')
    
    # 2. Image with region overlays
    axes[0, 1].imshow(orig_image)
    img_width, img_height = orig_image.size
    
    for i, region in enumerate(regions):
        region_name = region['name']
        bbox = region['bbox']
        color = region_colors[i]
        
        # Convert normalized bbox to pixel coordinates
        x1 = int(bbox[0] * img_width)
        y1 = int(bbox[1] * img_height)
        x2 = int(bbox[2] * img_width)
        y2 = int(bbox[3] * img_height)
        
        # Draw region rectangle
        from matplotlib.patches import Rectangle
        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                        linewidth=3, edgecolor=color, facecolor=color, alpha=0.3)
        axes[0, 1].add_patch(rect)
        
        # Add region label
        axes[0, 1].text(x1+5, y1+15, region_name, fontsize=10, weight='bold', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))
    
    axes[0, 1].set_title('Custom Regions Overlay', fontsize=14)
    axes[0, 1].axis('off')
    
    # 3. Best token per region
    axes[0, 2].axis('off')
    axes[0, 2].set_xlim(0, 1)
    axes[0, 2].set_ylim(0, len(regions))
    axes[0, 2].set_title('Best Token per Region', fontsize=14)
    
    for i, region in enumerate(regions):
        region_name = region['name']
        if region_name in region_token_similarities:
            similarities = region_token_similarities[region_name]
            best_token_idx = np.argmax(similarities)
            best_token = tokens[best_token_idx]
            best_similarity = similarities[best_token_idx]
            color = region_colors[i]
            
            y_pos = len(regions) - 1 - i
            axes[0, 2].text(0.1, y_pos, f"{region_name}:", fontsize=12, weight='bold', va='center')
            axes[0, 2].text(0.6, y_pos, f"'{best_token}' ({best_similarity:.3f})", 
                           fontsize=12, va='center',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
    
    # 4. Similarity heatmap for all regions
    region_names = [r['name'] for r in regions if r['name'] in region_token_similarities]
    if region_names:
        similarity_matrix = np.array([region_token_similarities[name] for name in region_names])
        
        im = axes[1, 0].imshow(similarity_matrix, cmap='viridis', aspect='auto')
        axes[1, 0].set_ylabel('Regions')
        axes[1, 0].set_xlabel('Tokens')
        axes[1, 0].set_title('Region-Token Similarity Matrix', fontsize=14)
        axes[1, 0].set_yticks(range(len(region_names)))
        axes[1, 0].set_yticklabels(region_names)
        axes[1, 0].set_xticks(range(len(tokens)))
        axes[1, 0].set_xticklabels([f"'{t}'" for t in tokens], rotation=45, ha='right')
        plt.colorbar(im, ax=axes[1, 0], label='Similarity')
    
    # 5. Region similarity comparison
    if region_names:
        axes[1, 1].axis('off')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, len(tokens))
        axes[1, 1].set_title('Token Performance Across Regions', fontsize=14)
        
        for token_idx, token in enumerate(tokens):
            y_pos = len(tokens) - 1 - token_idx
            axes[1, 1].text(0.05, y_pos, f"'{token}':", fontsize=11, weight='bold', va='center')
            
            # Show best region for this token
            token_region_sims = {name: region_token_similarities[name][token_idx] for name in region_names}
            best_region = max(token_region_sims.keys(), key=lambda x: token_region_sims[x])
            best_sim = token_region_sims[best_region]
            
            region_idx = next(i for i, r in enumerate(regions) if r['name'] == best_region)
            color = region_colors[region_idx]
            
            axes[1, 1].text(0.4, y_pos, f"{best_region} ({best_sim:.3f})", 
                           fontsize=11, va='center',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
    
    # 6. Statistics summary
    axes[1, 2].axis('off')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_title('Analysis Summary', fontsize=14)
    
    summary_text = f"""Method: {aggregation_method}
Regions analyzed: {len(regions)}
Tokens analyzed: {len(tokens)}

Region Statistics:
"""
    
    if region_names:
        for region_name in region_names:
            similarities = region_token_similarities[region_name]
            max_sim = np.max(similarities)
            mean_sim = np.mean(similarities)
            summary_text += f"• {region_name}: max={max_sim:.3f}, avg={mean_sim:.3f}\n"
    
    axes[1, 2].text(0.05, 0.95, summary_text, fontsize=10, va='top', ha='left',
                   transform=axes[1, 2].transAxes, family='monospace')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_file = f"{save_path}_region_based_analysis_{aggregation_method}.png"
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_file}")
    
    plt.show()
    plt.close(fig)
    
    return region_token_similarities, region_patch_mappings

def robust_token_patch_analysis(similarity_np, tokens, image_tensor, orig_image, 
                              method='top_k_mean', k=3, threshold=0.1, save_path=None):
    """
    More robust analysis that doesn't rely on single highest similarity patch.
    
    Args:
        method: 'top_k_mean', 'threshold_based', 'spatial_smoothing'
        k: number of top patches to consider (for top_k_mean)
        threshold: minimum similarity threshold (for threshold_based)
    """
    print(f"\n=== ROBUST TOKEN-PATCH ANALYSIS ===")
    print(f"Method: {method}")
    
    grid_size = int(np.sqrt(similarity_np.shape[1]))
    num_tokens, num_patches = similarity_np.shape
    
    robust_results = {}
    
    for token_idx, token in enumerate(tokens):
        token_similarities = similarity_np[token_idx, :]
        
        if method == 'top_k_mean':
            # Take mean of top-k patches
            top_k_indices = np.argsort(token_similarities)[-k:]
            top_k_similarities = token_similarities[top_k_indices]
            aggregated_score = np.mean(top_k_similarities)
            relevant_patches = top_k_indices
            
        elif method == 'threshold_based':
            # Take all patches above threshold
            above_threshold = token_similarities >= threshold
            if np.sum(above_threshold) > 0:
                relevant_patches = np.where(above_threshold)[0]
                aggregated_score = np.mean(token_similarities[above_threshold])
            else:
                # Fallback to top patch if no patches above threshold
                relevant_patches = [np.argmax(token_similarities)]
                aggregated_score = np.max(token_similarities)
                
        elif method == 'spatial_smoothing':
            # Apply Gaussian smoothing to make attention more spatially coherent
            similarity_grid = token_similarities.reshape(grid_size, grid_size)
            if gaussian_filter:
                smoothed_grid = gaussian_filter(similarity_grid, sigma=0.5)
                smoothed_similarities = smoothed_grid.flatten()
                
                # Find top patch in smoothed version
                top_patch_idx = np.argmax(smoothed_similarities)
                aggregated_score = smoothed_similarities[top_patch_idx]
                relevant_patches = [top_patch_idx]
            else:
                print("Warning: scipy not available for spatial smoothing. Falling back to top patch.")
                relevant_patches = [np.argmax(token_similarities)]
                aggregated_score = np.max(token_similarities)
        
        robust_results[token] = {
            'score': aggregated_score,
            'patches': relevant_patches,
            'method': method
        }
        
        # Convert patch indices to grid coordinates for display
        patch_coords = []
        for patch_idx in relevant_patches:
            row = patch_idx // grid_size
            col = patch_idx % grid_size
            patch_coords.append((row, col))
        
        print(f"Token '{token}': score={aggregated_score:.4f}, patches={patch_coords}")
    
    return robust_results

def simple_patch_to_token_analysis(similarity_np, tokens, image_tensor, orig_image, top_k=3, save_path=None):
    """
    Simple patch-to-token analysis: For each image patch, show the top K most similar tokens.
    This is the inverse of token-to-patch - we start from patches and find their best matching tokens.
    
    Args:
        similarity_np: [num_tokens, num_patches] similarity matrix 
        tokens: List of token strings
        top_k: Number of top tokens to show for each patch (default: 3)
    """
    print(f"\n=== SIMPLE PATCH-TO-TOKEN ANALYSIS (TOP {top_k} TOKENS PER PATCH) ===")
    
    num_tokens, num_patches = similarity_np.shape
    grid_size = int(np.sqrt(num_patches))
    
    print(f"Grid size: {grid_size}×{grid_size} = {num_patches} patches")
    print(f"Finding top {top_k} tokens for each patch...\n")
    
    # For each patch (column), find which tokens (rows) have highest similarity
    # np.argsort sorts along axis=0 (tokens) for each patch
    top_k_token_indices = np.argsort(similarity_np, axis=0)[-top_k:]  # Shape: [top_k, num_patches]
    top_k_similarities = np.sort(similarity_np, axis=0)[-top_k:]      # Shape: [top_k, num_patches]
    
    # Print detailed results for each patch
    patch_results = []
    for patch_idx in range(num_patches):
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        
        print(f"Patch {patch_idx} [row={row}, col={col}]:")
        
        # Get top K tokens for this patch (highest first)
        patch_top_k_indices = top_k_token_indices[:, patch_idx][::-1]  # Reverse to get highest first
        patch_top_k_sims = top_k_similarities[:, patch_idx][::-1]
        
        patch_info = []
        for rank, (token_idx, similarity) in enumerate(zip(patch_top_k_indices, patch_top_k_sims), 1):
            token = tokens[token_idx]
            print(f"  #{rank}: '{token}' (similarity: {similarity:.4f})")
            patch_info.append((rank, token, similarity))
        
        patch_results.append({
            'patch_idx': patch_idx,
            'row': row, 
            'col': col,
            'top_tokens': patch_info
        })
        print()  # Empty line for readability
    
    # Create simple visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Original image
    axes[0].imshow(orig_image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # 2. Grid showing best token per patch
    # Create color map for tokens
    colors = plt.cm.Set3(np.linspace(0, 1, num_tokens))
    
    # Create grid visualization
    patch_grid = np.ones((grid_size, grid_size, 3)) * 0.9  # Light gray background
    
    for patch_idx in range(num_patches):
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        
        # Get best token for this patch (rank 1)
        best_token_idx = top_k_token_indices[-1, patch_idx]  # Last element is highest
        patch_grid[row, col] = colors[best_token_idx][:3]
    
    axes[1].imshow(patch_grid)
    axes[1].set_title(f'Best Token per Patch\n(Each patch colored by its top token)', fontsize=14)
    
    # Add text annotations showing top token names
    for patch_idx in range(num_patches):
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        
        best_token_idx = top_k_token_indices[-1, patch_idx]
        best_token = tokens[best_token_idx]
        best_similarity = top_k_similarities[-1, patch_idx]
        
        # Truncate long token names for display
        display_name = best_token if len(best_token) <= 4 else best_token[:3] + '..'
        
        axes[1].text(col, row, f'{display_name}\n{best_similarity:.2f}', 
                    ha='center', va='center', fontsize=8, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Add grid lines
    for i in range(grid_size + 1):
        axes[1].axhline(y=i-0.5, color='black', linewidth=0.5, alpha=0.3)
        axes[1].axvline(x=i-0.5, color='black', linewidth=0.5, alpha=0.3)
    
    axes[1].set_xlabel('Patch Column')
    axes[1].set_ylabel('Patch Row')
    
    # 3. Summary statistics
    axes[2].axis('off')
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].set_title('Analysis Summary', fontsize=14)
    
    # Count how many patches each token dominates
    token_dominance = {}
    for patch_idx in range(num_patches):
        best_token_idx = top_k_token_indices[-1, patch_idx]
        best_token = tokens[best_token_idx]
        token_dominance[best_token] = token_dominance.get(best_token, 0) + 1
    
    summary_text = f"Patch-to-Token Analysis Summary:\n\n"
    summary_text += f"Total patches: {num_patches}\n"
    summary_text += f"Total tokens: {num_tokens}\n"
    summary_text += f"Top {top_k} tokens shown per patch\n\n"
    summary_text += "Token dominance (# patches where token is #1):\n"
    
    for token, count in sorted(token_dominance.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / num_patches) * 100
        summary_text += f"• '{token}': {count} patches ({percentage:.1f}%)\n"
    
    axes[2].text(0.05, 0.95, summary_text, fontsize=10, va='top', ha='left',
                transform=axes[2].transAxes, family='monospace')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_file = f"{save_path}_simple_patch_to_token_top{top_k}.png"
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Saved visualization: {save_file}")
    
    plt.show()
    plt.close(fig)
    
    return patch_results

def main():
    # --- Setup ---
    print("INTERPRETABLE CLIP TOKEN HEATMAP VISUALIZATIONS")
    
    print("Loading interpretable CLIP model...")
    model = load_interpretable_clip("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Load image and set text prompt
    image_path = r"D:/Wajahat Ali Khan/CLIP/images/cat.png"
    image = Image.open(image_path).convert("RGB")
    text = "a photo of a cat"
    
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
    else:
        similarity_np = similarity
        
    grid_size = int(np.sqrt(similarity_np.shape[1]))
    
    # Print basic info
    print(f"\nResults:")
    print(f"  Tokens found: {tokens}")
    print(f"  Grid size: {grid_size}×{grid_size} = {similarity_np.shape[1]} patches")
    
    # --- COMPARISON: OLD vs NEW METHODS ---
    print("\n" + "="*80)
    print("COMPARISON: TRADITIONAL vs ROBUST ANALYSIS METHODS")
    print("="*80)
    
    # OLD METHOD: Single highest similarity (current approach)
    print("\n🔴 OLD METHOD (Single Max Patch):")
    for token_idx, token in enumerate(tokens):
        token_similarities = similarity_np[token_idx, :]
        max_patch_idx = np.argmax(token_similarities)
        max_similarity = token_similarities[max_patch_idx]
        max_row = max_patch_idx // grid_size
        max_col = max_patch_idx % grid_size
        print(f"  '{token}': Patch [{max_row},{max_col}] = {max_similarity:.4f}")
    
    # NEW METHOD 1: Top-K averaging
    print("\n🟢 NEW METHOD 1 (Top-5 Patch Average - More Robust):")
    robust_results = robust_token_patch_analysis(similarity_np, tokens, image_input, image, 'top_k_mean', k=5)
    
    # NEW METHOD 2: Custom semantic regions
    print("\n🟢 NEW METHOD 2 (Semantic Regions - Most Meaningful):")
    
    # Define regions based on your image content (customize these!)
    if "plant" in image_path.lower() or "potted" in image_path.lower():
        custom_regions = [
            {'name': 'plant_area', 'bbox': (0.2, 0.1, 0.8, 0.7)},      # Main plant area
            {'name': 'pot_base', 'bbox': (0.3, 0.6, 0.7, 0.9)},        # Pot/base area  
            {'name': 'background', 'bbox': (0.0, 0.0, 1.0, 0.4)},      # Background/wall
            {'name': 'left_context', 'bbox': (0.0, 0.0, 0.3, 1.0)},    # Left side
            {'name': 'right_context', 'bbox': (0.7, 0.0, 1.0, 1.0)},   # Right side
        ]
    else:
        # Generic regions for other images
        custom_regions = [
            {'name': 'main_object', 'bbox': (0.25, 0.25, 0.75, 0.75)}, # Center area
            {'name': 'top_area', 'bbox': (0.0, 0.0, 1.0, 0.4)},        # Top third
            {'name': 'bottom_area', 'bbox': (0.0, 0.6, 1.0, 1.0)},     # Bottom third
            {'name': 'left_side', 'bbox': (0.0, 0.0, 0.4, 1.0)},       # Left side
            {'name': 'right_side', 'bbox': (0.6, 0.0, 1.0, 1.0)},      # Right side
        ]
    
    # Analyze using different aggregation methods
    for agg_method in ['mean', 'max', 'top_k_mean']:
        print(f"\n  📊 Using '{agg_method}' aggregation:")
        region_similarities, _ = analyze_custom_regions(
            similarity_np, tokens, image_input, image, custom_regions, agg_method
        )
    
    # --- VISUALIZATIONS ---
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Generate SIMPLE patch-to-token analysis (NEW - what user requested)
    print("\n1. SIMPLE PATCH-TO-TOKEN ANALYSIS (TOP 3 TOKENS PER PATCH)...")
    simple_patch_to_token_analysis(similarity_np, tokens, image_input, image, top_k=3, save_path="results/heatmap_analysis")
    
    # Generate basic token heatmaps
    print("\n2. Traditional token heatmaps...")
    plot_token_heatmaps(similarity_np, tokens, image_input, image, text, save_path="results/heatmap_analysis")
    
    # Generate robust region-based analysis
    print("\n3. Region-based analysis (RECOMMENDED)...")
    visualize_region_token_analysis(similarity_np, tokens, image_input, image, custom_regions, 'mean', save_path="results/heatmap_analysis")
    
    # Generate patch similarity grid (traditional method)
    print("\n4. Traditional patch-token mapping...")
    visualize_patch_similarity_grid(similarity_np, tokens, image_input, image, save_path="results/heatmap_analysis")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\n📈 KEY IMPROVEMENTS:")
    print("✓ Robust aggregation reduces noise from single-patch artifacts")
    print("✓ Semantic regions provide more meaningful interpretations")  
    print("✓ Multiple aggregation methods show different aspects of attention")
    print("✓ Still preserves underlying 49-patch ViT architecture fidelity")
    print("\n🎯 RECOMMENDATION: Use region-based analysis for final results!")
    print("   It's more robust, semantically meaningful, and scientifically sound.")

if __name__ == "__main__":
    main() 