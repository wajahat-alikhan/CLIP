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

# Import from clip module
try:
    from clip.interpretable_clip import load_interpretable_clip, tokenize_text
except ImportError:
    # Fallback: try direct import
    import clip.interpretable_clip as interpretable_clip
    load_interpretable_clip = interpretable_clip.load_interpretable_clip
    tokenize_text = interpretable_clip.tokenize_text

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

def main():
    # --- Setup ---
    print("INTERPRETABLE CLIP TOKEN HEATMAP VISUALIZATIONS")
    
    print("Loading interpretable CLIP model...")
    model = load_interpretable_clip("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Load image and set text prompt
    image_path = r"D:/Wajahat Ali Khan/CLIP/images/catdog.png"
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
    else:
        similarity_np = similarity
        
    grid_size = int(np.sqrt(similarity_np.shape[1]))
    
    # Print basic info
    print(f"\nResults:")
    print(f"  Tokens found: {tokens}")
    print(f"  Grid size: {grid_size}×{grid_size} = {similarity_np.shape[1]} patches")
    
    # --- TOKEN HEATMAP VISUALIZATIONS ---
    print("\nGENERATING TOKEN HEATMAP VISUALIZATIONS")
    
    # Generate heatmaps for each token using the new visualization style
    plot_token_heatmaps(similarity_np, tokens, image_input, image, text, save_path="results/heatmap_analysis")
    
    print("\nTOKEN HEATMAP VISUALIZATION COMPLETE")

if __name__ == "__main__":
    main() 