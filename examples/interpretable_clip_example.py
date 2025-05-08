import torch
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clip.interpretable_clip import load_interpretable_clip
from clip.clip import tokenize

def main():
    # Load model
    model = load_interpretable_clip("ViT-B/32")
    
    # Load and preprocess image
    image = Image.open("D:/Wajahat Ali Khan/CLIP/cat.PNG")  # Change path as needed
    
    # Example text (change as needed)
    text = "A photo of a dog"
    
    # Visualize similarities for all tokens
    model.visualize_token_patch_similarity(image, text)
    
    # Get tokens and similarity matrix
    tokens, similarity = model.get_token_patch_similarity(
        model.preprocess(image).unsqueeze(0),
        tokenize([text])
    )
    print("Tokens and their indices:")
    for i, t in enumerate(tokens):
        print(f"{i}: '{t}'")
    
    # Example: let user select a token by index or substring
    # (for demo, we'll use the first token containing 'photo', or index 0 if not found)
    query = "dog"# Change this to any substring you want to search for
    matches = model.find_token_indices(tokens, query)
    if matches:
        selected_idx = matches[0]
        print(f"Visualizing overlay for token '{tokens[selected_idx]}' (idx={selected_idx})")
        model.visualize_token_patch_similarity(image, text, token_idx=selected_idx)
        model.visualize_token_patch_overlay(image, text, token_idx=selected_idx, alpha=0.5)
    else:
        print(f"No token containing '{query}' found. Showing overlay for first token.")
        model.visualize_token_patch_similarity(image, text, token_idx=0)
        model.visualize_token_patch_overlay(image, text, token_idx=0, alpha=0.5)
    
    # Plot token importance
    model.plot_token_importance(image, text)

    # Visualize all token overlays in one figure for easy comparison
    model.visualize_all_token_patch_overlays(image, text, alpha=0.5, max_cols=4)

if __name__ == "__main__":
    main() 