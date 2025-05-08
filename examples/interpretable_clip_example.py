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
    image = Image.open("D:/Wajahat Ali Khan/CLIP/cat.PNG")
    
    # Example text
    text = "Not a photo of a cat"
    
    # Visualize similarities for all tokens
    model.visualize_token_patch_similarity(image, text)
    
    # Visualize similarities for a specific token (e.g., "cat")
    tokens, _ = model.get_token_patch_similarity(
        model.preprocess(image).unsqueeze(0),
        tokenize([text])
    )
    
    # Print tokens for debugging
    print("Tokens:", tokens)

    # Find the token index containing 'cat'
    cat_idx = next((i for i, t in enumerate(tokens) if "cat" in t.lower()), None)
    if cat_idx is not None:
        model.visualize_token_patch_similarity(image, text, token_idx=cat_idx)
    else:
        print("No token containing 'cat' found:", tokens)

if __name__ == "__main__":
    main() 