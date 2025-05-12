import torch
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clip.interpretable_clip import load_interpretable_clip
from clip.clip import tokenize

def plot_token_patch_matrix(tokens, similarity):
    # Ensure similarity is a NumPy array
    if hasattr(similarity, 'detach'):
        similarity = similarity.detach().cpu().numpy()
    plt.figure(figsize=(10, max(4, len(tokens) * 0.5)))
    plt.imshow(similarity, aspect='auto', cmap='viridis')
    plt.colorbar(label='Cosine Similarity')
    plt.yticks(np.arange(len(tokens)), [str(t) for t in tokens])
    plt.xlabel('Patch Index')
    plt.ylabel('Token')
    plt.title('Token-Patch Similarity Matrix')
    plt.tight_layout()
    plt.show()

def main():
    # Load model
    model = load_interpretable_clip("ViT-B/32")
    
    # Load and preprocess image
    image = Image.open("D:/Wajahat Ali Khan/CLIP/human.png")
    
    # Example text (change as needed)
    text = "An image of a lady with a cat and a dog"
    
    # Print real tokens and their indices
    tokens, similarity = model.get_token_patch_similarity(
        model.preprocess(image).unsqueeze(0),
        tokenize([text])
    )
    print("Real tokens and their indices:")
    for i, t in enumerate(tokens):
        print(f"{i}: '{t}'")
    
    # Plot token-patch similarity matrix (confusion matrix style)
    plot_token_patch_matrix(tokens, similarity)
    
    # Visualize all token overlays in one figure for easy comparison
    #model.visualize_all_token_patch_overlays(image, text, alpha=0.5, max_cols=4)
    
    # Plot token importance
   #model.plot_token_importance(image, text)

if __name__ == "__main__":
    main() 