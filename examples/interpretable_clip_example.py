import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import torch
from clip.interpretable_clip import load_interpretable_clip

def main():
    # --- Setup ---
    print("Loading interpretable CLIP model...")
    model = load_interpretable_clip("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
    
    image_path = r"D:/Wajahat Ali Khan/CLIP/human.png"
    image = Image.open(image_path).convert("RGB")
    text = "an image of a dog and a cat"
    
    print(f"\nAnalyzing image: {image_path}")
    print(f"Using text prompt: '{text}'")
    
    # --- Demonstration of Different Visualization Styles ---
    print("\n" + "="*60)
    print("DEMONSTRATION: Different Gradient Heatmap Styles")
    print("="*60)
    
    
    # Option 2: Thermal/Heat style visualization
    print("\n Thermal Style (Hot colormap - heat map):")
    model.analyze_image_text_pair(image, text, show_matrix=True, heatmap_style='thermal')
    
    # --- NEW: Individual Token Impact Analysis ---
    print("\n" + "="*60)
    print("NEW FEATURE: Individual Token Impact Analysis")
    print("="*60)
    print("\nAnalyzing individual word impacts:")
    model.analyze_image_text_pair(image, text, show_matrix=False, heatmap_style='thermal', 
                                 show_individual_tokens=True)
    
if __name__ == "__main__":
    main() 