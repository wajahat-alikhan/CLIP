"""
Simple Demo: Interpretable CLIP Visualizations

This script demonstrates how to generate publication-quality visualizations
using your interpretable CLIP implementation.
"""

import sys
sys.path.append('experiments')

from experiments.visualize_interpretability import InterpretabilityVisualizer
import numpy as np
from PIL import Image

def quick_demo():
    """Quick demonstration of interpretable CLIP visualizations"""
    
    print("🎨 Quick Demo: Interpretable CLIP Visualizations")
    print("="*50)
    
    # Initialize visualizer with ViT-B/16 (good balance of detail vs speed)
    visualizer = InterpretabilityVisualizer("ViT-B/16", device="cpu")
    
    # Try to load your cat image, or use a dummy
    try:
        print("📷 Loading your cat image...")
        image_path = "D:\Wajahat Ali Khan\CLIP\images\cat.PNG"  # Adjust path as needed
        image, _ = visualizer.load_and_preprocess_image(image_path)
        print("✅ Loaded cat.PNG successfully!")
    except FileNotFoundError:
        print("📷 Creating demo image...")
        # Create a colorful demo image
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        print("✅ Created demo image!")
    
    # Demo 1: Basic spatial analysis
    print(f"\n🔍 Demo 1: Spatial Localization Analysis")
    text_prompt = "an image of a cat"
    print(f"   Analyzing: '{text_prompt}'")
    
    tokens, similarity = visualizer.analyze_spatial_localization(
        image, 
        text_prompt, 
        save_dir="demo_results"
    )
    
    print(f"✅ Analysis complete!")
    print(f"   - Found {len(tokens)} meaningful tokens: {tokens}")
    print(f"   - Generated {similarity.shape[0]} attention heatmaps")
    print(f"   - Similarity matrix: {similarity.shape}")
    
    # Analysis complete - no architecture comparison needed
    
    print(f"\n🎉 Demo completed!")
    print(f"📁 Results saved to: demo_results/")
    print(f"💡 Your interpretable CLIP is generating beautiful visualizations!")

def custom_experiment():
    """Optional: Add your own custom experiments here"""
    
    print("\n🧪 Custom Experiment (Optional)")
    print("-" * 30)
    print("   You can add your own experiments here!")
    print("   Example: Test different text prompts with the same image")
    print("   Example: Analyze different types of objects")
    print("   Example: Compare how the model sees different concepts")

if __name__ == "__main__":
    # Run quick demo
    quick_demo()
    
    # Run custom experiment
    custom_experiment()
    
    print(f"\n{'='*50}")
    print(f"🚀 Ready for your own experiments!")
    print(f"💡 Tips:")
    print(f"   - Try different text prompts with the same image")
    print(f"   - Test with different types of images") 
    print(f"   - Analyze how the model localizes different concepts")
    print(f"   - Use the red heatmaps to see attention patterns!")
    print(f"{'='*50}") 