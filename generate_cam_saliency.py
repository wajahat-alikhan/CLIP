"""
CAM-Style Saliency Map Generator for Interpretable CLIP

This module creates Class Activation Map (CAM) style saliency visualizations
that show exactly what CLIP sees - no coordinate transformations applied.
Generates both individual token saliency maps and overall sentence saliency maps.

Features:
- Research-quality CAM-style visualizations  
- Raw CLIP spatial attention (no flips/rotations)
- Individual token saliency maps
- Combined sentence saliency maps  
- Smooth gradients with proper upsampling
- Professional appearance suitable for publications
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pathlib import Path
import matplotlib.cm as cm
from clip.interpretable_clip_attention import load_interpretable_clip, tokenize_text

class CAMSaliencyGenerator:
    """Generates CAM-style saliency maps for interpretable CLIP"""
    
    def __init__(self, model_name="ViT-B/16", device="cpu"):
        """Initialize the CAM generator with a CLIP model"""
        self.model_name = model_name
        self.device = device
        self.model = load_interpretable_clip(model_name, device=device)
        self.grid_size = self.model.visual.input_resolution // self.model.visual.conv1.kernel_size[0]
        
        print(f"🎯 CAM Saliency Generator initialized")
        print(f"   Model: {model_name}")
        print(f"   Grid: {self.grid_size}×{self.grid_size} = {self.grid_size**2} patches")
    
    def create_cam_saliency(self, image, similarity_vector, title="Saliency Map", 
                           colormap='jet', alpha=0.4, save_path=None):
        """
        Create a single CAM-style saliency map from similarity scores
        
        Args:
            image: PIL Image - original image
            similarity_vector: 1D tensor/array of similarity scores for patches
            title: string - title for the visualization
            colormap: string - matplotlib colormap ('jet', 'hot', 'viridis', etc.)
            alpha: float - transparency of overlay (0.0-1.0)
            save_path: Path - where to save the result
        """
        # Convert to numpy if needed
        if hasattr(similarity_vector, 'detach'):
            sim_np = similarity_vector.detach().cpu().numpy()
        else:
            sim_np = similarity_vector
        
        # Reshape to spatial grid (no transformations - show exactly what CLIP sees)
        saliency_grid = sim_np.reshape(self.grid_size, self.grid_size)
        
        # Get image dimensions
        img_w, img_h = image.size
        
        # Upsample to image resolution using cubic interpolation for smoothness
        saliency_resized = cv2.resize(
            saliency_grid.astype(np.float32), 
            (img_w, img_h), 
            interpolation=cv2.INTER_CUBIC
        )
        
        # Apply Gaussian blur for CAM-style smoothness
        # Kernel size proportional to image size for consistent appearance
        kernel_size = max(21, min(img_w, img_h) // 10)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        saliency_smooth = cv2.GaussianBlur(
            saliency_resized, 
            (kernel_size, kernel_size), 
            0
        )
        
        # Normalize to [0, 1] for proper colormap application
        saliency_norm = (saliency_smooth - saliency_smooth.min()) / (
            saliency_smooth.max() - saliency_smooth.min() + 1e-8
        )
        
        # Create the CAM visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Display original image
        ax.imshow(image)
        
        # Apply colormap and overlay
        im = ax.imshow(saliency_norm, cmap=colormap, alpha=alpha, vmin=0, vmax=1)
        
        # Clean professional styling
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Saliency Intensity', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved CAM saliency map: {save_path}")
        
        plt.show()
        
        return saliency_norm
    
    def generate_token_saliency_maps(self, image_path, text_prompt, save_dir="cam_results"):
        """
        Generate individual CAM saliency maps for each token in the text
        
        Args:
            image_path: Path to the image file
            text_prompt: Text prompt to analyze
            save_dir: Directory to save results
        """
        print(f"🎯 Generating token saliency maps")
        print(f"   Image: {image_path}")
        print(f"   Text: '{text_prompt}'")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_input = self.model.preprocess(image).unsqueeze(0).to(self.device)
        text_input = tokenize_text(text_prompt).to(self.device)
        
        # Get token-patch similarities
        tokens, similarity_matrix = self.model.get_token_patch_similarity(image_input, text_input)
        
        print(f"   Found {len(tokens)} tokens: {tokens}")
        
        # Create output directory
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Generate CAM for each token
        token_maps = {}
        for i, token in enumerate(tokens):
            print(f"   Generating CAM for '{token}'...")
            
            save_path = save_dir / f"cam_token_{token}_{self.model_name.replace('/', '_')}.png"
            title = f'Token Saliency: "{token}"'
            
            # Create CAM-style saliency map
            saliency_map = self.create_cam_saliency(
                image=image,
                similarity_vector=similarity_matrix[i, :],
                title=title,
                colormap='jet',  # Classic CAM colormap
                alpha=0.4,
                save_path=save_path
            )
            
            token_maps[token] = saliency_map
        
        print(f"✅ Generated {len(tokens)} token saliency maps")
        return tokens, token_maps
    
    def generate_sentence_saliency_map(self, image_path, text_prompt, save_dir="cam_results"):
        """
        Generate overall sentence CAM saliency map combining all tokens
        
        Args:
            image_path: Path to the image file
            text_prompt: Text prompt to analyze
            save_dir: Directory to save results
        """
        print(f"🎯 Generating sentence saliency map")
        print(f"   Image: {image_path}")
        print(f"   Text: '{text_prompt}'")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_input = self.model.preprocess(image).unsqueeze(0).to(self.device)
        text_input = tokenize_text(text_prompt).to(self.device)
        
        # Get token-patch similarities
        tokens, similarity_matrix = self.model.get_token_patch_similarity(image_input, text_input)
        
        # Combine all token similarities (average across tokens)
        combined_similarity = similarity_matrix.mean(dim=0)
        
        print(f"   Combined similarity from {len(tokens)} tokens")
        
        # Create output directory
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Generate combined CAM
        save_path = save_dir / f"cam_sentence_{self.model_name.replace('/', '_')}.png"
        title = f'Sentence Saliency: "{text_prompt}"'
        
        sentence_map = self.create_cam_saliency(
            image=image,
            similarity_vector=combined_similarity,
            title=title,
            colormap='jet',  # Classic research paper style
            alpha=0.4,
            save_path=save_path
        )
        
        print(f"✅ Generated sentence saliency map")
        return sentence_map
    
    def generate_comparative_cam(self, image_path, text_prompt, save_dir="cam_results"):
        """
        Generate both token and sentence CAMs in one comprehensive analysis
        """
        print(f"🎯 Generating Comprehensive CAM Analysis")
        print(f"="*60)
        
        # Generate individual token CAMs
        tokens, token_maps = self.generate_token_saliency_maps(image_path, text_prompt, save_dir)
        
        print(f"\n" + "-"*40)
        
        # Generate sentence CAM
        sentence_map = self.generate_sentence_saliency_map(image_path, text_prompt, save_dir)
        
        print(f"\n" + "="*60)
        print(f"🎉 CAM Analysis Complete!")
        print(f"   📁 Results saved to: {Path(save_dir).absolute()}")
        print(f"   📊 Generated {len(tokens)} token CAMs + 1 sentence CAM")
        print(f"   🎨 Research-quality visualizations ready!")
        
        return {
            'tokens': tokens,
            'token_maps': token_maps, 
            'sentence_map': sentence_map
        }

def demo_cam_generation():
    """Demonstration of CAM saliency map generation"""
    
    print("🎯 CAM Saliency Map Demo")
    print("="*50)
    
    # Initialize CAM generator
    cam_generator = CAMSaliencyGenerator("ViT-B/16", device="cpu")
    
    # Try to use your cat image or create a demo
    try:
        image_path = "D:/Wajahat Ali Khan/CLIP/images/dog.PNG"
        text_prompt = "a photo of a dog"
        
        print(f"📷 Using your cat image")
        
    except FileNotFoundError:
        print("📷 Cat image not found - you can update the path in the script")
        return
    
    # Generate comprehensive CAM analysis
    results = cam_generator.generate_comparative_cam(
        image_path=image_path,
        text_prompt=text_prompt,
        save_dir="cam_saliency_results"
    )
    
    print(f"\n💡 Your CAM saliency maps are ready!")
    print(f"   Individual token maps show where each word focuses")
    print(f"   Sentence map shows overall attention pattern") 
    print(f"   Perfect for research papers and presentations!")

if __name__ == "__main__":
    demo_cam_generation() 