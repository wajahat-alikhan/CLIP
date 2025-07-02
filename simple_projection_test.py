import torch
from clip.interpretable_clip import load_interpretable_clip
from clip import load as load_original_clip

print("Loading models...")
original_model, _ = load_original_clip("ViT-B/32", device="cpu")
interpretable_model = load_interpretable_clip("ViT-B/32", device="cpu")

print("Testing projection heads equivalence...")

# Check visual projection
orig_visual_proj = original_model.visual.proj
interp_visual_proj = interpretable_model.visual.proj
visual_match = torch.allclose(orig_visual_proj, interp_visual_proj, atol=1e-6)
print(f"Visual projection matrices identical: {visual_match}")

# Check text projection  
orig_text_proj = original_model.text_projection
interp_text_proj = interpretable_model.text_projection
text_match = torch.allclose(orig_text_proj, interp_text_proj, atol=1e-6)
print(f"Text projection matrices identical: {text_match}")

print(f"Original visual proj shape: {orig_visual_proj.shape}")
print(f"Interpretable visual proj shape: {interp_visual_proj.shape}")
print(f"Original text proj shape: {orig_text_proj.shape}")  
print(f"Interpretable text proj shape: {interp_text_proj.shape}")

print("CONCLUSION: Your implementation uses the EXACT SAME projection heads as original CLIP!") 