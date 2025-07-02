import clip

# Load model
model, preprocess = clip.load('ViT-B/32')

print("=== CLIP ViT-B/32 Architecture Analysis ===")
print(f"Visual proj shape: {model.visual.proj.shape}")
print(f"Text projection shape: {model.text_projection.shape}")
print(f"Final embedding dimension: {model.text_projection.shape[1]}")

# Check visual encoder details
print(f"\nVisual encoder type: {type(model.visual)}")
if hasattr(model.visual, 'conv1'):
    print(f"Visual conv1 out_channels (width): {model.visual.conv1.out_channels}")
if hasattr(model.visual, 'proj'):
    print(f"Visual projection: {model.visual.conv1.out_channels} -> {model.visual.proj.shape[1]}")

# Check text encoder details
print(f"\nText encoder details:")
print(f"Token embedding: {model.token_embedding.weight.shape}")
print(f"Transformer width: {model.transformer.width}")
print(f"Text projection: {model.text_projection.shape[0]} -> {model.text_projection.shape[1]}")

# Test a larger model too
print("\n" + "="*50)
print("=== CLIP ViT-L/14 Architecture Analysis ===")
try:
    model_large, _ = clip.load('ViT-L/14')
    print(f"Visual proj shape: {model_large.visual.proj.shape}")
    print(f"Text projection shape: {model_large.text_projection.shape}")
    print(f"Final embedding dimension: {model_large.text_projection.shape[1]}")
    print(f"Visual width: {model_large.visual.conv1.out_channels}")
    print(f"Transformer width: {model_large.transformer.width}")
except Exception as e:
    print(f"Could not load ViT-L/14: {e}") 