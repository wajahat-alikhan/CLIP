#!/usr/bin/env python3
"""
Test the simplified attention analysis approach.
"""

import torch
from clip.simple_attention_analysis import analyze_clip_simple

def test_simple_analysis():
    print("🎯 TESTING SIMPLIFIED CLIP ATTENTION ANALYSIS")
    print("=" * 70)
    
    # Test simplified attention analysis
    results = analyze_clip_simple(
        image_path='images/cat.PNG',
        texts=['a dog', 'a cat'],
        text_index=1,  # analyze 'a cat'
        debug=True
    )
    
    print(f'\n📊 SIMPLIFIED ANALYSIS RESULTS:')
    print(f'Image attention shape: {results["image_attention"].shape}')
    print(f'Image attention std: {results["image_attention"].std():.6f}')
    print(f'Image attention range: [{results["image_attention"].min():.6f}, {results["image_attention"].max():.6f}]')
    print(f'Selected text: {results["selected_text"]}')
    
    # Check if we got meaningful results
    if results["image_attention"].std() > 0.001:
        print("✅ Image attention shows variation - likely working!")
        print("🎉 SUCCESS: Gradients are being captured correctly!")
    else:
        print("❌ Image attention is uniform - still not working")
        
    # Show attention heatmap values
    attention = results["image_attention"]
    grid_size = int(len(attention) ** 0.5)
    if grid_size * grid_size == len(attention):
        attention_2d = attention.reshape(grid_size, grid_size)
        print(f'\n📊 Attention heatmap ({grid_size}x{grid_size}):')
        print(attention_2d.numpy())

if __name__ == "__main__":
    test_simple_analysis() 