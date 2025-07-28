#!/usr/bin/env python3
"""
Test the proper attention analysis approach.
"""

import torch
from clip.proper_attention_analysis import analyze_clip_proper

def test_proper_analysis():
    print("🎯 TESTING PROPER CLIP ATTENTION ANALYSIS")
    print("=" * 70)
    
    # Test proper attention analysis
    results = analyze_clip_proper(
        image_path='images/cat.PNG',
        texts=['a dog', 'a cat'],
        text_index=1,  # analyze 'a cat'
        debug=True
    )
    
    print(f'\n📊 PROPER ANALYSIS RESULTS:')
    print(f'Image attention shape: {results["image_attention"].shape}')
    print(f'Image attention std: {results["image_attention"].std():.6f}')
    print(f'Image attention range: [{results["image_attention"].min():.6f}, {results["image_attention"].max():.6f}]')
    print(f'Text attention shape: {results["text_attention"].shape}')
    print(f'Text attention std: {results["text_attention"].std():.6f}')
    print(f'Tokens: {[t[0] for t in results["tokens"]]}')
    
    # Check if we got meaningful results
    if results["image_attention"].std() > 0.001:
        print("✅ Image attention shows variation - likely working!")
    else:
        print("❌ Image attention is uniform - likely not working")
        
    if results["text_attention"].std() > 0.001:
        print("✅ Text attention shows variation - likely working!")
    else:
        print("❌ Text attention is uniform - likely not working")

if __name__ == "__main__":
    test_proper_analysis() 