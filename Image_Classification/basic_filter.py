from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import os
import numpy as np

def apply_wild_neon_filter(image_path, output_path="neon_tiger.png"):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((512, 512))
        
        # Enhance color saturation for vibrant look
        enhancer = ImageEnhance.Color(img_resized)
        img_saturated = enhancer.enhance(1.8)
        
        # Enhance contrast to make features pop
        contrast = ImageEnhance.Contrast(img_saturated)
        img_contrast = contrast.enhance(1.4)
        
        # Enhance sharpness for crisp edges
        sharpness = ImageEnhance.Sharpness(img_contrast)
        img_sharp = sharpness.enhance(2.0)
        
        # Add a slight edge enhancement
        img_edges = img_sharp.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        # Blend original sharp image with edge-enhanced version
        img_final = Image.blend(img_sharp, img_edges, 0.3)
        
        # Display the result
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_resized)
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_final)
        plt.title("Wild Neon Filter")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"Processed image saved as '{output_path}'.")
        
        return img_final
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    print("Wild Neon Filter - Tiger Edition (type 'exit' to quit)\n")
    while True:
        image_path = input("Enter image filename (or 'exit' to quit): ").strip()
        
        if image_path.lower() == 'exit':
            print("Goodbye!")
            break
        
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue
        
        # Derive output filename
        base, ext = os.path.splitext(image_path)
        output_file = f"{base}_wild_neon{ext}"
        
        apply_wild_neon_filter(image_path, output_file)