# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from bilateral import jbf

def main():
    print("=" * 60)
    print("Experiment 3-2: Joint Bilateral Filter")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))
    lab2_img = os.path.join('D:', '2026第二学期', '2026数媒数字图像处理实验说明', 'Pset2_Geometry and Calibration', '提供图像', 'lab2.png')
    default_img = os.path.join(project_root, 'static', 'imgs', 'test0', '1.bmp')

    img_path = sys.argv[1] if len(sys.argv) > 1 else lab2_img
    if not os.path.exists(img_path):
        img_path = default_img

    img = Image.open(img_path).convert('RGB')
    input_img = np.array(img)

    sigma_f = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    sigma_g = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    w = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    print(f"Input: {img_path}")
    print(f"Size: {input_img.shape[0]}x{input_img.shape[1]}")
    print(f"Window size (w): {w}")
    print(f"Spatial sigma (sigma_f): {sigma_f}")
    print(f"Range sigma (sigma_g): {sigma_g}")
    print("Processing...")

    output_img = jbf(input_img, input_img, w, sigma_f, sigma_g)

    output_dir = os.path.join(project_root, 'static', 'imgs', 'output')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'bilateral_filtered.png')

    result = Image.fromarray(output_img)
    result.save(save_path)
    print(f"Saved: {save_path}")

    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Joint Bilateral Filter', fontsize=14)
        axes[0].imshow(input_img)
        axes[0].set_title('Original')
        axes[0].axis('off')
        axes[1].imshow(input_img)
        axes[1].set_title('Guide Image')
        axes[1].axis('off')
        axes[2].imshow(output_img)
        axes[2].set_title('Bilateral Filtered')
        axes[2].axis('off')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Cannot display: {e}")

if __name__ == '__main__':
    main()