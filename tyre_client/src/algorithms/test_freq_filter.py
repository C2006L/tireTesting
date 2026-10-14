# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from freq_filter import freq_filter_sobel, freq_filter_laplacian

def main():
    print("=" * 60)
    print("Experiment 4-2: Frequency Domain Filtering")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))
    lab2_img = os.path.join('D:', '2026第二学期', '2026数媒数字图像处理实验说明', 'Pset4_Fourier and Frequency domain Filtering', '实验4提供的图像', 'moon.png')
    default_img = os.path.join(project_root, 'static', 'imgs', 'test0', '1.bmp')

    img_path = sys.argv[1] if len(sys.argv) > 1 else lab2_img
    if not os.path.exists(img_path):
        img_path = default_img

    img = Image.open(img_path).convert('L')
    input_img = np.array(img, dtype=np.float64)
    H, W = input_img.shape

    print(f"Input: {img_path}")
    print(f"Size: {H}x{W}")
    print("4-2-1: Sobel edge detection in frequency domain...")

    result_sobel = freq_filter_sobel(input_img)

    print("4-2-2: Laplacian sharpening in frequency domain...")

    result_lap = freq_filter_laplacian(input_img, D0=30)

    output_dir = os.path.join(project_root, 'static', 'imgs', 'output')
    os.makedirs(output_dir, exist_ok=True)

    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Experiment 4-2: Frequency Domain Filtering', fontsize=14)

        axes[0].imshow(input_img, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(result_sobel, cmap='gray')
        axes[1].set_title('Sobel (Frequency Domain)')
        axes[1].axis('off')

        axes[2].imshow(result_lap, cmap='gray')
        axes[2].set_title('Laplacian Sharpening (D0=30)')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Cannot display: {e}")

if __name__ == '__main__':
    main()