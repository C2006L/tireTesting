# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from freq_filter import freq_filter_sobel, freq_filter_laplacian

def main(convolve=None):
    print("=" * 60)
    print("Experiment 4-2: Frequency Domain Filtering")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))

    house_img_path = r"D:\2026第二学期\2026数媒数字图像处理实验说明\Pset4_Fourier and Frequency domain Filtering\实验4提供的图像\building-600by600.tif"
    moon_img_path = r"D:\2026第二学期\2026数媒数字图像处理实验说明\Pset4_Fourier and Frequency domain Filtering\实验4提供的图像\blurry-moon.tif"
    default_img = os.path.join(project_root, 'static', 'imgs', 'test0', '1.bmp')

    # ========== 4-2-1: Sobel频域滤波 (House图像) ==========
    print("\n4-2-1: Sobel Edge Detection on House Image")

    if os.path.exists(house_img_path):
        img_path_sobel = house_img_path
    else:
        img_path_sobel = default_img

    img_house = Image.open(img_path_sobel).convert('L')
    input_house = np.array(img_house, dtype=np.float64)
    print(f"Input: {img_path_sobel}, Size: {input_house.shape}")

    # 频域Sobel
    result_sobel_freq = freq_filter_sobel(input_house)

    # 空域Sobel对比
    sobel_kernel = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]], dtype=np.float64)
    from scipy.ndimage import convolve
    result_sobel_spatial = convolve(input_house, sobel_kernel)
    result_sobel_spatial = np.abs(result_sobel_spatial)
    s_min, s_max = result_sobel_spatial.min(), result_sobel_spatial.max()
    if s_max - s_min > 1e-10:
        result_sobel_spatial = (result_sobel_spatial - s_min) / (s_max - s_min) * 255.0
    result_sobel_spatial = np.clip(result_sobel_spatial, 0, 255)

    # ========== 4-2-2: Laplacian锐化 (Moon图像) ==========
    print("4-2-2: Laplacian Sharpening on Moon Image")

    if os.path.exists(moon_img_path):
        img_path_lap = moon_img_path
    else:
        img_path_lap = default_img

    img_moon = Image.open(img_path_lap).convert('L')
    input_moon = np.array(img_moon, dtype=np.float64)
    print(f"Input: {img_path_lap}, Size: {input_moon.shape}")

    # Laplacian锐化 - 调整参数让效果更明显
    result_lap = freq_filter_laplacian(input_moon, D0=25, strength=2.0)

    output_dir = os.path.join(project_root, 'static', 'imgs', 'output')
    os.makedirs(output_dir, exist_ok=True)

    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Experiment 4-2: Frequency Domain Filtering', fontsize=16)

        # 第一行：Sobel (House)
        axes[0, 0].imshow(result_sobel_freq, cmap='gray')
        axes[0, 0].set_title('Frequency Domain Sobel\n(House 600x600)')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(result_sobel_spatial, cmap='gray')
        axes[0, 1].set_title('Spatial Domain Sobel\n(Direct Convolution)')
        axes[0, 1].axis('off')

        # 第二行：Laplacian (Moon)
        axes[1, 0].imshow(input_moon, cmap='gray')
        axes[1, 0].set_title('Original Blurry Moon')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(result_lap, cmap='gray')
        axes[1, 1].set_title('Laplacian Sharpened Moon\n(D0=25, Strength=2.0)')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Cannot display: {e}")

if __name__ == '__main__':
    main()