#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from noise import add_gaussian_noise
from restore import median_filter, adaptive_median_filter, motion_blur_kernel, wiener_filter


def main():
    print("=" * 60)
    print("实验5-2：含噪声图像处理")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))
    output_dir = os.path.join(project_root, 'static', 'imgs', 'output')
    os.makedirs(output_dir, exist_ok=True)

    img_dir = r"D:\2026第二学期\2026数媒数字图像处理实验说明\Pset5_Restoration\实验5提供的图像"
    sp_path = os.path.join(img_dir, 'circuitboard-saltandpep.tif')
    blur_path = os.path.join(img_dir, 'book-cover-blurred.tif')

    for p in (sp_path, blur_path):
        if not os.path.exists(p):
            print(f"图像不存在: {p}")
            return

    # ---------- 5-2-1 自适应中值滤波 vs 7x7中值滤波 ----------
    print("\n[5-2-1] 自适应中值滤波 vs 7x7 中值滤波")
    sp_img = np.array(Image.open(sp_path).convert('L'), dtype=np.float64)
    print(f"输入图像: {sp_path}, 尺寸: {sp_img.shape}")

    print("执行 7x7 中值滤波 ...")
    med7 = median_filter(sp_img, size=7)

    print("执行自适应中值滤波 (Smax=7) ...")
    amf = adaptive_median_filter(sp_img, smax=7)

    Image.fromarray(med7.astype(np.uint8)).save(os.path.join(output_dir, 'median7.png'))
    Image.fromarray(amf.astype(np.uint8)).save(os.path.join(output_dir, 'adaptive_median.png'))
    print(f"结果已保存到: {output_dir}")

    # ---------- 5-2-2 运动模糊 + 高斯噪声 -> 维纳滤波 ----------
    print("\n[5-2-2] 维纳滤波复原（运动模糊 a=b=0.1, T=1 + 高斯噪声 var=650）")
    blurred = np.array(Image.open(blur_path).convert('L'), dtype=np.float64)
    print(f"输入图像: {blur_path}, 尺寸: {blurred.shape}")

    noise_var = 650.0
    noisy = add_gaussian_noise(blurred, mean=0, sigma=np.sqrt(noise_var), seed=4).astype(np.float64)

    Hf = motion_blur_kernel(blurred.shape[0], blurred.shape[1], a=0.1, b=0.1, T=1.0)

    K = 650.0 / (np.var(blurred) + 1e-10)
    K = float(np.clip(K, 1e-4, 0.05))
    print(f"噪信功率比 K = {K:.4f}")

    print("执行维纳滤波 ...")
    restored = wiener_filter(noisy, Hf, K=K)

    Image.fromarray(noisy.astype(np.uint8)).save(os.path.join(output_dir, 'blurred_noisy.png'))
    Image.fromarray(restored.astype(np.uint8)).save(os.path.join(output_dir, 'wiener_restored.png'))
    print(f"结果已保存到: {output_dir}")

    # ---------- 显示 ----------
    try:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Experiment 5-2: Noise Image Processing', fontsize=15)

        axes[0, 0].imshow(sp_img, cmap='gray', vmin=0, vmax=255)
        axes[0, 0].set_title('Salt & Pepper Noise (P=0.25)')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(med7, cmap='gray', vmin=0, vmax=255)
        axes[0, 1].set_title('7x7 Median Filter')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(amf, cmap='gray', vmin=0, vmax=255)
        axes[0, 2].set_title('Adaptive Median Filter (Smax=7)')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(noisy, cmap='gray', vmin=0, vmax=255)
        axes[1, 0].set_title('Motion Blur + Gaussian Noise (var=650)')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(restored, cmap='gray', vmin=0, vmax=255)
        axes[1, 1].set_title(f'Wiener Filter Restored (K={K:.3f})')
        axes[1, 1].axis('off')

        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"无法显示图像对比图: {e}")

    print("=" * 60)


if __name__ == '__main__':
    main()