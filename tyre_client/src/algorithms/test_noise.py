#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from noise import add_gaussian_noise, add_rayleigh_noise, add_salt_pepper_noise, compute_histogram


def main():
    print("=" * 60)
    print("实验5-1：噪声图像生成及其直方图")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))
    output_dir = os.path.join(project_root, 'static', 'imgs', 'output')
    os.makedirs(output_dir, exist_ok=True)

    lab_img = r"D:\2026第二学期\2026数媒数字图像处理实验说明\Pset5_Restoration\实验5提供的图像\test-pattern.tif"
    img_path = sys.argv[1] if len(sys.argv) > 1 else lab_img

    if not os.path.exists(img_path):
        print(f"图像不存在: {img_path}")
        return

    img = np.array(Image.open(img_path).convert('L'))
    print(f"输入图像: {img_path}")
    print(f"图像尺寸: {img.shape[0]}x{img.shape[1]}")

    gaussian_sigma = 25.0
    rayleigh_b = 25.0
    sp_p = 0.05

    print(f"\n生成高斯噪声 (mean=0, sigma={gaussian_sigma}) ...")
    img_gaussian = add_gaussian_noise(img, mean=0, sigma=gaussian_sigma, seed=1)

    print(f"生成瑞利噪声 (a=0, b={rayleigh_b}) ...")
    img_rayleigh = add_rayleigh_noise(img, a=0, b=rayleigh_b, seed=2)

    print(f"生成椒盐噪声 (p={sp_p}) ...")
    img_sp = add_salt_pepper_noise(img, p=sp_p, seed=3)

    results = [
        ('Gaussian', img_gaussian, 'gaussian_noise.png'),
        ('Rayleigh', img_rayleigh, 'rayleigh_noise.png'),
        ('Salt and Pepper', img_sp, 'salt_pepper_noise.png'),
    ]

    for name, im, fname in results:
        save_path = os.path.join(output_dir, fname)
        Image.fromarray(im).save(save_path)
        print(f"结果已保存: {save_path}")

    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle('Experiment 5-1: Noise Images and Their Histograms', fontsize=15)

        for k, (name, im, _) in enumerate(results):
            axes[0, k].imshow(im, cmap='gray', vmin=0, vmax=255)
            axes[0, k].set_title(name, fontsize=13)
            axes[0, k].axis('off')

            hist = compute_histogram(im)
            axes[1, k].bar(np.arange(256), hist, width=1.0, color='black')
            axes[1, k].set_title(f'{name} Histogram', fontsize=12)
            axes[1, k].set_xlim(0, 255)
            axes[1, k].set_xlabel('Gray level')
            axes[1, k].set_ylabel('Pixel count')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"无法显示图像对比图: {e}")

    print("=" * 60)


if __name__ == '__main__':
    main()