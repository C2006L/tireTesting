#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from gaussian import Gaussian


def main():
    print("=" * 60)
    print("实验3-1：图像高斯滤波处理（Gaussian Smoothing）")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))
    default_img = os.path.join(project_root, 'static', 'imgs', 'test0', '1.bmp')
    lab2_img = os.path.join('D:', os.sep, '2026第二学期', '2026数媒数字图像处理实验说明',
                            'Pset1_Deployment of real-time monitoring system for tire defects',
                            'Tire testing', '实验3提供图像', '3-1', 'a.jpg')

    img_path = sys.argv[1] if len(sys.argv) > 1 else lab2_img

    if not os.path.exists(img_path):
        img_path = default_img

    if len(sys.argv) > 2:
        sigma = float(sys.argv[2])
    else:
        sigma = 1.5

    img = Image.open(img_path).convert('RGB')
    input_img = np.array(img)

    print(f"输入图像: {img_path}")
    print(f"原始尺寸: {input_img.shape[0]}x{input_img.shape[1]}x{input_img.shape[2]}")
    print(f"高斯标准差(sigma): {sigma}")

    window_size = int((6 * sigma - 1) // 2 * 2 + 1)
    if window_size < 3:
        window_size = 3
    print(f"滤波窗口大小: {window_size}x{window_size}")
    print("正在执行高斯滤波（可分离性：先水平后垂直）...")

    output_img = Gaussian(input_img, sigma)

    output_dir = os.path.join(project_root, 'static', 'imgs', 'output')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'gaussian_smoothed.png')

    result = Image.fromarray(output_img)
    result.save(save_path)

    print(f"结果已保存: {save_path}")

    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'Experiment 3-1: Gaussian Smoothing (sigma={sigma}, window={window_size}x{window_size})',
                     fontsize=14)

        axes[0].imshow(input_img)
        axes[0].set_title(f'Original: {input_img.shape[0]}x{input_img.shape[1]}')
        axes[0].axis('off')

        axes[1].imshow(output_img)
        axes[1].set_title(f'Gaussian Smoothed: {output_img.shape[0]}x{output_img.shape[1]}')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"无法显示图像对比图: {e}")

    print("=" * 60)


if __name__ == '__main__':
    main()