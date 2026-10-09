#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验2-1 测试脚本：图像缩放

用法:
    python test_scale.py [图像路径] [sx] [sy]
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(__file__))
from scale import Scale

try:
    from PIL import Image
    USE_PIL = True
except ImportError:
    USE_PIL = False


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))
    default_img = os.path.join(project_root, 'static', 'imgs', 'test0', '1.bmp')

    img_path = sys.argv[1] if len(sys.argv) > 1 else default_img
    sx = float(sys.argv[2]) if len(sys.argv) > 2 else 1.8
    sy = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8

    print("=" * 60)
    print("实验2-1：图像缩放（双线性插值法）")
    print("=" * 60)

    if USE_PIL:
        img = Image.open(img_path)
        input_img = np.array(img)
    else:
        import cv2
        input_img = cv2.imread(img_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    if input_img.ndim == 2:
        input_img = np.stack([input_img] * 3, axis=-1)

    print(f"输入图像: {img_path}")
    print(f"原始尺寸: {input_img.shape[0]} x {input_img.shape[1]} x {input_img.shape[2]}")
    print(f"缩放倍数: sx={sx}, sy={sy}")

    output_img = Scale(input_img, sx, sy)

    print(f"输出尺寸: {output_img.shape[0]} x {output_img.shape[1]} x {output_img.shape[2]}")

    output_dir = os.path.join(project_root, 'static', 'imgs', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'sizechange.png')

    if USE_PIL:
        result = Image.fromarray(output_img)
        result.save(output_path)
    else:
        import cv2
        output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_bgr)

    print(f"结果已保存: {output_path}")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(input_img)
        axes[0].set_title(f'Original: {input_img.shape[0]}x{input_img.shape[1]}')
        axes[0].axis('off')
        axes[1].imshow(output_img)
        axes[1].set_title(f'Scaled: {output_img.shape[0]}x{output_img.shape[1]} (sx={sx}, sy={sy})')
        axes[1].axis('off')
        plt.suptitle('Experiment 2-1: Image Scaling (Bilinear Interpolation)')
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("(安装matplotlib可可视化对比结果)")


if __name__ == '__main__':
    main()