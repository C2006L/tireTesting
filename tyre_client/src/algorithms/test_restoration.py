#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
实验5 测试脚本：图像修复

运行方式：
    python test_restoration.py
    或指定图像：python test_restoration.py <图像路径>

输出：
    - 5-1：三种噪声图像（高斯/瑞利/椒盐）及其直方图
    - 5-2-1：椒盐噪声图像 -> 7×7 中值滤波 vs 自适应中值滤波 对比
    - 5-2-2：book-cover 图像 -> 运动模糊+高斯噪声 退化 -> 维纳滤波复原
    - 结果图片保存到 tyre_client/static/imgs/output/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from restoration import (
    add_gaussian_noise, add_rayleigh_noise, add_salt_pepper_noise,
    compute_histogram, median_filter, adaptive_median_filter,
    apply_motion_blur, wiener_filter,
)

# 实验提供的图像目录
LAB5_DIR = r"D:\2026第二学期\2026数媒数字图像处理实验说明\Pset5_Restoration\实验5提供的图像"


def load_gray(path):
    """读取图像并转成灰度 numpy 数组（float64）"""
    return np.array(Image.open(path).convert('L'), dtype=np.float64)


def main():
    print("=" * 60)
    print("实验5：图像修复（Image Restoration）")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))
    output_dir = os.path.join(project_root, 'static', 'imgs', 'output')
    os.makedirs(output_dir, exist_ok=True)

    # ==================== 5-1 噪声图像生成及其直方图 ====================
    print("\n[5-1] 噪声图像生成及其直方图")
    img_51_path = os.path.join(LAB5_DIR, 'test-pattern.tif')
    if not os.path.exists(img_51_path):
        img_51_path = os.path.join(LAB5_DIR, 'circuitboard.tif')
    img_51 = load_gray(img_51_path)
    print(f"    输入图像: {img_51_path}, 尺寸: {img_51.shape}")

    # 分别生成三种噪声图像
    np.random.seed(42)  # 固定随机种子，保证结果可复现
    noisy_gauss = add_gaussian_noise(img_51, mean=0, sigma=15)
    noisy_rayleigh = add_rayleigh_noise(img_51, sigma=20)
    noisy_sp = add_salt_pepper_noise(img_51, pa=0.05, pb=0.05)

    # 计算各噪声图像的直方图（按顺序存放，方便与图像一一对应）
    hist_list = [
        compute_histogram(img_51),          # 原图
        compute_histogram(noisy_gauss),     # 高斯噪声
        compute_histogram(noisy_rayleigh),  # 瑞利噪声
        compute_histogram(noisy_sp),        # 椒盐噪声
    ]

    # 保存噪声图像
    Image.fromarray(noisy_gauss.astype(np.uint8)).save(
        os.path.join(output_dir, 'noise_gaussian.png'))
    Image.fromarray(noisy_rayleigh.astype(np.uint8)).save(
        os.path.join(output_dir, 'noise_rayleigh.png'))
    Image.fromarray(noisy_sp.astype(np.uint8)).save(
        os.path.join(output_dir, 'noise_saltpepper.png'))
    print(f"    噪声图像已保存到: {output_dir}")

    # 绘图：上排为图像，下排为对应直方图
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor='black')
    fig.suptitle('Experiment 5-1: Noise Generation & Histograms', fontsize=15, color='white')

    imgs = [img_51, noisy_gauss, noisy_rayleigh, noisy_sp]
    names = ['Original', 'Gaussian Noise', 'Rayleigh Noise', 'Salt & Pepper']

    def _norm_hist(h, bins=64):
        """
        统计灰度直方图并归一化：
          - 以"中间区域的最大峰"为归一化基准（排除紧贴边界 0/255 的堆积柱），
            使主要分布的峰顶满纵轴，分布形状清晰饱满；
          - 超过 1.0 的边界堆积柱裁剪到 1.0。
        （测试图背景为大面积灰度0，加噪后在0处堆积，若直接按全局最大值归一化会把
          中间的主峰压得很矮，故采用此归一化方式以便观察分布形状）
        """
        cnt = np.histogram(np.clip(h, 0, 255), bins=bins, range=(0, 256))[0].astype(np.float64)
        inner_max = cnt[2:-2].max()
        base = inner_max if inner_max > 0 else cnt.max()
        return np.clip(cnt / base, 0, 1.0)

    for k in range(4):
        axes[0, k].imshow(imgs[k], cmap='gray', vmin=0, vmax=255)
        axes[0, k].set_title(names[k], color='white')
        axes[0, k].axis('off')

        # 下排：对应的灰度直方图（64 个 bin 归一化后柱状绘制，黑底白柱、峰值顶满）
        hn = _norm_hist(imgs[k])
        bw = 256.0 / len(hn)                       # 每个柱对应的灰度宽度
        axes[1, k].bar(np.arange(len(hn)) * bw + bw / 2, hn, width=bw * 0.9, color='white')
        axes[1, k].set_title(f'Histogram - {names[k]}', color='white', fontsize=11)
        axes[1, k].set_xlim(0, 255)
        axes[1, k].set_ylim(0, 1.05)               # 峰值顶到顶部
        axes[1, k].set_facecolor('black')          # 黑色背景
        axes[1, k].tick_params(colors='white', labelsize=8)
        axes[1, k].set_ylabel('normalized count', color='white', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp5_1_noise_hist.png'), dpi=100, facecolor='black')
    plt.show()

    # ==================== 5-2-1 自适应中值滤波 vs 7×7 中值滤波 ====================
    print("\n[5-2-1] 自适应中值滤波 vs 7×7 中值滤波")
    img_521_path = os.path.join(LAB5_DIR, 'circuitboard-saltandpep.tif')
    if not os.path.exists(img_521_path):
        img_521_path = os.path.join(LAB5_DIR, 'circuitboard.tif')
    img_521 = load_gray(img_521_path)
    print(f"    输入图像: {img_521_path}, 尺寸: {img_521.shape}")

    print("    正在执行 7×7 中值滤波...")
    med7 = median_filter(img_521, ksize=7)

    print("    正在执行自适应中值滤波（smax=7，可能稍慢，请耐心等待）...")
    adaptive = adaptive_median_filter(img_521, smax=7)

    Image.fromarray(med7.astype(np.uint8)).save(
        os.path.join(output_dir, 'median_7x7.png'))
    Image.fromarray(adaptive.astype(np.uint8)).save(
        os.path.join(output_dir, 'adaptive_median.png'))
    print(f"    结果已保存到: {output_dir}")

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle('Experiment 5-2-1: Median vs Adaptive Median Filter', fontsize=15)
    for ax, im, title in zip(
            axes2,
            [img_521, med7, adaptive],
            [f'Noisy (Salt & Pepper)', '7x7 Median Filter', 'Adaptive Median Filter']):
        ax.imshow(im, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp5_2_1_median_compare.png'), dpi=100)
    plt.show()

    # ==================== 5-2-2 维纳滤波复原 ====================
    print("\n[5-2-2] 运动模糊 + 高斯噪声 -> 维纳滤波")
    img_522_path = os.path.join(LAB5_DIR, 'book-cover.tif')
    if not os.path.exists(img_522_path):
        img_522_path = os.path.join(LAB5_DIR, 'book-cover-blurred.tif')
    img_522 = load_gray(img_522_path)
    print(f"    输入图像: {img_522_path}, 尺寸: {img_522.shape}")

    # 1) 运动模糊（a=b=0.1, T=1）
    blurred, H = apply_motion_blur(img_522, a=0.1, b=0.1, T=1.0)
    # 2) 叠加均值为 0、方差为 650 的加性高斯噪声（sigma = sqrt(650) ≈ 25.5）
    sigma_noise = float(np.sqrt(650))
    np.random.seed(7)
    degraded = np.clip(blurred + sigma_noise * np.random.randn(*blurred.shape), 0, 255)
    print(f"    已生成退化图像（运动模糊 + 高斯噪声, sigma={sigma_noise:.2f}）")

    # 3) 维纳滤波复原
    restored = wiener_filter(degraded, H, K=0.02)

    Image.fromarray(degraded.astype(np.uint8)).save(
        os.path.join(output_dir, 'blurred_noisy.png'))
    Image.fromarray(restored.astype(np.uint8)).save(
        os.path.join(output_dir, 'wiener_restored.png'))
    print(f"    结果已保存到: {output_dir}")

    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
    fig3.suptitle('Experiment 5-2-2: Wiener Filter (Motion Blur + Gaussian Noise)',
                  fontsize=15)
    for ax, im, title in zip(
            axes3,
            [img_522, degraded, restored],
            ['Original', 'Degraded (Blur + Noise)', 'Wiener Restored']):
        ax.imshow(im, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp5_2_2_wiener.png'), dpi=100)
    plt.show()

    print("\n" + "=" * 60)
    print("实验5 全部完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
