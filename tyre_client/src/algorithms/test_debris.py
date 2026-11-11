#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
实验6 测试脚本：胎侧杂物检测

运行方式：
    python test_debris.py                      # 处理"缺陷图像"文件夹中的全部图像
    python test_debris.py <图像路径>            # 处理指定单张图像

输出：
    - 每张图的缺陷二值图（debris_binary_<文件名>.png）
    - 每张图标注缺陷后的图像（debris_marked_<文件名>.png）
    - 汇总对比图（exp6_debris_result.png）
    结果保存到 tyre_client/static/imgs/output/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from debris import detect_debris, column_gray_normalize, to_gray, otsu_threshold, binarize, black_hat

# 实验提供的胎侧缺陷图像目录
LAB6_DIR = r"D:\2026第二学期\2026数媒数字图像处理实验说明\Pset6_Debris detection of tire sidewall\缺陷图像"


def main():
    print("=" * 60)
    print("实验6：胎侧杂物检测算法（Debris Detection）")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))
    output_dir = os.path.join(project_root, 'static', 'imgs', 'output')
    os.makedirs(output_dir, exist_ok=True)

    # 确定要处理的图像列表
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        img_files = [sys.argv[1]]
    else:
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
        img_files = [os.path.join(LAB6_DIR, f) for f in sorted(os.listdir(LAB6_DIR))
                     if f.lower().endswith(exts)]
    print(f"待处理图像数量: {len(img_files)}")

    # 用于汇总展示的前 3 张图
    show_data = []

    for idx, img_path in enumerate(img_files, 1):
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n[{idx}/{len(img_files)}] 处理: {name}")

        # 读取原始彩色图像
        rgb = np.array(Image.open(img_path).convert('RGB'))
        gray = to_gray(rgb)

        # 执行完整检测流程
        binary, marked, regions = detect_debris(
            rgb, min_area=150, max_area=8000, min_h=10, min_w=10, max_ratio=3.0,
            morph_size=5, bg_size=25, edge_margin=20, flatten_block=64,
            min_mean_response=95.0, dark_is_defect=True, verbose=True)

        # 保存二值缺陷图和标记图
        Image.fromarray((binary * 255).astype(np.uint8)).save(
            os.path.join(output_dir, f'debris_binary_{name}.png'))
        Image.fromarray(marked).save(
            os.path.join(output_dir, f'debris_marked_{name}.png'))

        # 打印缺陷信息
        for r in regions:
            print(f"    缺陷区域: 位置(x0,y0,x1,y1)={r['bbox']}, 面积={r['area']} 像素, 响应={r['mean_response']:.1f}")

        # 收集前 3 张用于汇总展示
        if idx <= 3:
            response = black_hat(column_gray_normalize(gray), 25)
            T = otsu_threshold(response)
            bin_otsu = (response > T).astype(np.uint8) * 255   # 二值图放大到 0/255 便于显示
            show_data.append((name, rgb, response, bin_otsu, marked))

    print(f"\n结果已保存到: {output_dir}")

    # ==================== 汇总对比展示 ====================
    if show_data:
        n = len(show_data)
        fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
        if n == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle('Experiment 6: Tire Sidewall Debris Detection', fontsize=15)
        titles = ['Original', 'Black-Hat Response', 'Otsu Binary', 'Detected (Marked)']
        for row, (name, rgb, norm, bin_otsu, marked) in enumerate(show_data):
            for col, (im, title, cmap) in enumerate(zip(
                    [rgb, norm, bin_otsu, marked],
                    titles,
                    [None, 'gray', 'gray', None])):
                axes[row, col].imshow(im, cmap=cmap, vmin=0, vmax=255)
                axes[row, col].set_title(f'{name} - {title}')
                axes[row, col].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'exp6_debris_result.png'), dpi=100)
        plt.show()

    print("\n" + "=" * 60)
    print("实验6 全部完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
