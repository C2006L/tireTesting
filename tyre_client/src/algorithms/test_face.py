#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
实验7 测试脚本：人脸面部识别

运行方式：
    python test_face.py                    # 处理实验提供的两张人脸图
    python test_face.py <图像路径>          # 处理指定图像

输出：
    - 人脸检测结果图（原图 + 红框）face_result_<文件名>.png
    - 肤色分割中间结果 face_skin_<文件名>.png
    结果保存到 tyre_client/static/imgs/output/exp7_人脸识别/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from face import detect_faces, draw_face_boxes

# 实验提供的人脸图像目录
LAB7_DIR = r"D:\2026第二学期\2026数媒数字图像处理实验说明\Pset7_Recognition\提供图像"


def main():
    print("=" * 60)
    print("实验7：人脸面部识别（Face Detection）")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))
    output_dir = os.path.join(project_root, 'static', 'imgs', 'output', 'exp7_人脸识别')
    os.makedirs(output_dir, exist_ok=True)

    # 确定要处理的图像
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        img_files = [sys.argv[1]]
    else:
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        img_files = [os.path.join(LAB7_DIR, f) for f in sorted(os.listdir(LAB7_DIR))
                     if f.lower().endswith(exts)]

    show_data = []
    for idx, img_path in enumerate(img_files, 1):
        name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n[{idx}/{len(img_files)}] 处理: {name}")

        img = np.array(Image.open(img_path).convert('RGB'))

        # 人脸检测
        faces, mask = detect_faces(img)
        print(f"    检测到 {len(faces)} 个人脸区域")
        for (x, y, w, h) in faces:
            print(f"      人脸框: (x={x}, y={y}, w={w}, h={h}), 高宽比={h/w:.2f}")

        # 画框并保存
        result = draw_face_boxes(img, faces)
        Image.fromarray(result).save(os.path.join(output_dir, f'face_result_{name}.png'))
        Image.fromarray(mask).save(os.path.join(output_dir, f'face_skin_{name}.png'))
        show_data.append((name, img, mask, result))

    print(f"\n结果已保存到: {output_dir}")

    # 汇总展示
    if show_data:
        n = len(show_data)
        fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
        if n == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle('Experiment 7: Face Detection', fontsize=15)
        titles = ['Original', 'Skin Mask', 'Detected Face']
        for row, (name, img, mask, result) in enumerate(show_data):
            for col, (im, title, cmap) in enumerate(zip([img, mask, result], titles, [None, 'gray', None])):
                axes[row, col].imshow(im, cmap=cmap)
                axes[row, col].set_title(f'{name} - {title}')
                axes[row, col].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'exp7_face_result.png'), dpi=100)
        plt.show()

    print("\n" + "=" * 60)
    print("实验7 全部完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
