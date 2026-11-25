#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
实验8 测试脚本：胎冠裂纹检测

运行方式：
    python test_crack.py                # 处理前 5 张
    python test_crack.py all            # 处理全部图像
    python test_crack.py <图像路径>      # 处理单张

输出（保存到 tyre_client/static/imgs/output/exp8_胎冠裂纹检测/）：
    - 8-1_<名字>_标定结果.png   裂纹标定结果（红线=有裂纹，蓝线=未检出）
    - 8-1_<名字>_线密度曲线.png  线密度曲线与一阶导数
    - 8-1_汇总网格.png          多张结果的汇总
"""
import sys
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crack import detect_crack, draw_crack_lines

LAB8_DIR = r"D:\2026第二学期\2026数媒数字图像处理实验说明\Pset8_Crack detection of tire crown\图像数据"


def get_font(size):
    """加载中文字体"""
    for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        try:
            return ImageFont.truetype(os.path.join(r"C:\Windows\Fonts", name), size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_curve(curve, deriv, peak_index, title):
    """绘制线密度曲线（黑）+ 一阶导数（蓝）+ 裂纹位置（红竖线）"""
    width, height = 900, 320
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = get_font(18)
    n = len(curve)
    lo, hi = float(curve.min()), float(curve.max())
    span = max(hi - lo, 1e-6)
    for i in range(1, n):
        x0 = int((i - 1) / (n - 1) * (width - 1))
        x1 = int(i / (n - 1) * (width - 1))
        y0 = int(height - 40 - (curve[i - 1] - lo) / span * (height - 90))
        y1 = int(height - 40 - (curve[i] - lo) / span * (height - 90))
        draw.line([(x0, y0), (x1, y1)], fill=(0, 0, 0), width=2)
    px = int(peak_index / max(n - 1, 1) * (width - 1))
    draw.line([(px, 20), (px, height - 30)], fill=(255, 0, 0), width=2)
    dmax = float(np.max(np.abs(deriv))) or 1.0
    for i in range(1, n):
        x0 = int((i - 1) / (n - 1) * (width - 1))
        x1 = int(i / (n - 1) * (width - 1))
        base = height - 20
        y0 = int(base - deriv[i - 1] / dmax * 26)
        y1 = int(base - deriv[i] / dmax * 26)
        draw.line([(x0, y0), (x1, y1)], fill=(0, 120, 255), width=1)
    draw.text((8, 4), title, font=font, fill=(0, 0, 0))
    draw.text((8, height - 24), "黑：线密度曲线   蓝：一阶导数   红：裂纹位置", font=font, fill=(90, 90, 90))
    return np.array(canvas, dtype=np.uint8)


def main():
    print("=" * 72)
    print("实验8：胎冠裂纹检测（Radon 变换 + 线密度投影）")
    print("=" * 72)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))
    out_dir = os.path.join(project_root, 'static', 'imgs', 'output', 'exp8_胎冠裂纹检测')
    os.makedirs(out_dir, exist_ok=True)

    # 选择图像
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        files = [sys.argv[1]]
    else:
        names = [n for n in sorted(os.listdir(LAB8_DIR))
                 if n.lower().endswith((".bmp", ".jpg", ".png", ".tif"))]
        if not (len(sys.argv) > 1 and sys.argv[1] == 'all'):
            names = names[:5]
        files = [os.path.join(LAB8_DIR, n) for n in names]

    print("共 %d 张图像\n" % len(files))
    print("%-34s %6s %6s %8s %8s %s" % ("图像", "角度", "偏移", "线密度", "跳变", "判定"))

    results = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        gray = np.array(Image.open(path).convert("L"), dtype=np.float64)
        rgb = np.stack([gray] * 3, axis=2).astype(np.uint8)

        cracks, info = detect_crack(gray)
        ok = info['has_crack']
        print("%-34s %6d %6d %8.2f %8.2f %s" % (
            name, info['theta'], info['rho'], info['score'], info['jump'],
            "有裂纹" if ok else "未检出"))

        # 标定结果图
        marked = draw_crack_lines(rgb, cracks)
        Image.fromarray(marked).save(os.path.join(out_dir, "8-1_%s_标定结果.png" % name))
        # 线密度曲线图
        curve_img = draw_curve(info['curve'], info['deriv'], info['peak_index'],
                               "%s  θ=%d°" % (name, info['theta']))
        Image.fromarray(curve_img).save(os.path.join(out_dir, "8-1_%s_线密度曲线.png" % name))
        results.append((name, marked, info['theta'], ok))

    hit = sum(1 for r in results if r[3])
    print("-" * 72)
    print("判定有裂纹 %d / %d 张" % (hit, len(results)))

    # 汇总网格（最多 9 张）
    picked = results[:9]
    if picked:
        cell = 320
        gw = 3
        gh = (len(picked) + gw - 1) // gw
        sheet = Image.new("RGB", (gw * cell, gh * (cell + 24)), (255, 255, 255))
        draw = ImageDraw.Draw(sheet)
        font = get_font(16)
        for i, (name, marked, theta, ok) in enumerate(picked):
            im = Image.fromarray(marked).resize((cell, cell), Image.BILINEAR)
            gx = (i % gw) * cell
            gy = (i // gw) * (cell + 24)
            sheet.paste(im, (gx, gy + 24))
            draw.text((gx + 4, gy + 4), "%s  θ=%d" % (name, theta), font=font, fill=(0, 0, 0))
        sheet.save(os.path.join(out_dir, "8-1_汇总网格.png"))

    print("[完成] 结果目录: %s" % out_dir)
    print("=" * 72)


if __name__ == '__main__':
    main()