#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
实验8：胎冠裂纹检测算法（Radon 变换 + 线密度投影）

算法流程：
  读图 → 裁剪白边 → 灰度归一化 → 高通滤波(去背景) → Radon 变换(各角度投影)
  → 线密度投影(total/cover) → 平滑 → 找"偏离中值最大"的位置(裂纹) → 导数跳变确认
  → 由 (θ, ρ) 写出裂纹直线并在图上标定

核心思想：
  - 高通滤波去掉大尺度背景，只保留细线/边缘结构（这一步很关键，否则背景会把裂纹淹没）；
  - Radon 变换把图像沿各方向投影，若存在与投影方向垂直的直线状裂纹，投影曲线上会出现突出的峰；
  - 线密度投影 = 投影累加 / 覆盖像素数 = 平均灰度，消除直线长短带来的影响；
  - 裂纹位置 = 平滑后的密度曲线偏离中值最大的位置；导数跳变用于确认灰度突变强度。
"""
import numpy as np

# ---- 算法参数 ----
THETAS = np.array([d for d in range(180) if 10 <= d <= 80 or 100 <= d <= 170])
WHITE_LEVEL = 245        # 判定"白边"的灰度阈值
WHITE_FRAC = 0.30        # 整行/列的白色像素比例超过该值视为白边
MARGIN = 12              # 裁边时额外保留的边距
HP_K = 31                # 高通滤波窗口（减去大窗口均值 = 去背景）
SMOOTH_K = 5             # 密度曲线平滑窗口
MIN_COVER = 0.45         # 最小覆盖比例（排除边缘覆盖不足的直线）
SCORE_TH = 2.5           # 判定"有裂纹"的线密度评分阈值


# ------------------------------------------------------------------
# 1. 预处理：裁剪白边 / 归一化 / 高通滤波
#先裁剪：按行/列的白色像素比例找到有效区域、去掉图像白边。
# ------------------------------------------------------------------
def crop_frame(img):
    """
    裁剪白边：按行、列的白色像素比例找出有效区域，并去掉 MARGIN 边距
    返回：(裁剪后的子图, 起始行 r0, 起始列 c0)
    """
    white = img > WHITE_LEVEL
    rows = np.where(white.mean(axis=1) < WHITE_FRAC)[0]
    cols = np.where(white.mean(axis=0) < WHITE_FRAC)[0]
    if len(rows) == 0 or len(cols) == 0:
        return img, 0, 0
    r0, r1 = int(rows[0]) + MARGIN, int(rows[-1]) - MARGIN
    c0, c1 = int(cols[0]) + MARGIN, int(cols[-1]) - MARGIN
    if r1 <= r0 or c1 <= c0:
        return img, 0, 0
    return img[r0:r1 + 1, c0:c1 + 1], r0, c0


def normalize(img):
    """灰度线性拉伸到 [0, 255]"""
    lo, hi = img.min(), img.max()
    return (img - lo) / max(hi - lo, 1e-12) * 255.0


def box_blur(img, k):
    """方框均值滤波（用于高通：原图 − 大窗口均值，去掉大尺度背景）"""
    pad = np.pad(img, k // 2, mode="reflect")
    win = np.lib.stride_tricks.sliding_window_view(pad, (k, k))
    return win.mean(axis=(-2, -1))


def smooth1d(values, k):
    """一维平滑"""
    pad = np.pad(values, k // 2, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, k)
    return win.mean(axis=-1)


# ------------------------------------------------------------------
# 2. Radon 变换 + 线密度投影
# ------------------------------------------------------------------
def radon(img, thetas=None):
    """
    Radon 变换：对每个角度，把图像投影到"法向偏移 ρ"上
    同时记录每个位置的灰度累加和覆盖像素数
    """
    if thetas is None:
        thetas = THETAS
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    y0 = yy - (h - 1) / 2.0
    x0 = xx - (w - 1) / 2.0
    shift = int(np.ceil(0.5 * np.hypot(h, w))) + 2
    nbins = 2 * shift + 1
    total = np.zeros((len(thetas), nbins))
    cover = np.zeros((len(thetas), nbins))
    for i, deg in enumerate(thetas):
        t = np.deg2rad(deg)
        rho = x0 * np.cos(t) + y0 * np.sin(t)
        idx = np.round(rho).astype(np.int64) + shift
        flat = idx.ravel()
        total[i] = np.bincount(flat, weights=img.ravel(), minlength=nbins)
        cover[i] = np.bincount(flat, minlength=nbins)
    return total, cover, shift


def line_density(total, cover):
    """线密度投影 = 累加 / 覆盖数（平均灰度），消除直线长短的影响"""
    return total / np.maximum(cover, 1.0)


# ------------------------------------------------------------------
# 3. 裂纹定位
# ------------------------------------------------------------------
def find_best_line(density, cover, shift):
    """
    在所有角度中找"偏离中值最大"的那条直线（裂纹候选）
    再算一阶导数，用导数跳变确认灰度突变强度
    """
    smooth = np.apply_along_axis(smooth1d, 1, density, SMOOTH_K)
    best = None
    for i in range(smooth.shape[0]):
        med = float(np.median(smooth[i]))
        usable = cover[i] >= MIN_COVER * cover[i].max()
        candidate = np.where(usable, smooth[i] - med, -1e9)
        rho_index = int(np.argmax(candidate))
        if best is None or candidate[rho_index] > best[0]:
            best = (float(candidate[rho_index]), i, rho_index)
    score, theta_index, rho_index = best
    curve = smooth[theta_index]
    deriv = np.gradient(curve)
    lo = max(0, rho_index - 4)
    hi = min(len(deriv), rho_index + 5)
    jump = float(np.max(np.abs(deriv[lo:hi])))
    return score, theta_index, rho_index - shift, jump, curve, deriv, rho_index


def line_endpoints(h, w, deg, rho):
    """由 (角度, 偏移) 求裂纹直线与图像边界的交点"""
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    points = []
    if abs(s) > 1e-9:
        for x in (0.0, float(w - 1)):
            y = cy + (rho - (x - cx) * c) / s
            if -1.0 <= y <= h:
                points.append((x, y))
    if abs(c) > 1e-9:
        for y in (0.0, float(h - 1)):
            x = cx + (rho - (y - cy) * s) / c
            if -1.0 <= x <= w:
                points.append((x, y))
    return points


# ------------------------------------------------------------------
# 4. 对外接口
# ------------------------------------------------------------------
def detect_crack(img, thetas=None):
    """
      把上面串起来；评分 ≥2.5 判定为'有裂纹'
    """
    if thetas is None:
        thetas = THETAS
    sub, r0, c0 = crop_frame(img)
    norm = normalize(sub)
    hp = norm - box_blur(norm, HP_K)                  # 高通滤波：去背景、留细线结构
    total, cover, shift = radon(hp, thetas)
    density = line_density(total, cover)
    score, theta_index, rho, jump, curve, deriv, peak_index = find_best_line(density, cover, shift)
    deg = int(thetas[theta_index])
    has_crack = score >= SCORE_TH

    sub_h, sub_w = sub.shape
    pts = line_endpoints(sub_h, sub_w, deg, rho)
    cracks = []
    if len(pts) >= 2:
        (x1, y1), (x2, y2) = pts[0], pts[-1]
        cracks.append({
            'angle': float(deg), 'offset': float(rho), 'score': float(score),
            'jump': float(jump),
            'p1': (x1 + c0, y1 + r0),                 # 换算回原图坐标
            'p2': (x2 + c0, y2 + r0),
            'has_crack': bool(has_crack),
        })

    info = {'density': density, 'cover': cover, 'curve': curve, 'deriv': deriv,
            'theta': deg, 'rho': rho, 'score': float(score), 'jump': float(jump),
            'has_crack': bool(has_crack), 'sub': sub, 'r0': r0, 'c0': c0,
            'peak_index': peak_index}
    return cracks, info


def draw_crack_lines(img_rgb, cracks, thickness=2):
    """在图像上标定裂纹直线：检出裂纹用红色，未检出用蓝色（与参考实现一致）"""
    out = img_rgb.copy()
    H, W = out.shape[:2]
    for c in cracks:
        color = (255, 0, 0) if c.get('has_crack', True) else (0, 160, 255)
        (x1, y1), (x2, y2) = c['p1'], c['p2']
        n = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
        xs = np.linspace(x1, x2, n)
        ys = np.linspace(y1, y2, n)
        for dx in range(-thickness, thickness + 1):
            for dy in range(-thickness, thickness + 1):
                xi = np.round(xs + dx).astype(int)
                yi = np.round(ys + dy).astype(int)
                valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
                out[yi[valid], xi[valid]] = color
    return out