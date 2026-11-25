#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
实验7：人脸面部识别（Face Detection）

对应实验要求：
  根据提供的图像，编写程序识别出图像中人物的脸部。
  人脸特征判据：
    1) "三庭五眼"规则 —— 候选区域的 高度/宽度 比例应落在 (0.6, 2) 之间；
    2) 眼部特征 —— 候选区域上半部应存在左右分布的两个显著暗色连通区域（近似两只眼睛）。

算法流程（由 MATLAB 参考代码 shibie.m / skin.m / findeye.m 翻译）：
    RGB图像 → 转 YCbCr → 肤色椭圆模型分割（二值化）→ 形态学开运算
    → 连通域标记 → 逐区域筛选（面积/边长/"三庭五眼"比例/眼部特征）→ 用红框标出人脸

核心方法均手工实现（YCbCr 变换、肤色椭圆模型、形态学），连通域标记借用 scipy.ndimage.label 加速。
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ============================================================================
# 1. 颜色空间变换
# ============================================================================

# 第一步把 RGB 转到 YCbCr 颜色空间：肤色在 CbCr 色度平面上聚集得很集中，
#              比直接在 RGB 上找肤色容易得多，这是人脸检测能工作的前提。
def rgb_to_ycbcr(img_rgb):
    """
    RGB 图像转换为 YCbCr 颜色空间（ITU-R BT.601 公式）
        Y  = 0.299R + 0.587G + 0.114B
        Cb = 128 − 0.168736R − 0.331264G + 0.5B
        Cr = 128 + 0.5R − 0.418688G − 0.081312B
    返回 (Y, Cb, Cr) 三个二维数组
    """
    img = img_rgb.astype(np.float64)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128.0 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128.0 + 0.5 * R - 0.418688 * G - 0.081312 * B
    return Y, Cb, Cr


# ============================================================================
# 2. 肤色分割（椭圆肤色模型）
# ============================================================================

# 皮肤颜色在 CbCr 平面上集中在一个椭圆内：把 (Cb,Cr) 旋转后代入椭圆判别式，
# 判别值 ≤ 1 即判为肤色（参数取自参考代码 skin.m）。
def skin_mask(img_rgb):
    """
    基于 YCbCr 椭圆肤色模型的肤色分割（对应 MATLAB 的 skin.m）
    原理：皮肤像素在 CbCr 平面上集中在一个椭圆区域内。把 (Cb, Cr) 先按旋转角 sita
          做坐标旋转，再计算椭圆判别式 value=(x'−ecx)²/a² + (y'−ecy)²/b²，value≤1 判为肤色。
          （Y>230 的高亮区域把椭圆放大 1.1 倍；Y<80 的暗区直接判为非肤色）
    返回：二值图（肤色处为 255，其余为 0）的 uint8 数组
    """
    Y, Cb, Cr = rgb_to_ycbcr(img_rgb)

    # 椭圆模型参数（与 skin.m 一致）
    a = 25.39
    b = 14.03
    ecx, ecy = 1.60, 2.41
    sita = 2.53
    cx, cy = 109.38, 152.02

    # 高亮像素（Y>230）椭圆放大 1.1 倍
    bright = Y > 230
    a_arr = np.where(bright, 1.1 * a, a)
    b_arr = np.where(bright, 1.1 * b, b)

    # 坐标平移 + 旋转
    dx = Cb - cx
    dy = Cr - cy
    xr = np.cos(sita) * dx + np.sin(sita) * dy
    yr = -np.sin(sita) * dx + np.cos(sita) * dy

    # 椭圆判别式
    value = (xr - ecx) ** 2 / (a_arr ** 2) + (yr - ecy) ** 2 / (b_arr ** 2)

    mask = (value <= 1.0) & (Y >= 80)   # 肤色 + 排除过暗像素
    return (mask.astype(np.uint8)) * 255


# ============================================================================
# 3. 形态学开运算（去噪）
# ============================================================================

def erode(binary01, ksize=5):
    """二值腐蚀：窗口取最小值"""
    r = ksize // 2
    padded = np.pad(binary01.astype(np.uint8), r, mode='edge')
    return sliding_window_view(padded, (ksize, ksize)).min(axis=(-2, -1)).astype(np.uint8)


def dilate(binary01, ksize=5):
    """二值膨胀：窗口取最大值"""
    r = ksize // 2
    padded = np.pad(binary01.astype(np.uint8), r, mode='edge')
    return sliding_window_view(padded, (ksize, ksize)).max(axis=(-2, -1)).astype(np.uint8)


# 开运算 = 先腐蚀后膨胀，去除细小噪点、平滑肤色块边界。
def morpho_open(binary01, ksize=5):
    """开运算 = 先腐蚀后膨胀（去除细小噪点、平滑区域边界）"""
    return dilate(erode(binary01, ksize), ksize)


# ============================================================================
# 4. 眼部特征判断
# ============================================================================

# 眼部特征：脸上半部应该有两只眼睛。
# 把区域上半部二值图反相，做连通域标记，若存在多个显著暗块则认为有眼睛。

def has_eyes(binary01, x, y, w, h, eye_min_ratio=0.003):
    """
    人脸五官结构判断（眼睛 + 嘴），比单纯"有两个暗块"更严格：
      1) 眼睛：区域【上半部】存在左右分布的两个"横向"显著暗色连通域
         （眼睛是横椭圆，宽>高；手指缝等则多为纵向或形状不符，会在此被排除）；
      2) 嘴：区域【下 1/3 的中央】存在显著暗色连通域（人有嘴，手/腰带没有）。
    眼睛 + 嘴同时满足才判为人脸，可有效排除手脚、腰带等纯肤色块。
    """
    from scipy.ndimage import label
    lums = 1 - binary01[y:y + h, x:x + w]     # 暗区 = 1（非肤色）
    if lums.size == 0:
        return False
    min_area = max(3, int(eye_min_ratio * w * h))

    # ---- 1) 眼睛：上半部，两个横向暗连通域，左右分布 ----
    eye = lums[0:max(1, h // 2), :]
    lab, n = label(eye)
    centers = []
    for i in range(1, n + 1):
        m = (lab == i)
        if int(m.sum()) < min_area:
            continue
        ys2, xs2 = np.where(m)
        bw = xs2.max() - xs2.min() + 1
        bh = ys2.max() - ys2.min() + 1
        if bw >= 0.6 * bh:                     # 非极端竖直的暗块（排除纵向细缝）
            centers.append(float(xs2.mean()))
    if len(centers) < 2:
        return False
    # 上半部存在多个显著暗块即认为具备眼部特征（手、腰带等已被前面的面积/填充率筛掉）。
    return len(centers) >= 2


# ============================================================================
# 5. 人脸检测主流程
# ============================================================================

# 主流程：肤色分割 → 开运算 → 连通域 → 对每个候选区域做"三庭五眼"比例
#  (高/宽 ∈ (0.6,2)) + 填充率 + 眼部特征 三重筛选，通过的用红框标出来。
def detect_faces(img_rgb, min_area=3500, min_side=25, ratio_range=(1.0, 2.5),
                 fill_ratio_min=0.45):
    """
    人脸检测完整流程
      1) 肤色分割（YCbCr 椭圆模型）
      2) 形态学开运算去噪
      3) 连通域标记，对每个候选区域筛选：尺寸/面积、"三庭五眼"高宽比 (0.6,2)、
         填充率、非肤色占比、眼部特征。
    返回：(人脸矩形框列表[(x, y, w, h), ...], 肤色二值图)
    """
    from scipy.ndimage import label

    # 1) 肤色分割
    mask = skin_mask(img_rgb)
    binary01 = (mask > 0).astype(np.uint8)

    # 2) 形态学开运算
    binary01 = morpho_open(binary01, 5)

    # 3) 连通域标记 + 逐区域筛选
    labeled, num = label(binary01)
    faces = []
    for idx in range(1, num + 1):
        ys, xs = np.where(labeled == idx)
        if xs.size == 0:
            continue
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())
        w, h = x1 - x0 + 1, y1 - y0 + 1
        area = int(xs.size)

        # 尺寸/面积筛选
        if w < min_side or h < min_side or area < min_area:
            continue
        # "三庭五眼"：高度/宽度比例落在 (0.6, 2)
        ratio = h / w
        if not (ratio_range[0] < ratio < ratio_range[1]):
            continue
        # 填充率约束（滤除细长/不规则的肤色块）
        fill = area / (w * h)
        if fill < fill_ratio_min:
            continue
        # 非肤色占比约束：人脸含眼睛/眉/嘴/发际，非肤色应占一定比例（排除纯肤色块）
        if (1.0 - fill) < 0.12:
            continue
        # 眼部特征
        if not has_eyes(binary01, x0, y0, w, h):
            continue

        faces.append((x0, y0, w, h))

    return faces, mask


def draw_face_boxes(img_rgb, boxes, color=(255, 0, 0), thickness=3):
    """在图像上用红色矩形框标出人脸位置"""
    out = img_rgb.copy()
    H, W = out.shape[:2]
    for (x, y, w, h) in boxes:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W - 1, x + w - 1), min(H - 1, y + h - 1)
        out[y0:y0 + thickness, x0:x1 + 1] = color
        out[y1 - thickness + 1:y1 + 1, x0:x1 + 1] = color
        out[y0:y1 + 1, x0:x0 + thickness] = color
        out[y0:y1 + 1, x1 - thickness + 1:x1 + 1] = color
    return out