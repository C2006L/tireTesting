#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
实验6：胎侧杂物检测算法（Debris Detection of Tire Sidewall）

对应实验要求：
  基于轮胎X光图像，设计并实现一种能够检测胎侧杂物缺陷的图像处理算法：
    灰度值调整 -> 最大类间方差(Otsu)阈值分割 -> 形态学处理 -> 标记缺陷

  算法流程：
    输入胎侧图像 -> 灰度值调整 -> 最大类间方差阈值分割 -> 形态学滤波与膨胀运算 -> 输出缺陷检测结果

  输出：
    1) 胎侧杂物缺陷的检测结果图像（二值图）
    2) 标定缺陷后的图像（在原图上用红色矩形框标记缺陷位置）

关键设计说明：
  实验提供的胎侧 X 光图像存在两类"背景干扰"：
    (1) 胎壁厚度导致的列方向灰度不均（出现竖向暗带）—— 用"逐列灰度归一化"消除；
    (2) 帘布层产生的横向细密条纹 —— 用"黑帽变换(black-hat)"抑制，同时突出比背景暗的孤立小斑点(缺陷)。
  因此"灰度值调整"这一步 = 逐列灰度归一化 + 黑帽变换背景抑制；
  再经 Otsu 阈值分割、形态学处理与面积/形状过滤，最终得到缺陷区域。

  核心算法（归一化、Otsu、腐蚀/膨胀、黑帽变换）均手工实现，不直接调用成型滤波函数。
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ============================================================================
# 1. 灰度值调整（逐列灰度归一化 + 背景抑制）
# ============================================================================

def to_gray(img):
    """将 RGB 图像转换为灰度图（加权平均法：0.299R + 0.587G + 0.114B）"""
    if img.ndim == 3:
        return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return img.astype(np.float64)


def column_gray_normalize(gray):
    """
    逐列灰度归一化（灰度值调整第 1 步）
    目的：胎壁厚度不同 -> X 射线吸收不同 -> 图像列方向灰度不均（出现"竖向暗带"）。
          通过逐列归一化，使每列灰度水平趋于一致，从而消除厚度差异带来的灰度波动。

    公式：out(i, j) = gray(i, j) · (全局平均灰度 / 第 j 列平均灰度)
    """
    col_mean = gray.mean(axis=0)          # 每一列的平均灰度（表征列方向的灰度水平）
    global_mean = gray.mean()             # 整幅图像的平均灰度（全局参考值）
    col_mean_safe = np.where(col_mean < 1e-6, 1e-6, col_mean)
    out = gray * (global_mean / col_mean_safe)
    return np.clip(out, 0, 255)


def block_gray_normalize(gray, block=64):
    """
    分块灰度归一化（二维背景估计，灰度值调整的增强步骤）
    目的：逐列归一化只能消除"列方向"的整体灰度差异，对于二维的大尺度灰度不均
          （例如图像左侧一条宽达上百像素的竖向暗带、以及整体亮度渐变）效果有限。
          本步骤把图像分成 block×block 的块，用每块的均值估计局部背景，
          再上采样回原尺寸，最后用 out = gray · (全局均值 / 背景) 做归一化，
          可同时消除行、列两个方向的大尺度灰度不均，使背景趋于均匀。

    参数：block 为分块大小（应大于缺陷尺寸，小于背景不均区域的尺度）
    """
    from PIL import Image as _PILImage
    h, w = gray.shape
    bh, bw = max(1, h // block), max(1, w // block)
    gh, gw = bh * block, bw * block
    # 1) 分块求均值，得到低分辨率的背景估计
    blocks = gray[:gh, :gw].reshape(bh, block, bw, block).mean(axis=(1, 3))
    # 2) 用双线性插值把背景上采样回原尺寸
    bg = np.asarray(_PILImage.fromarray(blocks.astype(np.float64)).resize((w, h), _PILImage.BILINEAR),
                    dtype=np.float64)
    bg = np.where(bg < 1e-6, 1e-6, bg)
    # 3) 用背景做归一化（平场校正）
    out = gray * (gray.mean() / bg)
    return np.clip(out, 0, 255)


def gray_dilate(gray, ksize):
    """
    灰度膨胀：窗口内取最大值（把亮区扩展、填充暗的细微结构）
    """
    r = ksize // 2
    padded = np.pad(gray.astype(np.float64), r, mode='edge')
    windows = sliding_window_view(padded, (ksize, ksize))
    return windows.max(axis=(-2, -1))


def gray_erode(gray, ksize):
    """
    灰度腐蚀：窗口内取最小值（把暗区扩展）
    """
    r = ksize // 2
    padded = np.pad(gray.astype(np.float64), r, mode='edge')
    windows = sliding_window_view(padded, (ksize, ksize))
    return windows.min(axis=(-2, -1))


def black_hat(gray, ksize=25):
    """
    黑帽变换（灰度值调整第 2 步：背景抑制）
    原理：先对灰度图做闭运算（先膨胀后腐蚀，用较大窗口），得到"填充了所有暗细节"的背景估计，
          再用 闭运算结果 - 原图，即可提取出"比周围背景暗的小区域"。
    作用：横向帘线条纹和残余灰度不均属于大面积/线状结构，会被闭运算吸收进背景；
          而孤立的暗斑缺陷尺寸小于结构元，会被保留下来，从而被凸显。

    参数：ksize 为结构元（窗口）大小，应大于缺陷尺寸、且能覆盖条纹间隔
    """
    closed = gray_erode(gray_dilate(gray, ksize), ksize)   # 闭运算 = 先膨胀后腐蚀
    return np.clip(closed - gray, 0, None)                 # 暗细节响应图（缺陷处较大）


# ============================================================================
# 2. 最大类间方差阈值分割（Otsu 方法）
# ============================================================================

def otsu_threshold(gray):
    """
    最大类间方差法（Otsu）自动求取分割阈值

    原理：把图像分为前景（目标）和背景两类，遍历所有可能的阈值 T，
          使类间方差 sigma_b² = w0·w1·(m0-m1)² 最大的 T 即为最佳阈值。
          （w0/w1 为两类像素比例，m0/m1 为两类平均灰度）
    返回：最佳阈值 T
    """
    hist = np.bincount(np.clip(np.round(gray), 0, 255).astype(np.uint8).ravel(),
                       minlength=256).astype(np.float64)
    total = hist.sum()
    prob = hist / total
    gray_levels = np.arange(256)
    cum_prob = np.cumsum(prob)
    cum_mean = np.cumsum(prob * gray_levels)
    global_mean = cum_mean[-1]

    best_T = 0
    best_var = -1.0
    for t in range(256):
        w0 = cum_prob[t]
        w1 = 1.0 - w0
        if w0 < 1e-9 or w1 < 1e-9:
            continue
        m0 = cum_mean[t] / w0
        m1 = (global_mean - cum_mean[t]) / w1
        var_between = w0 * w1 * (m0 - m1) ** 2
        if var_between > best_var:
            best_var = var_between
            best_T = t
    return best_T


def binarize(gray, threshold, dark_is_defect=True):
    """利用阈值对图像二值化（默认灰度 < 阈值判为缺陷）"""
    if dark_is_defect:
        binary = (gray < threshold)
    else:
        binary = (gray > threshold)
    return binary.astype(np.uint8)


# ============================================================================
# 3. 形态学处理（腐蚀 / 膨胀 / 开运算）
# ============================================================================

def erode(binary, ksize=3, h=None, w=None):
    """
    腐蚀运算：对二值图中任意一点 O，取其邻域窗口内的最小值作为该点新值。
    效果：消除细小噪点、缩小目标。
    ksize 为正方形窗口边长；h/w 可分别指定窗口的高和宽（用于各向异性结构元）
    """
    hh = h if h is not None else ksize
    ww = w if w is not None else ksize
    ph, pw = hh // 2, ww // 2
    padded = np.pad(binary.astype(np.uint8), ((ph, ph), (pw, pw)), mode='edge')
    windows = sliding_window_view(padded, (hh, ww))
    return windows.min(axis=(-2, -1)).astype(np.uint8)


def dilate(binary, ksize=3, h=None, w=None):
    """
    膨胀运算：对二值图中任意一点 O，取其邻域窗口内的最大值作为该点新值。
    效果：填补目标空洞、恢复因腐蚀缩小的目标尺寸。
    """
    hh = h if h is not None else ksize
    ww = w if w is not None else ksize
    ph, pw = hh // 2, ww // 2
    padded = np.pad(binary.astype(np.uint8), ((ph, ph), (pw, pw)), mode='edge')
    windows = sliding_window_view(padded, (hh, ww))
    return windows.max(axis=(-2, -1)).astype(np.uint8)


def opening(binary, ksize=3):
    """开运算 = 先腐蚀后膨胀：去除细小噪点、断开窄连接"""
    return dilate(erode(binary, ksize), ksize)


def remove_small_objects(binary, response=None, min_mean_response=0.0,
                         min_area=50, max_area=None, min_h=0, min_w=0, max_ratio=4.0):
    """
    连通域过滤：
      - 面积小于 min_area 的视为噪点，去除；
      - 面积大于 max_area 的视为大面积背景残留，去除（max_area=None 表示不过滤）；
      - 高度小于 min_h 或宽度小于 min_w 的"线状/扁平"区域去除；
      - 长宽比（较长边/较短边）大于 max_ratio 的细长条带去除（缺陷应为紧凑块状）；
      - 区域内"暗细节响应值"的平均值小于 min_mean_response 的去除
        （用于剔除亮度较弱的帘线暗纹等背景结构，只保留"足够黑"的真实缺陷）。
    返回：过滤后的二值图、连通域标记图、保留下来的连通域信息列表
    """
    from scipy.ndimage import label
    labeled, num = label(binary)
    if num == 0:
        return binary, labeled, []

    cleaned = np.zeros_like(binary)
    regions = []
    for idx in range(1, num + 1):
        mask = (labeled == idx)
        area = int(mask.sum())
        if area < min_area:
            continue
        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        if max_area is not None and area > max_area:
            continue
        if bh < min_h or bw < min_w:      # 剔除过于扁平的线状结构
            continue
        ratio = max(bw, bh) / max(1, min(bw, bh))   # 长宽比
        if ratio > max_ratio:             # 剔除细长的条带形干扰
            continue
        # 响应强度过滤：剔除暗细节响应较弱的背景结构（如帘线暗纹）
        if response is not None and min_mean_response > 0:
            if float(response[mask].mean()) < min_mean_response:
                continue
        cleaned[mask] = 1
        regions.append({
            'area': area,
            'bbox': (x0, y0, x1, y1),
            'center': (int(xs.mean()), int(ys.mean())),
            'mean_response': float(response[mask].mean()) if response is not None else 0.0,
        })
    return cleaned, labeled, regions


# ============================================================================
# 4. 缺陷标记 与 完整检测流程
# ============================================================================

def draw_defects(rgb_img, regions, color=(255, 0, 0), thickness=2):
    """在原始彩色图像上用红色矩形框标记缺陷位置"""
    out = rgb_img.copy()
    h, w = out.shape[:2]
    for reg in regions:
        x0, y0, x1, y1 = reg['bbox']
        x0, y0 = max(0, x0 - 5), max(0, y0 - 5)
        x1, y1 = min(w - 1, x1 + 5), min(h - 1, y1 + 5)
        out[y0:y0 + thickness, x0:x1 + 1] = color
        out[y1 - thickness + 1:y1 + 1, x0:x1 + 1] = color
        out[y0:y1 + 1, x0:x0 + thickness] = color
        out[y0:y1 + 1, x1 - thickness + 1:x1 + 1] = color
    return out


def detect_debris(img, min_area=150, max_area=8000, morph_size=5, bg_size=25,
                  min_h=10, min_w=10, max_ratio=3.0, edge_margin=20,
                  thresh_scale=1.0, flatten_block=64, min_mean_response=95.0,
                  dark_is_defect=True, verbose=False):
    """
    胎侧杂物检测完整流程
      1) 灰度化
      2) 灰度值调整：逐列灰度归一化（消除厚度引起的竖向灰度不均）
      3) 灰度值调整（背景抑制）：黑帽变换，凸显比背景暗的孤立小斑（抑制横向帘线条纹）
      4) Otsu 最大类间方差阈值分割
      5) 形态学处理：开运算（腐蚀去噪 + 膨胀恢复）
      6) 连通域过滤：去除面积过小/过大、以及线状（扁平）的干扰区域
      7) 在原图上用红框标记缺陷

    参数：
      img : 输入图像（RGB 或灰度，numpy 数组）
      min_area / max_area : 缺陷面积的合理范围（像素）
      morph_size : 形态学开运算窗口大小
      bg_size    : 黑帽变换结构元大小（需大于缺陷尺寸）
      min_h / min_w : 缺陷区域最小高度/宽度（用于剔除线状干扰）
    返回：(二值缺陷图, 标记后的彩色图, 缺陷信息列表)
    """
    # 原始 RGB 图（用于标记）
    if img.ndim == 3:
        rgb_img = img.astype(np.uint8).copy()
    else:
        rgb_img = np.stack([img.astype(np.uint8)] * 3, axis=2)

    # 1) 灰度化
    gray = to_gray(img)

    # 2) 灰度值调整：逐列灰度归一化（消除列方向厚度不均）
    normalized = column_gray_normalize(gray)
    # 2.1) 灰度值调整（增强）：分块二维背景归一化，进一步消除大尺度灰度不均（如竖向暗带）
    if flatten_block and flatten_block > 0:
        normalized = block_gray_normalize(normalized, block=flatten_block)

    # 3) 黑帽变换背景抑制（凸显比背景暗的小斑缺陷）
    response = black_hat(normalized, bg_size).astype(np.float64)

    # 3.1) 边缘抑制：图像最外圈受黑帽边界效应影响会产生伪响应（如左侧暗带边缘），
    #      将距边缘 edge_margin 像素内的响应置 0，避免边缘误检
    if edge_margin > 0:
        m = int(edge_margin)
        response[:m, :] = 0
        response[-m:, :] = 0
        response[:, :m] = 0
        response[:, -m:] = 0

    # 4) Otsu 阈值分割（对"暗细节响应图"求阈值）
    #    thresh_scale 用于把阈值适当抬高，抑制响应较弱的背景残留（默认 1.0 即纯 Otsu）
    T = otsu_threshold(response) * thresh_scale
    binary = (response > T).astype(np.uint8)
    if verbose:
        print(f"    黑帽响应 Otsu 阈值 T = {T:.1f}")

    # 5) 形态学处理：开运算去噪
    morphed = opening(binary, morph_size)

    # 6) 连通域过滤：面积 + 形状 + 响应强度（剔除细小噪点、过大背景残留、线状条带和弱响应背景结构）
    cleaned, labeled, regions = remove_small_objects(
        morphed, response=response, min_mean_response=min_mean_response,
        min_area=min_area, max_area=max_area,
        min_h=min_h, min_w=min_w, max_ratio=max_ratio)
    if verbose:
        print(f"    检测到 {len(regions)} 个候选缺陷区域")

    # 7) 标记缺陷
    marked = draw_defects(rgb_img, regions)

    return cleaned, marked, regions
