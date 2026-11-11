#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
"""
实验5：图像修复（Image Restoration）

对应实验要求：
  5-1 噪声图像生成及其直方图
      1) 噪声图像的生成（Rayleigh 噪声、高斯噪声、椒盐噪声）
      2) 生成噪声图像的直方图

  5-2 含噪声图像处理
      1) 实现自适应中值滤波算法（adaptive median filter），并与 7×7 中值滤波器比较
      2) 用维纳滤波器对"运动模糊（a=b=0.1, T=1）+ 均值为0、方差为650的加性高斯噪声"污染的图像进行处理

说明：所有核心算法均手工实现（噪声采样、中值滤波、自适应中值滤波、运动模糊退化函数、维纳滤波），
      不直接调用图像处理库的滤波/复原函数，以符合实验"手写算法"的要求。
"""

import numpy as np
import math
from numpy.lib.stride_tricks import sliding_window_view


# ============================================================================
# 5-1 噪声图像生成
# ============================================================================

def add_gaussian_noise(img, mean=0.0, sigma=25.0):
    """
    添加高斯噪声
    高斯分布概率密度：p(z) = 1/(sqrt(2π)·σ) · exp(-(z-mean)²/(2σ²))

    实现：用 Box-Muller 变换，把两组 [0,1) 均匀随机数转换为标准正态分布随机数：
          z = sqrt(-2·ln(U1)) · cos(2π·U2)   ，再线性变换 z -> mean + sigma·z
    参数：
        mean : 噪声均值
        sigma: 噪声标准差
    """
    h, w = img.shape
    # 生成两组独立均匀分布随机数 U1, U2 ∈ [0,1)
    u1 = np.random.rand(h, w)
    u2 = np.random.rand(h, w)
    # Box-Muller 变换 -> 标准正态 N(0,1)
    # 加 1e-12 防止 log(0) 出现 -inf
    z = np.sqrt(-2.0 * np.log(u1 + 1e-12)) * np.cos(2.0 * math.pi * u2)
    noise = mean + sigma * z
    # 叠加噪声并截断到合法灰度范围 [0, 255]
    return np.clip(img.astype(np.float64) + noise, 0, 255)


def add_rayleigh_noise(img, sigma=30.0):
    """
    添加瑞利（Rayleigh）噪声
    瑞利分布概率密度：p(z) = (z-a)/b · exp(-(z-a)²/(2b)) , z >= a

    实现：采用逆变换采样。令 U ~ Uniform(0,1)，则
          z = a + sqrt(-2b·ln(1-U)) 服从瑞利分布（取 a = 0）
          这里取尺度参数 b = sigma²，使噪声强度随 sigma 增大而增大。
    """
    h, w = img.shape
    u = np.random.rand(h, w)
    b = float(sigma) ** 2                      # 尺度参数
    noise = np.sqrt(-2.0 * b * np.log(1.0 - u + 1e-12))   # 瑞利分布随机数（a=0）
    return np.clip(img.astype(np.float64) + noise, 0, 255)


def add_salt_pepper_noise(img, pa=0.05, pb=0.05):
    """
    添加椒盐噪声
    pa: "盐"噪声（亮点，灰度 255）出现的概率
    pb: "椒"噪声（暗点，灰度 0）出现的概率

    实现：生成与图像同尺寸的 [0,1) 随机矩阵 r
           r < pa          -> 置为 255（盐）
           pa <= r < pa+pb -> 置为 0  （椒）
    """
    out = img.astype(np.float64).copy()
    h, w = img.shape
    r = np.random.rand(h, w)
    out[r < pa] = 255.0                              # 盐噪声
    out[(r >= pa) & (r < pa + pb)] = 0.0             # 椒噪声
    return out


def compute_histogram(img):
    """
    计算灰度直方图（256 个灰度级）
    返回：长度为 256 的数组，hist[g] 表示灰度值为 g 的像素个数
    （用 np.bincount 统计，等价于逐像素计数，速度更快）
    """
    gray = np.clip(np.round(img), 0, 255).astype(np.uint8).ravel()
    hist = np.bincount(gray, minlength=256)
    return hist


# ============================================================================
# 5-2-1 中值滤波 与 自适应中值滤波
# ============================================================================

def median_filter(img, ksize=7):
    """
    普通中值滤波：用 ksize×ksize 窗口内的灰度中值替代窗口中心像素。
    参数：ksize 为窗口大小（应为奇数，如 7 表示 7×7）

    实现：边界用 'edge' 方式填充（相当于把边界像素复制出去，等价于 clamp），
          再用滑动窗口视图一次性取出所有窗口，对每个窗口取中值（向量化加速）。
    """
    r = ksize // 2
    padded = np.pad(img.astype(np.float64), r, mode='edge')      # 边界复制填充
    windows = sliding_window_view(padded, (ksize, ksize))        # 形状 (h, w, ksize, ksize)
    return np.median(windows, axis=(-2, -1))


def adaptive_median_filter(img, smax=7):
    """
    自适应中值滤波（Adaptive Median Filter）
    思想：当窗口内中值本身是脉冲（椒盐）时，说明窗口内噪声较多，需要扩大窗口；
          当窗口内中值不是脉冲时，再判断中心像素本身是否为脉冲：
              是脉冲 -> 输出窗口内中值（把它替换掉）
              不是   -> 保持原像素值（保护细节）
    两个层次：
      Level A：z_med - z_min > 0 且 z_med - z_max < 0 （中值不是脉冲）-> 进入 Level B
               否则扩大窗口（size += 2），直到 size > smax
      Level B：z_xy - z_min > 0 且 z_xy - z_max < 0 （中心像素不是脉冲）-> 输出 z_xy
               否则输出 z_med
    参数：smax 为允许的最大窗口大小
    """
    h, w = img.shape
    src = img.astype(np.float64)
    out = src.copy()

    for i in range(h):
        for j in range(w):
            zxy = src[i, j]
            size = 3                      # 从 3×3 窗口开始
            zmed = zxy                    # 默认值，防止 smax<3 时未定义
            while size <= smax:
                r = size // 2
                # 取窗口（越界处 clamp 到边界，保证窗口始终为 size×size）
                i0, i1 = max(0, i - r), min(h, i + r + 1)
                j0, j1 = max(0, j - r), min(w, j + r + 1)
                win = src[i0:i1, j0:j1]
                zmin = win.min()
                zmax = win.max()
                zmed = np.median(win)

                # ---------- Level A ----------
                if (zmed - zmin) > 0 and (zmed - zmax) < 0:
                    # 中值不是脉冲，进入 Level B
                    # ---------- Level B ----------
                    if (zxy - zmin) > 0 and (zxy - zmax) < 0:
                        out[i, j] = zxy        # 中心像素不是脉冲 -> 保持不变
                    else:
                        out[i, j] = zmed       # 中心像素是脉冲 -> 用中值替换
                    break
                else:
                    size += 2                  # 中值本身是脉冲 -> 扩大窗口
            else:
                # 窗口已超过 smax，直接输出中值
                out[i, j] = zmed
    return out


# ============================================================================
# 5-2-2 运动模糊退化 与 维纳滤波
# ============================================================================

def motion_blur_psf(h, w, a=0.1, b=0.1, T=1.0):
    """
    生成运动模糊的退化函数 H(u,v)（频域，频率坐标已中心化到图像中心）

    退化模型（匀速直线运动模糊）：
        H(u,v) = T/(π(ua+vb)) · sin(π(ua+vb)) · exp(-j·π(ua+vb))
        当 (ua+vb) = 0 时，取极限 H(u,v) = T

    参数：a, b 为运动分量（实验要求 a=b=0.1），T 为曝光时间（实验要求 T=1）
    """
    # 中心化频率坐标：u ∈ [-h/2, h/2)，v ∈ [-w/2, w/2)
    u = np.arange(h) - h // 2
    v = np.arange(w) - w // 2
    U, V = np.meshgrid(v, u)          # U 沿列(v)方向，V 沿行(u)方向
    UV = U * a + V * b                # ua + vb

    H = np.zeros((h, w), dtype=np.complex128)
    # 除零保护：|ua+vb| 很小的位置直接取极限值 T
    small = np.abs(UV) < 1e-6
    H[small] = T
    H[~small] = (T / (math.pi * UV[~small])) * \
                np.sin(math.pi * UV[~small]) * \
                np.exp(-1j * math.pi * UV[~small])
    return H


def apply_motion_blur(img, a=0.1, b=0.1, T=1.0):
    """
    对图像施加运动模糊（频域相乘：G = H · F）
    返回：(模糊后的空间域图像, 退化函数 H)
    """
    h, w = img.shape
    H = motion_blur_psf(h, w, a, b, T)
    # 图像做 FFT 并移中（低频到中心），使其与中心化的 H 频率坐标对应
    F = np.fft.fftshift(np.fft.fft2(img.astype(np.float64)))
    G = F * H                                          # 频域相乘 = 空域卷积（退化）
    blurred = np.real(np.fft.ifft2(np.fft.ifftshift(G)))   # 逆变换回空域
    return np.clip(blurred, 0, 255), H


def wiener_filter(degraded, H, K=0.02):
    """
    维纳（Wiener）滤波复原
    公式：F_hat(u,v) = [ H*(u,v) / (|H(u,v)|² + K) ] · G(u,v)
        其中 H* 为 H 的共轭，K 近似表示噪信比（|N|²/|F|²）的常数估计，
        K 越小复原越锐利但更容易放大噪声，K 越大越平滑。

    参数：
        degraded : 退化（模糊+噪声）图像
        H        : 运动模糊退化函数
        K        : 噪信比参数（经验常数）
    """
    G = np.fft.fftshift(np.fft.fft2(degraded.astype(np.float64)))
    F_hat = (np.conj(H) / (np.abs(H) ** 2 + K)) * G     # 维纳滤波器核心公式
    restored = np.real(np.fft.ifft2(np.fft.ifftshift(F_hat)))
    return np.clip(restored, 0, 255)
