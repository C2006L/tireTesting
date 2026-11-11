# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def median_filter(image, size=7):
    """标准中值滤波: size×size窗口取中值"""
    img = np.array(image, dtype=np.float64)
    r = size // 2
    padded = np.pad(img, r, mode='edge')
    windows = sliding_window_view(padded, (size, size))
    return np.median(windows, axis=(-1, -2))


def adaptive_median_filter(image, smax=7):
    """自适应中值滤波: 窗口3×3起, 最大smax, Gonzalez三级判断"""
    img = np.array(image, dtype=np.float64)
    H, W = img.shape
    rmax = smax // 2
    padded = np.pad(img, rmax, mode='edge')

    decided = np.zeros((H, W), dtype=bool)
    out = img.copy()
    zmed = np.zeros((H, W))

    for size in range(3, smax + 1, 2):
        r = size // 2
        off = rmax - r
        windows = sliding_window_view(padded, (size, size))[off:off + H, off:off + W]
        zmin = windows.min(axis=(-1, -2))
        zmax = windows.max(axis=(-1, -2))
        zmed = np.median(windows, axis=(-1, -2))

        stage_a = (zmin < zmed) & (zmed < zmax)
        stage_b = (zmin < img) & (img < zmax)

        # A成立且中心像素不是噪声: 保留原值
        keep = ~decided & stage_a & stage_b
        # A成立且中心像素是噪声: 用中值替换
        replace = ~decided & stage_a & ~stage_b

        out[keep] = img[keep]
        out[replace] = zmed[replace]
        decided |= stage_a

    # 窗口已到smax仍无法判断: 输出中值
    out[~decided] = zmed[~decided]
    return out


def motion_blur_kernel(H, W, a=0.1, b=0.1, T=1.0):
    """运动模糊频域响应: H(u,v)=T/(π(ua+vb))·sin(π(ua+vb))·e^(-jπ(ua+vb)), 原点取T"""
    u = np.arange(H) - H // 2
    v = np.arange(W) - W // 2
    U, V = np.meshgrid(v, u)
    phase = np.pi * (U * a + V * b)

    Hf = np.ones_like(phase, dtype=np.complex128) * T
    nz = np.abs(phase) > 1e-10
    Hf[nz] = T / phase[nz] * np.sin(phase[nz]) * np.exp(-1j * phase[nz])
    return Hf


def wiener_filter(image, Hf, K=0.02):
    """维纳滤波: F̂ = H*/(|H|^2+K) · G, K为噪信功率比"""
    img = np.array(image, dtype=np.float64)
    G = np.fft.fftshift(np.fft.fft2(img))
    F = np.conj(Hf) / (np.abs(Hf) ** 2 + K) * G
    f = np.fft.ifft2(np.fft.ifftshift(F)).real
    return np.clip(f, 0, 255)