# -*- coding: utf-8 -*-
import numpy as np
import math


def fft2d(image):
    """
    二维傅立叶变换（空域→频域）

    参数:
        image: 输入图像(numpy数组, HxW, 灰度图)

    返回:
        F: 频域复数矩阵
    """
    H, W = image.shape
    F = np.zeros((H, W), dtype=np.complex128)

    for u in range(H):
        for v in range(W):
            val = 0.0
            for x in range(H):
                for y in range(W):
                    angle = -2 * math.pi * (u * x / H + v * y / W)
                    val += image[x, y] * (math.cos(angle) + 1j * math.sin(angle))
            F[u, v] = val

    return F


def ifft2d(F):
    """
    二维傅立叶逆变换（频域→空域）

    参数:
        F: 频域复数矩阵

    返回:
        image: 空域图像(numpy数组, HxW)
    """
    H, W = F.shape
    image = np.zeros((H, W), dtype=np.float64)

    for x in range(H):
        for y in range(W):
            val = 0.0
            for u in range(H):
                for v in range(W):
                    angle = 2 * math.pi * (u * x / H + v * y / W)
                    val += F[u, v].real * math.cos(angle) - F[u, v].imag * math.sin(angle)
            image[x, y] = val / (H * W)

    return image


def fft_shift(F):
    """
    频谱移中（将低频移到中心）

    参数:
        F: 频域复数矩阵

    返回:
        F_shifted: 移中后的频域矩阵
    """
    H, W = F.shape
    F_shifted = np.zeros_like(F)

    for u in range(H):
        for v in range(W):
            u_new = (u + H // 2) % H
            v_new = (v + W // 2) % W
            F_shifted[u_new, v_new] = F[u, v]

    return F_shifted


def ifft_shift(F_shifted):
    """
    频谱移回（将中心低频移回原点）

    参数:
        F_shifted: 移中后的频域矩阵

    返回:
        F: 原始频域矩阵
    """
    H, W = F_shifted.shape
    F = np.zeros_like(F_shifted)

    for u in range(H):
        for v in range(W):
            u_new = (u + H // 2) % H
            v_new = (v + W // 2) % W
            F[u_new, v_new] = F_shifted[u, v]

    return F


def create_sine_wave(H, W, freq_x, freq_y):
    """
    生成正弦波形图像

    参数:
        H, W: 图像高宽
        freq_x, freq_y: X/Y方向频率

    返回:
        image: 正弦波图像
    """
    image = np.zeros((H, W), dtype=np.float64)
    for x in range(H):
        for y in range(W):
            image[x, y] = 128 + 127 * math.sin(2 * math.pi * (freq_x * x / H + freq_y * y / W))
    return image


def combine_waves(wave_list):
    """
    组合多个波形

    参数:
        wave_list: 波形列表[(wave, weight), ...]

    返回:
        combined: 组合后的图像
    """
    H, W = wave_list[0][0].shape
    combined = np.zeros((H, W), dtype=np.float64)
    for wave, weight in wave_list:
        combined += wave * weight
    combined = np.clip(combined, 0, 255)
    return combined