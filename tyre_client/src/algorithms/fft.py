# -*- coding: utf-8 -*-
import numpy as np
import math


def fft2d(image):
    """二维FFT: F(u,v) = ΣΣ f(x,y)·e^(-j2π(ux/H+vy/W))"""
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
    """二维IFFT: f(x,y) = (1/HW)·ΣΣ F(u,v)·e^(j2π(ux/H+vy/W))"""
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
    """频谱移中: 低频从左上角移到中心"""
    H, W = F.shape
    F_shifted = np.zeros_like(F)

    for u in range(H):
        for v in range(W):
            u_new = (u + H // 2) % H
            v_new = (v + W // 2) % W
            F_shifted[u_new, v_new] = F[u, v]

    return F_shifted


def ifft_shift(F_shifted):
    """频谱移回: 中心低频移回左上角"""
    H, W = F_shifted.shape
    F = np.zeros_like(F_shifted)

    for u in range(H):
        for v in range(W):
            u_new = (u + H // 2) % H
            v_new = (v + W // 2) % W
            F[u_new, v_new] = F_shifted[u, v]

    return F


def create_sine_wave(H, W, freq_x, freq_y):
    """生成正弦波: 128+127·sin(2π(freq_x·x/H+freq_y·y/W))"""
    image = np.zeros((H, W), dtype=np.float64)
    for x in range(H):
        for y in range(W):
            image[x, y] = 128 + 127 * math.sin(2 * math.pi * (freq_x * x / H + freq_y * y / W))
    return image


def combine_waves(wave_list):
    """叠加多个波形: Σ(weight_i × wave_i)"""
    H, W = wave_list[0][0].shape
    combined = np.zeros((H, W), dtype=np.float64)
    for wave, weight in wave_list:
        combined += weight * wave
    combined = np.clip(combined, 0, 255)
    return combined