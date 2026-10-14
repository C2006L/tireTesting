# -*- coding: utf-8 -*-
import numpy as np
import math


def freq_filter_sobel(image, kernel_size=3):
    #频域Sobel边缘检测: 空域卷积=频域相乘
    H, W = image.shape

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)

    # Sobel核补零到图像大小
    kernel_padded = np.zeros((H, W), dtype=np.float64)
    kh, kw = sobel_x.shape
    kernel_padded[:kh, :kw] = sobel_x

    # FFT → 频域相乘 → IFFT
    F_img = np.fft.fft2(image)
    F_kernel = np.fft.fft2(kernel_padded)
    F_result = F_img * F_kernel
    result = np.fft.ifft2(F_result).real

    # 取绝对值 + 去背景噪声 + 线性拉伸到[0,255]
    result = np.abs(result)
    result[result < result.max() * 0.02] = 0
    result_min = result.min()
    result_max = result.max()
    if result_max - result_min > 1e-10:
        result = (result - result_min) / (result_max - result_min) * 255.0
    result = np.clip(result, 0, 255)
    return result


def freq_filter_laplacian(image, D0=30, strength=1.0):
    """频域Laplacian锐化: 原图 + strength × 高频分量, H=1-exp(-D²/(2D0²))"""
    H, W = image.shape

    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)

    # 高斯高通滤波器
    u = np.arange(H)
    v = np.arange(W)
    U, V = np.meshgrid(v, u)
    center_u, center_v = H // 2, W // 2
    D = np.sqrt((U - center_v) ** 2 + (V - center_u) ** 2)
    H_filter = 1.0 - np.exp(-(D ** 2) / (2 * D0 ** 2))

    # 提取高频分量
    F_high = F_shifted * H_filter
    F_high_unshifted = np.fft.ifftshift(F_high)
    high_freq = np.fft.ifft2(F_high_unshifted).real

    # 锐化 = 原图 + 高频增强
    result = image + strength * high_freq
    result = np.clip(result, 0, 255)
    return result