# -*- coding: utf-8 -*-
import numpy as np
import math


def freq_filter_sobel(image, kernel_size=3):
    """
    实验4-2-1: 使用3x3的Sobel核，在频域中滤波

    原理: 空域卷积 = 频域相乘
    1) 对图像做FFT
    2) 对Sobel核做FFT（补零到图像大小）
    3) 频域相乘
    4) IFFT回到空域

    参数:
        image: 输入图像(HxW, 灰度图)
        kernel_size: Sobel核大小(默认3)

    返回:
        result: 滤波后图像
    """
    H, W = image.shape

    # Sobel核(水平方向边缘检测)
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)

    # 将Sobel核补零到图像大小
    kernel_padded = np.zeros((H, W), dtype=np.float64)
    kh, kw = sobel_x.shape
    kernel_padded[:kh, :kw] = sobel_x

    # 对图像和核都做FFT
    F_img = np.fft.fft2(image)
    F_kernel = np.fft.fft2(kernel_padded)

    # 频域相乘
    F_result = F_img * F_kernel

    # IFFT回到空域
    result = np.fft.ifft2(F_result).real

    # 归一化到0-255
    result = np.clip(result, 0, 255)
    return result


def freq_filter_laplacian(image, D0=30):
    """
    实验4-2-2: 使用Laplacian高通滤波器，在频域中锐化

    Laplacian高通滤波器公式:
    H(u,v) = 1 + c * (u^2 + v^2) / D0^2
    或理想高通: H(u,v) = 1  if D(u,v) > D0, else 0

    这里使用高斯高通滤波器:
    H(u,v) = 1 - exp(-D(u,v)^2 / (2*D0^2))

    参数:
        image: 输入图像(HxW, 灰度图)
        D0: 截止频率(默认30)

    返回:
        result: 锐化后图像
    """
    H, W = image.shape

    # FFT
    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)

    # 构建Laplacian高通滤波器
    u = np.arange(H)
    v = np.arange(W)
    U, V = np.meshgrid(v, u)

    # 中心坐标
    center_u, center_v = H // 2, W // 2

    # 计算每个频率点到中心的距离
    D = np.sqrt((U - center_v) ** 2 + (V - center_u) ** 2)

    # 高斯高通滤波器
    H_filter = 1.0 - np.exp(-(D ** 2) / (2 * D0 ** 2))

    # 频域滤波
    F_filtered = F_shifted * H_filter

    # 移回 + IFFT
    F_unshifted = np.fft.ifftshift(F_filtered)
    result = np.fft.ifft2(F_unshifted).real

    # 归一化
    result = np.clip(result, 0, 255)
    return result