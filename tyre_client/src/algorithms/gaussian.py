#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math


def gaussian_kernel_1d(sigma, size):
    center = size // 2
    x = np.arange(-center, center + 1, dtype=np.float64)
    kernel = (1.0 / (math.sqrt(2 * math.pi) * sigma)) * np.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def convolve_1d(image, kernel, axis):
    r = len(kernel) // 2
    output = np.zeros_like(image, dtype=np.float64)

    if axis == 1:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                val = 0.0
                for k in range(-r, r + 1):
                    j_src = max(0, min(j + k, image.shape[1] - 1))
                    val += image[i, j_src] * kernel[k + r]
                output[i, j] = val
    elif axis == 0:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                val = 0.0
                for k in range(-r, r + 1):
                    i_src = max(0, min(i + k, image.shape[0] - 1))
                    val += image[i_src, j] * kernel[k + r]
                output[i, j] = val

    return output


def Gaussian(input_img, sigma):
    input_arr = np.array(input_img, dtype=np.float64)

    window_size = int((6 * sigma - 1) // 2 * 2 + 1)
    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1

    kernel = gaussian_kernel_1d(sigma, window_size)

    if len(input_arr.shape) == 2:
        result_h = convolve_1d(input_arr, kernel, axis=1)
        result_v = convolve_1d(result_h, kernel, axis=0)
        output_arr = result_v
    elif len(input_arr.shape) == 3:
        channels = []
        for c in range(input_arr.shape[2]):
            channel = input_arr[:, :, c]
            result_h = convolve_1d(channel, kernel, axis=1)
            result_v = convolve_1d(result_h, kernel, axis=0)
            channels.append(result_v)
        output_arr = np.stack(channels, axis=2)

    output_arr = np.clip(output_arr, 0, 255)
    if input_img.dtype == np.uint8:
        output_arr = output_arr.astype(np.uint8)

    return output_arr