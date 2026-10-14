#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import math


def Warp(input_img):
    """
    图像变形 —— 旋转变形

    参数:
        input_img: 输入图像 (numpy数组, H×W×C, uint8)

    返回:
        output_img: 变形后的图像 (numpy数组, uint8)
    """

    width, height, dimension = input_img.shape
    output_img = np.zeros((width, height, dimension), dtype=np.uint8)

    for i in range(width):
        for j in range(height):
            #中心归一化
            tempx = (i + 1 - 0.5 * width) / (0.5 * width)
            tempy = (j + 1 - 0.5 * height) / (0.5 * height)
            #计算旋转角度
            r = math.sqrt(tempx ** 2 + tempy ** 2)
            K = (1 - r ) ** 4
            #逆映射变换
            if r >= 1:
                x = tempx
                y = tempy
            else:
                x = math.cos(K) * tempx - math.sin(K) * tempy
                y = math.sin(K) * tempx + math.cos(K) * tempy
            #反归一化
            old_x = int((x + 1) * 0.5 * width)
            old_y = int((y + 1) * 0.5 * height)

            old_x = max(1, min(old_x, width))
            old_y = max(1, min(old_y, height))

            output_img[i, j, :] = input_img[old_x - 1, old_y - 1, :]

    return output_img