#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def Scale(input_img, sx, sy):
    """
    图像缩放 —— 双线性插值法（把图像放大或缩小，用周围4个像素加权平均填补新像素。）

    参数:
        input_img: 输入图像 (numpy数组, H×W×C, uint8)
        sx: X方向(行方向)缩放倍数, 对应MATLAB中的x
        sy: Y方向(列方向)缩放倍数, 对应MATLAB中的y

    返回:
        output_img: 缩放后的图像 (numpy数组, uint8)
    """

    width, height, dimension = input_img.shape

    new_width = round(width * sx)
    new_height = round(height * sy)

    output_img = np.zeros((new_width, new_height, dimension), dtype=np.uint8)

    for i in range(new_width):
        for j in range(new_height):

            tempx = int(i / sx)
            tempy = int(j / sy)

            if tempx == 0 or tempy == 0 or tempx == width - 1 or tempy == height - 1:
                output_img[0, j, :] = input_img[0, tempy, :]
                output_img[i, 0, :] = input_img[tempx, 0, :]
            else:
                a = i / sx - tempx
                b = j / sy - tempy

                output_img[i, j, :] = (
                    input_img[tempx, tempy, :] * (1 - a) * (1 - b)
                    + input_img[tempx, tempy + 1, :] * (1 - a) * b
                    + input_img[tempx + 1, tempy, :] * a * (1 - b)
                    + input_img[tempx + 1, tempy + 1, :] * a * b
                )

    return output_img