#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math


def jbf(D, C, w, sigma_f, sigma_g):
    D_arr = np.array(D, dtype=np.float64)
    C_arr = np.array(C, dtype=np.float64)

    r = w // 2

    if len(D_arr.shape) == 2:
        output = np.zeros_like(D_arr, dtype=np.float64)

        for i in range(D_arr.shape[0]):
            for j in range(D_arr.shape[1]):
                p_val = 0.0
                w_sum = 0.0
                I_p_hat = C_arr[i, j]

                for di in range(-r, r + 1):
                    for dj in range(-r, r + 1):
                        ni = max(0, min(i + di, D_arr.shape[0] - 1))
                        nj = max(0, min(j + dj, D_arr.shape[1] - 1))
                        # 对每个像素p，遍历窗口内每个邻域像素q：

                        # 1) 空间权重f：距离越远权重越小
                        dist_sq = di * di + dj * dj
                        f_weight = math.exp(-dist_sq / (2 * sigma_f ** 2))
                        # 2) 值域权重g：灰度差异越大权重越小（保边缘的关键！）
                        I_q_hat = C_arr[ni, nj]
                        intensity_diff = abs(float(I_p_hat) - float(I_q_hat))
                        g_weight = math.exp(-(intensity_diff ** 2) / (2 * sigma_g ** 2))
                        # 3) 总权重 = 空间权重 × 值域权重
                        total_weight = f_weight * g_weight
                        p_val += total_weight * D_arr[ni, nj]
                        w_sum += total_weight
                # 4) 加权平均
                if w_sum > 1e-10:
                    output[i, j] = p_val / w_sum
                else:
                    output[i, j] = D_arr[i, j]

        output_arr = output

    elif len(D_arr.shape) == 3:
        channels = []
        for c in range(D_arr.shape[2]):
            channel_D = D_arr[:, :, c]
            channel_C = C_arr[:, :, c]

            output = np.zeros_like(channel_D, dtype=np.float64)

            for i in range(channel_D.shape[0]):
                for j in range(channel_D.shape[1]):
                    p_val = 0.0
                    w_sum = 0.0
                    I_p_hat = channel_C[i, j]

                    for di in range(-r, r + 1):
                        for dj in range(-r, r + 1):
                            ni = max(0, min(i + di, channel_D.shape[0] - 1))
                            nj = max(0, min(j + dj, channel_D.shape[1] - 1))

                            dist_sq = di * di + dj * dj
                            f_weight = math.exp(-dist_sq / (2 * sigma_f ** 2))

                            I_q_hat = channel_C[ni, nj]
                            intensity_diff = abs(float(I_p_hat) - float(I_q_hat))
                            g_weight = math.exp(-(intensity_diff ** 2) / (2 * sigma_g ** 2))

                            total_weight = f_weight * g_weight
                            p_val += total_weight * channel_D[ni, nj]
                            w_sum += total_weight

                    if w_sum > 1e-10:
                        output[i, j] = p_val / w_sum
                    else:
                        output[i, j] = channel_D[i, j]

            channels.append(output)

        output_arr = np.stack(channels, axis=2)

    output_arr = np.clip(output_arr, 0, 255)
    if D.dtype == np.uint8:
        output_arr = output_arr.astype(np.uint8)

    return output_arr