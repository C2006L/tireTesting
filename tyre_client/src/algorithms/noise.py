# -*- coding: utf-8 -*-
# pylint: disable=invalid-name
import numpy as np


def add_gaussian_noise(image, mean=0.0, sigma=25.0, seed=None):
    """加性高斯噪声: z = f + n, n ~ N(mean, sigma^2)"""
    img = np.array(image, dtype=np.float64)
    rng = np.random.default_rng(seed)
    noise = rng.normal(mean, sigma, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def add_rayleigh_noise(image, a=0.0, b=30.0, seed=None):
    """瑞利噪声: z = f + a + b*sqrt(-2*ln(1-U)), U ~ U(0,1)"""
    img = np.array(image, dtype=np.float64)
    rng = np.random.default_rng(seed)
    u = rng.random(img.shape)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    noise = a + b * np.sqrt(-2.0 * np.log(1.0 - u))
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, p=0.05, seed=None):
    """椒盐噪声: 概率p/2置0(椒), 概率p/2置255(盐)"""
    img = np.array(image, dtype=np.float64)
    rng = np.random.default_rng(seed)
    r = rng.random(img.shape)
    img[r < p / 2.0] = 0
    img[r > 1.0 - p / 2.0] = 255
    return img.astype(np.uint8)


def compute_histogram(image, bins=256):
    """灰度直方图: 返回每个灰度级的像素个数"""
    img = np.array(image)
    hist, _ = np.histogram(img.ravel(), bins=bins, range=(0, bins))
    return hist