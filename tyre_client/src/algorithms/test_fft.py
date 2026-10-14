# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from fft import fft2d, ifft2d, fft_shift, create_sine_wave, combine_waves

def main():
    print("=" * 60)
    print("Experiment 4-1: Fourier Transform")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))
    lab2_img = r"D:\2026第二学期\2026数媒数字图像处理实验说明\Pset4_Fourier and Frequency domain Filtering\实验4提供的图像\building-600by600.tif"
    default_img = os.path.join(project_root, 'static', 'imgs', 'test0', '1.bmp')

    img_path = sys.argv[1] if len(sys.argv) > 1 else lab2_img
    if not os.path.exists(img_path):
        img_path = default_img

    img = Image.open(img_path).convert('L')
    input_img = np.array(img, dtype=np.float64)
    H, W = input_img.shape

    print(f"Input: {img_path}")
    print(f"Size: {H}x{W}")

    # 使用numpy内置FFT（手写FFT2d是O(N^4)，大图跑不完）
    print("Computing FFT...")

    # 1) 空域到频域变换（原点在左上角）
    F_original = np.fft.fft2(input_img)
    F_original_mag = np.log(1 + np.abs(F_original))

    # 2) 频谱移中（低频移到中心）
    F_shifted = np.fft.fftshift(F_original)
    F_shifted_mag = np.log(1 + np.abs(F_shifted))

    # 3) 频域到空域变换（IFFT）
    F_unshifted = np.fft.ifftshift(F_shifted)
    reconstructed = np.fft.ifft2(F_unshifted).real

    # 4) 手写FFT验证（仅用小图验证原理正确性）
    print("Verifying hand-written FFT on 32x32 crop...")
    small_crop = input_img[:32, :32]
    F_hand = fft2d(small_crop)
    F_np = np.fft.fft2(small_crop)
    diff = np.max(np.abs(F_hand - F_np))
    print(f"  Hand-written FFT vs numpy FFT max diff: {diff:.2e} (should be ~0)")

    # 5) 正弦波合成演示
    wave1 = create_sine_wave(256, 256, 10, 0)
    wave2 = create_sine_wave(256, 256, 0, 10)
    wave3 = create_sine_wave(256, 256, 10, 10)
    combined = combine_waves([(wave1, 0.5), (wave2, 0.5), (wave3, 0.3)])

    output_dir = os.path.join(project_root, 'static', 'imgs', 'output')
    os.makedirs(output_dir, exist_ok=True)

    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Experiment 4-1: Fourier Transform', fontsize=14)

        axes[0, 0].imshow(input_img, cmap='gray')
        axes[0, 0].set_title('Original Image\n(Spatial Domain)')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(F_original_mag, cmap='gray')
        axes[0, 1].set_title('FFT Magnitude\n(Origin at Top-Left)')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(F_shifted_mag, cmap='gray')
        axes[0, 2].set_title('FFT Magnitude Shifted\n(Low-Freq at Center)')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(reconstructed, cmap='gray')
        axes[1, 0].set_title('Reconstructed (IFFT)')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(wave1, cmap='gray')
        axes[1, 1].set_title('Sine Wave (fx=10, fy=0)\nHorizontal Stripes')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(combined, cmap='gray')
        axes[1, 2].set_title('Combined Waves\n(Multiple Frequencies)')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Cannot display: {e}")

if __name__ == '__main__':
    main()