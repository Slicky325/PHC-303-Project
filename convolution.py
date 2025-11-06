"""
Efficient Convolution using FFT (Convolution Theorem)
conv(f, g) = IFFT(FFT(f) * FFT(g))

Complexity: Direct O(n²) vs FFT O(n log n) for 1D
           Direct O(n⁴) vs FFT O(n² log n) for 2D
"""

import numpy as np
from fft import fft_iterative, ifft, fft_2d, ifft_2d


def convolve_1d_direct(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Direct (spatial domain) convolution for comparison.
    
    Time Complexity: O(n * m) where n = len(signal), m = len(kernel)
    Space Complexity: O(n + m)
    
    This is the naive implementation for benchmarking purposes.
    """
    n = len(signal)
    m = len(kernel)
    result = np.zeros(n + m - 1)
    
    for i in range(n + m - 1):
        for j in range(m):
            if 0 <= i - j < n:
                result[i] += signal[i - j] * kernel[j]
    
    return result


def convolve_1d_fft(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    FFT-based convolution (frequency domain).
    
    Algorithm:
    1. Pad both signals to length (n + m - 1) rounded to next power of 2
    2. Compute FFT of both signals
    3. Multiply element-wise in frequency domain
    4. Compute IFFT to get result
    
    Time Complexity: O(n log n) where n = len(signal) + len(kernel)
    Space Complexity: O(n)
    
    Speedup: ~(n/log n) times faster than direct convolution
    """
    n = len(signal)
    m = len(kernel)
    
    # Output length for linear convolution
    output_len = n + m - 1
    
    # Pad to next power of 2 for optimal FFT performance
    fft_len = 1 << (output_len - 1).bit_length()
    
    # Pad both signals
    signal_padded = np.pad(signal, (0, fft_len - n), mode='constant')
    kernel_padded = np.pad(kernel, (0, fft_len - m), mode='constant')
    
    # FFT of both signals
    signal_fft = fft_iterative(signal_padded)
    kernel_fft = fft_iterative(kernel_padded)
    
    # Multiply in frequency domain (convolution theorem)
    result_fft = signal_fft * kernel_fft
    
    # IFFT to get result in time domain
    result = ifft(result_fft)
    
    # Return only the valid portion (remove padding)
    return np.real(result[:output_len])


def convolve_2d_direct(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Direct 2D convolution for comparison.
    
    Time Complexity: O((n₁ * n₂) * (m₁ * m₂))
    Space Complexity: O((n₁ + m₁) * (n₂ + m₂))
    
    For square n×n image and m×m kernel: O(n² * m²)
    """
    h_img, w_img = image.shape
    h_ker, w_ker = kernel.shape
    
    result = np.zeros((h_img + h_ker - 1, w_img + w_ker - 1))
    
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            for ki in range(h_ker):
                for kj in range(w_ker):
                    img_i = i - ki
                    img_j = j - kj
                    if 0 <= img_i < h_img and 0 <= img_j < w_img:
                        result[i, j] += image[img_i, img_j] * kernel[ki, kj]
    
    return result


def convolve_2d_fft(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    FFT-based 2D convolution (frequency domain).
    
    Algorithm:
    1. Pad both image and kernel to size (h₁+h₂-1) × (w₁+w₂-1)
    2. Compute 2D FFT of both
    3. Multiply element-wise in frequency domain
    4. Compute 2D IFFT to get result
    
    Time Complexity: O(n² log n) for n×n image
    Space Complexity: O(n²)
    
    Speedup: For large images, this is MUCH faster than direct convolution
    Example: For 512×512 image with 32×32 kernel:
        - Direct: ~268 million operations
        - FFT: ~2.4 million operations (100x speedup!)
    """
    h_img, w_img = image.shape
    h_ker, w_ker = kernel.shape
    
    # Output dimensions
    out_h = h_img + h_ker - 1
    out_w = w_img + w_ker - 1
    
    # Pad to next power of 2 for optimal FFT
    fft_h = 1 << (out_h - 1).bit_length()
    fft_w = 1 << (out_w - 1).bit_length()
    
    # Pad both image and kernel
    image_padded = np.pad(image, ((0, fft_h - h_img), (0, fft_w - w_img)), 
                          mode='constant')
    kernel_padded = np.pad(kernel, ((0, fft_h - h_ker), (0, fft_w - w_ker)), 
                           mode='constant')
    
    # 2D FFT of both
    image_fft = fft_2d(image_padded)
    kernel_fft = fft_2d(kernel_padded)
    
    # Element-wise multiplication in frequency domain
    result_fft = image_fft * kernel_fft
    
    # 2D IFFT to get spatial domain result
    result = ifft_2d(result_fft)
    
    # Return only valid portion
    return np.real(result[:out_h, :out_w])


def convolve_2d_fft_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution with 'same' output size (matches input)."""
    full_conv = convolve_2d_fft(image, kernel)
    h_ker, w_ker = kernel.shape
    h_img, w_img = image.shape
    h_start, w_start = h_ker // 2, w_ker // 2
    return full_conv[h_start:h_start + h_img, w_start:w_start + w_img]
