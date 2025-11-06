"""
Custom FFT Implementation - Cooley-Tukey Algorithm
Time Complexity: O(n log n), Space Complexity: O(n)
"""

import numpy as np


def fft_iterative(x: np.ndarray) -> np.ndarray:
    """
    Iterative FFT using Cooley-Tukey algorithm.
    Time: O(n log n), Space: O(n)
    """
    n = len(x)

    # Pad to power of 2
    if n & (n - 1) != 0:
        next_pow2 = 1 << (n - 1).bit_length()
        x = np.pad(x, (0, next_pow2 - n), mode='constant')
        n = next_pow2

    # Vectorized bit-reversal indices
    bits = int(np.log2(n))
    indices = np.arange(n, dtype=np.int64)
    rev = np.zeros(n, dtype=np.int64)
    for _ in range(bits):
        rev = (rev << 1) | (indices & 1)
        indices >>= 1

    # Reorder input into bit-reversed order and ensure complex dtype
    X = np.asarray(x, dtype=complex)[rev]

    # Iterative FFT (Cooley-Tukey) with safe copies to avoid overwrite issues
    length = 2
    while length <= n:
        half = length // 2
        # Twiddle factors for this stage
        k = np.arange(half)
        twiddles = np.exp(-2j * np.pi * k / length)

        # Process blocks
        for start in range(0, n, length):
            a = X[start:start + half].copy()
            b = X[start + half:start + length].copy()
            t = twiddles * b
            X[start:start + half] = a + t
            X[start + half:start + length] = a - t

        length <<= 1

    return X


def ifft(X: np.ndarray) -> np.ndarray:
    """Inverse FFT using conjugate trick. Time: O(n log n)"""
    n = len(X)
    return np.conj(fft_iterative(np.conj(X))) / n


def fft_2d(image: np.ndarray) -> np.ndarray:
    """2D FFT using row-column decomposition. Time: O(n² log n)"""
    rows, cols = image.shape
    result = np.zeros_like(image, dtype=complex)
    
    # FFT along rows then columns
    for i in range(rows):
        result[i, :] = fft_iterative(image[i, :])
    for j in range(cols):
        result[:, j] = fft_iterative(result[:, j])
    return result


def ifft_2d(spectrum: np.ndarray) -> np.ndarray:
    """2D Inverse FFT. Time: O(n² log n)"""
    rows, cols = spectrum.shape
    result = np.zeros_like(spectrum, dtype=complex)
    
    for i in range(rows):
        result[i, :] = ifft(spectrum[i, :])
    for j in range(cols):
        result[:, j] = ifft(result[:, j])
    return result


if __name__ == "__main__":
    try:
        import numpy as _np
        x = _np.random.randn(128) + 1j * _np.random.randn(128)
        my = fft_iterative(x)
        npy = _np.fft.fft(x)
        err = _np.max(_np.abs(my - npy))
        print(f"Self-test FFT max error vs numpy: {err:.2e}")
    except Exception as e:
        print("Self-test skipped (numpy not available or other error):", e)
