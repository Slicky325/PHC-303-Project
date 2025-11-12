import numpy as np


def fft_iterative(signal: np.ndarray) -> np.ndarray:
    signal_length = len(signal)

    if signal_length & (signal_length - 1) != 0:
        next_power_of_two = 1 << (signal_length - 1).bit_length()
        signal = np.pad(signal, (0, next_power_of_two - signal_length), mode='constant')
        signal_length = next_power_of_two

    bit_count = int(np.log2(signal_length))
    original_indices = np.arange(signal_length, dtype=np.int64)
    reversed_indices = np.zeros(signal_length, dtype=np.int64)
    
    for _ in range(bit_count):
        reversed_indices = (reversed_indices << 1) | (original_indices & 1)
        original_indices >>= 1

    frequency_domain = np.asarray(signal, dtype=complex)[reversed_indices]

    butterfly_size = 2
    while butterfly_size <= signal_length:
        half_size = butterfly_size // 2
        frequency_indices = np.arange(half_size)
        twiddle_factors = np.exp(-2j * np.pi * frequency_indices / butterfly_size)

        for block_start in range(0, signal_length, butterfly_size):
            even_samples = frequency_domain[block_start:block_start + half_size].copy()
            odd_samples = frequency_domain[block_start + half_size:block_start + butterfly_size].copy()
            rotated_odd = twiddle_factors * odd_samples
            
            frequency_domain[block_start:block_start + half_size] = even_samples + rotated_odd
            frequency_domain[block_start + half_size:block_start + butterfly_size] = even_samples - rotated_odd

        butterfly_size <<= 1

    return frequency_domain


def ifft(frequency_domain: np.ndarray) -> np.ndarray:
    signal_length = len(frequency_domain)
    return np.conj(fft_iterative(np.conj(frequency_domain))) / signal_length


def fft_2d(image: np.ndarray) -> np.ndarray:
    row_count, col_count = image.shape
    frequency_domain = np.zeros_like(image, dtype=complex)
    
    for row_index in range(row_count):
        frequency_domain[row_index, :] = fft_iterative(image[row_index, :])
    for col_index in range(col_count):
        frequency_domain[:, col_index] = fft_iterative(frequency_domain[:, col_index])
    
    return frequency_domain


def ifft_2d(frequency_domain: np.ndarray) -> np.ndarray:
    row_count, col_count = frequency_domain.shape
    image = np.zeros_like(frequency_domain, dtype=complex)
    
    for row_index in range(row_count):
        image[row_index, :] = ifft(frequency_domain[row_index, :])
    for col_index in range(col_count):
        image[:, col_index] = ifft(image[:, col_index])
    
    return image


if __name__ == "__main__":
    try:
        import numpy as _np
        test_signal = _np.random.randn(128) + 1j * _np.random.randn(128)
        our_result = fft_iterative(test_signal)
        numpy_result = _np.fft.fft(test_signal)
        max_error = _np.max(_np.abs(our_result - numpy_result))
        print(f"Self-test FFT max error vs numpy: {max_error:.2e}")
    except Exception as e:
        print("Self-test skipped (numpy not available or other error):", e)
