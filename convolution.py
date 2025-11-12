import numpy as np
from fft import fft_iterative, ifft, fft_2d, ifft_2d


def convolve_1d_direct(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    signal_length = len(signal)
    kernel_length = len(kernel)
    output_length = signal_length + kernel_length - 1
    result = np.zeros(output_length)
    
    for output_index in range(output_length):
        for kernel_index in range(kernel_length):
            signal_index = output_index - kernel_index
            if 0 <= signal_index < signal_length:
                result[output_index] += signal[signal_index] * kernel[kernel_index]
    
    return result


def convolve_1d_fft(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    signal_length = len(signal)
    kernel_length = len(kernel)
    output_length = signal_length + kernel_length - 1
    
    fft_size = 1 << (output_length - 1).bit_length()
    
    padded_signal = np.pad(signal, (0, fft_size - signal_length), mode='constant')
    padded_kernel = np.pad(kernel, (0, fft_size - kernel_length), mode='constant')
    
    signal_frequency = fft_iterative(padded_signal)
    kernel_frequency = fft_iterative(padded_kernel)
    
    result_frequency = signal_frequency * kernel_frequency
    
    result = ifft(result_frequency)
    
    return np.real(result[:output_length])


def convolve_2d_direct(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    output_height = image_height + kernel_height - 1
    output_width = image_width + kernel_width - 1
    result = np.zeros((output_height, output_width))
    
    for output_row in range(output_height):
        for output_col in range(output_width):
            for kernel_row in range(kernel_height):
                for kernel_col in range(kernel_width):
                    image_row = output_row - kernel_row
                    image_col = output_col - kernel_col
                    if 0 <= image_row < image_height and 0 <= image_col < image_width:
                        result[output_row, output_col] += image[image_row, image_col] * kernel[kernel_row, kernel_col]
    
    return result


def convolve_2d_fft(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    output_height = image_height + kernel_height - 1
    output_width = image_width + kernel_width - 1
    
    fft_height = 1 << (output_height - 1).bit_length()
    fft_width = 1 << (output_width - 1).bit_length()
    
    padded_image = np.pad(image, ((0, fft_height - image_height), (0, fft_width - image_width)), mode='constant')
    padded_kernel = np.pad(kernel, ((0, fft_height - kernel_height), (0, fft_width - kernel_width)), mode='constant')
    
    image_frequency = fft_2d(padded_image)
    kernel_frequency = fft_2d(padded_kernel)
    
    result_frequency = image_frequency * kernel_frequency
    
    result = ifft_2d(result_frequency)
    
    return np.real(result[:output_height, :output_width])


def convolve_2d_fft_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    full_result = convolve_2d_fft(image, kernel)
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    start_row = kernel_height // 2
    start_col = kernel_width // 2
    return full_result[start_row:start_row + image_height, start_col:start_col + image_width]
