import numpy as np
import time
from fft import fft_iterative, ifft
from convolution import convolve_1d_direct, convolve_1d_fft, convolve_2d_direct, convolve_2d_fft, convolve_2d_fft_same


def demo_1d_convolution():
    print("="*80)
    print("1D SIGNAL CONVOLUTION DEMO")
    print("="*80)
    
    signal = np.array([1, 2, 3, 4, 5])
    kernel = np.array([0.5, 1, 0.5])
    
    print(f"\nSignal: {signal}")
    print(f"Kernel: {kernel}")
    
    direct_result = convolve_1d_direct(signal, kernel)
    fft_result = convolve_1d_fft(signal, kernel)
    
    print(f"Direct result: {direct_result}")
    print(f"FFT result:    {fft_result}")
    print(f"Match: {np.allclose(direct_result, fft_result)}")
    
    print("\n" + "-"*80)
    print("PERFORMANCE BENCHMARK")
    print("-"*80)
    
    signal_sizes = [128, 256, 512, 1024]
    kernel_size = 32
    
    print(f"Kernel size: {kernel_size}\n")
    print(f"{'Size':<10} {'Direct (ms)':<15} {'FFT (ms)':<15} {'Speedup':<10}")
    print("-"*80)
    
    for signal_size in signal_sizes:
        test_signal = np.random.randn(signal_size)
        test_kernel = np.random.randn(kernel_size)
        
        start_time = time.perf_counter()
        _ = convolve_1d_direct(test_signal, test_kernel)
        direct_duration = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        _ = convolve_1d_fft(test_signal, test_kernel)
        fft_duration = (time.perf_counter() - start_time) * 1000
        
        speedup = direct_duration / fft_duration
        print(f"{signal_size:<10} {direct_duration:<15.3f} {fft_duration:<15.3f} {speedup:<10.2f}x")


def demo_2d_image_convolution():
    print("\n" + "="*80)
    print("2D IMAGE CONVOLUTION DEMO")
    print("="*80)
    
    image_size = 64
    test_image = np.zeros((image_size, image_size))
    center = image_size // 2
    y_coords, x_coords = np.ogrid[:image_size, :image_size]
    circle_mask = (x_coords - center)**2 + (y_coords - center)**2 <= (image_size//4)**2
    test_image[circle_mask] = 1.0
    
    blur_kernel = np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16.0
    
    print(f"\nImage size: {test_image.shape}")
    print(f"Kernel:\n{blur_kernel}")
    
    filtered_image = convolve_2d_fft_same(test_image, blur_kernel)
    
    print(f"Filtered image range: [{filtered_image.min():.3f}, {filtered_image.max():.3f}]")
    
    print("\n" + "-"*80)
    print("PERFORMANCE BENCHMARK")
    print("-"*80)
    
    image_sizes = [32, 64, 128]
    kernel_dimensions = (5, 5)
    
    print(f"Kernel size: {kernel_dimensions}\n")
    print(f"{'Size':<15} {'Direct (ms)':<15} {'FFT (ms)':<15} {'Speedup':<10}")
    print("-"*80)
    
    for image_size in image_sizes:
        test_image = np.random.randn(image_size, image_size)
        test_kernel = np.random.randn(*kernel_dimensions)
        
        if image_size <= 128:
            start_time = time.perf_counter()
            _ = convolve_2d_direct(test_image, test_kernel)
            direct_duration = (time.perf_counter() - start_time) * 1000
        else:
            direct_duration = float('inf')
        
        start_time = time.perf_counter()
        _ = convolve_2d_fft(test_image, test_kernel)
        fft_duration = (time.perf_counter() - start_time) * 1000
        
        if direct_duration != float('inf'):
            speedup = direct_duration / fft_duration
            print(f"{image_size}×{image_size:<10} {direct_duration:<15.3f} {fft_duration:<15.3f} {speedup:<10.2f}x")
        else:
            print(f"{image_size}×{image_size:<10} {'too slow':<15} {fft_duration:<15.3f} {'N/A':<10}")


def demo_image_filters():
    print("\n" + "="*80)
    print("IMAGE FILTERING EXAMPLES")
    print("="*80)
    
    image_size = 128
    test_image = np.zeros((image_size, image_size))
    center = image_size // 2
    y_coords, x_coords = np.ogrid[:image_size, :image_size]
    circle_mask = (x_coords - center)**2 + (y_coords - center)**2 <= (image_size//4)**2
    test_image[circle_mask] = 1.0
    test_image += np.random.randn(image_size, image_size) * 0.1
    
    filter_kernels = {
        'Blur 3×3': np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16,
        'Sharpen': np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
        'Edge (Sobel X)': np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
        'Edge (Sobel Y)': np.array([[-1,-2,-1],[0,0,0],[1,2,1]]),
    }
    
    print(f"\nImage size: {test_image.shape}")
    print(f"Applying filters using FFT convolution...\n")
    
    for filter_name, filter_kernel in filter_kernels.items():
        start_time = time.perf_counter()
        filtered_result = convolve_2d_fft_same(test_image, filter_kernel)
        elapsed_time = (time.perf_counter() - start_time) * 1000
        
        print(f"{filter_name:<20} Time: {elapsed_time:>6.2f} ms  Range: [{filtered_result.min():>6.2f}, {filtered_result.max():>6.2f}]")


def demo_complexity_analysis():
    print("\n" + "="*80)
    print("THEORETICAL COMPLEXITY ANALYSIS")
    print("="*80)
    
    print("\n1D Convolution:")
    print("-"*80)
    
    for signal_length in [256, 512, 1024, 2048]:
        kernel_length = 32
        output_length = signal_length + kernel_length - 1
        fft_size = 1 << (output_length - 1).bit_length()
        
        direct_operations = signal_length * kernel_length
        fft_operations = 3 * fft_size * np.log2(fft_size)
        theoretical_speedup = direct_operations / fft_operations
        
        print(f"Signal: {signal_length:4d}, Kernel: {kernel_length:2d}")
        print(f"  Direct: {direct_operations:,} ops (O(n×m))")
        print(f"  FFT:    {int(fft_operations):,} ops (O(n log n))")
        print(f"  Speedup: {theoretical_speedup:.2f}x\n")
    
    print("2D Convolution:")
    print("-"*80)
    
    for image_size in [128, 256, 512]:
        kernel_size = 5
        direct_operations = (image_size ** 2) * (kernel_size ** 2)
        fft_size = 1 << ((image_size + kernel_size - 1) - 1).bit_length()
        fft_operations = 3 * (fft_size ** 2) * np.log2(fft_size) * 2
        theoretical_speedup = direct_operations / fft_operations
        
        print(f"Image: {image_size}×{image_size}, Kernel: {kernel_size}×{kernel_size}")
        print(f"  Direct: {direct_operations:,} ops (O(n²×m²))")
        print(f"  FFT:    {int(fft_operations):,} ops (O(n² log n))")
        print(f"  Speedup: {theoretical_speedup:.2f}x\n")


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║             EFFICIENT CONVOLUTION USING FFT - COMPLETE DEMO                   ║
║                                                                               ║
║  Demonstrates the Convolution Theorem: conv(f,g) = IFFT(FFT(f) × FFT(g))    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    demo_1d_convolution()
    demo_2d_image_convolution()
    demo_image_filters()
    demo_complexity_analysis()
    
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print("\nKey Insights:")
    print("  • FFT transforms O(n²) convolution to O(n log n)")
    print("  • Speedup increases with signal/image size")
    print("  • Same algorithm works for 1D signals and 2D images")
    print("  • All implemented from scratch without external FFT libraries")


if __name__ == "__main__":
    main()
