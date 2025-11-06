"""
Complete Demo: Efficient Convolution using FFT
Demonstrates 1D signal and 2D image convolution with complexity analysis.
"""

import numpy as np
import time
from fft import fft_iterative, ifft
from convolution import convolve_1d_direct, convolve_1d_fft, convolve_2d_direct, convolve_2d_fft, convolve_2d_fft_same


def demo_1d_convolution():
    """Demonstrate 1D signal convolution."""
    print("="*80)
    print("1D SIGNAL CONVOLUTION DEMO")
    print("="*80)
    
    # Simple example
    signal = np.array([1, 2, 3, 4, 5])
    kernel = np.array([0.5, 1, 0.5])
    
    print(f"\nSignal: {signal}")
    print(f"Kernel: {kernel}")
    
    result_direct = convolve_1d_direct(signal, kernel)
    result_fft = convolve_1d_fft(signal, kernel)
    
    print(f"Direct result: {result_direct}")
    print(f"FFT result:    {result_fft}")
    print(f"Match: {np.allclose(result_direct, result_fft)}")
    
    # Performance comparison
    print("\n" + "-"*80)
    print("PERFORMANCE BENCHMARK")
    print("-"*80)
    
    sizes = [128, 256, 512, 1024]
    kernel_size = 32
    
    print(f"Kernel size: {kernel_size}\n")
    print(f"{'Size':<10} {'Direct (ms)':<15} {'FFT (ms)':<15} {'Speedup':<10}")
    print("-"*80)
    
    for size in sizes:
        sig = np.random.randn(size)
        ker = np.random.randn(kernel_size)
        
        # Time direct
        start = time.perf_counter()
        _ = convolve_1d_direct(sig, ker)
        direct_time = (time.perf_counter() - start) * 1000
        
        # Time FFT
        start = time.perf_counter()
        _ = convolve_1d_fft(sig, ker)
        fft_time = (time.perf_counter() - start) * 1000
        
        speedup = direct_time / fft_time
        print(f"{size:<10} {direct_time:<15.3f} {fft_time:<15.3f} {speedup:<10.2f}x")


def demo_2d_image_convolution():
    """Demonstrate 2D image convolution."""
    print("\n" + "="*80)
    print("2D IMAGE CONVOLUTION DEMO")
    print("="*80)
    
    # Create test image (circle)
    size = 64
    image = np.zeros((size, size))
    center = size // 2
    y, x = np.ogrid[:size, :size]
    mask = (x - center)**2 + (y - center)**2 <= (size//4)**2
    image[mask] = 1.0
    
    # Blur kernel
    blur = np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16.0
    
    print(f"\nImage size: {image.shape}")
    print(f"Kernel:\n{blur}")
    
    # Apply filter
    filtered = convolve_2d_fft_same(image, blur)
    
    print(f"Filtered image range: [{filtered.min():.3f}, {filtered.max():.3f}]")
    
    # Performance comparison
    print("\n" + "-"*80)
    print("PERFORMANCE BENCHMARK")
    print("-"*80)
    
    sizes = [32, 64, 128]
    kernel_size = (5, 5)
    
    print(f"Kernel size: {kernel_size}\n")
    print(f"{'Size':<15} {'Direct (ms)':<15} {'FFT (ms)':<15} {'Speedup':<10}")
    print("-"*80)
    
    for size in sizes:
        img = np.random.randn(size, size)
        ker = np.random.randn(*kernel_size)
        
        # Time direct (skip for large sizes)
        if size <= 128:
            start = time.perf_counter()
            _ = convolve_2d_direct(img, ker)
            direct_time = (time.perf_counter() - start) * 1000
        else:
            direct_time = float('inf')
        
        # Time FFT
        start = time.perf_counter()
        _ = convolve_2d_fft(img, ker)
        fft_time = (time.perf_counter() - start) * 1000
        
        if direct_time != float('inf'):
            speedup = direct_time / fft_time
            print(f"{size}×{size:<10} {direct_time:<15.3f} {fft_time:<15.3f} {speedup:<10.2f}x")
        else:
            print(f"{size}×{size:<10} {'too slow':<15} {fft_time:<15.3f} {'N/A':<10}")


def demo_image_filters():
    """Demonstrate various image filters."""
    print("\n" + "="*80)
    print("IMAGE FILTERING EXAMPLES")
    print("="*80)
    
    # Create test image
    size = 128
    image = np.zeros((size, size))
    center = size // 2
    y, x = np.ogrid[:size, :size]
    mask = (x - center)**2 + (y - center)**2 <= (size//4)**2
    image[mask] = 1.0
    image += np.random.randn(size, size) * 0.1
    
    # Define filters
    filters = {
        'Blur 3×3': np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16,
        'Sharpen': np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
        'Edge (Sobel X)': np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
        'Edge (Sobel Y)': np.array([[-1,-2,-1],[0,0,0],[1,2,1]]),
    }
    
    print(f"\nImage size: {image.shape}")
    print(f"Applying filters using FFT convolution...\n")
    
    for name, kernel in filters.items():
        start = time.perf_counter()
        filtered = convolve_2d_fft_same(image, kernel)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"{name:<20} Time: {elapsed:>6.2f} ms  Range: [{filtered.min():>6.2f}, {filtered.max():>6.2f}]")


def demo_complexity_analysis():
    """Show theoretical complexity analysis."""
    print("\n" + "="*80)
    print("THEORETICAL COMPLEXITY ANALYSIS")
    print("="*80)
    
    print("\n1D Convolution:")
    print("-"*80)
    
    for sig_len in [256, 512, 1024, 2048]:
        ker_len = 32
        output_len = sig_len + ker_len - 1
        fft_len = 1 << (output_len - 1).bit_length()
        
        direct_ops = sig_len * ker_len
        fft_ops = 3 * fft_len * np.log2(fft_len)
        speedup = direct_ops / fft_ops
        
        print(f"Signal: {sig_len:4d}, Kernel: {ker_len:2d}")
        print(f"  Direct: {direct_ops:,} ops (O(n×m))")
        print(f"  FFT:    {int(fft_ops):,} ops (O(n log n))")
        print(f"  Speedup: {speedup:.2f}x\n")
    
    print("2D Convolution:")
    print("-"*80)
    
    for img_size in [128, 256, 512]:
        ker_size = 5
        direct_ops = (img_size ** 2) * (ker_size ** 2)
        fft_size = 1 << ((img_size + ker_size - 1) - 1).bit_length()
        fft_ops = 3 * (fft_size ** 2) * np.log2(fft_size) * 2
        speedup = direct_ops / fft_ops
        
        print(f"Image: {img_size}×{img_size}, Kernel: {ker_size}×{ker_size}")
        print(f"  Direct: {direct_ops:,} ops (O(n²×m²))")
        print(f"  FFT:    {int(fft_ops):,} ops (O(n² log n))")
        print(f"  Speedup: {speedup:.2f}x\n")


def main():
    """Run complete demonstration."""
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
