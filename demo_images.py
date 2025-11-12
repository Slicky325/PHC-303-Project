import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import requests

from convolution import convolve_2d_fft_same, convolve_2d_direct


output_directory = Path("demo_images_out")
output_directory.mkdir(exist_ok=True)

image_a_urls = [
    "https://picsum.photos/800/600",
    "https://placekitten.com/800/600",
]
image_b_urls = [
    "https://picsum.photos/seed/kernel/64/64",
    "https://placekitten.com/64/64",
]

target_image_size = (256, 256)
max_kernel_dimension = 31


def download_image(url: str) -> Image.Image:
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
        "python-requests/2.x",
    ]

    last_error = None
    for attempt in range(3):
        user_agent = user_agents[attempt % len(user_agents)]
        headers = {"User-Agent": user_agent}
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception as error:
            last_error = error
            time.sleep(0.5 * (2 ** attempt))

    raise RuntimeError(f"Failed to download {url}: {last_error}")


def try_download_from_list(url_list):
    last_error = None
    for url in url_list:
        try:
            return download_image(url)
        except Exception as error:
            last_error = error
            print(f"Download failed for {url}: {error}")
    raise RuntimeError(f"All downloads failed; last error: {last_error}")


def generate_synthetic_image(shape=(512, 512)) -> Image.Image:
    height, width = shape
    x_coords = np.linspace(0.0, 1.0, width)
    y_coords = np.linspace(0.0, 1.0, height)[:, None]
    gradient = (0.6 * x_coords + 0.4 * y_coords)
    noise = 0.05 * np.random.RandomState(0).randn(height, width)
    image_array = np.clip(gradient + noise, 0.0, 1.0)
    image = Image.fromarray((image_array * 255.0).astype(np.uint8))
    return image


def generate_gaussian_kernel(size=31, sigma=None) -> np.ndarray:
    if sigma is None:
        sigma = size / 6.0
    axis = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    x_grid, y_grid = np.meshgrid(axis, axis)
    kernel = np.exp(-(x_grid**2 + y_grid**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def to_grayscale_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("L"), dtype=np.float64) / 255.0


def save_image_array(image_array: np.ndarray, path: Path) -> None:
    clipped_array = np.clip(image_array, 0.0, 1.0)
    image = Image.fromarray((clipped_array * 255.0).astype(np.uint8))
    image.save(path)


def make_kernel_from_image(image_array: np.ndarray, max_dimension: int) -> np.ndarray:
    height, width = image_array.shape
    if height > max_dimension or width > max_dimension:
        scale_factor = max_dimension / max(height, width)
        new_height = int(round(height * scale_factor))
        new_width = int(round(width * scale_factor))
        image = Image.fromarray((image_array * 255.0).astype(np.uint8))
        image = image.resize((new_width, new_height), Image.LANCZOS)
        kernel_array = np.array(image, dtype=np.float64) / 255.0
    else:
        kernel_array = image_array.copy()

    kernel_sum = kernel_array.sum()
    if kernel_sum == 0:
        return kernel_array
    return kernel_array / kernel_sum


def main():
    print("Downloading images...")
    try:
        image_a = try_download_from_list(image_a_urls)
    except Exception:
        print("All remote image A downloads failed — using synthetic image.")
        image_a = generate_synthetic_image(target_image_size)

    try:
        image_b = try_download_from_list(image_b_urls)
    except Exception:
        print("All remote image B downloads failed — using synthetic kernel image.")
        gaussian_kernel = generate_gaussian_kernel(size=max_kernel_dimension)
        image_b = Image.fromarray((np.clip(gaussian_kernel, 0.0, 1.0) * 255.0).astype(np.uint8))

    image_a.save(output_directory / "image_A_orig.png")
    image_b.save(output_directory / "image_B_orig.png")

    print("Converting to grayscale arrays and resizing to target size...")
    image_a = image_a.resize(target_image_size, Image.LANCZOS)
    image_b = image_b.resize(target_image_size, Image.LANCZOS)

    image_a_array = to_grayscale_array(image_a)
    image_b_array = to_grayscale_array(image_b)

    print(f"Image A shape: {image_a_array.shape}, Image B shape: {image_b_array.shape}")
    
    kernel = image_b_array / image_b_array.sum() if image_b_array.sum() > 0 else image_b_array
    print(f"Kernel shape: {kernel.shape}")

    save_image_array(image_a_array, output_directory / "image_A_gray.png")
    save_image_array(kernel, output_directory / "image_B_kernel.png")

    image_height, image_width = image_a_array.shape
    kernel_height, kernel_width = kernel.shape
    image_pixels = image_height * image_width
    kernel_pixels = kernel_height * kernel_width

    print("\n=== Complexity Analysis ===")
    print(f"Image size: {image_height}×{image_width} = {image_pixels:,} pixels")
    print(f"Kernel size: {kernel_height}×{kernel_width} = {kernel_pixels:,} pixels")
    print(f"\nTheoretical complexity:")
    print(f"  Direct:   O(n² · m²) ≈ O({image_pixels} · {kernel_pixels}) = {image_pixels * kernel_pixels:,} operations")
    
    output_height = image_height + kernel_height - 1
    output_width = image_width + kernel_width - 1
    fft_height = 1 << (output_height - 1).bit_length()
    fft_width = 1 << (output_width - 1).bit_length()
    fft_operations = fft_height * fft_width * (np.log2(fft_height) + np.log2(fft_width))
    print(f"  FFT:      O(n² log n) ≈ {int(fft_operations):,} operations (padded to {fft_height}×{fft_width})")
    print(f"  Expected speedup: ~{(image_pixels * kernel_pixels) / fft_operations:.1f}x\n")

    print("Running DIRECT convolution (spatial domain)...")
    direct_start_time = time.perf_counter()
    direct_result = convolve_2d_direct(image_a_array, kernel)
    direct_end_time = time.perf_counter()
    direct_duration = direct_end_time - direct_start_time
    print(f"  Time: {direct_duration:.3f} s")

    print("\nRunning FFT-based convolution (frequency domain)...")
    fft_start_time = time.perf_counter()
    fft_result = convolve_2d_fft_same(image_a_array, kernel)
    fft_end_time = time.perf_counter()
    fft_duration = fft_end_time - fft_start_time
    print(f"  Time: {fft_duration:.3f} s")

    start_row = kernel_height // 2
    start_col = kernel_width // 2
    direct_result_cropped = direct_result[start_row:start_row + image_height, start_col:start_col + image_width]
    
    max_difference = np.max(np.abs(direct_result_cropped - fft_result))
    print(f"\n=== Results ===")
    print(f"Max absolute difference (direct vs FFT): {max_difference:.2e}")
    print(f"Match: {'✓ PASS' if max_difference < 1e-6 else '✗ FAIL'}")
    
    measured_speedup = direct_duration / fft_duration if fft_duration > 0 else float('inf')
    print(f"\nMeasured speedup: {measured_speedup:.2f}x (FFT vs Direct)")
    print(f"FFT is {'FASTER' if measured_speedup > 1 else 'SLOWER'} than direct for this size.")

    save_image_array(fft_result, output_directory / "image_A_convolved_fft.png")
    save_image_array(direct_result_cropped, output_directory / "image_A_convolved_direct.png")
    
    print(f"\nSaved results to {output_directory.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
