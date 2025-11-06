import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import requests

from convolution import convolve_2d_fft_same, convolve_2d_direct


# === Configuration ===
OUT_DIR = Path("demo_images_out")
OUT_DIR.mkdir(exist_ok=True)

# Two example image URL lists (try multiple hosts to avoid 403). If all fail we generate local fallbacks.
IMG_A_URLS = [
    "https://picsum.photos/800/600",  # random image service
    "https://placekitten.com/800/600",
]
IMG_B_URLS = [
    "https://picsum.photos/seed/kernel/64/64",
    "https://placekitten.com/64/64",
]

# Target size for both images (to compare complexity analysis fairly)
TARGET_IMAGE_SIZE = (256, 256)
# If kernel is larger than this, it will be resized down to keep direct convolution reasonable
MAX_KERNEL_DIM = 31


def download_image(url: str) -> Image.Image:
    # Try a few common user-agents and simple retries to avoid 403 from some hosts
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
        "python-requests/2.x",
    ]

    last_exc = None
    for attempt in range(3):
        ua = user_agents[attempt % len(user_agents)]
        headers = {"User-Agent": ua}
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content))
        except Exception as e:
            last_exc = e
            # simple exponential backoff
            time.sleep(0.5 * (2 ** attempt))

    # If we reach here, all attempts failed — surface a helpful error
    raise RuntimeError(f"Failed to download {url}: {last_exc}")


def try_download_from_list(urls):
    last_exc = None
    for url in urls:
        try:
            return download_image(url)
        except Exception as e:
            last_exc = e
            print(f"Download failed for {url}: {e}")
    raise RuntimeError(f"All downloads failed; last error: {last_exc}")


def generate_synthetic_image(shape=(512, 512)) -> Image.Image:
    # Create a smooth gradient + slight noise to act as a test image
    h, w = shape
    x = np.linspace(0.0, 1.0, w)
    y = np.linspace(0.0, 1.0, h)[:, None]
    grad = (0.6 * x + 0.4 * y)
    noise = 0.05 * np.random.RandomState(0).randn(h, w)
    arr = np.clip(grad + noise, 0.0, 1.0)
    img = Image.fromarray((arr * 255.0).astype(np.uint8))
    return img


def generate_gaussian_kernel(size=31, sigma=None) -> np.ndarray:
    if sigma is None:
        sigma = size / 6.0
    ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def to_grayscale_array(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"), dtype=np.float64) / 255.0


def save_image_array(arr: np.ndarray, path: Path) -> None:
    # Clip and convert to uint8
    arr = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((arr * 255.0).astype(np.uint8))
    img.save(path)


def make_kernel_from_image(img_arr: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img_arr.shape
    if h > max_dim or w > max_dim:
        # resize preserving aspect ratio
        scale = max_dim / max(h, w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        img = Image.fromarray((img_arr * 255.0).astype(np.uint8))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        arr = np.array(img, dtype=np.float64) / 255.0
    else:
        arr = img_arr.copy()

    # Normalize kernel to sum to 1 (so output is not arbitrarily bright/dark)
    s = arr.sum()
    if s == 0:
        return arr
    return arr / s


def main():
    print("Downloading images...")
    try:
        a = try_download_from_list(IMG_A_URLS)
    except Exception:
        print("All remote image A downloads failed — using synthetic image.")
        a = generate_synthetic_image(TARGET_IMAGE_SIZE)

    try:
        b = try_download_from_list(IMG_B_URLS)
    except Exception:
        print("All remote image B downloads failed — using synthetic kernel image.")
        # create a synthetic kernel image (grayscale) from gaussian
        g = generate_gaussian_kernel(size=MAX_KERNEL_DIM)
        # convert kernel array to PIL Image for saving/visualization
        b = Image.fromarray((np.clip(g, 0.0, 1.0) * 255.0).astype(np.uint8))

    a.save(OUT_DIR / "image_A_orig.png")
    b.save(OUT_DIR / "image_B_orig.png")

    print("Converting to grayscale arrays and resizing to target size...")
    # Resize both images to TARGET_IMAGE_SIZE for fair complexity comparison
    a = a.resize(TARGET_IMAGE_SIZE, Image.LANCZOS)
    b = b.resize(TARGET_IMAGE_SIZE, Image.LANCZOS)

    a_arr = to_grayscale_array(a)
    b_arr = to_grayscale_array(b)

    print(f"Image A shape: {a_arr.shape}, Image B shape: {b_arr.shape}")
    
    # Normalize B to use as kernel (both images are same size now)
    kernel = b_arr / b_arr.sum() if b_arr.sum() > 0 else b_arr
    print(f"Kernel shape: {kernel.shape}")

    save_image_array(a_arr, OUT_DIR / "image_A_gray.png")
    save_image_array(kernel, OUT_DIR / "image_B_kernel.png")

    # === Complexity Analysis ===
    h, w = a_arr.shape
    k_h, k_w = kernel.shape
    n_pixels = h * w
    k_pixels = k_h * k_w

    print("\n=== Complexity Analysis ===")
    print(f"Image size: {h}×{w} = {n_pixels:,} pixels")
    print(f"Kernel size: {k_h}×{k_w} = {k_pixels:,} pixels")
    print(f"\nTheoretical complexity:")
    print(f"  Direct:   O(n² · m²) ≈ O({n_pixels} · {k_pixels}) = {n_pixels * k_pixels:,} operations")
    
    # FFT pads to next power of 2
    out_h = h + k_h - 1
    out_w = w + k_w - 1
    fft_h = 1 << (out_h - 1).bit_length()
    fft_w = 1 << (out_w - 1).bit_length()
    fft_ops = fft_h * fft_w * (np.log2(fft_h) + np.log2(fft_w))
    print(f"  FFT:      O(n² log n) ≈ {int(fft_ops):,} operations (padded to {fft_h}×{fft_w})")
    print(f"  Expected speedup: ~{(n_pixels * k_pixels) / fft_ops:.1f}x\n")

    # === Run Direct Convolution ===
    print("Running DIRECT convolution (spatial domain)...")
    t0_direct = time.perf_counter()
    out_direct = convolve_2d_direct(a_arr, kernel)
    t1_direct = time.perf_counter()
    time_direct = t1_direct - t0_direct
    print(f"  Time: {time_direct:.3f} s")

    # === Run FFT Convolution ===
    print("\nRunning FFT-based convolution (frequency domain)...")
    t0_fft = time.perf_counter()
    out_fft = convolve_2d_fft_same(a_arr, kernel)
    t1_fft = time.perf_counter()
    time_fft = t1_fft - t0_fft
    print(f"  Time: {time_fft:.3f} s")

    # === Compare Results ===
    # Crop direct output to match 'same' size
    h_start, w_start = k_h // 2, k_w // 2
    out_direct_cropped = out_direct[h_start:h_start + h, w_start:w_start + w]
    
    max_diff = np.max(np.abs(out_direct_cropped - out_fft))
    print(f"\n=== Results ===")
    print(f"Max absolute difference (direct vs FFT): {max_diff:.2e}")
    print(f"Match: {'✓ PASS' if max_diff < 1e-6 else '✗ FAIL'}")
    
    speedup = time_direct / time_fft if time_fft > 0 else float('inf')
    print(f"\nMeasured speedup: {speedup:.2f}x (FFT vs Direct)")
    print(f"FFT is {'FASTER' if speedup > 1 else 'SLOWER'} than direct for this size.")

    # Save outputs
    save_image_array(out_fft, OUT_DIR / "image_A_convolved_fft.png")
    save_image_array(out_direct_cropped, OUT_DIR / "image_A_convolved_direct.png")
    
    print(f"\nSaved results to {OUT_DIR.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
