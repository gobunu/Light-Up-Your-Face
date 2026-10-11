"""Linear-light wavelet adjustment for fill-light strength."""
from collections import deque
import numpy as np
from dtcwt.numpy import Transform2d, Pyramid

def fill_mask_holes(mask01: np.ndarray) -> np.ndarray:
    mask = mask01 > 0.5
    (H, W) = mask.shape
    bg = ~mask
    visited = np.zeros((H, W), dtype=np.bool_)
    q = deque()
    for j in range(W):
        if bg[0, j] and (not visited[0, j]):
            visited[0, j] = True
            q.append((0, j))
        if bg[H - 1, j] and (not visited[H - 1, j]):
            visited[H - 1, j] = True
            q.append((H - 1, j))
    for i in range(H):
        if bg[i, 0] and (not visited[i, 0]):
            visited[i, 0] = True
            q.append((i, 0))
        if bg[i, W - 1] and (not visited[i, W - 1]):
            visited[i, W - 1] = True
            q.append((i, W - 1))
    while q:
        (i, j) = q.popleft()
        ni = i - 1
        if ni >= 0 and bg[ni, j] and (not visited[ni, j]):
            visited[ni, j] = True
            q.append((ni, j))
        ni = i + 1
        if ni < H and bg[ni, j] and (not visited[ni, j]):
            visited[ni, j] = True
            q.append((ni, j))
        nj = j - 1
        if nj >= 0 and bg[i, nj] and (not visited[i, nj]):
            visited[i, nj] = True
            q.append((i, nj))
        nj = j + 1
        if nj < W and bg[i, nj] and (not visited[i, nj]):
            visited[i, nj] = True
            q.append((i, nj))
    holes = bg & ~visited
    filled = mask | holes
    return filled.astype(np.float32)

def srgb_to_linear(x01: np.ndarray) -> np.ndarray:
    x = np.clip(x01.astype(np.float32), 0.0, 1.0)
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    return np.clip(lin, 0.0, 1.0)

def linear_to_srgb(lin01: np.ndarray) -> np.ndarray:
    x = np.clip(lin01.astype(np.float32), 0.0, 1.0)
    a = 0.055
    srgb = np.where(x <= 0.0031308, 12.92 * x, (1 + a) * x ** (1 / 2.4) - a)
    return np.clip(srgb, 0.0, 1.0)

def gaussian_kernel1d(sigma: float, radius: int | None=None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    if radius is None:
        radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum() + 1e-12
    return k.astype(np.float32)

def gaussian_blur2d_reflect(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img.astype(np.float32, copy=False)
    k = gaussian_kernel1d(sigma)
    r = len(k) // 2
    tmp = np.pad(img, ((0, 0), (r, r)), mode='reflect')
    tmp = np.apply_along_axis(lambda m: np.convolve(m, k, mode='valid'), 1, tmp)
    tmp2 = np.pad(tmp, ((r, r), (0, 0)), mode='reflect')
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode='valid'), 0, tmp2)
    return out.astype(np.float32)

def binary_erosion_3x3(mask01: np.ndarray, iters: int=1) -> np.ndarray:
    x = mask01.astype(np.float32)
    if x.ndim == 3 and x.shape[-1] == 3:
        y = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
        m = y > 0.0
    else:
        m = x > 0.0
    m = m.astype(np.float32)
    (H, W) = m.shape
    iters = max(0, int(iters))
    for _ in range(iters):
        p = np.pad(m, ((1, 1), (1, 1)), mode='edge')
        neigh = [p[i:i + H, j:j + W] for i in range(3) for j in range(3)]
        m = np.minimum.reduce(neigh)
    return m.astype(np.float32)

def feather_positive_delta(delta: np.ndarray, feather_sigma: float) -> np.ndarray:
    delta_pos = np.maximum(delta, 0.0).astype(np.float32)
    mask = (delta_pos > 0.01).astype(np.float32)
    mask = fill_mask_holes(mask)
    mask = binary_erosion_3x3(mask, iters=2)
    delta_pos_eroded = delta_pos * mask
    mask2 = (delta_pos_eroded > 0.0).astype(np.float32)
    mask2 = fill_mask_holes(mask2)
    mask_f = gaussian_blur2d_reflect(mask2, feather_sigma)
    mask_f = np.clip(mask_f, 0.0, 1.0)
    return delta_pos_eroded * mask_f

def reconstruct_rgb_with_mixed_LL_dtcwt(img1_rgb: np.ndarray, img2_rgb: np.ndarray, img3_rgb: np.ndarray, level: int, x: float, feather_sigma: float=0.0) -> np.ndarray:
    t = Transform2d()
    out = np.zeros_like(img1_rgb, dtype=np.float32)
    for c in range(3):
        ch1 = img1_rgb[..., c].astype(np.float32)
        ch2 = img2_rgb[..., c].astype(np.float32)
        ch3 = img3_rgb[..., c].astype(np.float32)
        pyr1 = t.forward(ch1, nlevels=level)
        pyr2 = t.forward(ch2, nlevels=level)
        pyr3 = t.forward(ch3, nlevels=level)
        (yl1, yh1) = (pyr1.lowpass, pyr1.highpasses)
        (yl2, _yh2) = (pyr2.lowpass, pyr2.highpasses)
        (yl3, _yh3) = (pyr3.lowpass, pyr3.highpasses)
        delta = x * (yl1 - yl2)
        sigma_low = feather_sigma / 2 ** level if feather_sigma > 0 else 0.0
        delta_pos_feather = feather_positive_delta(delta, sigma_low)
        yl_mixed = yl3 + delta_pos_feather
        yh_mixed = yh1
        pyr_out = Pyramid(lowpass=yl_mixed, highpasses=yh_mixed)
        rec_c = t.inverse(pyr_out)
        out[..., c] = rec_c.real.astype(np.float32)
    return out

def adjust_strength(prediction, source, strength):
    """Retain predicted detail while adjusting positive low-frequency light."""
    height, width = source.shape[:2]
    padding = ((0, height % 2), (0, width % 2), (0, 0))
    prediction = np.pad(prediction, padding, mode='edge')
    source = np.pad(source, padding, mode='edge')
    pred_linear = srgb_to_linear(prediction.astype(np.float32)) * 255.0
    source_linear = srgb_to_linear(source.astype(np.float32)) * 255.0
    result = reconstruct_rgb_with_mixed_LL_dtcwt(
        pred_linear, source_linear, source_linear, level=1,
        x=strength, feather_sigma=2,
    )
    return linear_to_srgb(np.clip(result / 255.0, 0.0, 1.0))[:height, :width]
