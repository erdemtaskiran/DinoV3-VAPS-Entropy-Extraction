"""Image loading, resizing/padding, patch validity mask and image covariates."""
import numpy as np
from PIL import Image, ImageOps

SIZE, PATCH, PAD = 896, 16, 128                      # 896 px -> 56 x 56 patches
MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet normalisation
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.uint8)


def resize_pad(rgb, size=SIZE):
    """Long side -> size (Lanczos, aspect kept), centred on a grey canvas."""
    h, w = rgb.shape[:2]
    s = size / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    img = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.Resampling.LANCZOS))
    canvas = np.full((size, size, 3), PAD, np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = img
    return canvas, (x0, y0, nw, nh)


def valid_mask(box, size=SIZE, patch=PATCH, min_frac=0.5):
    """Patches with >= 50% painting pixels (from geometry, not pixel values)."""
    x0, y0, w, h = box
    e = np.arange(0, size + 1, patch)
    oy = np.clip(np.minimum(e[1:], y0 + h) - np.maximum(e[:-1], y0), 0, None)
    ox = np.clip(np.minimum(e[1:], x0 + w) - np.maximum(e[:-1], x0), 0, None)
    return np.outer(oy, ox) / patch ** 2 >= min_frac


def normalise(canvas):
    return ((canvas.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)


def covariates(rgb, box):
    """Computed on the original (unpadded) painting."""
    x = rgb.astype(np.float64) / 255.0
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    lum = 0.2126 * R + 0.7152 * G + 0.0722 * B
    mx, mn = x.max(-1), x.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-12), 0.0)
    rg, yb = R - G, 0.5 * (R + G) - B
    colourful = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())
    return {"mean_luminance": lum.mean(), "contrast": lum.std(),
            "saturation_mean": sat.mean(), "colorfulness": colourful,
            "pad_fraction": 1 - box[2] * box[3] / SIZE ** 2}


def luminance_512(rgb):
    """Rec. 709 luminance, long side <= 512 px (pixel-level H baseline)."""
    x = rgb.astype(np.float64) / 255.0
    lum = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    if max(lum.shape) > 512:
        s = 512 / max(lum.shape)
        im = Image.fromarray((lum * 255).astype(np.uint8)).resize(
            (max(1, int(lum.shape[1] * s)), max(1, int(lum.shape[0] * s))),
            Image.Resampling.LANCZOS)
        lum = np.asarray(im, dtype=np.float64) / 255.0
    return lum
