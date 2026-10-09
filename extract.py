"""DINOv3 patch-global similarity maps and H, C, FuzzyEn, LZC for every painting.

usage:  python extract.py --images VAPS/ --weights dinov3-vitb16/ [--device mps]
writes: results/per_image_measures.csv, results/ordinal_pattern_counts.csv
        (--save-maps also writes the similarity maps; keep these private, as
        they show the paintings' composition)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoModel

import measures as ms
import preprocess as pp

SEED, DRAWS = 20250902, 100


def n_windows(box):
    v = pp.valid_mask(box)
    r, c = v.any(1).sum(), v.any(0).sum()
    return (r - 1) * (c - 1)


def box_of(path):
    with Image.open(path) as im:
        w, h = ImageOps.exif_transpose(im).size
    s = pp.SIZE / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    return ((pp.SIZE - nw) // 2, (pp.SIZE - nh) // 2, nw, nh)


@torch.no_grad()
def similarity_map(model, canvas, valid, device):
    """Cosine similarity of each patch token to the normalised mean of the
    valid (painting) patch tokens. Class and register tokens are dropped."""
    x = torch.from_numpy(pp.normalise(canvas)).unsqueeze(0).to(device)
    h = model(pixel_values=x).last_hidden_state[0]
    p = h[1 + model.config.num_register_tokens:]
    F = torch.nn.functional.normalize(p, dim=-1).cpu().numpy()
    g = F[valid.ravel()].mean(0)
    g = g / np.linalg.norm(g)
    return (F @ g).reshape(valid.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results")
    ap.add_argument("--save-maps", default=None, help="optional .npz path (not for sharing)")
    a = ap.parse_args()

    paths = sorted(p for p in Path(a.images).rglob("*") if p.suffix.lower() == ".jpg")
    # same number of 2x2 windows for every painting = dataset minimum (1155)
    n_target = min(n_windows(box_of(p)) for p in paths)
    print(f"{len(paths)} paintings, {n_target} windows per painting, {DRAWS} draws")

    model = AutoModel.from_pretrained(a.weights).eval().to(a.device)
    rows, counts, maps = [], [], {}
    for p in tqdm(paths):
        rgb = pp.load_rgb(p)
        canvas, box = pp.resize_pad(rgb)
        valid = pp.valid_mask(box)
        M = similarity_map(model, canvas, valid, a.device)
        U = ms.valid_rect(M, valid)
        H, C = ms.hc_fixed_n(M, valid, n_target, DRAWS, SEED)
        Hpx, _ = ms.hc_fixed_n(pp.luminance_512(rgb), None, n_target, DRAWS, SEED)
        rows.append({"image_id": p.stem, "H": H, "C": C,
                     "FuzzyEn": ms.fuzzy_entropy_2d(U), "LZC": ms.lzc_2d(U, seed=SEED),
                     "H_pixel": Hpx, **pp.covariates(rgb, box),
                     "map_rows": U.shape[0], "map_cols": U.shape[1],
                     "n_windows": int(len(ms.ordinal_codes(M, valid)))})
        counts.append([p.stem, *np.bincount(ms.ordinal_codes(M, valid), minlength=24)])
        maps[p.stem] = U.astype(np.float32)

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "per_image_measures.csv", index=False, float_format="%.10g")
    pd.DataFrame(counts, columns=["image_id"] + [f"pattern_{i + 1}" for i in range(24)]
                 ).to_csv(out / "ordinal_pattern_counts.csv", index=False)
    if a.save_maps:
        np.savez_compressed(a.save_maps, **maps)


if __name__ == "__main__":
    main()
