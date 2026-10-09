# DinoV3-VAPS-Entropy-Extraction

# Entropy of DINOv3 representations and perceived complexity in paintings

Code and per-painting results for: permutation entropy (H), statistical complexity (C), fuzzy entropy (FuzzyEn) and Lempel–Ziv complexity (LZC) computed on DINOv3 patch-similarity maps of the 996 VAPS paintings, and their relation to rated visual complexity, arousal, valence and liking.

## Pipeline

1. **Preprocessing** (`preprocess.py`): RGB with EXIF orientation applied; long side resized to 896 px (Lanczos, aspect kept); centred on a grey (128) 896 × 896 canvas; ImageNet normalisation. Patches with ≥ 50% painting pixels are valid.
2. **DINOv3** (`extract.py`): frozen `facebook/dinov3-vitb16-pretrain-lvd1689m`, final layer, 56 × 56 patch tokens (class and 4 register tokens dropped).
3. **Map**: cosine similarity of each L2-normalised patch token to the normalised mean of the valid patch tokens.
4. **Measures** (`measures.py`):
   - **H, C**: 2 × 2 ordinal patterns (24 possible), 1,155 windows per painting (dataset minimum), averaged over 100 random draws; MPR complexity C = D(P,U) · H / D*.
   - **FuzzyEn**: z-scored map, mean-removed 2 × 2 vs 3 × 3 templates, Chebyshev distance, exp(−d²/0.2). Not bounded above.
   - **LZC**: median-binarised map, LZ76 of row- and column-major readouts, each divided by the mean of 10 shuffled copies.
   - **Pixel H**: the same H on luminance (long side ≤ 512 px).
5. **Statistics** (`analysis.py`): correlations (raw, adjusted for image covariates, within genre × period), tertiles, hierarchical regression, separate rater groups, reliability, cross-validation, familiarity, and exploratory category-specific analyses (per content category, raw and within period).

## Run

```bash
pip install -r requirements.txt
python extract.py --images path/to/VAPS --weights path/to/dinov3-vitb16-pretrain-lvd1689m
python analysis.py --vaps path/to/VAPS_Data_per_Artwork.csv
```

Running on CPU (default) reproduces the files in `results/` exactly. Seed: 20250902.

## Data

VAPS images and ratings are **not included**. They are available from the VAPS authors under their licence (Fekete et al., 2022, https://doi.org/10.1037/aca0000460; https://osf.io/a7xcr/). DINOv3 weights are available from Hugging Face (licence-gated). Three VAPS paintings (10317, 10324, 10328) had no image file and are not included.

## Results (`results/`)

| File | Content |
|---|---|
| `per_image_measures.csv` | One row per painting (VAPS ID): H, C, FuzzyEn, LZC, H_pixel, image covariates (mean_luminance, contrast, saturation_mean, colorfulness, pad_fraction), map size (patches), number of valid 2 × 2 windows |
| `ordinal_pattern_counts.csv` | Counts of the 24 ordinal patterns over all valid windows of each map |
| `stats_*.csv` | Output of `analysis.py` |

No images, maps, titles or artist names are distributed; paintings are identified only by their VAPS ID. `extract.py --save-maps maps.npz` writes the similarity maps locally for users with VAPS access.
