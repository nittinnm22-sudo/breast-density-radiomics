# breast-density-radiomics

PET/CT viewer with CT breast 3D volumetric density, tissue complexity, and FDG uptake assessment module.

## Files

- `petct_viewer_v21_full_metric_export_orient_fix_v5_fixed2.py` — main PET/CT viewer (v21)
- `petct_breast_density_unified_workstation.py` — unified breast CT density + parenchymal complexity workstation

## Breast CT density + complexity engine

The unified workstation implements a research-grade pipeline:

1. **Whole-breast segmentation** (HU `-300` to `+200`) with morphological cleanup and right/left split.
2. **Fibroglandular segmentation** within whole breast (HU `>-25`).
3. **Parenchymal ROI cleanup** to reduce tumor-presence bias by excluding:
   - skin shell
   - posterior pectoralis-like high-HU voxels
   - clip/marker-like very high HU small components
   - largest high-HU candidate component (primary tumor placeholder for review)

### Density metrics

The density engine reports bilateral and side-specific volumetric outputs including:
- right/left total breast volume (cc)
- right/left fibroglandular volume (cc)
- right/left fat volume (cc)
- right/left volumetric density (%)
- bilateral density (%)
- density asymmetry (%)
- fibroglandular volume asymmetry (cc)
- absolute fibroglandular volume (cc)

### Parenchymal complexity features

Complexity is computed from the **cleaned parenchymal mask** (not whole-breast mask), including:
- Shape/morphology features (surface/volume, sphericity, PCA-derived axes)
- First-order HU features
- Texture families: GLCM, GLRLM, GLSZM, GLDM, NGTDM

The script exposes a manuscript shortlist panel with 21 features:
`surface_to_volume_ratio`, `sphericity`, `elongation`, `flatness`,
`std_hu`, `skewness`, `kurtosis`, `entropy`, `uniformity`,
`glcm_contrast`, `glcm_correlation`, `glcm_joint_entropy`,
`glrlm_short_run_emphasis`, `glrlm_long_run_emphasis`, `glrlm_run_entropy`,
`glszm_zone_entropy`, `glszm_small_area_emphasis`, `glszm_large_area_emphasis`,
`gldm_dependence_non_uniformity`, `ngtdm_coarseness`, `ngtdm_busyness`.

## Dependencies

- Required: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`, `scikit-image`
- Optional texture backend: `pyradiomics` (+ `SimpleITK`)

## Example usage

```python
import numpy as np
from petct_breast_density_unified_workstation import (
    BreastCTDensityEngine,
    BreastSegmentor,
    ParenchymalComplexityEngine,
)

ct = np.random.normal(-100, 80, size=(128, 256, 256)).astype(np.float32)
spacing = (1.0, 1.0, 1.0)

density = BreastCTDensityEngine().compute_volumetric_density(ct, spacing)
segmentor = BreastSegmentor()
masks = segmentor.segment_whole_breast(ct)
fg = segmentor.segment_fibroglandular(ct, masks["right_mask"] | masks["left_mask"])
parenchymal = segmentor.exclude_non_parenchymal(fg, ct)
complexity = ParenchymalComplexityEngine(ct, parenchymal, spacing).compute_all()
```
