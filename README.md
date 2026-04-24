# breast-density-radiomics

PET/CT viewer with CT breast 3D volumetric density, tissue complexity, and FDG uptake assessment module.

## Files

- `petct_viewer_v21_full_metric_export_orient_fix_v5_fixed2.py` — main PET/CT viewer (v21)
- `breast_density_module.py` — breast density & radiomics complexity module (to be added)

## Installation

```bash
pip install -r requirements.txt
```

> **Important — TotalSegmentator (required for accurate breast segmentation)**
>
> TotalSegmentator derives the diaphragm boundary and posterior vertebral
> constraint from real anatomical segmentations.  Without it the breast
> segmentor falls back to a HU-based heuristic that may extend below the
> thorax on some cases.
>
> Install once (PyTorch is pulled in automatically):
> ```bash
> pip install totalsegmentator nibabel
> ```
> A CUDA GPU is optional but speeds up TotalSegmentator inference.
> On first use TotalSegmentator will download model weights (~200 MB).

## Dependencies

| Package | Purpose |
|---|---|
| `PyQt6` | GUI |
| `pyqtgraph` | Real-time image display |
| `pydicom` | DICOM loading |
| `SimpleITK` | Image registration & resampling |
| `numpy`, `scipy`, `scikit-image` | Array maths & morphology |
| `pyradiomics` | Radiomics feature extraction |
| `totalsegmentator`, `nibabel` | Thoracic constraint boundary (required) |
| `pandas`, `matplotlib`, `scikit-learn` | Analysis & visualisation (optional) |
