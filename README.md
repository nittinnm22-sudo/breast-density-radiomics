# breast-density-radiomics

PET/CT viewer with CT breast 3D volumetric density, tissue complexity, and FDG uptake assessment module.

## Unified workstation

Use the unified script:

- `petct_breast_density_unified_workstation.py` — integrated PET/CT GUI workstation + breast density analyzer + ML ROC analysis

## Dependencies

Required:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

Optional image I/O:

- `Pillow`
- `SimpleITK`

## Run

```bash
python petct_breast_density_unified_workstation.py
```

Behavior:

- Runs breast density ML ROC analysis from `breast_density_data.csv` if available (saves `roc_curve.png`)
- Launches the unified PET/CT workstation with inlined breast density dialog and export tools
