# An Open-Source Python Workstation for Automated FDG PET/CT Metabolic Quantification, Breast Density Analysis, and IBSI-Compliant Radiomics Feature Extraction

---

## Abstract

FDG PET/CT is the dominant functional oncologic imaging modality, yet reproducible extraction of metabolic and texture-based radiomic features remains a challenge in multi-center research settings. Commercial platforms impose licensing costs, restrict algorithmic transparency, and rarely integrate breast density analysis alongside PET metabolic quantification. We present an open-source Python/PyQt5 GUI workstation — distributed as a single deployable script — that combines (1) SUV-based PET metabolic feature extraction, (2) fully automated bilateral breast segmentation with fibroglandular tissue quantification, and (3) a 50+ feature IBSI-compliant CT radiomics pipeline. The software runs on any Python ≥ 3.8 environment, requires no server or database infrastructure, and exports structured CSV results suitable for direct statistical analysis. All algorithms are fully transparent and peer-reviewable. The workstation is freely available on GitHub under an open-source license, with a documented requirements file and anonymized example data. This tool is designed to accelerate reproducible PET/CT oncologic research, breast density studies, and radiomics biomarker discovery in clinical cohort studies and trials.

**Keywords:** PET/CT, radiomics, breast density, fibroglandular tissue, SUV, IBSI, open-source, Python

---

## 1. Introduction

FDG PET/CT is the cornerstone of staging, response assessment, and surveillance across the major oncologic indications [CITATION]. Clinical interpretation relies primarily on visual assessment supplemented by semi-quantitative standardized uptake value (SUV) metrics; however, reproducibility of SUV measurements across sites and scanner generations remains a persistent challenge that limits multi-center biomarker studies [Boellaard 2015].

Beyond metabolic quantification, there is growing recognition that CT texture features extracted from the same examination — when computed in a standardized, reproducible fashion — provide complementary prognostic and predictive information [Zwanenburg 2020]. In breast oncology, the proportion of fibroglandular tissue (breast density) measured on CT has been proposed as an imaging biomarker of cancer risk and treatment response [CITATION]. Yet no freely available, integrated tool currently combines all three capabilities — PET metabolic metrics, automated breast segmentation with density quantification, and IBSI-compliant CT radiomics — within a single interactive workstation.

Existing open tools address parts of this gap. PETSurfer provides automated PET region parcellation but lacks a general-purpose GUI and does not extract CT radiomics [CITATION]. LIFEx offers a GUI for PET texture analysis but is not fully open-source and does not support breast segmentation [CITATION]. PyRadiomics is the de facto standard for IBSI-compliant radiomics but is a library, not a clinical-workflow GUI, and provides no PET metabolic computation or breast segmentation [van Griethuysen 2017].

We present a workstation that fills this gap. The primary contributions are:
1. A fully open, single-script GUI workstation integrating PET metabolic extraction, multi-planar rendering, automated breast segmentation, fibroglandular tissue quantification, and IBSI-compliant CT radiomics.
2. A novel multi-step breast segmentation pipeline with diaphragm detection, anatomy-constrained left/right separation, and two-stage fibroglandular sub-segmentation.
3. IBSI-compliant radiomics configuration (fixed 25 HU bin width, 1.2 mm isotropic resampling, 50-feature shortlist) aligned with published consensus recommendations.

---

## 2. Software Architecture

### 2.1 Technology Stack

The workstation is implemented in Python 3 and depends entirely on freely available libraries:

| Library | Role |
|---|---|
| PyQt5 | GUI framework, event loop, widget toolkit |
| SimpleITK | DICOM I/O, image resampling, spatial transforms |
| pydicom | DICOM tag parsing (SUV calibration, patient metadata) |
| NumPy / SciPy | Array operations, morphological processing, connected components |
| matplotlib | Multi-planar rendering canvas (axial/coronal/sagittal) |
| pyradiomics | IBSI-compliant feature extraction |
| scikit-image | Supplementary morphological operations |
| TotalSegmentator *(optional)* | Deep-learning anatomy priors for breast segmentation |

The entire application is contained within a single Python script (`petct_viewer_v47.py`), requiring no server, database, or installation beyond a standard pip environment. This design choice maximises portability and reproducibility: a specific script version is unambiguously associated with a specific set of algorithmic behaviours.

### 2.2 Module Overview

| Module | Primary Class / Function | Responsibility |
|---|---|---|
| DICOM loader | `PETCTApp` | Series discovery, CT/PET stack loading, orientation normalisation |
| Multi-planar viewer | `MPRCanvas` | Axial/coronal/sagittal rendering, SUV windowing, overlay compositing |
| Metabolic engine | `MetabolicEngine` | SUVmax/mean/peak, MTV, TLG, histogram |
| Breast segmentor | `BreastSegmentor` | Whole-breast → fibroglandular two-stage pipeline |
| Radiomics engine | `ParenchymalComplexityEngine` | IBSI-compliant CT texture features via pyradiomics |
| Export | `_on_export_figure`, CSV writer | Publication-ready PNG figure + structured CSV batch export |

---

## 3. Functional Modules — Technical Detail

### 3.1 DICOM Loading and PET/CT Co-registration

DICOM series within a study folder are automatically discovered and grouped by `StudyInstanceUID`. PET and CT series are identified by `Modality` tag and paired. If the CT and PET voxel grids differ (different slice thickness, reconstruction matrix, or field of view), SimpleITK resamples the PET volume to the CT grid using B-spline interpolation, preserving spatial correspondence.

SUV conversion uses the body-weight formula:

```
SUV = pixel_value [Bq/mL] × body_weight [g] / (injected_dose [Bq])
```

where `body_weight` is read from the DICOM `PatientWeight` tag, `injected_dose` is derived from `RadiopharmaceuticalInformationSequence` after decay correction to scan start time using `AcquisitionTime` and `RadiopharmaceuticalStartTime`.

### 3.2 PET Metabolic Feature Extraction

The following metrics are computed within a user-defined VOI (manual bounding box or auto-thresholded):

| Metric | Definition |
|---|---|
| SUVmax | Maximum voxel SUV within VOI |
| SUVmean | Mean voxel SUV within segmented volume |
| SUVpeak | Mean SUV within 1-cm³ sphere centred on SUVmax voxel |
| MTV | Metabolic tumour volume: voxel count × voxel volume for SUV ≥ threshold |
| TLG | Total lesion glycolysis: MTV × SUVmean |
| SUV heterogeneity index | Coefficient of variation of SUV distribution |

MTV is computed at user-selectable thresholds (SUV 2.0, 2.5, and 40% of SUVmax). The ISO 50% threshold method is also supported for lesion delineation. A SUV histogram and cumulative distribution are displayed interactively.

### 3.3 Automated Bilateral Breast Segmentation

Breast segmentation is implemented in `BreastSegmentor` as a multi-step pipeline applied to the CT volume.

**Step 1 — Diaphragm detection (`_bd_detect_thorax_z_range`):**  
Axial HU profiles are analysed to detect the diaphragm position. A self-calibrating threshold `max(0.05, peak_fraction × 0.20)` is applied with a 3-slice look-ahead buffer. A hard superior cap at 75% of the image z-extent prevents bowel gas artefacts from pushing the detected diaphragm into the abdomen.

**Step 2 — Anterior thorax restriction:**  
The volume is cropped to the anterior 55% of the AP dimension. A body outline mask (HU > −200) is eroded to remove the patient table and air background.

**Step 3 — Left/right split:**  
The anterior thorax mask is divided at the image midline column to produce independent left and right candidate regions.

**Step 4 — Largest anterior connected component (`_bd_keep_largest_anterior_component`):**  
For each side, 3D connected-component labelling is performed. Only the largest connected component whose centroid lies in the anterior 65% of the FOV (y < ny × 0.65) is retained. This step discards liver, spleen, and other posterior abdominal fragments that may survive the anterior crop.

**Step 5 — Optional TotalSegmentator anatomy constraints (`_bd_ts_anatomy_constraints`):**  
If TotalSegmentator is available, it is called in fast mode with the `lung` and `vertebrae` ROI subset to derive anatomy-based bounding constraints. The lung superior boundary and vertebrae posterior boundary are used to further restrict the breast mask. If TotalSegmentator is unavailable, HU-heuristic fallback constraints are applied.

**Fibroglandular tissue sub-segmentation (two-stage):**

*Stage A (`segment_fibroglandular`)*: Voxels within the breast mask with HU in [−25, +100] are retained as fibroglandular candidates.

*Stage B (`exclude_non_parenchymal`)*: The candidate mask is refined by:
- 4-voxel skin erosion (`_FG_SKIN_EROSION_VOX = 4`) to remove the skin layer
- Posterior 40% pectoralis exclusion (voxels with y > breast_mask_posterior × 0.60 are discarded)
- Surgical clip removal (small high-HU components)
- Minimum connected-component size filter (100 voxels)

**Quality control flags (`compute_fg_qc_warnings`):**  
A warning is raised if fibroglandular tissue occupies >15% of the skin zone or >20% of the pectoralis zone, indicating possible segmentation spillover.

### 3.4 IBSI-Compliant CT Radiomics

Radiomics feature extraction is performed by `ParenchymalComplexityEngine`, which wraps pyradiomics with a fixed IBSI-aligned configuration:

| Parameter | Value |
|---|---|
| Resampled voxel spacing | 1.2 × 1.2 × 1.2 mm (B-spline interpolation) |
| Discretisation method | Fixed bin width |
| Bin width | 25 HU |
| voxelArrayShift | 0 |
| Feature families | First-order, GLCM, GLRLM, GLSZM, NGTDM, shape |
| Feature shortlist | 50 features |

The 50-feature shortlist was selected to cover the feature families recommended by the IBSI consensus document (Zwanenburg et al. 2020) while excluding redundant and highly correlated features.

For the bilateral merged mask, shape features are suppressed (`compute_all(skip_shape=True)`) because the merged mask does not represent a single anatomical structure. Per-breast analysis uses the full feature set including shape descriptors.

### 3.5 Multi-Planar Rendering and Figure Export

The MPR canvas renders axial, coronal, and sagittal views simultaneously. Anatomically correct aspect ratios are maintained: coronal views use `aspect = sz / sx`, sagittal views use `aspect = sz / sy`, where sx, sy, sz are the physical voxel dimensions.

Overlay rendering uses a layered compositing order designed to preserve visibility of all tissue classes:
1. Tumour exclusion zone (α = 0.55) — rendered first so subsequent layers are visible on top
2. Skin layer
3. Pectoralis layer
4. Fibroglandular tissue layer

A safety check blocks tumour exclusion export if the exclusion ROI exceeds 30% of any breast volume, preventing inadvertent exclusion of the majority of the breast parenchyma.

Export produces:
- A publication-ready PNG figure (all three MPR planes with overlays and colourbar)
- A structured CSV file containing all metabolic metrics and all radiomics features for each breast / bilateral ROI

---

## 4. Validation

*Note: The following subsections describe the validation experiments to be performed. Quantitative results (N, ICC values, Dice scores) will be populated from prospective data collection.*

### 4.1 Metabolic Metrics

SUV metrics computed by the workstation will be compared against values obtained from a validated commercial platform (GE AW Workstation / Siemens syngo.via) on a cohort of N = [TBD] clinical FDG PET/CT examinations. Statistical analysis will include Pearson correlation coefficient, Bland-Altman limits of agreement, and intra-class correlation coefficient (ICC, two-way mixed, absolute agreement).

### 4.2 Breast Segmentation

Automated breast and fibroglandular segmentation will be evaluated against manual expert contours (N = [TBD] patients). Evaluation metrics:
- Dice similarity coefficient (DSC) for whole-breast and fibroglandular masks
- Volumetric overlap error
- Mean surface distance

Results will be stratified by ACR BI-RADS breast density category (A–D) to assess performance across the full range of breast compositions.

### 4.3 Radiomics Reproducibility

Feature reproducibility will be assessed using test-retest scans (N = [TBD]) or the IBSI digital phantom reference dataset. Intra-class correlation coefficients will be computed for all 50 features. Features with ICC ≥ 0.75 will be designated as reproducible for downstream biomarker analysis.

---

## 5. Usage Workflow

### 5.1 Installation

```bash
# Clone repository
git clone https://github.com/nittinnm22-sudo/breast-density-radiomics.git
cd breast-density-radiomics

# Install dependencies
pip install -r requirements.txt

# Launch workstation
python petct_viewer_v47.py
```

TotalSegmentator is an optional dependency. If installed (`pip install TotalSegmentator`), anatomy-constrained breast segmentation is automatically enabled.

### 5.2 Step-by-Step Workflow

1. **Load DICOM**: Click *File → Open DICOM folder*. The workstation automatically discovers and pairs CT and PET series by StudyInstanceUID.
2. **Review series**: Axial, coronal, and sagittal views are displayed with SUV windowing. PET/CT fusion opacity is adjustable.
3. **Define VOI**: Draw a bounding box over the lesion of interest in any MPR plane, or use the auto-threshold tool (ISO 50% or SUV threshold).
4. **Compute PET metrics**: Click *Compute Metabolic Metrics*. Results populate the metrics panel.
5. **Run breast segmentation**: Click *Segment Breast (L)*, *Segment Breast (R)*, or *Bilateral*. Segmentation overlays appear on the MPR views.
6. **Compute radiomics**: Click *Compute Radiomics*. Progress bar indicates feature extraction status.
7. **Export**: Click *Export Figure + CSV*. A PNG figure and CSV file are saved to the output directory.

---

## 6. Comparison with Existing Tools

| Feature | This work | LIFEx | PETSurfer | PyRadiomics (standalone) |
|---|---|---|---|---|
| PET SUV metrics (SUVmax, SUVmean, SUVpeak, MTV, TLG) | ✓ | ✓ | ✓ | ✗ |
| Automated breast segmentation | ✓ | ✗ | ✗ | ✗ |
| Fibroglandular tissue quantification | ✓ | ✗ | ✗ | ✗ |
| IBSI-compliant CT radiomics | ✓ | Partial | ✗ | ✓ |
| Open source (full code access) | ✓ | ✗ | ✓ | ✓ |
| Interactive GUI | ✓ | ✓ | ✗ | ✗ |
| Single-script deployable | ✓ | ✗ | ✗ | ✗ |
| No server / database required | ✓ | ✓ | ✓ | ✓ |

---

## 7. Limitations and Future Work

**Current limitations:**
- Single-workstation application; no PACS push or DICOM SR output
- Deep-learning breast segmentation (e.g., nnU-Net) is not yet integrated; TotalSegmentator anatomy priors are optional
- Validation on prospective multi-center datasets has not yet been performed
- No automated multi-lesion delineation or total metabolic tumour volume (TMTV) computation

**Planned developments:**
- Docker container for zero-dependency deployment
- nnU-Net-based breast segmentation model fine-tuned on CT data
- Multi-lesion MTV / TMTV pipeline for lymphoma response assessment
- DICOM SR and DICOM SEG output for PACS integration
- Prospective validation study across ≥3 PET/CT scanner platforms

---

## 8. Conclusion

We present an open-source Python workstation that, for the first time, integrates automated PET metabolic quantification, bilateral breast segmentation with fibroglandular tissue analysis, and IBSI-compliant CT radiomics within a single, freely deployable GUI. The software is designed for research cohort studies and clinical trials requiring reproducible, auditable, and cost-free imaging biomarker extraction. IBSI compliance and full algorithmic transparency support reproducibility across sites and time points. Future work will focus on deep-learning segmentation integration, TMTV computation, and prospective multi-center validation.

---

## References

1. Zwanenburg A, Vallières M, Abdalah MA, et al. The Image Biomarker Standardization Initiative: Standardized Quantitative Radiomics for High-Throughput Image-based Phenotyping. *Radiology*. 2020;295(2):328–338. doi:10.1148/radiol.2020191145

2. Boellaard R, Delgado-Bolton R, Oyen WJG, et al. FDG PET/CT: EANM procedure guidelines for tumour imaging: version 2.0. *Eur J Nucl Med Mol Imaging*. 2015;42(2):328–354. doi:10.1007/s00259-014-2961-x

3. Wasserthal J, Breit HC, Meyer MT, et al. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. *Radiology: Artificial Intelligence*. 2023;5(5):e230024. doi:10.1148/ryai.230024

4. van Griethuysen JJM, Fedorov A, Parmar C, et al. Computational Radiomics System to Decode the Radiographic Phenotype. *Cancer Research*. 2017;77(21):e104–e107. doi:10.1158/0008-5472.CAN-17-0339

5. American College of Radiology. *ACR BI-RADS Atlas: Breast Imaging Reporting and Data System*. 5th ed. Reston, VA: ACR; 2013.

6. [Additional references to be added during manuscript finalisation]

---

## Supplementary Materials

### S1. Software Availability

- **Repository**: https://github.com/nittinnm22-sudo/breast-density-radiomics
- **Primary script**: `petct_viewer_v47.py`
- **License**: [MIT / open-source — to be confirmed]
- **Python version**: ≥ 3.8
- **Dependencies**: see `requirements.txt`

### S2. IBSI Radiomics Configuration

The complete pyradiomics YAML configuration used for feature extraction:

```yaml
setting:
  binWidth: 25
  voxelArrayShift: 0
  resampledPixelSpacing: [1.2, 1.2, 1.2]
  interpolator: sitkBSpline
  label: 1

featureClass:
  firstorder: []
  glcm: []
  glrlm: []
  glszm: []
  ngtdm: []
  shape: []
```

### S3. 50-Feature Shortlist

| # | Feature Name | Family | IBSI Code |
|---|---|---|---|
| 1 | Energy | First-order | OJKH |
| 2 | Entropy | First-order | X8ZN |
| 3 | Kurtosis | First-order | IPH6 |
| 4 | Mean | First-order | Q4LE |
| 5 | Median | First-order | Y12H |
| 6 | Skewness | First-order | 88K1 |
| 7 | Variance | First-order | EIBY |
| 8 | 10th Percentile | First-order | GPMT |
| 9 | 90th Percentile | First-order | OZ0C |
| 10 | Uniformity | First-order | BJ5W |
| 11 | GLCM Autocorrelation | GLCM | QWB0 |
| 12 | GLCM Cluster Prominence | GLCM | AE86 |
| 13 | GLCM Cluster Shade | GLCM | 7NFM |
| 14 | GLCM Contrast | GLCM | ACUI |
| 15 | GLCM Correlation | GLCM | NI2N |
| 16 | GLCM Difference Average | GLCM | TF7R |
| 17 | GLCM Difference Entropy | GLCM | NTRS |
| 18 | GLCM Difference Variance | GLCM | D3YU |
| 19 | GLCM Energy | GLCM | 8ZQL |
| 20 | GLCM Entropy | GLCM | TU9B |
| 21 | GLCM Homogeneity | GLCM | SKES |
| 22 | GLCM Imc1 | GLCM | R8DG |
| 23 | GLCM Imc2 | GLCM | JN9H |
| 24 | GLCM Idmn | GLCM | 1QCO |
| 25 | GLCM Inverse Variance | GLCM | E8JP |
| 26 | GLCM Joint Average | GLCM | 60VM |
| 27 | GLCM MCC | GLCM | 9J26 |
| 28 | GLCM Maximum Probability | GLCM | GYBY |
| 29 | GLCM Sum Average | GLCM | ZGXS |
| 30 | GLCM Sum Entropy | GLCM | P6QZ |
| 31 | GLRLM Run Entropy | GLRLM | HJ9O |
| 32 | GLRLM Run Length Non-uniformity | GLRLM | R2L0 |
| 33 | GLRLM Run Length Non-uniformity Normalised | GLRLM | OJQ7 |
| 34 | GLRLM Run Percentage | GLRLM | 9ZK5 |
| 35 | GLRLM Short Run Emphasis | GLRLM | 22OV |
| 36 | GLRLM Long Run Emphasis | GLRLM | W4KF |
| 37 | GLRLM Short Run High Grey Level Emphasis | GLRLM | GD3A |
| 38 | GLRLM Long Run High Grey Level Emphasis | GLRLM | UNPE |
| 39 | GLSZM Zone Entropy | GLSZM | GU8N |
| 40 | GLSZM Zone Size Non-uniformity | GLSZM | 4JP3 |
| 41 | GLSZM Small Zone Emphasis | GLSZM | 5QRC |
| 42 | GLSZM Large Zone Emphasis | GLSZM | 48P8 |
| 43 | GLSZM Zone Percentage | GLSZM | P30P |
| 44 | NGTDM Busyness | NGTDM | NQ30 |
| 45 | NGTDM Coarseness | NGTDM | QCDE |
| 46 | NGTDM Complexity | NGTDM | HDEZ |
| 47 | NGTDM Contrast | NGTDM | 65HE |
| 48 | NGTDM Strength | NGTDM | 1X9X |
| 49 | Shape — Mesh Volume | Shape | RNU0 |
| 50 | Shape — Sphericity | Shape | QCFX |

*Note: Shape features (rows 49–50) are computed for per-breast analysis only; they are suppressed for the bilateral merged mask.*

### S4. Diaphragm Detection Algorithm (Pseudocode)

```
function detect_thorax_z_range(ct_array, voxel_size_z):
    # Compute fraction of voxels with HU > 0 per axial slice
    for each slice z:
        body_fraction[z] = count(HU[z] > 0) / slice_area

    # Self-calibrating threshold
    peak_frac = max(body_fraction)
    threshold = max(0.05, peak_frac * 0.20)

    # Descend from superior end to find diaphragm
    z_diaphragm = None
    for z in range(top_25_percent, bottom):
        if body_fraction[z] < threshold:
            # Look ahead 3 slices to confirm
            if all(body_fraction[z+1:z+4] < threshold):
                z_diaphragm = z
                break

    # Apply 75% hard cap
    z_diaphragm = min(z_diaphragm, 0.75 * n_slices)

    return z_diaphragm
```

### S5. Fibroglandular Segmentation Algorithm (Pseudocode)

```
function segment_fibroglandular_two_stage(ct_array, breast_mask):
    # Stage A: HU threshold
    fg_candidates = breast_mask AND (ct_array >= -25) AND (ct_array <= 100)

    # Stage B: exclude non-parenchymal tissue
    # 4-voxel skin erosion
    fg_no_skin = morphological_erosion(fg_candidates, radius=4)

    # Posterior 40% pectoralis exclusion
    posterior_boundary = compute_posterior_boundary(breast_mask)
    pectoralis_zone = y > (posterior_boundary * 0.60)
    fg_no_pectoralis = fg_no_skin AND NOT pectoralis_zone

    # Remove surgical clips (small high-HU components)
    fg_no_clips = remove_high_hu_components(fg_no_pectoralis)

    # Minimum component size filter
    fg_final = remove_small_components(fg_no_clips, min_size=100)

    # QC flags
    if fg_overlap_skin_zone(fg_final, breast_mask) > 0.15:
        warn("FG > 15% skin zone — possible skin spillover")
    if fg_overlap_pectoralis_zone(fg_final, breast_mask) > 0.20:
        warn("FG > 20% pectoralis zone — possible muscle contamination")

    return fg_final
```
