"""Unified PET/CT breast density and parenchymal complexity workstation.

This module provides a research-grade breast CT analysis workflow for clinical
manuscript preparation. It includes whole-breast and fibroglandular
segmentation, volumetric density quantification, exclusion of non-parenchymal
regions, and computation of shape/intensity/texture complexity radiomics.

Usage:
    - Import `BreastCTDensityEngine` and `ParenchymalComplexityEngine` for
      programmatic analysis.
    - Launch `BreastDensityDialog` for interactive review and export.
    - Run this script directly to execute density and complexity ML analyses.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.stats import entropy as scipy_entropy
from scipy.stats import kurtosis, skew
from skimage.measure import marching_cubes, mesh_surface_area
from skimage.morphology import binary_closing, binary_opening, remove_small_objects
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ═══ SECTION 0 — Constants ═══
HU_FAT_MAX = -25
HU_BREAST_MIN = -300
HU_BREAST_MAX = 200
HU_MUSCLE_MIN = 40
HU_CLIP_MIN = 400
VOXEL_BINS = 32

MANUSCRIPT_SHORTLIST_FEATURES = [
    "surface_to_volume_ratio",
    "sphericity",
    "elongation",
    "flatness",
    "std_hu",
    "skewness",
    "kurtosis",
    "entropy",
    "uniformity",
    "glcm_contrast",
    "glcm_correlation",
    "glcm_joint_entropy",
    "glrlm_short_run_emphasis",
    "glrlm_long_run_emphasis",
    "glrlm_run_entropy",
    "glszm_zone_entropy",
    "glszm_small_area_emphasis",
    "glszm_large_area_emphasis",
    "gldm_dependence_non_uniformity",
    "ngtdm_coarseness",
    "ngtdm_busyness",
]

try:
    import SimpleITK as sitk
    from radiomics import featureextractor

    PYRADIOMICS_AVAILABLE = True
except Exception:
    PYRADIOMICS_AVAILABLE = False


def _safe_div(numerator: float, denominator: float) -> float:
    """Safely divide two values and return 0 for zero denominator."""
    return float(numerator / denominator) if denominator else 0.0


def _voxel_volume_cc(voxel_spacing_mm: Tuple[float, float, float]) -> float:
    """Return single-voxel volume in cubic centimeters."""
    return float(np.prod(voxel_spacing_mm) / 1000.0)


def _prepare_quantized_volume(ct_volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Clip HU and quantize masked voxels into VOXEL_BINS bins."""
    clipped = np.clip(ct_volume, HU_BREAST_MIN, HU_BREAST_MAX)
    normalized = (clipped - HU_BREAST_MIN) / (HU_BREAST_MAX - HU_BREAST_MIN + 1e-8)
    quantized = np.floor(normalized * (VOXEL_BINS - 1)).astype(np.int32)
    quantized[~mask] = 0
    return quantized


# ═══ SECTION 1 — Segmentation and Density ═══
class BreastSegmentor:
    """Performs whole-breast and fibroglandular segmentation on CT volumes."""

    def segment_whole_breast(self, ct_volume: np.ndarray) -> Dict[str, np.ndarray]:
        """Segment right and left whole-breast masks using HU thresholding and morphology."""
        breast = (ct_volume >= HU_BREAST_MIN) & (ct_volume <= HU_BREAST_MAX)
        breast = binary_closing(breast, footprint=np.ones((3, 3, 3)))
        breast = remove_small_objects(breast, min_size=2_000)
        midline = ct_volume.shape[2] // 2
        right_mask = np.zeros_like(breast, dtype=bool)
        left_mask = np.zeros_like(breast, dtype=bool)
        right_mask[:, :, :midline] = breast[:, :, :midline]
        left_mask[:, :, midline:] = breast[:, :, midline:]
        right_mask = remove_small_objects(right_mask, min_size=1_000)
        left_mask = remove_small_objects(left_mask, min_size=1_000)
        return {"right_mask": right_mask, "left_mask": left_mask}

    def segment_fibroglandular(self, ct_volume: np.ndarray, breast_mask: np.ndarray) -> np.ndarray:
        """Segment fibroglandular tissue within a provided whole-breast mask."""
        fg = (ct_volume > HU_FAT_MAX) & breast_mask
        fg = binary_opening(fg, footprint=np.ones((3, 3, 3)))
        fg = binary_closing(fg, footprint=np.ones((3, 3, 3)))
        return remove_small_objects(fg, min_size=200)

    def exclude_non_parenchymal(self, fg_mask: np.ndarray, ct_volume: np.ndarray) -> np.ndarray:
        """Exclude skin, pectoralis, clips, and tumor-like components from fibroglandular mask."""
        cleaned = fg_mask.copy()
        if not cleaned.any():
            print("[Segmentor] excluded 0 voxels: skin shell")
            print("[Segmentor] excluded 0 voxels: pectoralis muscle")
            print("[Segmentor] excluded 0 voxels: clips/markers")
            print("[Segmentor] excluded 0 voxels: primary tumor placeholder")
            return cleaned

        breast_mask = (ct_volume >= HU_BREAST_MIN) & (ct_volume <= HU_BREAST_MAX)
        breast_mask = binary_closing(breast_mask, footprint=np.ones((3, 3, 3)))
        inner = ndi.binary_erosion(breast_mask, iterations=2)
        skin_shell = breast_mask & ~inner
        removed = int(np.count_nonzero(cleaned & skin_shell))
        cleaned &= ~skin_shell
        print(f"[Segmentor] excluded {removed} voxels: skin shell")

        posterior_region = np.zeros_like(cleaned, dtype=bool)
        posterior_start = int(ct_volume.shape[1] * 0.66)
        posterior_region[:, posterior_start:, :] = True
        pectoralis = posterior_region & (ct_volume > HU_MUSCLE_MIN)
        removed = int(np.count_nonzero(cleaned & pectoralis))
        cleaned &= ~pectoralis
        print(f"[Segmentor] excluded {removed} voxels: pectoralis muscle")

        clips = (ct_volume > HU_CLIP_MIN) & cleaned
        labels, n_labels = ndi.label(clips)
        if n_labels:
            sizes = np.bincount(labels.ravel())           # sizes[i] = voxels in component i
            small_ids = np.nonzero(sizes[1:] <= 500)[0] + 1  # 1-based label IDs ≤500 voxels
            if small_ids.size:
                clip_mask = np.isin(labels, small_ids)
                removed = int(np.count_nonzero(cleaned & clip_mask))
                cleaned &= ~clip_mask
            else:
                removed = 0
        else:
            removed = 0
        print(f"[Segmentor] excluded {removed} voxels: clips/markers")

        tumor_candidates = cleaned & (ct_volume > 20)
        labels, n_labels = ndi.label(tumor_candidates)
        if n_labels:
            sizes = np.bincount(labels.ravel())
            largest_idx = int(np.argmax(sizes[1:]) + 1)
            largest_size = int(sizes[largest_idx])
            cleaned &= labels != largest_idx
            print(f"[Segmentor] excluded {largest_size} voxels: primary tumor placeholder (review advised)")
        else:
            print("[Segmentor] excluded 0 voxels: primary tumor placeholder")

        return remove_small_objects(cleaned, min_size=100)


class BreastCTDensityEngine:
    """Computes bilateral volumetric breast density metrics from CT."""

    def __init__(self) -> None:
        """Initialize density engine with a segmentor instance."""
        self.segmentor = BreastSegmentor()

    def compute_volumetric_density(
        self, ct_volume: np.ndarray, voxel_spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> Dict[str, float]:
        """Compute right/left/bilateral volumetric density and asymmetry metrics."""
        masks = self.segmentor.segment_whole_breast(ct_volume)
        voxel_cc = _voxel_volume_cc(voxel_spacing_mm)

        def _side_metrics(side: str) -> Dict[str, float]:
            whole = masks[f"{side}_mask"]
            fg = self.segmentor.segment_fibroglandular(ct_volume, whole)
            total_cc = float(np.count_nonzero(whole) * voxel_cc)
            fg_cc = float(np.count_nonzero(fg) * voxel_cc)
            fat_cc = max(total_cc - fg_cc, 0.0)
            density_pct = _safe_div(fg_cc * 100.0, total_cc)
            return {
                f"{side}_total_breast_volume_cc": total_cc,
                f"{side}_fibroglandular_volume_cc": fg_cc,
                f"{side}_fat_volume_cc": fat_cc,
                f"{side}_volumetric_density_pct": density_pct,
            }

        right = _side_metrics("right")
        left = _side_metrics("left")
        right_fg = right["right_fibroglandular_volume_cc"]
        left_fg = left["left_fibroglandular_volume_cc"]
        right_total = right["right_total_breast_volume_cc"]
        left_total = left["left_total_breast_volume_cc"]

        combined_fg = right_fg + left_fg
        combined_total = right_total + left_total
        bilateral_density = _safe_div(combined_fg * 100.0, combined_total)

        return {
            **right,
            **left,
            "bilateral_density_pct": bilateral_density,
            "density_asymmetry_pct": abs(
                right["right_volumetric_density_pct"] - left["left_volumetric_density_pct"]
            ),
            "fibroglandular_volume_asymmetry_cc": abs(right_fg - left_fg),
            "absolute_fibroglandular_volume_cc": combined_fg,
        }


# ═══ SECTION 2 — Complexity Feature Engines ═══
class ShapeFeatures:
    """Shape and morphology feature extraction from parenchymal mask."""

    def compute(self, parenchymal_mask: np.ndarray, voxel_spacing_mm: Tuple[float, float, float]) -> Dict[str, float]:
        """Compute volume, area, compactness, PCA axes, and geometric ratios."""
        if not np.any(parenchymal_mask):
            return {k: float("nan") for k in [
                "parenchymal_volume_cc", "surface_area_mm2", "surface_to_volume_ratio", "sphericity",
                "elongation", "flatness", "max_3d_diameter_mm", "major_axis_length_mm", "minor_axis_length_mm"
            ]}

        voxel_mm3 = float(np.prod(voxel_spacing_mm))
        volume_mm3 = float(np.count_nonzero(parenchymal_mask) * voxel_mm3)
        volume_cc = volume_mm3 / 1000.0

        verts, faces, _, _ = marching_cubes(parenchymal_mask.astype(np.float32), level=0.5, spacing=voxel_spacing_mm)
        surface_area = float(mesh_surface_area(verts, faces))

        coords = np.argwhere(parenchymal_mask).astype(np.float64)
        coords *= np.asarray(voxel_spacing_mm)
        cov = np.cov(coords, rowvar=False)
        eigvals, _ = np.linalg.eig(cov)
        eigvals = np.sort(np.real(eigvals))[::-1]
        major, intermediate, minor = [max(float(v), 1e-8) for v in eigvals]

        major_axis = float(4.0 * np.sqrt(major))
        minor_axis = float(4.0 * np.sqrt(minor))

        bbox_mm = (coords.max(axis=0) - coords.min(axis=0))
        max_diameter = float(np.linalg.norm(bbox_mm))

        surface_to_volume = _safe_div(surface_area, volume_mm3)
        sphericity = _safe_div((np.pi ** (1.0 / 3.0)) * ((6.0 * volume_mm3) ** (2.0 / 3.0)), surface_area)

        return {
            "parenchymal_volume_cc": volume_cc,
            "surface_area_mm2": surface_area,
            "surface_to_volume_ratio": surface_to_volume,
            "sphericity": sphericity,
            "elongation": float(np.sqrt(_safe_div(minor, intermediate))),
            "flatness": float(np.sqrt(_safe_div(minor, major))),
            "max_3d_diameter_mm": max_diameter,
            "major_axis_length_mm": major_axis,
            "minor_axis_length_mm": minor_axis,
        }


class FirstOrderFeatures:
    """First-order HU statistics from parenchymal ROI."""

    def compute(self, ct_volume: np.ndarray, parenchymal_mask: np.ndarray) -> Dict[str, float]:
        """Compute descriptive HU moments and histogram-derived entropy/uniformity."""
        values = ct_volume[parenchymal_mask]
        if values.size == 0:
            return {k: float("nan") for k in [
                "mean_hu", "median_hu", "std_hu", "percentile_10_hu", "percentile_90_hu", "iqr_hu",
                "skewness", "kurtosis", "entropy", "uniformity"
            ]}

        hist, _ = np.histogram(values, bins=256, density=False)
        probs = hist.astype(np.float64)
        probs /= max(probs.sum(), 1.0)

        return {
            "mean_hu": float(np.mean(values)),
            "median_hu": float(np.median(values)),
            "std_hu": float(np.std(values)),
            "percentile_10_hu": float(np.percentile(values, 10)),
            "percentile_90_hu": float(np.percentile(values, 90)),
            "iqr_hu": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "skewness": float(skew(values)) if values.size > 2 else float("nan"),
            "kurtosis": float(kurtosis(values)) if values.size > 3 else float("nan"),
            "entropy": float(scipy_entropy(probs + 1e-12, base=2)),
            "uniformity": float(np.sum(probs ** 2)),
        }


class _TextureFeatureBase:
    """Shared utilities for texture feature extractors."""

    @staticmethod
    def _pyradiomics_features(ct_volume: np.ndarray, mask: np.ndarray, feature_class: str, names: Dict[str, str]) -> Dict[str, float]:
        if not PYRADIOMICS_AVAILABLE:
            return {}
        try:
            image = sitk.GetImageFromArray(ct_volume.astype(np.float32))
            image_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
            extractor = featureextractor.RadiomicsFeatureExtractor(binCount=VOXEL_BINS)
            extractor.disableAllFeatures()
            extractor.enableFeatureClassByName(feature_class)
            result = extractor.execute(image, image_mask)
            output = {}
            for out_name, rad_name in names.items():
                key = f"original_{feature_class}_{rad_name}"
                output[out_name] = float(result.get(key, float("nan")))
            return output
        except Exception as exc:
            print(f"[Texture] pyradiomics backend unavailable for {feature_class}: {exc}")
            return {}


class GLCMFeatures(_TextureFeatureBase):
    """3D isotropic GLCM feature extraction."""

    def compute(self, ct_volume: np.ndarray, parenchymal_mask: np.ndarray) -> Dict[str, float]:
        """Compute contrast, correlation, joint entropy, and inverse difference."""
        pyr = self._pyradiomics_features(
            ct_volume,
            parenchymal_mask,
            "glcm",
            {
                "glcm_contrast": "Contrast",
                "glcm_correlation": "Correlation",
                "glcm_joint_entropy": "JointEntropy",
                "glcm_inverse_difference": "Id",
            },
        )
        if pyr:
            return pyr

        q = _prepare_quantized_volume(ct_volume, parenchymal_mask)
        glcm = np.zeros((VOXEL_BINS, VOXEL_BINS), dtype=np.float64)
        directions = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1),
            (0, 1, 1), (0, 1, -1), (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        ]
        for dz, dy, dx in directions:
            shifted_q = np.roll(q, shift=(dz, dy, dx), axis=(0, 1, 2))
            shifted_mask = np.roll(parenchymal_mask, shift=(dz, dy, dx), axis=(0, 1, 2))
            valid = parenchymal_mask & shifted_mask
            i_vals = q[valid]
            j_vals = shifted_q[valid]
            np.add.at(glcm, (i_vals, j_vals), 1)

        glcm = glcm + glcm.T
        p = glcm / max(glcm.sum(), 1.0)
        i = np.arange(VOXEL_BINS, dtype=np.float64)
        j = np.arange(VOXEL_BINS, dtype=np.float64)
        ii, jj = np.meshgrid(i, j, indexing="ij")

        mu_i = float(np.sum(ii * p))
        mu_j = float(np.sum(jj * p))
        sig_i = float(np.sqrt(np.sum(((ii - mu_i) ** 2) * p)))
        sig_j = float(np.sqrt(np.sum(((jj - mu_j) ** 2) * p)))
        correlation = _safe_div(np.sum((ii - mu_i) * (jj - mu_j) * p), sig_i * sig_j)

        return {
            "glcm_contrast": float(np.sum(((ii - jj) ** 2) * p)),
            "glcm_correlation": float(correlation),
            "glcm_joint_entropy": float(-np.sum(p * np.log2(p + 1e-12))),
            "glcm_inverse_difference": float(np.sum(p / (1.0 + np.abs(ii - jj)))),
        }


class GLRLMFeatures(_TextureFeatureBase):
    """Lightweight GLRLM-style run-length features."""

    def compute(self, ct_volume: np.ndarray, parenchymal_mask: np.ndarray) -> Dict[str, float]:
        """Compute short/long run emphasis, run entropy, and gray-level non-uniformity."""
        pyr = self._pyradiomics_features(
            ct_volume,
            parenchymal_mask,
            "glrlm",
            {
                "glrlm_short_run_emphasis": "ShortRunEmphasis",
                "glrlm_long_run_emphasis": "LongRunEmphasis",
                "glrlm_run_entropy": "RunEntropy",
                "glrlm_gray_level_non_uniformity": "GrayLevelNonUniformity",
            },
        )
        if pyr:
            return pyr

        q = _prepare_quantized_volume(ct_volume, parenchymal_mask)
        # Vectorised run-length encoding along the x-axis (axis 2).
        n_g = VOXEL_BINS
        max_run = max(ct_volume.shape)
        rlm = np.zeros((n_g, max_run + 1), dtype=np.float64)
        Z, Y, X = q.shape
        q_2d = q.reshape(-1, X)
        m_2d = parenchymal_mask.reshape(-1, X)
        prev_m = np.zeros_like(m_2d)
        prev_m[:, 1:] = m_2d[:, :-1]
        prev_q = np.zeros_like(q_2d)
        prev_q[:, 1:] = q_2d[:, :-1]
        is_start = m_2d & (~prev_m | (q_2d != prev_q))
        run_id = np.cumsum(is_start.ravel()).reshape(Z * Y, X)
        run_id = np.where(m_2d, run_id, 0)
        max_id = int(run_id.max())
        if max_id == 0:
            return {
                "glrlm_short_run_emphasis": float("nan"),
                "glrlm_long_run_emphasis": float("nan"),
                "glrlm_run_entropy": float("nan"),
                "glrlm_gray_level_non_uniformity": float("nan"),
            }
        rids_flat = run_id.ravel()
        grays_flat = q_2d.ravel()
        mask_flat = m_2d.ravel()
        masked_rids = rids_flat[mask_flat]
        masked_grays = grays_flat[mask_flat]
        rl = np.bincount(masked_rids, minlength=max_id + 1)[1:]
        gl = np.zeros(max_id + 1, dtype=q_2d.dtype)
        np.maximum.at(gl, masked_rids, masked_grays)
        gl = gl[1:]
        valid = (gl >= 0) & (gl < n_g) & (rl >= 1) & (rl <= max_run)
        np.add.at(rlm, (gl[valid], rl[valid]), 1)
        p = rlm / max(rlm.sum(), 1.0)
        run_lengths = np.arange(p.shape[1], dtype=np.float64)

        sre = float(np.sum(p[:, 1:] / (run_lengths[1:] ** 2)))
        lre = float(np.sum(p[:, 1:] * (run_lengths[1:] ** 2)))
        run_entropy = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
        glnu = float(np.sum(np.sum(rlm, axis=1) ** 2) / max(rlm.sum(), 1.0))
        return {
            "glrlm_short_run_emphasis": sre,
            "glrlm_long_run_emphasis": lre,
            "glrlm_run_entropy": run_entropy,
            "glrlm_gray_level_non_uniformity": glnu,
        }


class GLSZMFeatures(_TextureFeatureBase):
    """Lightweight GLSZM-style zone-size features."""

    def compute(self, ct_volume: np.ndarray, parenchymal_mask: np.ndarray) -> Dict[str, float]:
        """Compute small/large area emphasis, zone entropy, and gray-level non-uniformity."""
        pyr = self._pyradiomics_features(
            ct_volume,
            parenchymal_mask,
            "glszm",
            {
                "glszm_small_area_emphasis": "SmallAreaEmphasis",
                "glszm_large_area_emphasis": "LargeAreaEmphasis",
                "glszm_zone_entropy": "ZoneEntropy",
                "glszm_gray_level_non_uniformity": "GrayLevelNonUniformity",
            },
        )
        if pyr:
            return pyr

        q = _prepare_quantized_volume(ct_volume, parenchymal_mask)
        structure = np.ones((3, 3, 3), dtype=np.uint8)
        # Vectorised: for each gray level, label connected components then get sizes
        # via bincount — avoids the O(n_labels × volume_size) loop per component.
        zones_g: list = []
        zones_s: list = []
        for g in range(VOXEL_BINS):
            level_mask = parenchymal_mask & (q == g)
            labels, n_labels = ndi.label(level_mask, structure=structure)
            if n_labels:
                sizes = np.bincount(labels.ravel())[1:]   # voxel count per component
                zones_g.extend([g] * n_labels)
                zones_s.extend(sizes.tolist())

        if not zones_g:
            return {
                "glszm_small_area_emphasis": float("nan"),
                "glszm_large_area_emphasis": float("nan"),
                "glszm_zone_entropy": float("nan"),
                "glszm_gray_level_non_uniformity": float("nan"),
            }

        max_zone = int(max(zones_s))
        zones_g_arr = np.array(zones_g, dtype=np.int64)
        zones_s_arr = np.array(zones_s, dtype=np.int64)
        szm = np.zeros((VOXEL_BINS, max_zone + 1), dtype=np.float64)
        np.add.at(szm, (zones_g_arr, zones_s_arr), 1)

        p = szm / max(szm.sum(), 1.0)
        sizes = np.arange(p.shape[1], dtype=np.float64)
        sae = float(np.sum(p[:, 1:] / (sizes[1:] ** 2)))
        lae = float(np.sum(p[:, 1:] * (sizes[1:] ** 2)))
        zone_entropy = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
        glnu = float(np.sum(np.sum(szm, axis=1) ** 2) / max(szm.sum(), 1.0))
        return {
            "glszm_small_area_emphasis": sae,
            "glszm_large_area_emphasis": lae,
            "glszm_zone_entropy": zone_entropy,
            "glszm_gray_level_non_uniformity": glnu,
        }


class GLDMFeatures(_TextureFeatureBase):
    """Lightweight GLDM-style dependence features."""

    def compute(self, ct_volume: np.ndarray, parenchymal_mask: np.ndarray) -> Dict[str, float]:
        """Compute dependence emphases and non-uniformity features."""
        pyr = self._pyradiomics_features(
            ct_volume,
            parenchymal_mask,
            "gldm",
            {
                "gldm_small_dependence_emphasis": "SmallDependenceEmphasis",
                "gldm_large_dependence_emphasis": "LargeDependenceEmphasis",
                "gldm_dependence_non_uniformity": "DependenceNonUniformity",
                "gldm_gray_level_non_uniformity": "GrayLevelNonUniformity",
            },
        )
        if pyr:
            return pyr

        q = _prepare_quantized_volume(ct_volume, parenchymal_mask)
        # Vectorised: count same-gray (within ±1 bin) neighbours using 26-connected
        # shifted copies, avoiding the O(n_voxels × 26) Python loop.
        dep_count = np.zeros(q.shape, dtype=np.int32)
        all_offsets_3d = [
            (dz, dy, dx)
            for dz in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if not (dz == dy == dx == 0)
        ]
        q_int = q.astype(np.int32)
        for dz, dy, dx in all_offsets_3d:
            shifted_q = np.roll(np.roll(np.roll(q_int, -dz, axis=0), -dy, axis=1), -dx, axis=2)
            shifted_m = np.roll(np.roll(np.roll(parenchymal_mask, -dz, axis=0), -dy, axis=1), -dx, axis=2)
            # Clear wrapped boundaries for all three axes.
            if dz > 0:
                shifted_m[:dz] = False
            elif dz < 0:
                shifted_m[dz:] = False
            if dy > 0:
                shifted_m[:, :dy] = False
            elif dy < 0:
                shifted_m[:, dy:] = False
            if dx > 0:
                shifted_m[:, :, :dx] = False
            elif dx < 0:
                shifted_m[:, :, dx:] = False
            same_gray = parenchymal_mask & shifted_m & (np.abs(q_int - shifted_q) <= 1)
            dep_count += same_gray.astype(np.int32)

        if not parenchymal_mask.any():
            return {
                "gldm_small_dependence_emphasis": float("nan"),
                "gldm_large_dependence_emphasis": float("nan"),
                "gldm_dependence_non_uniformity": float("nan"),
                "gldm_gray_level_non_uniformity": float("nan"),
            }
        grays = q[parenchymal_mask]
        deps = dep_count[parenchymal_mask] + 1   # +1 matches original behaviour
        max_dep = int(deps.max())
        gldm = np.zeros((VOXEL_BINS, max_dep + 1), dtype=np.float64)
        valid = (grays >= 0) & (grays < VOXEL_BINS)
        np.add.at(gldm, (grays[valid], deps[valid]), 1)

        p = gldm / max(gldm.sum(), 1.0)
        deps = np.arange(p.shape[1], dtype=np.float64)
        sde = float(np.sum(p[:, 1:] / (deps[1:] ** 2)))
        lde = float(np.sum(p[:, 1:] * (deps[1:] ** 2)))
        dnu = float(np.sum(np.sum(gldm, axis=0) ** 2) / max(gldm.sum(), 1.0))
        glnu = float(np.sum(np.sum(gldm, axis=1) ** 2) / max(gldm.sum(), 1.0))
        return {
            "gldm_small_dependence_emphasis": sde,
            "gldm_large_dependence_emphasis": lde,
            "gldm_dependence_non_uniformity": dnu,
            "gldm_gray_level_non_uniformity": glnu,
        }


class NGTDMFeatures(_TextureFeatureBase):
    """Lightweight NGTDM-style neighborhood-tone features."""

    def compute(self, ct_volume: np.ndarray, parenchymal_mask: np.ndarray) -> Dict[str, float]:
        """Compute coarseness, contrast, and busyness on quantized parenchyma."""
        pyr = self._pyradiomics_features(
            ct_volume,
            parenchymal_mask,
            "ngtdm",
            {
                "ngtdm_coarseness": "Coarseness",
                "ngtdm_contrast": "Contrast",
                "ngtdm_busyness": "Busyness",
            },
        )
        if pyr:
            return pyr

        q = _prepare_quantized_volume(ct_volume, parenchymal_mask)
        s_i = np.zeros(VOXEL_BINS, dtype=np.float64)
        n_i = np.zeros(VOXEL_BINS, dtype=np.float64)

        # Vectorised: accumulate neighbour sums/counts with 26 shifted copies.
        all_offsets_3d = [
            (dz, dy, dx)
            for dz in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if not (dz == dy == dx == 0)
        ]
        neighbour_sum = np.zeros(q.shape, dtype=np.float64)
        neighbour_cnt = np.zeros(q.shape, dtype=np.float64)
        q_float = q.astype(np.float64)
        for dz, dy, dx in all_offsets_3d:
            shifted_q = np.roll(np.roll(np.roll(q_float, -dz, axis=0), -dy, axis=1), -dx, axis=2)
            shifted_m = np.roll(np.roll(np.roll(parenchymal_mask, -dz, axis=0), -dy, axis=1), -dx, axis=2)
            if dz > 0:
                shifted_m[:dz] = False
            elif dz < 0:
                shifted_m[dz:] = False
            if dy > 0:
                shifted_m[:, :dy] = False
            elif dy < 0:
                shifted_m[:, dy:] = False
            if dx > 0:
                shifted_m[:, :, :dx] = False
            elif dx < 0:
                shifted_m[:, :, dx:] = False
            valid_nb = parenchymal_mask & shifted_m
            neighbour_sum += np.where(valid_nb, shifted_q, 0.0)
            neighbour_cnt += valid_nb.astype(np.float64)
        has_nb = parenchymal_mask & (neighbour_cnt > 0)
        if has_nb.any():
            grays = q[has_nb]
            avg_nb = neighbour_sum[has_nb] / neighbour_cnt[has_nb]
            diffs = np.abs(grays.astype(np.float64) - avg_nb)
            valid = (grays >= 0) & (grays < VOXEL_BINS)
            np.add.at(n_i, grays[valid], 1.0)
            np.add.at(s_i, grays[valid], diffs[valid])

        total = float(np.sum(n_i))
        if total == 0:
            return {"ngtdm_coarseness": float("nan"), "ngtdm_contrast": float("nan"), "ngtdm_busyness": float("nan")}

        p_i = n_i / total
        coarseness = _safe_div(1.0, np.sum(p_i * s_i) + 1e-8)

        i = np.arange(VOXEL_BINS, dtype=np.float64)
        ii, jj = np.meshgrid(i, i, indexing="ij")
        pij = np.outer(p_i, p_i)
        contrast_num = np.sum(pij * ((ii - jj) ** 2))
        ng = max(float(np.count_nonzero(p_i)), 1.0)
        contrast = _safe_div(contrast_num, ng * (ng - 1.0 + 1e-8)) * _safe_div(np.sum(s_i), total)

        busy_den = np.sum(np.abs(ii * p_i[:, None] - jj * p_i[None, :]))
        busyness = _safe_div(np.sum(p_i * s_i), busy_den)

        return {
            "ngtdm_coarseness": float(coarseness),
            "ngtdm_contrast": float(contrast),
            "ngtdm_busyness": float(busyness),
        }


class ParenchymalComplexityEngine:
    """Orchestrates shape, first-order, and texture parenchymal complexity extraction."""

    def __init__(
        self,
        ct_volume: np.ndarray,
        parenchymal_mask: np.ndarray,
        voxel_spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """Initialize complexity engine with CT volume and cleaned parenchymal mask."""
        self.ct_volume = ct_volume
        self.parenchymal_mask = parenchymal_mask.astype(bool)
        self.voxel_spacing_mm = voxel_spacing_mm

    def compute_all(self) -> Dict[str, Dict[str, float] | float]:
        """Compute full complexity feature set and 21-feature manuscript shortlist."""
        shape_features = ShapeFeatures().compute(self.parenchymal_mask, self.voxel_spacing_mm)
        first_order = FirstOrderFeatures().compute(self.ct_volume, self.parenchymal_mask)
        glcm = GLCMFeatures().compute(self.ct_volume, self.parenchymal_mask)
        glrlm = GLRLMFeatures().compute(self.ct_volume, self.parenchymal_mask)
        glszm = GLSZMFeatures().compute(self.ct_volume, self.parenchymal_mask)
        gldm = GLDMFeatures().compute(self.ct_volume, self.parenchymal_mask)
        ngtdm = NGTDMFeatures().compute(self.ct_volume, self.parenchymal_mask)

        all_features: Dict[str, float] = {}
        for group in (shape_features, first_order, glcm, glrlm, glszm, gldm, ngtdm):
            all_features.update(group)

        shortlist = {k: float(all_features.get(k, float("nan"))) for k in MANUSCRIPT_SHORTLIST_FEATURES}
        all_features["manuscript_shortlist"] = shortlist
        return all_features


# ═══ SECTION 3 — GUI Workstation Dialog ═══
class BreastDensityDialog:
    """Tkinter dialog for density metrics, complexity features, and segmentation preview."""

    def __init__(
        self,
        master,
        ct_volume: np.ndarray,
        voxel_spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """Create dialog, run segmentation/feature engines, and render tabs."""
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self._ttk = ttk
        self.ct_volume = ct_volume
        self.window = tk.Toplevel(master)
        self.window.title("Breast CT Density + Parenchymal Complexity")
        self.window.geometry("1100x800")

        segmentor = BreastSegmentor()
        self.whole_masks = segmentor.segment_whole_breast(ct_volume)
        bilateral_fg = (
            segmentor.segment_fibroglandular(ct_volume, self.whole_masks["right_mask"]) |
            segmentor.segment_fibroglandular(ct_volume, self.whole_masks["left_mask"])
        )
        self.cleaned_parenchymal_mask = segmentor.exclude_non_parenchymal(bilateral_fg, ct_volume)

        self.density_results = BreastCTDensityEngine().compute_volumetric_density(ct_volume, voxel_spacing_mm)
        self.complexity_results = ParenchymalComplexityEngine(
            ct_volume,
            self.cleaned_parenchymal_mask,
            voxel_spacing_mm,
        ).compute_all()

        notebook = ttk.Notebook(self.window)
        tab_density = ttk.Frame(notebook)
        tab_complexity = ttk.Frame(notebook)
        tab_preview = ttk.Frame(notebook)
        notebook.add(tab_density, text="Density")
        notebook.add(tab_complexity, text="Complexity")
        notebook.add(tab_preview, text="Segmentation Preview")
        notebook.pack(fill="both", expand=True, padx=8, pady=8)

        self._build_kv_table(tab_density, self.density_results)
        self._build_kv_table(tab_complexity, self.complexity_results["manuscript_shortlist"])

        fig = self._build_segmentation_figure()
        canvas = FigureCanvasTkAgg(fig, master=tab_preview)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill="x", padx=8, pady=8)
        ttk.Button(button_frame, text="Export Full Report", command=self.export_full_report).pack(side="left", padx=4)
        ttk.Button(button_frame, text="Export Manuscript Panel", command=self.export_manuscript_panel).pack(side="left", padx=4)

    def _build_kv_table(self, parent, values: Dict[str, float]) -> None:
        """Render key/value table in the provided Tkinter parent frame."""
        columns = ("metric", "value")
        tree = self._ttk.Treeview(parent, columns=columns, show="headings")
        tree.heading("metric", text="Metric")
        tree.heading("value", text="Value")
        tree.column("metric", width=420, anchor="w")
        tree.column("value", width=220, anchor="e")
        for key, value in values.items():
            if key == "manuscript_shortlist":
                continue
            tree.insert("", "end", values=(key, f"{float(value):.6f}"))
        tree.pack(fill="both", expand=True, padx=8, pady=8)

    def _build_segmentation_figure(self):
        """Build matplotlib figure with axial CT slice and segmentation overlays."""
        mid = self.ct_volume.shape[0] // 2
        ct_slice = self.ct_volume[mid]
        whole = (self.whole_masks["right_mask"] | self.whole_masks["left_mask"])[mid]
        fg = self.cleaned_parenchymal_mask[mid]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(ct_slice, cmap="gray")
        whole_overlay = np.zeros((*whole.shape, 4), dtype=np.float32)
        whole_overlay[..., 1] = whole.astype(np.float32)
        whole_overlay[..., 3] = whole.astype(np.float32) * 0.25
        fg_overlay = np.zeros((*fg.shape, 4), dtype=np.float32)
        fg_overlay[..., 0] = fg.astype(np.float32)
        fg_overlay[..., 3] = fg.astype(np.float32) * 0.35
        ax.imshow(whole_overlay)
        ax.imshow(fg_overlay)
        ax.set_title("Axial Mid-Slice: Whole Breast (green) + Parenchyma (red)")
        ax.axis("off")
        fig.tight_layout()
        return fig

    def export_full_report(self) -> str:
        """Export all density and complexity features to a timestamped CSV file."""
        segmentation_valid = bool(self.cleaned_parenchymal_mask.any())
        payload = {
            "segmentation_valid": segmentation_valid,
            **self.density_results,
            **{k: v for k, v in self.complexity_results.items() if k != "manuscript_shortlist"},
        }
        output = f"breast_full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame([payload]).to_csv(output, index=False)
        print(f"[Export] Full report saved: {output}")
        return output

    def export_manuscript_panel(self) -> str:
        """Export only 21-feature manuscript shortlist to a timestamped CSV file."""
        segmentation_valid = bool(self.cleaned_parenchymal_mask.any())
        payload = {
            "segmentation_valid": segmentation_valid,
            **self.complexity_results["manuscript_shortlist"],
        }
        output = f"breast_manuscript_panel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame([payload]).to_csv(output, index=False)
        print(f"[Export] Manuscript panel saved: {output}")
        return output


# ═══ SECTION 4 — ML Analysis ═══
def _train_classifiers(df: pd.DataFrame, feature_columns: List[str], target_col: str):
    """Train 4 baseline classifiers and return AUCs with ROC coordinates."""
    X = df[feature_columns]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1_000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True),
    }
    auc_results = {}
    curves = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        scores = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_results[name] = float(auc(fpr, tpr))
        curves[name] = (fpr, tpr)
    return auc_results, curves


def run_breast_density_ml_analysis(csv_path: str = "breast_density_data.csv") -> Dict[str, float]:
    """Train baseline ML models on density-oriented tabular features and save ROC plot."""
    df = pd.read_csv(csv_path)
    if "diagnosis" not in df.columns:
        raise ValueError("CSV must include 'diagnosis' column.")
    numeric_cols = [c for c in df.columns if c != "diagnosis" and pd.api.types.is_numeric_dtype(df[c])]
    auc_results, curves = _train_classifiers(df.dropna(subset=["diagnosis"] + numeric_cols), numeric_cols, "diagnosis")
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr) in curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_results[name]:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Breast Density Features")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    plt.close()
    return auc_results


def run_complexity_ml_analysis(csv_path: str = "breast_density_data.csv") -> Dict[str, float]:
    """Train baseline ML models on manuscript-shortlist complexity features and save ROC plot."""
    df = pd.read_csv(csv_path)
    if "diagnosis" not in df.columns:
        raise ValueError("CSV must include 'diagnosis' column.")
    available_features = [col for col in MANUSCRIPT_SHORTLIST_FEATURES if col in df.columns]
    if not available_features:
        raise ValueError("CSV does not include manuscript shortlist columns for complexity ML analysis.")
    usable = df.dropna(subset=["diagnosis"] + available_features)
    auc_results, curves = _train_classifiers(usable, available_features, "diagnosis")
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr) in curves.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_results[name]:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Parenchymal Complexity Features")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve_complexity.png", dpi=150)
    plt.close()
    return auc_results


# ═══ SECTION 5 — Main Entry ═══
if __name__ == "__main__":
    """Run density and complexity ML analyses when executed as a script."""
    csv = "breast_density_data.csv"
    if os.path.exists(csv):
        print("Running breast density ML analysis...")
        print(run_breast_density_ml_analysis(csv))
        print("Running parenchymal complexity ML analysis...")
        print(run_complexity_ml_analysis(csv))
    else:
        print(f"Skipping ML analysis: {csv} not found.")
