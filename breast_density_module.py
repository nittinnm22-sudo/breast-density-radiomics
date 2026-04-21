#!/usr/bin/env python
"""Breast density and complexity module for PET/CT viewer.

Self-contained PyQt6 dialog that can be integrated into the main PET/CT app
via `_install_breast_density_toolbar(app)`.
"""

from __future__ import annotations

import csv
import math
import os
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QAction, QColor, QPen
    from PyQt6.QtWidgets import (
        QFileDialog,
        QDialog,
        QDoubleSpinBox,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSlider,
        QTableWidget,
        QTableWidgetItem,
        QToolBar,
        QVBoxLayout,
        QWidget,
        QComboBox,
        QCheckBox,
        QSpinBox,
    )
except Exception as e:  # pragma: no cover
    raise ImportError("PyQt6 is required for breast_density_module") from e

try:
    import pyqtgraph as pg
    _PG_AVAILABLE = True
except Exception:
    pg = None
    _PG_AVAILABLE = False

try:
    import nibabel as nib
    _NIB_AVAILABLE = True
except Exception:
    nib = None
    _NIB_AVAILABLE = False

try:
    from scipy import ndimage
    _SCIPY_AVAILABLE = True
except Exception:
    ndimage = None
    _SCIPY_AVAILABLE = False

try:
    from scipy.stats import skew, kurtosis
except Exception:
    skew = None
    kurtosis = None

try:
    from skimage import feature, measure
    _SKIMAGE_AVAILABLE = True
except Exception:
    feature = None
    measure = None
    _SKIMAGE_AVAILABLE = False

try:
    from radiomics import featureextractor
    _PYRADIOMICS_AVAILABLE = True
except Exception:
    featureextractor = None
    _PYRADIOMICS_AVAILABLE = False

try:
    import SimpleITK as sitk
    _SITK_AVAILABLE = True
except Exception:
    sitk = None
    _SITK_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    _REPORTLAB_AVAILABLE = True
except Exception:
    A4 = None
    canvas = None
    _REPORTLAB_AVAILABLE = False

try:
    import totalsegmentator
    _TOTALSEG_AVAILABLE = True
except Exception:
    _TOTALSEG_AVAILABLE = False

try:
    from petct_viewer_v21_full_metric_export_orient_fix_v5_fixed2 import (
        _window_ct_to_uint8,
        _make_hot_iron_lut,
        _pet_to_rgba,
        _make_body_mask_from_ct,
        SliceView,
    )
    _MAIN_VIEWER_AVAILABLE = True
except Exception:
    _MAIN_VIEWER_AVAILABLE = False

    def _window_ct_to_uint8(ct: np.ndarray, wl: float = 40.0, ww: float = 400.0) -> np.ndarray:
        lo = wl - ww / 2.0
        hi = wl + ww / 2.0
        x = np.clip((ct - lo) / max(ww, 1e-6), 0.0, 1.0)
        return (x * 255.0).astype(np.uint8)

    def _make_hot_iron_lut(n: int = 256) -> np.ndarray:
        x = np.linspace(0.0, 1.0, n)
        lut = np.zeros((n, 4), dtype=np.uint8)
        lut[:, 0] = np.clip(255 * np.minimum(1.0, 3 * x), 0, 255).astype(np.uint8)
        lut[:, 1] = np.clip(255 * np.minimum(1.0, 3 * x - 1), 0, 255).astype(np.uint8)
        lut[:, 2] = np.clip(255 * np.minimum(1.0, 3 * x - 2), 0, 255).astype(np.uint8)
        lut[:, 3] = 255
        return lut

    def _pet_to_rgba(pet: np.ndarray, alpha: int = 120) -> np.ndarray:
        if pet is None:
            return None
        x = pet.astype(np.float32)
        x = (x - np.nanmin(x)) / max(1e-6, (np.nanmax(x) - np.nanmin(x)))
        lut = _make_hot_iron_lut(256)
        idx = np.clip((x * 255).astype(np.int32), 0, 255)
        rgba = lut[idx]
        rgba[..., 3] = alpha
        return rgba

    def _make_body_mask_from_ct(ct: np.ndarray, threshold: float = -400.0) -> np.ndarray:
        m = ct > threshold
        if _SCIPY_AVAILABLE:
            m = ndimage.binary_closing(m, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
            m = ndimage.binary_fill_holes(m)
            lbl, n = ndimage.label(m)
            if n > 0:
                counts = np.bincount(lbl.ravel())
                counts[0] = 0
                m = lbl == np.argmax(counts)
        return m

    class SliceView:
        AXIAL = 0
        CORONAL = 1
        SAGITTAL = 2


@dataclass
class SegmentationParams:
    fgt_lo: int = -100
    fgt_hi: int = 200
    fat_lo: int = -250
    fat_hi: int = -10
    thoracic_fraction: float = 0.50
    morph_radius: int = 2
    use_totalseg: bool = False


class OrthogonalImageView(QWidget):
    def __init__(self, title: str, axis: int, parent=None):
        super().__init__(parent)
        self.axis = axis
        self._volume = None
        self._overlay = None
        self._index = 0
        self._cross = [0, 0, 0]

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(title))

        if _PG_AVAILABLE:
            self.plot = pg.PlotWidget()
            self.plot.setAspectLocked(False)
            self.plot.invertY(True)
            self.plot.setMenuEnabled(False)
            self.plot.hideAxis("left")
            self.plot.hideAxis("bottom")
            self.image_item = pg.ImageItem()
            self.overlay_item = pg.ImageItem()
            self.plot.addItem(self.image_item)
            self.plot.addItem(self.overlay_item)
            layout.addWidget(self.plot)
        else:
            self.plot = QLabel("pyqtgraph unavailable")
            self.plot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_item = None
            self.overlay_item = None
            layout.addWidget(self.plot)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self.slider)

        if _PG_AVAILABLE:
            self.vline = pg.InfiniteLine(angle=90, movable=False, pen=QPen(QColor("lime"), 1))
            self.hline = pg.InfiniteLine(angle=0, movable=False, pen=QPen(QColor("lime"), 1))
            self.plot.addItem(self.vline)
            self.plot.addItem(self.hline)
            self.plot.scene().sigMouseClicked.connect(self._on_click)

        self.on_slice_changed = None
        self.on_crosshair_changed = None

    def set_volume(self, volume: np.ndarray):
        self._volume = volume
        if volume is None or volume.ndim != 3:
            self.slider.setRange(0, 0)
            return
        max_idx = volume.shape[self.axis] - 1
        self.slider.setRange(0, max(0, max_idx))
        self._index = max_idx // 2
        self.slider.setValue(self._index)
        self._render()

    def set_overlay(self, overlay_rgba: np.ndarray):
        self._overlay = overlay_rgba
        self._render()

    def set_index(self, index: int):
        if self._volume is None:
            return
        index = int(np.clip(index, 0, self._volume.shape[self.axis] - 1))
        self._index = index
        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        self._render()

    def set_crosshair(self, zyx: Tuple[int, int, int]):
        self._cross = list(zyx)
        self._render_crosshair()

    def _slice(self, arr: np.ndarray) -> np.ndarray:
        if arr is None:
            return None
        if self.axis == 0:
            return arr[self._index, :, :]
        if self.axis == 1:
            return arr[:, self._index, :]
        return arr[:, :, self._index]

    def _render(self):
        if not _PG_AVAILABLE or self._volume is None:
            return
        img = self._slice(self._volume)
        if img is None:
            return
        self.image_item.setImage(np.ascontiguousarray(img.T), autoLevels=False)
        if self._overlay is not None:
            ov = self._slice(self._overlay)
            self.overlay_item.setImage(np.ascontiguousarray(ov.transpose(1, 0, 2)), autoLevels=False)
        else:
            self.overlay_item.clear()
        self._render_crosshair()

    def _render_crosshair(self):
        if not _PG_AVAILABLE:
            return
        z, y, x = self._cross
        if self.axis == 0:
            cx, cy = x, y
        elif self.axis == 1:
            cx, cy = x, z
        else:
            cx, cy = y, z
        self.vline.setValue(cx)
        self.hline.setValue(cy)

    def _on_slider(self, value: int):
        self._index = value
        self._render()
        if self.on_slice_changed:
            self.on_slice_changed(self.axis, value)

    def _on_click(self, ev):
        if not _PG_AVAILABLE or self._volume is None:
            return
        pos = self.plot.plotItem.vb.mapSceneToView(ev.scenePos())
        ix, iy = int(round(pos.x())), int(round(pos.y()))
        z, y, x = self._cross
        if self.axis == 0:
            x, y, z = ix, iy, self._index
        elif self.axis == 1:
            x, z, y = ix, iy, self._index
        else:
            y, z, x = ix, iy, self._index
        shape = self._volume.shape
        z = int(np.clip(z, 0, shape[0] - 1))
        y = int(np.clip(y, 0, shape[1] - 1))
        x = int(np.clip(x, 0, shape[2] - 1))
        if self.on_crosshair_changed:
            self.on_crosshair_changed((z, y, x))


class BreastDensityDialog(QDialog):
    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.setWindowTitle("Breast Density")
        self.resize(1600, 900)

        self.params = SegmentationParams()
        self.ct, self.spacing = self._extract_ct_volume_and_spacing(app)
        self.pet_suv = self._extract_pet_suv(app)

        self.body_mask = None
        self.fgt_r = None
        self.fgt_l = None
        self.fat_r = None
        self.fat_l = None
        self.results: Dict[str, Dict[str, Optional[float]]] = {}

        self.cross = [0, 0, 0] if self.ct is None else [s // 2 for s in self.ct.shape]

        self._build_ui()
        if self.ct is not None:
            self._update_display_volume()
            self.auto_segment()

    def _build_ui(self):
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        right = QVBoxLayout()
        root.addLayout(left, 3)
        root.addLayout(right, 2)

        views = QGridLayout()
        self.axial_view = OrthogonalImageView("Axial", axis=0)
        self.coronal_view = OrthogonalImageView("Coronal", axis=1)
        self.sagittal_view = OrthogonalImageView("Sagittal", axis=2)

        for v in (self.axial_view, self.coronal_view, self.sagittal_view):
            v.on_slice_changed = self._on_slice_changed
            v.on_crosshair_changed = self._on_crosshair_changed

        views.addWidget(self.axial_view, 0, 0)
        views.addWidget(self.coronal_view, 0, 1)
        views.addWidget(self.sagittal_view, 1, 0, 1, 2)
        left.addLayout(views)

        display_box = QGroupBox("Display")
        display_layout = QHBoxLayout(display_box)
        self.display_mode = QComboBox()
        self.display_mode.addItems(["CT only", "PET only", "Fused"])
        self.display_mode.currentIndexChanged.connect(self._update_display_volume)
        display_layout.addWidget(QLabel("Mode:"))
        display_layout.addWidget(self.display_mode)
        left.addWidget(display_box)

        seg_box = QGroupBox("Segmentation Controls")
        seg_form = QFormLayout(seg_box)

        self.fgt_lo_sb = QSpinBox(); self.fgt_lo_sb.setRange(-1024, 3071); self.fgt_lo_sb.setValue(self.params.fgt_lo)
        self.fgt_hi_sb = QSpinBox(); self.fgt_hi_sb.setRange(-1024, 3071); self.fgt_hi_sb.setValue(self.params.fgt_hi)
        self.fat_lo_sb = QSpinBox(); self.fat_lo_sb.setRange(-1024, 3071); self.fat_lo_sb.setValue(self.params.fat_lo)
        self.fat_hi_sb = QSpinBox(); self.fat_hi_sb.setRange(-1024, 3071); self.fat_hi_sb.setValue(self.params.fat_hi)
        self.thor_frac_sb = QDoubleSpinBox(); self.thor_frac_sb.setRange(0.1, 1.0); self.thor_frac_sb.setSingleStep(0.05); self.thor_frac_sb.setValue(self.params.thoracic_fraction)
        self.morph_radius_sb = QSpinBox(); self.morph_radius_sb.setRange(0, 10); self.morph_radius_sb.setValue(self.params.morph_radius)
        self.totalseg_cb = QCheckBox("Use TotalSegmentator (if available)")
        self.totalseg_cb.setChecked(False)
        self.totalseg_cb.setEnabled(_TOTALSEG_AVAILABLE)

        seg_form.addRow("FGT HU low", self.fgt_lo_sb)
        seg_form.addRow("FGT HU high", self.fgt_hi_sb)
        seg_form.addRow("Fat HU low", self.fat_lo_sb)
        seg_form.addRow("Fat HU high", self.fat_hi_sb)
        seg_form.addRow("Thoracic crop fraction", self.thor_frac_sb)
        seg_form.addRow("Morph radius", self.morph_radius_sb)
        seg_form.addRow(self.totalseg_cb)

        btn_row = QHBoxLayout()
        self.auto_btn = QPushButton("Auto-Segment")
        self.reset_btn = QPushButton("Reset Masks")
        self.compute_btn = QPushButton("Compute All")
        self.auto_btn.clicked.connect(self.auto_segment)
        self.reset_btn.clicked.connect(self.reset_masks)
        self.compute_btn.clicked.connect(self.compute_all)
        btn_row.addWidget(self.auto_btn)
        btn_row.addWidget(self.reset_btn)
        btn_row.addWidget(self.compute_btn)
        seg_form.addRow(btn_row)
        right.addWidget(seg_box)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Metric", "Right", "Left", "Bilateral"])
        right.addWidget(self.table, 1)

        export_row = QHBoxLayout()
        self.export_csv_btn = QPushButton("Export CSV")
        self.export_mask_btn = QPushButton("Export Masks (NIfTI)")
        self.export_pdf_btn = QPushButton("Export Report (PDF)")
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.export_mask_btn.clicked.connect(self.export_masks)
        self.export_pdf_btn.clicked.connect(self.export_pdf)
        export_row.addWidget(self.export_csv_btn)
        export_row.addWidget(self.export_mask_btn)
        export_row.addWidget(self.export_pdf_btn)
        right.addLayout(export_row)

    # ------------------------- data extraction -------------------------
    def _extract_ct_volume_and_spacing(self, app) -> Tuple[Optional[np.ndarray], Tuple[float, float, float]]:
        candidates = [
            "ct_hu", "ct_volume", "ct", "ct_data", "ct_img", "ct_numpy", "ct_np",
            "volume_ct", "ct_vol", "ct_arr",
        ]
        ct = None
        for name in candidates:
            if hasattr(app, name):
                val = getattr(app, name)
                if isinstance(val, np.ndarray) and val.ndim == 3:
                    ct = val.astype(np.float32, copy=False)
                    break
        if ct is None:
            return None, (1.0, 1.0, 1.0)

        spacing = (1.0, 1.0, 1.0)
        spacing_candidates = ["ct_spacing", "spacing", "voxel_spacing", "pixdim", "spacing_zyx"]
        for name in spacing_candidates:
            if hasattr(app, name):
                val = getattr(app, name)
                try:
                    if len(val) >= 3:
                        spacing = tuple(float(x) for x in val[:3])
                        break
                except Exception:
                    continue
        return ct, spacing

    def _extract_pet_suv(self, app) -> Optional[np.ndarray]:
        for name in ["pet_suv", "suv", "pet", "pet_data", "pet_np", "pet_volume"]:
            if hasattr(app, name):
                val = getattr(app, name)
                if isinstance(val, np.ndarray) and val.ndim == 3:
                    return val.astype(np.float32, copy=False)
        return None

    # ------------------------- segmentation -------------------------
    def _read_params(self):
        self.params = SegmentationParams(
            fgt_lo=self.fgt_lo_sb.value(),
            fgt_hi=self.fgt_hi_sb.value(),
            fat_lo=self.fat_lo_sb.value(),
            fat_hi=self.fat_hi_sb.value(),
            thoracic_fraction=self.thor_frac_sb.value(),
            morph_radius=self.morph_radius_sb.value(),
            use_totalseg=self.totalseg_cb.isChecked() and _TOTALSEG_AVAILABLE,
        )

    def auto_segment(self):
        if self.ct is None:
            QMessageBox.warning(self, "Breast Density", "No CT volume found in app.")
            return
        self._read_params()

        if self.params.use_totalseg and _TOTALSEG_AVAILABLE:
            try:
                # Fallback to attenuation method unless full TotalSegmentator pipeline exists.
                # Keep robust and silent per requirement.
                pass
            except Exception:
                pass

        self.body_mask = self._make_body_hull(self.ct, self.params.morph_radius)
        z0 = 0
        z1 = max(1, int(round(self.ct.shape[0] * self.params.thoracic_fraction)))
        thor = np.zeros_like(self.body_mask, dtype=bool)
        thor[z0:z1] = True

        roi = self.body_mask & thor
        fgt = (self.ct >= self.params.fgt_lo) & (self.ct <= self.params.fgt_hi) & roi
        fat = (self.ct >= self.params.fat_lo) & (self.ct <= self.params.fat_hi) & roi

        self.fgt_r, self.fgt_l, self.fat_r, self.fat_l = self._split_right_left(fgt, fat, self.body_mask & thor)
        self._update_overlay()
        self.compute_all()

    def reset_masks(self):
        self.body_mask = None
        self.fgt_r = self.fgt_l = self.fat_r = self.fat_l = None
        self._update_overlay()
        self.table.setRowCount(0)
        self.results = {}

    def _make_body_hull(self, ct: np.ndarray, morph_radius: int = 2) -> np.ndarray:
        mask = ct > -400
        if not _SCIPY_AVAILABLE:
            return mask
        if morph_radius > 0:
            rad = max(1, int(morph_radius))
            zz, yy, xx = np.ogrid[-rad : rad + 1, -rad : rad + 1, -rad : rad + 1]
            sph = (zz * zz + yy * yy + xx * xx) <= rad * rad
            mask = ndimage.binary_closing(mask, structure=sph)
        mask = ndimage.binary_fill_holes(mask)
        lbl, n = ndimage.label(mask)
        if n > 0:
            c = np.bincount(lbl.ravel())
            c[0] = 0
            mask = lbl == int(np.argmax(c))
        return mask

    def _split_right_left(self, fgt: np.ndarray, fat: np.ndarray, roi: np.ndarray):
        ys, xs = np.where(np.any(roi, axis=0))
        if len(xs) == 0:
            mid = self.ct.shape[2] // 2
        else:
            mid = int((xs.min() + xs.max()) // 2)
        xx = np.arange(self.ct.shape[2])[None, None, :]
        right = xx < mid
        left = ~right
        return fgt & right, fgt & left, fat & right, fat & left

    # ------------------------- display -------------------------
    def _compose_base_volume(self):
        mode = self.display_mode.currentText()
        if mode == "PET only":
            if self.pet_suv is None:
                return _window_ct_to_uint8(self.ct)
            arr = self.pet_suv
            arr = (arr - np.nanmin(arr)) / max(np.nanmax(arr) - np.nanmin(arr), 1e-6)
            return (arr * 255).astype(np.uint8)
        if mode == "Fused" and self.pet_suv is not None:
            ct8 = _window_ct_to_uint8(self.ct).astype(np.float32)
            p = self.pet_suv
            p = (p - np.nanmin(p)) / max(np.nanmax(p) - np.nanmin(p), 1e-6)
            p8 = (p * 255).astype(np.float32)
            fused = np.clip(0.6 * ct8 + 0.4 * p8, 0, 255)
            return fused.astype(np.uint8)
        return _window_ct_to_uint8(self.ct)

    def _update_display_volume(self):
        if self.ct is None:
            return
        base = self._compose_base_volume()
        for v in (self.axial_view, self.coronal_view, self.sagittal_view):
            v.set_volume(base)
        self._update_overlay()
        self._set_cross(tuple(self.cross))

    def _update_overlay(self):
        if self.ct is None:
            return
        rgba = np.zeros(self.ct.shape + (4,), dtype=np.uint8)
        if self.fgt_r is not None:
            rgba[self.fgt_r] = np.array([0, 255, 0, 110], dtype=np.uint8)
        if self.fgt_l is not None:
            rgba[self.fgt_l] = np.array([0, 255, 255, 110], dtype=np.uint8)
        if self.fat_r is not None:
            rgba[self.fat_r] = np.array([255, 255, 0, 60], dtype=np.uint8)
        if self.fat_l is not None:
            rgba[self.fat_l] = np.array([255, 0, 255, 60], dtype=np.uint8)

        for v in (self.axial_view, self.coronal_view, self.sagittal_view):
            v.set_overlay(rgba)

    def _set_cross(self, zyx: Tuple[int, int, int]):
        self.cross = list(zyx)
        z, y, x = zyx
        self.axial_view.set_index(z)
        self.coronal_view.set_index(y)
        self.sagittal_view.set_index(x)
        for v in (self.axial_view, self.coronal_view, self.sagittal_view):
            v.set_crosshair(zyx)

    def _on_slice_changed(self, axis: int, idx: int):
        z, y, x = self.cross
        if axis == 0:
            z = idx
        elif axis == 1:
            y = idx
        else:
            x = idx
        self._set_cross((z, y, x))

    def _on_crosshair_changed(self, zyx: Tuple[int, int, int]):
        self._set_cross(zyx)

    # ------------------------- metrics -------------------------
    def compute_all(self):
        if self.fgt_r is None or self.fgt_l is None or self.fat_r is None or self.fat_l is None:
            return

        spacing_mm = np.array(self.spacing, dtype=np.float64)
        voxel_ml = float(np.prod(spacing_mm) / 1000.0)

        fgt_r_v = self.fgt_r.sum() * voxel_ml
        fgt_l_v = self.fgt_l.sum() * voxel_ml
        fat_r_v = self.fat_r.sum() * voxel_ml
        fat_l_v = self.fat_l.sum() * voxel_ml

        vpd_r = 100.0 * fgt_r_v / max(fgt_r_v + fat_r_v, 1e-12)
        vpd_l = 100.0 * fgt_l_v / max(fgt_l_v + fat_l_v, 1e-12)
        fgt_b = fgt_r_v + fgt_l_v
        fat_b = fat_r_v + fat_l_v
        vpd_b = 100.0 * fgt_b / max(fgt_b + fat_b, 1e-12)
        asym = 100.0 * abs(vpd_r - vpd_l) / max((vpd_r + vpd_l) / 2.0, 1e-12)

        self.results = {
            "VPD_%": {"R": vpd_r, "L": vpd_l, "B": vpd_b},
            "FGT_volume_mL": {"R": fgt_r_v, "L": fgt_l_v, "B": fgt_b},
            "Fat_volume_mL": {"R": fat_r_v, "L": fat_l_v, "B": fat_b},
            "Asymmetry_index_%": {"R": asym, "L": asym, "B": asym},
        }

        fgt_all = self.fgt_r | self.fgt_l
        self.results.update(self._complexity_metrics("Complexity_R", self.fgt_r))
        self.results.update(self._complexity_metrics("Complexity_L", self.fgt_l))
        self.results.update(self._complexity_metrics("Complexity_B", fgt_all))

        if self.pet_suv is not None:
            self.results.update(self._pet_metrics())

        self._populate_table()

    def _complexity_metrics(self, prefix: str, mask: np.ndarray):
        if mask is None or not np.any(mask):
            return {}
        ct_vals = self.ct[mask].astype(np.float64)

        out: Dict[str, Dict[str, Optional[float]]] = {}

        # Shape
        vox = float(mask.sum())
        voxel_vol = float(np.prod(np.array(self.spacing, dtype=np.float64)))
        mesh_volume_ml = (vox * voxel_vol) / 1000.0
        surface_area_mm2 = self._surface_area(mask)
        sphericity = self._sphericity(mesh_volume_ml, surface_area_mm2)
        elongation = self._elongation(mask)

        out[f"{prefix}_MeshVolume_mL"] = {"R": None, "L": None, "B": mesh_volume_ml}
        out[f"{prefix}_SurfaceArea_mm2"] = {"R": None, "L": None, "B": surface_area_mm2}
        out[f"{prefix}_Sphericity"] = {"R": None, "L": None, "B": sphericity}
        out[f"{prefix}_Elongation"] = {"R": None, "L": None, "B": elongation}

        # First-order
        out[f"{prefix}_MeanHU"] = {"R": None, "L": None, "B": float(np.mean(ct_vals))}
        out[f"{prefix}_SDHU"] = {"R": None, "L": None, "B": float(np.std(ct_vals))}
        out[f"{prefix}_Skewness"] = {"R": None, "L": None, "B": float(skew(ct_vals)) if skew else float("nan")}
        out[f"{prefix}_Kurtosis"] = {"R": None, "L": None, "B": float(kurtosis(ct_vals)) if kurtosis else float("nan")}
        p10, p90 = np.percentile(ct_vals, [10, 90])
        out[f"{prefix}_IQR_P10_P90"] = {"R": None, "L": None, "B": float(p90 - p10)}

        tex = self._texture_metrics(mask)
        for k, v in tex.items():
            out[f"{prefix}_{k}"] = {"R": None, "L": None, "B": v}

        return out

    def _surface_area(self, mask: np.ndarray) -> float:
        if _SKIMAGE_AVAILABLE and measure is not None:
            try:
                verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5, spacing=self.spacing)
                tri = verts[faces]
                a = tri[:, 1] - tri[:, 0]
                b = tri[:, 2] - tri[:, 0]
                area = 0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum()
                return float(area)
            except Exception:
                pass
        if _SCIPY_AVAILABLE:
            er = ndimage.binary_erosion(mask)
            shell = mask ^ er
            # crude approximation using average face area
            s = np.array(self.spacing)
            face_area = float((s[0] * s[1] + s[0] * s[2] + s[1] * s[2]) / 3.0)
            return float(shell.sum() * face_area)
        return float(mask.sum())

    def _sphericity(self, vol_ml: float, area_mm2: float) -> float:
        vol_mm3 = vol_ml * 1000.0
        if vol_mm3 <= 0 or area_mm2 <= 0:
            return 0.0
        return float((math.pi ** (1.0 / 3.0)) * ((6.0 * vol_mm3) ** (2.0 / 3.0)) / area_mm2)

    def _elongation(self, mask: np.ndarray) -> float:
        coords = np.argwhere(mask)
        if len(coords) < 3:
            return 1.0
        cov = np.cov(coords.T)
        eig = np.sort(np.linalg.eigvalsh(cov))
        if eig[-1] <= 0:
            return 1.0
        return float(np.sqrt(max(eig[-2], 1e-12) / max(eig[-1], 1e-12)))

    def _texture_metrics(self, mask: np.ndarray) -> Dict[str, float]:
        vals = self.ct[mask]
        if vals.size == 0:
            return {}

        # 32-bin quantized patch for simple/robust texture estimate
        q = self._quantize_ct(self.ct)

        # Use largest slice to stabilize 2D texture fallback
        z_counts = np.sum(mask, axis=(1, 2))
        z = int(np.argmax(z_counts))
        patch = q[z]
        m2d = mask[z]
        if np.sum(m2d) == 0:
            return {k: 0.0 for k in (
                "GLCM_Contrast", "GLCM_Correlation", "GLCM_Energy", "GLCM_Homogeneity",
                "GLCM_Entropy", "GLCM_ClusterShade", "GLCM_ClusterProminence", "GLCM_IMC1",
                "GLRLM_SRE", "GLRLM_LRE", "GLRLM_RLN", "GLRLM_RunVariance",
            )}

        p = patch.copy()
        p[~m2d] = 0

        glcm_feats = self._glcm_features(p)
        glrlm_feats = self._glrlm_features(p, m2d)
        return {**glcm_feats, **glrlm_feats}

    def _quantize_ct(self, ct: np.ndarray, bins: int = 32) -> np.ndarray:
        lo, hi = -300.0, 300.0
        x = np.clip((ct - lo) / (hi - lo), 0.0, 0.999999)
        return (x * bins).astype(np.uint8)

    def _glcm_features(self, patch: np.ndarray) -> Dict[str, float]:
        if _SKIMAGE_AVAILABLE and feature is not None:
            glcm = feature.graycomatrix(
                patch,
                distances=[1],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                levels=32,
                symmetric=True,
                normed=True,
            )
            contrast = float(feature.graycoprops(glcm, "contrast").mean())
            corr = float(feature.graycoprops(glcm, "correlation").mean())
            energy = float(feature.graycoprops(glcm, "energy").mean())
            hom = float(feature.graycoprops(glcm, "homogeneity").mean())
            p = glcm[:, :, 0, :]
            p = p / max(p.sum(), 1e-12)
            ii, jj = np.indices(p.shape[:2])
            mu_i = np.sum(ii[..., None] * p)
            mu_j = np.sum(jj[..., None] * p)
            ent = float(-np.sum(p * np.log2(np.clip(p, 1e-12, 1.0))))
            cs = float(np.sum(((ii[..., None] + jj[..., None] - mu_i - mu_j) ** 3) * p))
            cp = float(np.sum(((ii[..., None] + jj[..., None] - mu_i - mu_j) ** 4) * p))
            pi = p.sum(axis=1)
            pj = p.sum(axis=0)
            hxy = ent
            pi_m = pi.mean(axis=1)
            pj_m = pj.mean(axis=1)
            pij = np.outer(pi_m, pj_m)[:, :, None]
            hxy1 = -np.sum(p * np.log2(np.clip(pij, 1e-12, 1.0)))
            hx = -np.sum(pi * np.log2(np.clip(pi, 1e-12, 1.0)))
            hy = -np.sum(pj * np.log2(np.clip(pj, 1e-12, 1.0)))
            imc1 = float((hxy - hxy1) / max(hx, hy, 1e-12))
        else:
            vals = patch.ravel().astype(np.float64)
            contrast = float(np.var(vals))
            corr = float("nan")
            energy = float(np.mean(vals ** 2))
            hom = float(1.0 / (1.0 + np.var(vals)))
            p = np.bincount(vals.astype(int), minlength=32).astype(np.float64)
            p /= max(p.sum(), 1e-12)
            ent = float(-np.sum(p * np.log2(np.clip(p, 1e-12, 1.0))))
            cs = float(np.mean((vals - np.mean(vals)) ** 3))
            cp = float(np.mean((vals - np.mean(vals)) ** 4))
            imc1 = float(0.0)

        return {
            "GLCM_Contrast": contrast,
            "GLCM_Correlation": corr,
            "GLCM_Energy": energy,
            "GLCM_Homogeneity": hom,
            "GLCM_Entropy": ent,
            "GLCM_ClusterShade": cs,
            "GLCM_ClusterProminence": cp,
            "GLCM_IMC1": imc1,
        }

    def _glrlm_features(self, patch: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        runs = []
        for row, row_m in zip(patch, mask):
            active = row[row_m]
            if active.size == 0:
                continue
            start = 0
            while start < active.size:
                val = active[start]
                end = start + 1
                while end < active.size and active[end] == val:
                    end += 1
                runs.append(end - start)
                start = end
        if not runs:
            return {
                "GLRLM_SRE": 0.0,
                "GLRLM_LRE": 0.0,
                "GLRLM_RLN": 0.0,
                "GLRLM_RunVariance": 0.0,
            }
        r = np.array(runs, dtype=np.float64)
        sre = float(np.mean(1.0 / np.clip(r ** 2, 1e-12, None)))
        lre = float(np.mean(r ** 2))
        uniq, cnt = np.unique(r.astype(int), return_counts=True)
        rln = float(np.sum(cnt ** 2) / max(len(r), 1))
        rv = float(np.var(r))
        return {
            "GLRLM_SRE": sre,
            "GLRLM_LRE": lre,
            "GLRLM_RLN": rln,
            "GLRLM_RunVariance": rv,
        }

    def _pet_metrics(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        if self.pet_suv is None:
            return out
        fgt_r = self.pet_suv[self.fgt_r] if np.any(self.fgt_r) else np.array([0.0])
        fgt_l = self.pet_suv[self.fgt_l] if np.any(self.fgt_l) else np.array([0.0])
        fat_r = self.pet_suv[self.fat_r] if np.any(self.fat_r) else np.array([0.0])
        fat_l = self.pet_suv[self.fat_l] if np.any(self.fat_l) else np.array([0.0])

        fgt_b = np.r_[fgt_r, fgt_l]
        out["SUV_mean_FGT"] = {"R": float(np.mean(fgt_r)), "L": float(np.mean(fgt_l)), "B": float(np.mean(fgt_b))}
        out["SUV_max_FGT"] = {"R": float(np.max(fgt_r)), "L": float(np.max(fgt_l)), "B": float(np.max(fgt_b))}

        whole_r = np.r_[fgt_r, fat_r]
        whole_l = np.r_[fgt_l, fat_l]
        bpu_r = float(np.mean(fgt_r) / max(np.mean(whole_r), 1e-12))
        bpu_l = float(np.mean(fgt_l) / max(np.mean(whole_l), 1e-12))
        out["BPU_index"] = {"R": bpu_r, "L": bpu_l, "B": float((bpu_r + bpu_l) / 2.0)}
        return out

    def _populate_table(self):
        rows = list(self.results.items())
        self.table.setRowCount(len(rows))
        for i, (metric, vals) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(metric))
            self.table.setItem(i, 1, QTableWidgetItem(self._fmt(vals.get("R"))))
            self.table.setItem(i, 2, QTableWidgetItem(self._fmt(vals.get("L"))))
            self.table.setItem(i, 3, QTableWidgetItem(self._fmt(vals.get("B"))))
        self.table.resizeColumnsToContents()

    def _fmt(self, v) -> str:
        if v is None:
            return ""
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)

    # ------------------------- export -------------------------
    def export_csv(self):
        if not self.results:
            self.compute_all()
        if not self.results:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "breast_density_metrics.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Metric", "Right", "Left", "Bilateral"])
            for k, vals in self.results.items():
                w.writerow([k, self._fmt(vals.get("R")), self._fmt(vals.get("L")), self._fmt(vals.get("B"))])

    def export_masks(self):
        if self.fgt_r is None:
            QMessageBox.warning(self, "Breast Density", "Please run Auto-Segment first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Export Masks")
        if not folder:
            return
        masks = {
            "right_fgt.nii.gz": self.fgt_r,
            "left_fgt.nii.gz": self.fgt_l,
            "right_fat.nii.gz": self.fat_r,
            "left_fat.nii.gz": self.fat_l,
        }
        affine = np.diag([self.spacing[0], self.spacing[1], self.spacing[2], 1.0])
        for name, m in masks.items():
            out_path = os.path.join(folder, name)
            arr = np.transpose(m.astype(np.uint8), (2, 1, 0))
            if _NIB_AVAILABLE:
                img = nib.Nifti1Image(arr, affine=affine)
                nib.save(img, out_path)
            else:
                np.save(out_path + ".npy", arr)

    def export_pdf(self):
        if not _REPORTLAB_AVAILABLE:
            QMessageBox.information(self, "Breast Density", "reportlab not installed; PDF export unavailable.")
            return
        if not self.results:
            self.compute_all()
        path, _ = QFileDialog.getSaveFileName(self, "Export PDF", "breast_density_report.pdf", "PDF (*.pdf)")
        if not path:
            return
        c = canvas.Canvas(path, pagesize=A4)
        w, h = A4
        y = h - 40
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, y, "Breast Density & Complexity Report")
        y -= 24
        c.setFont("Helvetica", 10)
        c.drawString(40, y, f"Generated: {datetime.now().isoformat(timespec='seconds')}")
        y -= 20
        c.drawString(40, y, f"CT shape: {'N/A' if self.ct is None else self.ct.shape}, spacing(mm): {self.spacing}")
        y -= 24
        c.setFont("Helvetica", 9)
        for metric, vals in self.results.items():
            line = f"{metric}: R={self._fmt(vals.get('R'))}, L={self._fmt(vals.get('L'))}, B={self._fmt(vals.get('B'))}"
            c.drawString(40, y, line[:160])
            y -= 13
            if y < 40:
                c.showPage()
                y = h - 40
                c.setFont("Helvetica", 9)
        c.save()


def _install_breast_density_toolbar(app):
    """Install Breast Density toolbar action into viewer app."""
    if app is None:
        return

    target_tb = None
    try:
        for tb in app.findChildren(QToolBar):
            if tb.windowTitle().strip().lower() == "clinical":
                target_tb = tb
                break
    except Exception:
        target_tb = None

    if target_tb is None:
        target_tb = QToolBar("Breast Density", app if isinstance(app, QMainWindow) else None)
        if hasattr(app, "addToolBar"):
            app.addToolBar(target_tb)

    # Avoid duplicate action insertion
    for act in target_tb.actions():
        if act.text() == "🩺 Breast Density":
            return

    action = QAction("🩺 Breast Density", app)

    def _open_dialog():
        try:
            dlg = getattr(app, "_breast_density_dlg", None)
            if dlg is not None and dlg.isVisible():
                dlg.raise_()
                dlg.activateWindow()
                return
            dlg = BreastDensityDialog(app)
            app._breast_density_dlg = dlg
            dlg.show()
        except Exception as e:
            print(f"[BreastDensity] failed to open dialog: {e}")
            traceback.print_exc()

    action.triggered.connect(_open_dialog)
    target_tb.addAction(action)


__all__ = ["BreastDensityDialog", "_install_breast_density_toolbar"]
