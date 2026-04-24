#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PET/CT Viewer v21 — Fixed 3D Rotating MIP + Radiomics handlers (Bundled YAML Support)
---------------------------------------------------------------
- Accurate 3D rotating MIP in SUV domain (CT/PET body hull masking).
- Percentile-based contrast (Auto per-frame or Global), outlier clip, gamma.
- Play/Stop controls; Space=Play/Pause, S=Stop (PyQt6 QtGui.QShortcut).
- Radiomics button handlers updated to load config_clinical.yaml and config_texture.yaml.
- MIP player window auto-resizes to screen; minimize & close buttons enabled.
- FIXED: DICOM series loading with better modality detection and fallback logic
- FIXED: PERCIST and RECIST tabs now properly access and use ROI masks
"""

from __future__ import annotations

import os
import sys
import math
import csv
import json
import zipfile
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets


def _v21_qt_no_edit_triggers():
    """Qt5/Qt6 compatible 'no edit triggers' flag."""
    # PyQt6: QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
    try:
        return QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
    except Exception:
        pass
    # PyQt5: QtWidgets.QAbstractItemView.NoEditTriggers
    try:
        return QtWidgets.QAbstractItemView.NoEditTriggers
    except Exception:
        pass
    # Fallback
    try:
        return QtWidgets.QAbstractItemView.EditTriggers(0)
    except Exception:
        return 0


def _v21_qt_edit_triggers(*names):
    """Return a Qt5/Qt6 compatible EditTriggers flag value."""
    cls = QtWidgets.QAbstractItemView
    # Qt6: enums are nested under QAbstractItemView.EditTrigger / SelectionBehavior
    try:
        enum = getattr(cls, "EditTrigger", None)
        if enum is not None:
            val = enum.NoEditTriggers
            for n in names:
                if hasattr(enum, n):
                    val = val | getattr(enum, n)
            return val
    except Exception:
        pass
    # Qt5 fallback: flags exposed as attributes directly on QAbstractItemView
    val = 0
    for n in names:
        val |= int(getattr(cls, n, 0))
    return val


def _v21_qt_select_rows():
    """Qt5/Qt6 compatible 'SelectRows' selection behavior."""
    # PyQt6: QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
    try:
        return QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
    except Exception:
        pass
    # PyQt5: QtWidgets.QAbstractItemView.SelectRows
    try:
        return QtWidgets.QAbstractItemView.SelectRows
    except Exception:
        pass
    # Fallback
    try:
        return QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems
    except Exception:
        return 0

import pyqtgraph as pg

# High-quality rendering defaults (helps CT/PET appear less blocky)
try:
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
except Exception:
    pass
try:
    # If OpenGL is available this can improve smooth zoom/pan
    pg.setConfigOptions(useOpenGL=True)
except Exception:
    pass

import pydicom
import SimpleITK as sitk

# --- Optional Radiomics (safe) ---
try:
    from radiomics import featureextractor
    import radiomics
    radiomics.setVerbosity(40)
    RADIOMICS_AVAILABLE = True
except Exception:
    RADIOMICS_AVAILABLE = False


# ----------------------------- Utilities ---------------------------------

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def _log(msg: str) -> None:
    print(msg, flush=True)

def _safe_float(x, default=None):
    try:
        if x is None: return default
        return float(x)
    except Exception:
        return default

def _make_image_item():
    """Create a pyqtgraph ImageItem with high-quality display defaults.

    - axisOrder row-major to match numpy ZYX slicing logic
    - disable auto-downsample (prevents blocky CT/PET appearance on fit-to-view)
    """
    try:
        it = pg.ImageItem(axisOrder="row-major")
    except Exception:
        it = pg.ImageItem()
        try:
            it.setOpts(axisOrder="row-major")
        except Exception:
            pass
    try:
        # Critical for sharp CT rendering (avoids low-res downsampling artefacts)
        it.setAutoDownsample(False)
    except Exception:
        pass
    try:
        # Prefer smooth scaling if available (pyqtgraph >= 0.13 supports it via setOpts)
        it.setOpts(smooth=True)
    except Exception:
        pass
    return it

def _parse_dicom_dt(dt_str: Optional[str], d_str: Optional[str] = None, t_str: Optional[str] = None) -> Optional[datetime]:
    if dt_str:
        s = str(dt_str)
        try:
            if "." in s:
                base, frac = s.split(".", 1); base = base.ljust(14, "0"); frac = (frac + "000000")[:6]
                return datetime.strptime(base + frac, "%Y%m%d%H%M%S%f")
            s = s.ljust(14, "0")
            return datetime.strptime(s, "%Y%m%d%H%M%S")
        except Exception:
            pass
    if d_str and t_str:
        d = str(d_str); t = str(t_str)
        try:
            if "." in t:
                tbase, frac = t.split(".", 1); tbase = tbase.ljust(6, "0"); frac = (frac + "000000")[:6]
                return datetime.strptime(d + tbase + frac, "%Y%m%d%H%M%S%f")
            t = t.ljust(6, "0")
            return datetime.strptime(d + t, "%Y%m%d%H%M%S")
        except Exception:
            return None
    return None

def _dicom_series_index(dicom_root: str) -> Dict[str, Dict]:
    series: Dict[str, Dict] = {}
    for base, _, files in os.walk(dicom_root):
        for fn in files:
            fp = os.path.join(base, fn)
            if fn.lower().endswith((".json", ".txt", ".csv", ".xml", ".nii", ".nii.gz", ".png", ".jpg")): continue
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            except Exception:
                continue
            suid = getattr(ds, "SeriesInstanceUID", None)
            if not suid: continue
            if suid not in series: series[suid] = {"files": [], "example": ds}
            series[suid]["files"].append(fp)
    if not series: raise RuntimeError("No readable DICOM series found.")
    return series

def _sort_series_files(files: List[str]) -> List[str]:
    recs = []
    for fp in files:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            ipp = getattr(ds, "ImagePositionPatient", None)
            iop = getattr(ds, "ImageOrientationPatient", None)
            inst = getattr(ds, "InstanceNumber", None)
            if ipp and iop and len(ipp) == 3 and len(iop) == 6:
                r = np.array(iop[:3], dtype=float); c = np.array(iop[3:], dtype=float); n = np.cross(r, c)
                pos = float(np.dot(np.array(ipp, dtype=float), n))
                key = ("pos", pos)
            elif inst is not None:
                key = ("inst", int(inst))
            else:
                key = ("name", fp)
            recs.append((key, fp))
        except Exception:
            continue
    if not recs: return sorted(files)
    tag = recs[0][0][0]
    recs.sort(key=(lambda x: x[0][1]) if tag in ("pos", "inst") else (lambda x: x[1]))
    return [fp for _, fp in recs]

def _sitk_read_series(files: List[str]) -> sitk.Image:
    reader = sitk.ImageSeriesReader(); reader.SetFileNames(files)
    return reader.Execute()

@dataclass
class SUVMeta:
    units: str; mode: str; weight_kg: Optional[float]
    injected_dose_bq: Optional[float]; injected_dose_decay_corrected_bq: Optional[float]
    half_life_s: Optional[float]; t_inj: Optional[str]; t_acq: Optional[str]
    dt_s: Optional[float]; used_scale: float; note: str

def _compute_suvbw_from_pet_bqml(pet_bqml: np.ndarray, pt_ds: pydicom.Dataset, time_ref: str = "AcquisitionDateTime", suv_scale: float = 1.0) -> Tuple[np.ndarray, SUVMeta]:
    units = str(getattr(pt_ds, "Units", "") or "")
    weight_kg = _safe_float(getattr(pt_ds, "PatientWeight", None))
    injected_dose_bq = None; half_life_s = None; t_inj = None
    if hasattr(pt_ds, "RadiopharmaceuticalInformationSequence") and pt_ds.RadiopharmaceuticalInformationSequence:
        r = pt_ds.RadiopharmaceuticalInformationSequence[0]
        injected_dose_bq = _safe_float(getattr(r, "RadionuclideTotalDose", None))
        half_life_s = _safe_float(getattr(r, "RadionuclideHalfLife", None))
        t_inj_dt = _parse_dicom_dt(getattr(r, "RadiopharmaceuticalStartDateTime", None),
                                   getattr(pt_ds, "SeriesDate", None),
                                   getattr(r, "RadiopharmaceuticalStartTime", None))
        if t_inj_dt is None:
            t_inj_dt = _parse_dicom_dt(None,
                                       getattr(r, "RadiopharmaceuticalStartDate", None),
                                       getattr(r, "RadiopharmaceuticalStartTime", None))
        t_inj = t_inj_dt.isoformat() if t_inj_dt else None
    else:
        t_inj_dt = None
    if time_ref == "AcquisitionDateTime":
        acq_dt = _parse_dicom_dt(getattr(pt_ds, "AcquisitionDateTime", None),
                                 getattr(pt_ds, "AcquisitionDate", None),
                                 getattr(pt_ds, "AcquisitionTime", None))
    elif time_ref == "SeriesDateTime":
        acq_dt = _parse_dicom_dt(None, getattr(pt_ds, "SeriesDate", None),
                                 getattr(pt_ds, "SeriesTime", None))
    else:
        acq_dt = _parse_dicom_dt(getattr(pt_ds, "ContentDateTime", None),
                                 getattr(pt_ds, "ContentDate", None),
                                 getattr(pt_ds, "ContentTime", None))
    t_acq = acq_dt.isoformat() if acq_dt else None
    dose_corr = None; dt_s = None
    if injected_dose_bq and half_life_s and t_inj_dt and acq_dt:
        dt_s = (acq_dt - t_inj_dt).total_seconds()
        lam = math.log(2.0) / half_life_s
        dose_corr = injected_dose_bq * math.exp(-lam * dt_s)
    if weight_kg is None or injected_dose_bq is None:
        return pet_bqml.astype(np.float32), SUVMeta(units, "RAW", weight_kg, injected_dose_bq, dose_corr, half_life_s, t_inj, t_acq, dt_s, suv_scale, "Missing weight/dose")
    denom = dose_corr if (dose_corr is not None and dose_corr > 0) else injected_dose_bq
    if denom <= 0: denom = injected_dose_bq
    weight_g = weight_kg * 1000.0
    suv = (pet_bqml.astype(np.float32) * (weight_g / denom)) * float(suv_scale)
    return suv, SUVMeta(units, "SUVBW", weight_kg, injected_dose_bq, dose_corr, half_life_s, t_inj, t_acq, dt_s, float(suv_scale), f"Ref={time_ref}")

def _resample_to_reference(mov: sitk.Image, ref: sitk.Image) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref); resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1000); resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(mov)

def _window_ct_to_uint8(ct_hu: np.ndarray, wl: float, ww: float) -> np.ndarray:
    lo = wl - ww / 2.0; hi = wl + ww / 2.0
    x = np.clip(ct_hu, lo, hi); x = (x - lo) / max(hi - lo, 1e-6)
    return (x * 255.0).astype(np.uint8)

def _make_hot_iron_lut(n: int = 256) -> np.ndarray:
    x = np.linspace(0, 1, n, dtype=np.float32)
    r = np.clip(3 * x, 0, 1); g = np.clip(3 * x - 1, 0, 1); b = np.clip(3 * x - 2, 0, 1)
    return (np.stack([r, g, b], axis=1) * 255.0).astype(np.uint8)

def _pet_to_rgba(pet_suv: np.ndarray, vmax: float, alpha: float, lut: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    if vmax <= 1e-6: vmax = 1.0
    x = pet_suv / vmax
    x = np.clip(x, 0.0, 1.0)
    x = np.power(x, gamma)  # why: reveal mid-range uptake
    mask_bg = x < 0.03
    idx = (x * 255.0).astype(np.uint8)
    rgb = lut[idx]
    a = (x * 255.0 * alpha).astype(np.uint8)
    a[mask_bg] = 0
    return np.dstack([rgb, a])

def _largest_component_mask(bin_img: sitk.Image) -> sitk.Image:
    cc = sitk.ConnectedComponent(bin_img)
    stats = sitk.LabelShapeStatisticsImageFilter(); stats.Execute(cc)
    if stats.GetNumberOfLabels() == 0:
        return bin_img * 0
    areas = [(l, stats.GetNumberOfPixels(l)) for l in stats.GetLabels()]
    label = max(areas, key=lambda t: t[1])[0]
    return sitk.BinaryThreshold(cc, lowerThreshold=label, upperThreshold=label, insideValue=1, outsideValue=0)

def _make_body_mask_from_ct(ct_img: sitk.Image) -> sitk.Image:
    # why: CT-based hull is robust; HU>-400 approximates body/soft tissues
    ct_hu = sitk.Cast(ct_img, sitk.sitkFloat32)
    body = sitk.BinaryThreshold(ct_hu, lowerThreshold=-400.0, upperThreshold=10000.0, insideValue=1, outsideValue=0)
    body = sitk.BinaryMorphologicalClosing(body, (2,2,2))
    body = sitk.BinaryFillhole(body)
    body = _largest_component_mask(body)
    return sitk.Cast(body, sitk.sitkUInt8)

def _make_body_mask_from_pet_suv(pet_suv_img: sitk.Image) -> sitk.Image:
    thr = sitk.BinaryThreshold(pet_suv_img, lowerThreshold=0.3, upperThreshold=1e9, insideValue=1, outsideValue=0)
    thr = sitk.BinaryMorphologicalClosing(thr, (2,2,2))
    thr = _largest_component_mask(thr)
    return sitk.Cast(thr, sitk.sitkUInt8)

def _calculate_lesion_dimensions(mask_array: np.ndarray, spacing: Tuple[float, float, float]) -> Dict[str, float]:
    if np.sum(mask_array) == 0: return {"Dim_Transverse_mm": 0.0, "Dim_AP_mm": 0.0, "Dim_CC_mm": 0.0}
    z_indices, y_indices, x_indices = np.where(mask_array > 0)
    sz, sy, sx = spacing
    width_x = (np.max(x_indices) - np.min(x_indices) + 1) * sx
    height_y = (np.max(y_indices) - np.min(y_indices) + 1) * sy
    depth_z = (np.max(z_indices) - np.min(z_indices) + 1) * sz
    return {
        "Dim_Transverse_mm": round(width_x, 2),
        "Dim_AP_mm": round(height_y, 2),
        "Dim_CC_mm": round(depth_z, 2)
    }


# ----------------------------- Slice View (safe) ---------------------------------
class ROIViewBox(pg.ViewBox):
    def __init__(self, slice_view, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slice_view = slice_view
        self.setMouseEnabled(x=True, y=True); self.setMenuEnabled(False)
    def wheelEvent(self, ev, axis=None):
        if ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier: super().wheelEvent(ev, axis)
        else: self.slice_view.on_wheel(1 if ev.delta() > 0 else -1); ev.accept()
    def mouseDragEvent(self, ev, axis=None):
        if self.slice_view.is_mip_active: return
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.slice_view.mode in ("paint", "erase"):
                p = self.slice_view.img_ct.mapFromScene(ev.scenePos())
                self.slice_view.apply_brush(float(p.x()), float(p.y()), commit=ev.isFinish()); ev.accept(); return
            if self.slice_view.mode == "lasso_paint":
                p = self.slice_view.img_ct.mapFromScene(ev.scenePos())
                if ev.isStart(): self.slice_view.lasso_begin(p.x(), p.y())
                elif ev.isFinish(): self.slice_view.lasso_end(p.x(), p.y())
                else: self.slice_view.lasso_add(p.x(), p.y())
                ev.accept(); return
        super().mouseDragEvent(ev, axis)
    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            if self.slice_view.is_mip_active: return
            p = self.slice_view.img_ct.mapFromScene(ev.scenePos())
            self.slice_view.set_crosshair_from_click(p.x(), p.y())
            if self.slice_view.mode == "region_grow": self.slice_view.request_region_grow(p.x(), p.y())
            elif self.slice_view.mode in ("paint", "erase"): self.slice_view.apply_brush(p.x(), p.y(), commit=True)
            ev.accept(); return
        super().mouseClickEvent(ev)

class SliceView(QtWidgets.QWidget):
    sig_crosshair_changed = QtCore.pyqtSignal(int, int, int)
    sig_slice_changed = QtCore.pyqtSignal(str, int)
    sig_mask_changed = QtCore.pyqtSignal()
    sig_region_grow_req = QtCore.pyqtSignal(int, int, int)

    def __init__(self, title, view_type, parent=None):
        super().__init__(parent); self.title, self.view_type = title, view_type
        self.ct_hu = self.pet_suv = self.mask = None; self.spacing_zyx = (1.0, 1.0, 1.0)
        self.mask_primary = None
        self.meta_masks = []  # list of bool masks (metastatic lesions)
        self.active_roi_class = "primary"  # 'primary' or 'meta'
        self.active_meta_index = 0
        self.cross_z = self.cross_y = self.cross_x = 0; self.slice_idx = 0
        self.mode = "pan"; self.brush_mm = 8.0; self.pet_alpha = 0.55; self.pet_vmax = 15.0; self.pet_gamma = 1.0; self.pet_intensity = 1.0; self.pet_contrast = 1.0
        self.ct_wl = 40.0; self.ct_ww = 400.0; self.view_mode = 'fused'
        self.show_mask = True
        self.is_mip_active = False
        # Segmentation overlays: dict mapping (R, G, B, A) colour tuple → 3-D bool ndarray
        self.seg_overlays: dict = {}
        self._lut = _make_hot_iron_lut(256); self._build_ui()

    @property
    def current_zyx(self):
        return self.cross_z, self.cross_y, self.cross_x

    def _build_ui(self):
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        hdr = QtWidgets.QHBoxLayout()
        self.lbl = QtWidgets.QLabel(f"<b>{self.title}</b>"); hdr.addWidget(self.lbl)
        self.lbl_info = QtWidgets.QLabel(""); hdr.addWidget(self.lbl_info, 1)
        btn_reset = QtWidgets.QPushButton("Reset"); btn_reset.clicked.connect(self.reset_view); hdr.addWidget(btn_reset)
        lay.addLayout(hdr)
        self.glw = pg.GraphicsLayoutWidget()

        # Improve visual quality (smooth scaling)
        try:
            self.glw.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        except Exception:
            try:
                self.glw.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
            except Exception:
                pass

        self.vb = ROIViewBox(self)
        self.vb.setAspectLocked(True); self.vb.invertY(False)
        self.plot = self.glw.addPlot(viewBox=self.vb); self.plot.hideAxis('left'); self.plot.hideAxis('bottom')
        self.plot.setMenuEnabled(False)
        self.img_ct = _make_image_item(); self.img_pet = _make_image_item(); self.img_mask = _make_image_item()
        self.plot.addItem(self.img_ct); self.plot.addItem(self.img_pet); self.plot.addItem(self.img_mask)
        self.vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((0, 255, 0, 180), width=1))
        self.hline = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen((0, 255, 0, 180), width=1))
        self.plot.addItem(self.vline); self.plot.addItem(self.hline)
        self.lasso_item = pg.PlotDataItem(pen=pg.mkPen((0, 255, 255), width=2)); self.lasso_item.setVisible(False)
        self.plot.addItem(self.lasso_item); lay.addWidget(self.glw, 1)
        slay = QtWidgets.QHBoxLayout()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._on_slider); slay.addWidget(self.slider, 1)
        self.lbl_slice = QtWidgets.QLabel("0/0"); slay.addWidget(self.lbl_slice); lay.addLayout(slay)

    def set_volumes(self, ct_hu, pet_suv, mask, spacing_zyx,
                    mask_primary=None, meta_masks=None, active_meta_index=0, active_roi_class="primary"):
        """Set/refresh the backing volumes and ROI masks.
        - mask: kept for backward compatibility (used as primary if mask_primary is None)
        - mask_primary: primary-tumor ROI mask (bool ndarray)
        - meta_masks: list of metastatic lesion masks (bool ndarrays)
        - active_roi_class: 'primary' or 'meta' (which mask is editable)
        """
        self.ct_hu, self.pet_suv, self.spacing_zyx = ct_hu, pet_suv, spacing_zyx
        self.mask_primary = mask_primary if mask_primary is not None else mask
        self.meta_masks = meta_masks if meta_masks is not None else []
        self.active_roi_class = active_roi_class if active_roi_class in ("primary", "meta") else "primary"
        self.active_meta_index = int(active_meta_index) if self.meta_masks else 0
        # choose editable mask reference
        if self.active_roi_class == "meta" and self.meta_masks and 0 <= self.active_meta_index < len(self.meta_masks):
            self.mask = self.meta_masks[self.active_meta_index]
        else:
            self.mask = self.mask_primary

        nz, ny, nx = pet_suv.shape
        max_idx = nz if self.view_type == "axial" else (ny if self.view_type == "coronal" else nx)
        self.slider.setRange(0, max_idx - 1); self.slice_idx = max_idx // 2
        self.cross_z, self.cross_y, self.cross_x = nz // 2, ny // 2, nx // 2
        self.slider.blockSignals(True); self.slider.setValue(self.slice_idx); self.slider.blockSignals(False)
        self.update_view(reset=True)

    def set_active_mask(self, active_roi_class="primary", active_meta_index=0):
        """Switch which ROI mask is editable without changing the base volumes."""
        self.active_roi_class = active_roi_class if active_roi_class in ("primary", "meta") else "primary"
        self.active_meta_index = int(active_meta_index) if self.meta_masks else 0
        if self.active_roi_class == "meta" and self.meta_masks and 0 <= self.active_meta_index < len(self.meta_masks):
            self.mask = self.meta_masks[self.active_meta_index]
        else:
            self.mask = self.mask_primary
        if self.pet_suv is not None:
            self.update_view(reset=False)

    def _slice_mask(self, m):
        if m is None: 
            return None
        if self.view_type == "coronal" and self.is_mip_active:
            try:
                return np.max(m, axis=1)
            except Exception:
                return None
        if self.view_type == "axial":
            return m[self.slice_idx][::-1]
        if self.view_type == "coronal":
            return m[:, self.slice_idx]
        return m[:, :, self.slice_idx]

    def set_seg_overlays(self, overlays: dict):
        """Set colour-coded segmentation overlays for display in all three views.

        Args:
            overlays: dict mapping (R, G, B, A) colour tuples to 3-D boolean
                      numpy arrays (same shape as the loaded CT volume).
                      Pass an empty dict {} to clear all segmentation overlays.
        """
        self.seg_overlays = dict(overlays) if overlays else {}
        if self.pet_suv is not None:
            self.update_view()

    def set_mip_mode(self, active):
        if self.view_type == "coronal": self.is_mip_active = active; self.update_view()

    def set_view_mode(self, mode): self.view_mode = mode; self.update_view()
    def _on_slider(self, v):
        if self.is_mip_active: return
        self.slice_idx = int(v); self.sig_slice_changed.emit(self.view_type, self.slice_idx)
        self._sync_crosshair_axis_from_slice(); self.update_view()
    def on_wheel(self, step):
        if self.is_mip_active: return
        if self.slider.maximum() > 0: self.slider.setValue(np.clip(self.slider.value() + step, 0, self.slider.maximum()))
    def _sync_crosshair_axis_from_slice(self):
        if self.pet_suv is None: return
        if self.view_type == "axial": self.cross_z = self.slice_idx
        elif self.view_type == "coronal": self.cross_y = self.slice_idx
        else: self.cross_x = self.slice_idx
        self.sig_crosshair_changed.emit(self.cross_z, self.cross_y, self.cross_x)
    def _get_slice_arrays(self):
        ct, pet, m = self.ct_hu, self.pet_suv, self.mask
        if self.view_type == "coronal" and self.is_mip_active:
            ct_slice = ct[:, self.slice_idx]
            pet_mip = np.max(pet, axis=1)
            mask_mip = np.max(m, axis=1)
            return ct_slice, pet_mip, mask_mip
        if self.view_type == "axial": return ct[self.slice_idx][::-1], pet[self.slice_idx][::-1], m[self.slice_idx][::-1]
        if self.view_type == "coronal": return ct[:, self.slice_idx], pet[:, self.slice_idx], m[:, self.slice_idx]
        return ct[:, :, self.slice_idx], pet[:, :, self.slice_idx], m[:, :, self.slice_idx]
    def display_to_voxel(self, x_disp, y_disp):
        nz, ny, nx = self.pet_suv.shape; xi, yi = int(round(x_disp)), int(round(y_disp))
        if self.view_type == "axial": return self.slice_idx, np.clip((ny-1)-yi, 0, ny-1), np.clip(xi, 0, nx-1)
        if self.view_type == "coronal": return np.clip(yi, 0, nz-1), self.slice_idx, np.clip(xi, 0, nx-1)
        return np.clip(yi, 0, nz-1), np.clip(xi, 0, ny-1), self.slice_idx
    def voxel_to_display(self, z, y, x):
        """Return mm scene coordinates for a voxel position (used for crosshair lines).

        After _apply_image_rects the ViewBox scene is in physical mm, so
        crosshair lines must also be positioned in mm.
        """
        sz, sy, sx = self.spacing_zyx
        nz, ny, nx = self.pet_suv.shape
        if self.view_type == "axial":
            return float(x) * sx, float((ny - 1) - y) * sy
        if self.view_type == "coronal":
            return float(x) * sx, float(z) * sz
        return float(y) * sy, float(z) * sz
    def set_crosshair(self, z, y, x, update_slice=False):
        self.cross_z, self.cross_y, self.cross_x = int(z), int(y), int(x)
        if update_slice and not self.is_mip_active:
            idx = self.cross_z if self.view_type=="axial" else (self.cross_y if self.view_type=="coronal" else self.cross_x)
            self.slider.setValue(idx)
        self.update_crosshair_lines()
    def set_crosshair_from_click(self, x_disp, y_disp):
        z, y, x = self.display_to_voxel(x_disp, y_disp); self.cross_z, self.cross_y, self.cross_x = z, y, x
        self.sig_crosshair_changed.emit(z, y, x); self.update_crosshair_lines()
    def request_region_grow(self, x_disp, y_disp):
        z, y, x = self.display_to_voxel(x_disp, y_disp); self.sig_region_grow_req.emit(z, y, x)

    # ------------------------------------------------------------------
    # Physical-spacing helpers (fix left-right stretching artifact)
    # ------------------------------------------------------------------
    def _slice_spacing_mm(self):
        """Return (row_spacing_mm, col_spacing_mm) for the slice shown in this view.

        Slices are stored as (rows, cols):
          axial   → rows = Y voxels (sy),  cols = X voxels (sx)
          coronal → rows = Z voxels (sz),  cols = X voxels (sx)
          sagittal→ rows = Z voxels (sz),  cols = Y voxels (sy)
        """
        sz, sy, sx = self.spacing_zyx
        if self.view_type == "axial":
            return float(sy), float(sx)
        if self.view_type == "coronal":
            return float(sz), float(sx)
        return float(sz), float(sy)

    def _apply_image_rects(self, nrows: int, ncols: int) -> None:
        """Scale ImageItems so each pixel occupies its true physical size in mm.

        After setRect the ViewBox scene coordinate is in mm, eliminating the
        left-right (transverse) stretch that occurs when voxels are not isotropic.
        """
        row_sp, col_sp = self._slice_spacing_mm()
        rect = QtCore.QRectF(0.0, 0.0, float(ncols) * col_sp, float(nrows) * row_sp)
        for img_item in (self.img_ct, self.img_pet, self.img_mask):
            try:
                img_item.setRect(rect)
            except Exception:
                pass
    # ------------------------------------------------------------------

    def apply_brush(self, x_disp, y_disp, commit=True):
        if self.mask is None: return
        z, y, x = self.display_to_voxel(x_disp, y_disp)
        self.cross_z, self.cross_y, self.cross_x = z, y, x
        self.sig_crosshair_changed.emit(z, y, x)
        sz, sy, sx = self.spacing_zyx; r_mm = float(self.brush_mm)
        if self.view_type == "axial":
            ry, rx = max(1, int(round(r_mm/sy))), max(1, int(round(r_mm/sx)))
            y0, y1 = max(0, y-ry), min(self.mask.shape[1]-1, y+ry); x0, x1 = max(0, x-rx), min(self.mask.shape[2]-1, x+rx)
            yy, xx = np.ogrid[y0:y1+1, x0:x1+1]; mask_local = ((yy-y)*sy)**2 + ((xx-x)*sx)**2 <= r_mm**2
            target_slice = self.mask[z, y0:y1+1, x0:x1+1]
        elif self.view_type == "coronal":
            rz, rx = max(1, int(round(r_mm/sz))), max(1, int(round(r_mm/sx)))
            z0, z1 = max(0, z-rz), min(self.mask.shape[0]-1, z+rz); x0, x1 = max(0, x-rx), min(self.mask.shape[2]-1, x+rx)
            zz, xx = np.ogrid[z0:z1+1, x0:x1+1]; mask_local = ((zz-z)*sz)**2 + ((xx-x)*sx)**2 <= r_mm**2
            target_slice = self.mask[z0:z1+1, y, x0:x1+1]
        else:
            rz, ry = max(1, int(round(r_mm/sz))), max(1, int(round(r_mm/sy)))
            z0, z1 = max(0, z-rz), min(self.mask.shape[0]-1, z+rz); y0, y1 = max(0, y-ry), min(self.mask.shape[1]-1, y+ry)
            zz, yy = np.ogrid[z0:z1+1, y0:y1+1]; mask_local = ((zz-z)*sz)**2 + ((yy-y)*sy)**2 <= r_mm**2
            target_slice = self.mask[z0:z1+1, y0:y1+1, x]
        if self.mode == "paint": target_slice[mask_local] = True
        elif self.mode == "erase": target_slice[mask_local] = False
        self.update_view(); self.update_crosshair_lines()
        # Always emit so the other two orthogonal views update in real-time during painting
        self.sig_mask_changed.emit()
    def lasso_begin(self, x, y):
        # x, y are image-local pixel coords (from img_ct.mapFromScene after setRect)
        self._lasso_pts_px = [(float(x), float(y))]
        row_sp, col_sp = self._slice_spacing_mm()
        self.lasso_item.setData([float(x) * col_sp], [float(y) * row_sp])
        self.lasso_item.setVisible(True)
    def lasso_add(self, x, y):
        if not getattr(self, "_lasso_pts_px", None): self.lasso_begin(x, y); return
        lx, ly = self._lasso_pts_px[-1]
        if (x-lx)**2 + (y-ly)**2 < 0.25: return  # min-distance threshold in pixel space
        self._lasso_pts_px.append((float(x), float(y)))
        row_sp, col_sp = self._slice_spacing_mm()
        self.lasso_item.setData([p[0]*col_sp for p in self._lasso_pts_px],
                                 [p[1]*row_sp for p in self._lasso_pts_px])
    def lasso_end(self, x, y):
        self.lasso_add(x, y); pts = getattr(self, "_lasso_pts_px", []); self.lasso_item.setVisible(False)
        if len(pts) < 3: self._lasso_pts_px = []; return
        xs, ys = [p[0] for p in pts] + [pts[0][0]], [p[1] for p in pts] + [pts[0][1]]
        z, y, x = self.current_zyx
        if self.view_type == "axial": self.mask[z, :, :] |= _polyfill_2d(self.mask.shape[1], self.mask.shape[2], xs, ys)[::-1]
        elif self.view_type == "coronal": self.mask[:, y, :] |= _polyfill_2d(self.mask.shape[0], self.mask.shape[2], xs, ys)
        else: self.mask[:, :, x] |= _polyfill_2d(self.mask.shape[0], self.mask.shape[1], xs, ys)
        self._lasso_pts_px = []; self.update_view(); self.sig_mask_changed.emit()
    def update_crosshair_lines(self):
        x, y = self.voxel_to_display(self.cross_z, self.cross_y, self.cross_x)
        self.vline.setPos(x); self.hline.setPos(y)
    def update_view(self, reset=False):
        if self.ct_hu is None: return
        ct2, pet2, m2 = self._get_slice_arrays()
        show_ct = (self.view_mode in ['fused', 'ct']); show_pet = (self.view_mode in ['fused', 'pet'])
        if show_ct:
            ct8 = _window_ct_to_uint8(ct2, self.ct_wl, self.ct_ww)
            self.img_ct.setImage(ct8, levels=(0, 255), autoLevels=False)
        else: self.img_ct.clear()
        if show_pet:
            alpha = self.pet_alpha if self.view_mode == 'fused' else 1.0
            rgba = _pet_to_rgba(pet2, self.pet_vmax, alpha, _make_hot_iron_lut(), gamma=self.pet_gamma, intensity=self.pet_intensity, contrast=self.pet_contrast)
            self.img_pet.setImage(rgba, levels=(0, 255), autoLevels=False)
        else: self.img_pet.clear()
        if self.show_mask:
            h, w = m2.shape
            ov = np.zeros((h, w, 4), dtype=np.uint8)

            # Segmentation overlays (breast density module) — rendered first so
            # the primary/meta ROI markers always appear on top.
            # Colour key: cyan = right whole-breast, steel-blue = left whole-breast,
            #             orange = right fibroglandular, gold = left fibroglandular.
            for (r, g, b, a), m_seg in self.seg_overlays.items():
                try:
                    ms = self._slice_mask(m_seg)
                    if ms is not None and ms.shape == (h, w):
                        ov[ms.astype(bool)] = [r, g, b, a]
                except Exception:
                    pass

            # Primary tumor mask (green)
            try:
                m_primary = self._slice_mask(self.mask_primary)
                if m_primary is not None:
                    ov[m_primary.astype(bool)] = [0, 255, 0, 110]
            except Exception:
                pass

            # Metastatic lesions (magenta) + active lesion highlight (yellow)
            try:
                if self.meta_masks:
                    meta_union = np.zeros((h, w), dtype=bool)
                    for idx, mm in enumerate(self.meta_masks):
                        ms = self._slice_mask(mm)
                        if ms is None: 
                            continue
                        meta_union |= ms.astype(bool)
                    ov[meta_union] = [255, 0, 255, 85]

                    if self.active_roi_class == "meta" and 0 <= self.active_meta_index < len(self.meta_masks):
                        m_active = self._slice_mask(self.meta_masks[self.active_meta_index])
                        if m_active is not None:
                            ov[m_active.astype(bool)] = [255, 255, 0, 160]
            except Exception:
                pass

            self.img_mask.setImage(ov, levels=(0, 255), autoLevels=False)
        else: 
            self.img_mask.clear()
        # Apply physical-spacing rects so anatomy is displayed without transverse stretch
        self._apply_image_rects(ct2.shape[0], ct2.shape[1])
        self.lbl_slice.setText("MIP (Whole Body)" if self.is_mip_active else f"{self.slider.value()+1}/{self.slider.maximum()+1}")
        self.update_crosshair_lines()
        if reset: self.reset_view()
    def reset_view(self):
        if self.ct_hu is None: return
        ct2, _, _ = self._get_slice_arrays()
        row_sp, col_sp = self._slice_spacing_mm()
        w_mm = float(ct2.shape[1]) * col_sp
        h_mm = float(ct2.shape[0]) * row_sp
        self.vb.setRange(QtCore.QRectF(0.0, 0.0, w_mm, h_mm), padding=0.0)


# --- 3D Rotating MIP Player (fixed + sized to screen) ---
class RotatingMIPPlayer(QtWidgets.QDialog):
    def __init__(self, projections: List[np.ndarray], global_suv_stats: Dict[str, float], parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D Rotating MIP — Accurate SUV")

        # Ensure minimize & close buttons; keep resizable dialog.
        self.setWindowFlags(self.windowFlags()
                            | QtCore.Qt.WindowType.WindowMinimizeButtonHint
                            | QtCore.Qt.WindowType.WindowCloseButtonHint)

        # Size to available screen so controls don't get cut off.
        scr = QtGui.QGuiApplication.primaryScreen()
        avail = scr.availableGeometry() if scr else QtCore.QRect(0, 0, 1280, 800)
        # Conservative fractions of available area; with sensible mins/maxes.
        target_w = max(560, min(int(avail.width() * 0.7), 900))
        target_h = max(520, min(int(avail.height() * 0.8), 1000))
        self.resize(target_w, target_h)

        self.projections = projections
        self.global_stats = global_suv_stats
        self.current_frame = -1
        self.flip_v = False

        layout = QtWidgets.QVBoxLayout(self)

        # View
        self.glw = pg.GraphicsLayoutWidget()
        self.vb = pg.ViewBox(); self.vb.setAspectLocked(True); self.vb.invertY(False)
        self.plot = self.glw.addPlot(viewBox=self.vb); self.plot.hideAxis('left'); self.plot.hideAxis('bottom')
        self.img_item = pg.ImageItem(axisOrder='row-major'); self.plot.addItem(self.img_item)
        layout.addWidget(self.glw)

        # Controls
        grid = QtWidgets.QGridLayout()
        self.chk_flip = QtWidgets.QCheckBox("Flip Vertical"); self.chk_flip.toggled.connect(lambda s: setattr(self, "flip_v", s)); grid.addWidget(self.chk_flip, 0, 0)
        self.chk_auto = QtWidgets.QCheckBox("Auto (per-frame)"); self.chk_auto.setChecked(True); grid.addWidget(self.chk_auto, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Percentile:"), 1, 0)
        self.slider_pct = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.slider_pct.setRange(900, 1000); self.slider_pct.setValue(990)
        self.lbl_pct = QtWidgets.QLabel("99.0")
        grid.addWidget(self.slider_pct, 1, 1); grid.addWidget(self.lbl_pct, 1, 2)
        self.slider_pct.valueChanged.connect(lambda v: self.lbl_pct.setText(f"{v/10.0:.1f}"))

        grid.addWidget(QtWidgets.QLabel("Clip %:"), 2, 0)
        self.slider_clip = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.slider_clip.setRange(990, 999); self.slider_clip.setValue(997)
        self.lbl_clip = QtWidgets.QLabel("99.7")
        grid.addWidget(self.slider_clip, 2, 1); grid.addWidget(self.lbl_clip, 2, 2)
        self.slider_clip.valueChanged.connect(lambda v: self.lbl_clip.setText(f"{v/10.0:.1f}"))

        grid.addWidget(QtWidgets.QLabel("Gamma:"), 3, 0)
        self.slider_gamma = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.slider_gamma.setRange(4, 30); self.slider_gamma.setValue(10)
        self.lbl_gamma = QtWidgets.QLabel("1.0")
        grid.addWidget(self.slider_gamma, 3, 1); grid.addWidget(self.lbl_gamma, 3, 2)
        self.slider_gamma.valueChanged.connect(lambda v: self.lbl_gamma.setText(f"{v/10.0:.1f}"))

        grid.addWidget(QtWidgets.QLabel("Playback:"), 4, 0)
        pb_row = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        pb_row.addWidget(self.btn_play); pb_row.addWidget(self.btn_stop)
        pb_row.addWidget(QtWidgets.QLabel("FPS"))
        self.spin_fps = QtWidgets.QSpinBox(); self.spin_fps.setRange(1, 60); self.spin_fps.setValue(10); pb_row.addWidget(self.spin_fps)
        grid.addLayout(pb_row, 4, 1, 1, 2)
        layout.addLayout(grid)

        # Timer
        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self.next_frame)

        # Wires
        self.btn_play.clicked.connect(self.toggle_play_pause)
        self.btn_stop.clicked.connect(self.stop)
        self.spin_fps.valueChanged.connect(self._apply_fps)

        # Shortcuts (PyQt6 → QtGui.QShortcut)
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.toggle_play_pause)
        QtGui.QShortcut(QtGui.QKeySequence("S"), self, activated=self.stop)

        # Cache globals and autoplay
        self.global_percentiles = {p: self._percentile_all(p) for p in (95.0, 98.0, 99.0, 99.7, 99.9)}
        self._apply_fps(); self._autoplay()

    def _autoplay(self):
        self.timer.start()
        if self.current_frame < 0: self.next_frame()

    def _apply_fps(self):
        fps = max(1, int(self.spin_fps.value()))
        self.timer.setInterval(int(1000 / fps))

    def toggle_play_pause(self):
        if self.timer.isActive():
            self.timer.stop(); self.btn_play.setText("Play")
        else:
            self.timer.start(); self.btn_play.setText("Pause")

    def stop(self):
        self.timer.stop()
        self.current_frame = -1
        self.next_frame()
        self.btn_play.setText("Play")

    def _percentile_all(self, p: float) -> float:
        arr = np.concatenate([pr.flatten() for pr in self.projections]) if self.projections else np.array([10.0], dtype=np.float32)
        return float(np.percentile(arr, p))

    def _frame_vmax(self, frame: np.ndarray, p: float, clip_p: float, auto: bool) -> float:
        if auto:
            upper = float(np.percentile(frame, clip_p))
            base = np.clip(frame, 0, upper)
            vmax = float(np.percentile(base, p))
        else:
            upper = float(self.global_percentiles.get(clip_p, self._percentile_all(clip_p)))
            base_cap = self._percentile_all(p)
            vmax = max(min(upper, base_cap), 1e-3)
        return max(vmax, 1e-3)

    def next_frame(self):
        if not self.projections: return
        self.current_frame = (self.current_frame + 1) % len(self.projections)
        frame = self.projections[self.current_frame]
        if self.flip_v: frame = np.flipud(frame)
        p = float(self.slider_pct.value()) / 10.0
        cp = float(self.slider_clip.value()) / 10.0
        gamma = float(self.slider_gamma.value()) / 10.0
        vmax = self._frame_vmax(frame, p, cp, self.chk_auto.isChecked())
        lut = _make_hot_iron_lut(256)
        rgba = _pet_to_rgba(frame, vmax, 1.0, lut, gamma=gamma)
        self.img_item.setImage(rgba)


# --- Series selection (safe) ---
class SeriesSelectDialog(QtWidgets.QDialog):
    def __init__(self, series_index, parent=None):
        super().__init__(parent); self.setWindowTitle("Select PET and CT Series"); self.resize(500, 400)
        self.selected_pet = None; self.selected_ct = None
        self.series_index = series_index
        layout = QtWidgets.QVBoxLayout()
        self.list_widget = QtWidgets.QListWidget(); self.series_map = {}
        for suid, info in series_index.items():
            ds = info["example"]
            modality = getattr(ds, "Modality", "UNK")
            desc = getattr(ds, "SeriesDescription", "No Desc")
            size = len(info["files"])
            label = f"[{modality}] {desc} ({size} images)"
            self.series_map[label] = suid
            self.list_widget.addItem(label)
        layout.addWidget(QtWidgets.QLabel("Detected Series in Folder:")); layout.addWidget(self.list_widget)
        btn_box = QtWidgets.QHBoxLayout()
        self.btn_set_pet = QtWidgets.QPushButton("Set as PET"); self.btn_set_pet.clicked.connect(self.set_pet)
        self.btn_set_ct = QtWidgets.QPushButton("Set as CT"); self.btn_set_ct.clicked.connect(self.set_ct)
        btn_box.addWidget(self.btn_set_pet); btn_box.addWidget(self.btn_set_ct); layout.addLayout(btn_box)
        self.lbl_status = QtWidgets.QLabel("PET: None | CT: None"); layout.addWidget(self.lbl_status)
        ok_btn = QtWidgets.QPushButton("Load Selected"); ok_btn.clicked.connect(self.accept); layout.addWidget(ok_btn)
        self.setLayout(layout)
    def set_pet(self):
        cur = self.list_widget.currentItem()
        if cur: self.selected_pet = self.series_map[cur.text()]; self.update_status()
    def set_ct(self):
        cur = self.list_widget.currentItem()
        if cur: self.selected_ct = self.series_map[cur.text()]; self.update_status()
    def update_status(self):
        p = "Selected" if self.selected_pet else "None"; c = "Selected" if self.selected_ct else "None"
        self.lbl_status.setText(f"PET: {p} | CT: {c}")


# --- Metrics window (safe) ---
class MetricsWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent); self.setWindowTitle("ROI Metrics"); self.resize(350, 500)
        self.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)
        layout = QtWidgets.QVBoxLayout(self)
        self.table = QtWidgets.QTableWidget(); self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Metric", "Value"]); self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        btn_lay = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Save CSV"); self.btn_save.clicked.connect(self.save_to_csv); btn_lay.addWidget(self.btn_save)
        btn_close = QtWidgets.QPushButton("Close"); btn_close.clicked.connect(self.hide); btn_lay.addWidget(btn_close)
        layout.addLayout(btn_lay); self.current_data = {}
    def update_data(self, data: Dict[str, float]):
        self.current_data = data; self.table.setRowCount(0)
        for k, v in data.items():
            r = self.table.rowCount(); self.table.insertRow(r)
            val_str = f"{v:.4f}" if isinstance(v, (float, int)) else str(v)
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(k))); self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(val_str))
        self.show(); self.raise_(); self.activateWindow()
    def save_to_csv(self):
        if not self.current_data: return
        fp, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Metrics", "", "CSV Files (*.csv)")
        if not fp: return
        try:
            with open(fp, 'w', newline='') as f:
                writer = csv.writer(f); writer.writerow(["Metric", "Value"])
                for k, v in self.current_data.items(): writer.writerow([k, v])
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved to {fp}")
        except Exception as e: QtWidgets.QMessageBox.critical(self, "Error", f"Could not save: {str(e)}")


# --- App (MIP fixed; radiomics handlers folded in) ---

# --- Tumor burden window (Primary + Metastases) ---
class TumorBurdenWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Total Tumor Burden (Primary + Metastases)")
        self.resize(900, 520)
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)

        layout = QtWidgets.QVBoxLayout(self)

        self.lbl = QtWidgets.QLabel(
            "Per-lesion metrics (manual ROIs). Primary and metastatic lesions are tracked separately; totals are summed."
        )
        self.lbl.setWordWrap(True)
        layout.addWidget(self.lbl)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(11)
        self.table.setHorizontalHeaderLabels([
            "Lesion", "Type", "Vol (mL)", "SUVmax", "SUVmean",
            "MTV41 (mL)", "TLG41", "Dim LR (mm)", "Dim AP (mm)", "Dim CC (mm)",
            "CT mean/max (HU)"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Save CSV")
        self.btn_save.clicked.connect(self._save_csv)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_save)
        layout.addLayout(btn_row)

        self._rows_cache = []

    def set_rows(self, rows: List[Dict[str, object]]):
        self._rows_cache = rows or []
        self.table.setRowCount(len(self._rows_cache))
        for r, row in enumerate(self._rows_cache):
            def _set(c, v):
                it = QtWidgets.QTableWidgetItem("" if v is None else str(v))
                if row.get("_is_total", False):
                    f = it.font(); f.setBold(True); it.setFont(f)
                self.table.setItem(r, c, it)

            _set(0, row.get("Lesion", ""))
            _set(1, row.get("Type", ""))
            _set(2, row.get("Vol_ml", ""))
            _set(3, row.get("SUVmax", ""))
            _set(4, row.get("SUVmean", ""))
            _set(5, row.get("MTV41_ml", ""))
            _set(6, row.get("TLG41", ""))
            _set(7, row.get("Dim_LR_mm", ""))
            _set(8, row.get("Dim_AP_mm", ""))
            _set(9, row.get("Dim_CC_mm", ""))
            _set(10, row.get("CT_mean_max", ""))

        self.show()
        self.raise_()
        self.activateWindow()

    def _save_csv(self):
        if not self._rows_cache:
            return
        fp, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Tumor Burden", "", "CSV Files (*.csv)")
        if not fp:
            return
        try:
            cols = ["Lesion","Type","Vol_ml","SUVmax","SUVmean","MTV41_ml","TLG41","Dim_LR_mm","Dim_AP_mm","Dim_CC_mm","CT_mean_max"]
            with open(fp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(cols)
                for row in self._rows_cache:
                    w.writerow([row.get(c, "") for c in cols])
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved to {fp}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not save: {str(e)}")


class PETCTManualROIApp(QtWidgets.QMainWindow):
    def __init__(self, dicom_dir, out_dir, **kwargs):
        super().__init__()
        self.setWindowTitle("PET/CT Viewer v21 — Fixed MIP")
        self.resize(1600, 950)
        self.dicom_dir = dicom_dir; self.out_dir = out_dir if out_dir else os.path.join(os.getcwd(), "results")
        self.args = kwargs
        self.pet_suv = self.mask = None; self.pet_img_sitk_raw = None; self.ct_img_sitk_raw = None; self.body_mask_img = None
        self.temp_dirs = []
        self.metrics_win = MetricsWindow(self)
        self.meta_masks = []  # list of bool ndarray masks for metastases
        self.active_roi_class = "primary"  # 'primary' or 'meta'
        self.active_meta_index = 0
        self.burden_win = TumorBurdenWindow(self)
        self._build_ui()
        # Ensure the v21 dock tabs (includes Display sliders for CT/PET) are attached
        try:
            if "_v21_attach_dock_tabs" in globals():
                if not hasattr(self, "tabDock") or self.tabDock is None:
                    _v21_attach_dock_tabs(self)
                else:
                    try:
                        self.tabDock.setVisible(True)
                        self.tabDock.show()
                    except Exception:
                        pass
        except Exception:
            pass

        if self.dicom_dir and os.path.exists(self.dicom_dir): self._load_pet_ct()

    def _build_ui(self):
        cw = QtWidgets.QWidget(); self.setCentralWidget(cw); root = QtWidgets.QVBoxLayout(cw)
        load_bar = QtWidgets.QHBoxLayout()
        load_bar.addWidget(QtWidgets.QLabel("<b>Load Data:</b>"))
        btn_open_folder = QtWidgets.QPushButton("Open Folder"); btn_open_folder.clicked.connect(self.on_open_folder); load_bar.addWidget(btn_open_folder)
        btn_open_zip = QtWidgets.QPushButton("Open ZIP"); btn_open_zip.clicked.connect(self.on_open_zip); load_bar.addWidget(btn_open_zip)
        load_bar.addStretch(1)
        load_bar.addWidget(QtWidgets.QLabel("|  <b>View:</b>"))
        self.btn_view_fused = QtWidgets.QPushButton("Fused"); self.btn_view_ct = QtWidgets.QPushButton("CT Only"); self.btn_view_pet = QtWidgets.QPushButton("PET Only")
        for b in (self.btn_view_fused, self.btn_view_ct, self.btn_view_pet): b.setCheckable(True)
        self.btn_view_fused.setChecked(True)
        self.btn_view_fused.clicked.connect(lambda: self._set_view_mode('fused'))
        self.btn_view_ct.clicked.connect(lambda: self._set_view_mode('ct'))
        self.btn_view_pet.clicked.connect(lambda: self._set_view_mode('pet'))
        load_bar.addWidget(self.btn_view_fused); load_bar.addWidget(self.btn_view_ct); load_bar.addWidget(self.btn_view_pet)
        root.addLayout(load_bar)

        bar = QtWidgets.QHBoxLayout()
        self.btn_pan = QtWidgets.QPushButton("Pan"); self.btn_lasso = QtWidgets.QPushButton("Draw Contour")
        self.btn_paint = QtWidgets.QPushButton("Paint"); self.btn_erase = QtWidgets.QPushButton("Erase")
        self.btn_grow = QtWidgets.QPushButton("Smart Fill (3D)")
        self.btn_grow.setStyleSheet("font-weight: bold; color: #005500;")
        for b in (self.btn_pan, self.btn_lasso, self.btn_paint, self.btn_erase, self.btn_grow): b.setCheckable(True); bar.addWidget(b)
        self.btn_pan.clicked.connect(lambda: self._set_mode("pan")); self.btn_lasso.clicked.connect(lambda: self._set_mode("lasso_paint"))
        self.btn_paint.clicked.connect(lambda: self._set_mode("paint")); self.btn_erase.clicked.connect(lambda: self._set_mode("erase"))
        self.btn_grow.clicked.connect(lambda: self._set_mode("region_grow"))

        # --- ROI target: Primary vs Metastatic (separate masks) ---
        bar.addSpacing(12)
        bar.addWidget(QtWidgets.QLabel("ROI:"))
        self.btn_roi_primary = QtWidgets.QPushButton("Primary")
        self.btn_roi_meta = QtWidgets.QPushButton("Metastatic")
        for b in (self.btn_roi_primary, self.btn_roi_meta):
            b.setCheckable(True)
            bar.addWidget(b)
        self.btn_roi_primary.setChecked(True)
        self.btn_roi_primary.clicked.connect(lambda: self._set_roi_class("primary"))
        self.btn_roi_meta.clicked.connect(lambda: self._set_roi_class("meta"))

        self.btn_meta_add = QtWidgets.QPushButton("Add Lesion")
        self.btn_meta_next = QtWidgets.QPushButton("Next")
        self.btn_meta_del = QtWidgets.QPushButton("Delete")
        self.lbl_meta = QtWidgets.QLabel("")
        for b in (self.btn_meta_add, self.btn_meta_next, self.btn_meta_del):
            bar.addWidget(b)
        bar.addWidget(self.lbl_meta)

        self.btn_meta_add.clicked.connect(self._add_meta_lesion)
        self.btn_meta_next.clicked.connect(self._next_meta_lesion)
        self.btn_meta_del.clicked.connect(self._delete_meta_lesion)
        bar.addSpacing(12); bar.addWidget(QtWidgets.QLabel("Fill(%):"))
        self.spin_grow_pct = QtWidgets.QDoubleSpinBox(); self.spin_grow_pct.setValue(41.0); bar.addWidget(self.spin_grow_pct)
        bar.addWidget(QtWidgets.QLabel("Brush:")); self.spin_brush = QtWidgets.QDoubleSpinBox()
        self.spin_brush.setValue(8.0); self.spin_brush.valueChanged.connect(self._update_brush); bar.addWidget(self.spin_brush)

        bar.addWidget(QtWidgets.QLabel("Vmax:")); self.spin_vmax = QtWidgets.QDoubleSpinBox()
        self.spin_vmax.setRange(0.1, 99999999.0); self.spin_vmax.setValue(15.0)
        self.spin_vmax.valueChanged.connect(self._update_pet_vmax); bar.addWidget(self.spin_vmax)

        self.chk_mip = QtWidgets.QCheckBox("Static MIP"); self.chk_mip.toggled.connect(self._toggle_mip); bar.addWidget(self.chk_mip)

        self.btn_cine = QtWidgets.QPushButton("Cinematic MIP"); self.btn_cine.setStyleSheet("background-color: #AAFFAA; font-weight: bold;")
        self.btn_cine.clicked.connect(self._generate_rotating_mip); bar.addWidget(self.btn_cine)

        self.btn_save = QtWidgets.QPushButton("Basic Metrics"); self.btn_save.clicked.connect(self._compute_and_save)
        bar.addStretch(1); bar.addWidget(self.btn_save)
        self.btn_burden = QtWidgets.QPushButton("Tumor Burden"); self.btn_burden.clicked.connect(self._show_tumor_burden)
        bar.addWidget(self.btn_burden)
        root.addLayout(bar)
        self._update_meta_ui()

        rad_bar = QtWidgets.QHBoxLayout()
        rad_lbl = QtWidgets.QLabel("<b>IBSI Radiomics:</b>"); rad_bar.addWidget(rad_lbl)
        rad_bar.addWidget(QtWidgets.QLabel("Source:"))
        self.cmb_rad_source = QtWidgets.QComboBox(); self.cmb_rad_source.addItems(["PET (SUV)", "CT (HU)"]); rad_bar.addWidget(self.cmb_rad_source)
        rad_bar.addSpacing(10)
        self.btn_rad_clinical = QtWidgets.QPushButton("A: Clinical")
        self.btn_rad_first = QtWidgets.QPushButton("B: 1st Order")
        self.btn_rad_second = QtWidgets.QPushButton("C: 2nd Order")
        self.btn_rad_higher = QtWidgets.QPushButton("D: Higher Order")
        for b in [self.btn_rad_clinical, self.btn_rad_first, self.btn_rad_second, self.btn_rad_higher]:
            b.setEnabled(RADIOMICS_AVAILABLE); rad_bar.addWidget(b)
        if not RADIOMICS_AVAILABLE: rad_lbl.setText("<b>IBSI Radiomics (Not Installed):</b>")
        self.btn_rad_clinical.clicked.connect(self._on_btn_clinical_click)
        self.btn_rad_first.clicked.connect(self._on_btn_first_order_click)
        self.btn_rad_second.clicked.connect(self._on_btn_glcm_click)
        self.btn_rad_higher.clicked.connect(self._on_btn_higher_order_click)
        rad_bar.addStretch(1); self.btn_show_table = QtWidgets.QPushButton("Show Results Table")
        self.btn_show_table.clicked.connect(self.metrics_win.show); rad_bar.addWidget(self.btn_show_table)
        root.addLayout(rad_bar)

        grid = QtWidgets.QGridLayout()
        self.view_ax = SliceView("Axial", "axial"); self.view_co = SliceView("Coronal", "coronal"); self.view_sa = SliceView("Sagittal", "sagittal")
        for i, v in enumerate((self.view_ax, self.view_co, self.view_sa)):
            grid.addWidget(v, 0, i); v.sig_crosshair_changed.connect(self._on_crosshair_changed)
            v.sig_slice_changed.connect(self._on_slice_changed); v.sig_region_grow_req.connect(self._on_region_grow)
            v.sig_mask_changed.connect(self._refresh_all_views)
        root.addLayout(grid, 1)

        self.status = QtWidgets.QTextEdit(); self.status.setFixedHeight(60); self.status.setPlaceholderText("Log..."); root.addWidget(self.status)

    def on_open_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if folder: self.dicom_dir = folder; self._load_pet_ct()
    def on_open_zip(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select DICOM Archive", "", "Archives (*.zip)")
        if not fname: return
        self._status(f"Extracting {fname}...")
        try:
            temp_dir = tempfile.mkdtemp(); self.temp_dirs.append(temp_dir)
            with zipfile.ZipFile(fname, 'r') as zf: zf.extractall(temp_dir)
            self.dicom_dir = temp_dir; self._load_pet_ct()
        except Exception as e: QtWidgets.QMessageBox.critical(self, "Extraction Error", f"Failed to open ZIP:\n{str(e)}")
    def closeEvent(self, event):
        for d in self.temp_dirs:
            if os.path.exists(d): shutil.rmtree(d, ignore_errors=True)
        super().closeEvent(event)
    def _status(self, s): self.status.append(s); _log(s)
    def _set_mode(self, mode):
        for b in (self.btn_pan, self.btn_lasso, self.btn_paint, self.btn_erase, self.btn_grow):
            b.blockSignals(True); b.setChecked(False); b.blockSignals(False)
        {"pan": self.btn_pan, "lasso_paint": self.btn_lasso, "paint": self.btn_paint, "erase": self.btn_erase, "region_grow": self.btn_grow}[mode].setChecked(True)
        for v in (self.view_ax, self.view_co, self.view_sa): v.mode = mode
    def _set_view_mode(self, mode):
        self.btn_view_fused.blockSignals(True); self.btn_view_ct.blockSignals(True); self.btn_view_pet.blockSignals(True)
        self.btn_view_fused.setChecked(mode == 'fused'); self.btn_view_ct.setChecked(mode == 'ct'); self.btn_view_pet.setChecked(mode == 'pet')
        self.btn_view_fused.blockSignals(False); self.btn_view_ct.blockSignals(False); self.btn_view_pet.blockSignals(False)
        for v in (self.view_ax, self.view_co, self.view_sa): v.set_view_mode(mode)
    def _toggle_mip(self, checked): self.view_co.set_mip_mode(checked)
    def _update_brush(self, v): 
        for vw in (self.view_ax, self.view_co, self.view_sa): vw.brush_mm = float(v)
    def _update_pet_vmax(self, v):
        for vw in (self.view_ax, self.view_co, self.view_sa): vw.pet_vmax = float(v); vw.update_view()
    def _on_crosshair_changed(self, z, y, x):
        # Store last crosshair (z,y,x) so other tools (e.g., PERCIST liver ROI) can use it
        self.cross_z, self.cross_y, self.cross_x = int(z), int(y), int(x)
        for v in (self.view_ax, self.view_co, self.view_sa): v.set_crosshair(z, y, x)
    def _on_slice_changed(self, name, idx):
        if name=="axial": self._on_crosshair_changed(idx, self.view_ax.cross_y, self.view_ax.cross_x)
        elif name=="coronal": self._on_crosshair_changed(self.view_co.cross_z, idx, self.view_co.cross_x)
        else: self._on_crosshair_changed(self.view_sa.cross_z, self.view_sa.cross_y, idx)
    def _refresh_all_views(self):
        self.view_ax.update_view()
        self.view_co.update_view()
        self.view_sa.update_view()
        # Refresh clinical tabs if they exist
        self._refresh_clinical_tabs()
    def _on_region_grow(self, z, y, x):
        if self.pet_suv is None:
            return
        seed_val = float(self.pet_suv[z, y, x])
        pct = float(self.spin_grow_pct.value())
        self._status(f"Growing from seed SUV={seed_val:.2f} (> {pct}%)")
        new_mask = _perform_3d_region_grow(self.pet_suv, (z, y, x), pct)
        target = self._get_active_mask()
        if target is None:
            self._status("No active ROI mask (switch to Primary ROI or create a Metastatic lesion).")
            return
        target |= new_mask
        self._refresh_all_views()
        self._status(f"Added {np.count_nonzero(new_mask)} voxels.")

    def _load_pet_ct(self):
        if not self.dicom_dir: return
        try:
            idx = _dicom_series_index(self.dicom_dir)
            pet_uid, ct_uid = None, None
            if len(idx) > 1:
                dlg = SeriesSelectDialog(idx, self)
                if dlg.exec(): pet_uid, ct_uid = dlg.selected_pet, dlg.selected_ct
            elif len(idx) == 1: pet_uid = list(idx.keys())[0]
            if not pet_uid: self._status("Load cancelled."); return
            
            # FIXED: Use improved series loading with better fallback
            pet_img, pet_files, pet_uid = self._load_best_series_fixed(idx, pet_uid, prefer_modalities=("PT", "PET", "NM"))
            
            # Load CT if selected (keep CT native geometry; resample PET SUV to CT for display/fusion)
            if ct_uid:
                ct_raw, ct_files, ct_uid = self._load_best_series_fixed(idx, ct_uid, prefer_modalities=("CT",))
                self.ct_image = ct_raw
                self.ct_hu = sitk.GetArrayFromImage(self.ct_image).astype(np.float32)
            else:
                self.ct_image = None
                self.ct_hu = None

            self.pet_img_sitk_raw = pet_img
            pet_np = sitk.GetArrayFromImage(pet_img).astype(np.float32)
            ds0 = pydicom.dcmread(pet_files[0], stop_before_pixels=True)
            self.pet_suv, self.suv_meta = _compute_suvbw_from_pet_bqml(pet_np, ds0)

            # If CT exists, upsample PET SUV to CT grid (prevents CT/fusion being forced to PET low-res geometry)
            if self.ct_image is not None:
                try:
                    _suv_img = sitk.GetImageFromArray(self.pet_suv.astype(np.float32))
                    _suv_img.CopyInformation(self.pet_img_sitk_raw)
                    _suv_img_ct = _resample_to_reference(_suv_img, self.ct_image)
                    self.pet_suv = sitk.GetArrayFromImage(_suv_img_ct).astype(np.float32)
                    # Update spacing to reference geometry (CT)
                    sp = self.ct_image.GetSpacing()
                    self.spacing_zyx = (float(sp[2]), float(sp[1]), float(sp[0]))
                except Exception as _e:
                    self._status(f"[WARN] PET→CT resample failed, using PET grid: {_e}")
                    sp = self.pet_img_sitk_raw.GetSpacing()
                    self.spacing_zyx = (float(sp[2]), float(sp[1]), float(sp[0]))
            else:
                sp = self.pet_img_sitk_raw.GetSpacing()
                self.spacing_zyx = (float(sp[2]), float(sp[1]), float(sp[0]))
            min_val, max_val = float(np.nanmin(self.pet_suv)), float(np.nanmax(self.pet_suv))
            self._status(f"SUV Range: [{min_val:.2f}, {max_val:.2f}]")
            p99 = np.percentile(self.pet_suv[self.pet_suv>0], 99) if np.any(self.pet_suv>0) else 10.0
            self.spin_vmax.setRange(0.1, max(1.0, float(max_val * 1.5))); self.spin_vmax.setValue(float(p99))
            self.mask = np.zeros_like(self.pet_suv, dtype=bool)
            # metastases are stored as separate lesion masks (same geometry as primary)
            self.meta_masks = []
            self.active_roi_class = "primary"
            self.active_meta_index = 0
            for v in (self.view_ax, self.view_co, self.view_sa):
                v.set_volumes(self.ct_hu if self.ct_hu is not None else np.zeros_like(self.pet_suv),
                              self.pet_suv, self.mask, self.spacing_zyx,
                              mask_primary=self.mask, meta_masks=self.meta_masks,
                              active_meta_index=self.active_meta_index, active_roi_class=self.active_roi_class)
            suv_img = sitk.GetImageFromArray(self.pet_suv.astype(np.float32)); suv_img.CopyInformation(self.ct_image if self.ct_image is not None else self.pet_img_sitk_raw)
            self.body_mask_img = _make_body_mask_from_ct(self.ct_image) if self.ct_image is not None else _make_body_mask_from_pet_suv(suv_img)
            self._set_mode("pan")
            self._status(f"Loaded. SUV={self.suv_meta.mode}. Fusion={'Yes' if self.ct_hu is not None else 'No'}")
            # --- Ensure crosshair coordinates are initialized (used by Clinical Tools: PERCIST liver ROI, etc.)
            # Some workflows show a visual crosshair immediately but the shared (app-level) crosshair fields
            # may still be unset until the user clicks. Initialize to the volume center so tools work out-of-box.
            try:
                vol = None
                if getattr(self, 'pet_suv', None) is not None:
                    vol = self.pet_suv
                elif getattr(self, 'ct_hu', None) is not None:
                    vol = self.ct_hu
                if vol is not None and getattr(vol, 'ndim', 0) >= 3:
                    z0 = int(vol.shape[0] // 2)
                    y0 = int(vol.shape[1] // 2)
                    x0 = int(vol.shape[2] // 2)
                    # propagate to all views + store on main app
                    self._on_crosshair_changed(z0, y0, x0)
            except Exception:
                pass
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", str(e)); self._status(f"Error: {e}")

    def _load_best_series_fixed(self, idx: dict, initial_uid: str, prefer_modalities=("PT",)):
        """Improved series loading with better modality detection and fallback"""
        candidates = []
        if initial_uid:
            candidates.append(initial_uid)
        
        # Add all UIDs as candidates
        for uid in idx.keys():
            if uid not in candidates:
                candidates.append(uid)
        
        tried = []
        last_err = None
        
        # First pass: try exact modality match
        for uid in candidates:
            try:
                entry = idx.get(uid)
                if not entry: continue
                ds = entry.get("example")
                mod = getattr(ds, "Modality", "").upper() if ds else ""
                
                if prefer_modalities and mod not in [m.upper() for m in prefer_modalities]:
                    tried.append((uid, mod, "wrong modality"))
                    continue
                    
                files = _sort_series_files(entry["files"])
                if len(files) < 2:
                    tried.append((uid, mod, f"only {len(files)} files"))
                    continue
                    
                img = _sitk_read_series(files)
                arr = sitk.GetArrayFromImage(img)
                
                # Basic volume check
                if arr.shape[0] < 2 or arr.shape[1] < 64 or arr.shape[2] < 64:
                    tried.append((uid, mod, f"small shape {arr.shape}"))
                    continue
                    
                return img, files, uid
                
            except Exception as e:
                last_err = e
                tried.append((uid, mod, f"error: {str(e)[:50]}"))
                continue
        
        # Second pass: try any series with enough files
        for uid in candidates:
            try:
                entry = idx.get(uid)
                if not entry: continue
                files = entry["files"]
                if len(files) < 10:
                    continue
                    
                files = _sort_series_files(files)
                img = _sitk_read_series(files)
                arr = sitk.GetArrayFromImage(img)
                
                if arr.shape[0] < 2 or arr.shape[1] < 64 or arr.shape[2] < 64:
                    continue
                    
                return img, files, uid
                
            except Exception as e:
                last_err = e
                continue
        
        msg = f"Could not load a valid series for modalities={prefer_modalities}. Tried: {tried}"
        if last_err is not None:
            msg += f" (last error: {last_err})"
        raise RuntimeError(msg)

    # ---------- FIXED 3D Rotating MIP ----------
    def _generate_rotating_mip(self):
        if self.pet_img_sitk_raw is None:
            QtWidgets.QMessageBox.warning(self, "No Image", "Load PET data first."); return
        self._status("Generating 360° MIP (masked, percentile)...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            suv_img = sitk.GetImageFromArray(self.pet_suv.astype(np.float32)); suv_img.CopyInformation(self.ct_image if self.ct_image is not None else self.pet_img_sitk_raw)
            body = self.body_mask_img if self.body_mask_img is not None else _make_body_mask_from_pet_suv(suv_img)

            size = suv_img.GetSize()
            center_idx = [sz/2.0 for sz in size]
            center_phys = suv_img.TransformContinuousIndexToPhysicalPoint(center_idx)

            projections: List[np.ndarray] = []
            for angle_deg in range(0, 360, 10):
                angle_rad = np.deg2rad(angle_deg)
                transform = sitk.Euler3DTransform(); transform.SetCenter(center_phys); transform.SetRotation(0, 0, angle_rad)

                res = sitk.ResampleImageFilter(); res.SetReferenceImage(suv_img); res.SetTransform(transform)
                res.SetInterpolator(sitk.sitkLinear); res.SetDefaultPixelValue(0.0)
                rot_suv = res.Execute(suv_img)

                rot_mask = sitk.Resample(body, suv_img, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
                rot_suv_masked = sitk.Mask(rot_suv, rot_mask)

                arr = sitk.GetArrayFromImage(rot_suv_masked)  # (Z,Y,X)
                mip = np.max(arr, axis=1)                     # coronal (Z,X)
                projections.append(mip.astype(np.float32))
                QtWidgets.QApplication.processEvents()

            all_vals = np.concatenate([p.flatten() for p in projections]) if projections else np.array([10.0], dtype=np.float32)
            global_stats = {"max": float(np.max(all_vals)),
                            "p95": float(np.percentile(all_vals, 95)),
                            "p98": float(np.percentile(all_vals, 98)),
                            "p99": float(np.percentile(all_vals, 99)),
                            "p997": float(np.percentile(all_vals, 99.7))}

            self.mip_player = RotatingMIPPlayer(projections, global_stats, self)
            self.mip_player.show()
            self._status("MIP ready.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    # ---------- Basic Metrics (safe) ----------
    def _compute_and_save(self):
        mask = self._get_active_mask()
        if mask is None:
            QtWidgets.QMessageBox.warning(self, "Empty ROI", "No ROI array available."); return
        if not np.any(mask):
            QtWidgets.QMessageBox.warning(self, "Empty ROI", "Please draw an ROI first."); return
        try:
            pet = self.pet_suv
            sp = getattr(self, "spacing_zyx", None)
            if sp is None:
                if self.pet_img_sitk_raw is not None:
                    sp = (self.pet_img_sitk_raw.GetSpacing()[2], self.pet_img_sitk_raw.GetSpacing()[1], self.pet_img_sitk_raw.GetSpacing()[0])
                else:
                    sp = (1.0, 1.0, 1.0)
                self.spacing_zyx = sp

            voxels = int(mask.sum())
            vol_mm3 = float(voxels * np.prod(sp))
            vals = pet[mask]; vals = vals[np.isfinite(vals)]
            suvmax = float(vals.max()) if vals.size else 0.0
            suvmean = float(vals.mean()) if vals.size else 0.0
            dims = _calculate_lesion_dimensions(mask.astype(np.uint8), sp)

            results = {"SUVmax": round(suvmax, 3), "SUVmean": round(suvmean, 3), "Vol_ml": round(vol_mm3/1000.0, 3),
                       "Dim_L-R_mm": dims["Dim_Transverse_mm"], "Dim_A-P_mm": dims["Dim_AP_mm"], "Dim_C-C_mm": dims["Dim_CC_mm"]}
            self.metrics_win.update_data(results)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Metrics Error", str(e))


    # ---------- Primary vs Metastatic ROI mode ----------
    def _get_active_mask(self):
        """Return the currently editable mask (primary or active metastatic lesion)."""
        if getattr(self, "pet_suv", None) is None:
            return None
        if getattr(self, "active_roi_class", "primary") == "meta":
            if getattr(self, "meta_masks", None) and 0 <= int(getattr(self, "active_meta_index", 0)) < len(self.meta_masks):
                return self.meta_masks[self.active_meta_index]
            return None
        return self.mask

    def _set_roi_class(self, which: str):
        which = "meta" if str(which).lower().startswith("m") else "primary"
        self.active_roi_class = which

        # create the first metastatic lesion mask lazily
        if which == "meta" and getattr(self, "pet_suv", None) is not None:
            if self.mask is None:
                return
            if not self.meta_masks:
                self.meta_masks.append(np.zeros_like(self.mask, dtype=bool))
            self.active_meta_index = int(np.clip(self.active_meta_index, 0, len(self.meta_masks)-1))

        # update view editable mask pointers
        for v in (self.view_ax, self.view_co, self.view_sa):
            try:
                v.set_active_mask(self.active_roi_class, self.active_meta_index)
            except Exception:
                pass

        self._update_meta_ui()
        try:
            self._refresh_all_views()
        except Exception:
            pass

    def _add_meta_lesion(self):
        if getattr(self, "pet_suv", None) is None or self.mask is None:
            QtWidgets.QMessageBox.warning(self, "No Study Loaded", "Load a PET/CT study first.")
            return
        if not hasattr(self, "meta_masks") or self.meta_masks is None:
            self.meta_masks = []
        self.meta_masks.append(np.zeros_like(self.mask, dtype=bool))
        self.active_roi_class = "meta"
        self.active_meta_index = len(self.meta_masks) - 1
        for v in (self.view_ax, self.view_co, self.view_sa):
            v.set_active_mask(self.active_roi_class, self.active_meta_index)
        self._update_meta_ui()
        self._refresh_all_views()

    def _next_meta_lesion(self):
        if not getattr(self, "meta_masks", None):
            return
        self.active_roi_class = "meta"
        self.active_meta_index = (int(self.active_meta_index) + 1) % len(self.meta_masks)
        for v in (self.view_ax, self.view_co, self.view_sa):
            v.set_active_mask(self.active_roi_class, self.active_meta_index)
        self._update_meta_ui()
        self._refresh_all_views()

    def _delete_meta_lesion(self):
        if not getattr(self, "meta_masks", None):
            return
        idx = int(getattr(self, "active_meta_index", 0))
        if idx < 0 or idx >= len(self.meta_masks):
            return
        # simple confirm
        if QtWidgets.QMessageBox.question(self, "Delete Lesion", f"Delete metastatic lesion {idx+1}?",
                                         QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No) != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.meta_masks.pop(idx)
        if not self.meta_masks:
            # fall back to primary mode
            self.active_roi_class = "primary"
            self.active_meta_index = 0
        else:
            self.active_roi_class = "meta"
            self.active_meta_index = min(idx, len(self.meta_masks)-1)
        for v in (self.view_ax, self.view_co, self.view_sa):
            v.set_active_mask(self.active_roi_class, self.active_meta_index)
        self._update_meta_ui()
        self._refresh_all_views()

    def _update_meta_ui(self):
        # guard if UI not built yet
        if not hasattr(self, "btn_roi_primary"):
            return
        self.btn_roi_primary.setChecked(getattr(self, "active_roi_class", "primary") == "primary")
        self.btn_roi_meta.setChecked(getattr(self, "active_roi_class", "primary") == "meta")

        in_meta = getattr(self, "active_roi_class", "primary") == "meta"
        has_meta = bool(getattr(self, "meta_masks", []))
        for b in (self.btn_meta_add, self.btn_meta_next, self.btn_meta_del, self.lbl_meta):
            b.setVisible(True)
        self.btn_meta_next.setEnabled(in_meta and has_meta and len(self.meta_masks) > 1)
        self.btn_meta_del.setEnabled(in_meta and has_meta)
        if in_meta and has_meta:
            self.lbl_meta.setText(f"Lesion {int(self.active_meta_index)+1}/{len(self.meta_masks)}")
        else:
            self.lbl_meta.setText("")

    # ---------- Tumor burden summary (Primary + Metastases) ----------
        # Refresh dock tabs (scan info / 3D metrics / NHOC-NHOP)
        try:
            for w in getattr(self, '_v21_dock_tab_widgets', []) or []:
                if hasattr(w, 'refresh'):
                    w.refresh()
        except Exception:
            pass

    def _compute_mask_metrics(self, mask: np.ndarray) -> Dict[str, object]:
        sp = self.spacing_zyx if getattr(self, "spacing_zyx", None) is not None else (1.0, 1.0, 1.0)
        vox = int(mask.sum()) if mask is not None else 0
        vox_vol_mm3 = float(np.prod(sp))
        vol_ml = float(vox * vox_vol_mm3 / 1000.0) if vox > 0 else 0.0

        suvmax = suvmean = 0.0
        mtv41_ml = tlg41 = 0.0
        if getattr(self, "pet_suv", None) is not None and vox > 0:
            vals = self.pet_suv[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                suvmax = float(np.max(vals))
                suvmean = float(np.mean(vals))
                thr = 0.41 * suvmax
                mtv_mask = mask & (self.pet_suv >= thr)
                mtv_vox = int(mtv_mask.sum())
                mtv41_ml = float(mtv_vox * vox_vol_mm3 / 1000.0) if mtv_vox > 0 else 0.0
                if mtv_vox > 0:
                    mv = self.pet_suv[mtv_mask]
                    mv = mv[np.isfinite(mv)]
                    tlg41 = float(np.mean(mv) * mtv41_ml) if mv.size else 0.0

        dims = {"Dim_LR_mm": "", "Dim_AP_mm": "", "Dim_CC_mm": ""}
        try:
            d = _calculate_lesion_dimensions(mask.astype(np.uint8), sp)
            dims = {"Dim_LR_mm": d.get("Dim_Transverse_mm", d.get("Dim_L-R_mm", "")),
                    "Dim_AP_mm": d.get("Dim_AP_mm", ""),
                    "Dim_CC_mm": d.get("Dim_CC_mm", "")}
        except Exception:
            pass

        ct_str = ""
        try:
            if getattr(self, "ct_hu", None) is not None and vox > 0:
                cv = self.ct_hu[mask]
                cv = cv[np.isfinite(cv)]
                if cv.size:
                    ct_str = f"{float(np.mean(cv)):.1f}/{float(np.max(cv)):.1f}"
        except Exception:
            ct_str = ""

        return {
            "Vol_ml": round(vol_ml, 3),
            "SUVmax": round(suvmax, 3),
            "SUVmean": round(suvmean, 3),
            "MTV41_ml": round(mtv41_ml, 3),
            "TLG41": round(tlg41, 3),
            "Dim_LR_mm": dims["Dim_LR_mm"],
            "Dim_AP_mm": dims["Dim_AP_mm"],
            "Dim_CC_mm": dims["Dim_CC_mm"],
            "CT_mean_max": ct_str
        }

    def _show_tumor_burden(self):
        if getattr(self, "pet_suv", None) is None or self.mask is None:
            QtWidgets.QMessageBox.warning(self, "No Study Loaded", "Load a PET/CT study first.")
            return
        rows = []

        # Primary
        if np.any(self.mask):
            m = self._compute_mask_metrics(self.mask)
            rows.append({"Lesion": "P1", "Type": "Primary", **m})
        else:
            rows.append({"Lesion": "P1", "Type": "Primary", "Vol_ml": 0, "SUVmax": 0, "SUVmean": 0,
                         "MTV41_ml": 0, "TLG41": 0, "Dim_LR_mm": "", "Dim_AP_mm": "", "Dim_CC_mm": "", "CT_mean_max": ""})

        # Metastases
        meta_rows = []
        if getattr(self, "meta_masks", None):
            for i, mm in enumerate(self.meta_masks, 1):
                if mm is None:
                    continue
                if not np.any(mm):
                    meta_rows.append({"Lesion": f"M{i}", "Type": "Metastasis", "Vol_ml": 0, "SUVmax": 0, "SUVmean": 0,
                                      "MTV41_ml": 0, "TLG41": 0, "Dim_LR_mm": "", "Dim_AP_mm": "", "Dim_CC_mm": "", "CT_mean_max": ""})
                else:
                    m = self._compute_mask_metrics(mm)
                    meta_rows.append({"Lesion": f"M{i}", "Type": "Metastasis", **m})

        rows.extend(meta_rows)

        # Totals (sum of lesions)
        def _sum(k): 
            return float(np.sum([float(r.get(k, 0) or 0) for r in rows if isinstance(r.get(k, 0) or 0, (int, float))]))
        tot = {
            "Vol_ml": round(_sum("Vol_ml"), 3),
            "MTV41_ml": round(_sum("MTV41_ml"), 3),
            "TLG41": round(_sum("TLG41"), 3),
        }
        # For totals, SUVmax = max across lesions, SUVmean = volume-weighted mean across lesions (manual ROI)
        try:
            suvmax_tot = max([float(r.get("SUVmax", 0) or 0) for r in rows]) if rows else 0.0
        except Exception:
            suvmax_tot = 0.0
        try:
            vols = np.array([float(r.get("Vol_ml", 0) or 0) for r in rows], dtype=float)
            suvm = np.array([float(r.get("SUVmean", 0) or 0) for r in rows], dtype=float)
            suvmean_tot = float(np.sum(vols * suvm) / np.sum(vols)) if np.sum(vols) > 0 else 0.0
        except Exception:
            suvmean_tot = 0.0

        rows.append({"Lesion": "TOTAL", "Type": "All lesions", "_is_total": True,
                     "Vol_ml": tot["Vol_ml"], "SUVmax": round(suvmax_tot, 3), "SUVmean": round(suvmean_tot, 3),
                     "MTV41_ml": tot["MTV41_ml"], "TLG41": tot["TLG41"],
                     "Dim_LR_mm": "", "Dim_AP_mm": "", "Dim_CC_mm": "", "CT_mean_max": ""})

        self.burden_win.set_rows(rows)

    # ---------- Radiomics helpers + handlers (bundled yaml support) ----------
    def _get_radiomics_image_mask(self):
        mask = self._get_active_mask()
        if self.pet_suv is None or mask is None: return None, None
        source_mode = self.cmb_rad_source.currentText()
        if source_mode == "PET (SUV)":
            target_arr = self.pet_suv; ref_sitk = self.pet_img_sitk_raw
        else:
            if getattr(self, "ct_image", None) is None or getattr(self, "ct_hu", None) is None: return None, None
            target_arr = self.ct_hu; ref_sitk = self.pet_img_sitk_raw  # geometry match
        img_sitk = sitk.GetImageFromArray(target_arr.astype(np.float32)); img_sitk.CopyInformation(ref_sitk)
        mask_sitk = sitk.GetImageFromArray(self.mask.astype(np.uint8)); mask_sitk.CopyInformation(ref_sitk)
        return img_sitk, mask_sitk

    def _run_radiomics(self, feature_classes, config_file):
        """
        Runs radiomics using settings from the bundled YAML file, 
        but only enables the specific feature_classes requested by the button.
        """
        if not RADIOMICS_AVAILABLE:
            QtWidgets.QMessageBox.information(self, "Radiomics", "PyRadiomics not installed.")
            return

        img, mask = self._get_radiomics_image_mask()
        if img is None or mask is None:
            QtWidgets.QMessageBox.warning(self, "Radiomics", "No valid image/mask for selected source.")
            return

        try:
            # 1. Locate the YAML file (works in .exe and .py)
            param_path = resource_path(config_file)
            if not os.path.exists(param_path):
                 QtWidgets.QMessageBox.critical(self, "Error", f"Config file not found: {param_path}")
                 return

            self._status(f"Loading config: {config_file}...")

            # 2. Initialize Extractor with the YAML file 
            # (This loads settings like binWidth, resampledPixelSpacing, etc.)
            extractor = featureextractor.RadiomicsFeatureExtractor(param_path)

            # 3. Override features: Disable all, then enable only what the button asked for
            extractor.disableAllFeatures()
            for fc in feature_classes:
                extractor.enableFeatureClassByName(fc)

            self._status(f"Radiomics running (Classes: {feature_classes})...")
            
            # 4. Execute
            res = extractor.execute(img, mask)

            out = {}
            # Add manual dimensions if available
            try:
                dims = _calculate_lesion_dimensions(self.mask.astype(np.uint8), self.spacing_zyx)
                out.update(dims)
            except Exception:
                pass

            # Clean up output keys
            for k, v in res.items():
                if "diagnostics" in k: continue
                out[k.replace("original_", "")] = float(v) if isinstance(v, (int, float, np.floating)) else v
            
            self.metrics_win.update_data(out)
            self._status("Radiomics done.")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Radiomics Error", str(e))
            self._status(f"Error: {str(e)}")

    def _on_btn_clinical_click(self): 
        self._run_radiomics(["shape", "firstorder"], "config_clinical.yaml")

    def _on_btn_first_order_click(self): 
        self._run_radiomics(["firstorder"], "config_clinical.yaml")

    def _on_btn_glcm_click(self): 
        self._run_radiomics(["glcm"], "config_texture.yaml")

    def _on_btn_higher_order_click(self): 
        self._run_radiomics(["glrlm", "glszm", "gldm", "ngtdm"], "config_texture.yaml")

    # ---------- Clinical Tools Refresh ----------
    def _refresh_clinical_tabs(self):
        """Refresh PERCIST and RECIST tabs when ROIs change"""
        try:
            # Get the clinical dialog
            dlg = getattr(self, "_v21_clinical_dlg", None)
            if dlg:
                # Refresh PERCIST tab
                percist_tab = getattr(dlg, "percist_tab", None)
                if percist_tab and hasattr(percist_tab, "refresh_dropdown"):
                    percist_tab.refresh_dropdown()
                
                # Refresh RECIST tab
                recist_tab = getattr(dlg, "recist_tab", None)
                if recist_tab and hasattr(recist_tab, "refresh_dropdown"):
                    recist_tab.refresh_dropdown()
        except Exception:
            pass


# --- Region grow helper (safe) ---
def _perform_3d_region_grow(pet_suv: np.ndarray, seed_zyx: Tuple[int, int, int], percent_threshold: float = 41.0) -> np.ndarray:
    z, y, x = seed_zyx
    if not (0 <= z < pet_suv.shape[0] and 0 <= y < pet_suv.shape[1] and 0 <= x < pet_suv.shape[2]):
        return np.zeros(pet_suv.shape, dtype=bool)
    seed_val = pet_suv[z, y, x]
    if seed_val <= 0: return np.zeros(pet_suv.shape, dtype=bool)
    lower_thr = seed_val * (percent_threshold / 100.0)
    img_sitk = sitk.GetImageFromArray(pet_suv.astype(np.float32))
    seed_sitk = (int(x), int(y), int(z))
    seg = sitk.ConnectedThreshold(image1=img_sitk, seedList=[seed_sitk], lower=float(lower_thr), upper=999999999.0, replaceValue=1)
    return sitk.GetArrayFromImage(seg).astype(bool)

def _polyfill_2d(h: int, w: int, xs: List[float], ys: List[float]) -> np.ndarray:
    if xs is None or ys is None or len(xs) < 3 or len(ys) < 3: return np.zeros((h, w), dtype=bool)
    xs = np.asarray(xs, dtype=np.float64); ys = np.asarray(ys, dtype=np.float64)
    if xs[0] != xs[-1] or ys[0] != ys[-1]: xs = np.r_[xs, xs[0]]; ys = np.r_[ys, ys[0]]
    xmin = int(np.floor(xs.min())); xmax = int(np.ceil(xs.max())); ymin = int(np.floor(ys.min())); ymax = int(np.ceil(ys.max()))
    xmin = max(0, min(w - 1, xmin)); xmax = max(0, min(w - 1, xmax)); ymin = max(0, min(h - 1, ymin)); ymax = max(0, min(h - 1, ymax))
    out = np.zeros((h, w), dtype=bool)
    if xmax < xmin or ymax < ymin: return out
    n = len(xs) - 1
    for y in range(ymin, ymax + 1):
        y_f = float(y) + 0.5
        xints: List[float] = []
        for i in range(n):
            x1, y1 = float(xs[i]), float(ys[i]); x2, y2 = float(xs[i + 1]), float(ys[i + 1])
            if (y1 > y_f) != (y2 > y_f):
                denom = (y2 - y1)
                if denom == 0.0: continue
                x = x1 + (y_f - y1) * (x2 - x1) / denom
                xints.append(x)
        if len(xints) < 2: continue
        xints.sort()
        for j in range(0, len(xints) - 1, 2):
            x_start = int(math.floor(xints[j])); x_end = int(math.ceil(xints[j + 1]))
            if x_end < 0 or x_start > (w - 1): continue
            x_start = max(0, x_start); x_end = min(w - 1, x_end)
            if x_end >= x_start: out[y, x_start : x_end + 1] = True
    return out


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = PETCTManualROIApp(None, None)
    win.show()
    sys.exit(app.exec())


# === V21 ADDON: ScanInfo+PERCIST+RECIST+Display+Export (dock tabs) ===
# Appended to preserve original code and functions. Adds new docked tab panels + SUV override logic.

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
except Exception:
    A4 = None

# Global fused PET gamma (used by overridden _pet_to_rgba when gamma not explicitly provided)
_V21_FUSED_PET_GAMMA = 1.0

def _v21_parse_dt_ymd_hms(s: str):
    """Parse 'YYYYMMDD HHMMSS' or ISO 'YYYY-MM-DD HH:MM:SS'."""
    import datetime
    if not s:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        if "-" in s and ":" in s:
            return datetime.datetime.fromisoformat(s)
        parts = s.split()
        d = parts[0]
        t = parts[1] if len(parts) > 1 else "000000"
        t = t.replace(":", "").split(".")[0]
        t = (t + "000000")[:6]
        return datetime.datetime(int(d[0:4]), int(d[4:6]), int(d[6:8]),
                                 int(t[0:2]), int(t[2:4]), int(t[4:6]))
    except Exception:
        return None

def _v21_lbm_janmahasatian(sex: str, weight_kg: float, height_cm: float):
    """Lean body mass (Janmahasatian 2005)."""
    try:
        h_m = float(height_cm) / 100.0
        w = float(weight_kg)
        bmi = w / (h_m*h_m + 1e-9)
        if str(sex).lower().startswith("m"):
            return 9270.0*w/(6680.0 + 216.0*bmi)
        return 9270.0*w/(8780.0 + 244.0*bmi)
    except Exception:
        return None

def _v21_to_kg(val, unit):
    try:
        v = float(val)
        return v * 0.45359237 if str(unit).lower() == "lb" else v
    except Exception:
        return None

def _v21_to_cm(val, unit):
    try:
        v = float(val)
        u = str(unit).lower()
        if u == "cm":
            return v
        if u == "m":
            return v * 100.0
        if u in ("in", "inch", "inches"):
            return v * 2.54
        return v
    except Exception:
        return None

def _v21_to_bq(val, unit):
    """Bq / kBq / MBq / mCi"""
    try:
        v = float(val)
        u = str(unit).lower()
        if u == "bq": return v
        if u == "kbq": return v * 1e3
        if u == "mbq": return v * 1e6
        if u == "mci": return v * 37e6
        return v
    except Exception:
        return None

def _v21_glucose_to_mmolL(val, unit):
    try:
        v = float(val)
        return v/18.0 if str(unit).lower() == "mg/dl" else v
    except Exception:
        return None

def _v21_compute_suvbw_from_bqml(pet_bqml, weight_kg, net_inj_bq, inj_dt, acq_dt, half_life_s):
    """SUVbw from Bq/mL using decay-corrected net injected dose."""
    import numpy as np, math
    if pet_bqml is None:
        return None
    wkg = float(weight_kg)
    dose = float(net_inj_bq)

    if inj_dt and acq_dt and half_life_s and float(half_life_s) > 0:
        dt = (acq_dt - inj_dt).total_seconds()
        dose = dose * math.exp(-math.log(2.0) * dt / float(half_life_s))

    w_g = wkg * 1000.0
    denom = dose / (w_g + 1e-9)  # Bq / g
    return (pet_bqml.astype(np.float32) / (denom + 1e-18)).astype(np.float32)

def _v21_glucose_correct(suv, glucose_mmolL, ref_mmolL=5.0):
    import numpy as np
    try:
        g = float(glucose_mmolL)
        return (suv.astype(np.float32) * (g / float(ref_mmolL))).astype(np.float32)
    except Exception:
        return suv

# ---- Override _pet_to_rgba to support fused gamma without modifying existing calls ----
try:
    _v21_orig_pet_to_rgba = _pet_to_rgba
except Exception:
    _v21_orig_pet_to_rgba = None

def _pet_to_rgba(pet_suv, vmax, alpha, lut, gamma=None, intensity=None, contrast=None):
    """Convert PET SUV slice to an RGBA uint8 image.

    Robust to LUT shape (Nx3 or Nx4) and LUT dtype (0..1 float or 0..255 uint8).
    Applies gamma/intensity/contrast on normalized PET before LUT mapping.
    """
    import numpy as _np

    if gamma is None:
        gamma = _V21_FUSED_PET_GAMMA
    if intensity is None:
        intensity = 1.0
    if contrast is None:
        contrast = 1.0

    # Normalize PET into [0, 1]
    v = _np.asarray(pet_suv, dtype=_np.float32)
    vmax_f = float(vmax) if vmax is not None else float(_np.nanmax(v) if _np.isfinite(v).any() else 1.0)
    vmax_f = max(vmax_f, 1e-6)
    x = _np.clip(v / vmax_f, 0.0, 1.0)

    # Intensity (gain) + contrast around midpoint
    intensity = max(float(intensity), 0.0)
    contrast = max(float(contrast), 0.0)
    x = _np.clip(x * intensity, 0.0, 1.0)
    x = _np.clip((x - 0.5) * contrast + 0.5, 0.0, 1.0)

    # Gamma correction (keep gamma>0)
    g = max(float(gamma), 1e-3)
    x = _np.clip(x ** g, 0.0, 1.0)

    idx = _np.clip((x * 255.0).astype(_np.uint8), 0, 255)

    # Build an RGBA LUT in uint8
    lut_arr = _np.asarray(lut)
    if lut_arr.ndim != 2 or lut_arr.shape[0] < 2 or lut_arr.shape[1] < 3:
        lut_arr = _make_hot_iron_lut(256)

    lut_arr = _np.asarray(lut_arr)
    rgb = lut_arr[:, :3]
    if rgb.dtype.kind in ("f", "c"):
        if _np.nanmax(rgb) <= 1.5:
            rgb = rgb * 255.0
        rgb = _np.clip(rgb, 0.0, 255.0).astype(_np.uint8)
    else:
        rgb = _np.clip(rgb, 0, 255).astype(_np.uint8)

    if lut_arr.shape[1] >= 4:
        a = lut_arr[:, 3]
        if a.dtype.kind in ("f", "c"):
            if _np.nanmax(a) <= 1.5:
                a = a * 255.0
            a = _np.clip(a, 0.0, 255.0).astype(_np.uint8)
        else:
            a = _np.clip(a, 0, 255).astype(_np.uint8)
    else:
        a = _np.full((rgb.shape[0],), 255, dtype=_np.uint8)

    lut_rgba = _np.concatenate([rgb, a[:, None]], axis=1)

    rgba = lut_rgba[idx]  # (H, W, 4) uint8

    # Alpha: only where uptake > 0
    a_mask = (x > 0.0).astype(_np.float32)
    a_out = _np.clip(a_mask * float(alpha) * 255.0, 0.0, 255.0).astype(_np.uint8)
    rgba[..., 3] = a_out

    return rgba

class V21ScanInfoTab(QtWidgets.QWidget):
    """Shows DICOM tags + editable SUV parameters (height, residual dose, glucose)."""
    def __init__(self, app):
        super().__init__()
        self.app = app
        self._build()

    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)

        box = QtWidgets.QGroupBox("SUV correction inputs (editable overrides)")
        grid = QtWidgets.QGridLayout(box)

        r = 0
        grid.addWidget(QtWidgets.QLabel("Weight:"), r, 0)
        self.ed_w = QtWidgets.QLineEdit("")
        self.cb_wu = QtWidgets.QComboBox(); self.cb_wu.addItems(["kg","lb"])
        grid.addWidget(self.ed_w, r, 1); grid.addWidget(self.cb_wu, r, 2)

        grid.addWidget(QtWidgets.QLabel("Height:"), r, 3)
        self.ed_h = QtWidgets.QLineEdit("")
        self.cb_hu = QtWidgets.QComboBox(); self.cb_hu.addItems(["cm","m","in"])
        grid.addWidget(self.ed_h, r, 4); grid.addWidget(self.cb_hu, r, 5)

        r += 1
        grid.addWidget(QtWidgets.QLabel("Injected dose:"), r, 0)
        self.ed_dose = QtWidgets.QLineEdit("")
        self.cb_du = QtWidgets.QComboBox(); self.cb_du.addItems(["MBq","mCi","kBq","Bq"])
        grid.addWidget(self.ed_dose, r, 1); grid.addWidget(self.cb_du, r, 2)

        grid.addWidget(QtWidgets.QLabel("Residual dose:"), r, 3)
        self.ed_res = QtWidgets.QLineEdit("")
        self.cb_ru = QtWidgets.QComboBox(); self.cb_ru.addItems(["MBq","mCi","kBq","Bq"])
        grid.addWidget(self.ed_res, r, 4); grid.addWidget(self.cb_ru, r, 5)

        r += 1
        grid.addWidget(QtWidgets.QLabel("Half-life:"), r, 0)
        self.ed_hl = QtWidgets.QLineEdit("")
        self.cb_hlu = QtWidgets.QComboBox(); self.cb_hlu.addItems(["s","min"])
        grid.addWidget(self.ed_hl, r, 1); grid.addWidget(self.cb_hlu, r, 2)

        grid.addWidget(QtWidgets.QLabel("Fasting glucose:"), r, 3)
        self.ed_glu = QtWidgets.QLineEdit("")
        self.cb_gu = QtWidgets.QComboBox(); self.cb_gu.addItems(["mg/dL","mmol/L"])
        grid.addWidget(self.ed_glu, r, 4); grid.addWidget(self.cb_gu, r, 5)

        r += 1
        grid.addWidget(QtWidgets.QLabel("Injection datetime:"), r, 0)
        self.ed_inj = QtWidgets.QLineEdit(""); self.ed_inj.setPlaceholderText("YYYYMMDD HHMMSS")
        grid.addWidget(self.ed_inj, r, 1, 1, 2)

        grid.addWidget(QtWidgets.QLabel("Acquisition datetime:"), r, 3)
        self.ed_acq = QtWidgets.QLineEdit(""); self.ed_acq.setPlaceholderText("YYYYMMDD HHMMSS")
        grid.addWidget(self.ed_acq, r, 4, 1, 2)

        r += 1
        self.chk_gluc = QtWidgets.QCheckBox("Compute glucose-corrected SUV (SUVgluc) (optional)")
        grid.addWidget(self.chk_gluc, r, 0, 1, 4)

        self.btn_apply = QtWidgets.QPushButton("Apply & recompute SUV")
        self.btn_apply.clicked.connect(self.apply)
        grid.addWidget(self.btn_apply, r, 4, 1, 2)

        lay.addWidget(box)

        # DICOM tag table
        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("DICOM tags"))
        self.ed_search = QtWidgets.QLineEdit(); self.ed_search.setPlaceholderText("Search tag/keyword/value...")
        self.ed_search.textChanged.connect(self._filter)
        h.addWidget(self.ed_search, 1)
        lay.addLayout(h)

        self.tbl = QtWidgets.QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["Modality","Tag","Keyword","Value"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setEditTriggers(_v21_qt_no_edit_triggers())
        lay.addWidget(self.tbl, 1)

        self.refresh()

    def refresh(self):
        self.tbl.setRowCount(0)

        app = getattr(self, "app", None)
        if app is None:
            return

        ds_pt = getattr(app, "pet_ds0", None)
        ds_ct = getattr(app, "ct_ds0", None)

        # Prefer PET for demographics; fall back to CT.
        ds0 = ds_pt or ds_ct
        if ds0 is None:
            self._add('APP', 'Status', 'Scan Info', 'No DICOM metadata loaded yet (load a study first).')
            self.tbl.resizeColumnsToContents()
            return

        def _safe_attr(ds, name, default=""):
            try:
                return getattr(ds, name)
            except Exception:
                return default

        mod0 = "PT" if ds_pt is not None else "CT"

        # --- Patient / Study ---
        self._add(mod0, "PatientName", "Patient", str(_safe_attr(ds0, "PatientName")))
        self._add(mod0, "PatientID", "Patient", str(_safe_attr(ds0, "PatientID")))
        self._add(mod0, "PatientSex", "Patient", str(_safe_attr(ds0, "PatientSex")))
        self._add(mod0, "PatientBirthDate", "Patient", str(_safe_attr(ds0, "PatientBirthDate")))
        self._add(mod0, "StudyDate", "Study", str(_safe_attr(ds0, "StudyDate")))
        self._add(mod0, "StudyTime", "Study", str(_safe_attr(ds0, "StudyTime")))
        self._add(mod0, "AccessionNumber", "Study", str(_safe_attr(ds0, "AccessionNumber")))
        self._add(mod0, "InstitutionName", "Study", str(_safe_attr(ds0, "InstitutionName")))
        self._add(mod0, "ReferringPhysicianName", "Study", str(_safe_attr(ds0, "ReferringPhysicianName")))

        # --- Scanner ---
        self._add(mod0, "Manufacturer", "Scanner", str(_safe_attr(ds0, "Manufacturer")))
        self._add(mod0, "ManufacturerModelName", "Scanner", str(_safe_attr(ds0, "ManufacturerModelName")))

        # --- PET radiopharm (if available) ---
        if ds_pt is not None:
            self._add("PT", "SeriesDescription", "Series", str(_safe_attr(ds_pt, "SeriesDescription")))
            self._add("PT", "SeriesNumber", "Series", str(_safe_attr(ds_pt, "SeriesNumber")))
            self._add("PT", "Units", "PET", str(_safe_attr(ds_pt, "Units")))
            self._add("PT", "DecayCorrection", "PET", str(_safe_attr(ds_pt, "DecayCorrection")))
            self._add("PT", "PatientWeight", "Patient", str(_safe_attr(ds_pt, "PatientWeight")))
            self._add("PT", "PatientSize", "Patient", str(_safe_attr(ds_pt, "PatientSize")))

            try:
                rph = getattr(ds_pt, "RadiopharmaceuticalInformationSequence", None)
                if rph and len(rph) > 0:
                    r0 = rph[0]
                    self._add("PT", "Radiopharmaceutical", "Tracer", str(getattr(r0, "Radiopharmaceutical", "")))
                    self._add("PT", "RadionuclideTotalDose", "Dose", str(getattr(r0, "RadionuclideTotalDose", "")))
                    self._add("PT", "RadionuclideHalfLife", "Dose", str(getattr(r0, "RadionuclideHalfLife", "")))
                    self._add("PT", "RadiopharmaceuticalStartTime", "Dose", str(getattr(r0, "RadiopharmaceuticalStartTime", "")))
            except Exception:
                pass

        # --- CT series info (if available) ---
        if ds_ct is not None:
            self._add("CT", "SeriesDescription", "Series", str(_safe_attr(ds_ct, "SeriesDescription")))
            self._add("CT", "SeriesNumber", "Series", str(_safe_attr(ds_ct, "SeriesNumber")))
            self._add("CT", "KVP", "CT", str(_safe_attr(ds_ct, "KVP")))
            self._add("CT", "SliceThickness", "CT", str(_safe_attr(ds_ct, "SliceThickness")))

        # --- Derived (viewer) ---
        try:
            pet_suv = getattr(app, "pet_suv", None)
            if pet_suv is not None:
                min_val = float(np.nanmin(pet_suv))
                max_val = float(np.nanmax(pet_suv))
                self._add("APP", "SUVRange", "Derived", f"[{min_val:.2f}, {max_val:.2f}]")
        except Exception:
            pass

        self.tbl.resizeColumnsToContents()
    
    def _add(self, modality, tag, keyword, value):
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        self.tbl.setItem(r, 0, QtWidgets.QTableWidgetItem(modality))
        self.tbl.setItem(r, 1, QtWidgets.QTableWidgetItem(tag))
        self.tbl.setItem(r, 2, QtWidgets.QTableWidgetItem(keyword))
        self.tbl.setItem(r, 3, QtWidgets.QTableWidgetItem(str(value)))

    def _filter(self, q):
        q = (q or "").strip().lower()
        for r in range(self.tbl.rowCount()):
            rowtxt = " ".join(self.tbl.item(r,c).text().lower() for c in range(4) if self.tbl.item(r,c))
            self.tbl.setRowHidden(r, bool(q) and (q not in rowtxt))

    def apply(self):
        if getattr(self.app, "pet_bqml", None) is None:
            QtWidgets.QMessageBox.warning(self, "SUV", "Raw PET activity (Bq/mL) not available in this session.")
            return
        wkg = _v21_to_kg(self.ed_w.text().strip(), self.cb_wu.currentText())
        hcm = _v21_to_cm(self.ed_h.text().strip(), self.cb_hu.currentText())
        dose_bq = _v21_to_bq(self.ed_dose.text().strip(), self.cb_du.currentText())
        res_bq = _v21_to_bq(self.ed_res.text().strip() or "0", self.cb_ru.currentText()) or 0.0
        try:
            net_bq = float(dose_bq) - float(res_bq)
            if net_bq <= 0:
                net_bq = float(dose_bq)
        except Exception:
            net_bq = dose_bq

        try:
            hl_s = float(self.ed_hl.text().strip() or "0")
            if self.cb_hlu.currentText() == "min":
                hl_s *= 60.0
            if hl_s <= 0:
                hl_s = None
        except Exception:
            hl_s = None

        inj_dt = _v21_parse_dt_ymd_hms(self.ed_inj.text().strip())
        acq_dt = _v21_parse_dt_ymd_hms(self.ed_acq.text().strip())

        glu = _v21_glucose_to_mmolL(self.ed_glu.text().strip(), self.cb_gu.currentText())
        do_gluc = self.chk_gluc.isChecked()

        if wkg is None or dose_bq is None:
            QtWidgets.QMessageBox.warning(self, "SUV", "Please enter valid weight and injected dose.")
            return

        suv = _v21_compute_suvbw_from_bqml(self.app.pet_bqml, wkg, net_bq, inj_dt, acq_dt, hl_s)
        if do_gluc and glu is not None:
            suv = _v21_glucose_correct(suv, glu, ref_mmolL=5.0)

        # Store overrides for PERCIST
        self.app.suv_overrides = dict(weight_kg=wkg, height_cm=hcm, dose_bq=dose_bq, residual_bq=res_bq,
                                      net_dose_bq=net_bq, half_life_s=hl_s, inj_dt=inj_dt, acq_dt=acq_dt,
                                      glucose_mmolL=glu, glucose_corr=do_gluc)

        self.app.pet_suv = suv

        # Refresh views
        try:
            for v in (getattr(self.app, "view_ax", None), getattr(self.app, "view_co", None), getattr(self.app, "view_sa", None)):
                if v is None: 
                    continue
                v.set_volumes(self.app.ct_hu, self.app.pet_suv, self.app.mask, self.app.spacing_zyx,
                              mask_primary=getattr(self.app, "mask_primary", None),
                              meta_masks=getattr(self.app, "meta_masks", None),
                              active_meta_index=getattr(self.app, "active_meta_index", 0),
                              active_roi_class=getattr(self.app, "active_roi_class", "primary"))
                v.update_view()
        except Exception:
            pass

        QtWidgets.QMessageBox.information(self, "SUV", "SUV recalculated and applied.")

class V21PercistTab(QtWidgets.QWidget):
    """PERCIST (simplified) helper.

    - Liver background ROI placed at the current crosshair (prefer fused view)
    - Allows collecting multiple lesions (>=5) and lists SULpeak for each
    """
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.liver_center_zyx = None
        self._build()

    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)

        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(QtWidgets.QLabel("Sex:"))
        self.cb_sex = QtWidgets.QComboBox()
        self.cb_sex.addItems(["Male", "Female"])
        row1.addWidget(self.cb_sex)

        row1.addWidget(QtWidgets.QLabel("Weight (kg):"))
        self.sp_wt = QtWidgets.QDoubleSpinBox()
        self.sp_wt.setRange(1, 500)
        self.sp_wt.setDecimals(1)
        self.sp_wt.setValue(70.0)
        row1.addWidget(self.sp_wt)

        row1.addWidget(QtWidgets.QLabel("Height (cm):"))
        self.sp_ht = QtWidgets.QDoubleSpinBox()
        self.sp_ht.setRange(30, 250)
        self.sp_ht.setDecimals(0)
        self.sp_ht.setValue(170.0)
        row1.addWidget(self.sp_ht)

        row1.addStretch(1)
        lay.addLayout(row1)

        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Liver ROI diameter (mm):"))
        self.sp_liver_diam = QtWidgets.QDoubleSpinBox()
        self.sp_liver_diam.setRange(5, 80)
        self.sp_liver_diam.setDecimals(0)
        self.sp_liver_diam.setValue(30.0)
        row2.addWidget(self.sp_liver_diam)

        row2.addWidget(QtWidgets.QLabel("Lesion:"))
        self.cb_lesion = QtWidgets.QComboBox()
        row2.addWidget(self.cb_lesion, 2)

        self.btn_set_liver = QtWidgets.QPushButton("Set Liver ROI at current crosshair")
        self.btn_add = QtWidgets.QPushButton("Add selected lesion")
        self.btn_del = QtWidgets.QPushButton("Delete selected row")
        self.btn_compute = QtWidgets.QPushButton("Compute PERCIST summary")
        row2.addWidget(self.btn_set_liver)
        row2.addWidget(self.btn_add)
        row2.addWidget(self.btn_del)
        row2.addWidget(self.btn_compute)
        lay.addLayout(row2)

        self.tbl = QtWidgets.QTableWidget(0, 5)
        self.tbl.setHorizontalHeaderLabels(["#", "Lesion", "SULpeak", "Measurable", "Notes"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSelectionBehavior(_v21_qt_select_rows())
        self.tbl.setEditTriggers(_v21_qt_edit_triggers('DoubleClicked','EditKeyPressed'))
        lay.addWidget(self.tbl, 1)

        self.lbl = QtWidgets.QLabel("Place liver ROI using the crosshair, then add lesions (often up to 5).")
        self.lbl.setStyleSheet("color: #999;")
        lay.addWidget(self.lbl)

        self.btn_set_liver.clicked.connect(self.set_liver)
        self.btn_add.clicked.connect(self.add_selected_lesion)
        self.btn_del.clicked.connect(self.delete_selected_row)
        self.btn_compute.clicked.connect(self.compute_summary)

        self.refresh_dropdown()

    def refresh_dropdown(self):
        self.cb_lesion.clear()
        items = []
        
        # Get primary mask - use self.app.mask (not self.app.mask_primary)
        if hasattr(self.app, "mask") and self.app.mask is not None:
            # Check if primary has any active voxels
            if np.any(self.app.mask):
                items.append(("Primary", "Primary"))
        
        # Get metastatic masks - use self.app.meta_masks (not self.app.mask_meta_list)
        metas = getattr(self.app, "meta_masks", None)
        if metas:
            for i, m in enumerate(metas, 1):
                if m is not None and np.any(m):
                    items.append((f"Metastasis #{i}", f"Metastasis #{i}"))
        
        # Add option for active ROI (if currently drawing)
        active_class = getattr(self.app, "active_roi_class", "primary")
        active_idx = getattr(self.app, "active_meta_index", 0)
        
        if active_class == "meta" and metas and 0 <= active_idx < len(metas):
            active_label = f"Current Active (Metastasis #{active_idx+1})"
            active_key = f"Active_#{active_idx}"
            # Add at the beginning for easy access
            items.insert(0, (active_label, active_key))
        elif active_class == "primary":
            active_label = "Current Active (Primary)"
            active_key = "Active_Primary"
            items.insert(0, (active_label, active_key))
        
        for label, key in items:
            self.cb_lesion.addItem(label, key)
    
    def _mask_for_key(self, key):
        if key == "Primary":
            return getattr(self.app, "mask", None)
        elif key == "Active_Primary":
            return getattr(self.app, "mask", None)
        elif key.startswith("Active_#"):
            try:
                idx = int(key.split("#")[1].strip())
                metas = getattr(self.app, "meta_masks", None) or []
                if 0 <= idx < len(metas):
                    return metas[idx]
            except Exception:
                return None
        elif key.startswith("Metastasis #"):
            try:
                idx = int(key.split("#")[1].strip()) - 1
                metas = getattr(self.app, "meta_masks", None) or []
                if 0 <= idx < len(metas):
                    return metas[idx]
            except Exception:
                return None
        return None

    def _sul_factor(self):
        sex = self.cb_sex.currentText().strip().lower()
        wt = float(self.sp_wt.value())
        ht = float(self.sp_ht.value())
        try:
            lbm = _v21_lbm_janmahasatian(sex, wt, ht)
        except Exception:
            lbm = wt  # fallback
        if wt <= 0:
            return 1.0
        return float(lbm) / wt

    def _sphere_mask(self, center_zyx, radius_mm):
        import numpy as np
        sp = getattr(self.app, "spacing_zyx", None)
        if sp is None:
            return None
        spz, spy, spx = float(sp[0]), float(sp[1]), float(sp[2])
        rz = max(1, int(math.ceil(radius_mm / spz)))
        ry = max(1, int(math.ceil(radius_mm / spy)))
        rx = max(1, int(math.ceil(radius_mm / spx)))

        z0, y0, x0 = [int(v) for v in center_zyx]
        pet = getattr(self.app, "pet_suv", None)
        if pet is None or getattr(pet, "shape", None) is None:
            return None
        Z, Y, X = pet.shape

        zmin, zmax = max(0, z0 - rz), min(Z - 1, z0 + rz)
        ymin, ymax = max(0, y0 - ry), min(Y - 1, y0 + ry)
        xmin, xmax = max(0, x0 - rx), min(X - 1, x0 + rx)

        zz = np.arange(zmin, zmax + 1)[:, None, None]
        yy = np.arange(ymin, ymax + 1)[None, :, None]
        xx = np.arange(xmin, xmax + 1)[None, None, :]

        dz2 = ((zz - z0) * spz) ** 2
        dy2 = ((yy - y0) * spy) ** 2
        dx2 = ((xx - x0) * spx) ** 2
        dist2 = dz2 + dy2 + dx2
        return (dist2 <= (radius_mm ** 2)), (zmin, ymin, xmin)

    def set_liver(self):
        # Prefer the fused-view crosshair; we store last crosshair in app.cross_z/y/x
        z = getattr(self.app, "cross_z", None)
        y = getattr(self.app, "cross_y", None)
        x = getattr(self.app, "cross_x", None)

        # Fallbacks: some sessions may not have propagated the app-level crosshair yet.
        if z is None or y is None or x is None:
            try:
                # Prefer axial view's current crosshair if available
                vax = getattr(self.app, "view_ax", None)
                if vax is not None:
                    z = getattr(vax, "cross_z", None)
                    y = getattr(vax, "cross_y", None)
                    x = getattr(vax, "cross_x", None)
            except Exception:
                pass

        if z is None or y is None or x is None:
            try:
                vol = getattr(self.app, "pet_suv", None) or getattr(self.app, "ct_hu", None)
                if vol is not None and getattr(vol, "ndim", 0) >= 3:
                    z = int(vol.shape[0] // 2)
                    y = int(vol.shape[1] // 2)
                    x = int(vol.shape[2] // 2)
                    # propagate so future clicks/tools share the same fields
                    try:
                        self.app._on_crosshair_changed(int(z), int(y), int(x))
                    except Exception:
                        self.app.cross_z, self.app.cross_y, self.app.cross_x = int(z), int(y), int(x)
            except Exception:
                pass

        if z is None or y is None or x is None:
            QtWidgets.QMessageBox.warning(
                self, "PERCIST",
                "Crosshair not available. Click inside the fused view first (or move the crosshair), then try again."
            )
            return

        self.liver_center_zyx = (int(z), int(y), int(x))
        QtWidgets.QMessageBox.information(self, "PERCIST", f"Liver ROI center set at (z,y,x)=({int(z)},{int(y)},{int(x)}).")
    
    def _liver_stats(self):
        import numpy as np
        if self.liver_center_zyx is None:
            return None
        diam = float(self.sp_liver_diam.value())
        radius_mm = max(1.0, diam / 2.0)

        pet = getattr(self.app, "pet_suv", None)
        if pet is None:
            return None
        fac = self._sul_factor()
        sul = pet.astype(np.float32) * fac

        sm = self._sphere_mask(self.liver_center_zyx, radius_mm)
        if sm is None:
            return None
        mask_local, (zmin, ymin, xmin) = sm
        slab = sul[zmin:zmin + mask_local.shape[0], ymin:ymin + mask_local.shape[1], xmin:xmin + mask_local.shape[2]]
        vals = slab[mask_local]
        if vals.size < 10:
            return None
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        return mu, sd

    def _sulpeak_for_lesion(self, lesion_mask):
        import numpy as np
        pet = getattr(self.app, "pet_suv", None)
        if pet is None or lesion_mask is None:
            return None
        if lesion_mask.shape != pet.shape:
            return None
        m = lesion_mask > 0
        if m.sum() == 0:
            return None

        fac = self._sul_factor()
        sul = pet.astype(np.float32) * fac

        # Find max SUL voxel within lesion
        sul_in = np.where(m, sul, -1e9)
        flat = int(np.argmax(sul_in))
        z, y, x = np.unravel_index(flat, sul.shape)

        # 1 cm^3 sphere => 1000 mm^3
        V = 1000.0
        radius_mm = (3.0 * V / (4.0 * math.pi)) ** (1.0 / 3.0)  # ~6.2 mm

        sm = self._sphere_mask((z, y, x), radius_mm)
        if sm is None:
            return None
        mask_local, (zmin, ymin, xmin) = sm
        slab = sul[zmin:zmin + mask_local.shape[0], ymin:ymin + mask_local.shape[1], xmin:xmin + mask_local.shape[2]]
        # keep inside lesion (optional but safer)
        lesion_slab = m[zmin:zmin + mask_local.shape[0], ymin:ymin + mask_local.shape[1], xmin:xmin + mask_local.shape[2]]
        use = mask_local & lesion_slab
        if use.sum() < 5:
            use = mask_local  # fallback
        return float(np.mean(slab[use]))

    def add_selected_lesion(self):
        if self.liver_center_zyx is None:
            QtWidgets.QMessageBox.warning(self, "PERCIST", "Set Liver ROI at current crosshair first.")
            return
        self.refresh_dropdown()
        key = self.cb_lesion.currentData()
        if not key:
            QtWidgets.QMessageBox.warning(self, "PERCIST", "No lesion ROI available yet.")
            return
        lesion = self._mask_for_key(key)
        if lesion is None:
            QtWidgets.QMessageBox.warning(self, "PERCIST", "Selected lesion mask not found.")
            return

        sulpeak = self._sulpeak_for_lesion(lesion)
        if sulpeak is None:
            QtWidgets.QMessageBox.warning(self, "PERCIST", "Could not compute SULpeak for this lesion.")
            return

        liver = self._liver_stats()
        if liver is None:
            QtWidgets.QMessageBox.warning(self, "PERCIST", "Liver stats unavailable (check crosshair location & diameter).")
            return
        mu, sd = liver
        thresh = 1.5 * mu + 2.0 * sd
        measurable = "Yes" if sulpeak >= thresh else "No"

        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        it0 = QtWidgets.QTableWidgetItem(str(r + 1)); it0.setFlags(it0.flags() & ~QtCore.Qt.ItemIsEditable)
        it1 = QtWidgets.QTableWidgetItem(str(key)); it1.setFlags(it1.flags() & ~QtCore.Qt.ItemIsEditable)
        it2 = QtWidgets.QTableWidgetItem(f"{float(sulpeak):.3f}"); it2.setFlags(it2.flags() & ~QtCore.Qt.ItemIsEditable)
        it3 = QtWidgets.QTableWidgetItem(measurable); it3.setFlags(it3.flags() & ~QtCore.Qt.ItemIsEditable)
        it4 = QtWidgets.QTableWidgetItem("")
        self.tbl.setItem(r, 0, it0)
        self.tbl.setItem(r, 1, it1)
        self.tbl.setItem(r, 2, it2)
        self.tbl.setItem(r, 3, it3)
        self.tbl.setItem(r, 4, it4)

        # Persist results
        self.app.percist_results = self._collect_results()

    def delete_selected_row(self):
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()}, reverse=True)
        for r in rows:
            self.tbl.removeRow(r)
        for r in range(self.tbl.rowCount()):
            it = self.tbl.item(r, 0)
            if it:
                it.setText(str(r + 1))
        self.app.percist_results = self._collect_results()

    def _collect_results(self):
        out = {"liver": None, "threshold": None, "lesions": []}
        liver = self._liver_stats()
        if liver is None:
            return out
        mu, sd = liver
        out["liver"] = {"sulmean": mu, "sulsd": sd}
        out["threshold"] = 1.5 * mu + 2.0 * sd
        for r in range(self.tbl.rowCount()):
            k = self.tbl.item(r, 1).text() if self.tbl.item(r, 1) else ""
            sp = float(self.tbl.item(r, 2).text()) if self.tbl.item(r, 2) else float("nan")
            meas = self.tbl.item(r, 3).text() if self.tbl.item(r, 3) else ""
            out["lesions"].append({"lesion": k, "sulpeak": sp, "measurable": meas})
        return out

    def compute_summary(self):
        res = self._collect_results()
        liver = res.get("liver", None)
        lesions = res.get("lesions", [])
        if liver is None:
            QtWidgets.QMessageBox.warning(self, "PERCIST", "Liver stats unavailable. Set Liver ROI at crosshair first.")
            return
        mu = liver["sulmean"]; sd = liver["sulsd"]
        thresh = res.get("threshold", None)

        # Sort lesions by SULpeak, take top 5
        peaks = [d for d in lesions if isinstance(d.get("sulpeak", None), (int, float))]
        peaks = [d for d in peaks if math.isfinite(d["sulpeak"])]
        peaks.sort(key=lambda d: d["sulpeak"], reverse=True)
        top5 = peaks[:5]
        txt = f"Liver SULmean±SD: {mu:.3f} ± {sd:.3f}\nThreshold (1.5×mean + 2×SD): {thresh:.3f}\n"
        txt += f"Lesions in table: {len(lesions)} (Top5 used: {len(top5)})\n"
        if top5:
            txt += "Top lesions (SULpeak):\n" + "\n".join([f"  - {d['lesion']}: {d['sulpeak']:.3f} ({d.get('measurable','')})" for d in top5])
        QtWidgets.QMessageBox.information(self, "PERCIST", txt)

        self.app.percist_results = res

    def on_show(self):
        """Called when tab is shown - refresh dropdown"""
        self.refresh_dropdown()

class V21RecistTab(QtWidgets.QWidget):
    """RECIST 1.1 target lesion manager (ROI-based)."""
    def __init__(self, app):
        super().__init__()
        self.app = app
        self._baseline_sum = None
        self._build()

    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)

        title = QtWidgets.QLabel("<b>RECIST 1.1</b> — target lesion manager (ROI-based)")
        lay.addWidget(title)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Lesion:"))
        self.cb_lesion = QtWidgets.QComboBox()
        top.addWidget(self.cb_lesion, 2)

        top.addWidget(QtWidgets.QLabel("Entry type:"))
        self.cb_entry_type = QtWidgets.QComboBox()
        self.cb_entry_type.addItems(["Target lesion (long axis)", "Lymph node (short axis)"])
        top.addWidget(self.cb_entry_type, 2)

        self.btn_add = QtWidgets.QPushButton("Add selected")
        self.btn_del = QtWidgets.QPushButton("Delete selected row")
        self.btn_base = QtWidgets.QPushButton("Set Baseline")
        self.btn_eval = QtWidgets.QPushButton("Evaluate")
        top.addWidget(self.btn_add)
        top.addWidget(self.btn_del)
        top.addWidget(self.btn_base)
        top.addWidget(self.btn_eval)
        lay.addLayout(top)

        # Table
        self.tbl = QtWidgets.QTableWidget(0, 8)
        self.tbl.setHorizontalHeaderLabels([
            "#", "Entry", "Lesion", "Long axis (mm)", "Short axis (mm)", "Used (mm)", "Included", "Notes"
        ])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setSelectionBehavior(_v21_qt_select_rows())
        self.tbl.setEditTriggers(_v21_qt_edit_triggers('DoubleClicked','EditKeyPressed'))
        lay.addWidget(self.tbl, 1)

        bot = QtWidgets.QHBoxLayout()
        self.lbl_sum = QtWidgets.QLabel("Sum of diameters: 0.0 mm")
        self.lbl_hint = QtWidgets.QLabel("Tip: RECIST commonly uses up to 5 target lesions (max 2 per organ).")
        self.lbl_hint.setStyleSheet("color: #999;")
        bot.addWidget(self.lbl_sum)
        bot.addStretch(1)
        bot.addWidget(self.lbl_hint)
        lay.addLayout(bot)

        self.btn_add.clicked.connect(self.add_selected)
        self.btn_del.clicked.connect(self.delete_selected)
        self.btn_base.clicked.connect(self.set_baseline)
        self.btn_eval.clicked.connect(self.evaluate)

        self.refresh_dropdown()

    def refresh_dropdown(self):
        self.cb_lesion.clear()
        items = []
        
        # Get primary mask
        if hasattr(self.app, "mask") and self.app.mask is not None:
            if np.any(self.app.mask):
                items.append(("Primary", "Primary"))
        
        # Get metastatic masks
        metas = getattr(self.app, "meta_masks", None)
        if metas:
            for i, m in enumerate(metas, 1):
                if m is not None and np.any(m):
                    items.append((f"Metastasis #{i}", f"Metastasis #{i}"))
        
        # Add option for active ROI
        active_class = getattr(self.app, "active_roi_class", "primary")
        active_idx = getattr(self.app, "active_meta_index", 0)
        
        if active_class == "meta" and metas and 0 <= active_idx < len(metas):
            active_label = f"Current Active (Metastasis #{active_idx+1})"
            active_key = f"Active_#{active_idx}"
            items.insert(0, (active_label, active_key))
        elif active_class == "primary":
            active_label = "Current Active (Primary)"
            active_key = "Active_Primary"
            items.insert(0, (active_label, active_key))
        
        for label, key in items:
            self.cb_lesion.addItem(label, key)
    
    def _mask_for_key(self, key):
        if key == "Primary":
            return getattr(self.app, "mask", None)
        elif key == "Active_Primary":
            return getattr(self.app, "mask", None)
        elif key.startswith("Active_#"):
            try:
                idx = int(key.split("#")[1].strip())
                metas = getattr(self.app, "meta_masks", None) or []
                if 0 <= idx < len(metas):
                    return metas[idx]
            except Exception:
                return None
        elif key.startswith("Metastasis #"):
            try:
                idx = int(key.split("#")[1].strip()) - 1
                metas = getattr(self.app, "meta_masks", None) or []
                if 0 <= idx < len(metas):
                    return metas[idx]
            except Exception:
                return None
        return None

    def _axes_mm_for_mask(self, mask, spacing_zyx):
        """Return (long_mm, short_mm) using maximal-axial-area slice bbox."""
        import numpy as np
        if mask is None:
            return 0.0, 0.0
        spz, spy, spx = float(spacing_zyx[0]), float(spacing_zyx[1]), float(spacing_zyx[2])

        # Ensure boolean
        m = (mask > 0)
        if m.ndim != 3 or m.sum() == 0:
            return 0.0, 0.0

        # Choose axial slice (z) with maximal area
        areas = m.sum(axis=(1, 2))
        z = int(np.argmax(areas))
        sl = m[z]
        ys, xs = np.where(sl)
        if ys.size == 0:
            return 0.0, 0.0

        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())

        dy_mm = (y1 - y0 + 1) * spy
        dx_mm = (x1 - x0 + 1) * spx

        long_mm = float(max(dx_mm, dy_mm))
        short_mm = float(min(dx_mm, dy_mm))

        return long_mm, short_mm

    def add_selected(self):
        self.refresh_dropdown()
        key = self.cb_lesion.currentData()
        if not key:
            QtWidgets.QMessageBox.warning(self, "RECIST", "No lesion ROI available yet. Draw Primary/Metastasis ROIs first.")
            return
        mask = self._mask_for_key(key)
        if mask is None:
            QtWidgets.QMessageBox.warning(self, "RECIST", "Selected lesion mask not found.")
            return

        spacing = getattr(self.app, "spacing_zyx", (1.0, 1.0, 1.0))
        long_mm, short_mm = self._axes_mm_for_mask(mask, spacing)

        entry = self.cb_entry_type.currentText()
        use_mm = short_mm if "Lymph node" in entry else long_mm

        r = self.tbl.rowCount()
        self.tbl.insertRow(r)

        def _set_item(c, txt, editable=True):
            it = QtWidgets.QTableWidgetItem(txt)
            if not editable:
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
            self.tbl.setItem(r, c, it)

        _set_item(0, str(r + 1), editable=False)
        _set_item(1, "Node" if "Lymph node" in entry else "Lesion", editable=False)
        _set_item(2, str(key), editable=False)
        _set_item(3, f"{long_mm:.1f}", editable=False)
        _set_item(4, f"{short_mm:.1f}", editable=False)
        _set_item(5, f"{use_mm:.1f}", editable=False)

        # Included checkbox
        chk = QtWidgets.QTableWidgetItem("")
        chk.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        chk.setCheckState(QtCore.Qt.Checked)
        self.tbl.setItem(r, 6, chk)

        _set_item(7, "", editable=True)

        self._recalc_sum()

    def delete_selected(self):
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()}, reverse=True)
        for r in rows:
            self.tbl.removeRow(r)
        # Renumber
        for r in range(self.tbl.rowCount()):
            it = self.tbl.item(r, 0)
            if it:
                it.setText(str(r + 1))
        self._recalc_sum()

    def _recalc_sum(self):
        s = 0.0
        for r in range(self.tbl.rowCount()):
            inc = self.tbl.item(r, 6)
            if inc is not None and inc.checkState() != QtCore.Qt.Checked:
                continue
            val_it = self.tbl.item(r, 5)
            if val_it is None:
                continue
            try:
                s += float(val_it.text())
            except Exception:
                pass

        self.lbl_sum.setText(f"Sum of diameters: {s:.1f} mm")
        self.app.recist_results = {"sum_mm": s, "n_rows": self.tbl.rowCount()}

    def set_baseline(self):
        self._recalc_sum()
        rr = getattr(self.app, "recist_results", None) or {}
        self._baseline_sum = float(rr.get("sum_mm", 0.0))
        QtWidgets.QMessageBox.information(self, "RECIST", f"Baseline saved: {self._baseline_sum:.1f} mm")

    def evaluate(self):
        self._recalc_sum()
        rr = getattr(self.app, "recist_results", None) or {}
        if self._baseline_sum is None:
            QtWidgets.QMessageBox.warning(self, "RECIST", "Set baseline first.")
            return
        b = float(self._baseline_sum)
        c = float(rr.get("sum_mm", 0.0))
        if b <= 0:
            QtWidgets.QMessageBox.warning(self, "RECIST", "Baseline sum is 0.")
            return

        pct = (c - b) / b * 100.0
        resp = "SD"
        # Simplified RECIST (requires non-target lesions/new lesions clinically)
        if c == 0:
            resp = "CR"
        elif pct <= -30.0:
            resp = "PR"
        elif pct >= 20.0 and (c - b) >= 5.0:
            resp = "PD"
        else:
            resp = "SD"

        QtWidgets.QMessageBox.information(
            self, "RECIST",
            f"Baseline: {b:.1f} mm\nCurrent: {c:.1f} mm\nChange: {pct:+.1f}%\nResponse: {resp}"
        )

    def on_show(self):
        """Called when tab is shown - refresh dropdown"""
        self.refresh_dropdown()

class V21DisplayTab(QtWidgets.QWidget):
    """Display adjustments (CT windowing; fused PET gamma/vmax/opacity)"""
    def __init__(self, app):
        super().__init__()
        self.app = app
        self._build()

    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)

        ctbox = QtWidgets.QGroupBox("CT window (WL/WW)")
        g = QtWidgets.QGridLayout(ctbox)
        g.addWidget(QtWidgets.QLabel("Preset:"), 0, 0)
        self.cb_preset = QtWidgets.QComboBox()
        self.cb_preset.addItems(["Soft tissue (40/400)", "Lung (-600/1500)", "Bone (300/1500)", "Brain (40/80)"])
        g.addWidget(self.cb_preset, 0, 1, 1, 2)
        self.btn_apply_preset = QtWidgets.QPushButton("Apply")
        g.addWidget(self.btn_apply_preset, 0, 3)

        g.addWidget(QtWidgets.QLabel("Brightness (WL):"), 1, 0)
        self.sl_wl = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.sl_wl.setRange(-1000, 1000); self.sl_wl.setValue(40)
        self.lbl_wl = QtWidgets.QLabel("40")
        g.addWidget(self.sl_wl, 1, 1, 1, 2); g.addWidget(self.lbl_wl, 1, 3)

        g.addWidget(QtWidgets.QLabel("Contrast (WW):"), 2, 0)
        self.sl_ww = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); self.sl_ww.setRange(1, 4000); self.sl_ww.setValue(400)
        self.lbl_ww = QtWidgets.QLabel("400")
        g.addWidget(self.sl_ww, 2, 1, 1, 2); g.addWidget(self.lbl_ww, 2, 3)

        lay.addWidget(ctbox)

        petbox = QtWidgets.QGroupBox("Fused PET display")
        gp = QtWidgets.QGridLayout(petbox)

        gp.addWidget(QtWidgets.QLabel("PET Vmax:"), 0, 0)
        self.lbl_vmax = QtWidgets.QLabel("")
        gp.addWidget(self.lbl_vmax, 0, 3)
        self.sl_vmax = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_vmax.setRange(2, 50); self.sl_vmax.setValue(15)
        gp.addWidget(self.sl_vmax, 0, 1, 1, 2)

        gp.addWidget(QtWidgets.QLabel("Gamma:"), 1, 0)
        self.lbl_gamma = QtWidgets.QLabel("")
        gp.addWidget(self.lbl_gamma, 1, 3)
        self.sl_gamma = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_gamma.setRange(4, 30); self.sl_gamma.setValue(10)
        gp.addWidget(self.sl_gamma, 1, 1, 1, 2)

        gp.addWidget(QtWidgets.QLabel("Intensity:"), 2, 0)
        self.lbl_intensity = QtWidgets.QLabel("")
        gp.addWidget(self.lbl_intensity, 2, 3)
        self.sl_intensity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_intensity.setRange(10, 300); self.sl_intensity.setValue(100)
        gp.addWidget(self.sl_intensity, 2, 1, 1, 2)

        gp.addWidget(QtWidgets.QLabel("Contrast:"), 3, 0)
        self.lbl_contrast = QtWidgets.QLabel("")
        gp.addWidget(self.lbl_contrast, 3, 3)
        self.sl_contrast = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_contrast.setRange(10, 300); self.sl_contrast.setValue(100)
        gp.addWidget(self.sl_contrast, 3, 1, 1, 2)

        gp.addWidget(QtWidgets.QLabel("Opacity:"), 4, 0)
        self.sl_alpha = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_alpha.setRange(0, 100); self.sl_alpha.setValue(55)
        self.lbl_alpha = QtWidgets.QLabel("")
        gp.addWidget(self.sl_alpha, 4, 1, 1, 2); gp.addWidget(self.lbl_alpha, 4, 3)

        # set initial label text
        self.lbl_vmax.setText(f"{self.sl_vmax.value():.1f}")
        self.lbl_gamma.setText(f"{self.sl_gamma.value()/10.0:.1f}")
        self.lbl_intensity.setText(f"{self.sl_intensity.value()/100.0:.2f}")
        self.lbl_contrast.setText(f"{self.sl_contrast.value()/100.0:.2f}")
        self.lbl_alpha.setText(f"{self.sl_alpha.value()/100.0:.2f}")

        lay.addWidget(petbox)
        lay.addStretch(1)
        # signals
        self.sl_wl.valueChanged.connect(lambda v: (self.lbl_wl.setText(str(v)), self.apply_ct()))
        self.sl_ww.valueChanged.connect(lambda v: (self.lbl_ww.setText(str(v)), self.apply_ct()))
        self.btn_apply_preset.clicked.connect(self.apply_preset)

        self.sl_vmax.valueChanged.connect(lambda v: (self.lbl_vmax.setText(str(v)), self.apply_pet()))
        self.sl_gamma.valueChanged.connect(lambda v: (self.lbl_gamma.setText(f"{v/10.0:.1f}"), self.apply_pet()))
        self.sl_intensity.valueChanged.connect(lambda v: (self.lbl_intensity.setText(f"{v/100.0:.2f}"), self.apply_pet()))
        self.sl_contrast.valueChanged.connect(lambda v: (self.lbl_contrast.setText(f"{v/100.0:.2f}"), self.apply_pet()))
        self.sl_alpha.valueChanged.connect(lambda v: (self.lbl_alpha.setText(f"{v/100.0:.2f}"), self.apply_pet()))

    def on_show(self):
        # read existing values from axial view if possible
        v = getattr(self.app, "view_ax", None)
        if v is not None:
            try:
                self.sl_wl.setValue(int(round(v.ct_wl)))
                self.sl_ww.setValue(int(round(v.ct_ww)))
                self.sl_vmax.setValue(int(round(v.pet_vmax)))
                self.sl_alpha.setValue(int(round(v.pet_alpha*100.0)))
            except Exception:
                pass

    def apply_preset(self):
        txt = self.cb_preset.currentText()
        if "40/400" in txt: wl, ww = 40, 400
        elif "-600/1500" in txt: wl, ww = -600, 1500
        elif "300/1500" in txt: wl, ww = 300, 1500
        else: wl, ww = 40, 80
        self.sl_wl.setValue(int(wl)); self.sl_ww.setValue(int(ww))
        self.apply_ct()

    def apply_ct(self):
        wl = float(self.sl_wl.value())
        ww = float(self.sl_ww.value())
        for v in (getattr(self.app, "view_ax", None), getattr(self.app, "view_co", None), getattr(self.app, "view_sa", None)):
            if v is None: 
                continue
            v.ct_wl = wl; v.ct_ww = ww
            try: v.update_view()
            except Exception: pass

    def apply_pet(self):
        global _V21_FUSED_PET_GAMMA
        gamma = float(self.sl_gamma.value()) / 10.0
        intensity = float(self.sl_intensity.value()) / 100.0
        contrast = float(self.sl_contrast.value()) / 100.0
        vmax = float(self.sl_vmax.value())
        alpha = float(self.sl_alpha.value()) / 100.0

        # Keep legacy global for fused overlay, but also set per-view attributes
        _V21_FUSED_PET_GAMMA = gamma

        for v in (getattr(self.app, "view_ax", None), getattr(self.app, "view_co", None), getattr(self.app, "view_sa", None)):
            if v is None:
                continue
            v.pet_vmax = vmax
            v.pet_alpha = alpha
            v.pet_gamma = gamma
            v.pet_intensity = intensity
            v.pet_contrast = contrast
            try:
                v.update_view()
            except Exception:
                pass

class V21ExportTab(QtWidgets.QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self._build()

    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)
        btns = QtWidgets.QHBoxLayout()
        self.btn_jpg = QtWidgets.QPushButton("Export current views (JPG)")
        self.btn_pdf = QtWidgets.QPushButton("Export PERCIST + RECIST report (PDF)")
        btns.addWidget(self.btn_jpg); btns.addWidget(self.btn_pdf); btns.addStretch(1)
        lay.addLayout(btns)
        # --- NEW: Export fused metric images (with ROI selector) ---
        gb = QtWidgets.QGroupBox("Export fused PET/CT images with metrics (PNG/JPG)")
        g = QtWidgets.QGridLayout(gb)
        r = 0
        g.addWidget(QtWidgets.QLabel("ROI:"), r, 0)
        self.cmb_roi = QtWidgets.QComboBox()
        self.cmb_roi.addItems([
            "Primary VOI",
            "Active ROI (current selection)",
            "All lesions (total tumour burden)"
        ])
        g.addWidget(self.cmb_roi, r, 1, 1, 2)
        r += 1
        g.addWidget(QtWidgets.QLabel("Export view:"), r, 0)
        self.cmb_view = QtWidgets.QComboBox()
        self.cmb_view.addItems([
            "Axial fused @ SUVmax (markers + labels)",
            "Coronal fused @ SUVmax (markers + labels)",
            "Coronal fused @ Centroid (markers + labels)",
            "Coronal MIP fused + whole-body Dmax (markers + line)"
        ])
        g.addWidget(self.cmb_view, r, 1, 1, 2)
        r += 1
        g.addWidget(QtWidgets.QLabel("Format:"), r, 0)
        self.cmb_fmt = QtWidgets.QComboBox(); self.cmb_fmt.addItems(["PNG", "JPG"])
        g.addWidget(self.cmb_fmt, r, 1)
        g.addWidget(QtWidgets.QLabel("JPG quality:"), r, 2)
        self.spin_quality = QtWidgets.QSpinBox(); self.spin_quality.setRange(10, 100); self.spin_quality.setValue(95)
        g.addWidget(self.spin_quality, r, 3)
        r += 1
        g.addWidget(QtWidgets.QLabel("gETU a:"), r, 0)
        self.spin_getu_a = QtWidgets.QDoubleSpinBox()
        self.spin_getu_a.setDecimals(2); self.spin_getu_a.setRange(0.05, 10.0); self.spin_getu_a.setSingleStep(0.05); self.spin_getu_a.setValue(1.50)
        g.addWidget(self.spin_getu_a, r, 1)
        g.addWidget(QtWidgets.QLabel("Orientation:"), r, 2)
        self.cmb_orient = QtWidgets.QComboBox()
        self.cmb_orient.addItems([
            "Radiological (patient R on left)",
            "Anatomical (patient R on right)"
        ])
        g.addWidget(self.cmb_orient, r, 3)
        r += 1
        g.addWidget(QtWidgets.QLabel("Flip vertically:"), r, 0)
        self.chk_flip_ud = QtWidgets.QCheckBox("Yes")
        self.chk_flip_ud.setChecked(False)
        
        # Auto-orientation helper: most DICOM viewers show coronal with opposite superior–inferior
        # compared to raw array indexing. To match the on-screen viewer, default Flip vertically
        # ON for coronal export presets and OFF for axial presets. User can still override manually.
        def _auto_flip_for_export_view():
            try:
                t = self.cmb_view.currentText().lower()
            except Exception:
                t = ""
            if "coronal" in t:
                self.chk_flip_ud.setChecked(True)
            else:
                self.chk_flip_ud.setChecked(False)

        self.cmb_view.currentIndexChanged.connect(_auto_flip_for_export_view)
        _auto_flip_for_export_view()
        g.addWidget(self.chk_flip_ud, r, 1)
        self.btn_export_metrics = QtWidgets.QPushButton("Export metric image")
        g.addWidget(self.btn_export_metrics, r, 2, 1, 2)
        r += 1
        self.ed_out = QtWidgets.QLineEdit()
        import os as _os
        self.ed_out.setText(getattr(self.app, "out_dir", _os.path.join(_os.getcwd(), "results")))
        self.btn_browse = QtWidgets.QPushButton("Browse...")
        g.addWidget(QtWidgets.QLabel("Output folder:"), r, 0)
        g.addWidget(self.ed_out, r, 1, 1, 2)
        g.addWidget(self.btn_browse, r, 3)
        lay.addWidget(gb)
        self.btn_export_metrics.clicked.connect(self.export_metric_image)
        self.btn_browse.clicked.connect(self._choose_out_dir)

        self.log = QtWidgets.QTextEdit(); self.log.setReadOnly(True)
        lay.addWidget(self.log, 1)
        self.btn_jpg.clicked.connect(self.export_jpg)
        self.btn_pdf.clicked.connect(self.export_pdf)

    def _log(self, s):
        self.log.append(str(s))

    def _choose_out_dir(self):
        try:
            import os
            out = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder", self.ed_out.text().strip() or os.getcwd())
            if out:
                self.ed_out.setText(out)
                try:
                    self.app.out_dir = out
                except Exception:
                    pass
        except Exception as e:
            self._log(f"Browse failed: {e}")

    def _resolve_mask_for_export(self):
        """Return (mask, mask_name). Keeps primary/meta logic consistent with the viewer."""
        sel = self.cmb_roi.currentText().strip() if hasattr(self, "cmb_roi") else "Primary VOI"
        app = self.app
        if sel.startswith("Primary"):
            return getattr(app, "mask", None), "primary"
        if sel.startswith("Active"):
            cls = getattr(app, "active_roi_class", "primary")
            if cls == "meta":
                mi = int(getattr(app, "active_meta_index", 0))
                mm = (getattr(app, "meta_masks", []) or [])
                if mm and 0 <= mi < len(mm):
                    return mm[mi], f"meta_{mi+1}"
            return getattr(app, "mask", None), "primary"
        # All lesions (union)
        m_primary = getattr(app, "mask", None)
        if m_primary is None:
            return None, "all"
        out = m_primary.astype(bool).copy()
        for m in (getattr(app, "meta_masks", []) or []):
            if m is not None and np.count_nonzero(m) > 0:
                out |= m.astype(bool)
        return out, "all"

    def _compose_fused_pil(self, ct2d: np.ndarray, pet2d: np.ndarray, wl: float, ww: float, pet_vmax: float, pet_alpha: float, pet_gamma: float, pet_intensity: float, pet_contrast: float):
        """Return PIL.Image in RGB for fused CT+PET."""
        from PIL import Image
        ct8 = _window_ct_to_uint8(ct2d, wl, ww)
        ct_rgb = np.stack([ct8, ct8, ct8], axis=-1).astype(np.float32)
        rgba = _pet_to_rgba(pet2d, pet_vmax, float(pet_alpha), _make_hot_iron_lut(), gamma=float(pet_gamma), intensity=float(pet_intensity), contrast=float(pet_contrast))
        pet_rgb = rgba[..., :3].astype(np.float32)
        a = (rgba[..., 3:4].astype(np.float32) / 255.0)
        fused = ct_rgb * (1.0 - a) + pet_rgb * a
        fused = np.clip(fused, 0, 255).astype(np.uint8)
        return Image.fromarray(fused, mode="RGB")

    def export_metric_image(self):
        """Export fused PET/CT images with computed metrics and markers."""
        import os, time
        from PIL import Image, ImageDraw, ImageFont

        app = self.app
        if getattr(app, "pet_suv", None) is None:
            QtWidgets.QMessageBox.warning(self, "Export", "Load a PET/CT study first.")
            return

        out_dir = (self.ed_out.text().strip() if hasattr(self, "ed_out") else "") or getattr(app, "out_dir", os.path.join(os.getcwd(), "results"))
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")

        spacing = getattr(app, "spacing_zyx", (1.0, 1.0, 1.0))
        ct = getattr(app, "ct_hu", None)
        pet = getattr(app, "pet_suv", None)

        # View parameters (use current axial view as canonical defaults)
        vax = getattr(app, "view_ax", None)
        wl = float(getattr(vax, "ct_wl", 40.0))
        ww = float(getattr(vax, "ct_ww", 400.0))
        pet_vmax = float(getattr(vax, "pet_vmax", np.percentile(pet[pet > 0], 99) if np.any(pet > 0) else 10.0))
        pet_alpha = float(getattr(vax, "pet_alpha", 0.55))
        pet_gamma = float(getattr(vax, "pet_gamma", 1.0))
        pet_intensity = float(getattr(vax, "pet_intensity", 1.0))
        pet_contrast = float(getattr(vax, "pet_contrast", 1.0))

        fmt = (self.cmb_fmt.currentText().strip().lower() if hasattr(self, "cmb_fmt") else "png")
        q = int(self.spin_quality.value()) if hasattr(self, "spin_quality") else 95
        mode = (self.cmb_view.currentText() if hasattr(self, "cmb_view") else "Axial fused @ SUVmax (markers + labels)")
        orient = (self.cmb_orient.currentText().strip() if hasattr(self, "cmb_orient") else "Radiological (patient R on left)")
        flip_ud = bool(self.chk_flip_ud.isChecked()) if hasattr(self, "chk_flip_ud") else False

        # resolve mask(s)
        msel, mname = self._resolve_mask_for_export()
        if msel is None or np.count_nonzero(msel) == 0:
            QtWidgets.QMessageBox.warning(self, "Export", "Selected ROI is empty. Draw/segment a VOI first.")
            return

        # hotspot + centroid
        hh = _v21_hotspot_zyx(msel, pet)
        hotspot_zyx = hh[0] if isinstance(hh, tuple) else None
        if hotspot_zyx is None:
            QtWidgets.QMessageBox.warning(self, "Export", "Could not find SUVmax voxel in the selected ROI.")
            return
        hz, hy, hx = [int(v) for v in hotspot_zyx]
        centroid_mm = _v21_centroid_phys_mm(msel, spacing)
        # centroid voxel (approx)
        if centroid_mm is not None:
            dz, dy, dx = [float(x) for x in spacing]
            cz = int(round(centroid_mm[0] / dz)); cy = int(round(centroid_mm[1] / dy)); cx = int(round(centroid_mm[2] / dx))
        else:
            cz, cy, cx = hz, hy, hx

        # For 2D exports, centroid may not lie on the chosen hotspot slice (expected).
        # We report slice offsets in mm so users understand why centroid marker can appear outside the VOI on that slice.
        try:
            sz_mm, sy_mm, sx_mm = [float(x) for x in spacing]
        except Exception:
            sz_mm, sy_mm, sx_mm = (1.0, 1.0, 1.0)
        dz_cent_mm = abs(int(cz) - int(hz)) * sz_mm
        dy_cent_mm = abs(int(cy) - int(hy)) * sy_mm

        # compute metrics
        nh = _v21_compute_nhoc_nhop(msel, pet, spacing) or {}
        a_getu = float(self.spin_getu_a.value()) if hasattr(self, "spin_getu_a") else 1.5
        getu = _v21_calc_getu_stable(msel, pet, spacing, a=a_getu)
        d_intra, p1_intra, p2_intra = _v21_intra_dmax_mm(msel, spacing)

        # WB dissemination (if needed)
        masks_all = []
        m_primary = getattr(app, "mask", None)
        if m_primary is not None and np.count_nonzero(m_primary) > 0:
            masks_all.append(m_primary.astype(bool))
        for mm in (getattr(app, "meta_masks", []) or []):
            if mm is not None and np.count_nonzero(mm) > 0:
                masks_all.append(mm.astype(bool))
        d_wb, cent_pair, c1_mm, c2_mm = _v21_dmax_centroid_detail_mm(masks_all, spacing)
        d_wb_vox, p1_wb, p2_wb = _v21_dmax_vox_detail_mm(masks_all, spacing)

        # Build image
        if ct is None:
            ct = np.zeros_like(pet, dtype=np.float32)

        # IMPORTANT: to ensure exports match the on-screen viewer exactly (including any built-in flips),
        # we reuse the SliceView slice extraction + voxel_to_display mapping.
        view_ax = getattr(app, "view_ax", None)
        view_co = getattr(app, "view_co", None)

        title_tag = ""
        ct2 = pet2 = None

        if mode.lower().startswith("axial"):
            if view_ax is not None:
                # preserve state
                old_idx = int(getattr(view_ax, "slice_idx", hz))
                old_mip = bool(getattr(view_ax, "is_mip_active", False))
                try:
                    view_ax.is_mip_active = False
                except Exception:
                    pass
                view_ax.slice_idx = int(hz)
                ct2, pet2, _ = view_ax._get_slice_arrays()
                # marker locations in display coordinates
                hot_xy = view_ax.voxel_to_display(hz, hy, hx)
                cen_xy = view_ax.voxel_to_display(cz, cy, cx)
                # restore
                view_ax.slice_idx = old_idx
                try:
                    view_ax.is_mip_active = old_mip
                except Exception:
                    pass
            else:
                # fallback: mimic viewer logic (axial is flipped vertically)
                ct2 = ct[hz][::-1]
                pet2 = pet[hz][::-1]
                ny = pet.shape[1]
                hot_xy = (hx, (ny - 1) - hy)
                cen_xy = (cx, (ny - 1) - cy)

            im = self._compose_fused_pil(ct2, pet2, wl, ww, pet_vmax, pet_alpha, pet_gamma, pet_intensity, pet_contrast)
            title_tag = "axial"

        elif mode.lower().startswith("coronal fused"):
            # coronal slice is selected along Y axis
            plane_y = int(hy)
            if "centroid" in mode.lower():
                plane_y = int(cy)

            if view_co is not None:
                old_idx = int(getattr(view_co, "slice_idx", plane_y))
                old_mip = bool(getattr(view_co, "is_mip_active", False))
                try:
                    view_co.is_mip_active = False
                except Exception:
                    pass
                view_co.slice_idx = int(plane_y)
                ct2, pet2, _ = view_co._get_slice_arrays()
                hot_xy = view_co.voxel_to_display(hz, plane_y, hx)
                # centroid is a 3D point; on a different plane it is a projection (still useful)
                cen_xy = view_co.voxel_to_display(cz, plane_y, cx)
                # restore
                view_co.slice_idx = old_idx
                try:
                    view_co.is_mip_active = old_mip
                except Exception:
                    pass
            else:
                ct2 = ct[:, plane_y]
                pet2 = pet[:, plane_y]
                hot_xy = (hx, hz)
                cen_xy = (cx, cz)

            im = self._compose_fused_pil(ct2, pet2, wl, ww, pet_vmax, pet_alpha, pet_gamma, pet_intensity, pet_contrast)
            title_tag = "coronal_centroid" if "centroid" in mode.lower() else "coronal"

        else:
            # Coronal MIP (PET) + CT at mid Y
            y_mid = int(pet.shape[1] // 2)

            if view_co is not None:
                old_idx = int(getattr(view_co, "slice_idx", y_mid))
                old_mip = bool(getattr(view_co, "is_mip_active", False))
                try:
                    view_co.is_mip_active = True
                except Exception:
                    pass
                view_co.slice_idx = int(y_mid)
                ct2, pet2, _ = view_co._get_slice_arrays()
                hot_xy = view_co.voxel_to_display(hz, y_mid, hx)
                cen_xy = view_co.voxel_to_display(cz, y_mid, cx)
                # restore
                view_co.slice_idx = old_idx
                try:
                    view_co.is_mip_active = old_mip
                except Exception:
                    pass
            else:
                ct2 = ct[:, y_mid]
                pet2 = np.max(pet, axis=1)
                hot_xy = (hx, hz)
                cen_xy = (cx, cz)

            im = self._compose_fused_pil(ct2, pet2, wl, ww, pet_vmax, pet_alpha, pet_gamma, pet_intensity, pet_contrast)
            title_tag = "coronal_mip_dmax"

        # Optional forced vertical flip (rare; only if you want to match a different convention)
        if flip_ud:
            from PIL import Image
            im = im.transpose(Image.FLIP_TOP_BOTTOM)
            h_img = im.size[1]
            hot_xy = (int(hot_xy[0]), (h_img - 1) - int(hot_xy[1]))
            cen_xy = (int(cen_xy[0]), (h_img - 1) - int(cen_xy[1]))
        draw = ImageDraw.Draw(im)
        # Left-right orientation control
        if "Anatomical" in orient:
            # Flip image left-right and remap x coordinates for markers/lines
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            draw = ImageDraw.Draw(im)
            w = im.size[0]
            hot_xy = ((w - 1) - int(hot_xy[0]), int(hot_xy[1]))
            cen_xy = ((w - 1) - int(cen_xy[0]), int(cen_xy[1]))

        # Simple markers
        def circle(xy, r, outline):
            x, y = xy
            draw.ellipse((x-r, y-r, x+r, y+r), outline=outline, width=2)
        def cross(xy, r, outline):
            x, y = xy
            draw.line((x-r, y-r, x+r, y+r), fill=outline, width=2)
            draw.line((x-r, y+r, x+r, y-r), fill=outline, width=2)
        def triangle(xy, r, outline):
            x, y = xy
            pts = [(x, y-r), (x-r, y+r), (x+r, y+r)]
            draw.polygon(pts, outline=outline)

        # R/L labels (helps interpret radiological vs anatomical convention)
        try:
            font_rl = ImageFont.load_default()
        except Exception:
            font_rl = None
        # In radiological convention, patient Right is on image Left.
        if "Radiological" in orient:
            lab_left, lab_right = "R", "L"
        else:
            lab_left, lab_right = "L", "R"
        draw.text((10, im.size[1]//2), lab_left, fill=(255,255,255), font=font_rl)
        draw.text((im.size[0]-20, im.size[1]//2), lab_right, fill=(255,255,255), font=font_rl)

        # SUVmax voxel (green circle)
        circle(hot_xy, 6, outline=(0, 255, 0))
        # VOI centroid (cyan triangle)
        triangle(cen_xy, 7, outline=(0, 255, 255))
        # If centroid is off the chosen slice, annotate that it is a projection (not an error)
        try:
            if title_tag == 'axial' and dz_cent_mm > 0:
                draw.text((int(cen_xy[0]) + 10, int(cen_xy[1]) - 10), 'C(proj)', fill=(0, 255, 255), font=font_rl)
            if title_tag == 'coronal' and dy_cent_mm > 0:
                draw.text((int(cen_xy[0]) + 10, int(cen_xy[1]) - 10), 'C(proj)', fill=(0, 255, 255), font=font_rl)
        except Exception:
            pass

        # Dmax overlays for MIP mode
        if title_tag.startswith("coronal_mip") and c1_mm is not None and c2_mm is not None:
            dz, dy, dx = [float(x) for x in spacing]
            # centroids in voxel coords (z,y,x) for plotting on coronal MIP (z,x)
            c1_zyx = (int(round(c1_mm[0] / dz)), int(round(c1_mm[1] / dy)), int(round(c1_mm[2] / dx)))
            c2_zyx = (int(round(c2_mm[0] / dz)), int(round(c2_mm[1] / dy)), int(round(c2_mm[2] / dx)))

            # Map to export-image pixel coords in the *pre-flip* image space, then apply the same flips
            if view_co is not None:
                p1 = view_co.voxel_to_display(int(c1_zyx[0]), int(y_mid), int(c1_zyx[2]))
                p2 = view_co.voxel_to_display(int(c2_zyx[0]), int(y_mid), int(c2_zyx[2]))
            else:
                p1 = (c1_zyx[2], c1_zyx[0])
                p2 = (c2_zyx[2], c2_zyx[0])

            def _xf(pt):
                x, y = int(pt[0]), int(pt[1])
                if flip_ud:
                    y = (im.size[1] - 1) - y
                if "Anatomical" in orient:
                    x = (im.size[0] - 1) - x
                return (x, y)

            p1 = _xf(p1)
            p2 = _xf(p2)

            draw.line((p1[0], p1[1], p2[0], p2[1]), fill=(0, 255, 255), width=2)
            circle(p1, 6, outline=(0, 255, 255))
            circle(p2, 6, outline=(0, 255, 255))

            if p1_wb is not None and p2_wb is not None:
                if view_co is not None:
                    pv1 = view_co.voxel_to_display(int(p1_wb[0]), int(y_mid), int(p1_wb[2]))
                    pv2 = view_co.voxel_to_display(int(p2_wb[0]), int(y_mid), int(p2_wb[2]))
                else:
                    pv1 = (int(p1_wb[2]), int(p1_wb[0]))
                    pv2 = (int(p2_wb[2]), int(p2_wb[0]))
                pv1 = _xf(pv1)
                pv2 = _xf(pv2)
                cross(pv1, 6, outline=(255, 255, 0))
                cross(pv2, 6, outline=(255, 255, 0))

        # Text box
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        lines_txt = [
            f"ROI: {mname}",
            f"NHOCmax: {float(nh.get('nhocmax', 0.0)):.3f}",

            f"NHOPmax: {float(nh.get('nhopmax', 0.0)):.3f}",
            f"Intra Dmax (mm): {float(d_intra):.1f}",
            f"gETU(a={a_getu:.2f}): {float(getu):.3f}",
        ]
        # Explain 2D visualization: centroid marker is projected onto the exported slice;
        # it may fall outside the VOI on that slice if centroid is in a different plane.
        if title_tag == 'coronal' and dy_cent_mm > 0.5:
            lines_txt.append(f"Centroid offset from this slice: Δy={dy_cent_mm:.1f} mm")
        if title_tag == 'axial' and dz_cent_mm > 0.5:
            lines_txt.append(f"Centroid offset from this slice: Δz={dz_cent_mm:.1f} mm")

        # Coronal exports: slice is along Y axis
        if title_tag == 'coronal' and dy_cent_mm > 0.5:
            lines_txt.append(f"Centroid offset from this slice: Δy={dy_cent_mm:.1f} mm")
        if title_tag == 'coronal_centroid':
            dy_hot_mm = abs(int(hy) - int(cy)) * sy_mm
            if dy_hot_mm > 0.5:
                lines_txt.append(f"Hotspot offset from this slice: Δy={dy_hot_mm:.1f} mm")

        if title_tag.startswith("coronal_mip"):
            lines_txt += [
                f"WB Dmax centroid (mm): {float(d_wb):.1f}",
                f"WB DmaxVox (mm): {float(d_wb_vox):.1f}",
                f"Lesions used: {len(masks_all)}"
            ]
        text = "\n".join(lines_txt)
        # background rectangle
        pad = 6
        try:
            bbox = draw.multiline_textbbox((0, 0), text, font=font)
            tw, th = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        except Exception:
            try:
                tw, th = draw.multiline_textsize(text, font=font)
            except Exception:
                tw, th = (max(1, max((len(s) for s in lines_txt), default=1)) * 6, max(1, len(lines_txt)) * 12)
        draw.rectangle((5, 5, 5+tw+pad*2, 5+th+pad*2), fill=(0, 0, 0))
        draw.multiline_text((5+pad, 5+pad), text, fill=(255, 255, 255), font=font)

        # Save
        base = f"metric_{title_tag}_{mname}_{ts}"
        fp = os.path.join(out_dir, f"{base}.{fmt}")
        try:
            if fmt in ("jpg", "jpeg"):
                im.convert("RGB").save(fp, "JPEG", quality=q)
            else:
                im.save(fp, "PNG")
            self._log(f"Saved {fp}")
        except Exception as e:
            self._log(f"Export failed: {e}")

    def export_jpg(self):
        import os, time
        out_dir = getattr(self.app, "out_dir", os.path.join(os.getcwd(), "results"))
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")

        targets = []
        for nm in ("view_ax", "view_co", "view_sa"):
            w = getattr(self.app, nm, None)
            if w is not None:
                targets.append((nm, w))

        if not targets:
            self._log("No views found to export.")
            return

        for nm, w in targets:
            try:
                pm = w.grab()
                fp = os.path.join(out_dir, f"{nm}_{ts}.jpg")
                pm.save(fp, "JPG")
                self._log(f"Saved {fp}")
            except Exception as e:
                self._log(f"Failed {nm}: {e}")

    def export_pdf(self):
        if A4 is None:
            QtWidgets.QMessageBox.warning(self, "PDF", "reportlab not installed. Install: pip install reportlab")
            return
        import os, time
        out_dir = getattr(self.app, "out_dir", os.path.join(os.getcwd(), "results"))
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fp = os.path.join(out_dir, f"PERCIST_RECIST_Report_{ts}.pdf")

        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph("PERCIST & RECIST 1.1 Measurement Report", styles["Title"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph("PERCIST", styles["Heading2"]))
        pr = getattr(self.app, "percist_results", None)
        if pr:
            data = [["Field","Value"],
                    ["Lesion", pr.get("lesion","")],
                    ["Liver SULmean", f"{pr.get('liver_mean',0):.3f}"],
                    ["Liver SD", f"{pr.get('liver_sd',0):.3f}"],
                    ["Threshold", f"{pr.get('threshold',0):.3f}"],
                    ["SULmax", f"{pr.get('sulmax',0):.3f}"],
                    ["SULmean", f"{pr.get('sulmean',0):.3f}"],
                    ["SULpeak", f"{pr.get('sulpeak',0):.3f}"]]
        else:
            data = [["PERCIST results not computed."]]
        t = Table(data, hAlign="LEFT")
        t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.grey),
                               ("BACKGROUND",(0,0),(-1,0),colors.lightgrey)]))
        story.append(t); story.append(Spacer(1, 12))

        story.append(Paragraph("RECIST 1.1", styles["Heading2"]))
        rr = getattr(self.app, "recist_results", None)
        if rr and rr.get("lesions"):
            data2 = [["#", "Type", "Lesion", "Long axis (mm)"]]
            for i, row in enumerate(rr["lesions"], start=1):
                data2.append([str(i), row.get("type",""), row.get("lesion",""), f"{row.get('long_axis_mm',0):.1f}"])
            data2.append(["", "", "Sum", f"{rr.get('sum_mm',0):.1f}"])
        else:
            data2 = [["RECIST table not available."]]
        t2 = Table(data2, hAlign="LEFT")
        t2.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.grey),
                                ("BACKGROUND",(0,0),(-1,0),colors.lightgrey)]))
        story.append(t2)

        doc = SimpleDocTemplate(fp, pagesize=A4)
        doc.build(story)
        QtWidgets.QMessageBox.information(self, "PDF", f"Saved: {fp}")


# ---- V21 ADDON: Tumor 3D metrics (volume/surface) + NHOC/NHOPmax ----

def _v21_spacing_xyz_from_zyx(spacing_zyx):
    """Convert (sz,sy,sx) -> (sx,sy,sz) for SimpleITK spacing."""
    try:
        sz, sy, sx = [float(x) for x in spacing_zyx]
        return (sx, sy, sz)
    except Exception:
        return (1.0, 1.0, 1.0)

def _v21_mask_volume_mm3(mask: np.ndarray, spacing_zyx) -> float:
    try:
        sz, sy, sx = [float(x) for x in spacing_zyx]
        return float(np.count_nonzero(mask)) * (sz * sy * sx)
    except Exception:
        return 0.0


def _v21_mask_volume_ml(mask, spacing_zyx):
    """Return volume in mL given a binary mask and spacing (z,y,x) in mm.

    1 mL = 1 cm^3 = 1000 mm^3.
    """
    try:
        mm3 = _v21_mask_volume_mm3(mask, spacing_zyx)
        return float(mm3) / 1000.0
    except Exception:
        return 0.0

def _v21_mask_surface_area_mm2(mask: np.ndarray, spacing_zyx) -> float:
    """
    Voxel-surface approximation (axis-aligned faces).
    Counts boundary faces along each axis and multiplies by the corresponding face area.
    """
    try:
        sz, sy, sx = [float(x) for x in spacing_zyx]
        m = (mask.astype(np.uint8) > 0)
        if m.ndim != 3 or not np.any(m):
            return 0.0

        # Faces orthogonal to Z: compare slices (z) and include outer boundaries
        z_faces = np.count_nonzero(m[0, :, :]) + np.count_nonzero(m[-1, :, :])
        z_faces += np.count_nonzero(m[1:, :, :] != m[:-1, :, :])
        # Faces orthogonal to Y
        y_faces = np.count_nonzero(m[:, 0, :]) + np.count_nonzero(m[:, -1, :])
        y_faces += np.count_nonzero(m[:, 1:, :] != m[:, :-1, :])
        # Faces orthogonal to X
        x_faces = np.count_nonzero(m[:, :, 0]) + np.count_nonzero(m[:, :, -1])
        x_faces += np.count_nonzero(m[:, :, 1:] != m[:, :, :-1])

        area = (z_faces * (sy * sx)) + (y_faces * (sz * sx)) + (x_faces * (sz * sy))
        return float(area)
    except Exception:
        return 0.0

def _v21_equiv_sphere_radius_mm(vol_mm3: float) -> float:
    """Radius R of a sphere with the same volume (mm)."""
    try:
        v = float(vol_mm3)
        if v <= 0:
            return 0.0
        return float((3.0 * v / (4.0 * math.pi)) ** (1.0 / 3.0))
    except Exception:
        return 0.0

def _v21_centroid_phys_mm(mask: np.ndarray, spacing_zyx):
    """Centroid in physical mm coordinates (z_mm, y_mm, x_mm) using voxel centers."""
    try:
        m = mask.astype(bool)
        if not np.any(m):
            return None
        coords = np.argwhere(m)  # z,y,x
        cz, cy, cx = coords.mean(axis=0)
        sz, sy, sx = [float(x) for x in spacing_zyx]
        return (float(cz) * sz, float(cy) * sy, float(cx) * sx)
    except Exception:
        return None

def _v21_hotspot_zyx(mask: np.ndarray, pet_suv: np.ndarray):
    """Return (z,y,x), suvmax for SUVmax voxel within mask."""
    try:
        m = mask.astype(bool)
        if not np.any(m) or pet_suv is None:
            return None, 0.0
        vals = pet_suv[m]
        if vals.size == 0:
            return None, 0.0
        imax = int(np.argmax(vals))
        suvmax = float(vals[imax])
        coords = np.argwhere(m)
        zyx = tuple(int(x) for x in coords[imax])
        return zyx, suvmax
    except Exception:
        return None, 0.0

def _v21_distance_hotspot_to_boundary_mm(mask: np.ndarray, hotspot_zyx, spacing_zyx) -> float:
    """
    Distance from hotspot voxel to closest tumor perimeter (mm).
    Uses Signed Maurer distance map (SimpleITK) with image spacing.
    """
    try:
        if hotspot_zyx is None:
            return 0.0
        m = (mask.astype(np.uint8) > 0)
        if not np.any(m):
            return 0.0
        img = sitk.GetImageFromArray(m.astype(np.uint8))
        img.SetSpacing(_v21_spacing_xyz_from_zyx(spacing_zyx))
        # Inside positive distance to boundary
        dist = sitk.SignedMaurerDistanceMap(img, insideIsPositive=True, squaredDistance=False, useImageSpacing=True)
        z, y, x = [int(v) for v in hotspot_zyx]
        # SITK index order: x,y,z
        return float(dist.GetPixel(int(x), int(y), int(z)))
    except Exception:
        # Fallback: approximate by counting one voxel to boundary if anything goes wrong
        return 0.0

def _v21_compute_nhoc_nhop(mask: np.ndarray, pet_suv: np.ndarray, spacing_zyx):
    """
    NHOCmax: normalized distance from SUVmax voxel to tumor centroid.
    NHOPmax: normalized distance from SUVmax voxel to tumor perimeter (closest border).
    Normalization uses R = radius of a sphere with the same tumor volume.
    """
    try:
        if mask is None or pet_suv is None:
            return None
        vol = _v21_mask_volume_mm3(mask, spacing_zyx)
        R = _v21_equiv_sphere_radius_mm(vol)
        if R <= 0:
            return dict(vol_mm3=0.0, R_mm=0.0, suvmax=0.0, nhocmax=0.0, nhopmax=0.0,
                        d_centroid_mm=0.0, d_perimeter_mm=0.0,
                        centroid_zyx_mm=None, hotspot_zyx=None)
        centroid_mm = _v21_centroid_phys_mm(mask, spacing_zyx)
        hotspot_zyx, suvmax = _v21_hotspot_zyx(mask, pet_suv)
        if centroid_mm is None or hotspot_zyx is None:
            return dict(vol_mm3=vol, R_mm=R, suvmax=float(suvmax), nhocmax=0.0, nhopmax=0.0,
                        d_centroid_mm=0.0, d_perimeter_mm=0.0,
                        centroid_zyx_mm=centroid_mm, hotspot_zyx=hotspot_zyx)

        sz, sy, sx = [float(x) for x in spacing_zyx]
        hz, hy, hx = [int(v) for v in hotspot_zyx]
        hotspot_mm = (hz * sz, hy * sy, hx * sx)

        d_centroid = float(math.sqrt((hotspot_mm[0] - centroid_mm[0]) ** 2 +
                                     (hotspot_mm[1] - centroid_mm[1]) ** 2 +
                                     (hotspot_mm[2] - centroid_mm[2]) ** 2))
        d_perim = _v21_distance_hotspot_to_boundary_mm(mask, hotspot_zyx, spacing_zyx)

        nhoc = float(d_centroid / (R + 1e-9))
        nhop = float(d_perim / (R + 1e-9))

        return dict(
            vol_mm3=vol,
            R_mm=R,
            suvmax=float(suvmax),
            nhocmax=nhoc,
            nhopmax=nhop,
            d_centroid_mm=d_centroid,
            d_perimeter_mm=float(d_perim),
            centroid_zyx_mm=centroid_mm,  # (zmm,ymm,xmm)
            hotspot_zyx=hotspot_zyx       # (z,y,x)
        )
    except Exception:
        return None

def _v21_build_metastasis_label_mask(meta_masks: List[np.ndarray], shape_zyx) -> np.ndarray:
    """Create a label mask (0 background, 1..N lesions) from list of binary masks."""
    out = np.zeros(shape_zyx, dtype=np.uint16)
    if not meta_masks:
        return out
    lab = 1
    for mm in meta_masks:
        if mm is None:
            lab += 1
            continue
        try:
            out[mm.astype(bool)] = np.uint16(lab)
        except Exception:
            pass
        lab += 1
    return out

# --- Extra 3D metrics helpers (gETU, sphericity/asphericity, Dmax) ---

def _v21_is_reasonable_volume(arr: np.ndarray) -> bool:
    """Heuristic: reject tiny/thumbnail series accidentally selected from a ZIP."""
    try:
        if arr is None or not hasattr(arr, "shape"):
            return False
        if arr.ndim != 3:
            return False
        z, y, x = arr.shape
        if x < 16 or y < 16 or z < 2:
            return False
        # Reject obviously non-tomographic series (very few voxels overall)
        if (z * y * x) < 16 * 16 * 4:
            return False
        return True
    except Exception:
        return False


def _v21_series_score(entry: dict, prefer_modalities=("PT",)) -> float:
    """Score a series entry for auto-picking the most likely tomographic dataset."""
    try:
        ds = entry.get("ds0", None)
        n = len(entry.get("files", []))
        mod = str(entry.get("modality", "") or getattr(ds, "Modality", "")).upper()
        rows = int(getattr(ds, "Rows", 0) or 0) if ds is not None else 0
        cols = int(getattr(ds, "Columns", 0) or 0) if ds is not None else 0
        pix = rows * cols
        score = 0.0
        if mod in [m.upper() for m in prefer_modalities]:
            score += 1e6
        score += n * 500.0
        score += pix * 1.0
        # Prefer named series over empty description
        desc = str(getattr(ds, "SeriesDescription", "") or "")
        if desc.strip():
            score += 100.0
        # Penalize tiny images
        if rows and cols and (rows < 64 or cols < 64):
            score -= 5e5
        if n < 10:
            score -= 2e5
        return score
    except Exception:
        return 0.0


def _v21_rank_series_uids(idx: dict, prefer_modalities=("PT",)) -> list:
    uids = list(idx.keys())
    uids.sort(key=lambda uid: _v21_series_score(idx[uid], prefer_modalities), reverse=True)
    return uids


def _v21_load_best_series(idx: dict, initial_uid: str, prefer_modalities=("PT",)):
    """Try initial_uid first, then fall back to other series of the requested modality."""
    tried = []
    candidates = []
    if initial_uid:
        candidates.append(initial_uid)
    for uid in _v21_rank_series_uids(idx, prefer_modalities):
        if uid not in candidates:
            candidates.append(uid)

    last_err = None
    for uid in candidates:
        try:
            entry = idx.get(uid)
            if not entry:
                continue
            ds0 = entry.get("ds0", None)
            mod = str(entry.get("modality", "") or getattr(ds0, "Modality", "")).upper()
            if prefer_modalities and mod not in [m.upper() for m in prefer_modalities]:
                continue

            files = _sort_series_files(entry["files"])
            img = _sitk_read_series(files)
            arr = sitk.GetArrayFromImage(img)  # z,y,x
            if not _v21_is_reasonable_volume(arr):
                tried.append((uid, mod, arr.shape))
                continue
            return img, files, uid
        except Exception as e:
            last_err = e
            continue

    # If no series found with preferred modalities, try any series
    if not tried and prefer_modalities:
        for uid in candidates:
            try:
                entry = idx.get(uid)
                if not entry: continue
                files = _sort_series_files(entry["files"])
                img = _sitk_read_series(files)
                arr = sitk.GetArrayFromImage(img)
                if not _v21_is_reasonable_volume(arr):
                    tried.append((uid, "ANY", arr.shape))
                    continue
                return img, files, uid
            except Exception as e:
                last_err = e
                continue

    msg = f"Could not load a valid series for modalities={prefer_modalities}. Tried: {tried}"
    if last_err is not None:
        msg += f" (last error: {last_err})"
    raise RuntimeError(msg)


def _v21_mask_surface_sphere_mm2(vol_mm3: float) -> float:
    if vol_mm3 <= 0:
        return 0.0
    r = _v21_equiv_sphere_radius_mm(vol_mm3)
    return 4.0 * math.pi * (r * r)


def _v21_calc_sphericity_asphericity(mask: np.ndarray, spacing_zyx):
    vol_mm3 = _v21_mask_volume_mm3(mask, spacing_zyx)
    s_mm2 = _v21_mask_surface_area_mm2(mask, spacing_zyx)
    s_sph = _v21_mask_surface_sphere_mm2(vol_mm3)
    sphericity = (s_sph / s_mm2) if (s_mm2 > 0 and s_sph > 0) else 0.0
    asphericity_pct = ((s_mm2 / s_sph) - 1.0) * 100.0 if (s_sph > 0 and s_mm2 > 0) else 0.0
    return dict(vol_mm3=float(vol_mm3), surface_mm2=float(s_mm2), surface_sphere_mm2=float(s_sph),
                sphericity=float(sphericity), asphericity_pct=float(asphericity_pct))


def _v21_calc_getu(mask: np.ndarray, pet_suv: np.ndarray, spacing_zyx, a: float = 1.5):
    """gETU(a) = (Δv * Σ u_i^a)^(1/a), with Δv in mL."""
    if mask is None or pet_suv is None:
        return 0.0
    vals = pet_suv[mask.astype(bool)]
    if vals.size == 0:
        return 0.0
    # voxel volume in mL (1 mL = 1000 mm^3)
    dz, dy, dx = [float(x) for x in spacing_zyx]
    dv_ml = (dz * dy * dx) / 1000.0
    a = float(max(a, 1e-6))
    s = dv_ml * float(np.sum(np.power(vals.astype(np.float64), a)))
    if s <= 0:
        return 0.0
    return float(np.power(s, 1.0 / a))


def _v21_calc_getu_stable(mask: np.ndarray, pet_suv: np.ndarray, spacing_zyx, a: float = 1.5):
    """Numerically-stable gETU(a) using log-sum-exp. Keeps original _v21_calc_getu() intact."""
    if mask is None or pet_suv is None:
        return 0.0
    vals = pet_suv[mask.astype(bool)]
    if vals.size == 0:
        return 0.0
    vals = vals.astype(np.float64)
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0.0
    dz, dy, dx = [float(x) for x in spacing_zyx]
    dv_ml = (dz * dy * dx) / 1000.0
    a = float(max(a, 1e-6))
    logt = a * np.log(vals)
    m = float(np.max(logt))
    s = float(np.sum(np.exp(logt - m)))
    if s <= 0 or dv_ml <= 0:
        return 0.0
    logS = math.log(dv_ml) + (math.log(s) + m)
    return float(math.exp(logS / a))

def _v21_boundary_voxels(mask: np.ndarray, max_points: int = 4000):
    """Return subsampled boundary voxel indices (N,3) in z,y,x."""
    if mask is None or np.count_nonzero(mask) == 0:
        return np.zeros((0, 3), dtype=np.int32)
    try:
        from scipy.ndimage import binary_erosion
        struct = np.ones((3, 3, 3), dtype=bool)
        er = binary_erosion(mask.astype(bool), structure=struct)
        b = mask.astype(bool) & (~er)
    except Exception:
        b = mask.astype(bool)
    zyx = np.argwhere(b)
    if zyx.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    if zyx.shape[0] > max_points:
        step = int(math.ceil(zyx.shape[0] / max_points))
        zyx = zyx[::step]
    return zyx.astype(np.int32)

def _v21_intra_dmax_mm(mask: np.ndarray, spacing_zyx, max_points: int = 4000, chunk: int = 512):
    """Intra-lesion Dmax (Feret-like) using boundary voxels. Returns (d_mm, p1_zyx, p2_zyx).
    Exact for <= max_points points; for very large lesions this is an approximation via subsampling.
    """
    zyx = _v21_boundary_voxels(mask, max_points=max_points)
    if zyx.shape[0] < 2:
        return 0.0, None, None
    dz, dy, dx = [float(x) for x in spacing_zyx]
    pts = zyx.astype(np.float64)
    pts[:, 0] *= dz; pts[:, 1] *= dy; pts[:, 2] *= dx
    n = pts.shape[0]
    # Heuristic for very large N: farthest-point sampling (2-sweep)
    if n > 3000:
        mean = np.mean(pts, axis=0)
        i0 = int(np.argmax(np.sum((pts - mean) ** 2, axis=1)))
        d0 = np.sum((pts - pts[i0]) ** 2, axis=1)
        i1 = int(np.argmax(d0))
        d1 = np.sum((pts - pts[i1]) ** 2, axis=1)
        i2 = int(np.argmax(d1))
        return float(math.sqrt(float(d1[i2]))), tuple(map(int, zyx[i1])), tuple(map(int, zyx[i2]))
    # Exact chunked max distance
    maxd2 = -1.0; bi = bj = 0
    for i in range(0, n, chunk):
        a = pts[i:i+chunk]
        d2 = np.sum((a[:, None, :] - pts[None, :, :]) ** 2, axis=2)
        loc = np.unravel_index(int(np.argmax(d2)), d2.shape)
        v = float(d2[loc])
        if v > maxd2:
            maxd2 = v
            bi = i + int(loc[0])
            bj = int(loc[1])
    if maxd2 < 0:
        return 0.0, None, None
    return float(math.sqrt(maxd2)), tuple(map(int, zyx[bi])), tuple(map(int, zyx[bj]))

def _v21_dmax_centroid_detail_mm(masks: list, spacing_zyx):
    """Return (dmax_mm, (i,j), ci_mm, cj_mm) for centroid-based dissemination Dmax."""
    pts = []
    for idx, m in enumerate(masks or []):
        if m is None or np.count_nonzero(m) == 0:
            continue
        c = _v21_centroid_phys_mm_from_mask(m, spacing_zyx)
        if c is None:
            continue
        pts.append((idx, np.asarray(c, dtype=np.float64)))
    if len(pts) < 2:
        return 0.0, None, None, None
    maxd2 = -1.0; best = None
    for a in range(len(pts)):
        ia, pa = pts[a]
        for b in range(a+1, len(pts)):
            ib, pb = pts[b]
            d2 = float(np.sum((pa - pb) ** 2))
            if d2 > maxd2:
                maxd2 = d2; best = (ia, ib, pa, pb)
    if best is None:
        return 0.0, None, None, None
    ia, ib, pa, pb = best
    return float(math.sqrt(maxd2)), (ia, ib), pa, pb

def _v21_dmax_vox_detail_mm(masks: list, spacing_zyx, max_points_per_lesion: int = 4000, chunk: int = 256):
    """Return (dmaxvox_mm, p1_zyx, p2_zyx) for outermost-voxel dissemination DmaxVox (subsampled)."""
    # build boundary voxel sets per lesion
    bv = []
    for m in masks or []:
        if m is None or np.count_nonzero(m) == 0:
            continue
        zyx = _v21_boundary_voxels(m, max_points=max_points_per_lesion)
        if zyx.shape[0] > 0:
            bv.append(zyx)
    if len(bv) < 2:
        return 0.0, None, None
    dz, dy, dx = [float(x) for x in spacing_zyx]
    maxd2 = -1.0; best_p = best_q = None
    for i in range(len(bv)):
        A = bv[i].astype(np.float64)
        A_mm = A.copy(); A_mm[:,0]*=dz; A_mm[:,1]*=dy; A_mm[:,2]*=dx
        for j in range(i+1, len(bv)):
            B = bv[j].astype(np.float64)
            B_mm = B.copy(); B_mm[:,0]*=dz; B_mm[:,1]*=dy; B_mm[:,2]*=dx
            for k in range(0, A_mm.shape[0], chunk):
                a = A_mm[k:k+chunk]
                d2 = np.sum((a[:, None, :] - B_mm[None, :, :]) ** 2, axis=2)
                loc = np.unravel_index(int(np.argmax(d2)), d2.shape)
                v = float(d2[loc])
                if v > maxd2:
                    maxd2 = v
                    best_p = tuple(map(int, bv[i][k + int(loc[0])]))
                    best_q = tuple(map(int, bv[j][int(loc[1])]))
    if maxd2 < 0 or best_p is None or best_q is None:
        return 0.0, None, None
    return float(math.sqrt(maxd2)), best_p, best_q

def _v21_centroid_phys_mm_from_mask(mask: np.ndarray, spacing_zyx):
    """Return centroid (z,y,x) in mm. Wrapper around existing function."""
    return _v21_centroid_phys_mm(mask, spacing_zyx)


def _v21_dmax_centroid_mm(masks: list, spacing_zyx):
    pts = []
    for m in masks:
        if m is None:
            continue
        if np.count_nonzero(m) == 0:
            continue
        c = _v21_centroid_phys_mm_from_mask(m, spacing_zyx)
        if c is not None:
            pts.append(np.asarray(c, dtype=np.float64))
    if len(pts) < 2:
        return 0.0
    P = np.vstack(pts)  # (n,3) z,y,x in mm
    # pairwise max distance
    maxd = 0.0
    for i in range(len(P)):
        d = np.linalg.norm(P[i+1:] - P[i], axis=1) if i+1 < len(P) else np.array([])
        if d.size:
            md = float(np.max(d))
            if md > maxd:
                maxd = md
    return float(maxd)


def _v21_boundary_points_mm(mask: np.ndarray, spacing_zyx, max_points: int = 5000):
    """Return boundary voxel centers in mm as (N,3) with z,y,x ordering."""
    if mask is None or np.count_nonzero(mask) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # boundary voxels: mask minus eroded mask (26-connectivity via full 3x3x3 structure)
    try:
        from scipy.ndimage import binary_erosion
        struct = np.ones((3, 3, 3), dtype=bool)
        er = binary_erosion(mask.astype(bool), structure=struct)
        b = mask.astype(bool) & (~er)
    except Exception:
        # fallback: treat all voxels as candidates
        b = mask.astype(bool)

    zyx = np.argwhere(b)
    if zyx.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if zyx.shape[0] > max_points:
        # deterministic stride subsample
        step = int(math.ceil(zyx.shape[0] / max_points))
        zyx = zyx[::step]
    dz, dy, dx = [float(x) for x in spacing_zyx]
    pts = zyx.astype(np.float64)
    pts[:, 0] *= dz
    pts[:, 1] *= dy
    pts[:, 2] *= dx
    return pts


def _v21_dmax_vox_mm(masks: list, spacing_zyx, max_points_per_lesion: int = 5000, chunk: int = 512):
    """Optional DmaxVox: max distance between outermost voxels across lesions (approx, subsampled)."""
    # precompute boundary points per lesion
    pts_list = [_v21_boundary_points_mm(m, spacing_zyx, max_points=max_points_per_lesion) for m in masks if m is not None and np.count_nonzero(m) > 0]
    if len(pts_list) < 2:
        return 0.0

    maxd = 0.0
    for i in range(len(pts_list)):
        A = pts_list[i]
        for j in range(i + 1, len(pts_list)):
            B = pts_list[j]
            if A.size == 0 or B.size == 0:
                continue
            # chunked computation to avoid huge memory
            for k in range(0, A.shape[0], chunk):
                a = A[k:k+chunk]
                d2 = np.sum((a[:, None, :] - B[None, :, :]) ** 2, axis=2)
                md = float(np.sqrt(np.max(d2)))
                if md > maxd:
                    maxd = md
    return float(maxd)


def _v21_connected_components_masks(mask: np.ndarray, fully_connected: bool = True):
    """Return list of component masks using SimpleITK ConnectedComponent (26-connectivity if fully_connected)."""
    if mask is None:
        return []
    m = mask.astype(np.uint8)
    if np.count_nonzero(m) == 0:
        return []
    img = sitk.GetImageFromArray(m)  # z,y,x
    cc = sitk.ConnectedComponent(img, fullyConnected=bool(fully_connected))
    rel = sitk.RelabelComponent(cc, sortByObjectSize=True)
    lab = sitk.GetArrayFromImage(rel)
    n = int(lab.max())
    out = []
    for k in range(1, n + 1):
        out.append((lab == k))
    return out

class V21Tumor3DTab(QtWidgets.QWidget):
    """Tumor 3D metrics (single lesion + total burden), including shape + gETU + Dmax."""

    def __init__(self, app):
        super().__init__()
        self.app = app
        self._auto_masks = []
        self._build()

    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel(
            "3D tumor metrics computed from the current active VOI (primary or selected lesion)\n"
            "and from the total tumor burden (union of all lesions).\n\n"
            "Shape:\n"
            "• Sphericity Φ = S_sphere / S_voxel\n"
            "• Asphericity (%) = (S_voxel / S_sphere − 1) × 100\n"
            "where S_sphere is the surface area of a sphere with identical volume.\n\n"
            "gETU(a): (Δv · Σ SUVᵢᵃ)^(1/a) with Δv in mL (default a=1.5).\n"
            "Dmax: max Euclidean distance between lesion centroids (mm)."
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        # Controls
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("gETU exponent a:"))
        self.a_spin = QtWidgets.QDoubleSpinBox()
        self.a_spin.setDecimals(3)
        self.a_spin.setRange(0.05, 10.0)
        self.a_spin.setSingleStep(0.05)
        self.a_spin.setValue(1.5)
        self.a_spin.valueChanged.connect(self.refresh)
        ctrl.addWidget(self.a_spin)

        self.cb_dmaxvox = QtWidgets.QCheckBox("Compute DmaxVox (outermost voxels, slower)")
        self.cb_dmaxvox.stateChanged.connect(self.refresh)
        ctrl.addWidget(self.cb_dmaxvox)

        self.btn_auto = QtWidgets.QPushButton("Auto lesions from combined mask (26-connectivity)")
        self.btn_auto.clicked.connect(self._recompute_auto_lesions)
        ctrl.addWidget(self.btn_auto)

        ctrl.addStretch(1)
        lay.addLayout(ctrl)

        # Table
        self.tbl = QtWidgets.QTableWidget(0, 2)
        self.tbl.setHorizontalHeaderLabels(["Metric", "Value"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(_v21_qt_no_edit_triggers())
        lay.addWidget(self.tbl, 1)

        self.refresh()

    def _add_row(self, k: str, v: str):
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        self.tbl.setItem(r, 0, QtWidgets.QTableWidgetItem(k))
        self.tbl.setItem(r, 1, QtWidgets.QTableWidgetItem(v))

    def _get_total_burden_mask(self):
        masks = []
        m_primary = getattr(self.app, "mask", None)
        if m_primary is not None and np.count_nonzero(m_primary) > 0:
            masks.append(m_primary.astype(bool))
        for m in getattr(self.app, "meta_masks", []) or []:
            if m is not None and np.count_nonzero(m) > 0:
                masks.append(m.astype(bool))
        if not masks:
            return None
        out = masks[0].copy()
        for m in masks[1:]:
            out |= m
        return out

    def _recompute_auto_lesions(self):
        # Mode 2: user supplies a combined mask -> we derive lesions via 3D connected components (26-connectivity).
        combined = getattr(self.app, "combined_tumor_mask", None)
        if combined is None:
            combined = self._get_total_burden_mask()
        if combined is None:
            self._auto_masks = []
            self.refresh()
            return
        try:
            self._auto_masks = _v21_connected_components_masks(combined.astype(bool), fully_connected=True)
        except Exception:
            self._auto_masks = []
        self.refresh()

    def refresh(self):
        self.tbl.setRowCount(0)

        app = self.app
        mask_active = app._get_active_mask() if hasattr(app, "_get_active_mask") else getattr(app, "mask", None)
        pet_suv = getattr(app, "pet_suv", None)
        spacing = getattr(app, "spacing_zyx", (1.0, 1.0, 1.0))

        if mask_active is None or pet_suv is None:
            self._add_row("Status", "No PET/mask loaded.")
            return

        a = float(self.a_spin.value())

        # --- Active lesion ---
        self._add_row("Active VOI", f"{np.count_nonzero(mask_active)} voxels")
        vol_ml = _v21_mask_volume_ml(mask_active, spacing)
        shape = _v21_calc_sphericity_asphericity(mask_active, spacing)
        R = _v21_equiv_sphere_radius_mm(shape["vol_mm3"])

        self._add_row("Volume (mL)", f"{vol_ml:.3f}")
        self._add_row("Surface area (mm²)", f"{shape['surface_mm2']:.1f}")
        self._add_row("Equivalent radius R_eq (mm)", f"{R:.2f}")
        self._add_row("Sphericity Φ", f"{shape['sphericity']:.4f}")
        self._add_row("Asphericity (%)", f"{shape['asphericity_pct']:.2f}")

        # NHOC / NHOP (SUVmax hotspot)
        try:
            nh = _v21_compute_nhoc_nhop(mask_active, pet_suv, spacing)
        except Exception:
            nh = None
        if nh:
            self._add_row("SUVmax", f"{nh['suvmax']:.3f}")
            self._add_row("NHOCmax (hotspot→centroid / R_eq)", f"{nh['nhocmax']:.4f}  (d={nh['d_centroid_mm']:.2f} mm)")
            self._add_row("NHOPmax (hotspot→perimeter / R_eq)", f"{nh['nhopmax']:.4f}  (d={nh['d_perimeter_mm']:.2f} mm)")

        # gETU for active lesion
        getu_active = _v21_calc_getu(mask_active, pet_suv, spacing, a=a)
        self._add_row(f"gETU(a={a:.3f}) active", f"{getu_active:.4f}")

        # --- Total tumor burden ---
        total_mask = self._get_total_burden_mask()
        if total_mask is not None and np.count_nonzero(total_mask) > 0:
            vol_ml_t = _v21_mask_volume_ml(total_mask, spacing)
            shape_t = _v21_calc_sphericity_asphericity(total_mask, spacing)
            R_t = _v21_equiv_sphere_radius_mm(shape_t["vol_mm3"])
            getu_total = _v21_calc_getu(total_mask, pet_suv, spacing, a=a)

            self._add_row("", "")
            self._add_row("Total tumor burden (union)", f"{np.count_nonzero(total_mask)} voxels")
            self._add_row("Volume (mL) total", f"{vol_ml_t:.3f}")
            self._add_row("Sphericity Φ total", f"{shape_t['sphericity']:.4f}")
            self._add_row("Asphericity (%) total", f"{shape_t['asphericity_pct']:.2f}")
            self._add_row(f"gETU(a={a:.3f}) total", f"{getu_total:.4f}")
            self._add_row("R_eq total (mm)", f"{R_t:.2f}")

        # --- Dmax dissemination (two modes) ---
        self._add_row("", "")
        # Mode 1: manual lesions (meta_masks list)
        manual = [m.astype(bool) for m in (getattr(app, "meta_masks", []) or []) if m is not None and np.count_nonzero(m) > 0]
        if len(manual) >= 2:
            dmax = _v21_dmax_centroid_mm(manual, spacing)
            self._add_row("Dmax (manual lesions, centroid-centroid) [mm]", f"{dmax:.2f}")
            if self.cb_dmaxvox.isChecked():
                dmaxv = _v21_dmax_vox_mm(manual, spacing)
                self._add_row("DmaxVox (manual lesions, outer voxels) [mm]", f"{dmaxv:.2f}")
        else:
            self._add_row("Dmax (manual lesions)", "Need ≥2 manual lesions (Add Lesion).")

        # Mode 2: auto lesions (connected components from combined mask)
        if self._auto_masks and len(self._auto_masks) >= 2:
            dmax2 = _v21_dmax_centroid_mm(self._auto_masks, spacing)
            self._add_row("Dmax (auto lesions, centroid-centroid) [mm]", f"{dmax2:.2f}")
            if self.cb_dmaxvox.isChecked():
                dmaxv2 = _v21_dmax_vox_mm(self._auto_masks, spacing)
                self._add_row("DmaxVox (auto lesions, outer voxels) [mm]", f"{dmaxv2:.2f}")
        else:
            self._add_row("Dmax (auto lesions)", "Click 'Auto lesions…' to compute components (26-connectivity).")

        self.tbl.resizeColumnsToContents()

class V21NHOCNHOPTab(QtWidgets.QWidget):
    """Compute + export NHOCmax and NHOPmax for primary and each metastatic lesion."""
    def __init__(self, app):
        super().__init__()
        self.app = app
        self._build()

    def _build(self):
        lay = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel(
            "NHOCmax = distance(SUVmax voxel → lesion centroid) / R\n"
            "NHOPmax = distance(SUVmax voxel → lesion perimeter) / R\n"
            "R = radius of a sphere with the same lesion volume.\n"
            "Values are computed per-lesion from the current masks."
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        self.tbl = QtWidgets.QTableWidget(0, 10)
        self.tbl.setHorizontalHeaderLabels([
            "Label", "Type", "SUVmax", "Vol (mm³)", "R (mm)",
            "d_centroid (mm)", "d_perimeter (mm)", "NHOCmax", "NHOPmax", "Hotspot (z,y,x)"
        ])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.setEditTriggers(_v21_qt_no_edit_triggers())
        lay.addWidget(self.tbl, 1)

        btns = QtWidgets.QHBoxLayout()
        self.btn_refresh = QtWidgets.QPushButton("Compute / Refresh")
        self.btn_csv = QtWidgets.QPushButton("Export CSV")
        self.btn_masks = QtWidgets.QPushButton("Export label masks (.nii.gz)")
        btns.addWidget(self.btn_refresh)
        btns.addWidget(self.btn_csv)
        btns.addWidget(self.btn_masks)
        btns.addStretch(1)
        lay.addLayout(btns)

        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_csv.clicked.connect(self.export_csv)
        self.btn_masks.clicked.connect(self.export_label_masks)

        self._rows_cache = []
        self.refresh()

    def on_show(self):
        self.refresh()

    def refresh(self):
        self.tbl.setRowCount(0)
        self._rows_cache = []

        pet = getattr(self.app, "pet_suv", None)
        sp = getattr(self.app, "spacing_zyx", (1.0, 1.0, 1.0))
        if pet is None:
            return

        def _add_row(label, typ, res):
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)
            hotspot = res.get("hotspot_zyx")
            hotspot_txt = "" if hotspot is None else f"{hotspot[0]},{hotspot[1]},{hotspot[2]}"
            vals = [
                str(label),
                str(typ),
                f"{res.get('suvmax',0):.3f}",
                f"{res.get('vol_mm3',0):.1f}",
                f"{res.get('R_mm',0):.2f}",
                f"{res.get('d_centroid_mm',0):.2f}",
                f"{res.get('d_perimeter_mm',0):.2f}",
                f"{res.get('nhocmax',0):.4f}",
                f"{res.get('nhopmax',0):.4f}",
                hotspot_txt
            ]
            for c, v in enumerate(vals):
                self.tbl.setItem(r, c, QtWidgets.QTableWidgetItem(v))

            row = dict(label=label, type=typ, **res)
            row["hotspot_txt"] = hotspot_txt
            self._rows_cache.append(row)

        # Primary (label P1)
        pmask = getattr(self.app, "mask", None)
        if pmask is not None and np.any(pmask):
            res = _v21_compute_nhoc_nhop(pmask, pet, sp) or {}
            _add_row("P1", "Primary", res)
        else:
            _add_row("P1", "Primary", dict(vol_mm3=0.0, R_mm=0.0, suvmax=0.0, nhocmax=0.0, nhopmax=0.0,
                                           d_centroid_mm=0.0, d_perimeter_mm=0.0, hotspot_zyx=None))

        # Metastases (labels M1..MN)
        metas = getattr(self.app, "meta_masks", []) or []
        for i, mm in enumerate(metas, 1):
            if mm is None or not np.any(mm):
                _add_row(f"M{i}", "Metastasis", dict(vol_mm3=0.0, R_mm=0.0, suvmax=0.0, nhocmax=0.0, nhopmax=0.0,
                                                     d_centroid_mm=0.0, d_perimeter_mm=0.0, hotspot_zyx=None))
            else:
                res = _v21_compute_nhoc_nhop(mm, pet, sp) or {}
                _add_row(f"M{i}", "Metastasis", res)

        self.tbl.resizeColumnsToContents()

    def export_csv(self):
        if not self._rows_cache:
            QtWidgets.QMessageBox.information(self, "Export", "Nothing to export.")
            return
        fp, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save NHOC/NHOP table", "", "CSV Files (*.csv)")
        if not fp:
            return
        cols = ["Label","Type","SUVmax","Vol_mm3","R_mm","d_centroid_mm","d_perimeter_mm","NHOCmax","NHOPmax","Hotspot_zyx"]
        try:
            with open(fp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(cols)
                for r in self._rows_cache:
                    w.writerow([
                        r.get("label",""), r.get("type",""),
                        f"{r.get('suvmax',0):.6f}",
                        f"{r.get('vol_mm3',0):.3f}",
                        f"{r.get('R_mm',0):.6f}",
                        f"{r.get('d_centroid_mm',0):.6f}",
                        f"{r.get('d_perimeter_mm',0):.6f}",
                        f"{r.get('nhocmax',0):.8f}",
                        f"{r.get('nhopmax',0):.8f}",
                        r.get("hotspot_txt","")
                    ])
            QtWidgets.QMessageBox.information(self, "Export", f"Saved: {fp}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export", f"Failed to save CSV:\n{e}")

    def export_label_masks(self):
        """
        Export:
          - primary_mask.nii.gz (binary)
          - metastases_labelmask.nii.gz (0 background, 1..N metastases)
          - combined_labelmask.nii.gz (1 = primary, 2..N+1 metastases)
        """
        ref = getattr(self.app, "pet_img_sitk_raw", None)
        if ref is None:
            QtWidgets.QMessageBox.warning(self, "Export", "Reference geometry not available (load a study first).")
            return

        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output folder")
        if not out_dir:
            return

        try:
            pm = getattr(self.app, "mask", None)
            metas = getattr(self.app, "meta_masks", []) or []
            shape = None
            if pm is not None:
                shape = pm.shape
            elif metas and metas[0] is not None:
                shape = metas[0].shape
            else:
                QtWidgets.QMessageBox.information(self, "Export", "No masks available.")
                return

            # Primary
            primary_arr = (pm.astype(np.uint8) if pm is not None else np.zeros(shape, dtype=np.uint8))
            primary_img = sitk.GetImageFromArray(primary_arr)
            primary_img.CopyInformation(ref)
            sitk.WriteImage(primary_img, os.path.join(out_dir, "primary_mask.nii.gz"))

            # Metastasis label mask
            meta_label = _v21_build_metastasis_label_mask(metas, shape).astype(np.uint16)
            meta_img = sitk.GetImageFromArray(meta_label)
            meta_img.CopyInformation(ref)
            sitk.WriteImage(meta_img, os.path.join(out_dir, "metastases_labelmask.nii.gz"))

            # Combined
            comb = np.zeros(shape, dtype=np.uint16)
            if pm is not None:
                comb[pm.astype(bool)] = 1
            if np.any(meta_label):
                # metastases start from label 2
                comb[meta_label > 0] = (meta_label[meta_label > 0] + 1).astype(np.uint16)
            comb_img = sitk.GetImageFromArray(comb)
            comb_img.CopyInformation(ref)
            sitk.WriteImage(comb_img, os.path.join(out_dir, "combined_labelmask.nii.gz"))

            QtWidgets.QMessageBox.information(self, "Export", "Saved primary/multi-label masks.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export", f"Failed to export masks:\n{e}")

def _v21_attach_dock_tabs(app):
    """Add a right-side dock with Scan Info / PERCIST / RECIST / Display / Export / 3D Metrics / NHOC-NHOP tabs.
    Compatible with both PyQt5 and PyQt6 enum APIs.
    """
    if not isinstance(app, QtWidgets.QMainWindow):
        return

    # ---- Qt enum compatibility helpers ----
    def _qt_enum(root, *names, default=None):
        """Try nested enums (PyQt6) then flat attrs (PyQt5)."""
        cur = root
        for nm in names:
            if cur is None:
                return default
            cur = getattr(cur, nm, None)
        return cur if cur is not None else default

    def _dock_area(name: str):
        # PyQt5: QtCore.Qt.RightDockWidgetArea ; PyQt6: QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        return getattr(QtCore.Qt, name, None) or _qt_enum(QtCore.Qt, "DockWidgetArea", name)

    def _toolbar_area(name: str):
        # PyQt5: QtCore.Qt.TopToolBarArea ; PyQt6: QtCore.Qt.ToolBarArea.TopToolBarArea
        return getattr(QtCore.Qt, name, None) or _qt_enum(QtCore.Qt, "ToolBarArea", name)

    def _dock_feature(name: str):
        # PyQt5: QDockWidget.DockWidgetMovable ; PyQt6: QDockWidget.DockWidgetFeature.DockWidgetMovable
        return getattr(QtWidgets.QDockWidget, name, None) or _qt_enum(QtWidgets.QDockWidget, "DockWidgetFeature", name)

    # ---- Reuse existing dock if present ----
    existing = getattr(app, "_v21_tools_dock", None)
    if isinstance(existing, QtWidgets.QDockWidget):
        try:
            tabs = existing.widget()
            if isinstance(tabs, QtWidgets.QTabWidget):
                app._v21_tools_tabs = tabs
        except Exception:
            pass
        try:
            existing.setVisible(True)
            existing.show()
            existing.raise_()
        except Exception:
            pass
        return

    # ---- Build dock + tabs ----
    dock = QtWidgets.QDockWidget("Clinical Tools", app)
    allowed = (_dock_area("RightDockWidgetArea") or 0) | (_dock_area("LeftDockWidgetArea") or 0)
    try:
        dock.setAllowedAreas(allowed)
    except Exception:
        pass

    feats = (_dock_feature("DockWidgetMovable") or 0) | (_dock_feature("DockWidgetFloatable") or 0) | (_dock_feature("DockWidgetClosable") or 0)
    try:
        dock.setFeatures(feats)
    except Exception:
        pass

    tabs = QtWidgets.QTabWidget()
    tabs.setDocumentMode(True)
    app._v21_tools_tabs = tabs

    # Create tabs (owned by app so they can read app state)
    app.scaninfo_tab = V21ScanInfoTab(app)
    app.percist_tab = V21PercistTab(app)
    app.recist_tab = V21RecistTab(app)
    app.display_tab = V21DisplayTab(app)
    app.export_tab = V21ExportTab(app)
    app.metrics3d_tab = V21Tumor3DTab(app)
    app.nhoc_tab = V21NHOCNHOPTab(app)

    tabs.addTab(app.scaninfo_tab, "Scan Info")
    tabs.addTab(app.percist_tab, "PERCIST")
    tabs.addTab(app.recist_tab, "RECIST 1.1")
    tabs.addTab(app.display_tab, "Display")
    tabs.addTab(app.export_tab, "Export")
    tabs.addTab(app.metrics3d_tab, "3D Metrics")
    tabs.addTab(app.nhoc_tab, "NHOC/NHOP")

    # Auto-refresh on tab change
    def _refresh_current(_idx: int):
        w = tabs.currentWidget()
        for meth in ("on_show", "refresh", "update_ui", "rebuild", "populate", "_refresh"):
            fn = getattr(w, meth, None)
            if callable(fn):
                try:
                    fn()
                except Exception as e:
                    try:
                        QtWidgets.QMessageBox.warning(app, "Clinical Tools", f"Tab refresh failed: {e}")
                    except Exception:
                        pass
                break

    try:
        tabs.currentChanged.connect(_refresh_current)
    except Exception:
        pass

    dock.setWidget(tabs)

    area = _dock_area("RightDockWidgetArea") or _dock_area("LeftDockWidgetArea")
    try:
        if area is not None:
            app.addDockWidget(area, dock)
        else:
            # fallback
            app.addDockWidget(getattr(QtCore.Qt, "RightDockWidgetArea", QtCore.Qt.LeftDockWidgetArea), dock)
    except Exception:
        try:
            app.addDockWidget(getattr(QtCore.Qt, "RightDockWidgetArea", QtCore.Qt.LeftDockWidgetArea), dock)
        except Exception:
            try:
                app.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)
            except Exception:
                pass

    try:
        dock.setFloating(False)
        dock.show()
        dock.raise_()
    except Exception:
        pass

    app._v21_tools_dock = dock
    try:
        app.tabDock = dock  # back-compat for older patches
    except Exception:
        pass

# --- Capture PET raw Bq/mL and PET DICOM dataset during SUV computation ---
try:
    _v21_orig_compute = _compute_suvbw_from_pet_bqml
except Exception:
    _v21_orig_compute = None

def _compute_suvbw_from_pet_bqml(pet_bqml, pt_ds, *args, **kwargs):
    """Wrapper around the original SUV computation to capture PET raw Bq/mL and PET DICOM ds0 for Scan Info."""
    if _v21_orig_compute is None:
        return None, None
    suv, meta = _v21_orig_compute(pet_bqml, pt_ds, *args, **kwargs)
    try:
        globals()["_V21_LAST_PET_BQML"] = pet_bqml
        globals()["_V21_LAST_PET_DS0"] = pt_ds
    except Exception:
        pass
    return suv, meta


def _v21_find_first_pet_dataset(folder):
    try:
        import pydicom, os
        for root, _, files in os.walk(folder):
            for fn in files:
                fp = os.path.join(root, fn)
                try:
                    ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
                    mod = str(getattr(ds, "Modality", "")).upper()
                    if mod in ("PT", "PET"):
                        return ds
                except Exception:
                    continue
    except Exception:
        return None
    return None

def _v21_find_first_ct_dataset(folder):
    try:
        import pydicom, os
        for root, _, files in os.walk(folder):
            for fn in files:
                fp = os.path.join(root, fn)
                try:
                    ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
                    mod = str(getattr(ds, "Modality", "")).upper()
                    if mod == "CT":
                        return ds
                except Exception:
                    continue
    except Exception:
        return None
    return None

# --- Patch PETCTManualROIApp methods without editing their bodies ---
try:
    _v21_orig_load = PETCTManualROIApp._load_pet_ct
    def _load_pet_ct_patched(self, *a, **k):
        _v21_orig_load(self, *a, **k)
        # store PET Bq/mL and ds0 if captured
        self.pet_bqml = globals().get("_V21_LAST_PET_BQML", None)
        self.pet_ds0 = globals().get("_V21_LAST_PET_DS0", None)

        # Fallback: find PET/CT datasets for Scan Info
        if getattr(self, "dicom_dir", None):
            if self.pet_ds0 is None:
                try:
                    self.pet_ds0 = _v21_find_first_pet_dataset(self.dicom_dir)
                except Exception:
                    self.pet_ds0 = None
            try:
                self.ct_ds0 = _v21_find_first_ct_dataset(self.dicom_dir)
            except Exception:
                self.ct_ds0 = None

        # attach dock tabs after load (so metadata available)
        try:
            _v21_attach_dock_tabs(self)
        except Exception:
            pass
    PETCTManualROIApp._load_pet_ct = _load_pet_ct_patched
except Exception:
    pass

try:
    _v21_orig_build = PETCTManualROIApp._build_ui
    def _build_ui_patched(self, *a, **k):
        _v21_orig_build(self, *a, **k)
        try:
            _v21_attach_dock_tabs(self)
        except Exception:
            pass

        # Add a real toolbar/menu entry for Clinical Tools (prevents "floating, non-clickable" feel)
        try:
            def _jump_tab(title_contains: str):
                tabs = getattr(self, "_v21_tools_tabs", None)

                if tabs is None:
                     # Ensure dock/tabs exist (handles stale _v21_tools_dock attr)
                     try:
                         _v21_attach_dock_tabs(self)
                     except Exception:
                         pass
                     tabs = getattr(self, "_v21_tools_tabs", None)
                     if tabs is None:
                         return
                for i in range(tabs.count()):
                    if title_contains.lower() in tabs.tabText(i).lower():
                        tabs.setCurrentIndex(i)
                        return

            if not hasattr(self, "_v21_tools_toolbar"):
                tb = QtWidgets.QToolBar("Tools", self)
                tb.setMovable(False)
                self.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, tb)

                dock = getattr(self, "_v21_tools_dock", None)

                act_clin = QtGui.QAction("Clinical Tools", self)
                act_clin.setCheckable(True)
                act_clin.setChecked(True if (dock is None) else dock.isVisible())
                def _toggle_clin(on):
                    d = getattr(self, "_v21_tools_dock", None)
                    if d is not None:
                        d.setVisible(bool(on))
                        try:
                            d.raise_()
                        except Exception:
                            pass
                act_clin.toggled.connect(_toggle_clin)
                tb.addAction(act_clin)

                act_dock_right = QtGui.QAction("Dock Right", self)
                def _dock_right():
                    d = getattr(self, "_v21_tools_dock", None)
                    if d is None:
                        return
                    try:
                        d.setFloating(False)
                        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, d)
                        d.show(); d.raise_()
                    except Exception:
                        pass
                act_dock_right.triggered.connect(_dock_right)
                tb.addAction(act_dock_right)

                tb.addSeparator()

                act_burden = QtGui.QAction("Tumor Burden", self)
                act_burden.triggered.connect(lambda: self._show_tumor_burden())
                tb.addAction(act_burden)
                act_scan = QtGui.QAction("Scan Info", self)
                act_scan.triggered.connect(lambda: _jump_tab("Scan Info"))
                tb.addAction(act_scan)

                act_metrics = QtGui.QAction("3D Metrics", self)
                act_metrics.triggered.connect(lambda: _jump_tab("3D Metrics"))
                tb.addAction(act_metrics)

                act_nho = QtGui.QAction("NHOC/NHOP", self)
                act_nho.triggered.connect(lambda: _jump_tab("NHOC/NHOP"))
                tb.addAction(act_nho)

                self._v21_tools_toolbar = tb

            # Also expose in menu bar (Tools)
            mb = self.menuBar()
            tools = None
            for a in mb.actions():
                if a.text().lower() == "&tools" or a.text().lower() == "tools":
                    tools = a.menu()
                    break
            if tools is None:
                tools = mb.addMenu("Tools")
            # Ensure there is an action to toggle Clinical Tools
            if not hasattr(self, "_v21_menu_actions_added"):
                dock = getattr(self, "_v21_tools_dock", None)
                act = tools.addAction("Toggle Clinical Tools")
                act.triggered.connect(lambda: getattr(self, "_v21_tools_dock", None) and self._v21_tools_dock.setVisible(not self._v21_tools_dock.isVisible()))
                tools.addSeparator()
                tools.addAction("Show Tumor Burden").triggered.connect(lambda: self._show_tumor_burden())
                tools.addAction("Show Scan Info").triggered.connect(lambda: (getattr(self, "_v21_tools_dock", None) and self._v21_tools_dock.show(), _jump_tab("Scan Info")))
                tools.addAction("Show 3D Metrics").triggered.connect(lambda: (getattr(self, "_v21_tools_dock", None) and self._v21_tools_dock.show(), _jump_tab("3D Metrics")))
                tools.addAction("Show NHOC/NHOP").triggered.connect(lambda: (getattr(self, "_v21_tools_dock", None) and self._v21_tools_dock.show(), _jump_tab("NHOC/NHOP")))
                self._v21_menu_actions_added = True
        except Exception:
            pass
    PETCTManualROIApp._build_ui = _build_ui_patched
except Exception:
    pass





# =========================
# ══ BREAST DENSITY & COMPLEXITY MODULE ══
# Computation engines and PyQt6 tab integrated from petct_breast_density_unified_workstation
# All heavy dependencies are optional; guarded by BREAST_DENSITY_DEPS_AVAILABLE /
# BREAST_ML_DEPS_AVAILABLE flags so the viewer loads normally even without them.
# =========================

# --- Optional breast density computation dependencies ---
try:
    from scipy import ndimage as _bd_ndi
    from scipy.stats import entropy as _bd_scipy_entropy
    from scipy.stats import kurtosis as _bd_kurtosis, skew as _bd_skew
    from skimage.measure import marching_cubes as _bd_marching_cubes, mesh_surface_area as _bd_mesh_surface_area
    from skimage.morphology import (
        binary_closing as _bd_binary_closing,
        binary_opening as _bd_binary_opening,
        remove_small_objects as _bd_remove_small_objects,
    )
    BREAST_DENSITY_DEPS_AVAILABLE = True
except Exception:
    BREAST_DENSITY_DEPS_AVAILABLE = False

# --- Optional ML dependencies ---
try:
    import pandas as _bd_pd
    import matplotlib as _bd_matplotlib
    _bd_matplotlib.use("Agg")  # non-interactive backend safe for PyQt6
    import matplotlib.pyplot as _bd_plt
    from sklearn.ensemble import RandomForestClassifier as _bd_RFC
    from sklearn.linear_model import LogisticRegression as _bd_LR
    from sklearn.metrics import auc as _bd_auc, roc_curve as _bd_roc_curve
    from sklearn.model_selection import train_test_split as _bd_tts
    from sklearn.svm import SVC as _bd_SVC
    from sklearn.tree import DecisionTreeClassifier as _bd_DTC
    BREAST_ML_DEPS_AVAILABLE = True
except Exception:
    BREAST_ML_DEPS_AVAILABLE = False

# --- TotalSegmentator integration ---
# TotalSegmentator is REQUIRED for precise anatomical constraints.
# It derives the diaphragm Z-plane from real lung-lobe segmentations and the
# posterior Y-boundary from thoracic-vertebra segmentations.  Without it the
# breast segmentor falls back to a cruder HU-based heuristic that can extend
# below the diaphragm on difficult cases.
#
# Install once with:
#   pip install totalsegmentator nibabel
# (TotalSegmentator pulls in PyTorch automatically.)
try:
    from totalsegmentator.python_api import totalsegmentator as _bd_ts_api
    import nibabel as _bd_nib
    _BD_TS_AVAILABLE = True
except Exception as _bd_ts_import_exc:
    _BD_TS_AVAILABLE = False
    _bd_ts_api = None   # type: ignore[assignment]
    _bd_nib = None      # type: ignore[assignment]
    import sys as _sys
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════════════╗\n"
        "║  WARNING: TotalSegmentator / nibabel not found                  ║\n"
        "║                                                                  ║\n"
        "║  The breast segmentor will fall back to a HU-based heuristic    ║\n"
        "║  that is less accurate and may extend below the diaphragm on    ║\n"
        "║  difficult cases.                                                ║\n"
        "║                                                                  ║\n"
        "║  Install the required packages with:                            ║\n"
        "║      pip install totalsegmentator nibabel                       ║\n"
        "║                                                                  ║\n"
        "║  A CUDA GPU is optional but recommended for fast inference.     ║\n"
        f"║  Import error: {str(_bd_ts_import_exc)[:50]:<50s} ║\n"
        "╚══════════════════════════════════════════════════════════════════╝\n",
        file=_sys.stderr,
    )
    del _sys, _bd_ts_import_exc

# --- Constants ---
_BD_HU_FAT_MAX = -25
_BD_HU_BREAST_MIN = -300
_BD_HU_BREAST_MAX = 200
_BD_HU_MUSCLE_MIN = 40
_BD_HU_CLIP_MIN = 400
# IBSI-recommended fixed bin width for CT radiomics (HU units).
# VOXEL_BINS is derived so each bin spans exactly _BD_IBSI_BIN_WIDTH_HU Hounsfield units
# across the breast tissue range [_BD_HU_BREAST_MIN, _BD_HU_BREAST_MAX].
_BD_IBSI_BIN_WIDTH_HU = 25
_BD_VOXEL_BINS = int(np.ceil((_BD_HU_BREAST_MAX - _BD_HU_BREAST_MIN) / _BD_IBSI_BIN_WIDTH_HU))  # = 20

# Full 50-feature IBSI-aligned prespecified protocol panel.
# Shape (9), FirstOrder (11), GLCM (6), GLRLM (5), GLSZM (5), GLDM (5), NGTDM (5).
_BD_MANUSCRIPT_SHORTLIST_FEATURES = [
    # Shape (9)
    "parenchymal_volume_cc",
    "surface_area_mm2",
    "surface_to_volume_ratio",
    "sphericity",
    "max_3d_diameter_mm",
    "major_axis_length_mm",
    "minor_axis_length_mm",
    "elongation",
    "flatness",
    # First-order (11)
    "mean_hu",
    "median_hu",
    "percentile_10_hu",
    "percentile_90_hu",
    "iqr_hu",
    "variance_hu",
    "std_hu",
    "skewness",
    "kurtosis",
    "entropy",
    "uniformity",
    # GLCM (6)
    "glcm_joint_entropy",
    "glcm_contrast",
    "glcm_correlation",
    "glcm_difference_entropy",
    "glcm_idm",
    "glcm_imc1",
    # GLRLM (5)
    "glrlm_run_entropy",
    "glrlm_gray_level_non_uniformity",
    "glrlm_run_length_non_uniformity",
    "glrlm_short_run_emphasis",
    "glrlm_long_run_emphasis",
    # GLSZM (5)
    "glszm_zone_entropy",
    "glszm_gray_level_non_uniformity",
    "glszm_zone_size_non_uniformity",
    "glszm_small_area_emphasis",
    "glszm_large_area_emphasis",
    # GLDM (5)
    "gldm_dependence_entropy",
    "gldm_dependence_non_uniformity",
    "gldm_small_dependence_emphasis",
    "gldm_large_dependence_emphasis",
    "gldm_gray_level_non_uniformity",
    # NGTDM (5)
    "ngtdm_coarseness",
    "ngtdm_busyness",
    "ngtdm_complexity",
    "ngtdm_contrast",
    "ngtdm_strength",
]

# --- Helper functions ---
def _bd_safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _bd_sanitize_features(feats: dict) -> dict:
    """Replace any ``inf`` / ``-inf`` value in *feats* with ``float('nan')``.

    Infinite values (most commonly from division-by-zero in texture matrices)
    are invalid for statistical analysis and must not enter the final dataset.
    """
    return {
        k: (float("nan") if (isinstance(v, float) and not (v != v) and not (-1e308 < v < 1e308))
            else v)
        for k, v in feats.items()
    }


def _bd_voxel_volume_cc(voxel_spacing_mm) -> float:
    return float(np.prod(voxel_spacing_mm) / 1000.0)


def _bd_prepare_quantized_volume(ct_volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Clip HU and quantize masked voxels using fixed IBSI bin width.

    Fixed-bin-width discretization (IBSI recommendation for CT) ensures features
    are comparable across patients regardless of individual ROI intensity ranges.
    Each bin spans exactly _BD_IBSI_BIN_WIDTH_HU = 25 HU.
    """
    clipped = np.clip(ct_volume, _BD_HU_BREAST_MIN, _BD_HU_BREAST_MAX)
    quantized = np.floor((clipped - _BD_HU_BREAST_MIN) / _BD_IBSI_BIN_WIDTH_HU).astype(np.int32)
    quantized = np.clip(quantized, 0, _BD_VOXEL_BINS - 1)
    quantized[~mask] = 0
    return quantized


# --- Segmentation helpers ---
def _bd_detect_thorax_z_range(ct_volume: np.ndarray) -> tuple:
    """Estimate the Z-slice range covering the thorax (chest), excluding abdomen.

    Strategy:
    1. Compute the per-slice fraction of lung-equivalent voxels (HU -950 to -200).
    2. Use a *self-calibrating* threshold (max(0.05, peak_frac * 0.20)) so that
       scattered bowel/stomach gas — which typically occupies only 1–5 % of
       abdominal pixels — is rejected.  Real lung parenchyma routinely fills
       15–40 % of thoracic slice pixels.
    3. Find the superior lung border (z_top) and locate the diaphragm by
       detecting the first sustained drop in the lung-fraction signal: the
       last slice whose *smoothed* value still exceeds the threshold AND whose
       *raw* value 3 slices later has already fallen below a low-noise floor
       (0.02).  This prevents bowel gas below the diaphragm from pushing
       z_diaphragm into the abdomen.
    4. Apply a hard safety cap: z_diaphragm ≤ 75 % of the total slice count.
    5. Return (z_top, z_diaphragm).  Falls back to top 50 % of volume if no
       clear lung signal is found.
    """
    nz = ct_volume.shape[0]
    lung_hu_min, lung_hu_max = -950, -200

    # Per-slice lung voxel fraction
    lung_frac = np.array([
        float(np.mean((ct_volume[z] >= lung_hu_min) & (ct_volume[z] <= lung_hu_max)))
        for z in range(nz)
    ])

    # Smooth with a small window to reduce noise
    kernel = np.ones(5, dtype=np.float64) / 5.0
    lung_frac_smooth = np.convolve(lung_frac, kernel, mode="same")

    peak_frac = float(lung_frac_smooth.max())
    if peak_frac < 0.05:
        # No clear lung signal — fall back to top 50 % of volume
        return 0, nz // 2

    # Self-calibrating threshold: at least 5 % AND 20 % of the peak fraction.
    # This rejects bowel/stomach gas which fills only 1–5 % of abdominal pixels.
    threshold = max(0.05, peak_frac * 0.20)
    lung_present = lung_frac_smooth >= threshold
    lung_z = np.where(lung_present)[0]

    if lung_z.size < 5:
        return 0, nz // 2

    z_top = int(lung_z[0])
    z_lung_end = int(lung_z[-1])

    # Diaphragm detection: walk backwards from z_lung_end to find where the
    # lung signal *truly* disappears.  A 3-slice look-ahead in the raw (un-
    # smoothed) signal must drop below 0.02 to confirm the lung has ended.
    # This prevents a few gas-containing abdominal slices from dragging
    # z_lung_end further inferior.
    low_floor = 0.02
    look_ahead = 3
    for z in range(z_lung_end, z_top, -1):
        # Check that the next look_ahead raw slices are all below the noise floor
        future_end = min(z + look_ahead, nz - 1)
        if np.all(lung_frac[z:future_end + 1] < low_floor):
            z_lung_end = z
            break

    # Add a 2-slice inferior margin to capture the full inferior breast.
    z_diaphragm = min(z_lung_end + 2, nz - 1)

    # Hard cap: diaphragm cannot exceed 75 % of the total scan extent.
    z_diaphragm = min(z_diaphragm, int(nz * 0.75))

    return z_top, z_diaphragm


def _bd_get_body_outline(ct_volume: np.ndarray) -> np.ndarray:
    """Return a 3-D boolean mask covering the patient body (excluding table/air).

    Air outside the patient has HU < -500. Filling the background gives the body.
    The table is handled by taking the largest connected component of non-air voxels.
    """
    body = ct_volume > -500  # non-air
    body = _bd_binary_closing(body, footprint=np.ones((3, 3, 3)))
    # Keep largest connected component = body (excludes small noise islands)
    try:
        labeled, n = _bd_ndi.label(body)
        if n > 0:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0  # background
            largest = int(np.argmax(sizes))
            body = labeled == largest
    except Exception:
        pass
    return body.astype(bool)


def _bd_keep_largest_anterior_component(mask: np.ndarray, ny: int,
                                         anterior_y_fraction: float = 0.65) -> np.ndarray:
    """Return the largest connected component whose centroid lies in the
    anterior (low-Y) portion of the volume.

    After HU thresholding and Z/Y ROI filtering, residual liver, spleen, and
    subdiaphragmatic soft-tissue fragments may still survive.  They are not
    spatially connected to the anterior breast mound, so taking the largest
    anteriorly-centred connected component reliably isolates the true breast
    tissue for each side.

    Parameters
    ----------
    mask:
        Boolean 3-D candidate mask for one side (right or left).
    ny:
        Number of voxels in the Y (anterior–posterior) axis.
    anterior_y_fraction:
        Centroid Y must be below ``ny * anterior_y_fraction`` to be considered
        an anterior (breast) structure.  Default 0.65 keeps the anterior 65 %
        of AP depth, which safely includes the entire breast mound while
        excluding posterior abdominal structures.

    Returns
    -------
    Boolean 3-D array of the same shape as ``mask``.
    """
    if not mask.any():
        return mask

    labeled, n = _bd_ndi.label(mask)
    if n == 0:
        return mask

    y_threshold = ny * anterior_y_fraction  # centroid Y must be below this

    best_label = 0
    best_size = 0

    for lbl in range(1, n + 1):
        comp = labeled == lbl
        size = int(comp.sum())
        if size == 0:
            continue
        zz, yy, xx = np.nonzero(comp)
        centroid_y = float(yy.mean())
        if centroid_y < y_threshold and size > best_size:
            best_size = size
            best_label = lbl

    if best_label == 0:
        # No anteriorly-centred component found — fall back to overall largest
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        best_label = int(np.argmax(sizes))

    return (labeled == best_label).astype(bool)


# --- TotalSegmentator anatomy-constraint helpers ---

# Structures used to locate the diaphragm (inferior boundary of lungs)
_BD_TS_LUNG_STRUCTURES = [
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
]

# Thoracic vertebrae used to establish the posterior breast boundary
_BD_TS_VERTEBRAE_STRUCTURES = [
    "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4",
    "vertebrae_T5", "vertebrae_T6", "vertebrae_T7", "vertebrae_T8",
    "vertebrae_T9", "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
]


def _bd_sitk_to_nib(sitk_img: "sitk.Image") -> "_bd_nib.Nifti1Image":
    """Convert a SimpleITK image to a nibabel Nifti1Image.

    SimpleITK stores arrays as (Z, Y, X) in LPS physical coordinates.
    NIfTI / nibabel uses (X, Y, Z) in RAS coordinates.  This function
    transposes the data array and builds the correct RAS affine.
    """
    arr_zyx = sitk.GetArrayFromImage(sitk_img).astype(np.float32)  # (Z, Y, X)
    arr_xyz = arr_zyx.transpose(2, 1, 0)                           # (X, Y, Z)

    spacing = sitk_img.GetSpacing()     # (sx, sy, sz) mm  — ITK order
    origin  = sitk_img.GetOrigin()      # (ox, oy, oz) LPS
    direction = np.array(sitk_img.GetDirection()).reshape(3, 3)  # 3×3 LPS

    # Build LPS affine: p_LPS = direction @ diag(spacing) @ index + origin
    affine_lps = np.eye(4)
    affine_lps[:3, :3] = direction @ np.diag(spacing)
    affine_lps[:3, 3] = origin

    # Convert LPS → RAS (flip X and Y sign)
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine_ras = lps_to_ras @ affine_lps

    return _bd_nib.Nifti1Image(arr_xyz, affine_ras)


def _bd_ts_anatomy_constraints(ct_sitk: "sitk.Image") -> dict:
    """Run TotalSegmentator (lung lobes + thoracic vertebrae) and extract
    anatomical boundaries for breast segmentation.

    Uses ``roi_subset`` to request only the two structure groups so the run
    is faster than a full 104-class inference.  The ``fast=True`` flag uses
    the 3 mm low-resolution model which is sufficient for boundary detection.

    Returns a dict with any subset of:

    ``z_top``
        Superior-most Z index (array coords) that contains lung signal.
    ``z_diaphragm``
        Inferior-most Z index with lung signal + 5-slice margin.
    ``y_posterior_vertebrae``
        Anterior face of the thoracic vertebrae (array Y-axis) minus a
        5-voxel clearance margin.  Used as the posterior breast boundary.

    Returns an empty dict if TotalSegmentator is unavailable, the model
    weights are absent, or any other error occurs.  Callers fall back to
    the HU-based heuristics automatically.
    """
    if not _BD_TS_AVAILABLE or ct_sitk is None:
        return {}
    try:
        nib_img = _bd_sitk_to_nib(ct_sitk)
        roi_subset = _BD_TS_LUNG_STRUCTURES + _BD_TS_VERTEBRAE_STRUCTURES
        nz, ny, nx = sitk.GetArrayFromImage(ct_sitk).shape  # (Z, Y, X)

        with tempfile.TemporaryDirectory(prefix="bd_ts_") as tmpdir:
            out_dir = os.path.join(tmpdir, "segs")
            _bd_ts_api(
                nib_img, out_dir,
                task="total",
                roi_subset=roi_subset,
                fast=True,          # 3 mm model — sufficient for boundary detection
                verbose=False,
                statistics=False,
                radiomics=False,
            )

            def _load_mask(structures, shape_zyx):
                mask = np.zeros(shape_zyx, dtype=bool)
                for s in structures:
                    p = os.path.join(out_dir, f"{s}.nii.gz")
                    if os.path.exists(p):
                        # nibabel data is (X, Y, Z); transpose to (Z, Y, X)
                        seg_xyz = _bd_nib.load(p).get_fdata() > 0.5
                        mask |= seg_xyz.transpose(2, 1, 0)
                return mask

            lung_mask = _load_mask(_BD_TS_LUNG_STRUCTURES, (nz, ny, nx))
            vert_mask = _load_mask(_BD_TS_VERTEBRAE_STRUCTURES, (nz, ny, nx))

        result: dict = {}

        # --- Diaphragm from lung Z extent ---
        lung_z = np.where(lung_mask.any(axis=(1, 2)))[0]
        if lung_z.size >= 5:
            result["z_top"] = int(lung_z[0])
            result["z_diaphragm"] = min(int(lung_z[-1]) + 5, nz - 1)

        # --- Posterior boundary from vertebrae Y extent ---
        # In standard axial CT (supine, head-first) Y=0 is anterior and
        # Y increases toward posterior, so the vertebrae occupy high Y indices.
        # We use the minimum Y of the vertebral mask (its anterior face) as the
        # posterior breast boundary, minus a 5-voxel clearance.
        if vert_mask.any():
            vert_y = np.where(vert_mask.any(axis=(0, 2)))[0]
            if vert_y.size > 0:
                vert_y_mean = float(vert_y.mean())
                if vert_y_mean > ny / 2:
                    # Vertebrae are in the posterior (high-Y) half — standard orientation
                    y_boundary = max(1, int(vert_y.min()) - 5)
                else:
                    # Vertebrae are in the anterior (low-Y) half — flipped Y axis
                    y_boundary = min(ny - 1, int(vert_y.max()) + 5)
                result["y_posterior_vertebrae"] = y_boundary

        return result

    except Exception:
        return {}


# --- Segmentation ---
class BreastSegmentor:
    """Performs whole-breast and fibroglandular segmentation on CT volumes.

    Anatomical constraints applied to avoid liver/spleen/vertebrae false-positives:

    1. **Z-range (superior–inferior)**: segmentation is restricted to the thorax
       by detecting the diaphragm (inferior boundary of the lung fields).
       Everything below the diaphragm is excluded.
    2. **Anterior crop (A–P direction)**: breasts are anterior structures.
       The posterior 40 % of the AP depth (Y-axis) is excluded before any
       HU-based thresholding. This eliminates spine, vertebrae, and posterior
       chest-wall structures.
    3. **Body outline**: an air-threshold body mask is used to exclude the CT
       table and out-of-body noise.
    4. **Left/Right split**: the midline split is along the X-axis (columns)
       which corresponds to patient left/right in standard axial CT orientation.
    5. **Largest anterior connected component**: after per-side splitting, only
       the largest connected component whose centroid lies in the anterior 65 %
       of AP depth is retained.  This discards liver, spleen, and any
       subdiaphragmatic structures that are not contiguous with the breast mound.
    """

    # Fraction of the AP (Y) depth to keep as the anterior breast region.
    # 0.55 means the anterior 55 % of AP depth; posterior 45 % is excluded.
    _ANTERIOR_FRACTION = 0.55

    def _breast_roi_mask(self, ct_volume: np.ndarray,
                         ct_sitk=None) -> np.ndarray:
        """Build a 3-D Boolean mask restricting analysis to the anterior thorax.

        Combines:
          * Z range  — thorax only (lungs detected; below diaphragm excluded)
          * Y range  — anterior fraction only (eliminates spine/posterior structures)
          * Body outline — exclude air / CT table

        When TotalSegmentator is installed and ``ct_sitk`` is supplied the Z
        and Y boundaries are derived from actual lung-lobe and thoracic-vertebra
        segmentations, giving much more precise anatomical constraints than the
        HU-based fall-backs.
        """
        nz, ny, nx = ct_volume.shape

        # --- Try TotalSegmentator for precise anatomical boundaries first ---
        ts = _bd_ts_anatomy_constraints(ct_sitk) if (_BD_TS_AVAILABLE and ct_sitk is not None) else {}

        # 1. Detect thorax Z range
        if "z_top" in ts and "z_diaphragm" in ts:
            z_top  = ts["z_top"]
            z_diaph = ts["z_diaphragm"]
            self._last_seg_backend = "TotalSegmentator"
        else:
            try:
                z_top, z_diaph = _bd_detect_thorax_z_range(ct_volume)
            except Exception:
                z_top, z_diaph = 0, nz // 2
            self._last_seg_backend = "HU-heuristic"

        # 2. Posterior Y boundary (anterior breast limit)
        if "y_posterior_vertebrae" in ts:
            y_end = ts["y_posterior_vertebrae"]
        else:
            y_end = max(1, int(ny * self._ANTERIOR_FRACTION))

        # 3. Body outline to exclude air and table
        try:
            body = _bd_get_body_outline(ct_volume)
        except Exception:
            body = ct_volume > -500

        # Combine: thorax Z slices × anterior Y × body mask
        roi = np.zeros((nz, ny, nx), dtype=bool)
        roi[z_top:z_diaph + 1, :y_end, :] = True
        roi &= body
        return roi

    def segment_whole_breast(self, ct_volume: np.ndarray,
                             roi_mask: np.ndarray = None,
                             ct_sitk=None) -> dict:
        """Segment left and right whole-breast regions.

        Parameters
        ----------
        ct_volume:
            3-D CT array in HU, shape (Z, Y, X) in standard axial orientation.
        roi_mask:
            Optional pre-computed anterior-thorax ROI mask.  If not supplied it
            is computed automatically via ``_breast_roi_mask``.
        ct_sitk:
            Optional SimpleITK image of the same CT volume.  When supplied and
            TotalSegmentator is installed, the ROI mask is derived from actual
            anatomical segmentations rather than HU-based heuristics.

        Returns
        -------
        dict with keys ``right_mask`` and ``left_mask`` (Boolean 3-D arrays).
        """
        nz, ny, nx = ct_volume.shape

        # Step 1 — restrict to anterior thorax
        if roi_mask is None:
            roi_mask = self._breast_roi_mask(ct_volume, ct_sitk=ct_sitk)

        # Step 2 — HU-based breast tissue candidate mask within the ROI
        breast_hu = (ct_volume >= _BD_HU_BREAST_MIN) & (ct_volume <= _BD_HU_BREAST_MAX)
        breast = breast_hu & roi_mask

        # Step 3 — morphological clean-up
        breast = _bd_binary_closing(breast, footprint=np.ones((3, 3, 3)))
        breast = _bd_remove_small_objects(breast, min_size=2_000)

        # Step 4 — exclude posterior 40 % of AP depth (spine, pectoralis, ribs)
        #           within the breast HU candidates (belt-and-suspenders check)
        y_post_start = max(1, int(ny * self._ANTERIOR_FRACTION))
        breast[:, y_post_start:, :] = False

        # Step 5 — left / right split along X midline
        midline = nx // 2
        right_mask = np.zeros_like(breast, dtype=bool)
        left_mask = np.zeros_like(breast, dtype=bool)
        right_mask[:, :, :midline] = breast[:, :, :midline]
        left_mask[:, :, midline:] = breast[:, :, midline:]

        right_mask = _bd_remove_small_objects(right_mask, min_size=1_000)
        left_mask = _bd_remove_small_objects(left_mask, min_size=1_000)

        # Step 6 — keep only the largest anteriorly-centred connected component
        # per side.  Liver, spleen, and subdiaphragmatic soft-tissue fragments
        # that survive Z/Y/HU filtering are not spatially connected to the
        # anterior breast mound, so this step eliminates them reliably.
        right_mask = _bd_keep_largest_anterior_component(right_mask, ny)
        left_mask = _bd_keep_largest_anterior_component(left_mask, ny)

        return {"right_mask": right_mask, "left_mask": left_mask}

    def segment_fibroglandular(self, ct_volume: np.ndarray, breast_mask: np.ndarray) -> np.ndarray:
        fg = (ct_volume > _BD_HU_FAT_MAX) & breast_mask
        fg = _bd_binary_opening(fg, footprint=np.ones((3, 3, 3)))
        fg = _bd_binary_closing(fg, footprint=np.ones((3, 3, 3)))
        return _bd_remove_small_objects(fg, min_size=200)

    def exclude_non_parenchymal(self, fg_mask: np.ndarray, ct_volume: np.ndarray) -> np.ndarray:
        cleaned = fg_mask.copy()
        if not cleaned.any():
            return cleaned

        # Use the breast ROI boundary (HU range) to derive the skin shell —
        # only within the anterior thorax so we don't touch abdominal organs.
        breast_hu = (ct_volume >= _BD_HU_BREAST_MIN) & (ct_volume <= _BD_HU_BREAST_MAX)
        # Limit to anterior half to avoid accidental posterior structures
        ny = ct_volume.shape[1]
        y_post_start = max(1, int(ny * self._ANTERIOR_FRACTION))
        breast_hu[:, y_post_start:, :] = False
        breast_mask = _bd_binary_closing(breast_hu, footprint=np.ones((3, 3, 3)))

        inner = _bd_ndi.binary_erosion(breast_mask, iterations=2)
        skin_shell = breast_mask & ~inner
        cleaned &= ~skin_shell

        # Pectoralis: posterior region HU > muscle threshold
        posterior_region = np.zeros_like(cleaned, dtype=bool)
        # Posterior starts at 50% of AP depth (conservative — only flag rear half)
        posterior_start = int(ct_volume.shape[1] * 0.50)
        posterior_region[:, posterior_start:, :] = True
        pectoralis = posterior_region & (ct_volume > _BD_HU_MUSCLE_MIN)
        cleaned &= ~pectoralis

        # Metal clips: small bright objects
        clips = (ct_volume > _BD_HU_CLIP_MIN) & cleaned
        labels, n_labels = _bd_ndi.label(clips)
        if n_labels:
            sizes = np.bincount(labels.ravel())
            small_ids = np.nonzero(sizes[1:] <= 500)[0] + 1
            if small_ids.size:
                cleaned &= ~np.isin(labels, small_ids)

        # Remove single largest bright component (heuristic tumor exclusion)
        tumor_candidates = cleaned & (ct_volume > 20)
        labels, n_labels = _bd_ndi.label(tumor_candidates)
        if n_labels:
            sizes = np.bincount(labels.ravel())
            largest_idx = int(np.argmax(sizes[1:]) + 1)
            cleaned &= labels != largest_idx

        return _bd_remove_small_objects(cleaned, min_size=100)

    def compute_exclusion_masks(self, ct_volume: np.ndarray, breast_mask: np.ndarray) -> dict:
        """Return a dict of sub-tissue boolean masks for a single breast ROI.

        Keys
        ----
        ``skin``        — 2-voxel shell at the breast boundary
        ``pectoralis``  — posterior-region HU >40 (muscle) within the breast boundary
        ``clips``       — small high-HU objects (HU > 400) — metal clips / calcifications
        ``vessels``     — thin bright strands (HU 80–400) excluded after clip removal
        """
        inner = _bd_ndi.binary_erosion(breast_mask, iterations=2)
        skin = breast_mask & ~inner

        # Pectoralis: posterior 25 % of the breast's own AP (Y) extent, not
        # the global 50 % of the CT volume.  Using the breast mask's Y span
        # correctly restricts the pectoralis layer to the back of the breast
        # mound and avoids labelling structures outside the breast ROI.
        posterior_region = np.zeros(ct_volume.shape, dtype=bool)
        if breast_mask.any():
            y_coords = np.where(breast_mask.any(axis=(0, 2)))[0]
            if y_coords.size >= 2:
                y_ant = int(y_coords.min())
                y_post = int(y_coords.max())
                y_pec_thresh = y_ant + max(1, int((y_post - y_ant) * 0.75))
            else:
                y_pec_thresh = int(ct_volume.shape[1] * 0.50)
        else:
            y_pec_thresh = int(ct_volume.shape[1] * 0.50)
        posterior_region[:, y_pec_thresh:, :] = True
        pectoralis = posterior_region & (ct_volume > _BD_HU_MUSCLE_MIN) & breast_mask

        clips_raw = (ct_volume > _BD_HU_CLIP_MIN) & breast_mask
        labels, n_labels = _bd_ndi.label(clips_raw)
        clips = np.zeros_like(clips_raw)
        if n_labels:
            sizes = np.bincount(labels.ravel())
            small_ids = np.nonzero(sizes[1:] <= 500)[0] + 1
            if small_ids.size:
                clips = np.isin(labels, small_ids)

        # Vessels: bright soft-tissue strands (HU 80–400) not already assigned to pectoralis/clips
        vessels = (ct_volume >= 80) & (ct_volume <= _BD_HU_CLIP_MIN) & breast_mask & ~pectoralis & ~clips
        vessels = _bd_remove_small_objects(vessels.astype(bool), min_size=20)

        return {
            "skin": skin.astype(bool),
            "pectoralis": pectoralis.astype(bool),
            "clips": clips.astype(bool),
            "vessels": vessels.astype(bool),
        }


# --- Density Engine ---
class BreastCTDensityEngine:
    """Computes bilateral volumetric breast density metrics from CT."""

    def __init__(self) -> None:
        self.segmentor = BreastSegmentor()

    def compute_volumetric_density(self, ct_volume: np.ndarray,
                                   voxel_spacing_mm=(1.0, 1.0, 1.0),
                                   ct_sitk=None) -> dict:
        vox_cc = _bd_voxel_volume_cc(voxel_spacing_mm)
        whole_masks = self.segmentor.segment_whole_breast(ct_volume, ct_sitk=ct_sitk)

        def _side_metrics(side: str) -> dict:
            breast_mask = whole_masks[f"{side}_mask"]
            fg_mask = self.segmentor.segment_fibroglandular(ct_volume, breast_mask)
            cleaned = self.segmentor.exclude_non_parenchymal(fg_mask, ct_volume)
            whole_vol = float(np.count_nonzero(breast_mask)) * vox_cc
            fg_vol = float(np.count_nonzero(cleaned)) * vox_cc
            fat_vol = max(0.0, whole_vol - fg_vol)
            return {
                f"{side}_whole_breast_vol_cc": round(whole_vol, 3),
                f"{side}_fibroglandular_vol_cc": round(fg_vol, 3),
                f"{side}_fat_vol_cc": round(fat_vol, 3),
                f"{side}_volumetric_density_pct": round(_bd_safe_div(fg_vol, whole_vol) * 100.0, 3),
            }

        right = _side_metrics("right")
        left = _side_metrics("left")
        total_fg = right["right_fibroglandular_vol_cc"] + left["left_fibroglandular_vol_cc"]
        total_whole = right["right_whole_breast_vol_cc"] + left["left_whole_breast_vol_cc"]
        bilateral = {
            "bilateral_whole_breast_vol_cc": round(total_whole, 3),
            "bilateral_fibroglandular_vol_cc": round(total_fg, 3),
            "bilateral_volumetric_density_pct": round(_bd_safe_div(total_fg, total_whole) * 100.0, 3),
            "density_asymmetry_pct": round(
                abs(right["right_volumetric_density_pct"] - left["left_volumetric_density_pct"]), 3
            ),
        }
        return {**right, **left, **bilateral}


# --- Shape Features ---
class ShapeFeatures:
    """3-D shape radiomics for the parenchymal mask."""

    def compute(self, parenchymal_mask: np.ndarray, voxel_spacing_mm=(1.0, 1.0, 1.0)) -> dict:
        vox_cc = _bd_voxel_volume_cc(voxel_spacing_mm)
        volume_cc = float(np.count_nonzero(parenchymal_mask)) * vox_cc
        try:
            # voxel_spacing_mm is (z,y,x); marching_cubes 'spacing' follows the same axis order
            # as the array (row, col, depth) = (z, y, x), so pass them in (z,y,x) order.
            z_sp, y_sp, x_sp = voxel_spacing_mm
            spacing_zyx_mc = (z_sp, y_sp, x_sp)
            verts, faces, _, _ = _bd_marching_cubes(parenchymal_mask.astype(np.float32), level=0.5, spacing=spacing_zyx_mc)
            surface_mm2 = float(_bd_mesh_surface_area(verts, faces))
            volume_mm3 = volume_cc * 1000.0
            svr = _bd_safe_div(surface_mm2, volume_mm3)
            sphericity = _bd_safe_div((np.pi ** (1.0 / 3.0)) * ((6.0 * volume_mm3) ** (2.0 / 3.0)), surface_mm2)
        except Exception:
            svr = float("nan")
            sphericity = float("nan")
            surface_mm2 = float("nan")
            volume_mm3 = volume_cc * 1000.0

        try:
            coords = np.argwhere(parenchymal_mask)
            if len(coords) >= 3:
                scaled = coords * np.array(voxel_spacing_mm)
                cov = np.cov(scaled.T)
                eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
                eigvals = np.maximum(eigvals, 1e-8)
                elongation = float(np.sqrt(_bd_safe_div(eigvals[1], eigvals[0])))
                flatness = float(np.sqrt(_bd_safe_div(eigvals[2], eigvals[0])))
                major_axis = float(4.0 * np.sqrt(eigvals[0]))
                minor_axis = float(4.0 * np.sqrt(eigvals[2]))
                bbox_mm = scaled.max(axis=0) - scaled.min(axis=0)
                max_diameter = float(np.linalg.norm(bbox_mm))
            else:
                elongation = flatness = float("nan")
                major_axis = minor_axis = max_diameter = float("nan")
        except Exception:
            elongation = flatness = float("nan")
            major_axis = minor_axis = max_diameter = float("nan")

        return {
            "parenchymal_volume_cc": round(volume_cc, 4),
            "volume_cc": round(volume_cc, 4),
            "surface_area_mm2": round(surface_mm2, 4) if surface_mm2 == surface_mm2 else float("nan"),
            "surface_to_volume_ratio": round(svr, 6) if svr == svr else float("nan"),
            "sphericity": round(sphericity, 6) if sphericity == sphericity else float("nan"),
            "elongation": round(elongation, 6) if elongation == elongation else float("nan"),
            "flatness": round(flatness, 6) if flatness == flatness else float("nan"),
            "major_axis_length_mm": round(major_axis, 4) if major_axis == major_axis else float("nan"),
            "minor_axis_length_mm": round(minor_axis, 4) if minor_axis == minor_axis else float("nan"),
            "max_3d_diameter_mm": round(max_diameter, 4) if max_diameter == max_diameter else float("nan"),
        }


# --- First Order Features ---
class FirstOrderFeatures:
    """Intensity histogram statistics for masked voxels."""

    def compute(self, ct_volume: np.ndarray, parenchymal_mask: np.ndarray) -> dict:
        voxels = ct_volume[parenchymal_mask].astype(np.float64)
        if len(voxels) == 0:
            return {k: float("nan") for k in ["mean_hu", "median_hu", "std_hu", "min_hu", "max_hu",
                                      "range_hu", "skewness", "kurtosis", "entropy", "uniformity",
                                      "energy", "percentile_10_hu", "percentile_90_hu", "iqr_hu",
                                      "variance_hu"]}
        quantized = _bd_prepare_quantized_volume(ct_volume, parenchymal_mask)
        q_vals = quantized[parenchymal_mask]
        counts = np.bincount(q_vals, minlength=_BD_VOXEL_BINS).astype(np.float64)
        probs = counts / (counts.sum() + 1e-12)
        ent = float(_bd_scipy_entropy(probs + 1e-12))
        uniformity = float(np.sum(probs ** 2))
        return {
            "mean_hu": round(float(np.mean(voxels)), 4),
            "median_hu": round(float(np.median(voxels)), 4),
            "std_hu": round(float(np.std(voxels)), 4),
            "variance_hu": round(float(np.var(voxels)), 4),
            "min_hu": round(float(np.min(voxels)), 4),
            "max_hu": round(float(np.max(voxels)), 4),
            "range_hu": round(float(np.ptp(voxels)), 4),
            "skewness": round(float(_bd_skew(voxels)), 6),
            "kurtosis": round(float(_bd_kurtosis(voxels)), 6),
            "entropy": round(ent, 6),
            "uniformity": round(uniformity, 6),
            "energy": round(float(np.sum(voxels ** 2)), 2),
            "percentile_10_hu": round(float(np.percentile(voxels, 10)), 4),
            "percentile_90_hu": round(float(np.percentile(voxels, 90)), 4),
            "iqr_hu": round(float(np.percentile(voxels, 75) - np.percentile(voxels, 25)), 4),
        }


# --- Texture Feature Base ---
class _BDTextureFeatureBase:
    @staticmethod
    def _pyradiomics_features(ct_volume: np.ndarray, mask: np.ndarray, feature_class: str, names: dict) -> dict:
        """Run pyradiomics on the locked IBSI-compliant feature panel.

        Settings:
          binWidth=25          — fixed HU bin width (IBSI, not relative binCount)
          voxelArrayShift=0    — IBSI first-order alignment
          normalize=False      — raw HU values used (CT-appropriate)
          resampledPixelSpacing=[1.2,1.2,1.2] — isotropic resampling for reproducibility
        """
        try:
            import SimpleITK as _sitk
            from radiomics import featureextractor as _fex
            extractor = _fex.RadiomicsFeatureExtractor(
                binWidth=_BD_IBSI_BIN_WIDTH_HU,
                voxelArrayShift=0,
                normalize=False,
                force2D=False,
                resampledPixelSpacing=[1.2, 1.2, 1.2],
                interpolator="sitkBSpline",
                padDistance=5,
                correctMask=True,
                label=1,
            )
            extractor.disableAllFeatures()
            # Enable only the prespecified protocol features (locked panel).
            extractor.enableFeaturesByName(**{feature_class: list(names.keys())})
            img = _sitk.GetImageFromArray(ct_volume.astype(np.float32))
            msk = _sitk.GetImageFromArray(mask.astype(np.uint8))
            result = extractor.execute(img, msk)
            out = {}
            for rad_key, out_key in names.items():
                full = f"original_{feature_class}_{rad_key}"
                if full in result:
                    try:
                        out[out_key] = float(result[full])
                    except Exception:
                        out[out_key] = float("nan")
            return out
        except Exception:
            return {v: float("nan") for v in names.values()}


class GLCMFeatures(_BDTextureFeatureBase):
    def compute(self, ct_volume: np.ndarray, parenchymal_mask: np.ndarray) -> dict:
        names = {"Contrast": "glcm_contrast", "Correlation": "glcm_correlation",
                 "JointEnergy": "glcm_joint_energy", "JointEntropy": "glcm_joint_entropy",
                 "Homogeneity1": "glcm_homogeneity", "ClusterShade": "glcm_cluster_shade",
                 "ClusterProminence": "glcm_cluster_prominence", "MCC": "glcm_mcc",
                 "DifferenceVariance": "glcm_diff_variance", "SumEntropy": "glcm_sum_entropy",
                 "DifferenceEntropy": "glcm_difference_entropy",
                 "Idm": "glcm_idm",
                 "Imc1": "glcm_imc1"}
        rad_out = self._pyradiomics_features(ct_volume, parenchymal_mask, "glcm", names)
        if any(not (v != v) for v in rad_out.values()):
            return rad_out

        q = _bd_prepare_quantized_volume(ct_volume, parenchymal_mask)
        n = _BD_VOXEL_BINS
        glcm = np.zeros((n, n), dtype=np.float64)
        for dz, dy, dx in [(0, 0, 1), (0, 1, 0), (1, 0, 0)]:
            slc_a = (slice(None, -dz or None), slice(None, -dy or None), slice(None, -dx or None))
            slc_b = (slice(dz or None, None), slice(dy or None, None), slice(dx or None, None))
            ma = parenchymal_mask[slc_a] & parenchymal_mask[slc_b]
            ia = q[slc_a][ma]
            ib = q[slc_b][ma]
            np.add.at(glcm, (ia, ib), 1)
            np.add.at(glcm, (ib, ia), 1)
        total = glcm.sum() + 1e-12
        glcm /= total
        i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        diff = (i_idx - j_idx).astype(np.float64)
        contrast = float(np.sum(glcm * diff ** 2))
        mu_i = float(np.sum(i_idx * glcm))
        mu_j = float(np.sum(j_idx * glcm))
        std_i = float(np.sqrt(np.sum(glcm * (i_idx - mu_i) ** 2)))
        std_j = float(np.sqrt(np.sum(glcm * (j_idx - mu_j) ** 2)))
        correlation = float(np.sum(glcm * (i_idx - mu_i) * (j_idx - mu_j))) / (std_i * std_j + 1e-12)
        joint_entropy = float(-np.sum(glcm * np.log2(glcm + 1e-12)))
        joint_energy = float(np.sum(glcm ** 2))
        homogeneity = float(np.sum(glcm / (1.0 + diff ** 2)))
        # IDM (inverse difference moment) — standard formula uses |i-j|, not signed diff,
        # so the denominator is always >= 1 and never produces inf.
        idm = float(np.sum(glcm / (1.0 + np.abs(diff))))
        cluster_shade = float(np.sum(glcm * ((i_idx + j_idx - mu_i - mu_j) ** 3)))
        cluster_prominence = float(np.sum(glcm * ((i_idx + j_idx - mu_i - mu_j) ** 4)))
        # sum/diff distributions
        sum_idx = (i_idx + j_idx).astype(np.int32)
        diff_idx = np.abs(diff).astype(np.int32)
        p_xpy = np.zeros(2 * n, dtype=np.float64)
        np.add.at(p_xpy, sum_idx.ravel(), glcm.ravel())
        p_xmy = np.zeros(n, dtype=np.float64)
        np.add.at(p_xmy, diff_idx.ravel(), glcm.ravel())
        sum_entropy = float(-np.sum(p_xpy * np.log2(p_xpy + 1e-12)))
        diff_variance = float(np.sum(np.arange(n, dtype=np.float64) ** 2 * p_xmy))
        difference_entropy = float(-np.sum(p_xmy * np.log2(p_xmy + 1e-12)))
        # MCC
        p_x = glcm.sum(axis=1)
        p_y = glcm.sum(axis=0)
        p_x_s = np.where(p_x > 0, p_x, 1.0)
        p_y_s = np.where(p_y > 0, p_y, 1.0)
        q_mcc = (glcm / p_y_s[np.newaxis, :]) @ glcm.T / (p_x_s[:, np.newaxis] * p_x_s[np.newaxis, :])
        ev = np.sort(np.real(np.linalg.eigvals(q_mcc)))[::-1]
        mcc = float(np.sqrt(max(ev[1], 0.0))) if len(ev) > 1 else float("nan")
        # IMC1 (information measure of correlation 1)
        hx = float(-np.sum(p_x * np.log2(p_x + 1e-12)))
        hy = float(-np.sum(p_y * np.log2(p_y + 1e-12)))
        hxy1 = float(-np.sum(glcm * np.log2((p_x[:, None] * p_y[None, :]) + 1e-12)))
        imc1 = _bd_safe_div(joint_entropy - hxy1, max(hx, hy) + 1e-12)
        return {"glcm_contrast": round(contrast, 6), "glcm_correlation": round(correlation, 6),
                "glcm_joint_entropy": round(joint_entropy, 6), "glcm_joint_energy": round(joint_energy, 6),
                "glcm_homogeneity": round(homogeneity, 6), "glcm_cluster_shade": round(cluster_shade, 6),
                "glcm_cluster_prominence": round(cluster_prominence, 6), "glcm_mcc": round(mcc, 6),
                "glcm_diff_variance": round(diff_variance, 6), "glcm_sum_entropy": round(sum_entropy, 6),
                "glcm_difference_entropy": round(difference_entropy, 6),
                "glcm_idm": round(idm, 6),
                "glcm_imc1": round(imc1, 6)}


class GLRLMFeatures(_BDTextureFeatureBase):
    def compute(self, ct_volume: np.ndarray, parenchymal_mask: np.ndarray) -> dict:
        names = {"ShortRunEmphasis": "glrlm_short_run_emphasis", "LongRunEmphasis": "glrlm_long_run_emphasis",
                 "GrayLevelNonUniformity": "glrlm_gray_level_non_uniformity",
                 "RunLengthNonUniformity": "glrlm_run_length_non_uniformity",
                 "RunPercentage": "glrlm_run_percentage", "RunEntropy": "glrlm_run_entropy",
                 "ShortRunHighGrayLevelEmphasis": "glrlm_srhgle", "LongRunHighGrayLevelEmphasis": "glrlm_lrhgle"}
        rad_out = self._pyradiomics_features(ct_volume, parenchymal_mask, "glrlm", names)
        if any(not (v != v) for v in rad_out.values()):
            return rad_out

        q = _bd_prepare_quantized_volume(ct_volume, parenchymal_mask)
        n_g = _BD_VOXEL_BINS
        max_run = max(ct_volume.shape)
        glrlm = np.zeros((n_g, max_run), dtype=np.float64)
        # Vectorised run-length encoding along the x-axis (axis 2).
        # Flatten (z, y) → rows so all rows are processed simultaneously.
        Z, Y, X = q.shape
        q_2d = q.reshape(-1, X)                      # (Z*Y, X)
        m_2d = parenchymal_mask.reshape(-1, X)        # (Z*Y, X)
        # A new run starts when the mask turns on, or the gray level changes while masked.
        prev_m = np.zeros_like(m_2d)
        prev_m[:, 1:] = m_2d[:, :-1]
        prev_q = np.zeros_like(q_2d)
        prev_q[:, 1:] = q_2d[:, :-1]
        is_start = m_2d & (~prev_m | (q_2d != prev_q))
        # Give each run a unique integer ID via cumulative sum of starts.
        run_id = np.cumsum(is_start.ravel()).reshape(Z * Y, X)
        run_id = np.where(m_2d, run_id, 0)          # 0 = background (outside mask)
        max_id = int(run_id.max())
        if max_id > 0:
            rids = run_id.ravel()
            grays_flat = q_2d.ravel()
            mask_flat_ravel = m_2d.ravel()
            masked_rids = rids[mask_flat_ravel]
            masked_grays = grays_flat[mask_flat_ravel]
            # Run lengths: count voxels per run ID.
            rl = np.bincount(masked_rids, minlength=max_id + 1)[1:]   # (max_id,)
            # Gray level per run: all voxels in a run share the same gray level;
            # use the maximum (equivalent to any since they are equal).
            gl = np.zeros(max_id + 1, dtype=q_2d.dtype)
            np.maximum.at(gl, masked_rids, masked_grays)
            gl = gl[1:]                                                # (max_id,)
            valid = (gl >= 0) & (gl < n_g) & (rl >= 1) & (rl <= max_run)
            np.add.at(glrlm, (gl[valid], rl[valid] - 1), 1)
        total = glrlm.sum() + 1e-12
        r_idx = np.arange(1, max_run + 1, dtype=np.float64)
        sre = float(np.sum(glrlm / (r_idx ** 2 + 1e-12))) / total
        lre = float(np.sum(glrlm * (r_idx ** 2))) / total
        run_probs = glrlm / (glrlm.sum() + 1e-12)
        run_ent = float(-np.sum(run_probs * np.log2(run_probs + 1e-12)))
        glnu = float(np.sum(glrlm.sum(axis=1) ** 2)) / total
        rlnu = float(np.sum(glrlm.sum(axis=0) ** 2)) / total
        total_vox = float(np.sum(glrlm * r_idx[np.newaxis, :]))
        run_pct = _bd_safe_div(total, max(total_vox, 1.0))
        g_idx_rl = np.arange(glrlm.shape[0], dtype=np.float64)
        srhgle = float(np.sum(glrlm / (r_idx ** 2 + 1e-12) * (g_idx_rl[:, np.newaxis] ** 2))) / total
        lrhgle = float(np.sum(glrlm * (r_idx ** 2) * (g_idx_rl[:, np.newaxis] ** 2))) / total
        return {"glrlm_short_run_emphasis": round(sre, 6), "glrlm_long_run_emphasis": round(lre, 6),
                "glrlm_run_entropy": round(run_ent, 6), "glrlm_gray_level_non_uniformity": round(glnu, 6),
                "glrlm_run_length_non_uniformity": round(rlnu, 6), "glrlm_run_percentage": round(run_pct, 6),
                "glrlm_srhgle": round(srhgle, 6), "glrlm_lrhgle": round(lrhgle, 6)}


class GLSZMFeatures(_BDTextureFeatureBase):
    def compute(self, ct_volume: np.ndarray, parenchymal_mask: np.ndarray) -> dict:
        names = {"ZoneEntropy": "glszm_zone_entropy", "SmallAreaEmphasis": "glszm_small_area_emphasis",
                 "LargeAreaEmphasis": "glszm_large_area_emphasis",
                 "GrayLevelNonUniformity": "glszm_gray_level_non_uniformity",
                 "ZoneSizeNonUniformity": "glszm_zone_size_non_uniformity",
                 "ZonePercentage": "glszm_zone_percentage", "SmallAreaHighGrayLevelEmphasis": "glszm_sahgle"}
        rad_out = self._pyradiomics_features(ct_volume, parenchymal_mask, "glszm", names)
        if any(not (v != v) for v in rad_out.values()):
            return rad_out

        q = _bd_prepare_quantized_volume(ct_volume, parenchymal_mask)
        n_g = _BD_VOXEL_BINS
        q_masked = np.where(parenchymal_mask, q, -1)
        labeled, n_comp = _bd_ndi.label(parenchymal_mask)
        max_zone = max(1, n_comp)
        glszm = np.zeros((n_g, max_zone), dtype=np.float64)
        if n_comp > 0:
            # Vectorised: build a (n_comp × n_g) frequency table in one pass.
            lbls_flat = labeled.ravel()          # 0 = background, 1..n_comp = components
            q_flat = q_masked.ravel()
            in_mask = lbls_flat > 0
            lbls_m = lbls_flat[in_mask] - 1      # 0-based component index
            q_m = q_flat[in_mask].astype(np.int64)
            q_m = np.clip(q_m, 0, n_g - 1)
            combined = lbls_m * n_g + q_m
            freq = np.bincount(combined, minlength=n_comp * n_g).reshape(n_comp, n_g)
            gray_per_comp = np.argmax(freq, axis=1)           # modal gray per component
            # Zone size = voxel count - 1 (matches original behaviour)
            voxel_counts = np.bincount(lbls_flat)[1:]         # component sizes (1-based)
            zone_sizes = voxel_counts.astype(np.int64) - 1
            valid = (gray_per_comp >= 0) & (gray_per_comp < n_g) & \
                    (zone_sizes >= 0) & (zone_sizes < max_zone)
            np.add.at(glszm, (gray_per_comp[valid], zone_sizes[valid]), 1)
        total = glszm.sum() + 1e-12
        z_idx = np.arange(1, max_zone + 1, dtype=np.float64)
        sae = float(np.sum(glszm / (z_idx ** 2 + 1e-12))) / total
        lae = float(np.sum(glszm * (z_idx ** 2))) / total
        zone_probs = glszm / (glszm.sum() + 1e-12)
        zone_ent = float(-np.sum(zone_probs * np.log2(zone_probs + 1e-12)))
        glnu_sz = float(np.sum(glszm.sum(axis=1) ** 2)) / total
        zsnu = float(np.sum(glszm.sum(axis=0) ** 2)) / total
        total_vox_sz = float(np.sum(glszm * z_idx[np.newaxis, :]))
        zone_pct = _bd_safe_div(total, max(total_vox_sz, 1.0))
        g_idx_sz = np.arange(n_g, dtype=np.float64)
        sahgle = float(np.sum(glszm / (z_idx ** 2 + 1e-12) * (g_idx_sz[:, np.newaxis] ** 2))) / total
        return {"glszm_zone_entropy": round(zone_ent, 6), "glszm_small_area_emphasis": round(sae, 6),
                "glszm_large_area_emphasis": round(lae, 6), "glszm_gray_level_non_uniformity": round(glnu_sz, 6),
                "glszm_zone_size_non_uniformity": round(zsnu, 6), "glszm_zone_percentage": round(zone_pct, 6),
                "glszm_sahgle": round(sahgle, 6)}


class GLDMFeatures(_BDTextureFeatureBase):
    def compute(self, ct_volume: np.ndarray, parenchymal_mask: np.ndarray) -> dict:
        names = {"DependenceNonUniformity": "gldm_dependence_non_uniformity",
                 "DependenceEntropy": "gldm_dependence_entropy",
                 "SmallDependenceEmphasis": "gldm_small_dependence_emphasis",
                 "LargeDependenceEmphasis": "gldm_large_dependence_emphasis",
                 "GrayLevelNonUniformity": "gldm_gl_non_uniformity"}
        rad_out = self._pyradiomics_features(ct_volume, parenchymal_mask, "gldm", names)
        if any(not (v != v) for v in rad_out.values()):
            return rad_out

        q = _bd_prepare_quantized_volume(ct_volume, parenchymal_mask)
        n_g = _BD_VOXEL_BINS
        # Vectorised: count same-gray neighbours for every masked voxel simultaneously
        # using shifted copies of q and mask along each of the 6 face directions.
        dep_count = np.zeros(q.shape, dtype=np.int32)
        for axis, step in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
            shifted_q = np.roll(q, -step, axis=axis)
            shifted_m = np.roll(parenchymal_mask, -step, axis=axis)
            # Zero out the wrapped boundary so np.roll does not introduce phantom neighbours.
            sl = [slice(None)] * 3
            sl[axis] = slice(-1, None) if step > 0 else slice(None, 1)
            shifted_m[tuple(sl)] = False
            dep_count += (parenchymal_mask & shifted_m & (q == shifted_q)).astype(np.int32)
        if not parenchymal_mask.any():
            return {v: float("nan") for v in names.values()}
        grays = q[parenchymal_mask]
        deps = dep_count[parenchymal_mask]
        max_dep = int(deps.max()) + 1
        gldm = np.zeros((n_g, max_dep), dtype=np.float64)
        valid = (grays >= 0) & (grays < n_g)
        np.add.at(gldm, (grays[valid], deps[valid]), 1)
        total = gldm.sum() + 1e-12
        dep_idx = np.arange(max_dep, dtype=np.float64)
        dnu = float(np.sum(gldm.sum(axis=0) ** 2)) / total
        glnu_gldm = float(np.sum(gldm.sum(axis=1) ** 2)) / total
        dep_probs = gldm / (gldm.sum() + 1e-12)
        dep_ent = float(-np.sum(dep_probs * np.log2(dep_probs + 1e-12)))
        sde = float(np.sum(dep_probs[:, 1:] / ((dep_idx[1:] ** 2) + 1e-12))) if max_dep > 1 else float("nan")
        lde = float(np.sum(dep_probs[:, 1:] * (dep_idx[1:] ** 2))) if max_dep > 1 else float("nan")
        return {"gldm_dependence_non_uniformity": round(dnu, 6), "gldm_dependence_entropy": round(dep_ent, 6),
                "gldm_small_dependence_emphasis": round(sde, 6), "gldm_large_dependence_emphasis": round(lde, 6),
                "gldm_gray_level_non_uniformity": round(glnu_gldm, 6)}


class NGTDMFeatures(_BDTextureFeatureBase):
    def compute(self, ct_volume: np.ndarray, parenchymal_mask: np.ndarray) -> dict:
        names = {"Coarseness": "ngtdm_coarseness", "Busyness": "ngtdm_busyness",
                 "Complexity": "ngtdm_complexity", "Contrast": "ngtdm_contrast", "Strength": "ngtdm_strength"}
        rad_out = self._pyradiomics_features(ct_volume, parenchymal_mask, "ngtdm", names)
        if any(not (v != v) for v in rad_out.values()):
            return rad_out

        q = _bd_prepare_quantized_volume(ct_volume, parenchymal_mask)
        n_g = _BD_VOXEL_BINS
        # Vectorised: accumulate neighbour sums and counts using 6 shifted copies.
        neighbour_sum = np.zeros(q.shape, dtype=np.float64)
        neighbour_cnt = np.zeros(q.shape, dtype=np.float64)
        for axis, step in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
            shifted_q = np.roll(q, -step, axis=axis).astype(np.float64)
            shifted_m = np.roll(parenchymal_mask, -step, axis=axis)
            # Clear the wrapped boundary.
            sl = [slice(None)] * 3
            sl[axis] = slice(-1, None) if step > 0 else slice(None, 1)
            shifted_m[tuple(sl)] = False
            valid_nb = parenchymal_mask & shifted_m
            neighbour_sum += np.where(valid_nb, shifted_q, 0.0)
            neighbour_cnt += valid_nb.astype(np.float64)
        has_nb = parenchymal_mask & (neighbour_cnt > 0)
        n_i = np.zeros(n_g, dtype=np.float64)
        s_i = np.zeros(n_g, dtype=np.float64)
        if has_nb.any():
            grays = q[has_nb]
            avg_nb = neighbour_sum[has_nb] / neighbour_cnt[has_nb]
            diffs = np.abs(grays.astype(np.float64) - avg_nb)
            valid = (grays >= 0) & (grays < n_g)
            np.add.at(n_i, grays[valid], 1.0)
            np.add.at(s_i, grays[valid], diffs[valid])

        total_n = n_i.sum() + 1e-12
        p_i = n_i / total_n
        coarseness = float(1.0 / (np.sum(p_i * s_i) + 1e-12))
        g_idx = np.arange(n_g, dtype=np.float64)
        busyness_num = float(np.sum(p_i * s_i))
        busyness_den = float(np.sum(np.abs(np.outer(p_i * g_idx, np.ones(n_g)) - np.outer(np.ones(n_g), p_i * g_idx))))
        busyness = _bd_safe_div(busyness_num, busyness_den)
        # Contrast
        pij = np.outer(p_i, p_i)
        ii_m, jj_m = np.meshgrid(g_idx, g_idx, indexing="ij")
        ng = max(float(np.count_nonzero(p_i)), 1.0)
        contrast_num = float(np.sum(pij * (ii_m - jj_m) ** 2))
        contrast = _bd_safe_div(contrast_num, ng * (ng - 1.0 + 1e-8)) * _bd_safe_div(float(np.sum(s_i)), float(total_n))
        # Complexity
        abs_diff_m = np.abs(ii_m - jj_m)
        si_mat = s_i[:, None] * np.ones((1, n_g))
        sj_mat = np.ones((n_g, 1)) * s_i[None, :]
        p_sum_m = p_i[:, None] + p_i[None, :] + 1e-12
        complexity = float(np.sum(pij * abs_diff_m * (p_i[:, None] * si_mat + p_i[None, :] * sj_mat) / p_sum_m))
        # Strength
        s_total = float(np.sum(s_i)) + 1e-12
        strength = _bd_safe_div(float(np.sum(pij * (ii_m - jj_m) ** 2 * (p_i[:, None] + p_i[None, :]))), s_total)
        return {"ngtdm_coarseness": round(coarseness, 6), "ngtdm_busyness": round(busyness, 6),
                "ngtdm_complexity": round(complexity, 6), "ngtdm_contrast": round(contrast, 6),
                "ngtdm_strength": round(strength, 6)}


# --- Parenchymal Complexity Engine ---
class ParenchymalComplexityEngine:
    """Combines shape, first-order, and texture features into a single complexity report."""

    def __init__(self, ct_volume: np.ndarray, parenchymal_mask: np.ndarray, voxel_spacing_mm=(1.0, 1.0, 1.0)):
        self.ct_volume = ct_volume
        self.parenchymal_mask = parenchymal_mask
        self.voxel_spacing_mm = voxel_spacing_mm

    def compute_all(self, skip_shape: bool = False) -> dict:
        """Compute all parenchymal complexity features.

        Parameters
        ----------
        skip_shape:
            When *True* all shape features (``sphericity``, ``elongation``,
            ``flatness``, ``major_axis_length_mm``, ``max_3d_diameter_mm``,
            etc.) are set to ``NaN``.  Use this for bilateral merged masks
            where a disconnected two-sided object produces biologically
            meaningless shape metrics (e.g. MajorAxisLength > 1000 mm
            spanning both breasts).  Shape should only be extracted per
            breast, not from a union mask.
        """
        if skip_shape:
            # Compute volume (parenchymal_volume_cc) but nullify spatial shape.
            vox_cc = _bd_voxel_volume_cc(self.voxel_spacing_mm)
            vol_cc = round(float(np.count_nonzero(self.parenchymal_mask)) * vox_cc, 4)
            shape = {
                "parenchymal_volume_cc": vol_cc,
                "volume_cc": vol_cc,
                "surface_area_mm2": float("nan"),
                "surface_to_volume_ratio": float("nan"),
                "sphericity": float("nan"),
                "elongation": float("nan"),
                "flatness": float("nan"),
                "major_axis_length_mm": float("nan"),
                "minor_axis_length_mm": float("nan"),
                "max_3d_diameter_mm": float("nan"),
            }
        else:
            shape = ShapeFeatures().compute(self.parenchymal_mask, self.voxel_spacing_mm)
        first_order = FirstOrderFeatures().compute(self.ct_volume, self.parenchymal_mask)
        glcm = GLCMFeatures().compute(self.ct_volume, self.parenchymal_mask)
        glrlm = GLRLMFeatures().compute(self.ct_volume, self.parenchymal_mask)
        glszm = GLSZMFeatures().compute(self.ct_volume, self.parenchymal_mask)
        gldm = GLDMFeatures().compute(self.ct_volume, self.parenchymal_mask)
        ngtdm = NGTDMFeatures().compute(self.ct_volume, self.parenchymal_mask)
        all_feats = _bd_sanitize_features({**shape, **first_order, **glcm, **glrlm, **glszm, **gldm, **ngtdm})
        shortlist = {k: all_feats.get(k, float("nan")) for k in _BD_MANUSCRIPT_SHORTLIST_FEATURES}
        return {**all_feats, "manuscript_shortlist": shortlist}


# --- ML Analysis helpers ---
def _bd_train_classifiers(df, feature_columns, target_col):
    """Train 4 baseline classifiers; return AUC results and ROC coordinates."""
    X = df[feature_columns]
    y = df[target_col]
    X_train, X_test, y_train, y_test = _bd_tts(
        X, y, test_size=0.2, random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )
    classifiers = {
        "Logistic Regression": _bd_LR(max_iter=1_000),
        "Decision Tree": _bd_DTC(random_state=42),
        "Random Forest": _bd_RFC(random_state=42),
        "SVM": _bd_SVC(probability=True),
    }
    auc_results = {}
    curves = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        scores = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = _bd_roc_curve(y_test, scores)
        auc_results[name] = float(_bd_auc(fpr, tpr))
        curves[name] = (fpr, tpr)
    return auc_results, curves


# ═══ Background worker thread for long-running breast density / complexity ops ═══
class _BreastWorker(QtCore.QThread):
    """Generic QThread worker that calls a callable in a background thread.

    Signals:
        finished(dict) — emitted on success with a result payload
        error(str)     — emitted on exception with the error message string
    """
    finished = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._func(*self._args, **self._kwargs)
            self.finished.emit(result if isinstance(result, dict) else {})
        except Exception as exc:
            self.error.emit(str(exc))


# ═══ V21BreastDensityTab — PyQt6 widget for Clinical Tools dialog ═══
class V21BreastDensityTab(QtWidgets.QWidget):
    """Breast density and parenchymal complexity analysis tab for the Clinical Tools dialog.

    Workflow:
    1. Draw ROI in the main viewer (Paint / Lasso — any orthogonal view; all views sync in real time).
    2. Click '🔵 Label as LEFT Breast' or '🔴 Label as RIGHT Breast' to assign the drawn ROI.
       - Each label copies the current viewer mask into a per-breast store.
       - You can draw a new ROI and label the other breast separately.
    3. Click '🔬 Segment Fibroglandular Tissue' to run BreastSegmentor on the labelled breast masks
       and strip skin / pectoralis / clips — independently for each side.
    4. Alternatively, '▶ Compute Density (Auto)' performs fully automatic bilateral segmentation.
    5. '▶ Compute Complexity' runs radiomics on whichever parenchymal mask is active.
    6. '💾 Export CSV' saves all per-breast and bilateral metrics.
    """

    def __init__(self, app):
        super().__init__()
        self._app = app
        self._density_results: dict = {}
        self._complexity_results: dict = {}
        self._whole_masks: dict = {}          # keyed "left_mask" / "right_mask"
        self._fg_masks: dict = {}             # fibroglandular masks per side after segmentation
        self._parenchymal_mask = None         # active mask fed to complexity engine
        self._roi_left: Optional[np.ndarray] = None           # manually labelled left breast ROI
        self._roi_right: Optional[np.ndarray] = None          # manually labelled right breast ROI
        self._roi_tumor_exclusion: Optional[np.ndarray] = None  # user-drawn tumor exclusion ROI
        self._excl_detail_masks: dict = {}    # per-side skin/pectoralis/clips/vessels (for figure export)
        self._worker: Optional[_BreastWorker] = None          # background computation thread
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # --- Info banner ---
        if not BREAST_DENSITY_DEPS_AVAILABLE:
            warn = QtWidgets.QLabel(
                "⚠  scipy / scikit-image not installed — breast density computation unavailable.\n"
                "Install with:  pip install scipy scikit-image"
            )
            warn.setStyleSheet("color: #c0392b; background: #fdf0f0; padding: 6px; border-radius:4px;")
            warn.setWordWrap(True)
            root.addWidget(warn)

        # ── ROI guidance label ──────────────────────────────────────────
        roi_hint = QtWidgets.QLabel(
            "Tip — Draw breast ROI in the main viewer using Paint or Lasso on any view (sagittal recommended). "
            "Changes update all three orthogonal views in real time.\n"
            "Label each breast separately using the buttons below, then run fibroglandular segmentation "
            "or auto-density."
        )
        roi_hint.setStyleSheet("color: #1a5276; background: #eaf4fc; padding: 5px; border-radius: 4px;")
        roi_hint.setWordWrap(True)
        root.addWidget(roi_hint)

        # ── Per-breast ROI labelling row ────────────────────────────────
        label_grp = QtWidgets.QGroupBox("Step 1 — Label drawn ROI per breast")
        label_lay = QtWidgets.QVBoxLayout(label_grp)
        label_row = QtWidgets.QHBoxLayout()

        self._btn_label_right = QtWidgets.QPushButton("🔴  Label as RIGHT Breast")
        self._btn_label_right.setToolTip(
            "Copies the currently drawn ROI in the main viewer as the RIGHT breast mask.\n"
            "Draw ROI over the right breast first, then click this button."
        )
        self._btn_label_right.setStyleSheet("QPushButton { color: #c0392b; font-weight: bold; }")
        self._btn_label_right.clicked.connect(lambda: self._on_label_roi("right"))
        label_row.addWidget(self._btn_label_right)

        self._btn_label_left = QtWidgets.QPushButton("🔵  Label as LEFT Breast")
        self._btn_label_left.setToolTip(
            "Copies the currently drawn ROI in the main viewer as the LEFT breast mask.\n"
            "Draw ROI over the left breast, then click this button."
        )
        self._btn_label_left.setStyleSheet("QPushButton { color: #1a5276; font-weight: bold; }")
        self._btn_label_left.clicked.connect(lambda: self._on_label_roi("left"))
        label_row.addWidget(self._btn_label_left)

        self._btn_clear_labels = QtWidgets.QPushButton("✖  Clear Labels")
        self._btn_clear_labels.setToolTip("Remove stored left / right ROI labels.")
        self._btn_clear_labels.clicked.connect(self._on_clear_labels)
        label_row.addWidget(self._btn_clear_labels)
        label_row.addStretch()
        label_lay.addLayout(label_row)

        self._lbl_roi_status = QtWidgets.QLabel("No breast ROIs labelled yet.")
        self._lbl_roi_status.setStyleSheet("color: #5d6d7e; font-style: italic;")
        label_lay.addWidget(self._lbl_roi_status)

        # ── Tumor exclusion ROI row (within Step 1) ─────────────────────
        tumor_row = QtWidgets.QHBoxLayout()

        self._btn_label_tumor_excl = QtWidgets.QPushButton("🚫  Mark Tumor Exclusion ROI")
        self._btn_label_tumor_excl.setToolTip(
            "Draw the tumor region in the main viewer using Paint or Lasso, then click this button.\n"
            "The drawn region will be subtracted from the fibroglandular mask before density computation.\n\n"
            "This is applied on top of the automatic largest-component heuristic already built in to\n"
            "'Segment Fibroglandular Tissue' and 'Compute Density (Auto)'."
        )
        self._btn_label_tumor_excl.setStyleSheet("QPushButton { color: #6c3483; font-weight: bold; }")
        self._btn_label_tumor_excl.clicked.connect(self._on_label_tumor_exclusion)
        tumor_row.addWidget(self._btn_label_tumor_excl)

        self._btn_clear_tumor_excl = QtWidgets.QPushButton("✖  Clear Tumor Exclusion")
        self._btn_clear_tumor_excl.setToolTip("Remove the stored tumor exclusion ROI.")
        self._btn_clear_tumor_excl.clicked.connect(self._on_clear_tumor_exclusion)
        tumor_row.addWidget(self._btn_clear_tumor_excl)
        tumor_row.addStretch()
        label_lay.addLayout(tumor_row)

        self._lbl_tumor_excl_status = QtWidgets.QLabel("No tumor exclusion ROI set.")
        self._lbl_tumor_excl_status.setStyleSheet("color: #6c3483; font-style: italic;")
        label_lay.addWidget(self._lbl_tumor_excl_status)

        root.addWidget(label_grp)

        # ── Segmentation / density row ──────────────────────────────────
        seg_grp = QtWidgets.QGroupBox("Step 2 — Segment fibroglandular tissue & compute density")
        seg_lay = QtWidgets.QVBoxLayout(seg_grp)
        btn_row = QtWidgets.QHBoxLayout()

        self._btn_segment_fg = QtWidgets.QPushButton("🔬  Segment Fibroglandular Tissue")
        self._btn_segment_fg.setEnabled(BREAST_DENSITY_DEPS_AVAILABLE)
        self._btn_segment_fg.setToolTip(
            "Run BreastSegmentor.segment_fibroglandular + exclude_non_parenchymal on the\n"
            "labelled left and right breast ROIs independently.\n\n"
            "✅ Automatic exclusions (NO extra buttons needed):\n"
            "   • Skin shell   — 2-voxel erosion of breast boundary\n"
            "   • Pectoralis   — posterior-region HU >40 (muscle)\n"
            "   • Metal clips  — bright small objects HU >400\n\n"
            "After running, colour-coded overlays appear in all three views:\n"
            "   Cyan = right whole breast   Blue = left whole breast\n"
            "   Orange = right fibroglandular   Gold = left fibroglandular\n\n"
            "Requires at least one labelled breast ROI (Step 1)."
        )
        self._btn_segment_fg.clicked.connect(self._on_segment_fg)
        btn_row.addWidget(self._btn_segment_fg)

        self._btn_density = QtWidgets.QPushButton("▶  Compute Density (Auto)")
        self._btn_density.setEnabled(BREAST_DENSITY_DEPS_AVAILABLE)
        self._btn_density.setToolTip(
            "Run automatic whole-breast segmentation and volumetric density computation on loaded CT.\n"
            "Computes left and right breast metrics independently."
        )
        self._btn_density.clicked.connect(self._on_compute_density)
        btn_row.addWidget(self._btn_density)

        self._btn_complexity = QtWidgets.QPushButton("▶  Compute Complexity")
        self._btn_complexity.setEnabled(False)
        self._btn_complexity.setToolTip(
            "Run parenchymal complexity radiomics on the fibroglandular mask.\n"
            "Run 'Segment Fibroglandular Tissue' or 'Compute Density (Auto)' first."
        )
        self._btn_complexity.clicked.connect(self._on_compute_complexity)
        btn_row.addWidget(self._btn_complexity)

        self._btn_export = QtWidgets.QPushButton("💾  Export CSV")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._on_export_csv)
        btn_row.addWidget(self._btn_export)

        self._btn_export_fig = QtWidgets.QPushButton("📸  Export Segmentation Figure")
        self._btn_export_fig.setEnabled(False)
        self._btn_export_fig.setToolTip(
            "Save a publication-quality PNG figure (300 DPI) with all segmentation layers "
            "rendered in distinct colours over representative CT slices.\n\n"
            "Colour scheme:\n"
            "  Whole breast (R)      — cyan\n"
            "  Whole breast (L)      — steel-blue\n"
            "  Fibroglandular (R)    — orange\n"
            "  Fibroglandular (L)    — gold\n"
            "  Skin shell            — olive-green\n"
            "  Pectoralis muscle     — magenta\n"
            "  Metal clips / calcif. — yellow\n"
            "  Blood vessels         — lime-green\n"
            "  Tumor exclusion zone  — red\n\n"
            "Run 'Segment Fibroglandular Tissue' or 'Compute Density (Auto)' first."
        )
        self._btn_export_fig.clicked.connect(self._on_export_figure)
        btn_row.addWidget(self._btn_export_fig)
        btn_row.addStretch()
        seg_lay.addLayout(btn_row)
        root.addWidget(seg_grp)

        # ── Per-breast complexity row ───────────────────────────────────
        side_grp = QtWidgets.QGroupBox("Per-breast complexity (after fibroglandular segmentation)")
        side_lay = QtWidgets.QHBoxLayout(side_grp)

        self._btn_complexity_right = QtWidgets.QPushButton("🔴  Complexity: RIGHT")
        self._btn_complexity_right.setEnabled(False)
        self._btn_complexity_right.setToolTip("Run complexity radiomics on the RIGHT fibroglandular mask only.")
        self._btn_complexity_right.clicked.connect(lambda: self._on_compute_complexity_side("right"))
        side_lay.addWidget(self._btn_complexity_right)

        self._btn_complexity_left = QtWidgets.QPushButton("🔵  Complexity: LEFT")
        self._btn_complexity_left.setEnabled(False)
        self._btn_complexity_left.setToolTip("Run complexity radiomics on the LEFT fibroglandular mask only.")
        self._btn_complexity_left.clicked.connect(lambda: self._on_compute_complexity_side("left"))
        side_lay.addWidget(self._btn_complexity_left)
        side_lay.addStretch()
        root.addWidget(side_grp)

        # --- Results tabs ---
        self._result_tabs = QtWidgets.QTabWidget()
        self._result_tabs.setDocumentMode(True)

        # Density results
        density_w = QtWidgets.QWidget()
        density_lay = QtWidgets.QVBoxLayout(density_w)
        self._density_table = self._make_kv_table()
        density_lay.addWidget(self._density_table)
        self._result_tabs.addTab(density_w, "Density")

        # Complexity — manuscript shortlist
        shortlist_w = QtWidgets.QWidget()
        shortlist_lay = QtWidgets.QVBoxLayout(shortlist_w)
        self._shortlist_table = self._make_kv_table()
        shortlist_lay.addWidget(self._shortlist_table)
        self._result_tabs.addTab(shortlist_w, "Complexity (Shortlist)")

        # Complexity — full feature set
        full_w = QtWidgets.QWidget()
        full_lay = QtWidgets.QVBoxLayout(full_w)
        self._full_table = self._make_kv_table()
        full_lay.addWidget(self._full_table)
        self._result_tabs.addTab(full_w, "Complexity (Full)")

        root.addWidget(self._result_tabs, 2)

        # --- ML section ---
        ml_grp = QtWidgets.QGroupBox("ML Analysis (requires sklearn + pandas + exported CSV)")
        ml_lay = QtWidgets.QVBoxLayout(ml_grp)

        if not BREAST_ML_DEPS_AVAILABLE:
            ml_lay.addWidget(QtWidgets.QLabel(
                "sklearn / pandas / matplotlib not installed — ML analysis unavailable."
            ))
        else:
            csv_row = QtWidgets.QHBoxLayout()
            self._csv_path_edit = QtWidgets.QLineEdit()
            self._csv_path_edit.setPlaceholderText("Path to breast_density_data.csv …")
            csv_row.addWidget(self._csv_path_edit)
            browse_btn = QtWidgets.QPushButton("Browse…")
            browse_btn.clicked.connect(self._on_browse_csv)
            csv_row.addWidget(browse_btn)
            ml_lay.addLayout(csv_row)

            ml_btn_row = QtWidgets.QHBoxLayout()
            self._btn_ml_density = QtWidgets.QPushButton("Run Density ML")
            self._btn_ml_density.clicked.connect(lambda: self._on_run_ml("density"))
            ml_btn_row.addWidget(self._btn_ml_density)
            self._btn_ml_complexity = QtWidgets.QPushButton("Run Complexity ML")
            self._btn_ml_complexity.clicked.connect(lambda: self._on_run_ml("complexity"))
            ml_btn_row.addWidget(self._btn_ml_complexity)
            ml_btn_row.addStretch()
            ml_lay.addLayout(ml_btn_row)

            self._ml_result_label = QtWidgets.QLabel("")
            self._ml_result_label.setWordWrap(True)
            ml_lay.addWidget(self._ml_result_label)

        root.addWidget(ml_grp)

        # --- Log pane ---
        self._log = QtWidgets.QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(100)
        self._log.setPlaceholderText("Log output …")
        root.addWidget(self._log)

        # --- Busy status label (shown while background thread runs) ---
        self._lbl_busy = QtWidgets.QLabel("")
        self._lbl_busy.setStyleSheet(
            "color: #7d3c98; background: #f9f0ff; padding: 4px; border-radius: 4px; font-style: italic;"
        )
        self._lbl_busy.setVisible(False)
        root.addWidget(self._lbl_busy)

    # ------------------------------------------------------------------
    def _set_busy(self, busy: bool, message: str = "") -> None:
        """Disable/enable all action buttons and show a status message while a worker runs."""
        action_btns = [
            self._btn_segment_fg, self._btn_density, self._btn_complexity,
            self._btn_complexity_right, self._btn_complexity_left,
        ]
        if busy:
            for btn in action_btns:
                btn.setEnabled(False)
            self._lbl_busy.setText(f"⏳  {message}")
            self._lbl_busy.setVisible(True)
        else:
            self._lbl_busy.setVisible(False)
            # Re-enable buttons that should be active based on current state
            self._btn_segment_fg.setEnabled(BREAST_DENSITY_DEPS_AVAILABLE)
            self._btn_density.setEnabled(BREAST_DENSITY_DEPS_AVAILABLE)
            self._btn_complexity.setEnabled(BREAST_DENSITY_DEPS_AVAILABLE and self._parenchymal_mask is not None)
            self._btn_complexity_right.setEnabled(BREAST_DENSITY_DEPS_AVAILABLE and "right" in self._fg_masks)
            self._btn_complexity_left.setEnabled(BREAST_DENSITY_DEPS_AVAILABLE and "left" in self._fg_masks)

    # ------------------------------------------------------------------
    def _update_roi_status(self):
        parts = []
        if self._roi_right is not None:
            parts.append(f"🔴 RIGHT  ({int(np.sum(self._roi_right))} vox)")
        if self._roi_left is not None:
            parts.append(f"🔵 LEFT   ({int(np.sum(self._roi_left))} vox)")
        self._lbl_roi_status.setText("  |  ".join(parts) if parts else "No breast ROIs labelled yet.")
        if self._roi_tumor_exclusion is not None:
            n = int(np.sum(self._roi_tumor_exclusion))
            self._lbl_tumor_excl_status.setText(f"🚫 Tumor exclusion ROI set  ({n} vox) — will be subtracted from FG mask")
        else:
            self._lbl_tumor_excl_status.setText("No tumor exclusion ROI set.")

    def _push_seg_overlays(self):
        """Push colour-coded segmentation masks to all three slice views.

        Colour scheme:
          🔴 RIGHT whole breast  — cyan        (R=0,   G=200, B=200, A=55)
          🔵 LEFT  whole breast  — steel-blue  (R=70,  G=130, B=180, A=55)
          🔴 RIGHT fibroglandular— orange      (R=255, G=140, B=0,   A=140)
          🔵 LEFT  fibroglandular— gold        (R=220, G=200, B=0,   A=140)

        Skin, pectoralis, and clip exclusions are performed automatically inside
        BreastSegmentor.exclude_non_parenchymal() — no extra buttons are required.
        """
        overlays: dict = {}
        _wm = self._whole_masks
        right_whole = _wm.get("right_mask") if _wm.get("right_mask") is not None else self._roi_right
        left_whole  = _wm.get("left_mask")  if _wm.get("left_mask")  is not None else self._roi_left
        if right_whole is not None:
            overlays[(0, 200, 200, 55)] = right_whole       # cyan — right whole breast
        if left_whole is not None:
            overlays[(70, 130, 180, 55)] = left_whole       # steel-blue — left whole breast
        if "right" in self._fg_masks:
            overlays[(255, 140, 0, 140)] = self._fg_masks["right"]   # orange — right FG
        if "left" in self._fg_masks:
            overlays[(220, 200, 0, 140)] = self._fg_masks["left"]    # gold — left FG
        if self._roi_tumor_exclusion is not None:
            overlays[(220, 30, 30, 160)] = self._roi_tumor_exclusion  # red — tumor exclusion zone
        for attr in ("view_ax", "view_co", "view_sa"):
            v = getattr(self._app, attr, None)
            if v is not None and hasattr(v, "set_seg_overlays"):
                v.set_seg_overlays(overlays)

    # ------------------------------------------------------------------
    @staticmethod
    def _make_kv_table() -> QtWidgets.QTableWidget:
        tbl = QtWidgets.QTableWidget(0, 2)
        tbl.setHorizontalHeaderLabels(["Metric", "Value"])
        tbl.horizontalHeader().setStretchLastSection(True)
        tbl.setEditTriggers(_v21_qt_no_edit_triggers())
        tbl.setSelectionBehavior(_v21_qt_select_rows())
        tbl.verticalHeader().setVisible(False)
        return tbl

    def _fill_table(self, tbl: QtWidgets.QTableWidget, data: dict):
        tbl.setRowCount(0)
        for key, val in data.items():
            if key == "manuscript_shortlist":
                continue
            row = tbl.rowCount()
            tbl.insertRow(row)
            tbl.setItem(row, 0, QtWidgets.QTableWidgetItem(str(key)))
            try:
                tbl.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{float(val):.6f}"))
            except Exception:
                tbl.setItem(row, 1, QtWidgets.QTableWidgetItem(str(val)))

    def _log_msg(self, msg: str):
        self._log.appendPlainText(msg)

    # ------------------------------------------------------------------
    # ROI labelling
    # ------------------------------------------------------------------
    def _on_label_roi(self, side: str):
        """Snapshot the current viewer ROI mask and tag it as the given breast side."""
        roi_mask = getattr(self._app, "mask_primary", None)
        if roi_mask is None:
            roi_mask = getattr(self._app, "mask", None)
        if roi_mask is None or not np.any(roi_mask):
            QtWidgets.QMessageBox.warning(
                self, "Label ROI",
                f"No ROI drawn in the main viewer yet.\n\n"
                f"Switch to Paint or Lasso mode, draw the {'RIGHT' if side == 'right' else 'LEFT'} "
                f"breast region, then click this button."
            )
            return
        snap = roi_mask.astype(bool).copy()
        if side == "right":
            self._roi_right = snap
            self._log_msg(f"[ROI] RIGHT breast labelled — {int(np.sum(snap))} voxels.")
        else:
            self._roi_left = snap
            self._log_msg(f"[ROI] LEFT breast labelled — {int(np.sum(snap))} voxels.")
        self._update_roi_status()
        # Enable fibroglandular segmentation now that at least one side is labelled
        self._btn_segment_fg.setEnabled(BREAST_DENSITY_DEPS_AVAILABLE)

    def _on_clear_labels(self):
        self._roi_left = None
        self._roi_right = None
        self._fg_masks = {}
        self._parenchymal_mask = None
        self._whole_masks = {}
        self._update_roi_status()
        self._btn_complexity.setEnabled(False)
        self._btn_complexity_left.setEnabled(False)
        self._btn_complexity_right.setEnabled(False)
        # Clear segmentation overlays from all views
        for attr in ("view_ax", "view_co", "view_sa"):
            v = getattr(self._app, attr, None)
            if v is not None and hasattr(v, "set_seg_overlays"):
                v.set_seg_overlays({})
        self._log_msg("[ROI] Left / right labels cleared.")

    def _on_label_tumor_exclusion(self):
        """Snapshot the current viewer ROI mask as the tumor exclusion zone."""
        roi_mask = getattr(self._app, "mask_primary", None)
        if roi_mask is None:
            roi_mask = getattr(self._app, "mask", None)
        if roi_mask is None or not np.any(roi_mask):
            QtWidgets.QMessageBox.warning(
                self, "Tumor Exclusion ROI",
                "No ROI drawn in the main viewer yet.\n\n"
                "Switch to Paint or Lasso mode, draw the tumor region, then click this button."
            )
            return
        snap = roi_mask.astype(bool).copy()

        # Block saving if the drawn region covers more than 30 % of any breast ROI —
        # this prevents the common mistake of re-using the full breast ROI as the
        # tumor exclusion zone.  The user must draw a tight region around the lesion.
        blocked_sides = []
        for side_label, breast_roi in (("Right", self._roi_right), ("Left", self._roi_left)):
            if breast_roi is not None and breast_roi.any():
                overlap = int(np.sum(snap & breast_roi.astype(bool)))
                breast_total = int(np.sum(breast_roi))
                if breast_total > 0 and overlap / breast_total > 0.30:
                    blocked_sides.append(
                        f"{side_label}: ROI covers {100*overlap/breast_total:.0f} % "
                        f"({overlap}/{breast_total} vox)"
                    )
        if blocked_sides:
            QtWidgets.QMessageBox.critical(
                self, "Tumor Exclusion ROI — Region Too Large",
                "The drawn ROI covers more than 30 % of the following breast ROI(s):\n\n"
                + "\n".join(blocked_sides)
                + "\n\nThe tumor exclusion zone must be a tight region around the tumour "
                "only — not the whole breast.\n\n"
                "Steps to fix:\n"
                "  1. Clear the current viewer mask (paint/lasso clear).\n"
                "  2. Draw a small region that covers only the tumour / clip area.\n"
                "  3. Click 'Mark Tumor Exclusion ROI' again."
            )
            return

        self._roi_tumor_exclusion = snap
        self._log_msg(f"[ROI] Tumor exclusion ROI set — {int(np.sum(snap))} voxels will be subtracted from FG mask.")
        self._update_roi_status()
        self._push_seg_overlays()

    def _on_clear_tumor_exclusion(self):
        """Remove the stored tumor exclusion ROI."""
        self._roi_tumor_exclusion = None
        self._update_roi_status()
        self._push_seg_overlays()
        self._log_msg("[ROI] Tumor exclusion ROI cleared.")

    # ------------------------------------------------------------------
    # Fibroglandular segmentation on per-breast ROIs
    # ------------------------------------------------------------------
    def _on_segment_fg(self):
        """Run fibroglandular segmentation in a background thread (non-blocking)."""
        ct_hu = getattr(self._app, "ct_hu", None)
        if ct_hu is None:
            QtWidgets.QMessageBox.warning(self, "Segment FG", "No CT volume loaded.")
            return
        if self._roi_left is None and self._roi_right is None:
            QtWidgets.QMessageBox.warning(
                self, "Segment FG",
                "No breast ROIs labelled yet.\n\n"
                "Draw the right breast ROI in the viewer, click '🔴 Label as RIGHT Breast', "
                "then draw the left breast and click '🔵 Label as LEFT Breast'."
            )
            return
        spacing = getattr(self._app, "spacing_zyx", (1.0, 1.0, 1.0))
        voxel_spacing_mm = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
        vox_cc = float(np.prod(voxel_spacing_mm) / 1000.0)
        # Snapshot ROI references before launching thread (they must not change mid-run)
        roi_right = self._roi_right.copy() if self._roi_right is not None else None
        roi_left = self._roi_left.copy() if self._roi_left is not None else None
        tumor_excl = self._roi_tumor_exclusion.copy() if self._roi_tumor_exclusion is not None else None
        ct_snap = ct_hu.copy()

        def _compute():
            segmentor = BreastSegmentor()
            results: dict = {}
            fg_masks: dict = {}
            excl_detail: dict = {}
            for side, roi in (("right", roi_right), ("left", roi_left)):
                if roi is None:
                    continue
                whole_vol_cc = float(np.sum(roi)) * vox_cc
                fg_raw = segmentor.segment_fibroglandular(ct_snap, roi)
                fg_clean = segmentor.exclude_non_parenchymal(fg_raw, ct_snap)
                # Intersect with the drawn ROI so the cleaned fibroglandular mask
                # never extends beyond the manually defined breast boundary.
                fg_clean = fg_clean & roi
                # Apply optional user-drawn tumor exclusion zone
                if tumor_excl is not None:
                    fg_clean = fg_clean & ~tumor_excl
                fg_vol_cc = float(np.sum(fg_clean)) * vox_cc
                fat_vol_cc = max(0.0, whole_vol_cc - fg_vol_cc)
                fg_masks[side] = fg_clean
                excl_detail[side] = segmentor.compute_exclusion_masks(ct_snap, roi)
                results.update({
                    f"{side}_breast_roi_vol_cc": round(whole_vol_cc, 3),
                    f"{side}_fibroglandular_vol_cc": round(fg_vol_cc, 3),
                    f"{side}_fat_vol_cc": round(fat_vol_cc, 3),
                    f"{side}_volumetric_density_pct": round(
                        fg_vol_cc / whole_vol_cc * 100.0 if whole_vol_cc > 0 else 0.0, 3),
                    f"{side}_roi_source": "manual_viewer_roi",
                })
            if "right" in fg_masks and "left" in fg_masks:
                r_fg = results["right_fibroglandular_vol_cc"]
                l_fg = results["left_fibroglandular_vol_cc"]
                r_w = results["right_breast_roi_vol_cc"]
                l_w = results["left_breast_roi_vol_cc"]
                total_whole = r_w + l_w
                total_fg = r_fg + l_fg
                results.update({
                    "bilateral_whole_breast_vol_cc": round(total_whole, 3),
                    "bilateral_fibroglandular_vol_cc": round(total_fg, 3),
                    "bilateral_volumetric_density_pct": round(
                        total_fg / total_whole * 100.0 if total_whole > 0 else 0.0, 3),
                    "density_asymmetry_pct": round(
                        abs(results["right_volumetric_density_pct"] -
                            results["left_volumetric_density_pct"]), 3),
                })
            return {"results": results, "fg_masks": fg_masks, "excl_detail": excl_detail}

        self._log_msg("[FG] Running fibroglandular segmentation on labelled ROIs …")
        self._set_busy(True, "Segmenting fibroglandular tissue … please wait")
        self._worker = _BreastWorker(_compute)
        self._worker.finished.connect(self._on_segment_fg_done)
        self._worker.error.connect(self._on_segment_fg_error)
        self._worker.start()

    def _on_segment_fg_done(self, payload: dict):
        self._set_busy(False)
        results = payload.get("results", {})
        fg_masks = payload.get("fg_masks", {})
        self._fg_masks = fg_masks
        self._excl_detail_masks = payload.get("excl_detail", {})
        self._density_results = results
        self._fill_table(self._density_table, results)
        self._result_tabs.setCurrentIndex(0)
        fg_union = None
        for m in fg_masks.values():
            fg_union = m if fg_union is None else (fg_union | m)
        self._parenchymal_mask = fg_union
        self._btn_export.setEnabled(True)
        self._btn_export_fig.setEnabled(BREAST_ML_DEPS_AVAILABLE)  # needs matplotlib
        self._push_seg_overlays()
        for side, mask in fg_masks.items():
            spacing = getattr(self._app, "spacing_zyx", (1.0, 1.0, 1.0))
            vox_cc = float(np.prod([float(s) for s in spacing]) / 1000.0)
            self._log_msg(f"[FG] {side.upper()}: FG={float(np.sum(mask)) * vox_cc:.1f} cc, "
                          f"density={results.get(f'{side}_volumetric_density_pct', 0.0):.1f}%")
        self._log_msg("[FG] Done. Colour overlays: cyan=right whole, blue=left whole, "
                      "orange=right FG, gold=left FG. "
                      "Skin, pectoralis and clips are excluded automatically.")

    def _on_segment_fg_error(self, msg: str):
        self._set_busy(False)
        self._log_msg(f"[FG] ERROR: {msg}")
        QtWidgets.QMessageBox.critical(self, "FG Segmentation Error", msg)

    # ------------------------------------------------------------------
    def _on_compute_density(self):
        """Run automatic whole-breast density in a background thread (non-blocking)."""
        ct_hu = getattr(self._app, "ct_hu", None)
        if ct_hu is None:
            QtWidgets.QMessageBox.warning(self, "Breast Density", "No CT volume loaded. Load a CT study first.")
            return
        spacing = getattr(self._app, "spacing_zyx", (1.0, 1.0, 1.0))
        voxel_spacing_mm = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
        ct_snap = ct_hu.copy()
        ct_sitk_ref = getattr(self._app, "ct_image", None)   # SimpleITK image for TS
        tumor_excl = self._roi_tumor_exclusion.copy() if self._roi_tumor_exclusion is not None else None

        def _compute():
            segmentor = BreastSegmentor()
            whole_masks = segmentor.segment_whole_breast(ct_snap, ct_sitk=ct_sitk_ref)
            vox_cc = float(np.prod(voxel_spacing_mm) / 1000.0)
            fg_masks: dict = {}
            results: dict = {}
            excl_detail: dict = {}
            for side in ("right", "left"):
                breast_mask = whole_masks[f"{side}_mask"]
                fg_raw = segmentor.segment_fibroglandular(ct_snap, breast_mask)
                fg_clean = segmentor.exclude_non_parenchymal(fg_raw, ct_snap)
                # Apply optional user-drawn tumor exclusion zone
                if tumor_excl is not None:
                    fg_clean = fg_clean & ~tumor_excl
                fg_masks[side] = fg_clean
                excl_detail[side] = segmentor.compute_exclusion_masks(ct_snap, breast_mask)
                whole_vol_cc = float(np.sum(breast_mask)) * vox_cc
                fg_vol_cc = float(np.sum(fg_clean)) * vox_cc
                fat_vol_cc = max(0.0, whole_vol_cc - fg_vol_cc)
                results.update({
                    f"{side}_whole_breast_vol_cc": round(whole_vol_cc, 3),
                    f"{side}_fibroglandular_vol_cc": round(fg_vol_cc, 3),
                    f"{side}_fat_vol_cc": round(fat_vol_cc, 3),
                    f"{side}_volumetric_density_pct": round(
                        fg_vol_cc / whole_vol_cc * 100.0 if whole_vol_cc > 0 else 0.0, 3),
                    f"{side}_roi_source": "auto_segmentation",
                })
            r_w = results["right_whole_breast_vol_cc"]
            l_w = results["left_whole_breast_vol_cc"]
            r_fg = results["right_fibroglandular_vol_cc"]
            l_fg = results["left_fibroglandular_vol_cc"]
            total_whole = r_w + l_w
            total_fg = r_fg + l_fg
            results.update({
                "bilateral_whole_breast_vol_cc": round(total_whole, 3),
                "bilateral_fibroglandular_vol_cc": round(total_fg, 3),
                "bilateral_volumetric_density_pct": round(
                    total_fg / total_whole * 100.0 if total_whole > 0 else 0.0, 3),
                "density_asymmetry_pct": round(
                    abs(results["right_volumetric_density_pct"] -
                        results["left_volumetric_density_pct"]), 3),
            })
            return {"results": results, "fg_masks": fg_masks, "whole_masks": whole_masks,
                    "excl_detail": excl_detail,
                    "seg_backend": getattr(segmentor, "_last_seg_backend", "HU-heuristic")}

        if _BD_TS_AVAILABLE:
            ts_note = " (TotalSegmentator — precise thoracic constraints active)"
        else:
            ts_note = " (HU-heuristic fallback)"
            self._log_msg(
                "[Density] WARNING: TotalSegmentator is NOT installed. "
                "Breast segmentation will use a less-accurate HU-based diaphragm "
                "heuristic that may extend below the thorax on some cases. "
                "Install with:  pip install totalsegmentator nibabel"
            )
        self._log_msg(f"[Density] Starting auto-segmentation{ts_note} …")
        self._set_busy(True, "Auto-segmenting whole breast + fibroglandular tissue … please wait")
        self._worker = _BreastWorker(_compute)
        self._worker.finished.connect(self._on_density_done)
        self._worker.error.connect(self._on_density_error)
        self._worker.start()

    def _on_density_done(self, payload: dict):
        self._set_busy(False)
        results = payload.get("results", {})
        fg_masks = payload.get("fg_masks", {})
        whole_masks = payload.get("whole_masks", {})
        self._fg_masks = fg_masks
        self._excl_detail_masks = payload.get("excl_detail", {})
        self._whole_masks = whole_masks
        self._density_results = results
        self._fill_table(self._density_table, results)
        self._result_tabs.setCurrentIndex(0)
        if "right" in fg_masks and "left" in fg_masks:
            self._parenchymal_mask = fg_masks["right"] | fg_masks["left"]
        self._btn_export.setEnabled(True)
        self._btn_export_fig.setEnabled(BREAST_ML_DEPS_AVAILABLE)  # needs matplotlib
        self._push_seg_overlays()
        for side in ("right", "left"):
            self._log_msg(f"[Density] {side.upper()}: "
                          f"whole={results.get(f'{side}_whole_breast_vol_cc', 0.0):.1f} cc, "
                          f"FG={results.get(f'{side}_fibroglandular_vol_cc', 0.0):.1f} cc, "
                          f"density={results.get(f'{side}_volumetric_density_pct', 0.0):.1f}%")
        backend = payload.get("seg_backend", "HU-heuristic")
        self._log_msg(f"[Density] Done. Segmentation backend: {backend}. "
                      "Colour overlays: cyan=right whole, blue=left whole, "
                      "orange=right FG, gold=left FG. "
                      "Skin, pectoralis and clips are excluded automatically.")

    def _on_density_error(self, msg: str):
        self._set_busy(False)
        self._log_msg(f"[Density] ERROR: {msg}")
        QtWidgets.QMessageBox.critical(self, "Breast Density Error", msg)

    # ------------------------------------------------------------------
    def _on_compute_complexity(self):
        """Run bilateral complexity radiomics in a background thread (non-blocking)."""
        ct_hu = getattr(self._app, "ct_hu", None)
        if ct_hu is None or self._parenchymal_mask is None:
            QtWidgets.QMessageBox.warning(self, "Breast Complexity",
                                          "Run 'Segment Fibroglandular Tissue' or 'Compute Density (Auto)' first "
                                          "to generate a parenchymal mask.")
            return
        spacing = getattr(self._app, "spacing_zyx", (1.0, 1.0, 1.0))
        voxel_spacing_mm = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
        ct_snap = ct_hu.copy()
        mask_snap = self._parenchymal_mask.copy()

        def _compute():
            engine = ParenchymalComplexityEngine(ct_snap, mask_snap, voxel_spacing_mm)
            # skip_shape=True: bilateral merged mask spans both breasts as a disconnected
            # object — shape features (MajorAxisLength, sphericity, etc.) would be
            # biologically meaningless (e.g. >1000 mm axis).  Shape must only be
            # extracted per-breast via the individual side buttons.
            return {"complexity": engine.compute_all(skip_shape=True)}

        self._log_msg("[Complexity] Computing bilateral radiomics features "
                      "(shape features suppressed — use per-breast buttons for shape) …")
        self._set_busy(True, "Computing bilateral radiomics … please wait")
        self._worker = _BreastWorker(_compute)
        self._worker.finished.connect(self._on_complexity_done)
        self._worker.error.connect(self._on_complexity_error)
        self._worker.start()

    def _on_complexity_done(self, payload: dict):
        self._set_busy(False)
        complexity = payload.get("complexity", {})
        self._complexity_results = complexity
        prefixed = {f"bilateral_{k}" if not k.startswith("bilateral_") else k: v
                    for k, v in complexity.items() if k != "manuscript_shortlist"}
        self._fill_table(self._shortlist_table, complexity.get("manuscript_shortlist", {}))
        self._fill_table(self._full_table, prefixed)
        self._result_tabs.setCurrentIndex(1)
        self._btn_export.setEnabled(True)
        self._log_msg("[Complexity] Done (bilateral).")

    def _on_complexity_error(self, msg: str):
        self._set_busy(False)
        self._log_msg(f"[Complexity] ERROR: {msg}")
        QtWidgets.QMessageBox.critical(self, "Complexity Error", msg)

    def _on_compute_complexity_side(self, side: str):
        """Run per-breast complexity radiomics in a background thread (non-blocking)."""
        ct_hu = getattr(self._app, "ct_hu", None)
        fg_mask = self._fg_masks.get(side)
        if ct_hu is None or fg_mask is None:
            QtWidgets.QMessageBox.warning(
                self, f"Complexity: {side.upper()}",
                f"No fibroglandular mask for the {side.upper()} breast.\n"
                "Run 'Segment Fibroglandular Tissue' or 'Compute Density (Auto)' first."
            )
            return
        spacing = getattr(self._app, "spacing_zyx", (1.0, 1.0, 1.0))
        voxel_spacing_mm = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
        ct_snap = ct_hu.copy()
        mask_snap = fg_mask.copy()

        def _compute():
            engine = ParenchymalComplexityEngine(ct_snap, mask_snap, voxel_spacing_mm)
            return {"complexity": engine.compute_all(), "side": side}

        self._log_msg(f"[Complexity] Computing {side.upper()} breast radiomics …")
        self._set_busy(True, f"Computing {side.upper()} breast radiomics … please wait")
        self._worker = _BreastWorker(_compute)
        self._worker.finished.connect(self._on_complexity_side_done)
        self._worker.error.connect(lambda msg, s=side: self._on_complexity_side_error(s, msg))
        self._worker.start()

    def _on_complexity_side_done(self, payload: dict):
        self._set_busy(False)
        side = payload.get("side", "unknown")
        results = payload.get("complexity", {})
        prefixed = {f"{side}_{k}" if not k.startswith(side + "_") else k: v
                    for k, v in results.items() if k != "manuscript_shortlist"}
        self._complexity_results.update(prefixed)
        self._fill_table(self._shortlist_table, results.get("manuscript_shortlist", {}))
        full = {k: v for k, v in self._complexity_results.items() if k != "manuscript_shortlist"}
        self._fill_table(self._full_table, full)
        self._result_tabs.setCurrentIndex(2)
        self._btn_export.setEnabled(True)
        self._log_msg(f"[Complexity] Done ({side.upper()}).")

    def _on_complexity_side_error(self, side: str, msg: str):
        self._set_busy(False)
        self._log_msg(f"[Complexity] ERROR ({side.upper()}): {msg}")
        QtWidgets.QMessageBox.critical(self, f"Complexity {side.upper()} Error", msg)

    def _on_export_csv(self):
        if not self._density_results and not self._complexity_results:
            QtWidgets.QMessageBox.information(self, "Export", "No results to export yet.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Breast Density Report", "breast_density_report.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        payload = {**self._density_results,
                   **{k: v for k, v in self._complexity_results.items() if k != "manuscript_shortlist"}}
        try:
            import csv as _csv
            with open(path, "w", newline="") as f:
                writer = _csv.DictWriter(f, fieldnames=list(payload.keys()))
                writer.writeheader()
                writer.writerow(payload)
            self._log_msg(f"[Export] Saved to {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(exc))

    # ------------------------------------------------------------------
    def _on_export_figure(self):
        """Export a publication-quality segmentation figure (300 DPI PNG/SVG).

        Colour scheme
        -------------
        Whole breast (R)        cyan          (0,   200, 200)
        Whole breast (L)        steel-blue    (70,  130, 180)
        Fibroglandular (R)      orange        (255, 140, 0)
        Fibroglandular (L)      gold          (220, 200, 0)
        Skin shell              olive-green   (107, 142, 35)
        Pectoralis muscle       magenta       (200, 0,   200)
        Metal clips / calcif.   yellow        (255, 255, 0)
        Blood vessels           lime-green    (0,   230, 115)
        Tumor exclusion zone    red           (220, 30,  30)
        """
        if not BREAST_ML_DEPS_AVAILABLE:
            QtWidgets.QMessageBox.warning(
                self, "Export Figure",
                "matplotlib is required for figure export.\n"
                "Install with:  pip install matplotlib"
            )
            return
        ct_hu = getattr(self._app, "ct_hu", None)
        if ct_hu is None:
            QtWidgets.QMessageBox.warning(self, "Export Figure", "No CT volume loaded.")
            return
        if not self._fg_masks:
            QtWidgets.QMessageBox.warning(
                self, "Export Figure",
                "No segmentation results yet.\n"
                "Run 'Segment Fibroglandular Tissue' or 'Compute Density (Auto)' first."
            )
            return

        path, fmt = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Segmentation Figure",
            "breast_segmentation_figure.png",
            "PNG image (*.png);;SVG vector (*.svg);;TIFF image (*.tif)"
        )
        if not path:
            return

        # Collect masks (all optional)
        _wm = self._whole_masks
        right_whole = _wm.get("right_mask") if _wm.get("right_mask") is not None else self._roi_right
        left_whole  = _wm.get("left_mask")  if _wm.get("left_mask")  is not None else self._roi_left
        right_fg    = self._fg_masks.get("right")
        left_fg     = self._fg_masks.get("left")
        tumor_excl  = self._roi_tumor_exclusion

        def _get_side_detail(side: str, key: str):
            d = self._excl_detail_masks.get(side)
            return d.get(key) if d else None

        # Build per-side centroid specs: {label: (z_cent, y_cent, x_cent)}
        # For each active side we derive the representative slice index for each
        # orientation (axial→z, coronal→y, sagittal→x) from the breast centroid.
        def _mask_centroid_indices(mask):
            """Return (z_c, y_c, x_c) medial indices, or None if mask is empty."""
            if mask is None or not mask.any():
                return None
            z_idx = np.where(mask.any(axis=(1, 2)))[0]
            y_idx = np.where(mask.any(axis=(0, 2)))[0]
            x_idx = np.where(mask.any(axis=(0, 1)))[0]
            if not (z_idx.size and y_idx.size and x_idx.size):
                return None
            return (
                int(z_idx[len(z_idx) // 2]),
                int(y_idx[len(y_idx) // 2]),
                int(x_idx[len(x_idx) // 2]),
            )

        side_centroids: dict = {}  # label → (z_c, y_c, x_c)
        for label, whole_mask in (("RIGHT", right_whole), ("LEFT", left_whole)):
            c = _mask_centroid_indices(whole_mask)
            if c is None:
                # Fallback to fibroglandular mask centroid
                fg = right_fg if label == "RIGHT" else left_fg
                c = _mask_centroid_indices(fg)
            if c is not None:
                side_centroids[label] = c

        if not side_centroids:
            QtWidgets.QMessageBox.warning(
                self, "Export Figure",
                "Could not determine representative slices.\n"
                "Make sure at least one breast ROI or fibroglandular mask is non-empty."
            )
            return

        try:
            import matplotlib as _mpl
            if _mpl.get_backend().lower() not in ("agg",):
                try:
                    _mpl.use("Agg")
                except Exception:
                    pass  # backend already set; proceed anyway
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches

            spacing_zyx = getattr(self._app, "spacing_zyx", (1.0, 1.0, 1.0))
            # HU window: breast soft-tissue window
            hu_lo, hu_hi = -300.0, 200.0

            # ---------------------------------------------------------------
            # Three orientations × N sides grid
            # Row 0: Axial    (slice through Z at centroid z)
            # Row 1: Coronal  (slice through Y at centroid y) — Z rows, X cols
            # Row 2: Sagittal (slice through X at centroid x) — Z rows, Y cols
            # ---------------------------------------------------------------
            orientations = [
                ("Axial",    0,  "z", float(spacing_zyx[0])),
                ("Coronal",  1,  "y", float(spacing_zyx[1])),
                ("Sagittal", 2,  "x", float(spacing_zyx[2])),
            ]
            side_labels = list(side_centroids.keys())
            n_rows = len(orientations)
            n_cols = len(side_labels)

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(6.5 * n_cols, 6.0 * n_rows),
                dpi=300,
                facecolor="white",
                squeeze=False,
            )

            legend_handles = []
            legend_added: set = set()

            def _add_legend(colour_rgba, label):
                if label not in legend_added:
                    legend_added.add(label)
                    patch = mpatches.Patch(color=colour_rgba, label=label, alpha=0.85)
                    legend_handles.append(patch)

            def _get_ct_slice(orient_axis: int, idx: int) -> np.ndarray:
                """Return a 2-D CT slice for the given orientation and index."""
                if orient_axis == 0:        # axial: ct[z, :, :]
                    return ct_hu[idx].astype(float)
                elif orient_axis == 1:      # coronal: ct[:, y, :]  → (Z, X)
                    return ct_hu[:, idx, :].astype(float)
                else:                       # sagittal: ct[:, :, x] → (Z, Y)
                    return ct_hu[:, :, idx].astype(float)

            def _get_mask_slice(mask_3d, orient_axis: int, idx: int):
                """Return a 2-D boolean slice for the given orientation and index."""
                if mask_3d is None:
                    return None
                if orient_axis == 0:
                    return mask_3d[idx].astype(bool)
                elif orient_axis == 1:
                    return mask_3d[:, idx, :].astype(bool)
                else:
                    return mask_3d[:, :, idx].astype(bool)

            # Pixel aspect ratios per orientation so that anatomical proportions are
            # preserved when voxels are anisotropic (e.g. z-spacing >> x/y-spacing).
            #   axial    (Y rows, X cols): aspect = sy / sx
            #   coronal  (Z rows, X cols): aspect = sz / sx
            #   sagittal (Z rows, Y cols): aspect = sz / sy
            sz = float(spacing_zyx[0])
            sy = float(spacing_zyx[1])
            sx = float(spacing_zyx[2])
            orient_pixel_aspect = {
                0: sy / sx if sx > 0 else 1.0,
                1: sz / sx if sx > 0 else 1.0,
                2: sz / sy if sy > 0 else 1.0,
            }

            def _overlay_2d(ax, sl_2d, rgba, pix_aspect=1.0):
                """Blend a 2-D boolean slice as a solid-colour RGBA overlay."""
                if sl_2d is None or not sl_2d.any():
                    return
                h, w = sl_2d.shape
                img = np.zeros((h, w, 4), dtype=np.float32)
                img[sl_2d, 0] = rgba[0]
                img[sl_2d, 1] = rgba[1]
                img[sl_2d, 2] = rgba[2]
                img[sl_2d, 3] = rgba[3]
                ax.imshow(img, origin="upper", aspect=pix_aspect, interpolation="none")

            # Layer table: (mask_3d, RGBA, legend_label)
            # Rendering order matters: layers drawn later appear on top.
            # Tumor exclusion zone is placed FIRST (background) so that the
            # sub-tissue layers (skin, pectoralis, fibroglandular) remain visible
            # on top of the excluded region.  Its alpha is kept semi-transparent.
            layers = [
                (right_whole, (0.00, 0.78, 0.78, 0.30), "Whole breast (R)"),
                (left_whole,  (0.27, 0.51, 0.71, 0.30), "Whole breast (L)"),
                (tumor_excl,  (0.86, 0.12, 0.12, 0.55), "Tumor exclusion zone"),
                (_get_side_detail("right", "skin"),       (0.42, 0.56, 0.14, 0.75), "Skin shell"),
                (_get_side_detail("left",  "skin"),       (0.42, 0.56, 0.14, 0.75), "Skin shell"),
                (_get_side_detail("right", "pectoralis"), (0.78, 0.00, 0.78, 0.75), "Pectoralis muscle"),
                (_get_side_detail("left",  "pectoralis"), (0.78, 0.00, 0.78, 0.75), "Pectoralis muscle"),
                (_get_side_detail("right", "clips"),      (1.00, 1.00, 0.00, 0.85), "Clips / calcifications"),
                (_get_side_detail("left",  "clips"),      (1.00, 1.00, 0.00, 0.85), "Clips / calcifications"),
                (_get_side_detail("right", "vessels"),    (0.00, 0.90, 0.45, 0.70), "Blood vessels"),
                (_get_side_detail("left",  "vessels"),    (0.00, 0.90, 0.45, 0.70), "Blood vessels"),
                (right_fg,    (1.00, 0.55, 0.00, 0.80), "Fibroglandular (R)"),
                (left_fg,     (0.86, 0.78, 0.00, 0.80), "Fibroglandular (L)"),
            ]

            for r_idx, (orient_name, orient_axis, ax_letter, sp_mm) in enumerate(orientations):
                pix_aspect = orient_pixel_aspect[orient_axis]
                for c_idx, side_label in enumerate(side_labels):
                    ax = axes[r_idx, c_idx]
                    z_c, y_c, x_c = side_centroids[side_label]
                    # Pick the index appropriate for this orientation
                    slice_idx = (z_c, y_c, x_c)[orient_axis]

                    ct_sl = _get_ct_slice(orient_axis, slice_idx)
                    ct_norm = np.clip((ct_sl - hu_lo) / (hu_hi - hu_lo), 0.0, 1.0)
                    ax.imshow(ct_norm, cmap="gray", origin="upper", aspect=pix_aspect,
                              interpolation="bilinear", vmin=0.0, vmax=1.0)

                    for mask, rgba, lbl in layers:
                        sl_2d = _get_mask_slice(mask, orient_axis, slice_idx)
                        _overlay_2d(ax, sl_2d, rgba, pix_aspect)
                        if sl_2d is not None and sl_2d.any():
                            _add_legend(rgba, lbl)

                    ax.set_title(
                        f"{side_label}  —  {orient_name}  "
                        f"{ax_letter}={slice_idx}\n"
                        f"({ax_letter}-spacing {sp_mm:.2f} mm)",
                        fontsize=9, fontweight="bold", color="#1a1a2e",
                    )
                    ax.axis("off")

            # Shared legend at the bottom
            if legend_handles:
                fig.legend(
                    handles=legend_handles,
                    loc="lower center",
                    ncol=min(len(legend_handles), 5),
                    fontsize=8,
                    framealpha=0.9,
                    edgecolor="#aaaaaa",
                    bbox_to_anchor=(0.5, 0.0),
                )

            fig.suptitle("Breast CT Segmentation — Tissue Layers\n"
                         "(rows: Axial · Coronal · Sagittal)",
                         fontsize=12, fontweight="bold", y=1.01, color="#1a1a2e")
            fig.tight_layout(rect=[0, 0.06, 1, 1])

            ext = path.rsplit(".", 1)[-1].lower() if "." in path else "png"
            fig.savefig(path, dpi=300, bbox_inches="tight",
                        format=ext if ext in ("png", "svg", "tif", "tiff", "pdf") else "png",
                        facecolor="white")
            plt.close(fig)
            self._log_msg(f"[Figure] Saved to {path}")
            QtWidgets.QMessageBox.information(
                self, "Export Figure",
                f"Segmentation figure saved:\n{path}\n\n"
                f"Views: {', '.join(side_labels)}  ×  Axial / Coronal / Sagittal"
            )
        except Exception as exc:
            self._log_msg(f"[Figure] ERROR: {exc}")
            QtWidgets.QMessageBox.critical(self, "Export Figure Error", str(exc))

    def _on_browse_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open CSV for ML Analysis", "", "CSV files (*.csv)"
        )
        if path:
            self._csv_path_edit.setText(path)

    def _on_run_ml(self, mode: str):
        if not BREAST_ML_DEPS_AVAILABLE:
            return
        csv_path = self._csv_path_edit.text().strip()
        if not csv_path or not os.path.exists(csv_path):
            QtWidgets.QMessageBox.warning(self, "ML Analysis", "Select a valid CSV file first.")
            return
        self._log_msg(f"[ML] Running {mode} analysis on {csv_path} …")
        QtWidgets.QApplication.processEvents()
        try:
            df = _bd_pd.read_csv(csv_path)
            if "diagnosis" not in df.columns:
                raise ValueError("CSV must include a 'diagnosis' column.")
            if mode == "density":
                numeric_cols = [c for c in df.columns if c != "diagnosis" and
                                _bd_pd.api.types.is_numeric_dtype(df[c])]
                auc_results, curves = _bd_train_classifiers(
                    df.dropna(subset=["diagnosis"] + numeric_cols), numeric_cols, "diagnosis"
                )
                roc_out = os.path.join(os.path.dirname(csv_path), "roc_curve_density.png")
                title = "ROC Curve — Breast Density Features"
            else:
                available_features = [col for col in _BD_MANUSCRIPT_SHORTLIST_FEATURES if col in df.columns]
                if not available_features:
                    raise ValueError("CSV does not contain manuscript shortlist feature columns.")
                usable = df.dropna(subset=["diagnosis"] + available_features)
                auc_results, curves = _bd_train_classifiers(usable, available_features, "diagnosis")
                roc_out = os.path.join(os.path.dirname(csv_path), "roc_curve_complexity.png")
                title = "ROC Curve — Parenchymal Complexity Features"

            _bd_plt.figure(figsize=(7, 6))
            for name, (fpr, tpr) in curves.items():
                _bd_plt.plot(fpr, tpr, label=f"{name} (AUC={auc_results[name]:.2f})")
            _bd_plt.plot([0, 1], [0, 1], "k--")
            _bd_plt.xlabel("False Positive Rate")
            _bd_plt.ylabel("True Positive Rate")
            _bd_plt.title(title)
            _bd_plt.legend(loc="lower right")
            _bd_plt.tight_layout()
            _bd_plt.savefig(roc_out, dpi=150)
            _bd_plt.close()

            auc_txt = "  ".join(f"{n}: {v:.3f}" for n, v in auc_results.items())
            self._ml_result_label.setText(f"AUC results: {auc_txt}\nROC plot → {roc_out}")
            self._log_msg(f"[ML] {auc_txt}")
        except Exception as exc:
            self._log_msg(f"[ML] ERROR: {exc}")
            QtWidgets.QMessageBox.critical(self, "ML Error", str(exc))

    def on_show(self):
        pass  # nothing to auto-refresh; user triggers analysis explicitly


# =========================
# v21 UI patch v7 (Clinical Tools as dialog + toolbar menu; no floating dock)
# =========================
# NOTE: Use the existing Qt binding imported at the top of this file (PyQt6 in this build).
# Do NOT import PyQt5 here, or you'll create a mixed-binding crash (PyQt5 QApplication + PyQt6 QWidget).
def _v21_install_clinical_dialog_and_menu(app):
    try:
        _QMainWindow = QtWidgets.QMainWindow
    except Exception:
        return
    if not isinstance(app, _QMainWindow):
        return
        return

    # ---------- Dialog builders ----------
    def _get_dialog():
        dlg = getattr(app, "_v21_clinical_dlg", None)
        if isinstance(dlg, QtWidgets.QDialog):
            return dlg

        dlg = QtWidgets.QDialog(app)
        dlg.setWindowTitle("Clinical Tools")
        dlg.setModal(False)
        try:
            dlg.resize(1050, 760)
        except Exception:
            pass

        v = QtWidgets.QVBoxLayout(dlg)
        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)

        # Create fresh widgets owned by this dialog (avoid dock parenting issues)
        dlg.scaninfo_tab = V21ScanInfoTab(app)
        dlg.percist_tab = V21PercistTab(app)
        dlg.recist_tab = V21RecistTab(app)
        dlg.display_tab = V21DisplayTab(app)
        dlg.export_tab = V21ExportTab(app)
        dlg.metrics3d_tab = V21Tumor3DTab(app)
        dlg.nhoc_tab = V21NHOCNHOPTab(app)
        dlg.breast_density_tab = V21BreastDensityTab(app)

        tabs.addTab(dlg.scaninfo_tab, "Scan Info")
        tabs.addTab(dlg.percist_tab, "PERCIST")
        tabs.addTab(dlg.recist_tab, "RECIST 1.1")
        tabs.addTab(dlg.display_tab, "Display")
        tabs.addTab(dlg.export_tab, "Export")
        tabs.addTab(dlg.metrics3d_tab, "3D Metrics")
        tabs.addTab(dlg.nhoc_tab, "NHOC/NHOP")
        tabs.addTab(dlg.breast_density_tab, "Breast Density")

        def _on_tab(i):
            w = tabs.widget(i)
            if hasattr(w, "on_show"):
                try:
                    w.on_show()
                except Exception:
                    pass
            # Always refresh Scan Info when shown
            if w is getattr(dlg, "scaninfo_tab", None):
                try:
                    dlg.scaninfo_tab.refresh()
                except Exception:
                    pass

        tabs.currentChanged.connect(_on_tab)

        v.addWidget(tabs, 1)
        dlg._v21_tabs = tabs
        app._v21_clinical_dlg = dlg
        return dlg

    def _show_dialog(tab_title=None):
        # If nothing loaded, still allow dialog to open but warn gently
        if getattr(app, "pet_suv", None) is None and getattr(app, "ct_hu", None) is None:
            try:
                QtWidgets.QMessageBox.information(app, "Clinical Tools", "Load a PET/CT or CECT study first.")
            except Exception:
                pass

        dlg = _get_dialog()
        tabs = getattr(dlg, "_v21_tabs", None)

        if tab_title and isinstance(tabs, QtWidgets.QTabWidget):
            # find matching tab
            want = tab_title.strip().lower()
            for i in range(tabs.count()):
                if want in tabs.tabText(i).lower():
                    tabs.setCurrentIndex(i)
                    break

        try:
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
        except Exception:
            pass

    # Expose helper on app for other code paths
    try:
        app._v21_show_clinical_tools = _show_dialog
    except Exception:
        pass

    # ---------- Remove/close old dock if present (prevents tiny floating "Clinical T..." window) ----------
    old_dock = getattr(app, "_v21_tools_dock", None)
    if isinstance(old_dock, QtWidgets.QDockWidget):
        try:
            old_dock.hide()
            old_dock.setParent(None)
            old_dock.deleteLater()
        except Exception:
            pass
    for _attr in ("_v21_tools_dock", "_v21_tools_tabs", "tabDock"):
        if hasattr(app, _attr):
            try:
                delattr(app, _attr)
            except Exception:
                pass

    # ---------- Toolbar: Clinical Tools drop-down + quick buttons ----------
    # Remove previous toolbar if we created it earlier
    tb_old = getattr(app, "_v21_tools_toolbar", None)
    if isinstance(tb_old, QtWidgets.QToolBar):
        try:
            app.removeToolBar(tb_old)
            tb_old.deleteLater()
        except Exception:
            pass

    tb = QtWidgets.QToolBar("Clinical", app)
    tb.setMovable(False)
    try:
        app.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, tb)
    except Exception:
        app.addToolBar(tb)

    # Drop-down button
    btn = QtWidgets.QToolButton()
    btn.setText("Clinical Tools")
    btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
    btn.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
    menu = QtWidgets.QMenu(btn)
    menu.addAction("Open Clinical Tools Window", lambda: _show_dialog(None))
    menu.addSeparator()
    menu.addAction("Scan Info", lambda: _show_dialog("Scan Info"))
    menu.addAction("3D Metrics", lambda: _show_dialog("3D Metrics"))
    menu.addAction("NHOC/NHOP", lambda: _show_dialog("NHOC/NHOP"))
    menu.addSeparator()
    menu.addAction("Breast Density", lambda: _show_dialog("Breast Density"))
    menu.addSeparator()
    menu.addAction("PERCIST", lambda: _show_dialog("PERCIST"))
    menu.addAction("RECIST 1.1", lambda: _show_dialog("RECIST"))
    menu.addAction("Display", lambda: _show_dialog("Display"))
    menu.addAction("Export", lambda: _show_dialog("Export"))
    btn.setMenu(menu)
    tb.addWidget(btn)

    tb.addSeparator()
    tb.addAction(QtGui.QAction("Tumor Burden", app, triggered=lambda: app._show_tumor_burden()))
    tb.addAction(QtGui.QAction("Scan Info", app, triggered=lambda: _show_dialog("Scan Info")))
    tb.addAction(QtGui.QAction("3D Metrics", app, triggered=lambda: _show_dialog("3D Metrics")))
    tb.addAction(QtGui.QAction("NHOC/NHOP", app, triggered=lambda: _show_dialog("NHOC/NHOP")))
    tb.addAction(QtGui.QAction("Breast Density", app, triggered=lambda: _show_dialog("Breast Density")))

    app._v21_tools_toolbar = tb

    # ---------- Menu bar entry ----------
    try:
        mb = app.menuBar()
        # Remove any previous "Tools" injected items? We'll just add/refresh Clinical Tools menu.
        clin_menu = None
        for a in mb.actions():
            if a.text().replace("&","").strip().lower() == "clinical tools":
                clin_menu = a.menu()
                break
        if clin_menu is None:
            clin_menu = mb.addMenu("Clinical Tools")

        clin_menu.clear()
        clin_menu.addAction("Open Clinical Tools Window", lambda: _show_dialog(None))
        clin_menu.addSeparator()
        clin_menu.addAction("Tumor Burden", lambda: app._show_tumor_burden())
        clin_menu.addAction("Scan Info", lambda: _show_dialog("Scan Info"))
        clin_menu.addAction("3D Metrics", lambda: _show_dialog("3D Metrics"))
        clin_menu.addAction("NHOC/NHOP", lambda: _show_dialog("NHOC/NHOP"))
        clin_menu.addSeparator()
        clin_menu.addAction("Breast Density", lambda: _show_dialog("Breast Density"))
    except Exception:
        pass


# ---------- Override load/build patches to avoid creating the old floating dock ----------
try:
    _v21_orig_load  # noqa: F821
except Exception:
    _v21_orig_load = None

try:
    if _v21_orig_load is not None:
        def _load_pet_ct_v6(self, *a, **k):
            _v21_orig_load(self, *a, **k)

            # store PET Bq/mL and ds0 if captured
            try:
                self.pet_bqml = globals().get("_V21_LAST_PET_BQML", None)
                self.pet_ds0 = globals().get("_V21_LAST_PET_DS0", None)
            except Exception:
                pass

            # Fallback: find PET/CT datasets for Scan Info
            if getattr(self, "dicom_dir", None):
                if getattr(self, "pet_ds0", None) is None:
                    try:
                        self.pet_ds0 = _v21_find_first_pet_dataset(self.dicom_dir)  # noqa: F821
                    except Exception:
                        self.pet_ds0 = None
                try:
                    self.ct_ds0 = _v21_find_first_ct_dataset(self.dicom_dir)  # noqa: F821
                except Exception:
                    self.ct_ds0 = None

            # Install dialog/menu (no dock)
            try:
                _v21_install_clinical_dialog_and_menu(self)
            except Exception:
                pass

        PETCTManualROIApp._load_pet_ct = _load_pet_ct_v6
except Exception:
    pass

try:
    _v21_orig_build  # noqa: F821
except Exception:
    _v21_orig_build = None

try:
    if _v21_orig_build is not None:
        def _build_ui_v6(self, *a, **k):
            _v21_orig_build(self, *a, **k)
            try:
                _v21_install_clinical_dialog_and_menu(self)
            except Exception:
                pass

        PETCTManualROIApp._build_ui = _build_ui_v6
except Exception:
    pass

if __name__ == "__main__":
    main()