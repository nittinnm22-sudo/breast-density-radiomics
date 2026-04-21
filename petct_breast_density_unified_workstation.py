#!/usr/bin/env python

# ==============================
# Imports
# ==============================
import csv
import os
import threading
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    tk = None
    filedialog = None
    messagebox = None
    ttk = None
    FigureCanvasTkAgg = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None


# ==============================
# Breast Density Analysis Engine
# ==============================
class BreastDensityAnalyzer:
    def __init__(self, image_array):
        self.image_array = np.asarray(image_array) if image_array is not None else np.array([])

    def compute_density(self):
        if self.image_array.size == 0:
            return 0.0
        arr = self.image_array.astype(float)
        threshold = np.percentile(arr, 75)
        return float((np.sum(arr > threshold) / arr.size) * 100.0)

    def classify_birads(self, density_pct):
        if density_pct < 25:
            return "A"
        if density_pct < 50:
            return "B"
        if density_pct < 75:
            return "C"
        return "D"

    def generate_report(self):
        density_pct = self.compute_density()
        return {
            "density_pct": round(density_pct, 2),
            "birads": self.classify_birads(density_pct),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }


# ==============================
# Breast Density Dialog
# ==============================
if tk is not None:
    class BreastDensityDialog(tk.Toplevel):
        def __init__(self, parent, image_array=None):
            super().__init__(parent)
            self.title("Breast Density Analysis")
            self.geometry("640x520")
            self.analyzer = BreastDensityAnalyzer(image_array)
            self.report = self.analyzer.generate_report()
            self._build_ui()

        def _build_ui(self):
            body = ttk.Frame(self, padding=10)
            body.pack(fill="both", expand=True)

            ttk.Label(body, text=f"Density: {self.report['density_pct']:.2f}%", font=("Arial", 12, "bold")).pack(anchor="w")
            ttk.Label(body, text=f"BI-RADS Category: {self.report['birads']}", font=("Arial", 12)).pack(anchor="w", pady=(0, 8))

            fig, ax = plt.subplots(figsize=(6, 3))
            if self.analyzer.image_array.size:
                ax.hist(self.analyzer.image_array.ravel(), bins=40, color="#3366cc", alpha=0.85)
            else:
                ax.text(0.5, 0.5, "No image loaded", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Pixel Intensity Histogram")
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Frequency")
            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=body)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            ttk.Button(body, text="Export Report", command=self._export_report).pack(anchor="e", pady=(10, 0))

        def _export_report(self):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"breast_density_report_{timestamp}.csv"
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["density_pct", "birads", "timestamp"])
                writer.writeheader()
                writer.writerow(self.report)
            messagebox.showinfo("Export Complete", f"Report saved:\n{os.path.abspath(filename)}")
else:
    BreastDensityDialog = None


# ==============================
# Toolbar Integration Function
# ==============================
def _install_breast_density_toolbar(app_instance):
    if BreastDensityDialog is None:
        return

    def _open_dialog():
        BreastDensityDialog(app_instance, image_array=app_instance.current_image_array)

    ttk.Button(app_instance.toolbar_frame, text="Breast Density", command=_open_dialog).pack(side="left", padx=4)


# ==============================
# PET/CT Manual ROI Workstation
# ==============================
if tk is not None:
    class PETCTManualROIApp(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("PET/CT Breast Density Unified Workstation")
            self.geometry("1024x760")
            self.current_image_array = None
            self.current_image_title = ""
            self._build_ui()

        def _build_ui(self):
            menubar = tk.Menu(self)

            file_menu = tk.Menu(menubar, tearoff=0)
            file_menu.add_command(label="Load PET", command=self.load_pet)
            file_menu.add_command(label="Load CT", command=self.load_ct)
            file_menu.add_separator()
            file_menu.add_command(label="Exit", command=self.destroy)
            menubar.add_cascade(label="File", menu=file_menu)

            tools_menu = tk.Menu(menubar, tearoff=0)
            tools_menu.add_command(label="Draw ROI", command=self.draw_roi)
            tools_menu.add_command(label="Export Metrics", command=self.export_metrics)
            menubar.add_cascade(label="Tools", menu=tools_menu)

            help_menu = tk.Menu(menubar, tearoff=0)
            help_menu.add_command(label="About", command=lambda: messagebox.showinfo("About", "Unified PET/CT breast density workstation"))
            menubar.add_cascade(label="Help", menu=help_menu)
            self.config(menu=menubar)

            self.toolbar_frame = ttk.Frame(self, padding=6)
            self.toolbar_frame.pack(side="top", fill="x")
            ttk.Button(self.toolbar_frame, text="Load PET", command=self.load_pet).pack(side="left", padx=4)
            ttk.Button(self.toolbar_frame, text="Load CT", command=self.load_ct).pack(side="left", padx=4)
            ttk.Button(self.toolbar_frame, text="Draw ROI", command=self.draw_roi).pack(side="left", padx=4)
            ttk.Button(self.toolbar_frame, text="Export Metrics", command=self.export_metrics).pack(side="left", padx=4)
            _install_breast_density_toolbar(self)

            content = ttk.Frame(self, padding=8)
            content.pack(fill="both", expand=True)
            self.image_title = ttk.Label(content, text="No image loaded")
            self.image_title.pack(anchor="w")

            self.display_figure, self.display_ax = plt.subplots(figsize=(8, 6))
            self.display_ax.axis("off")
            self.display_canvas = FigureCanvasTkAgg(self.display_figure, master=content)
            self.display_canvas.draw()
            self.display_canvas.get_tk_widget().pack(fill="both", expand=True)

            self.status_var = tk.StringVar(value="Ready")
            ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w").pack(side="bottom", fill="x")

        def load_pet(self):
            self._load_and_display_image("Load PET Image")

        def load_ct(self):
            self._load_and_display_image("Load CT Image")

        def _load_and_display_image(self, title):
            file_path = filedialog.askopenfilename(
                title=title,
                filetypes=[("Image files", "*.npy *.nii *.nii.gz *.mha *.mhd *.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
            )
            if not file_path:
                return
            try:
                array = self._read_image_as_array(file_path)
            except Exception as exc:
                messagebox.showerror("Load Error", f"Could not load image:\n{exc}")
                return

            self.current_image_array = array
            self.current_image_title = os.path.basename(file_path)
            self._display_image(array, self.current_image_title)
            self.status_var.set(f"Loaded {self.current_image_title}")

        def _read_image_as_array(self, file_path):
            lower = file_path.lower()
            if lower.endswith(".npy"):
                return np.load(file_path)

            if sitk is not None and (lower.endswith(".nii") or lower.endswith(".nii.gz") or lower.endswith(".mha") or lower.endswith(".mhd")):
                image = sitk.ReadImage(file_path)
                arr = sitk.GetArrayFromImage(image)
                return arr[0] if arr.ndim >= 3 else arr

            if Image is not None:
                with Image.open(file_path) as img:
                    return np.array(img.convert("L"))

            raise RuntimeError("Install Pillow for standard images or SimpleITK for medical volumes.")

        def draw_roi(self):
            self.status_var.set("ROI drawing mode activated (placeholder)")
            messagebox.showinfo("Draw ROI", "ROI drawing mode is a placeholder in this unified script.")

        def export_metrics(self):
            if self.current_image_array is None:
                messagebox.showwarning("No Image", "Load an image before exporting metrics.")
                return

            arr = np.asarray(self.current_image_array, dtype=float)
            metrics = {
                "image": self.current_image_title or "unknown",
                "mean_intensity": float(np.mean(arr)),
                "std_intensity": float(np.std(arr)),
                "min_intensity": float(np.min(arr)),
                "max_intensity": float(np.max(arr)),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            filename = f"roi_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
                writer.writeheader()
                writer.writerow(metrics)

            self.status_var.set(f"Metrics exported to {filename}")
            messagebox.showinfo("Export Complete", f"Metrics saved:\n{os.path.abspath(filename)}")

        def _display_image(self, array, title):
            arr = np.asarray(array)
            if arr.ndim > 2:
                arr = arr[arr.shape[0] // 2]

            self.display_ax.clear()
            self.display_ax.imshow(arr, cmap="gray")
            self.display_ax.set_title(title)
            self.display_ax.axis("off")
            self.display_figure.tight_layout()
            self.display_canvas.draw()
            self.image_title.configure(text=f"Current image: {title}")
else:
    class PETCTManualROIApp:
        def __init__(self):
            raise RuntimeError("tkinter is required to run the GUI workstation.")


# ==============================
# ML / ROC Analysis Module
# ==============================
def run_breast_density_ml_analysis(csv_path="breast_density_data.csv"):
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV not found: {csv_path}. Skipping ML ROC analysis.")
        return {}

    data = pd.read_csv(csv_path)
    if "diagnosis" not in data.columns:
        print("[WARN] CSV missing required 'diagnosis' column. Skipping ML ROC analysis.")
        return {}

    X = data.drop("diagnosis", axis=1)
    y = data["diagnosis"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
    }

    results = {}
    plt.figure(figsize=(7, 6))
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        results[name] = roc_auc
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Breast Density ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.show(block=False)

    return results


# ==============================
# Main Entry Point
# ==============================
if __name__ == "__main__":
    threading.Thread(target=run_breast_density_ml_analysis, daemon=True).start()

    if tk is None:
        print("[ERROR] tkinter is not installed. GUI workstation cannot start.")
    else:
        app = PETCTManualROIApp()
        app.mainloop()
