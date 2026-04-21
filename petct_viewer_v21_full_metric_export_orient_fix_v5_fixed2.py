#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load the dataset
data = pd.read_csv('breast_density_data.csv')

# Preprocessing
# Assuming the dataset has features and a target column 'diagnosis'
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}

# Training and evaluation
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    results[name] = roc_auc

# Plotting ROC curves
plt.figure()
for name, clf in classifiers.items():
    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label='{} (area = {:.2f})'.format(name, results[name]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.show() 

# ============================================================
# BREAST DENSITY MODULE — integration patch (additive only)
# ============================================================
try:
    from breast_density_module import BreastDensityDialog as _BDD, _install_breast_density_toolbar as _ibd_tb

    _v21_orig_build_for_bd = PETCTManualROIApp._build_ui

    def _build_ui_with_breast_density(self, *a, **k):
        _v21_orig_build_for_bd(self, *a, **k)
        try:
            _ibd_tb(self)
        except Exception as _bd_e:
            print(f"[BreastDensity] toolbar install failed: {_bd_e}")

    PETCTManualROIApp._build_ui = _build_ui_with_breast_density
except ImportError:
    pass  # breast_density_module.py not present — skip silently
