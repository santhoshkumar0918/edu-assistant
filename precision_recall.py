#!/usr/bin/env python3
"""
🎯 Precision–Recall Curve — TNPSC Difficulty Prediction
✅ Correct PR curve that starts from (Recall=0, Precision≈1)
✅ Includes 3 classes: Easy, Medium, Hard
✅ Adds No Skill baseline, and Average Precision (AP) in legend
✅ Uses sklearn LogisticRegression on synthetic data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score

# ==============================
# 1️⃣ Generate Synthetic Data
# ==============================
X, y = make_classification(
    n_samples=1500,
    n_features=15,
    n_informative=10,
    n_redundant=3,
    n_classes=3,
    n_clusters_per_class=1,
    random_state=42
)

# Label names
classes = ["Easy", "Medium", "Hard"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.35, stratify=y, random_state=42
)

# ==============================
# 2️⃣ Train Model (One-vs-Rest)
# ==============================
model = LogisticRegression(max_iter=3000, solver="lbfgs", multi_class="ovr")
model.fit(X_train, y_train)
y_score = model.predict_proba(X_test)  # shape = (n_samples, n_classes)

# ==============================
# 3️⃣ Prepare Labels (Binarize)
# ==============================
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

# ==============================
# 4️⃣ Plot Precision–Recall Curves
# ==============================
plt.figure(figsize=(9, 7))

# Compute global no-skill baseline (mean positive rate)
no_skill = np.mean(y_test_bin)
plt.plot([0, 1], [no_skill, no_skill], linestyle="--", color="gray", label=f"No Skill (baseline = {no_skill:.3f})")

# Custom colors
colors = ["#E74C3C", "#3498DB", "#9B59B6"]

# For each class
for i, cls in enumerate(classes):
    y_true_i = y_test_bin[:, i]
    y_score_i = y_score[:, i]

    # Skip if no positives (safety)
    if np.sum(y_true_i) == 0:
        continue

    precision, recall, _ = precision_recall_curve(y_true_i, y_score_i)
    ap = average_precision_score(y_true_i, y_score_i)

    # Ensure curve starts at (0,1)
    if recall[0] != 0 or precision[0] != 1.0:
        recall = np.concatenate(([0.0], recall))
        precision = np.concatenate(([1.0], precision))

    plt.plot(
        recall,
        precision,
        color=colors[i % len(colors)],
        lw=2.5,
        marker="o",
        markevery=max(1, len(recall)//12),
        label=f"{cls} (AP = {ap:.3f})"
    )

# ==============================
# 5️⃣ Styling
# ==============================
plt.title("Precision–Recall Curve — TNPSC Difficulty Prediction", fontsize=15, fontweight="bold")
plt.xlabel("Recall", fontsize=13)
plt.ylabel("Precision", fontsize=13)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="lower left", fontsize=10, frameon=True)
plt.tight_layout()

# Add annotation
plt.text(0.98, 0.98, "Higher Precision → Better", transform=plt.gca().transAxes,
         fontsize=10, verticalalignment="top", horizontalalignment="right",
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

# ==============================
# 6️⃣ Show and Save
# ==============================
plt.savefig("outputs/tnpsc_precision_recall_curve.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n✅ Precision–Recall curve generated successfully!")
print("📁 Saved at: outputs/tnpsc_precision_recall_curve.png")
