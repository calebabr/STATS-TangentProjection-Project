# SPD Geometry & Manifold Learning — KTH-TIPS2 Project Guide

## Overview

Compare three SPD geometries (AI, LogE, BW) across three tasks:
1. Manifold UMAP vs Tangent-Space UMAP
2. Raw vs Standardized tangent features
3. Classification with PCA/UMAP dimension reduction

**Dataset:** 4,752 images → 4,752 symmetric positive definite (SPD) matrices of shape (22, 22).  
**Classes:** 11 texture materials (aluminium_foil, brown_bread, corduroy, cork, cotton, cracker, lettuce_leaf, linen, white_bread, wood, wool).  
**Tangent feature size:** 22×23/2 = **253** features after half-vectorization.

---

## Directory Layout

```
Archive (1)/
├── metric_functions.py       # All SPD geometry functions
├── guide.txt                 # Original task guide
├── KTH_TIPS2_dataset_description.txt
└── data/KTH-TIPS2-b/
    ├── covariance_matrices.npz   # X: (4752,22,22), y: (4752,) labels
    ├── metadata.csv              # material, sample, scale, image_label, relative_path
    └── Untitled-1.ipynb          # Working notebook (put code here)
```

---

## Environment Setup

```python
# Standard installs needed (if not already present):
# pip install umap-learn scikit-learn scipy numpy pandas matplotlib seaborn

import sys
import os

# Add parent directories so metric_functions.py is importable from the notebook
# (notebook lives in data/KTH-TIPS2-b/, metric_functions.py lives in Archive (1)/)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath("__file__")), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.manifold import trustworthiness
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from scipy.stats import spearmanr
import umap

from metric_functions import (
    stack_npp_to_ppn,
    compute_mean_projection,
    project_stack_to_tangent_features,
    compute_pairwise_distance_matrix,
)
```

> **Note on `ai_projection_mean_stable`:** `compute_mean_projection` internally calls
> `ai_projection_mean_stable` for `metric="ai"`, but only `ai_projection_mean` is defined
> in the file. If you get a `NameError`, either alias it at the top of `metric_functions.py`:
> ```python
> ai_projection_mean_stable = ai_projection_mean
> ```
> or patch it in your notebook before importing:
> ```python
> import metric_functions as mf
> mf.ai_projection_mean_stable = mf.ai_projection_mean
> ```

---

## Data Loading

```python
DATA_DIR = "."  # notebook is already inside data/KTH-TIPS2-b/

data = np.load(os.path.join(DATA_DIR, "covariance_matrices.npz"), allow_pickle=True)
X = data["X"]       # shape: (4752, 22, 22)  — SPD matrices
y = data["y"]       # shape: (4752,)          — string material labels

meta = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))

print(f"X shape: {X.shape}")
print(f"Classes: {np.unique(y)}")
print(f"Samples per class: {pd.Series(y).value_counts().to_dict()}")
```

### Recommended Subsets (start small, scale up)

```python
# --- Subset 1: Smallest debug set (396 matrices) ---
idx_small = (meta["sample"] == "sample_a") & (meta["scale"].isin([2, 6, 10]))
X_small = X[idx_small.values]
y_small = y[idx_small.values]

# --- Subset 2: One sample (432 matrices) ---
idx_a = meta["sample"] == "sample_a"
X_a = X[idx_a.values]
y_a = y[idx_a.values]

# --- Subset 3: Two samples (864 matrices) ---
idx_ab = meta["sample"].isin(["sample_a", "sample_b"])
X_ab = X[idx_ab.values]
y_ab = y[idx_ab.values]

# --- Full dataset (4752 matrices) ---
X_full, y_full = X, y
```

> **Performance tip:** Pairwise distance matrices are O(n²). For n=396, that's ~78k pairs.
> For n=4752 that's ~11M pairs — very slow. Always prototype on `X_small` first.

---

## Key Function Reference

### Stack conversion (required by mean functions)

```python
X_ppn = stack_npp_to_ppn(X_subset)   # (n,22,22) → (22,22,n)
```

### Computing the barycenter (mean) on the manifold

```python
M_bw,   info_bw   = compute_mean_projection(X_ppn, metric="bw")
M_ai,   info_ai   = compute_mean_projection(X_ppn, metric="ai")
M_loge, info_loge = compute_mean_projection(X_ppn, metric="loge")
# info dict: {"metric": str, "iters": int, "final": float}
```

### Tangent-space projection (half-vectorized, shape (n, 253))

```python
Z_bw   = project_stack_to_tangent_features(X_subset, M_bw,   metric="bw")
Z_ai   = project_stack_to_tangent_features(X_subset, M_ai,   metric="ai")
Z_loge = project_stack_to_tangent_features(X_subset, M_loge, metric="loge")
```

### Pairwise manifold distance matrix (for manifold UMAP)

```python
D_bw   = compute_pairwise_distance_matrix(X_subset, metric="bw",   verbose=True)
D_ai   = compute_pairwise_distance_matrix(X_subset, metric="ai",   verbose=True)
D_loge = compute_pairwise_distance_matrix(X_subset, metric="loge", verbose=True)
```

---

## Task 1 — Manifold UMAP vs Tangent-Space UMAP

**Question:** Does tangent projection preserve the same structure as the original manifold distances?

### Step 1: Compute for each geometry

```python
METRICS = ["bw", "ai", "loge"]
UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.1, random_state=42)

results_t1 = {}  # keyed by metric

for metric in METRICS:
    print(f"\n=== {metric.upper()} ===")

    # -- Manifold UMAP --
    D = compute_pairwise_distance_matrix(X_small, metric=metric)
    emb_manifold = umap.UMAP(metric="precomputed", **UMAP_PARAMS).fit_transform(D)

    # -- Tangent-Space UMAP --
    X_ppn = stack_npp_to_ppn(X_small)
    M, _ = compute_mean_projection(X_ppn, metric=metric)
    Z = project_stack_to_tangent_features(X_small, M, metric=metric)
    emb_tangent = umap.UMAP(metric="euclidean", **UMAP_PARAMS).fit_transform(Z)

    results_t1[metric] = {
        "D": D, "Z": Z,
        "emb_manifold": emb_manifold,
        "emb_tangent": emb_tangent,
    }
```

### Step 2: Plot (3 geometries × 2 methods = 6 panels)

```python
fig, axes = plt.subplots(3, 2, figsize=(12, 15))
label_set = np.unique(y_small)
colors = plt.cm.tab20(np.linspace(0, 1, len(label_set)))
label_to_color = dict(zip(label_set, colors))

for row, metric in enumerate(METRICS):
    for col, (key, title) in enumerate([("emb_manifold", "Manifold UMAP"),
                                         ("emb_tangent",  "Tangent-Space UMAP")]):
        ax = axes[row, col]
        emb = results_t1[metric][key]
        for lbl in label_set:
            mask = y_small == lbl
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=[label_to_color[lbl]], label=lbl, s=10, alpha=0.7)
        ax.set_title(f"{metric.upper()} — {title}")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

axes[0, 1].legend(fontsize=6, markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("task1_umap_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Step 3: Compute metrics

```python
def neighborhood_overlap(D_manifold, Z_tangent, k=15):
    """Fraction of k-NN in manifold space that are also k-NN in tangent space."""
    nn_m = NearestNeighbors(n_neighbors=k+1, metric="precomputed").fit(D_manifold)
    nn_t = NearestNeighbors(n_neighbors=k+1, metric="euclidean").fit(Z_tangent)
    _, idx_m = nn_m.kneighbors(D_manifold)
    _, idx_t = nn_t.kneighbors(Z_tangent)
    overlaps = []
    for i in range(len(Z_tangent)):
        s_m = set(idx_m[i, 1:])   # exclude self
        s_t = set(idx_t[i, 1:])
        overlaps.append(len(s_m & s_t) / k)
    return float(np.mean(overlaps))


def spearman_dist_corr(D_manifold, Z_tangent):
    """Spearman correlation between upper-triangle pairwise distances."""
    n = D_manifold.shape[0]
    idx = np.triu_indices(n, k=1)
    d_man = D_manifold[idx]
    # Euclidean distances from tangent features
    from sklearn.metrics import pairwise_distances
    D_tan = pairwise_distances(Z_tangent)
    d_tan = D_tan[idx]
    rho, _ = spearmanr(d_man, d_tan)
    return float(rho)


print("\nTask 1 Metrics:")
print(f"{'Metric':<8}  {'Nbr Overlap':>12}  {'Trustworthiness':>16}  {'Spearman ρ':>11}")
print("-" * 55)
for metric in METRICS:
    D   = results_t1[metric]["D"]
    Z   = results_t1[metric]["Z"]
    no  = neighborhood_overlap(D, Z, k=15)
    tw  = trustworthiness(D, Z, n_neighbors=15, metric="precomputed")
    sp  = spearman_dist_corr(D, Z)
    print(f"{metric.upper():<8}  {no:>12.4f}  {tw:>16.4f}  {sp:>11.4f}")
```

---

## Task 2 — Raw vs Standardized Tangent Features

**Questions:**
1. Do geometries produce differently-scaled features?
2. Does standardization change PCA variance structure?
3. Does standardization affect classification accuracy?

### Question 1: Feature standard deviations

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, metric in zip(axes, METRICS):
    X_ppn = stack_npp_to_ppn(X_small)
    M, _ = compute_mean_projection(X_ppn, metric=metric)
    Z = project_stack_to_tangent_features(X_small, M, metric=metric)
    sds = Z.std(axis=0)
    ax.boxplot(sds, vert=True)
    ax.set_title(f"{metric.upper()} — Feature SDs\n"
                 f"median={np.median(sds):.3f}, IQR=[{np.percentile(sds,25):.3f}, {np.percentile(sds,75):.3f}]")
    ax.set_ylabel("Standard Deviation")
    ax.set_xlabel("(all 253 features)")

plt.tight_layout()
plt.savefig("task2_feature_sds.png", dpi=150)
plt.show()
```

### Question 2: PCA cumulative variance explained

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
q_range = np.arange(1, 254)

for ax, metric in zip(axes, METRICS):
    X_ppn = stack_npp_to_ppn(X_small)
    M, _ = compute_mean_projection(X_ppn, metric=metric)
    Z_raw = project_stack_to_tangent_features(X_small, M, metric=metric)
    Z_std = StandardScaler().fit_transform(Z_raw)

    for Z, label, ls in [(Z_raw, "Raw", "-"), (Z_std, "Standardized", "--")]:
        pca = PCA().fit(Z)
        cve = np.cumsum(pca.explained_variance_ratio_)
        ax.plot(q_range, cve, ls, label=label)

    ax.axhline(0.90, color="gray", linestyle=":", alpha=0.7)
    ax.axhline(0.95, color="gray", linestyle="-.", alpha=0.7)
    ax.set_title(f"{metric.upper()} — Cumulative Variance Explained")
    ax.set_xlabel("Number of components q")
    ax.set_ylabel("CVE(q)")
    ax.legend()
    ax.set_xlim(0, 80)

plt.tight_layout()
plt.savefig("task2_pca_cve.png", dpi=150)
plt.show()
```

### Question 3: Standardization vs accuracy

Use the nested CV from Task 3 (below) and compare `standardize=True` vs `standardize=False`.
See the grouped barplot at the end of Task 3.

---

## Task 3 — Classification Pipelines with Nested CV

**Pipelines per geometry:**
- **A:** Full tangent features → classifier
- **B:** PCA (q components) → classifier
- **C:** UMAP (q components) → classifier

**Outer CV:** 5-fold stratified. **Inner CV:** 3-fold for hyperparameter tuning.

### Core evaluation function

```python
def run_nested_cv(X_mats, y_labels, metric, pipeline="A", q=10,
                  standardize=False, classifier="knn", n_outer=5, random_state=42):
    """
    Full nested cross-validation for one geometry + pipeline combination.

    Parameters
    ----------
    X_mats     : ndarray (n, p, p)
    y_labels   : ndarray (n,)
    metric     : 'bw', 'ai', 'loge'
    pipeline   : 'A' (full), 'B' (PCA), 'C' (UMAP)
    q          : latent dimension for B and C
    standardize: whether to apply StandardScaler after tangent projection
    classifier : 'knn', 'svm', 'lr'
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y_labels)
    n_classes = len(le.classes_)

    skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)

    accs, precs, recs, aucs, runtimes = [], [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_mats, y_enc)):
        t0 = time.time()

        X_train, X_test = X_mats[train_idx], X_mats[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        # 1. Compute barycenter on TRAINING only
        X_train_ppn = stack_npp_to_ppn(X_train)
        M, _ = compute_mean_projection(X_train_ppn, metric=metric)

        # 2. Project both sets using training barycenter
        Z_train = project_stack_to_tangent_features(X_train, M, metric=metric)
        Z_test  = project_stack_to_tangent_features(X_test,  M, metric=metric)

        # 3. Standardize (fit on train only)
        if standardize:
            scaler = StandardScaler()
            Z_train = scaler.fit_transform(Z_train)
            Z_test  = scaler.transform(Z_test)

        # 4. Dimension reduction (fit on train only)
        if pipeline == "B":
            dr = PCA(n_components=q, random_state=random_state)
            Z_train = dr.fit_transform(Z_train)
            Z_test  = dr.transform(Z_test)
        elif pipeline == "C":
            dr = umap.UMAP(n_components=q, metric="euclidean",
                           n_neighbors=15, min_dist=0.1, random_state=random_state)
            Z_train = dr.fit_transform(Z_train)
            Z_test  = dr.transform(Z_test)

        # 5. Build classifier
        if classifier == "knn":
            clf = KNeighborsClassifier(n_neighbors=5)
        elif classifier == "svm":
            clf = SVC(kernel="rbf", C=1.0, probability=True)
        elif classifier == "lr":
            clf = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial")
        else:
            raise ValueError(f"Unknown classifier: {classifier}")

        clf.fit(Z_train, y_train)
        y_pred = clf.predict(Z_test)

        # 6. Metrics
        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, average="macro", zero_division=0))
        recs.append(recall_score(y_test, y_pred, average="macro", zero_division=0))

        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(Z_test)
        else:
            from sklearn.preprocessing import label_binarize
            y_prob = label_binarize(y_pred, classes=np.arange(n_classes))

        try:
            auc_val = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                    average="macro", labels=np.arange(n_classes))
        except ValueError:
            auc_val = float("nan")
        aucs.append(auc_val)
        runtimes.append(time.time() - t0)

    return {
        "accuracy":   np.mean(accs),
        "acc_std":    np.std(accs),
        "precision":  np.mean(precs),
        "recall":     np.mean(recs),
        "auc":        np.nanmean(aucs),
        "runtime_s":  np.sum(runtimes),
    }
```

### Run the full comparison table

```python
# Use X_small / y_small to prototype; swap to X_a or X_ab for final results
X_use, y_use = X_small, y_small

Q_VALUES = [5, 10, 20]
CLASSIFIERS = ["knn"]   # add "svm", "lr" for full comparison

rows = []
for metric in METRICS:
    for standardize in [False, True]:
        for clf_name in CLASSIFIERS:
            # Pipeline A — full features
            r = run_nested_cv(X_use, y_use, metric=metric, pipeline="A",
                              standardize=standardize, classifier=clf_name)
            rows.append({"geometry": metric.upper(), "pipeline": "A-Full",
                         "q": "—", "standardize": standardize,
                         "classifier": clf_name, **r})

            for q in Q_VALUES:
                # Pipeline B — PCA
                r = run_nested_cv(X_use, y_use, metric=metric, pipeline="B", q=q,
                                  standardize=standardize, classifier=clf_name)
                rows.append({"geometry": metric.upper(), "pipeline": "B-PCA",
                             "q": q, "standardize": standardize,
                             "classifier": clf_name, **r})

                # Pipeline C — UMAP
                r = run_nested_cv(X_use, y_use, metric=metric, pipeline="C", q=q,
                                  standardize=standardize, classifier=clf_name)
                rows.append({"geometry": metric.upper(), "pipeline": "C-UMAP",
                             "q": q, "standardize": standardize,
                             "classifier": clf_name, **r})

df_results = pd.DataFrame(rows)
df_results = df_results.sort_values(["geometry", "pipeline", "q"])

# Display
display_cols = ["geometry", "pipeline", "q", "standardize",
                "accuracy", "acc_std", "precision", "recall", "auc", "runtime_s"]
print(df_results[display_cols].to_string(index=False, float_format="{:.4f}".format))
df_results.to_csv("task3_results.csv", index=False)
```

### Task 2 Q3 + Task 3 grouped barplot

```python
# Accuracy by geometry, pipeline, raw vs standardized
df_plot = df_results[df_results["pipeline"] != "A-Full"].copy()
df_plot["label"] = df_plot["pipeline"] + " q=" + df_plot["q"].astype(str)

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, metric in zip(axes, [m.upper() for m in METRICS]):
    sub = df_plot[df_plot["geometry"] == metric]
    x = np.arange(len(sub["label"].unique()))
    labels_ord = sub.groupby("label")["accuracy"].mean().index.tolist()
    raw_acc = [sub[(sub["label"]==l) & (~sub["standardize"])]["accuracy"].values[0]
               if len(sub[(sub["label"]==l) & (~sub["standardize"])]) else 0 for l in labels_ord]
    std_acc = [sub[(sub["label"]==l) & (sub["standardize"])]["accuracy"].values[0]
               if len(sub[(sub["label"]==l) & (sub["standardize"])]) else 0 for l in labels_ord]
    w = 0.35
    ax.bar(x - w/2, raw_acc, w, label="Raw", color="steelblue")
    ax.bar(x + w/2, std_acc, w, label="Standardized", color="darkorange")
    ax.set_xticks(x); ax.set_xticklabels(labels_ord, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"{metric}")
    ax.set_ylabel("Accuracy"); ax.legend()

plt.suptitle("Task 3: Classification Accuracy (Raw vs Standardized)", y=1.02)
plt.tight_layout()
plt.savefig("task3_accuracy_barplot.png", dpi=150, bbox_inches="tight")
plt.show()
```

---

## Common Pitfalls

| Pitfall | Fix |
|---|---|
| Data leakage: computing mean/scaler/PCA on full dataset before CV | Always fit inside each fold on training data only |
| `ai_projection_mean_stable` NameError | Add alias `ai_projection_mean_stable = ai_projection_mean` in `metric_functions.py` |
| Stack shape mismatch | Mean functions expect `(p,p,n)`; use `stack_npp_to_ppn()` |
| OOM / slowness for large n | Prototype on `X_small` (n=396), scale after |
| UMAP `transform` on test set | UMAP's `.transform()` requires `parametric_umap` or UMAP v0.5+; for standard UMAP fit on train, call `.transform(Z_test)` |
| Negative eigenvalues in SPD | Functions call `make_spd(eps=1e-12)` internally, already handled |
| AUC fails for folds with missing classes | Wrap `roc_auc_score` in try/except as shown above |

---

## Scaling Up Checklist

```
1. Debug on X_small  (n=396,  ~78k pairs)        ← get all code working
2. Scale to X_a      (n=432,  ~93k pairs)         ← verify no leakage, plots look right
3. Scale to X_ab     (n=864,  ~373k pairs)        ← final performance numbers
4. Full X_full       (n=4752, ~11M pairs)         ← only if runtime is acceptable
```

For large n, consider caching pairwise distance matrices to `.npy` files:
```python
np.save("D_bw_sample_ab.npy", D_bw)
D_bw = np.load("D_bw_sample_ab.npy")
```

---

## Output Files Summary

| File | Contents |
|---|---|
| `task1_umap_comparison.png` | 3×2 grid of UMAP plots (manifold vs tangent, all geometries) |
| `task2_feature_sds.png` | Boxplots of tangent feature SDs per geometry |
| `task2_pca_cve.png` | PCA cumulative variance explained, raw vs standardized |
| `task3_results.csv` | Full table: accuracy, std, precision, recall, AUC, runtime |
| `task3_accuracy_barplot.png` | Grouped barplot comparing raw vs standardized across pipelines |
