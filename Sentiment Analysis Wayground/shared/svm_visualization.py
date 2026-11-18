"""
Visualisasi SVM
Fungsi: Confusion Matrix, Distribusi Sentimen, PCA, t-SNE, Learning Curve, Metrik
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from typing import List, Optional, Dict
import base64
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

SENTIMENT_COLORS = {
    "Positif": "#22c55e",
    "Netral": "#3b82f6",
    "Negatif": "#ef4444"
}
SENTIMENT_CLASSES = ["Positif", "Netral", "Negatif"]
RANDOM_STATE = 42

def _to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img}"

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    if labels is None:
        labels = SENTIMENT_CLASSES
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    return _to_base64(fig)

def plot_confusion_matrix_normalized(y_true, y_pred, labels=None, title="Confusion Matrix Normalized"):
    if labels is None:
        labels = SENTIMENT_CLASSES
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    return _to_base64(fig)

def plot_sentiment_distribution(dist, title="Distribusi Sentimen"):
    labels = list(dist.keys())
    counts = [dist[l]["count"] for l in labels]
    colors = [SENTIMENT_COLORS.get(l, "#888") for l in labels]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    wedges, texts, autotexts = ax1.pie(counts, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax1.set_title("Pie Chart")
    bars = ax2.bar(labels, counts, color=colors, edgecolor="black")
    ax2.set_ylabel("Jumlah")
    ax2.set_title("Bar Chart")
    for bar, cnt in zip(bars, counts):
        ax2.annotate(f"{cnt}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return _to_base64(fig)

def plot_aspect_distribution(aspect_dist, title="Distribusi per Aspek"):
    aspects = list(aspect_dist.keys())
    x = np.arange(len(aspects))
    w = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, sent in enumerate(["Positif", "Netral", "Negatif"]):
        vals = [aspect_dist.get(a, {}).get(sent, 0) for a in aspects]
        ax.bar(x + i * w, vals, w, label=sent, color=SENTIMENT_COLORS.get(sent, "#888"))
    ax.set_xticks(x + w)
    ax.set_xticklabels(aspects, rotation=15, ha="right")
    ax.legend()
    ax.set_title(title)
    plt.tight_layout()
    return _to_base64(fig)

def plot_pca_2d(X, y, model=None, title="PCA 2D", show_hyperplane=True):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X)
    if model is not None:
        from sklearn.base import clone
        model_2d = clone(model)
        model_2d.fit(X_2d, y)
    else:
        model_2d = None
    fig, ax = plt.subplots(figsize=(12, 10))
    if show_hyperplane and model_2d is not None:
        _plot_decision_boundary(ax, X_2d, model_2d, resolution=150)
    markers = ["o", "s", "^"]
    for i, label in enumerate(SENTIMENT_CLASSES):
        mask = y == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=SENTIMENT_COLORS[label],
                   label=f"{label} (n={mask.sum()})",
                   marker=markers[i], s=60, alpha=0.7, edgecolors="#333", linewidths=0.5)
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _to_base64(fig)

def _plot_decision_boundary(ax, X, model, resolution=150):
    try:
        h = (X.max() - X.min()) / resolution
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z_idx = np.array([SENTIMENT_CLASSES.index(c) if c in SENTIMENT_CLASSES else -1 for c in Z]).reshape(xx.shape)
        if np.any(Z_idx >= 0):
            region_colors = [
                (34/255, 197/255, 94/255, 0.15),
                (59/255, 130/255, 246/255, 0.15),
                (239/255, 68/255, 68/255, 0.15)
            ]
            for idx, sentiment in enumerate(SENTIMENT_CLASSES):
                mask = Z_idx == idx
                if np.any(mask):
                    ax.contourf(xx, yy, mask.astype(float),
                               levels=[0.5, 1.5],
                               colors=[region_colors[idx]],
                               alpha=0.5)
    except Exception as e:
        print(f"Warning: {e}")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(34/255, 197/255, 94/255, 0.3), edgecolor="none", label="Region Positif"),
        Patch(facecolor=(59/255, 130/255, 246/255, 0.3), edgecolor="none", label="Region Netral"),
        Patch(facecolor=(239/255, 68/255, 68/255, 0.3), edgecolor="none", label="Region Negatif"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=8)

def plot_tsne_2d(X, y, model=None, title="t-SNE 2D", perplexity=30.0, n_iter=1000, show_hyperplane=True):
    print(f"Computing t-SNE (perp={perplexity}, iter={n_iter})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter,
                random_state=RANDOM_STATE, init="pca", learning_rate="auto")
    X_2d = tsne.fit_transform(X)
    if model is not None:
        from sklearn.base import clone
        model_2d = clone(model)
        model_2d.fit(X_2d, y)
    else:
        model_2d = None
    fig, ax = plt.subplots(figsize=(12, 10))
    if show_hyperplane and model_2d is not None:
        _plot_decision_boundary(ax, X_2d, model_2d, resolution=150)
    markers = ["o", "s", "^"]
    for i, label in enumerate(SENTIMENT_CLASSES):
        mask = y == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=SENTIMENT_COLORS[label],
                   label=f"{label} (n={mask.sum()})",
                   marker=markers[i], s=60, alpha=0.7,
                   edgecolors="#333", linewidths=0.5)
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_title(f"{title} (Perp={perplexity})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _to_base64(fig)

def plot_learning_curve(estimator, X, y, train_sizes=None, title="Learning Curve"):
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    train_abs, train_scores, val_scores = learning_curve(
        estimator, X, y, train_sizes=train_sizes,
        cv=5, scoring="f1_macro", n_jobs=-1, random_state=42)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(train_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
    ax.fill_between(train_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color="orange")
    ax.plot(train_abs, train_mean, "o-", color="blue", label="Training Score", linewidth=2)
    ax.plot(train_abs, val_mean, "o-", color="orange", label="Cross-Validation Score", linewidth=2)
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("F1-Macro Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _to_base64(fig)

def plot_metrics_comparison(metrics, title="Model Comparison"):
    models = list(metrics.keys())
    x = np.arange(len(models))
    w = 0.25
    metric_names = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    colors = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metric_names):
        vals = [metrics[m].get(metric, 0) for m in models]
        bars = ax.bar(x + i * w, vals, w, label=metric.replace("_", " ").title(),
                      color=colors[i], alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.annotate(f"{val:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title(title)
    plt.tight_layout()
    return _to_base64(fig)

if __name__ == "__main__":
    dist = {"Positif": {"count": 100, "percentage": 50.0},
            "Netral": {"count": 50, "percentage": 25.0},
            "Negatif": {"count": 50, "percentage": 25.0}}
    print("Testing svm_visualization...")
    r = plot_sentiment_distribution(dist)
    print(f"Saved: {len(r)} chars")

# --- Stub exports for backward compat (backed up original signatures ---
def plot_pca_3d(X, y, model=None, title="PCA 3D Visualization", save_path=None):
    return ""  # stub: 3D not supported via base64 web export

def plot_tsne_comparison(X, y, model=None, perplexities=None, save_path=None):
    return ""  # stub

def plot_pca_vs_tsne(X, y, model=None, title="PCA vs t-SNE", save_path=None):
    return ""  # stub

def plot_svm_hyperplane(X, y, model, title="SVM Hyperplane", reduction_method="pca", perplexity=30.0, save_path=None):
    return ""  # stub

def plot_detailed_hyperplane(X, y, model, title="Detailed Hyperplane", save_path=None):
    return ""  # stub

def generate_wordcloud(texts, title="Word Cloud", save_path=None):
    return ""  # stub

def plot_ngram_frequency(ngram_df, title="N-Gram", top_n=20, save_path=None):
    return ""  # stub

def plot_pca_2d_stub(X, y, model=None, title="PCA 2D", show_hyperplane=True, save_path=None):
    return ""  # stub
