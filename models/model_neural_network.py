"""
PyTorch MLP for the `category` target. One of the five comparison models.

Reads data/processed/la_crime_features.csv, reuses the shared train/test split,
takes its hyperparameters from config, and writes plots, metrics, and the saved
model to outputs/neural_network_output/. Same feature set and seed as the others.

The use_cuda flag in config picks an NVIDIA GPU when available and otherwise runs
on CPU (see resolve_device). CUDA is NVIDIA-only and needs a CUDA build of torch.
"""

import sys
import copy
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, log_loss, f1_score
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[ERROR] PyTorch not installed. Run: pip install torch")
    print("        (CPU build is fine; a CUDA build + NVIDIA GPU are only needed for GPU training.)")

# All paths / settings come from src/config.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config as cfg

# =============================================================================
# 0.  REPRODUCIBILITY  &  PATHS
# =============================================================================

MODEL_KEY   = "neural_network"
RANDOM_SEED = cfg.RANDOM_SEED
np.random.seed(RANDOM_SEED)
if HAS_TORCH:
    torch.manual_seed(RANDOM_SEED)

OUT_DIR = cfg.ensure_dir(cfg.MODEL_DIRS[MODEL_KEY])
PARAMS  = cfg.MODEL_PARAMS[MODEL_KEY]
ARCH    = PARAMS["architecture"]
TRAIN   = PARAMS["training"]
TUNE    = PARAMS["tuning"]

sns.set_theme(style="darkgrid", font_scale=1.0)
ACCENT, COOL, WARM, DARK_BG = "#E63946", "#457B9D", "#F4A261", "#1D3557"
CATEGORY_COLORS = {
    "Violent": "#E63946", "Property": "#457B9D",
    "Vehicle": "#F4A261", "Other": "#6C757D",
}

# Feature set shared by all models. weapon_desc / status_desc left out (leakage).
NUMERIC_FEATURE_COLUMNS     = ["lat", "lon", "vict_age", "hour", "month", "day_of_week"]
CATEGORICAL_FEATURE_COLUMNS = ["area_name", "vict_sex", "vict_descent", "premis_group"]
TARGET_COLUMN = cfg.TARGET_COLUMN

PERM_IMPORTANCE_SAMPLE = 2000   # rows used for permutation importance


def resolve_device():
    """Return cuda if use_cuda is set and a CUDA GPU is present, else cpu."""
    want_cuda = bool(TRAIN.get("use_cuda", False))
    if want_cuda and torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  Device       : CUDA ({torch.cuda.get_device_name(0)})")
    else:
        dev = torch.device("cpu")
        if want_cuda:
            print("  Device       : CPU  (use_cuda=True but no CUDA GPU available — "
                  "falling back; expected on AMD/Windows)")
        else:
            print("  Device       : CPU  (use_cuda=False)")
    return dev


# =============================================================================
# 1.  LOAD  &  FEATURE SELECTION
# =============================================================================

def find_features_file() -> Path:
    path = cfg.FEATURES_DATA_PATH
    if not path.exists():
        print(f"\n  [ERROR] Could not find the engineered features file at {path}")
        print(f"          Run data_cleaning.py then feature_engineering.py first.")
        sys.exit(1)
    return path


def load_data() -> pd.DataFrame:
    print(f"\n{'='*70}")
    print("  STEP 1 – LOADING ENGINEERED FEATURES DATASET")
    print(f"{'='*70}")
    path = find_features_file()
    print(f"  File  : {path}")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Shape : {df.shape}")
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*70}")
    print("  STEP 2 – FEATURE SELECTION")
    print(f"{'='*70}")
    print("  Groups: time, location, victim demographics, premise")
    print("  Left out: weapon_desc, status_desc (leak the target)")

    wanted  = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_FEATURE_COLUMNS
    present = [c for c in wanted if c in df.columns]
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        print(f"  [NOTE] Not in features file (skipped): {missing}")
    if TARGET_COLUMN not in df.columns:
        print(f"\n  [ERROR] Target column '{TARGET_COLUMN}' not found in features file.")
        sys.exit(1)

    work = df[[TARGET_COLUMN] + present].copy()
    print(f"  Final feature set ({len(present)} features + target): {present}")
    return work


# =============================================================================
# 2.  PREPROCESSING  (NN needs scaling -> impute+scale numeric, impute+one-hot cat)
# =============================================================================

def build_preprocessor(work: pd.DataFrame):
    print(f"\n{'='*70}")
    print("  STEP 3 – PRE-PROCESSING  (impute + scale numeric, one-hot categorical)")
    print(f"{'='*70}")

    numeric_features     = [c for c in NUMERIC_FEATURE_COLUMNS if c in work.columns]
    categorical_features = [c for c in CATEGORICAL_FEATURE_COLUMNS if c in work.columns]

    numeric_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),       # NNs train far better on standardized inputs
    ])
    categorical_pipeline = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
    print(f"  Numeric (impute+scale) : {numeric_features}")
    print(f"  Categorical (one-hot)  : {categorical_features}")
    return preprocessor


def _to_dense_float32(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


# =============================================================================
# 3.  SHARED TRAIN / TEST SPLIT  (identical held-out rows across all models)
# =============================================================================

def load_or_create_split(work: pd.DataFrame):
    print(f"\n{'='*70}")
    print("  STEP 4 – TRAIN / TEST SPLIT")
    print(f"{'='*70}")

    X = work.drop(columns=[TARGET_COLUMN])
    y = work[TARGET_COLUMN]
    split_path = cfg.SPLIT_INDICES_PATH

    if split_path.exists():
        split = pd.read_csv(split_path)
        train_idx = [i for i in split.loc[split["split"] == "train", "row_index"] if i in work.index]
        test_idx  = [i for i in split.loc[split["split"] == "test",  "row_index"] if i in work.index]
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test,  y_test  = X.loc[test_idx],  y.loc[test_idx]
        print(f"  Reusing shared split        : {split_path}")
    else:
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, work.index, test_size=cfg.TEST_SIZE,
            random_state=RANDOM_SEED, stratify=y if cfg.STRATIFY else None,
        )
        cfg.ensure_dir(split_path.parent)
        pd.DataFrame({
            "row_index": list(idx_train) + list(idx_test),
            "split": ["train"] * len(idx_train) + ["test"] * len(idx_test),
        }).to_csv(split_path, index=False)
        print(f"  No shared split found — created one and saved -> {split_path}")

    print(f"  Train shape  : {X_train.shape}  |  Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# =============================================================================
# 4.  MODEL  (MLP) + TRAINING
# =============================================================================

class MLP(nn.Module):
    """Feed-forward net: input -> [Linear+ReLU+Dropout]* -> Linear(n_classes) logits.
    CrossEntropyLoss applies the softmax, so the final layer outputs raw logits."""
    def __init__(self, input_dim, hidden_layers, n_classes, dropout):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def predict_proba(model, X_np, device, batch_size=4096):
    """Batched softmax probabilities for a (dense float32) feature matrix."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            xb = torch.from_numpy(X_np[i:i + batch_size]).to(device)
            out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.vstack(out)


def train_model(X_tr, y_tr, X_val, y_val, input_dim, n_classes, hidden_layers,
                dropout, lr, epochs, batch_size, patience, device, label="model"):
    """Train an MLP with early stopping on validation loss. Returns (model, history)."""
    model = MLP(input_dim, hidden_layers, n_classes, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size, shuffle=True,
    )
    Xv = torch.from_numpy(X_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val, best_state, bad = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb); seen += len(xb)
        train_loss = running / seen

        model.eval()
        with torch.no_grad():
            vout = model(Xv)
            val_loss = criterion(vout, yv).item()
            val_acc = (vout.argmax(1) == yv).float().mean().item()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch == 1 or epoch % 5 == 0:
            print(f"    [{label}] epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val - 1e-4:
            best_val, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                print(f"    [{label}] early stopping at epoch {epoch} (no val improvement for {patience}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def run_light_search(X_tr, y_tr, X_val, y_val, input_dim, n_classes, device):
    """Train each learning_rate x hidden_layers combo and keep the best by
    validation macro-F1. Stands in for GridSearchCV, which doesn't fit a raw net."""
    print(f"\n{'='*70}")
    print("  STEP 6 – HYPERPARAMETER TUNING (light manual search)")
    print(f"{'='*70}")
    print(f"  Grid: learning_rate={TUNE['learning_rate']}  hidden_layers={TUNE['hidden_layers']}")

    rows, best = [], None
    for lr in TUNE["learning_rate"]:
        for hl in TUNE["hidden_layers"]:
            tag = f"lr={lr},hl={hl}"
            model, history = train_model(
                X_tr, y_tr, X_val, y_val, input_dim, n_classes,
                hidden_layers=hl, dropout=ARCH["dropout"], lr=lr,
                epochs=TRAIN["epochs"], batch_size=TRAIN["batch_size"],
                patience=TRAIN["early_stopping_patience"], device=device, label=tag,
            )
            val_proba = predict_proba(model, X_val, device)
            val_f1 = f1_score(y_val, val_proba.argmax(1), average="macro", zero_division=0)
            rows.append({"learning_rate": lr, "hidden_layers": str(hl), "val_macro_f1": val_f1})
            print(f"  -> {tag}: val_macro_f1={val_f1:.4f}")
            if best is None or val_f1 > best["val_f1"]:
                best = {"model": model, "history": history, "val_f1": val_f1,
                        "params": {"learning_rate": lr, "hidden_layers": hl}}

    results = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False)
    results.to_csv(OUT_DIR / "search_results.csv", index=False)
    print(f"\n  Best: {best['params']} (val_macro_f1={best['val_f1']:.4f})")
    print(f"  -> saved: {OUT_DIR / 'search_results.csv'}")
    return best["model"], best["params"], best["history"]


# =============================================================================
# 5.  EVALUATION
# =============================================================================

def evaluate_model(model, X_test_np, y_test_enc, class_labels, le, device, label):
    y_proba = predict_proba(model, X_test_np, device)
    y_pred_enc = y_proba.argmax(1)
    y_test = le.inverse_transform(y_test_enc)
    y_pred = le.inverse_transform(y_pred_enc)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    ll = log_loss(y_test_enc, y_proba, labels=np.arange(len(class_labels)))
    y_test_bin = label_binarize(y_test, classes=class_labels)
    try:
        auc_macro = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
    except ValueError:
        auc_macro = np.nan

    print(f"\n  -- {label} Results --")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Macro Precision : {precision:.4f}")
    print(f"  Macro Recall    : {recall:.4f}")
    print(f"  Macro F1        : {f1:.4f}")
    print(f"  Macro ROC-AUC   : {auc_macro:.4f}")
    print(f"  Log Loss        : {ll:.4f}")
    print(f"\n  Per-class report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "label": label, "accuracy": acc, "macro_precision": precision,
        "macro_recall": recall, "macro_f1": f1, "macro_roc_auc": auc_macro,
        "log_loss": ll, "y_test": y_test, "y_pred": y_pred, "y_proba": y_proba,
    }


# =============================================================================
# 6.  VISUALS  (the 7 standard comparison plots + the NN-specific training curve)
# =============================================================================

def _save(fig, filename: str):
    path = OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> saved: {path}")


def plot_confusion_matrix(y_test, y_pred, class_labels, title: str, filename: str):
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={"label": "% of true class"}, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold", color=DARK_BG)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    _save(fig, filename)


def plot_roc_curves(y_test, y_proba, class_labels, filename: str):
    y_test_bin = label_binarize(y_test, classes=class_labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    for i, cls in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        auc_val = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC={auc_val:.3f})",
                color=CATEGORY_COLORS.get(cls, "#999999"), linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — One-vs-Rest per Category (Tuned Neural Network)",
                 fontsize=14, fontweight="bold", color=DARK_BG)
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save(fig, filename)


def plot_permutation_importance(model, X_np, y_enc, feature_names, device,
                                top_n=20, filename="feature_importance.png"):
    """Model-agnostic importance: shuffle each encoded feature and measure the drop in
    macro-F1 on a sample (NNs have no native feature_importances_)."""
    print(f"\n  [PLOT] Permutation feature importance (macro-F1 drop on a sample)")
    n = min(PERM_IMPORTANCE_SAMPLE, len(X_np))
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X_np), size=n, replace=False)
    Xs, ys = X_np[idx].copy(), y_enc[idx]

    base_f1 = f1_score(ys, predict_proba(model, Xs, device).argmax(1),
                       average="macro", zero_division=0)
    drops = np.zeros(Xs.shape[1], dtype=float)
    for j in range(Xs.shape[1]):
        Xp = Xs.copy()
        Xp[:, j] = Xp[rng.permutation(n), j]
        f1_j = f1_score(ys, predict_proba(model, Xp, device).argmax(1),
                        average="macro", zero_division=0)
        drops[j] = base_f1 - f1_j

    imp = pd.Series(drops, index=feature_names).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(imp.index[::-1], imp.values[::-1], color=COOL, edgecolor="white")
    ax.set_title(f"Top {top_n} Features by Permutation Importance\n(macro-F1 drop when shuffled)",
                 fontsize=13, fontweight="bold", color=DARK_BG)
    ax.set_xlabel("Macro-F1 decrease")
    plt.tight_layout()
    _save(fig, filename)


def plot_per_class_metrics(y_test, y_pred, class_labels, filename: str):
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=class_labels, zero_division=0)
    x = np.arange(len(class_labels)); w = 0.26
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w, p, w, label="Precision", color=COOL, edgecolor="white")
    ax.bar(x,     r, w, label="Recall",    color=WARM, edgecolor="white")
    ax.bar(x + w, f1, w, label="F1",        color=ACCENT, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(class_labels, rotation=20)
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1 (Tuned Neural Network)",
                 fontsize=13, fontweight="bold", color=DARK_BG)
    ax.legend()
    plt.tight_layout()
    _save(fig, filename)


def plot_baseline_vs_tuned(baseline, tuned, filename: str):
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_roc_auc"]
    labels  = ["Accuracy", "Macro P", "Macro R", "Macro F1", "ROC-AUC"]
    x = np.arange(len(metrics)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w/2, [baseline[m] for m in metrics], w, label="Baseline", color=COOL, edgecolor="white")
    ax.bar(x + w/2, [tuned[m] for m in metrics],    w, label="Tuned",    color=ACCENT, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Baseline vs Tuned — Neural Network", fontsize=13, fontweight="bold", color=DARK_BG)
    for i, m in enumerate(metrics):
        ax.text(i - w/2, baseline[m] + 0.01, f"{baseline[m]:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, tuned[m] + 0.01,    f"{tuned[m]:.2f}",    ha="center", fontsize=8)
    ax.legend()
    plt.tight_layout()
    _save(fig, filename)


def plot_class_distribution(work: pd.DataFrame, filename: str):
    counts = work[TARGET_COLUMN].value_counts()
    colors = [CATEGORY_COLORS.get(c, "#999999") for c in counts.index]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    ax.set_title("Target Class Distribution", fontsize=13, fontweight="bold", color=DARK_BG)
    ax.set_ylabel("Incidents"); ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    _save(fig, filename)


def plot_training_curve(history, filename: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, history["train_loss"], label="Train loss", color=COOL, linewidth=2)
    ax1.plot(epochs, history["val_loss"],   label="Val loss",   color=ACCENT, linewidth=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-entropy loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_acc"], label="Val accuracy", color=WARM,
             linewidth=2, linestyle="--")
    ax2.set_ylabel("Validation accuracy")
    ax1.set_title("Neural Network Training Curve (tuned model)",
                  fontsize=13, fontweight="bold", color=DARK_BG)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    plt.tight_layout()
    _save(fig, filename)


# =============================================================================
# 7.  COMPARISON SUMMARY
# =============================================================================

def write_comparison(baseline, tuned, best_params):
    print(f"\n{'='*70}")
    print("  STEP 8 – BASELINE vs TUNED COMPARISON")
    print(f"{'='*70}")
    comparison = pd.DataFrame([
        {"model": "Neural Network (Baseline)",
         "accuracy": baseline["accuracy"], "macro_precision": baseline["macro_precision"],
         "macro_recall": baseline["macro_recall"], "macro_f1": baseline["macro_f1"],
         "macro_roc_auc": baseline["macro_roc_auc"], "log_loss": baseline["log_loss"],
         "random_seed": RANDOM_SEED},
        {"model": f"Neural Network (Tuned: {best_params})",
         "accuracy": tuned["accuracy"], "macro_precision": tuned["macro_precision"],
         "macro_recall": tuned["macro_recall"], "macro_f1": tuned["macro_f1"],
         "macro_roc_auc": tuned["macro_roc_auc"], "log_loss": tuned["log_loss"],
         "random_seed": RANDOM_SEED},
    ])
    print(comparison.to_string(index=False))
    improvement = tuned["macro_f1"] - baseline["macro_f1"]
    print(f"\n  Macro-F1 change from tuning: {improvement:+.4f}")

    comparison.to_csv(OUT_DIR / "model_comparison_summary.csv", index=False)

    comparison = comparison.assign(model_key=MODEL_KEY)
    shared_path = cfg.COMPARISON_SUMMARY_PATH
    cfg.ensure_dir(shared_path.parent)
    if shared_path.exists():
        prior = pd.read_csv(shared_path)
        if "model_key" in prior.columns:
            prior = prior[prior["model_key"] != MODEL_KEY]
        combined = pd.concat([prior, comparison], ignore_index=True)
    else:
        combined = comparison
    combined.to_csv(shared_path, index=False)
    print(f"\n  -> saved (this model): {OUT_DIR / 'model_comparison_summary.csv'}")
    print(f"  -> updated (shared)  : {shared_path}")


# =============================================================================
# 8.  MAIN
# =============================================================================

def main():
    if not HAS_TORCH:
        sys.exit(1)

    print(f"\n{'#'*70}")
    print("  NEURAL NETWORK (PyTorch MLP) — LA Crime Category Prediction")
    print(f"  RANDOM SEED USED THROUGHOUT: {RANDOM_SEED}")
    print(f"{'#'*70}")
    device = resolve_device()

    df   = load_data()
    work = select_features(df)
    before = len(work)
    work = work.dropna(subset=[TARGET_COLUMN])
    if before != len(work):
        print(f"\n  Dropped {before - len(work):,} rows with missing target.")

    plot_class_distribution(work, "class_distribution.png")

    le = LabelEncoder().fit(work[TARGET_COLUMN])
    class_labels = list(le.classes_)
    n_classes = len(class_labels)
    print(f"\n  Target classes ({n_classes}): {class_labels}")

    preprocessor = build_preprocessor(work)
    X_train_df, X_test_df, y_train, y_test = load_or_create_split(work)

    # Fit preprocessing on TRAIN only, then transform both -> dense float32.
    X_train_full = _to_dense_float32(preprocessor.fit_transform(X_train_df))
    X_test       = _to_dense_float32(preprocessor.transform(X_test_df))
    y_train_full = le.transform(y_train).astype(np.int64)
    y_test_enc   = le.transform(y_test).astype(np.int64)
    input_dim = X_train_full.shape[1]
    feature_names = [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()]

    # Carve a validation set from TRAIN (for early stopping) — never touches the shared test set.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=TRAIN["val_size"],
        random_state=RANDOM_SEED, stratify=y_train_full,
    )
    print(f"  Train/val for fitting       : {X_tr.shape} / {X_val.shape}  (input_dim={input_dim})")

    # ---- Baseline (config architecture) ----
    print(f"\n{'='*70}")
    print("  STEP 5 – BASELINE MODEL  (config architecture)")
    print(f"{'='*70}")
    baseline_model, _ = train_model(
        X_tr, y_tr, X_val, y_val, input_dim, n_classes,
        hidden_layers=ARCH["hidden_layers"], dropout=ARCH["dropout"],
        lr=TRAIN["learning_rate"], epochs=TRAIN["epochs"],
        batch_size=TRAIN["batch_size"], patience=TRAIN["early_stopping_patience"],
        device=device, label="baseline",
    )
    baseline_results = evaluate_model(baseline_model, X_test, y_test_enc,
                                      class_labels, le, device, "BASELINE")
    plot_confusion_matrix(baseline_results["y_test"], baseline_results["y_pred"],
                          class_labels, "Confusion Matrix — Baseline NN",
                          "confusion_matrix_baseline.png")

    # ---- Tuned (light search) ----
    tuned_model, best_params, tuned_history = run_light_search(
        X_tr, y_tr, X_val, y_val, input_dim, n_classes, device
    )
    tuned_results = evaluate_model(tuned_model, X_test, y_test_enc,
                                   class_labels, le, device, "TUNED")

    # ---- Visuals ----
    plot_confusion_matrix(tuned_results["y_test"], tuned_results["y_pred"],
                          class_labels, "Confusion Matrix — Tuned NN",
                          "confusion_matrix_tuned.png")
    plot_roc_curves(tuned_results["y_test"], tuned_results["y_proba"],
                    class_labels, "roc_curves_tuned.png")
    plot_permutation_importance(tuned_model, X_test, y_test_enc, feature_names, device)
    plot_per_class_metrics(tuned_results["y_test"], tuned_results["y_pred"],
                           class_labels, "precision_recall_f1_by_class.png")
    plot_baseline_vs_tuned(baseline_results, tuned_results, "baseline_vs_tuned_metrics.png")
    plot_training_curve(tuned_history, "training_curve.png")

    # ---- Comparison summary ----
    write_comparison(baseline_results, tuned_results, best_params)

    # ---- Persist the tuned model (state_dict on CPU + everything to rebuild it) ----
    model_path = OUT_DIR / "model_neural_network.joblib"
    joblib.dump({
        "state_dict": {k: v.cpu() for k, v in tuned_model.state_dict().items()},
        "architecture": {"input_dim": input_dim, "hidden_layers": best_params["hidden_layers"],
                         "n_classes": n_classes, "dropout": ARCH["dropout"]},
        "preprocessor": preprocessor,
        "label_encoder": le,
        "feature_names": feature_names,
    }, model_path)
    print(f"\n  Serialized tuned model -> {model_path}")

    print(f"\n{'#'*70}")
    print(f"  DONE. All outputs saved in: {OUT_DIR}")
    print(f"  RANDOM SEED USED: {RANDOM_SEED}  <-- shared across all models")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
