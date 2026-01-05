import os
import re
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay
)
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# Default UCI Heart order & labels (used when these columns exist)
UCI_ORDER = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"
]
UCI_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type (cp)",
    "trestbps": "Resting Blood Pressure (trestbps)",
    "chol": "Serum Cholesterol (chol)",
    "fbs": "Fasting Blood Sugar > 120 mg/dl (fbs)",
    "restecg": "Resting ECG (restecg)",
    "thalach": "Max Heart Rate Achieved (thalach)",
    "exang": "Exercise Induced Angina (exang)",
    "oldpeak": "ST Depression (oldpeak)",
    "slope": "Slope (slope)",
    "ca": "Number of Major Vessels (ca)",
    "thal": "Thal (thal)",
}
UCI_CHOICES = {
    "sex": [(0, "Female"), (1, "Male")],
    "cp": [(0, "Typical angina"), (1, "Atypical angina"), (2, "Non-anginal pain"), (3, "Asymptomatic")],
    "fbs": [(0, "No"), (1, "Yes")],
    "restecg": [(0, "Normal"), (1, "ST-T abnormality"), (2, "LV hypertrophy")],
    "exang": [(0, "No"), (1, "Yes")],
    "slope": [(0, "Upsloping"), (1, "Flat"), (2, "Downsloping")],
    "ca": [(0, "0"), (1, "1"), (2, "2"), (3, "3"), (4, "4")],
    "thal": [(0, "Unknown"), (1, "Normal"), (2, "Fixed defect"), (3, "Reversible defect")],
}

COMMON_TARGET_NAMES = [
    "target","output","label","class","diagnosis","heartdisease","heart_disease","disease",
    "cardio","num","tenyearchd","chd","condition","heartdiseaseorattack"
]


def _ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def _safe_slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower() or "model"


def _to_numeric_if_possible(s: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(s)
    except Exception:
        return s


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # remove unnamed index columns
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    # standardize missing markers
    df.replace(["?", "NA", "N/A", "na", "n/a", "null", "NULL", "None", "none", "-"], np.nan, inplace=True)
    return df


def _infer_target(df: pd.DataFrame) -> Tuple[str, pd.Series, Dict[str, Any]]:
    """Infer a binary target column. Returns (target_col, y_binary, mapping_json)."""
    cols = [c for c in df.columns]
    lower_map = {str(c).lower(): c for c in cols}

    # 1) name-based match
    for key in COMMON_TARGET_NAMES:
        if key in lower_map:
            target = lower_map[key]
            break
    else:
        # 2) contains match
        target = None
        for c in cols:
            lc = str(c).lower()
            if any(k in lc for k in ["target", "label", "class", "disease", "diagnos", "chd", "heart"]):
                target = c
                break

    # 3) fallback: last column if it looks like label
    if target is None and cols:
        target = cols[-1]

    y_raw = df[target].copy() if target in df.columns else None
    if y_raw is None:
        raise ValueError("Could not identify a target column.")

    # try to make it numeric
    y_num = _to_numeric_if_possible(y_raw)

    # Map to binary
    mapping: Dict[str, Any] = {"original_target": str(target)}

    # If values are 0..4 like UCI (num), convert >0 to 1
    if pd.api.types.is_numeric_dtype(y_num):
        uniq = pd.Series(y_num.dropna().unique()).sort_values().tolist()
        mapping["unique_values"] = uniq
        if set(uniq).issubset({0, 1}):
            y_bin = y_num.astype(int)
            mapping["rule"] = "binary_as_is"
        elif all(isinstance(v, (int, float, np.integer, np.floating)) for v in uniq) and len(uniq) <= 6:
            # treat >0 as disease present
            y_bin = (y_num.fillna(0) > 0).astype(int)
            mapping["rule"] = "numeric_gt0_is_positive"
        else:
            # If numeric but not small cardinality, try median split (last resort)
            med = float(pd.Series(y_num.dropna()).median())
            y_bin = (y_num.fillna(med) > med).astype(int)
            mapping["rule"] = "median_split_fallback"
            mapping["median"] = med
    else:
        # String categories -> map common positives
        y_str = y_raw.astype(str).str.strip().str.lower()
        positives = {"1", "true", "yes", "y", "positive", "disease", "present"}
        y_bin = y_str.apply(lambda v: 1 if v in positives else 0).astype(int)
        mapping["rule"] = "string_positive_set"
        mapping["positives"] = sorted(list(positives))

    mapping["positive_label"] = 1
    mapping["negative_label"] = 0
    return target, y_bin, mapping


def _infer_features(df: pd.DataFrame, target_col: str) -> List[str]:
    cols = [c for c in df.columns if c != target_col]
    drop_like = {"id", "patient", "name", "date", "time"}
    out = []
    for c in cols:
        lc = str(c).lower()
        if lc in drop_like or lc.endswith("id"):
            continue
        out.append(c)
    # Prefer UCI order if possible
    if all(f in out for f in UCI_ORDER):
        ordered = [f for f in UCI_ORDER if f in out] + [c for c in out if c not in UCI_ORDER]
        return ordered
    return out


def _infer_types_and_meta(df: pd.DataFrame, features: List[str]) -> Tuple[List[str], List[str], Dict[str, Any]]:
    numeric: List[str] = []
    categorical: List[str] = []
    meta: Dict[str, Any] = {}

    for idx, col in enumerate(features):
        s = df[col]
        s_num = _to_numeric_if_possible(s)

        # Heuristic: treat as categorical if few uniques or object dtype
        nunique = int(s_num.dropna().nunique())
        is_obj = pd.api.types.is_object_dtype(s) or pd.api.types.is_bool_dtype(s)
        is_cat = is_obj or nunique <= 10

        if is_cat:
            categorical.append(col)
            # choices for UI
            vals = pd.Series(s.fillna("Missing").astype(str).unique()).tolist()
            # If UCI known column, override
            if str(col) in UCI_CHOICES:
                choices = [(str(v), lbl) for v, lbl in UCI_CHOICES[str(col)]]
            else:
                # Keep stable ordering
                vals_sorted = sorted(vals)
                choices = [(v, v) for v in vals_sorted]
            meta[str(col)] = {
                "type": "cat",
                "label": UCI_LABELS.get(str(col), str(col).replace("_", " ").title()),
                "choices": choices,
                "order": idx,
            }
        else:
            numeric.append(col)
            s_num = pd.to_numeric(s, errors="coerce")
            meta[str(col)] = {
                "type": "num",
                "label": UCI_LABELS.get(str(col), str(col).replace("_", " ").title()),
                "min": float(np.nanmin(s_num.values)) if np.isfinite(np.nanmin(s_num.values)) else None,
                "max": float(np.nanmax(s_num.values)) if np.isfinite(np.nanmax(s_num.values)) else None,
                "subtype": "int" if pd.api.types.is_integer_dtype(s_num.dropna()) else "float",
                "order": idx,
            }

    return numeric, categorical, meta


def _plot_missing(df: pd.DataFrame, outpath: str) -> None:
    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0]
    plt.figure(figsize=(10, 4))
    if len(miss) == 0:
        plt.text(0.5, 0.5, "No missing values detected", ha="center", va="center")
        plt.axis("off")
    else:
        plt.bar(range(len(miss.index)), miss.values)
        plt.xticks(range(len(miss.index)), miss.index, rotation=75, ha="right")
        plt.ylabel("Missing count")
        plt.title("Missing / Null Values per Column")
        plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def _plot_corr(df: pd.DataFrame, outpath: str) -> None:
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    if corr.shape[0] == 0:
        plt.text(0.5, 0.5, "No numeric columns for correlation", ha="center", va="center")
        plt.axis("off")
    else:
        plt.imshow(corr, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Correlation (Numeric Columns)")
        plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def _save_confusion(cm: np.ndarray, labels: List[str], outpath: str, title: str) -> None:
    plt.figure(figsize=(4.6, 4.1))
    plt.imshow(cm, aspect="auto")
    plt.title(title)
    plt.colorbar()
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(int(v)), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


@dataclass
class TrainOutputs:
    best_model_name: str
    selection_metric: str
    metrics: Dict[str, Dict[str, float]]
    artifacts_dir: str
    missing_plot: str
    correlation_plot: str
    roc_plot: str
    confusion_dir: str
    feature_importance_plot: str
    target_column: str
    feature_columns: List[str]
    feature_meta_json: Dict[str, Any]
    label_mapping_json: Dict[str, Any]
    n_rows: int
    n_features: int


def _onehot_encoder():
    # sklearn 1.4 uses sparse_output
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def train_all_models(
    csv_path: str,
    media_root: str,
    run_id: int,
    selection_metric: str = "roc_auc",
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainOutputs:
    df = pd.read_csv(csv_path)
    df = _clean_dataframe(df)

    target_col, y, label_mapping = _infer_target(df)
    features = _infer_features(df, target_col)

    if len(features) < 2:
        raise ValueError("Not enough feature columns detected (need at least 2).")

    X = df[features].copy()

    numeric, categorical, feature_meta = _infer_types_and_meta(df, features)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _onehot_encoder()),
            ]), categorical),
        ],
        remainder="drop"
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=500, random_state=random_state),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    artifacts_dir_abs = os.path.join(media_root, "artifacts", f"run_{run_id}")
    plots_dir_abs = os.path.join(media_root, "plots", f"run_{run_id}")
    confusion_dir_abs = os.path.join(plots_dir_abs, "confusion")
    _ensure_dirs(artifacts_dir_abs, plots_dir_abs, confusion_dir_abs)

    # EDA plots
    missing_plot_abs = os.path.join(plots_dir_abs, "missing_values.png")
    _plot_missing(df[features + [target_col]], missing_plot_abs)

    corr_plot_abs = os.path.join(plots_dir_abs, "correlation.png")
    _plot_corr(df[features + [target_col]], corr_plot_abs)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    metrics: Dict[str, Dict[str, float]] = {}

    # ROC plot
    plt.figure(figsize=(7, 6))
    roc_ax = plt.gca()

    best_name = None
    best_value = -1.0

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])

        cv = cross_validate(pipe, X_train, y_train, cv=skf, scoring=scoring, n_jobs=None)

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        y_proba = None
        if hasattr(pipe, "predict_proba"):
            try:
                y_proba = pipe.predict_proba(X_test)[:, 1]
            except Exception:
                y_proba = None

        hold_acc = accuracy_score(y_test, y_pred)
        hold_prec = precision_score(y_test, y_pred, zero_division=0)
        hold_rec = recall_score(y_test, y_pred, zero_division=0)
        hold_f1 = f1_score(y_test, y_pred, zero_division=0)
        hold_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cm_path_abs = os.path.join(confusion_dir_abs, f"{_safe_slug(name)}_cm.png")
        _save_confusion(cm, ["No", "Yes"], cm_path_abs, f"Confusion Matrix — {name}")

        try:
            RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=roc_ax, name=name)
        except Exception:
            pass

        model_path_abs = os.path.join(artifacts_dir_abs, f"{_safe_slug(name)}.joblib")
        joblib.dump(pipe, model_path_abs)

        metrics[name] = {
            "cv_accuracy": float(np.mean(cv["test_accuracy"])),
            "cv_precision": float(np.mean(cv["test_precision"])),
            "cv_recall": float(np.mean(cv["test_recall"])),
            "cv_f1": float(np.mean(cv["test_f1"])),
            "cv_roc_auc": float(np.mean(cv["test_roc_auc"])),
            "test_accuracy": float(hold_acc),
            "test_precision": float(hold_prec),
            "test_recall": float(hold_rec),
            "test_f1": float(hold_f1),
            "test_roc_auc": float(hold_auc),
            "artifact": os.path.basename(model_path_abs),
            "confusion_plot": os.path.relpath(cm_path_abs, media_root).replace("\\", "/"),
        }

        sel_val = metrics[name].get(f"cv_{selection_metric}", float("nan"))
        if not np.isnan(sel_val) and sel_val > best_value:
            best_value = float(sel_val)
            best_name = name

    roc_plot_abs = os.path.join(plots_dir_abs, "roc_curves.png")
    plt.title("ROC Curves (Holdout Test)")
    plt.tight_layout()
    plt.savefig(roc_plot_abs, dpi=180)
    plt.close()

    # Feature importance (permutation importance) for the best model
    fi_plot_abs = os.path.join(plots_dir_abs, "feature_importance.png")
    if best_name:
        try:
            best_pipe = joblib.load(os.path.join(artifacts_dir_abs, f"{_safe_slug(best_name)}.joblib"))
            r = permutation_importance(best_pipe, X_test, y_test, n_repeats=10, random_state=random_state)
            importances = pd.Series(r.importances_mean, index=list(X_test.columns)).sort_values(ascending=False)
            plt.figure(figsize=(10, 4.5))
            top = importances.head(15)
            plt.bar(range(len(top.index)), top.values)
            plt.xticks(range(len(top.index)), top.index, rotation=65, ha="right")
            plt.ylabel("Mean importance")
            plt.title(f"Permutation Feature Importance — {best_name}")
            plt.tight_layout()
            plt.savefig(fi_plot_abs, dpi=180)
            plt.close()
        except Exception:
            # fallback blank
            plt.figure(figsize=(10, 4))
            plt.text(0.5, 0.5, "Feature importance not available", ha="center", va="center")
            plt.axis("off")
            plt.savefig(fi_plot_abs, dpi=180)
            plt.close()

    # Save best metadata used by predictor
    best_info = {
        "best_model_name": best_name,
        "selection_metric": selection_metric,
        "best_cv_value": best_value,
        "features": [str(c) for c in features],
        "target": str(target_col),
        "feature_meta": feature_meta,
        "label_mapping": label_mapping,
    }
    with open(os.path.join(artifacts_dir_abs, "best_model.json"), "w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=2)

    # Return outputs with paths relative to MEDIA_ROOT
    return TrainOutputs(
        best_model_name=best_name or "",
        selection_metric=selection_metric,
        metrics=metrics,
        artifacts_dir=os.path.relpath(artifacts_dir_abs, media_root).replace("\\", "/"),
        missing_plot=os.path.relpath(missing_plot_abs, media_root).replace("\\", "/"),
        correlation_plot=os.path.relpath(corr_plot_abs, media_root).replace("\\", "/"),
        roc_plot=os.path.relpath(roc_plot_abs, media_root).replace("\\", "/"),
        confusion_dir=os.path.relpath(confusion_dir_abs, media_root).replace("\\", "/"),
        feature_importance_plot=os.path.relpath(fi_plot_abs, media_root).replace("\\", "/"),
        target_column=str(target_col),
        feature_columns=[str(c) for c in features],
        feature_meta_json=feature_meta,
        label_mapping_json=label_mapping,
        n_rows=int(df.shape[0]),
        n_features=int(len(features)),
    )
