import os
import json
import pandas as pd
import joblib
from django.conf import settings
from mlops.models import TrainingRun


def _latest_completed_run():
    return TrainingRun.objects.filter(status="COMPLETED").order_by("-created_at").first()


def predict_all_models(input_dict: dict):
    run = _latest_completed_run()
    if not run:
        raise RuntimeError("No trained model found. Ask admin to upload dataset and run training first.")

    artifacts_abs = os.path.join(settings.MEDIA_ROOT, run.artifacts_dir)
    best_meta = os.path.join(artifacts_abs, "best_model.json")
    with open(best_meta, "r", encoding="utf-8") as f:
        best_info = json.load(f)

    features = list(best_info.get("features") or run.feature_columns or [])
    if not features:
        raise RuntimeError("Training run does not have saved feature schema.")

    # Ensure DataFrame has correct column order
    row = {k: input_dict.get(k, None) for k in features}
    X = pd.DataFrame([row], columns=features)

    results = {}
    for model_name, m in (run.metrics_json or {}).items():
        artifact_file = m.get("artifact")
        if not artifact_file:
            continue
        path = os.path.join(artifacts_abs, artifact_file)
        if not os.path.exists(path):
            continue
        pipe = joblib.load(path)
        pred = int(pipe.predict(X)[0])
        proba = None
        if hasattr(pipe, "predict_proba"):
            try:
                proba = float(pipe.predict_proba(X)[0, 1])
            except Exception:
                proba = None
        results[model_name] = {
            "prediction": pred,
            "probability": proba,
            "cv_metric": float(m.get(f"cv_{run.selection_metric}", 0.0)),
        }

    best_model_name = best_info.get("best_model_name") or run.best_model_name
    best = results.get(best_model_name, None)
    return run, best_model_name, best, results
