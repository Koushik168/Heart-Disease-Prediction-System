from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.contrib import messages

from .forms import build_prediction_form
from .services.predict import predict_all_models
from mlops.models import PredictionLog, TrainingRun


def _latest_completed_run():
    return TrainingRun.objects.filter(status="COMPLETED").order_by("-created_at").first()


@login_required
def predict_view(request):
    run = _latest_completed_run()
    if not run:
        messages.error(request, "No trained model found yet. Ask the admin to upload a dataset and train models first.")
        return render(request, "predictor/predict.html", {"form": None, "run": None})

    form = build_prediction_form(run, request.POST or None)

    if request.method == "POST" and form.is_valid():
        cleaned = dict(form.cleaned_data)

        # Coerce types based on meta (ChoiceField values come as strings)
        meta = dict(run.feature_meta_json or {})
        for col, val in list(cleaned.items()):
            m = meta.get(col, {})
            if m.get("type") == "cat":
                # Try int -> float -> keep str
                try:
                    cleaned[col] = int(val)
                except Exception:
                    try:
                        cleaned[col] = float(val)
                    except Exception:
                        cleaned[col] = str(val)
            else:
                # numeric already coerced by form
                cleaned[col] = val

        run_obj, best_name, best, results = predict_all_models(cleaned)

        predicted_label = None
        best_probability = None
        if best:
            predicted_label = best.get("prediction")
            best_probability = best.get("probability")

        PredictionLog.objects.create(
            user=request.user,
            run=run_obj,
            input_json=cleaned,
            results_json=results,
            best_model_name=best_name,
            best_probability=best_probability,
            predicted_label=predicted_label,
        )
        best_percent = None
        if best_probability is not None:
            try:
                best_percent = round(float(best_probability) * 100.0, 2)
            except Exception:
                best_percent = None

        return render(request, "predictor/result.html", {
            "run": run_obj,
            "best_model_name": best_name,
            "best": best,
            "best_percent": best_percent,
            "results": results,
            "input": cleaned,
        })

    return render(request, "predictor/predict.html", {"form": form, "run": run})


@login_required
def history(request):
    logs = PredictionLog.objects.filter(user=request.user).select_related("run").order_by("-created_at")[:50]
    return render(request, "predictor/history.html", {"logs": logs})
