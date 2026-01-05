import os
from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings

from .forms import DatasetUploadForm
from .models import DatasetUpload, TrainingRun
from .services.training import train_all_models

def _is_staff(user):
    return user.is_staff

@login_required
@user_passes_test(_is_staff)
def upload_dataset(request):
    if request.method == "POST":
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            ds = form.save(commit=False)
            ds.uploaded_by = request.user
            ds.save()
            messages.success(request, "Dataset uploaded. Now you can run training.")
            return redirect("ml_train", dataset_id=ds.id)
    else:
        form = DatasetUploadForm()
    datasets = DatasetUpload.objects.order_by("-uploaded_at")[:10]
    return render(request, "mlops/upload_dataset.html", {"form": form, "datasets": datasets})

@login_required
@user_passes_test(_is_staff)
def train(request, dataset_id: int):
    ds = get_object_or_404(DatasetUpload, id=dataset_id)
    if request.method == "POST":
        run = TrainingRun.objects.create(dataset=ds, status="RUNNING")
        try:
            out = train_all_models(
                csv_path=ds.csv_file.path,
                media_root=str(settings.MEDIA_ROOT),
                run_id=run.id,
                selection_metric=request.POST.get("selection_metric", "roc_auc"),
            )
            run.status = "COMPLETED"
            run.best_model_name = out.best_model_name or ""
            run.selection_metric = out.selection_metric
            run.metrics_json = out.metrics
            run.artifacts_dir = out.artifacts_dir
            run.target_column = out.target_column
            run.feature_columns = out.feature_columns
            run.feature_meta_json = out.feature_meta_json
            run.label_mapping_json = out.label_mapping_json
            run.n_rows = out.n_rows
            run.n_features = out.n_features
            run.missing_plot = out.missing_plot
            run.feature_importance_plot = out.feature_importance_plot
            run.correlation_plot = out.correlation_plot
            run.roc_plot = out.roc_plot
            run.confusion_dir = out.confusion_dir
            run.save()
            messages.success(request, f"Training completed. Best model: {run.best_model_name}")
            return redirect("ml_run_detail", run_id=run.id)
        except Exception as e:
            run.status = "FAILED"
            run.save()
            messages.error(request, f"Training failed: {e}")
    return render(request, "mlops/train.html", {"dataset": ds})

@login_required
@user_passes_test(_is_staff)
def run_detail(request, run_id: int):
    run = get_object_or_404(TrainingRun, id=run_id)
    # Determine model ranking
    metric_key = f"cv_{run.selection_metric}"
    ranked = sorted(
        [(name, vals.get(metric_key, -1.0), vals) for name, vals in (run.metrics_json or {}).items()],
        key=lambda x: x[1],
        reverse=True
    )
    return render(request, "mlops/run_detail.html", {
        "run": run,
        "ranked": ranked,
        "metric_key": metric_key,
    })

@login_required
@user_passes_test(_is_staff)
def runs_list(request):
    runs = TrainingRun.objects.select_related("dataset").order_by("-created_at")[:30]
    return render(request, "mlops/runs_list.html", {"runs": runs})
