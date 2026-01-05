from django.db import models
from django.contrib.auth.models import User


class DatasetUpload(models.Model):
    """A raw CSV upload from an admin/staff user."""

    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    name = models.CharField(max_length=200, default="Heart Disease Dataset")
    csv_file = models.FileField(upload_to="datasets/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.uploaded_at:%Y-%m-%d %H:%M})"


class TrainingRun(models.Model):
    """A single end-to-end training run created from an uploaded dataset.

    Stores evaluation metrics + an inferred schema to build the customer input form,
    so the app can support future heart-disease datasets too.
    """

    dataset = models.ForeignKey(DatasetUpload, on_delete=models.CASCADE, related_name="runs")
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=30, default="COMPLETED")  # sync training for simplicity

    # Model selection & evaluation
    best_model_name = models.CharField(max_length=100, blank=True, default="")
    selection_metric = models.CharField(max_length=50, default="roc_auc")
    metrics_json = models.JSONField(default=dict)  # per-model metrics + artifact filenames
    artifacts_dir = models.CharField(max_length=300, blank=True, default="")

    # Inferred schema (flexible datasets)
    target_column = models.CharField(max_length=200, blank=True, default="")
    feature_columns = models.JSONField(default=list)  # ordered list
    feature_meta_json = models.JSONField(default=dict)  # {feature: {type,label,choices,min,max,order,...}}
    label_mapping_json = models.JSONField(default=dict)  # target mapping + positive label info
    n_rows = models.IntegerField(default=0)
    n_features = models.IntegerField(default=0)

    # Plot paths (relative to MEDIA_ROOT)
    missing_plot = models.CharField(max_length=300, blank=True, default="")
    correlation_plot = models.CharField(max_length=300, blank=True, default="")
    roc_plot = models.CharField(max_length=300, blank=True, default="")
    confusion_dir = models.CharField(max_length=300, blank=True, default="")
    feature_importance_plot = models.CharField(max_length=300, blank=True, default="")

    def __str__(self):
        return f"Run {self.id} — {self.best_model_name} — {self.created_at:%Y-%m-%d %H:%M}"


class PredictionLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    input_json = models.JSONField(default=dict)
    results_json = models.JSONField(default=dict)  # per-model output
    best_model_name = models.CharField(max_length=100, blank=True, default="")
    best_probability = models.FloatField(null=True, blank=True)
    predicted_label = models.IntegerField(null=True, blank=True)  # 0/1

    # Link prediction to a specific trained run (optional but useful for traceability)
    run = models.ForeignKey(TrainingRun, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return f"Prediction {self.id} — {self.user.username} — {self.created_at:%Y-%m-%d %H:%M}"
