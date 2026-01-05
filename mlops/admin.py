from django.contrib import admin
from .models import DatasetUpload, TrainingRun, PredictionLog

@admin.register(DatasetUpload)
class DatasetUploadAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "uploaded_by", "uploaded_at")

@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ("id", "dataset", "status", "best_model_name", "selection_metric", "created_at")

@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "best_model_name", "predicted_label", "best_probability", "created_at")
