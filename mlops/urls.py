from django.urls import path
from . import views

urlpatterns = [
    path("upload/", views.upload_dataset, name="ml_upload"),
    path("train/<int:dataset_id>/", views.train, name="ml_train"),
    path("runs/", views.runs_list, name="ml_runs"),
    path("runs/<int:run_id>/", views.run_detail, name="ml_run_detail"),
]
