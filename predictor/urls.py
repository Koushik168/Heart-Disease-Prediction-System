from django.urls import path
from . import views

urlpatterns = [
    path("", views.predict_view, name="predict"),
    path("history/", views.history, name="predict_history"),
]
