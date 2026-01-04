from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from mlops.models import TrainingRun


def _latest_run():
    return TrainingRun.objects.filter(status="COMPLETED").order_by("-created_at").first()


def home(request):
    return render(request, "core/home.html")


@login_required
def dashboard(request):
    return render(request, "core/dashboard.html", {"latest_run": _latest_run()})
