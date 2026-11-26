"""The single page of the website."""
from django.shortcuts import render

from . import predictor
from .forms import FlowForm


def index(request):
    """GET: show the form, pre-filled with a real test flow (?example=<category>).
    POST: run the model on the submitted flow and show the verdict above the form.
    The chosen example travels with both requests so the example menu stays in sync.
    """
    examples = predictor.examples()
    params = request.POST if request.method == "POST" else request.GET
    example = params.get("example") if params.get("example") in examples else "Normal"
    result = None
    if request.method == "POST":
        form = FlowForm(request.POST)
        if form.is_valid():
            result = predictor.predict(form.cleaned_data)
    else:
        form = FlowForm(initial=examples[example])
    return render(request, "detector/index.html", {"form": form, "result": result, "examples": examples,
                                                   "example": example, "just_loaded": request.method == "GET"})
