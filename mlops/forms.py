from django import forms
from .models import DatasetUpload


class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = DatasetUpload
        fields = ("name", "csv_file")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for n, f in self.fields.items():
            base = "form-control"
            if isinstance(f.widget, (forms.Select, forms.SelectMultiple)):
                base = "form-select"
            f.widget.attrs["class"] = (f.widget.attrs.get("class", "") + " " + base).strip()

    def clean_csv_file(self):
        f = self.cleaned_data["csv_file"]
        if not f.name.lower().endswith(".csv"):
            raise forms.ValidationError("Please upload a .csv file")
        return f
