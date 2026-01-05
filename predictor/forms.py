from django import forms


def _bootstrap_form(form: forms.Form) -> forms.Form:
    """Attach Bootstrap classes consistently."""
    for name, f in form.fields.items():
        base = "form-control"
        if isinstance(f.widget, (forms.Select, forms.SelectMultiple)):
            base = "form-select"
        existing = f.widget.attrs.get("class", "")
        f.widget.attrs["class"] = (existing + " " + base).strip()
        f.widget.attrs.setdefault("placeholder", f.label)
    return form


def build_prediction_form(run, post_data=None) -> forms.Form:
    """Build a dynamic prediction form based on the latest TrainingRun schema.

    Supports UCI-like datasets (nice labels + dropdowns) and any future heart disease dataset
    by using the inferred feature metadata stored in TrainingRun.feature_meta_json.
    """

    feature_columns = list(run.feature_columns or [])
    meta = dict(run.feature_meta_json or {})

    # Sort by stored order if present
    feature_columns = sorted(
        feature_columns,
        key=lambda c: int(meta.get(c, {}).get("order", 10_000))
    )

    fields = {}
    for col in feature_columns:
        m = meta.get(col, {})
        ftype = m.get("type", "num")
        label = m.get("label", col)
        help_text = m.get("help", "")

        if ftype == "cat":
            choices = m.get("choices", [])
            # choices can be list of [value,label] or list of dicts
            norm = []
            for ch in choices:
                if isinstance(ch, (list, tuple)) and len(ch) == 2:
                    norm.append((str(ch[0]), str(ch[1])))
                elif isinstance(ch, dict):
                    norm.append((str(ch.get("value")), str(ch.get("label", ch.get("value")))))
                else:
                    norm.append((str(ch), str(ch)))
            fields[col] = forms.ChoiceField(label=label, help_text=help_text, choices=norm)
        else:
            subtype = m.get("subtype", "float")
            min_v = m.get("min", None)
            max_v = m.get("max", None)
            if subtype == "int":
                fields[col] = forms.IntegerField(label=label, help_text=help_text, min_value=min_v, max_value=max_v)
            else:
                fields[col] = forms.FloatField(label=label, help_text=help_text, min_value=min_v, max_value=max_v)

    Dynamic = type("DynamicPredictionForm", (forms.Form,), fields)
    form = Dynamic(post_data)
    return _bootstrap_form(form)
