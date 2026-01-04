from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import User


def _bootstrap_fields(form):
    for name, f in form.fields.items():
        base = "form-control"
        if isinstance(f.widget, (forms.Select, forms.SelectMultiple)):
            base = "form-select"
        f.widget.attrs["class"] = (f.widget.attrs.get("class", "") + " " + base).strip()
        if not f.widget.attrs.get("placeholder"):
            f.widget.attrs["placeholder"] = f.label
    return form


class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _bootstrap_fields(self)


class LoginForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _bootstrap_fields(self)
