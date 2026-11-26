"""The flow form: one <select> per text feature, one number input per numeric feature."""
from django import forms

from .features import CATEGORICAL, NUMERIC_GROUPS
from .predictor import category_choices


class FlowForm(forms.Form):
    """All 42 features of one network flow. Fields are built from features.py."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        choices = category_choices()
        for name, label in CATEGORICAL.items():
            self.fields[name] = forms.ChoiceField(label=label, choices=[(v, v) for v in choices[name]])
        for _, _, fields in NUMERIC_GROUPS:
            for name, label in fields:
                # every feature of the dataset is a count, a size, a time or a flag: never negative
                self.fields[name] = forms.FloatField(label=label, min_value=0,
                                                     widget=forms.NumberInput(attrs={"step": "any"}))

    def categorical_fields(self):
        """The three <select> fields, for the template."""
        return [self[name] for name in CATEGORICAL]

    def numeric_groups(self):
        """[(title, open?, bound fields)] in page order, for the template."""
        return [(title, is_open, [self[name] for name, _ in fields]) for title, is_open, fields in NUMERIC_GROUPS]
