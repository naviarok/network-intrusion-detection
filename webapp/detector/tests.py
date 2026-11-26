"""Tests of the page: python manage.py test (run from webapp/)."""
from django.test import SimpleTestCase

from . import predictor


class IndexPageTests(SimpleTestCase):
    """The page renders, loads examples, agrees with the model and rejects invalid input."""

    def test_form_has_selects_for_text_features(self):
        page = self.client.get("/").content.decode()
        for name in ("proto", "service", "state"):
            self.assertIn(f'<select name="{name}"', page)

    def test_example_prefills_the_form(self):
        response = self.client.get("/?example=Exploits")
        self.assertEqual(response.context["example"], "Exploits")
        self.assertEqual(response.context["form"].initial, predictor.examples()["Exploits"])

    def test_normal_example_gives_no_alarm(self):
        response = self.client.post("/", predictor.examples()["Normal"])
        self.assertFalse(response.context["result"]["alarm"])
        self.assertEqual(response.context["result"]["category"], "Normal")

    def test_page_matches_the_saved_model(self):
        flow = predictor.examples()["Generic"]
        result = self.client.post("/", flow).context["result"]
        self.assertEqual(result, predictor.predict(flow))

    def test_negative_value_is_rejected(self):
        response = self.client.post("/", {**predictor.examples()["Normal"], "sbytes": -5})
        self.assertIsNone(response.context["result"])
        self.assertIn("sbytes", response.context["form"].errors)

    def test_unknown_category_is_rejected(self):
        response = self.client.post("/", {**predictor.examples()["Normal"], "state": "XYZ"})
        self.assertIn("state", response.context["form"].errors)
