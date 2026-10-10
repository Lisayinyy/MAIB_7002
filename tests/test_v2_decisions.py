"""Small unit fixtures test the decision code, not a synthetic experiment dataset."""
import unittest
import numpy as np
import pandas as pd
from finalproject_pricingml.v2 import choose_action, scores


def curve():
    return pd.DataFrame({"rate": [1., .95, .9, .85], "supported": [True]*4,
                         "selected_sales": [1., 1.08, 1.085, 1.2],
                         "rf_sales": [1., 1.08, 1.085, 1.2],
                         "cat_sales": [1., 1.08, 1.085, 1.2]})


class DecisionTests(unittest.TestCase):
    def test_no_baseline_is_explicit_abstention(self):
        c = curve(); c.loc[c.rate == 1, "supported"] = False
        r = choose_action(c, 0)
        self.assertEqual(r["status"], "abstain_support")
        self.assertTrue(np.isnan(r["recommended_rate"]))

    def test_stockout_requests_supply_review_not_automatic_full_price(self):
        self.assertEqual(choose_action(curve(), 1)["status"], "review_stockout")

    def test_no_evidence_of_inventory_age_forces_discount(self):
        c = curve(); c[["selected_sales", "rf_sales", "cat_sales"]] = 1.
        self.assertEqual(choose_action(c, 0)["status"], "keep_full_price")

    def test_value_floor_survives_screening(self):
        c = curve(); c.loc[c.rate == .85, ["selected_sales", "rf_sales", "cat_sales"]] = 1.1
        r = choose_action(c, 0)
        self.assertEqual(r["recommended_rate"], .95)
        self.assertGreaterEqual(r["value_share"], .95)

    def test_reference_disagreement_abstains(self):
        c = curve(); c["rf_sales"] = 1.
        self.assertEqual(choose_action(c, 0)["status"], "review_model_disagreement")

    def test_invalid_predictions_not_used(self):
        c = curve(); c.loc[1, "selected_sales"] = np.nan
        self.assertEqual(choose_action(c, 0)["status"], "review_invalid_prediction")

    def test_metric_calculation_rejects_nonfinite(self):
        with self.assertRaises(ValueError): scores([1,2], [1,np.nan])

    def test_milder_discount_wins_near_tie(self):
        c = curve().iloc[:3]
        self.assertEqual(choose_action(c, 0)["recommended_rate"], .95)


if __name__ == "__main__":
    unittest.main()
