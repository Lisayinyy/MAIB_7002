"""Next-day sales prediction for discounted perishables: data, evaluation harness, models and plots.

Typical use from the notebook::

    from finalproject_pricingml import config as C, data, evaluate as ev, models, plots
    feat = data.load_features()
    results = ev.cross_validate(feat, models.make_knn, [{"k": k} for k in (5, 25, 100)])
"""

from . import config, data, evaluate, models, plots  # noqa: F401
