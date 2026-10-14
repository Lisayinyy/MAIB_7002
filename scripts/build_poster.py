"""Builds the A1 poster (poster/poster.html) from the saved results, then renders it to PDF with headless Chrome.

Run from the project root:  uv run python scripts/build_poster.py
Charts are drawn as HTML and inline SVG so the type stays large and vector-sharp at 594 x 841 mm.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finalproject_pricingml import config as C, data, evaluate as ev, models, scenario  # noqa: E402

OUT = ROOT / "poster"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BLUE, ORANGE, GREEN, AMBER, GRAY = C.COLOURS["knn"], C.COLOURS["rf"], C.COLOURS["catboost"], C.COLOURS["ensemble"], C.COLOURS["baseline"]
INK, MUTED, GRID, LIGHT_BLUE = "#0b0b0b", "#52514e", "#e1e0d9", "#8db8ea"

# ----------------------------------------------------------------------------- data
feat = data.load_features()
train, test = ev.training_rows(feat), ev.test_rows(feat)
selected = pd.read_csv(C.PROCESSED / "selected_series.csv")
summary = pd.read_csv(C.RESULTS / "test_summary.csv", index_col=0)
preds = pd.read_csv(C.RESULTS / "test_predictions.csv", parse_dates=["dt"])
rf_grid = pd.read_csv(C.RESULTS / "rf_tuning_summary.csv", dtype={"max_depth": str})
curves = pd.read_csv(C.RESULTS / "catboost_learning_curves.csv", header=[0, 1], index_col=0)
oof = pd.read_csv(C.RESULTS / "oof_predictions.csv")

rel = feat.assign(relative_sales=feat[C.TARGET] / feat.groupby(C.KEY)[C.TARGET].transform("mean"))
bands6 = pd.cut(rel.discount_next, [0, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0], labels=["< 0.60", "0.60–0.70", "0.70–0.80", "0.80–0.90", "0.90–0.95", "none"])
signal = rel.groupby(bands6, observed=True).agg(rows=("relative_sales", "size"), relative_sales=("relative_sales", "mean"))

stockout_share = (feat.stockout_hours_t > 0).mean()
no_discount_share = (feat.discount_next >= 0.95).mean()
err_corr = (oof.rf - oof.actual).corr(oof.catboost - oof.actual)

# Phase 2: refit the locked ensemble and rerun the what-if (deterministic, about ten seconds)
rf = models.make_rf(max_depth="10", min_leaf=3)
cat = models.make_catboost(depth=5, learning_rate=0.03, iterations=500)
ensemble = models.make_ensemble([("rf", rf), ("catboost", cat)]).fit(train[C.FEATURES], train[C.TARGET])
response = scenario.discount_response(ensemble, test)
curve = scenario.response_curve(response)
support = scenario.supported_candidates(train, min_days=3)
supported = scenario.restrict(response, support)
actual = test.set_index(C.ROW_KEY).discount_next
band = lambda d: pd.cut(d, C.DISCOUNT_BAND_EDGES, labels=["deep", "moderate", "none"]).astype(str)
unguarded = scenario.recommend(supported, "max_value").set_index(C.ROW_KEY).reindex(actual.index)
policy = lambda **kw: scenario.guarded_recommend(scenario.reprice(supported, kw.pop("cost", 0.0)), test, "max_value", **kw).set_index(C.ROW_KEY).reindex(actual.index)
guarded = policy()
discounts = {"Manager (actual)": actual, "What-if alone": unguarded.discount, "+ stock rules": guarded.discount,
             "+ unit cost 0.7": policy(cost=0.7).discount, "+ cost 0.7, 10% hurdle": policy(cost=0.7, min_gain=0.10).discount}
bands_by_policy = pd.DataFrame({n: band(d).value_counts(normalize=True) * 100 for n, d in discounts.items()}).T.reindex(columns=["none", "moderate", "deep"]).fillna(0)
agreement = {n: (band(d) == band(actual)).mean() * 100 for n, d in discounts.items()}
decided = guarded.decided_by.value_counts(normalize=True) * 100
by_state = pd.DataFrame({"state": guarded.state, "manager": band(actual).values, "model": band(guarded.discount).values})
turning_agree = (by_state[by_state.state == scenario.STATES[2]].pipe(lambda x: (x.manager == x.model).mean())) * 100
mae = summary.mae
base = mae["Baseline (7-day mean)"]
ens = mae["RF + CatBoost (50/50)"]
mse_gain = (summary.mse["Baseline (7-day mean)"] - summary.mse["RF + CatBoost (50/50)"]) / summary.mse["Baseline (7-day mean)"] * 100
test_band = pd.cut(preds.discount_next, C.DISCOUNT_BAND_EDGES, labels=["deep", "moderate", "none"])
spike = preds.groupby(test_band, observed=True)[[C.TARGET, "pred_ensemble", "pred_baseline"]].mean()
deep_mae = (preds.target_sales - preds.pred_ensemble).abs().groupby(test_band, observed=True).mean()

# ----------------------------------------------------------------------------- helpers
def pct(x, d=0):
    return f"{x:.{d}f}%"

def hbar_rows(items, colour_of, fmt, max_value, label_w=62):
    """Horizontal bars as HTML. ``items``: list of (label, value)."""
    rows = []
    for label, value in items:
        rows.append(f'<div class="hb"><div class="hb-label" style="width:{label_w}mm">{label}</div>'
                    f'<div class="hb-track"><div class="hb-fill" style="width:{value / max_value * 100:.1f}%;background:{colour_of(label)}"></div></div>'
                    f'<div class="hb-value">{fmt(value)}</div></div>')
    return "\n".join(rows)

def vbar_chart(labels, values, fmt, colour=BLUE, height=58, reference=None, ref_label=""):
    """Vertical bars as HTML flex boxes. Heights scale to the max value."""
    vmax = max(values) * 1.18
    cols = []
    for label, value in zip(labels, values):
        cols.append(f'<div class="vb"><div class="vb-value">{fmt(value)}</div>'
                    f'<div class="vb-fill" style="height:{value / vmax * height:.1f}mm;background:{colour}"></div><div class="vb-label">{label}</div></div>')
    ref = ""
    if reference is not None:
        y = reference / vmax * height
        ref = f'<div class="vb-ref" style="bottom:{y + 9:.1f}mm"><span>{ref_label}</span></div>'
    return f'<div class="vbars" style="height:{height + 20}mm">{ref}{"".join(cols)}</div>'

def svg_lines(series, xlim, ylim, w, h, xlabel, ylabel, xticks, yticks, marks=(), xfmt=str, yfmt=str, reverse_x=False, hline=None):
    """A line chart as inline SVG in mm units. ``series``: list of (label, colour, xs, ys)."""
    pad_l, pad_b, pad_t, pad_r = 18, 14, 6, 6
    def X(x):
        f = (x - xlim[0]) / (xlim[1] - xlim[0])
        if reverse_x:
            f = 1 - f
        return pad_l + f * (w - pad_l - pad_r)
    def Y(y):
        return pad_t + (1 - (y - ylim[0]) / (ylim[1] - ylim[0])) * (h - pad_t - pad_b)
    parts = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;display:block;font-family:inherit">']
    for t in yticks:
        parts.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(t):.2f}" y2="{Y(t):.2f}" stroke="{GRID}" stroke-width="0.3"/>'
                     f'<text x="{pad_l - 2}" y="{Y(t) + 1.6:.2f}" font-size="4.2" text-anchor="end" fill="{MUTED}">{yfmt(t)}</text>')
    for t in xticks:
        parts.append(f'<text x="{X(t):.2f}" y="{h - pad_b + 6:.2f}" font-size="4.2" text-anchor="middle" fill="{MUTED}">{xfmt(t)}</text>')
    if hline is not None:
        parts.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(hline):.2f}" y2="{Y(hline):.2f}" stroke="{MUTED}" stroke-width="0.5"/>')
    parts.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{h - pad_b:.2f}" y2="{h - pad_b:.2f}" stroke="{MUTED}" stroke-width="0.4"/>')
    for label, colour, xs, ys in series:
        pts = " ".join(f"{X(x):.2f},{Y(y):.2f}" for x, y in zip(xs, ys))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="1.1" stroke-linejoin="round"/>')
        if len(xs) <= 12:
            for x, y in zip(xs, ys):
                parts.append(f'<circle cx="{X(x):.2f}" cy="{Y(y):.2f}" r="1.6" fill="{colour}" stroke="white" stroke-width="0.6"/>')
    for text, x, y, colour, anchor in marks:
        parts.append(f'<text x="{X(x):.2f}" y="{Y(y):.2f}" font-size="4.4" font-weight="600" fill="{colour}" text-anchor="{anchor}">{text}</text>')
    parts.append(f'<text x="{(pad_l + w - pad_r) / 2:.2f}" y="{h - 1:.2f}" font-size="4.2" text-anchor="middle" fill="{MUTED}">{xlabel}</text>')
    parts.append(f'<text transform="translate(5,{(pad_t + h - pad_b) / 2:.2f}) rotate(-90)" font-size="4.2" text-anchor="middle" fill="{MUTED}">{ylabel}</text>')
    parts.append("</svg>")
    return "\n".join(parts)

def svg_scatter(xs, ys, w, h, colour, lim):
    pad_l, pad_b, pad_t, pad_r = 14, 12, 4, 4
    X = lambda x: pad_l + x / lim * (w - pad_l - pad_r)
    Y = lambda y: pad_t + (1 - y / lim) * (h - pad_t - pad_b)
    parts = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;display:block;font-family:inherit">']
    for t in range(0, int(lim) + 1, 2):
        parts.append(f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{Y(t):.2f}" y2="{Y(t):.2f}" stroke="{GRID}" stroke-width="0.3"/>'
                     f'<text x="{pad_l - 2}" y="{Y(t) + 1.5:.2f}" font-size="4" text-anchor="end" fill="{MUTED}">{t}</text>'
                     f'<text x="{X(t):.2f}" y="{h - pad_b + 5.5:.2f}" font-size="4" text-anchor="middle" fill="{MUTED}">{t}</text>')
    parts.append(f'<line x1="{X(0):.2f}" y1="{Y(0):.2f}" x2="{X(lim):.2f}" y2="{Y(lim):.2f}" stroke="{MUTED}" stroke-width="0.5"/>')
    for x, y in zip(xs, ys):
        parts.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="0.75" fill="{colour}" fill-opacity="0.4"/>')
    parts.append(f'<text x="{X(lim) - 1:.2f}" y="{Y(lim) + 5:.2f}" font-size="4" text-anchor="end" fill="{MUTED}">perfect forecast</text>')
    parts.append(f'<text x="{(pad_l + w - pad_r) / 2:.2f}" y="{h - 1:.2f}" font-size="4.2" text-anchor="middle" fill="{MUTED}">Actual next-day sales</text>')
    parts.append(f'<text transform="translate(4,{(pad_t + h - pad_b) / 2:.2f}) rotate(-90)" font-size="4.2" text-anchor="middle" fill="{MUTED}">Predicted (ensemble)</text>')
    parts.append("</svg>")
    return "\n".join(parts)

# ----------------------------------------------------------------------------- icons (24x24 stroke icons)
def icon(name, colour=BLUE, size=14):
    paths = {
        "shelf": '<path d="M3 7h18M3 13h18M3 19h18"/><rect x="5" y="3" width="4" height="4" rx="0.8"/><rect x="11" y="3" width="4" height="4" rx="0.8"/><rect x="5" y="9" width="4" height="4" rx="0.8"/><rect x="15" y="15" width="4" height="4" rx="0.8"/>',
        "tag": '<path d="M3 11V4a1 1 0 0 1 1-1h7l10 10-8 8L3 11z"/><circle cx="7.5" cy="7.5" r="1.4"/>',
        "moon": '<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>',
        "scale": '<path d="M12 3v18M5 21h14M6 8l6-2 6 2M3 14a3 3 0 0 0 6 0L6 8l-3 6zM15 14a3 3 0 0 0 6 0l-3-6-3 6z"/>',
        "cloud": '<path d="M7 18a4 4 0 0 1-.6-7.95A6 6 0 0 1 18 9a4 4 0 0 1 0 9H7z"/><path d="M4 4l16 16" stroke="#eb6834"/>',
        "question": '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.4-1 .9-1 1.7M12 17h.01"/>',
        "eye": '<path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12z"/><circle cx="12" cy="12" r="3"/><path d="M4 4l16 16" stroke="#eb6834"/>',
        "pin": '<path d="M12 21s-6-5.5-6-11a6 6 0 0 1 12 0c0 5.5-6 11-6 11z"/><circle cx="12" cy="10" r="2.2"/>',
        "units": '<path d="M4 17V7l4 6 4-6v10M15 7h5M17.5 7v10M15 17h5"/><path d="M3 20L21 4" stroke="#eb6834"/>',
        "dice": '<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8" cy="8" r="1.3" fill="currentColor"/><circle cx="16" cy="8" r="1.3" fill="currentColor"/><circle cx="12" cy="12" r="1.3" fill="currentColor"/><circle cx="8" cy="16" r="1.3" fill="currentColor"/><circle cx="16" cy="16" r="1.3" fill="currentColor"/>',
        "calendar": '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/>',
        "brain": '<path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 3 3h3V4H9zM15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-3 3h-3V4h3z"/>',
        "spark": '<path d="M4 18l5-7 4 4 7-9"/><path d="M16 6h4v4"/>',
        "bolt": '<path d="M13 2L4 14h7l-1 8 9-12h-7l1-8z"/>',
        "empty": '<path d="M3 19h18M3 9h18"/><path d="M8 13h8" stroke-dasharray="1.5 2"/>',
    }
    return (f'<svg viewBox="0 0 24 24" style="width:{size}mm;height:{size}mm;flex:none" fill="none" stroke="{colour}" stroke-width="1.7" '
            f'stroke-linecap="round" stroke-linejoin="round" color="{colour}">{paths[name]}</svg>')

FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
shelf_art = f"""
<svg viewBox="0 0 200 120" style="width:100%;height:auto;display:block" xmlns="http://www.w3.org/2000/svg">
  <defs><linearGradient id="g" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#dfe9f8"/><stop offset="1" stop-color="#f6f9fe"/></linearGradient></defs>
  <rect x="10" y="8" width="180" height="104" rx="10" fill="url(#g)"/>
  <g stroke="#7a8794" stroke-width="2.2" stroke-linecap="round"><path d="M28 48h144M28 84h144"/><path d="M34 48v-4M166 48v-4M34 84v-4M166 84v-4"/></g>
  <g><circle cx="48" cy="38" r="8" fill="{ORANGE}"/><circle cx="62" cy="40" r="7" fill="{ORANGE}"/><circle cx="55" cy="30" r="7" fill="#f28a5c"/></g>
  <g><ellipse cx="100" cy="40" rx="15" ry="6" fill="#e0b66a"/><ellipse cx="100" cy="35" rx="15" ry="6" fill="#efcf8b"/></g>
  <g><rect x="130" y="26" width="12" height="20" rx="2" fill="{GREEN}"/><rect x="146" y="26" width="12" height="20" rx="2" fill="#5fcf9f"/></g>
  <g><circle cx="52" cy="74" r="7" fill="#a4d46a"/><circle cx="66" cy="76" r="7" fill="#8bc34a"/></g>
  <g><rect x="92" y="60" width="26" height="22" rx="3" fill="#f5e7c4" stroke="#d9c38f" stroke-width="1.5"/><path d="M96 66h18M96 72h18M96 78h12" stroke="#d9c38f" stroke-width="1.5" stroke-linecap="round"/></g>
  <g transform="translate(140 58) rotate(-12)"><path d="M0 8a8 8 0 0 1 8-8h22l14 14-14 14H8a8 8 0 0 1-8-8V8z" fill="{AMBER}"/><circle cx="8" cy="12" r="2.5" fill="white"/>
    <text x="27" y="17" font-size="12" font-weight="800" fill="#10213a" text-anchor="middle" font-family="{FONT}">&#8722;?%</text></g>
  <text x="100" y="105" font-size="7" fill="#52514e" text-anchor="middle" font-family="{FONT}">6 pm &#183; the shelf is still half full &#183; what is tomorrow's price?</text>
</svg>"""

# ----------------------------------------------------------------------------- chart pieces
order = ["Baseline (7-day mean)", "kNN (k=25)", "Random forest", "CatBoost", "RF + CatBoost (50/50)"]
nice = {"Baseline (7-day mean)": "7-day average (no ML)", "kNN (k=25)": "k-nearest neighbours", "Random forest": "Random forest", "CatBoost": "CatBoost", "RF + CatBoost (50/50)": "Forest + CatBoost blend"}
colour_of = {nice[k]: v for k, v in {"Baseline (7-day mean)": GRAY, "kNN (k=25)": BLUE, "Random forest": ORANGE, "CatBoost": GREEN, "RF + CatBoost (50/50)": AMBER}.items()}
mae_bars = hbar_rows([(nice[m], mae[m]) for m in order], lambda label: colour_of[label], lambda v: f"{v:.3f}", 0.345, label_w=64)

d = list(curve.index)
response_svg = svg_lines([("Predicted sales", BLUE, d, list(curve.sales_vs_no_discount)), ("Predicted value", ORANGE, d, list(curve.value_vs_no_discount))],
                         (0.6, 1.0), (0.9, 2.5), 150, 62, "Tomorrow's discount rate (deeper to the right)", "× no-discount level", d, [1.0, 1.5, 2.0, 2.5],
                         marks=[(f"sales {curve.sales_vs_no_discount.iloc[-1]:.1f}×", 0.66, 2.3, BLUE, "end"), (f"value {curve.value_vs_no_discount.iloc[-1]:.2f}×", 0.66, 1.5, ORANGE, "end")],
                         xfmt=lambda v: f"{v:.2f}", yfmt=lambda v: f"{v:.1f}", reverse_x=True, hline=1.0)

short = {"Manager (actual)": "The managers", "What-if alone": "Model alone", "+ stock rules": "Model + stock rules", "+ unit cost 0.7": "… + unit cost"}
policy_rows = []
for name, row in bands_by_policy.loc[list(short)].iterrows():
    segs = "".join(f'<div class="seg" style="width:{row[c]:.1f}%;background:{col};color:{ink}">{pct(row[c]) if row[c] >= 7 else ""}</div>'
                   for c, col, ink in [("none", "#d9d8d0", INK), ("moderate", LIGHT_BLUE, INK), ("deep", BLUE, "white")])
    policy_rows.append(f'<div class="pr"><div class="pr-label">{short[name]}</div><div class="pr-track">{segs}</div><div class="pr-agree"><div style="width:{agreement[name]:.1f}%"></div><span>{pct(agreement[name])}</span></div></div>')
policy_chart = "\n".join(policy_rows)

lc = curves["lr=0.03"].iloc[::10]
best_it = int(curves["lr=0.03"].validation.idxmin())
learning_svg = svg_lines([("Training", MUTED, list(lc.index + 1), list(lc.train)), ("Validation", GREEN, list(lc.index + 1), list(lc.validation))],
                         (0, 3000), (0.2, 0.6), 120, 48, "Trees added", "Error", [0, 1500, 3000], [0.2, 0.4, 0.6],
                         marks=[(f"new week: turns up after {best_it}", 2950, 0.35, GREEN, "end"), ("training rows: keeps falling", 2950, 0.212, MUTED, "end")], yfmt=lambda v: f"{v:.1f}")
deep_bars = vbar_chart(["no discount", "moderate", "deep"], [deep_mae["none"], deep_mae["moderate"], deep_mae["deep"]], lambda v: f"{v:.2f}", colour=AMBER, height=26)

flow_steps = [("shelf", "Sold out today?", "Then no discount tomorrow: the shelf is clear."),
              ("calendar", "A week with no sell-out?", "Then a discount is required; the model picks how deep."),
              ("brain", "Otherwise ask the model", "Only about discounts this product has actually run."),
              ("tag", "Choose", "Highest predicted value, net of unit cost, if the gain is worth it.")]
flow = "".join(f'<div class="step">{icon(ic, BLUE, 11)}<b>{t}</b><span>{b}</span></div>{"<i>&#8594;</i>" if i < 3 else ""}' for i, (ic, t, b) in enumerate(flow_steps))

# ----------------------------------------------------------------------------- page
html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Cut the price, or trust the shelf?</title>
<style>
@page {{ size: 594mm 841mm; margin: 0; }}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; width: 594mm; height: 841mm; background: #ffffff; color: {INK}; font-family: {FONT}; font-size: 18pt; line-height: 1.3; }}
.page {{ width: 594mm; height: 841mm; padding: 14mm 18mm 8mm; display: flex; flex-direction: column; gap: 7mm; }}
header {{ display: grid; grid-template-columns: 1fr 190mm; gap: 14mm; align-items: center; background: #eef3fb; border-radius: 8mm; padding: 10mm 16mm; }}
.kicker {{ font-size: 15pt; letter-spacing: 0.14em; text-transform: uppercase; color: {BLUE}; font-weight: 700; }}
h1 {{ font-size: 90pt; line-height: 0.98; margin: 5mm 0 6mm; font-weight: 800; letter-spacing: -0.02em; color: #10213a; }}
h1 span {{ color: {AMBER}; }}
.sub {{ font-size: 25pt; color: #33415a; max-width: 330mm; line-height: 1.3; }}
.who {{ margin-top: 6mm; font-size: 17pt; color: #52514e; }} .who b {{ color: #10213a; }}
h2 {{ font-size: 32pt; margin: 0 0 4.5mm; line-height: 1.1; color: #10213a; display: flex; align-items: center; gap: 4mm; }}
h2 i {{ font-style: normal; font-size: 14pt; color: white; background: {BLUE}; border-radius: 2.5mm; padding: 1mm 3mm; font-weight: 700; }}
section {{ background: white; border: 0.5mm solid {GRID}; border-radius: 6mm; padding: 7.5mm 10mm; }}
.row {{ display: grid; gap: 7mm; }}
.moments {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6mm; }}
.moment {{ background: #f7f9fc; border-radius: 5mm; padding: 6mm 7mm; display: flex; gap: 5mm; align-items: flex-start; }}
.moment b {{ display: block; font-size: 22pt; color: #10213a; margin-bottom: 1.5mm; }} .moment span {{ font-size: 16pt; color: #33415a; }}
.hero {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6mm; }}
.hero div {{ background: #fff7e6; border-radius: 5mm; padding: 6mm 7mm; }}
.hero b {{ display: block; font-size: 54pt; line-height: 1; color: {AMBER}; font-weight: 800; }} .hero span {{ display: block; margin-top: 2mm; font-size: 15.5pt; color: #52514e; }}
.hb {{ display: flex; align-items: center; gap: 4mm; margin: 2.6mm 0; }} .hb-label {{ font-size: 17.5pt; }} .hb-track {{ flex: 1; height: 11mm; background: #f3f2ee; border-radius: 1.5mm; overflow: hidden; }}
.hb-fill {{ height: 100%; border-radius: 0 1.5mm 1.5mm 0; }} .hb-value {{ width: 20mm; font-weight: 700; font-size: 18.5pt; text-align: right; }}
.cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4mm; }}
.card {{ border-left: 2.5mm solid; padding: 1mm 0 0 4mm; font-size: 16pt; color: #33415a; }} .card b {{ display: block; font-size: 18.5pt; color: {INK}; }}
.lesson {{ background: #f7f9fc; border-radius: 4mm; padding: 4.5mm 6mm; margin-top: 6mm; font-size: 17pt; }}
.flow {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4mm; margin: 2mm 0 0; }}
.step {{ background: #f7f9fc; border-radius: 4mm; padding: 5mm 6mm 4mm; font-size: 15pt; color: #33415a; display: flex; flex-direction: column; gap: 2mm; }}
.step b {{ font-size: 18pt; color: #10213a; }} .flow i {{ display: none; }}
.pr {{ display: flex; align-items: center; gap: 3mm; margin: 2.6mm 0; }} .pr-label {{ width: 50mm; font-size: 16.5pt; }}
.pr-track {{ flex: 1; display: flex; height: 11mm; border-radius: 1.5mm; overflow: hidden; gap: 0.6mm; }} .seg {{ display: flex; align-items: center; justify-content: center; font-size: 13.5pt; }}
.pr-agree {{ width: 46mm; position: relative; height: 11mm; background: #f3f2ee; border-radius: 1.5mm; overflow: hidden; }} .pr-agree div {{ height: 100%; background: {AMBER}; }}
.pr-agree span {{ position: absolute; left: 2.5mm; top: 0; line-height: 11mm; font-size: 13.5pt; font-weight: 700; }}
.legend {{ font-size: 14pt; color: {MUTED}; display: flex; gap: 5mm; margin: 1.5mm 0 0 53mm; white-space: nowrap; }} .legend b {{ display: inline-block; width: 4.5mm; height: 4.5mm; border-radius: 1mm; margin-right: 1.5mm; vertical-align: -0.6mm; }}
.small {{ font-size: 15.5pt; color: {MUTED}; }}
.tiles {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 7mm; }}
.tile {{ background: #f7f9fc; border-radius: 5mm; padding: 5.5mm 8mm; font-size: 16pt; color: #33415a; }}
.tile .top {{ display: flex; align-items: center; gap: 4mm; margin-bottom: 2.5mm; }} .tile b {{ font-size: 20pt; color: #10213a; line-height: 1.15; }}
.tile .num {{ font-size: 40pt; font-weight: 800; color: {BLUE}; line-height: 1; margin: 1mm 0 2mm; }}
.vbars {{ position: relative; display: flex; align-items: flex-end; gap: 2.5mm; padding: 0 1mm; }}
.vb {{ flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%; }}
.vb-fill {{ width: 100%; border-radius: 1.2mm 1.2mm 0 0; }} .vb-value {{ font-size: 12.5pt; margin-bottom: 0.8mm; }} .vb-label {{ font-size: 11.5pt; color: {MUTED}; margin-top: 1.2mm; height: 5mm; white-space: nowrap; }}
.vb-ref {{ position: absolute; left: 0; right: 0; border-top: 0.5mm dashed {MUTED}; }} .vb-ref span {{ position: absolute; right: 1mm; top: -5mm; font-size: 11pt; color: {MUTED}; }}
footer {{ display: flex; justify-content: space-between; font-size: 13pt; color: {MUTED}; padding: 0 2mm; }}
</style></head><body><div class="page">

<header>
  <div>
    <div class="kicker">HKU MAIB 7002 &#183; Group Q &#183; Machine learning for pricing</div>
    <h1>Cut the price,<br>or <span>trust the shelf?</span></h1>
    <div class="sub">We taught a model to predict what a perishable will sell tomorrow at any discount. Then we discovered it would discount everything, and had to teach it to read the shelf first.</div>
    <div class="who"><b>Aashish Omprakash Pareek &#183; Yuanyuan Yin &#183; Lingyu Chen</b> &nbsp;&#183;&nbsp; Dingdong FreshRetailNet-50K, a public dataset of 50,000 store-product histories</div>
  </div>
  <div>{shelf_art}</div>
</header>

<div class="row" style="grid-template-columns: 1.55fr 1fr">
  <section>
    <h2><i>01</i>A decision every evening, for every product</h2>
    <div class="moments">
      <div class="moment">{icon("moon", BLUE, 12)}<div><b>6 pm, the shelf is half full</b><span>Tomorrow's discount must be set tonight, before anyone knows how much will sell.</span></div></div>
      <div class="moment">{icon("scale", BLUE, 12)}<div><b>Two ways to lose</b><span>Hold the price and stock may be thrown away. Cut it and margin goes on sales that would have happened anyway.</span></div></div>
      <div class="moment">{icon("question", BLUE, 12)}<div><b>Our question</b><span>For one product, in one store, on one day: how much sells tomorrow at each discount, and should we discount at all?</span></div></div>
    </div>
  </section>
  <section>
    <h2><i>02</i>The raw material</h2>
    <div class="hero">
      <div><b>312</b><span>product histories in five stores, picked because their discounts vary</span></div>
      <div><b>{len(feat):,}</b><span>store-product-days; one unseen week of 2,184 kept for the test</span></div>
      <div><b>{stockout_share:.0%}</b><span>of days have an empty-shelf hour, a clue we later put to work</span></div>
    </div>
  </section>
</div>

<div class="row" style="grid-template-columns: 1fr 1fr">
  <section>
    <h2><i>03</i>Four models, one unseen week</h2>
    <p class="small" style="margin:0 0 3mm">Each model learns from the product's own recent sales, today's stock and tomorrow's planned discount. Error on the 2,184 held-out days; lower is better.</p>
    {mae_bars}
    <div class="lesson"><b>What we learned:</b> every model beat the no-ML average, the three tree models tie within noise, and choosing the inputs mattered more than any tuning dial. Dropping the weather was the single biggest gain.</div>
  </section>
  <section>
    <h2><i>04</i>The decisions behind the numbers</h2>
    <div class="cards">
      <div class="card" style="border-color:{BLUE}"><b>k-nearest neighbours &#183; 25 look-alike days</b>Finds the 25 most similar past days and averages them. Simple, and already 5% better than the average.</div>
      <div class="card" style="border-color:{ORANGE}"><b>Random forest &#183; 300 trees, depth 10</b>All 16 settings we tried beat the baseline every single week. A forgiving model.</div>
      <div class="card" style="border-color:{GREEN}"><b>CatBoost &#183; small and slow on purpose</b>Boosting over-fits in plain sight (see 06), so the best version is 500 shallow trees at a 0.03 learning rate.</div>
      <div class="card" style="border-color:{AMBER}"><b>Blend &#183; 50/50 forest + CatBoost</b>Their mistakes are 97% correlated, so blending gains only 0.001. We keep it as a safe choice, not a clever one.</div>
    </div>
    <div class="lesson">Splits are strictly in time order: every validation week is predicted by a model that has never seen it. Random splits would have flattered every model by putting near-identical days on both sides.</div>
  </section>
</div>

<section>
  <h2><i>05</i>The twist: a good forecaster is a reckless pricer</h2>
  <div class="row" style="grid-template-columns: 1fr 1.1fr 1.25fr; align-items: start">
    <div>
      <p style="margin:0 0 2mm"><b>Ask the model "what if I discount?"</b> and it answers: deeper is always better. At a 0.6 rate it predicts {curve.sales_vs_no_discount.iloc[-1]:.1f}× the sales.</p>
      {response_svg}
      <p class="small">It learned that from clearance days, which are fewer than 1% of the data. Left alone it would discount on 98% of days.</p>
    </div>
    <div>
      <p style="margin:0 0 2mm"><b>So we put the shelf before the model.</b> Three stock rules decide first; the model only chooses among discounts the product has actually run.</p>
      <div class="flow">{flow}</div>
    </div>
    <div>
      <p style="margin:0 0 2mm"><b>What each policy recommends</b> on the test week, and how often it lands in the same band as the real managers.</p>
      {policy_chart}
      <div class="legend"><span><b style="background:#d9d8d0"></b>no discount</span><span><b style="background:{LIGHT_BLUE}"></b>moderate</span><span><b style="background:{BLUE}"></b>deep</span><span style="margin-left:auto"><b style="background:{AMBER}"></b>agrees with manager</span></div>
      <p class="small" style="margin-top:3mm">Stock rules decide {100 - decided["model"]:.0f}% of days before the model is asked. Agreement rises from {agreement["What-if alone"]:.0f}% to {agreement["+ stock rules"]:.0f}%, and to {agreement["+ unit cost 0.7"]:.0f}% once a unit cost makes deep discounts cost something.</p>
    </div>
  </div>
</section>

<section>
  <h2><i>06</i>The challenges we found most interesting</h2>
  <div class="tiles">
    <div class="tile"><div class="top">{icon("empty", BLUE)}<b>The shelf lies</b></div>On {stockout_share:.0%} of days the shelf was empty for part of the day, so recorded sales are lower than real demand. The model learns from censored numbers.</div>
    <div class="tile"><div class="top">{icon("units", BLUE)}<b>Sales without units or prices</b></div>The data is normalised: 1.2 is not 1.2 kilograms or dollars. "Value" is sales × discount rate, a proxy, so profit cannot be computed.</div>
    <div class="tile"><div class="top">{icon("dice", BLUE)}<b>Managers are not random</b></div>Discounts were chosen on days managers expected demand. Part of the lift the model sees is their judgement, not the price cut. Correlation, not cause.</div>
    <div class="tile"><div class="top">{icon("eye", BLUE)}<b>No counterfactual</b></div>We only ever see what sold at the price the manager chose. Whether our policy sells more or wastes less needs a real experiment.</div>
    <div class="tile"><div class="top">{icon("cloud", BLUE)}<b>Weather made things worse</b></div><div class="num">0.327 &#8594; 0.312</div>Removing rain, temperature and humidity cut the error more than any amount of tuning did.</div>
    <div class="tile"><div class="top">{icon("spark", BLUE)}<b>Deep discounts are the hard days</b></div>{deep_bars}<span class="small">Error by tomorrow's discount: every model under-predicts the spike on exactly the days a pricing tool cares about.</span></div>
    <div class="tile"><div class="top">{icon("bolt", BLUE)}<b>Boosting over-fits in plain sight</b></div>{learning_svg}<span class="small">Training error keeps falling; error on a new week turns up after {best_it} trees.</span></div>
    <div class="tile"><div class="top">{icon("pin", BLUE)}<b>One city, one week, one guess at shelf life</b></div>Five stores in one city and a single test week that has been scored for four models. The "no sell-out for a week means old stock" rule assumes a shelf life the data never states.</div>
  </div>
</section>

<footer><span>Data: Dingdong-Inc/FreshRetailNet-50K on Hugging Face, CC BY 4.0 &#183; scikit-learn and CatBoost &#183; Errors are mean absolute error in normalised sales units</span><span>Full notebook, code and Docker image: github.com/Lisayinyy/MAIB_7002, branch pricing-ml-ap</span></footer>
</div></body></html>"""

OUT.mkdir(exist_ok=True)
(OUT / "poster.html").write_text(html)
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--no-margins",
                f"--print-to-pdf={OUT / 'pricing_ml_poster_A1.pdf'}", str(OUT / "poster.html")], check=True, capture_output=True)
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=2245,3179",
                f"--screenshot={OUT / 'pricing_ml_poster_A1_preview.png'}", str(OUT / "poster.html")], check=True, capture_output=True)
print("wrote", OUT / "pricing_ml_poster_A1.pdf")
