#!/usr/bin/env python3
"""Build the evidence-backed, single-page A1 poster from saved V2 results.

This is a presentation step, not a training or model-selection step. It never
changes result files and refuses to invent missing scores or an example.
Dependencies: reportlab (4.4.9 verified); optional pypdf for the independent audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from xml.sax.saxutils import escape

import reportlab
from reportlab.lib.colors import HexColor, Color
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
MM = 72 / 25.4
WIDTH, HEIGHT = 594 * MM, 841 * MM
MARGIN = 75
INNER = WIDTH - 2 * MARGIN
INK = HexColor("#152F35")
TEAL = HexColor("#05776D")
ORANGE = HexColor("#B84D23")
MUTED = HexColor("#52686B")
RULE = HexColor("#B5C6C4")
PALE = HexColor("#EAF3F0")
FONT, BOLD = "PosterVera", "PosterVeraBold"


def first(record, *names, default=None):
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return default


def finite(value, label):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite evidence: {label}")
    return number


def load_evidence(results: Path):
    paths = [results / name for name in ("comparison.csv", "selection.json", "policy_summary.json")]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Run the V2 pipeline before making the poster: missing {path}")
    with paths[0].open(newline="") as source:
        raw = list(csv.DictReader(source))
    selection = json.loads(paths[1].read_text())
    policy = json.loads(paths[2].read_text())
    rows = []
    for row in raw:
        model = first(row, "model", "name", "method")
        if not model:
            raise ValueError("comparison.csv needs a model/name/method column")
        rows.append({
            "model": str(model),
            "val_mae": finite(first(row, "validation_mae", "val_mae", "cv_mae", "mean_validation_mae"), f"{model} validation MAE"),
            "test_mae": finite(first(row, "test_mae", "final_mae", "mae"), f"{model} final MAE"),
            "test_rmse": finite(first(row, "test_rmse", "final_rmse", "rmse"), f"{model} final RMSE"),
        })
    selected = first(selection, "selected_model", "selected_name", "model")
    if isinstance(selected, dict):
        selected = first(selected, "name", "model")
    matching = [row for row in rows if row["model"] == selected]
    if len(matching) != 1:
        raise ValueError(f"Cannot uniquely match selected model {selected!r} to comparison.csv")
    baseline = [row for row in rows if "7" in row["model"] and any(t in row["model"].lower() for t in ("mean", "average"))]
    if len(baseline) != 1:
        raise ValueError("Expected exactly one seven-day mean baseline in comparison.csv")
    for name in ("recommendations.csv", "discount_response.csv", "features.parquet", "experiment_plan.json"):
        path = results / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing example/protocol evidence: {path}")
        paths.append(path)
    with (results / "recommendations.csv").open(newline="") as source:
        recs = list(csv.DictReader(source))
    key = lambda r: (int(r["store_id"]), int(r["product_id"]), r["dt"][:10])
    positive = [r for r in recs if r["status"] == "recommend_discount"]
    chosen = min(positive or recs, key=key)
    with (results / "discount_response.csv").open(newline="") as source:
        candidates = [r for r in csv.DictReader(source) if key(r) == key(chosen)]
    supported = [r for r in candidates if r["supported"].lower() in ("true", "1")]
    data_python = sys.executable
    try:
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError:
        local = ROOT / ".venv" / "bin" / "python"
        if not local.is_file():
            raise RuntimeError("Build with an interpreter containing pandas, pyarrow and reportlab")
        data_python = str(local)
    extract = """import json,sys,pandas as pd
f=pd.read_parquet(sys.argv[1]); s,p,d=int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
r=f[(f.store_id==s)&(f.product_id==p)&(f.dt.dt.strftime('%Y-%m-%d')==d)]
assert len(r)==1
print(json.dumps({'row':r.iloc[0].to_dict(),'n_series':len(f[['store_id','product_id']].drop_duplicates()),'test_rows':int((f['split']=='test').sum())},default=str))
"""
    extracted = subprocess.run([data_python, "-c", extract, str(results / "features.parquet"),
                                str(chosen["store_id"]), str(chosen["product_id"]), chosen["dt"][:10]],
                               check=True, capture_output=True, text=True)
    feature_evidence = json.loads(extracted.stdout)
    selection = selection | {"protocol": {"n_series": feature_evidence["n_series"],
                                          "test_rows": feature_evidence["test_rows"]}}
    selection["experiment_plan"] = json.loads((results / "experiment_plan.json").read_text())
    recommended = chosen["recommended_rate"]
    example = feature_evidence["row"] | {
        "reference_discount": 1.0, "recommended_discount": float(recommended) if recommended else None,
        "status": chosen["status"], "reason": chosen["reason"],
        "example_selection": "First recommend_discount row by store, SKU and date; first output row if none.",
        "candidates": [{"discount": float(r["rate"]), "pred_sales": float(r["selected_sales"]),
                        "pred_value": float(r["rate"]) * float(r["selected_sales"]),
                        "support_days": int(r["support_days"])} for r in (supported or candidates)],
    }
    transfer_dir = results.parent / "v2_transfer"
    transfer_paths = [transfer_dir / name for name in ("summary.csv", "run_receipt.json", "experiment_plan.json")]
    if all(path.is_file() for path in transfer_paths):
        with transfer_paths[0].open(newline="") as source:
            transfer_rows = {r["model"]: r for r in csv.DictReader(source)}
        receipt = json.loads(transfer_paths[1].read_text())
        transfer_plan = json.loads(transfer_paths[2].read_text())
        if transfer_plan["selected_model"] != selected or receipt["tuning_performed"]:
            raise ValueError("Transfer experiment must retain the locked selection without retuning")
        policy["transfer"] = {"selected": transfer_rows[selected],
                              "reference": transfer_rows["Teammate blend 50/50"], "receipt": receipt}
        paths.extend(transfer_paths)
    return rows, selection, policy, matching[0], baseline[0], example, paths


def register_fonts():
    fonts = Path(reportlab.__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont(FONT, str(fonts / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont(BOLD, str(fonts / "VeraBd.ttf")))
    pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=BOLD, italic=FONT, boldItalic=BOLD)


class Poster:
    """Top-origin drawing helpers with explicit paragraph overflow checks."""

    def __init__(self, path):
        self.canvas = canvas.Canvas(str(path), pagesize=(WIDTH, HEIGHT), pageCompression=1)
        self.canvas.setTitle("Daily Discount Decisions for Perishable Food | Group Q")
        self.canvas.setAuthor("Group Q: Aashish Omprakash Pareek, Yuanyuan Yin, Lingyu Chen")
        self.canvas.setSubject("MAIB7002: public-data sales forecasting and constrained discount scenarios")

    def paragraph(self, text, x, y, width, size=22, leading=None, color=INK,
                  bold=False, max_height=None):
        style = ParagraphStyle("poster", fontName=BOLD if bold else FONT, fontSize=size,
                               leading=leading or size * 1.32, textColor=color,
                               alignment=TA_LEFT, spaceBefore=0, spaceAfter=0)
        p = Paragraph(text, style)
        w, h = p.wrap(width, HEIGHT)
        if max_height is not None and h > max_height + 0.1:
            raise ValueError(f"Paragraph exceeds its {max_height:.1f}pt slot ({h:.1f}pt): {text[:90]}")
        if y + h > HEIGHT - 35:
            raise ValueError(f"Paragraph leaves page: {text[:90]}")
        p.drawOn(self.canvas, x, HEIGHT - y - h)
        return h

    def text(self, text, x, y, size=22, color=INK, bold=False, align="left"):
        c = self.canvas
        c.setFillColor(color)
        c.setFont(BOLD if bold else FONT, size)
        if align == "right":
            c.drawRightString(x, HEIGHT - y - size, str(text))
        else:
            c.drawString(x, HEIGHT - y - size, str(text))

    def line(self, x1, y1, x2, y2, color=RULE, thickness=1):
        c = self.canvas
        c.setStrokeColor(color)
        c.setLineWidth(thickness)
        c.line(x1, HEIGHT - y1, x2, HEIGHT - y2)

    def rect(self, x, y, width, height, color):
        c = self.canvas
        c.setFillColor(color)
        c.rect(x, HEIGHT - y - height, width, height, stroke=0, fill=1)

    def section(self, number, title, y, x=MARGIN, width=INNER):
        self.line(x, y, x + width, y, thickness=1.4)
        self.text(number, x, y + 15, 22, TEAL, True)
        self.text(title, x + 48, y + 11, 31, INK, True)

    def finish(self):
        self.canvas.showPage()
        self.canvas.save()


def model_label(name):
    label = str(name).replace("Random forest", "Random Forest").replace("HistGradientBoosting", "Hist. boosting")
    label = label.replace("Baseline (7-day mean)", "7-day mean baseline")
    label = label.replace("Same weekday last week", "Same-weekday baseline")
    label = label.replace("CatBoost history + IDs MAE", "CatBoost + history/IDs (MAE)")
    label = label.replace("CatBoost history + IDs", "CatBoost + history/IDs (RMSE)")
    label = label.replace("Teammate RF", "Original Random Forest").replace("Teammate CatBoost", "Original CatBoost")
    label = label.replace("Teammate blend 50/50", "Original blend (50/50)")
    label = label.replace("RF + validation best (25/75)", "Proposed blend (25/75)")
    return label


def model_explanation(name):
    lower = name.lower()
    if "ensemble" in lower or "blend" in lower or "validation best" in lower or ("forest" in lower and "catboost" in lower):
        return ("The ensemble combines tree-model predictions. Random Forest averages trees trained "
                "on resampled data; CatBoost adds trees sequentially to reduce prediction loss. "
                "Validation chooses the combination before benchmark scoring.")
    if "catboost" in lower:
        objective = "absolute prediction error" if "mae" in lower else "the specified prediction loss"
        return (f"CatBoost builds small trees sequentially. Each new tree follows loss gradients "
                f"to reduce {objective}; the final prediction sums their contributions with a "
                "learning-rate adjustment. Validation controls model choice.")
    if "forest" in lower or lower.startswith("rf"):
        return ("Random Forest fits many trees to resampled training rows. A tree splits feature "
                "space into leaves and predicts from observations in its leaf. Averaging trees "
                "reduces prediction variability; validation controls depth and leaf size.")
    if "knn" in lower or "nearest" in lower:
        return ("kNN scales each input using training statistics, finds similar historical rows "
                "in feature space, and averages their sales. The number of neighbours controls "
                "the balance between local detail and stability.")
    if "ridge" in lower:
        return ("Ridge regression learns coefficients for scaled inputs by balancing squared "
                "prediction error against a penalty on coefficient size. The penalty reduces "
                "unstable coefficients; its strength is chosen by validation.")
    if "hist" in lower or "boost" in lower:
        return ("Gradient boosting builds trees sequentially to reduce a prediction objective. "
                "Each new tree changes the current prediction using loss gradients. Tree size "
                "and learning rate control the fit and are evaluated chronologically.")
    if "tree" in lower:
        return ("A regression tree splits inputs into increasingly similar groups. The predicted "
                "sales come from the reached terminal leaf. Depth and minimum leaf size control "
                "how much detail the tree learns from historical observations.")
    raise ValueError(f"Add an accurate main-method explanation for {name!r}")


def draw_comparison(p, rows, selected, x, y, width):
    # Preserve the analysis order: no ranking or selection using final-period outcomes.
    primary = [r for r in rows if "baseline" in r["model"].lower() or "weekday" in r["model"].lower() or "7-day" in r["model"].lower()]
    others = [r for r in rows if r not in primary]
    shown = primary + others
    if len(shown) > 10:
        scope = {"7-day mean", "Same weekday", "Teammate RF", "Teammate CatBoost", "Teammate blend 50/50",
                 "CatBoost MAE", "CatBoost history + IDs", "CatBoost history + IDs MAE", selected["model"]}
        shown = [row for row in shown if row["model"] in scope]
    label_w, val_w = 345, 100
    bar_x = x + label_w
    bar_w = width - label_w - 220
    max_mae = max(r["test_mae"] for r in shown) * 1.06
    p.text("METHOD", x, y, 17, MUTED, True)
    p.text("FINAL-PERIOD MAE", bar_x, y, 17, MUTED, True)
    p.text("VAL. MAE", x + width, y, 17, MUTED, True, "right")
    row_h = min(45, 360 / len(shown))
    top = y + 40
    for i, row in enumerate(shown):
        yy = top + i * row_h
        is_selected = row["model"] == selected["model"]
        colour = TEAL if is_selected else HexColor("#829D9E") if row in primary else HexColor("#425E68")
        p.paragraph(escape(model_label(row["model"])), x, yy + 1, label_w - 18,
                    size=17.5, leading=18.5, bold=is_selected, max_height=row_h)
        p.rect(bar_x, yy + 5, bar_w * row["test_mae"] / max_mae, 21, colour)
        p.text(f"{row['test_mae']:.4f}", bar_x + bar_w + 18, yy + 1, 19, INK, is_selected)
        p.text(f"{row['val_mae']:.4f}", x + width, yy + 1, 19, INK, is_selected, "right")
    p.paragraph(f"MAE is mean absolute prediction error; lower is better. {len(shown)} of {len(rows)} "
                "methods shown; all are in the notebook. Teal marks the validation selection.", x, top + len(shown) * row_h + 9,
                width, size=17, color=MUTED, max_height=55)


def draw_example(p, example, x, y, width):
    required = ["store_id", "product_id", "dt", "sales_t", "sales_mean7", "stockout_hours_t"]
    for key in required:
        if key not in example:
            raise ValueError(f"Real worked example missing {key}")
    date = str(example["dt"])[:10]
    p.text(f"Store {example['store_id']} · SKU {example['product_id']} · {date}", x, y, 23, INK, True)
    p.paragraph(f"Known history: yesterday's sales <b>{finite(example['sales_t'], 'sales_t'):.3f}</b>; "
                f"7-day mean <b>{finite(example['sales_mean7'], 'sales_mean7'):.3f}</b>; "
                f"yesterday's stockout hours <b>{finite(example['stockout_hours_t'], 'stockout_hours_t'):g}</b>.",
                x, y + 45, width, size=21, max_height=84)
    candidates = example["candidates"]
    # Keep the reference and selected alternative when a case has many candidates.
    ref = finite(example["reference_discount"], "reference_discount")
    chosen = first(example, "recommended_discount", "selected_discount")
    chosen = None if chosen is None else finite(chosen, "recommended_discount")
    ordered = sorted(candidates, key=lambda r: finite(first(r, "discount", "rate"), "candidate discount"), reverse=True)
    if len(ordered) > 3:
        kept = [r for r in ordered if float(first(r, "discount", "rate")) in (ref, chosen)]
        for r in ordered:
            if r not in kept and len(kept) < 3:
                kept.append(r)
        ordered = sorted(kept, key=lambda r: float(first(r, "discount", "rate")), reverse=True)
    yy = y + 114
    p.line(x, yy - 8, x + width, yy - 8)
    cols = [x, x + width * .24, x + width * .48, x + width * .75]
    for heading, xx in zip(["PRICE RATE", "PRED. SALES", "RATE × SALES", "HISTORY DAYS"], cols):
        p.text(heading, xx, yy, 16, MUTED, True)
    yy += 33
    for candidate in ordered:
        rate = finite(first(candidate, "discount", "rate"), "candidate rate")
        sales = finite(first(candidate, "pred_sales", "prediction"), "candidate forecast")
        value = finite(first(candidate, "pred_value", "value", default=rate * sales), "candidate value")
        support = first(candidate, "support_days", "support_count", "n_days")
        if support is None:
            raise ValueError("Worked example candidate needs historical support count")
        is_chosen = chosen is not None and abs(rate - chosen) < 1e-8
        color = TEAL if is_chosen else INK
        suffix = "*" if is_chosen else ""
        for val, xx in zip([f"{rate:.2f}{suffix}", f"{sales:.3f}", f"{value:.3f}", str(int(support))], cols):
            p.text(val, xx, yy, 23, color, is_chosen)
        yy += 41
    p.line(x, yy + 2, x + width, yy + 2)
    reason = str(first(example, "reason", "decision_reason", default="See saved policy evidence for the decision."))
    if chosen is not None:
        reason = f"{chosen:.2f} selected (*). " + reason
    p.paragraph(f"<b>Decision:</b> {escape(reason)}", x, yy + 17, width, size=19, max_height=82)
    p.paragraph("A rate of 0.90 means 10% off. Alternative-rate predictions are "
                "what-if estimates; their outcomes were not observed.", x, yy + 108, width,
                size=17, color=MUTED, max_height=54)


def draw_poster(output, rows, selection, policy, selected, baseline, example):
    p = Poster(output)
    p.rect(0, 0, WIDTH, 16, TEAL)
    p.text("HKU  /  MAIB7002  /  GROUP Q", MARGIN, 53, 21, TEAL, True)
    p.paragraph("Daily Discount Decisions<br/>for Perishable Food", MARGIN, 108, INNER,
                size=66, leading=73, bold=True, max_height=152)
    p.paragraph("Forecast tomorrow's sales. Compare supported discounts. Explain when to abstain.",
                MARGIN, 286, INNER, size=27, color=MUTED, max_height=76)
    p.text("Aashish Omprakash Pareek  ·  Yuanyuan Yin  ·  Lingyu Chen", MARGIN, 340, 21)

    p.section("01", "The business question", 409)
    p.paragraph("Can historical sales help a category manager decide <b>whether to discount "
                "a perishable SKU tomorrow, and by how much</b>, while limiting loss of predicted sales value?",
                MARGIN, 470, 855, size=25, leading=34, max_height=146)
    protocol = first(selection, "protocol", "data", default={})
    cohort = first(protocol, "n_series", "series", "cohort_size", default=first(selection, "n_series"))
    n_test = first(protocol, "test_rows", "n_test", default=first(selection, "test_rows"))
    if cohort is None or n_test is None:
        raise ValueError("selection.json needs protocol.n_series and protocol.test_rows")
    p.paragraph(f"<b>Public observations, no synthetic labels.</b><br/>FreshRetailNet-50K "
                f"(Dingdong): {int(cohort):,} retrospectively selected store-SKU series; {int(n_test):,} final-period daily cases. "
                "Target: observed normalized sales, which can be limited by stockouts.",
                1005, 470, WIDTH - 1005 - MARGIN, size=22, max_height=150)

    p.rect(MARGIN, 641, INNER, 72, PALE)
    p.paragraph("<b>TIME ORDER (2024)</b>  Train from 4 April | Validate 5 expanding weeks, 22 May–25 June | "
                "Benchmark 26 June–2 July<br/>One day ahead, repeatedly: only earlier observed "
                "days update history. The final benchmark was already viewed during prior development.",
                MARGIN + 18, 651, INNER - 36, size=18.5, leading=25, max_height=52)

    p.section("02", "From history to a supported decision", 749)
    steps = [
        ("Build the context", "Recent sales, past stockouts, calendar and planned discount. Shift history before computing rolling features."),
        ("Learn sales patterns", "Fit models on earlier dates. Compare all methods with the same evaluation rows and preprocessing rules."),
        ("Select by validation", "Choose the smallest mean weekly MAE. Use the final period for a labelled benchmark, not further selection."),
        ("Score price options", "Require 3 nearby training days. Predict sales at supported rates, keeping context fixed. Value proxy = rate × predicted sales."),
        ("Recommend or abstain", "Require 5% sales gain, 95% of reference value and model agreement. Prefer milder near-best discounts; otherwise retain or review."),
    ]
    gap = 30
    step_w = (INNER - 4 * gap) / 5
    for i, (title, body) in enumerate(steps):
        xx = MARGIN + i * (step_w + gap)
        p.text(str(i + 1), xx, 817, 29, TEAL, True)
        p.paragraph(title, xx, 859, step_w, size=22, bold=True, max_height=61)
        p.paragraph(body, xx, 930, step_w, size=18, leading=25, max_height=153)
        if i < 4:
            p.line(xx + step_w + 3, 838, xx + step_w + gap - 8, 838, TEAL, 2)
            p.line(xx + step_w + gap - 15, 832, xx + step_w + gap - 8, 838, TEAL, 2)
            p.line(xx + step_w + gap - 15, 844, xx + step_w + gap - 8, 838, TEAL, 2)

    p.section("03", "Forecast evidence", 1116)
    draw_comparison(p, rows, selected, MARGIN, 1181, 1030)
    side_x, side_w = 1165, WIDTH - MARGIN - 1165
    p.text("SELECTED USING VALIDATION", side_x, 1182, 17, TEAL, True)
    weights = selection.get("blend_weights", {}).get(selected["model"])
    if weights:
        selected_label = " + ".join(f"{weight:.0%} " + ("RF" if "RF" in name else "CatBoost" if "CatBoost" in name else name)
                                    for name, weight in weights.items())
    else:
        selected_label = model_label(selected["model"])
    p.paragraph(escape(selected_label), side_x, 1219, side_w,
                size=28, leading=34, bold=True, max_height=78)
    delta = (selected["test_mae"] / baseline["test_mae"] - 1) * 100
    adjective = "lower" if delta < 0 else "higher"
    original_blend = next(r for r in rows if r["model"] == "Teammate blend 50/50")
    blend_delta = (selected["test_mae"] / original_blend["test_mae"] - 1) * 100
    blend_adjective = "lower" if blend_delta < 0 else "higher"
    p.paragraph(f"<b>{selected['test_mae']:.4f}</b> final-period MAE<br/>"
                f"{abs(delta):.1f}% {adjective} than the 7-day baseline.<br/>"
                f"{abs(blend_delta):.1f}% {blend_adjective} than the original blend.<br/>"
                f"RMSE: {selected['test_rmse']:.4f}.", side_x, 1317, side_w,
                size=21, leading=30, max_height=133)
    explanation = model_explanation(selected["model"])
    if weights and any("history + IDs MAE" in name for name in weights):
        explanation = ("RF averages resampled regression trees. CatBoost adds trees sequentially "
                       "to reduce absolute error, using recent history and store/SKU categories. "
                       "The weighted mean combines their forecasts. RMSE additionally diagnoses large errors.")
    p.paragraph(explanation, side_x, 1459, side_w,
                size=17.5, leading=24, max_height=176)
    by_name = {row["model"]: row for row in rows}
    experiment = "Training and validation use the same dates and rows. Prediction gains do not establish a causal discount effect."
    if {"CatBoost history + IDs", "CatBoost history + IDs MAE"} <= set(by_name):
        original = by_name["CatBoost history + IDs"]["val_mae"]
        revised = by_name["CatBoost history + IDs MAE"]["val_mae"]
        experiment = ("<b>Controlled technical test:</b> with history/IDs fixed, aligning CatBoost's loss "
                      "with MAE is expected to reduce absolute errors. Validation MAE: "
                      f"<b>{original:.4f} (RMSE loss) vs. {revised:.4f} (MAE loss)</b>. "
                      "The measured change is predictive, not a causal pricing gain.")
    p.paragraph(experiment, MARGIN, 1645, INNER, size=19, leading=26, color=INK, max_height=82)

    p.section("04", "A traceable discount example", 1752, width=842)
    p.section("05", "Business use and failures", 1752, x=996, width=WIDTH - MARGIN - 996)
    draw_example(p, example, MARGIN, 1814, 842)
    right_x, right_w = 996, WIDTH - MARGIN - 996
    total = int(policy["rows"])
    counts = policy["status_counts"]
    discounts = int(counts.get("recommend_discount", 0))
    stock_reviews = int(counts.get("review_stockout", 0))
    other_reviews = total - discounts - int(counts.get("keep_full_price", 0)) - stock_reviews
    eligible = total - stock_reviews
    recommendation_share = discounts / eligible if eligible else 0
    p.paragraph(f"<b>Screen all {total:,} cases:</b> {discounts:,} recommend a discount scenario; "
                f"{stock_reviews:,} require supply review; {other_reviews:,} other cases need review/abstention. "
                f"<b>{recommendation_share:.1%}</b> of the {eligible:,} cases without a recent-stockout flag "
                "still suggest discounting. This high rate needs further investigation.",
                right_x, 1814, right_w, size=19, leading=26, max_height=150)
    transfer = policy.get("transfer")
    if transfer:
        newer, older, receipt = transfer["selected"], transfer["reference"], transfer["receipt"]
        text = (f"<b>Transfer stress check:</b> {int(receipt['evaluation_rows']):,} cases across "
                f"{int(receipt['new_stores']):,} new stores / {int(receipt['new_cities']):,} other cities. "
                f"Proposed vs. original blend: MAE <b>{float(newer['mae']):.4f} vs. {float(older['mae']):.4f}</b>; "
                f"RMSE <b>{float(newer['rmse']):.4f} vs. {float(older['rmse']):.4f}</b>. "
                "The improvement does not transfer here. We kept the model frozen; no retuning.")
        p.paragraph(text, right_x, 1971, right_w, size=19, leading=26, max_height=166)
        limits_y = 2148
        limits_size, limits_leading, limits_height = 17, 23, 72
        limits = ("<b>Limits:</b> discounts are observational; stockouts do not reveal inventory age "
                  "or waste. Rate × sales is a proxy, not measured revenue or profit. "
                  "Pricing gains require intervention evidence.")
    else:
        limits_y = 2004
        limits_size, limits_leading, limits_height = 21, 29, 203
        limits = ("<b>Important limits.</b> Historical discounts were not randomized. "
                  "Stockout history does not reveal stock age, tomorrow's inventory or food waste. "
                  "The proxy is not measured currency revenue or profit. A previously viewed "
                  "benchmark cannot confirm an untouched-test improvement.")
    p.paragraph(limits, right_x, limits_y, right_w, size=limits_size,
                leading=limits_leading, max_height=limits_height)

    p.line(MARGIN, 2236, WIDTH - MARGIN, 2236, thickness=1.5)
    p.paragraph("<b>Data:</b> Dingdong-Inc / FreshRetailNet-50K, CC BY 4.0, "
                "huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K. "
                "<b>Code:</b> github.com/Lisayinyy/MAIB_7002. "
                "<b>Reproduce:</b> see the accompanying notebook and Docker README.",
                MARGIN, 2252, INNER, size=14, leading=19, max_height=40)
    p.paragraph("Generative-AI assistance: code review, implementation and presentation drafting; "
                "automated checks and saved row-level evidence are documented in the project. "
                "Each member must confirm their contribution and verification statement in the notebook.",
                MARGIN, 2303, INNER, size=12.5, leading=18, color=MUTED, max_height=39)
    p.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "v2")
    parser.add_argument("--output", type=Path, default=ROOT / "poster" / "freshretail_group_q_A1.pdf")
    args = parser.parse_args()
    rows, selection, policy, selected, baseline, example, source_paths = load_evidence(args.results_dir)
    register_fonts()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    draw_poster(args.output, rows, selection, policy, selected, baseline, example)
    manifest = {
        "format": "A1 portrait, one page", "width_mm": 594, "height_mm": 841,
        "selected_model": selected["model"], "output": args.output.name,
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reportlab_version": reportlab.Version,
        "status": "generated; inspect rendered PDF before submission",
    }
    (args.output.parent / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
