#!/usr/bin/env python3
"""Build the single-page A1 poster from the final project's saved evidence.

Run with Python and reportlab installed. No training or model selection occurs.
All charts and text are vector objects; the QR code links to the final notebook.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

import reportlab
from reportlab.graphics import renderPDF
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics.shapes import Drawing
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf"
MM = 72 / 25.4
W, H = 594 * MM, 841 * MM
M = 22 * MM
RIGHT = W - M
INNER = W - 2 * M
INK = "#153B32"
TEAL = "#28735F"
SAGE = "#A6C8B8"
PALE = "#EAF1EB"
PAPER = "#FAF8F1"
WHITE = "#FFFFFF"
MUTED = "#52645D"
LINE = "#CBD5CA"
AMBER = "#F0C35B"
RUST = "#AB5B33"
LIGHT = "#E5E9E2"
PIN = "a7e26dd979af88b4c14f98a7f029f6a64bffaedf"
NOTEBOOK = f"https://github.com/Lisayinyy/MAIB_7002/blob/{PIN}/discount_sales_prediction.ipynb"
QR_URL = "https://github.com/Lisayinyy/MAIB_7002/tree/pricing-ml-ap"
DATA_URL = "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K"


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def register_fonts():
    base = Path(reportlab.__file__).parent / "fonts"
    system = Path("/System/Library/Fonts/Supplemental")
    fonts = {
        "Body": system / "Arial.ttf",
        "Bold": system / "Arial Bold.ttf",
        "Display": system / "Georgia.ttf",
    }
    fallback = {"Body": "Vera.ttf", "Bold": "VeraBd.ttf", "Display": "VeraBd.ttf"}
    for name, path in fonts.items():
        pdfmetrics.registerFont(TTFont(name, str(path if path.is_file() else base / fallback[name])))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold", italic="Body", boldItalic="Bold")


class Poster:
    def __init__(self, path):
        self.c = canvas.Canvas(str(path), pagesize=(W, H), pageCompression=1)
        self.c.setTitle("Should we discount tomorrow? | MAIB7002 Group Q")
        self.c.setAuthor("Aashish Omprakash Pareek; Yuanyuan Yin; Lingyu Chen")
        self.c.setSubject("A1 portrait poster: FreshRetailNet forecasting and daily discount decision support")
        self.text_boxes = []

    def rect(self, x, top, w, h, color):
        self.c.setFillColor(HexColor(color))
        self.c.rect(x, H - top - h, w, h, stroke=0, fill=1)

    def line(self, x1, y1, x2, y2, color=LINE, weight=1.2):
        self.c.setStrokeColor(HexColor(color))
        self.c.setLineWidth(weight)
        self.c.line(x1, H - y1, x2, H - y2)

    def dot(self, x, top, r, color):
        self.c.setFillColor(HexColor(color))
        self.c.circle(x, H - top, r, stroke=0, fill=1)

    def text(self, value, x, top, size=24, color=INK, font="Body", align="left"):
        value = str(value)
        width = pdfmetrics.stringWidth(value, font, size)
        left = x if align == "left" else x - width if align == "right" else x - width / 2
        if left < 0 or left + width > W + .1 or top + size > H:
            raise ValueError(f"Text leaves page: {value}")
        self.c.setFillColor(HexColor(color))
        self.c.setFont(font, size)
        draw = self.c.drawString if align == "left" else self.c.drawRightString if align == "right" else self.c.drawCentredString
        draw(x, H - top - size * .85, value)
        self.text_boxes.append(dict(text=value, x=left, y=top, width=width, height=size))

    def para(self, text, x, top, width, size=25, color=INK, leading=None, max_h=None, bold=False):
        style = ParagraphStyle("body", fontName="Bold" if bold else "Body", fontSize=size,
                               leading=leading or size * 1.24, textColor=HexColor(color))
        p = Paragraph(text, style)
        _, h = p.wrap(width, H)
        if max_h is not None and h > max_h + .2:
            raise ValueError(f"Text exceeds {max_h:.1f}pt slot by {h-max_h:.1f}: {text}")
        if top + h > H - 12:
            raise ValueError(f"Paragraph leaves page: {text}")
        p.drawOn(self.c, x, H - top - h)
        self.text_boxes.append(dict(text=text, x=x, y=top, width=width, height=h))
        return h

    def arrow(self, x1, y, x2, color=TEAL, weight=2):
        self.line(x1, y, x2, y, color, weight)
        self.line(x2 - 8, y - 5, x2, y, color, weight)
        self.line(x2 - 8, y + 5, x2, y, color, weight)

    def section(self, number, title, x, top, size=33):
        self.text(number, x, top + 2, size=20, font="Bold", color=TEAL)
        self.text(title, x + 44, top, size=size, font="Bold")

    def qr(self, url, x, top, size):
        q = QrCodeWidget(url, barFillColor=HexColor(INK), barBorder=4)
        bounds = q.getBounds()
        dw, dh = bounds[2] - bounds[0], bounds[3] - bounds[1]
        d = Drawing(size, size, transform=[size/dw, 0, 0, size/dh, 0, 0])
        d.add(q)
        renderPDF.draw(d, self.c, x, H - top - size)
        self.c.linkURL(url, (x, H-top-size, x+size, H-top), relative=0, thickness=0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    register_fonts()
    score_path = ROOT / "results/test_summary.csv"
    weight_path = ROOT / "results/ensemble_weight_sweep.csv"
    case_path = ROOT / "results/visuals/one_product_discount_decision.csv"
    cohort_path = ROOT / "results/visuals/sample_selection.csv"
    policy_path = ROOT / "results/visuals/recommendation_distribution.csv"
    scores = {r[""]: {k: float(v) for k, v in r.items() if k} for r in read_csv(score_path)}
    weights = read_csv(weight_path)
    blend = scores["RF + CatBoost (50/50)"]
    baseline = scores["Baseline (7-day mean)"]
    improvement = 100 * (1 - blend["mae"] / baseline["mae"])
    case = read_csv(case_path)
    cohort = read_csv(cohort_path)
    policy = read_csv(policy_path)
    reference = next(r for r in case if float(r["discount"]) == 1.0)
    chosen = next(r for r in case if r["chosen"] == "True")
    assert chosen["eligible"] == "True" and float(chosen["discount"]) == .95
    assert int(cohort[0]["histories"]) == 50000 and int(cohort[-1]["histories"]) == 312
    assert len([r for r in case if r["eligible"] == "True"]) == 5
    near_optimal = [float(r["mae"]) for r in weights if .4 <= float(r["rf_weight"]) <= .7]
    policy_shares = {r["strategy"]: float(r["discount_share_percent"]) for r in policy}
    full_sales = float(reference["pred_sales"])
    discount_sales = float(chosen["pred_sales"])
    discount_value = float(chosen["pred_value"])
    # These source files are included with the visual edition, and its model,
    # feature and scenario code matches the pinned final branch byte-for-byte.
    assert abs(blend["mae"] - .28923224366438466) < 1e-10
    assert len(case) == 8
    path = OUT / "freshretail_a1_poster.pdf"
    p = Poster(path)
    p.rect(0, 0, W, H, PAPER)

    # A quiet editorial header with one large question rather than a dashboard.
    p.rect(M, 25*MM, 13, 15, AMBER)
    p.text("HKU  /  MAIB7002  /  GROUP Q", M+27, 24.7*MM, size=20, font="Bold")
    p.text("FRESH FOOD, DAILY DECISIONS", RIGHT, 24.7*MM, size=18, color=MUTED, align="right")
    p.text("Should we discount", M-2, 40*MM, size=89, font="Bold")
    p.text("tomorrow?", M-3, 74*MM, size=96, font="Display", color=TEAL)
    p.rect(M+569, 107.3*MM, 495, 5, AMBER)
    p.para("Forecasting next-day sales to help a Dingdong category manager<br/>decide <b>whether to discount and by how much.</b>",
           M, 114*MM, INNER, size=28, leading=35, max_h=76)
    p.text("Aashish Omprakash Pareek  |  Yuanyuan Yin  |  Lingyu Chen", M, 143*MM, size=21, color=MUTED)
    p.line(M, 158*MM, RIGHT, 158*MM, color=INK, weight=2)

    # 01: why these data, with a compact selection path.
    p.section("01", "Start with a decision the data can support", M, 169*MM)
    p.para("Fresh food loses value quickly. Too little discounting can leave stock unsold; too much gives up sales value. We study daily discount choices using public retail histories.",
           M, 189*MM, 244*MM, size=25, max_h=125)
    rx = 299*MM
    p.text("FreshRetailNet-50K", rx, 189*MM, size=29, font="Bold")
    p.para("Sales, discounts, stockout hours and calendar information match the inputs this decision needs.",
           rx, 204*MM, RIGHT-rx, size=24, max_h=87)
    # Native vector data selection diagram, exact denominators labelled.
    xs = [M, 167*MM, 318*MM, 469*MM]
    vals = ["50,000", "16,684", "5", "312"]
    labels = ["store-product histories", "eligible histories", "selected stores", "selected histories"]
    for i, (x, val, label) in enumerate(zip(xs, vals, labels)):
        p.text(val, x, 241*MM, size=45, font="Bold", color=TEAL if i == 3 else INK)
        p.text(label, x, 261*MM, size=20, color=MUTED)
        if i < 3:
            p.arrow(x + 105*MM, 251*MM, xs[i+1]-10*MM, color=TEAL)
    p.para("Eligibility: 15-85% discounted days (rate &lt; 0.95); 6+ discount-state changes; fewer than 20% zero-sales days.<br/>Choose the five stores with the most eligible histories. One city; cohort selected using the full 90-day training file.",
           M, 277*MM, INNER, size=20.5, leading=27, max_h=60)
    p.line(M, 306*MM, RIGHT, 306*MM)

    # 02: a readable method diagram with learning mechanics, inputs and output.
    p.section("02", "Predict sales first. Then choose a price.", M, 318*MM)
    col1, col2, col3 = M, 210*MM, 402*MM
    colw = 168*MM
    p.text("PAST + PLANNED INPUTS", col1, 340*MM, size=19, font="Bold", color=TEAL)
    p.para("Recent sales and stockouts,<br/>past discounts and activity,<br/>tomorrow's calendar and<br/><b>candidate price rate.</b>",
           col1, 354*MM, colw, size=24, leading=30, max_h=130)
    p.text("TWO LEARNERS, ONE FORECAST", col2, 340*MM, size=19, font="Bold", color=TEAL)
    p.para("<b>Random Forest</b> averages trees<br/>fit to bootstrapped samples.<br/><b>CatBoost</b> corrects errors sequentially.<br/><b>Final = 0.5 RF + 0.5 CatBoost.</b>",
           col2, 354*MM, colw+10, size=24, leading=30, max_h=130)
    p.text("SUPPORTED PRICE CHOICE", col3, 340*MM, size=19, font="Bold", color=TEAL)
    p.para("Compare eight price rates.<br/>Keep rates with historical<br/>support, apply stock rules,<br/>then maximize <b>rate x forecast.</b>",
           col3, 354*MM, RIGHT-col3, size=24, leading=30, max_h=130)
    p.arrow(187*MM, 377*MM, 202*MM, color=INK)
    p.arrow(380*MM, 377*MM, 395*MM, color=INK)
    p.para("The target is normalized observed sales, not units or unconstrained demand. Rate 0.95 means 5% off.<br/>Historical support: at least 3 training days within 0.025 of a rate. The default value proxy assumes zero unit cost.",
           M, 408*MM, INNER, size=20.5, leading=27, max_h=56)
    # Evaluation band doubles as a left-to-right chronology.
    p.rect(M, 438*MM, INNER, 27*MM, PALE)
    p.text("EVALUATION", M+18, 442*MM, size=17, font="Bold", color=TEAL)
    p.text("5 expanding weekly validation folds", M+18, 452*MM, size=22, font="Bold")
    p.arrow(213*MM, 455.5*MM, 244*MM)
    p.text("Refit on 25,896 rows", 254*MM, 452*MM, size=22, font="Bold")
    p.arrow(356*MM, 455.5*MM, 381*MM)
    p.text("2,184 next-day cases", 392*MM, 452*MM, size=22, font="Bold")
    p.text("26 Jun - 2 Jul 2024", RIGHT-15, 441*MM, size=17, color=MUTED, align="right")

    # 03/04: paired findings, one about forecast accuracy and one about decisions.
    left_w, right_x = 255*MM, 315*MM
    p.section("03", "A better sales forecast", M, 482*MM, size=31)
    p.section("04", "Pricing still needs rules", right_x, 482*MM, size=31)
    p.text(f"{improvement:.1f}%", M, 503*MM, size=55, font="Bold", color=TEAL)
    offset = pdfmetrics.stringWidth(f"{improvement:.1f}%", "Bold", 55) + 18
    p.para("lower final-week MAE<br/>than the seven-day mean", M+offset, 505*MM, left_w-offset, size=23, max_h=64)
    p.text("Mean absolute error (MAE); lower is better", M, 532*MM, size=20, color=MUTED)
    models = [
        ("7-day mean", "Baseline (7-day mean)", SAGE),
        ("Same weekday", "Same weekday last week", LIGHT),
        ("kNN", "kNN (k=25)", SAGE),
        ("Random Forest", "Random forest", SAGE),
        ("CatBoost", "CatBoost", SAGE),
        ("50/50 ensemble", "RF + CatBoost (50/50)", TEAL),
    ]
    bar_x, bar_w, row_gap = M + 212, 365, 32
    for i, (label, key, color) in enumerate(models):
        y = 548*MM + i*row_gap
        value = scores[key]["mae"]
        p.text(label, M, y+1, size=21, font="Bold" if i==5 else "Body")
        p.rect(bar_x, y, bar_w*value/.42, 23, color)
        p.text(f"{value:.4f}", bar_x+bar_w*value/.42+12, y+1, size=21, font="Bold" if i==5 else "Body")
    p.line(bar_x, 548*MM-3, bar_x, 548*MM+5*row_gap+27, LINE)
    p.text(f"Root mean square error (RMSE): {blend['rmse']:.4f}", M, 619*MM, size=21, font="Bold")
    p.para(f"Weight experiment: RF shares of 40-70% give validation MAE {min(near_optimal):.5f}-{max(near_optimal):.5f}. We keep 50/50 for simplicity; it adds little over CatBoost alone.",
           M, 634*MM, left_w, size=21, leading=27, max_h=91)

    p.para("Maximizing predicted value suggests a discount almost every day. Assumed stock rules change that behaviour.",
           right_x, 504*MM, RIGHT-right_x, size=24, leading=30, max_h=92)
    p.text("Share of cases recommending a discount", right_x, 543*MM, size=20, color=MUTED)
    policy_vals = [("Supported prices only", policy_shares["Supported model only"]), ("+ Stock rules", policy_shares["With stock rules"])]
    policy_w = RIGHT-right_x-115
    for i, (label, value) in enumerate(policy_vals):
        y = (559+i*25)*MM
        p.text(label, right_x, y, size=22, font="Bold" if i else "Body")
        p.rect(right_x, y+30, policy_w, 27, LIGHT)
        p.rect(right_x, y+30, policy_w*value/100, 27, AMBER if i == 0 else TEAL)
        p.text(f"{value:.1f}%", right_x+policy_w+14, y+28, size=29, font="Bold")
    p.para("<b>Stockout yesterday:</b> keep full price.<br/><b>No stockout in 7 days:</b> require a supported discount.<br/><b>Otherwise:</b> let the model compare eligible rates.",
           right_x, 615*MM, RIGHT-right_x, size=21, leading=28, max_h=112)
    p.para("Same 2,184 cases. These are policy assumptions, not evidence of inventory age or higher revenue.",
           right_x, 661*MM, RIGHT-right_x, size=20, leading=26, max_h=60)

    # 05: a grounded example; no cherry-picked gains or synthetic record.
    p.rect(M, 691*MM, INNER, 69*MM, INK)
    p.text("05  ONE PRODUCT, ONE DAY", M+24, 699*MM, size=18, color=SAGE, font="Bold")
    p.text("5% off", M+22, 714*MM, size=53, color=AMBER, font="Bold")
    p.text("recommended discount", M+25, 738*MM, size=19, color=WHITE)
    exx = 156*MM
    p.text("Store 18 / Product 11 / 26 June 2024", exx, 700*MM, size=24, color=WHITE, font="Bold")
    p.para("Yesterday: no stockout. Two stockout days in the past week.<br/>The model compares five historically supported rates.",
           exx, 713*MM, 254*MM, size=21, color=WHITE, leading=27, max_h=60)
    p.text("Full price", exx, 740*MM, size=22, color=SAGE)
    p.text(f"{full_sales:.2f}", exx+142, 740*MM, size=25, color=WHITE, font="Bold")
    p.arrow(exx+211, 745*MM, exx+261, color=AMBER)
    p.text("5% off", exx+280, 740*MM, size=22, color=SAGE)
    p.text(f"{discount_sales:.2f}", exx+382, 740*MM, size=25, color=WHITE, font="Bold")
    p.text("predicted sales", exx+459, 741*MM, size=18, color=SAGE)
    p.para(f"Value proxy<br/><b>0.95 x {discount_sales:.2f} = {discount_value:.2f}</b><br/>vs. {full_sales:.2f} at full price", 438*MM, 713*MM, 121*MM, size=23, leading=29, color=WHITE, max_h=93)
    p.text("Illustrative model scenario; not observed sales uplift.", exx, 753*MM, size=15.5, color=SAGE)

    # Close with an honest business interpretation and sources, never uplift claims.
    p.text("A decision aid, with business impact still to test.", M, 773*MM, size=31, font="Bold")
    p.para("Discounts are observational and stockouts can hide demand. We have no stock-age or waste outcomes; the final week was reused during development. Next: actual inventory/cost data and a controlled store pilot.",
           M, 792*MM, 446*MM, size=21, leading=27, max_h=80)
    p.text("Data: FreshRetailNet-50K / Dingdong / CC BY 4.0", M, 826*MM, size=15.5, color=MUTED)
    p.text("Final notebook: pricing-ml-ap @ a7e26dd  |  Code + Docker", 231*MM, 826*MM, size=15.5, color=MUTED)
    p.qr(QR_URL, 524*MM, 779*MM, 44*MM)
    p.text("EXPLORE THE PROJECT", 546*MM, 825*MM, size=13.5, color=TEAL, font="Bold", align="center")
    p.c.linkURL(NOTEBOOK, (M, H-132*MM, RIGHT, H-40*MM), relative=0, thickness=0)
    p.c.linkURL(DATA_URL, (M, H-834*MM, 216*MM, H-824*MM), relative=0, thickness=0)
    p.c.save()
    temp = ROOT / "tmp" / "pdfs"
    temp.mkdir(parents=True, exist_ok=True)
    (temp / "poster_layout.json").write_text(json.dumps(p.text_boxes, indent=2))
    evidence = {
        "final_notebook_commit": PIN,
        "final_notebook_url": NOTEBOOK,
        "paper": {"width_mm": 594, "height_mm": 841, "orientation": "portrait", "pages": 1},
        "group": "Q",
        "members": ["Aashish Omprakash Pareek", "Yuanyuan Yin", "Lingyu Chen"],
        "mae": blend["mae"], "rmse": blend["rmse"], "baseline_mae": baseline["mae"],
        "mae_reduction_percent": improvement,
        "selection": "50/50 for simplicity within a flat validation weight range; not the exact minimum",
        "source_file_hashes": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in [score_path, weight_path, case_path, cohort_path, policy_path]},
        "poster_pdf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "claims": "Forecast accuracy and model scenarios only; no causal sales, revenue or waste benefit established.",
        "example": "First date/store/product with model-decides state and at least two supported rates including full price.",
    }
    (OUT / "poster_evidence.json").write_text(json.dumps(evidence, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
