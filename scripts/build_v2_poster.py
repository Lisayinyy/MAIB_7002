#!/usr/bin/env python3
"""Build the evidence-backed, single-page A1 poster from saved V2 results.

This is a presentation step, not a training or model-selection step. It never
changes result files and refuses to invent missing scores or an example.
Dependencies: reportlab (4.4.9 verified); optional pypdf for the independent audit.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path
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
    for name in ("recommendations.csv", "discount_response.csv", "test_predictions.csv", "validation_predictions.csv", "experiment_plan.json", "story_evidence.json"):
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
    # Read observed history from verified row-level CSVs; no model dependencies are needed.
    selection["experiment_plan"] = json.loads((results / "experiment_plan.json").read_text())
    story = json.loads((results / "story_evidence.json").read_text())
    selection["story_evidence"] = story
    with (results / "test_predictions.csv").open(newline="") as source:
        test_predictions = list(csv.DictReader(source))
    with (results / "validation_predictions.csv").open(newline="") as source:
        validation_predictions = list(csv.DictReader(source))
    keyed_predictions = {key(r): r for r in validation_predictions + test_predictions}
    previous_date = (date.fromisoformat(chosen["dt"][:10]) - timedelta(days=1)).isoformat()
    prior_key = (int(chosen["store_id"]), int(chosen["product_id"]), previous_date)
    if prior_key not in keyed_predictions:
        raise ValueError("The real worked example's previous-day observed sales are unavailable")
    example_features = {"store_id": int(chosen["store_id"]), "product_id": int(chosen["product_id"]),
        "dt": chosen["dt"][:10], "sales_t": finite(keyed_predictions[prior_key]["target_sales"], "previous-day sales"),
        "history_date": previous_date}
    selection["protocol"] = {"n_series": story["selection_funnel"]["selected_series_in_top_five"],
                              "test_rows": len(test_predictions)}
    recommended = chosen["recommended_rate"]
    example = example_features | {
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


def rounded(p, x, y, w, h, radius, color, stroke=None):
    c = p.canvas
    c.setFillColor(color)
    if stroke:
        c.setStrokeColor(stroke)
    c.roundRect(x, HEIGHT-y-h, w, h, radius, fill=1, stroke=bool(stroke))


def circle(p, x, y, radius, color, stroke=None, line_width=2):
    c = p.canvas
    c.setFillColor(color)
    if stroke:
        c.setStrokeColor(stroke)
        c.setLineWidth(line_width)
    c.circle(x, HEIGHT-y, radius, fill=1, stroke=bool(stroke))


def polygon(p, vertices, color, stroke=None, thickness=2):
    c = p.canvas
    path = c.beginPath()
    path.moveTo(vertices[0][0], HEIGHT-vertices[0][1])
    for x,y in vertices[1:]:
        path.lineTo(x,HEIGHT-y)
    path.close()
    c.setFillColor(color)
    if stroke:
        c.setStrokeColor(stroke)
        c.setLineWidth(thickness)
    c.drawPath(path, fill=1, stroke=bool(stroke))


def arrow(p, x1, y, x2, color=TEAL, thickness=3):
    p.line(x1,y,x2-10,y,color,thickness)
    polygon(p,[(x2-15,y-8),(x2,y),(x2-15,y+8)],color)


def icon(p, kind, x, y, scale=1, color=TEAL):
    """Small original vector illustrations; no external media or private assets."""
    def ln(a,b,c,d):
        p.line(x+a*scale,y+b*scale,x+c*scale,y+d*scale,color,3*scale)
    if kind=='history':
        for i,h in enumerate([25,46,37,64]):
            p.rect(x+(8+i*21)*scale,y+(78-h)*scale,12*scale,h*scale,color)
        ln(0,84,95,84)
        ln(1,4,1,84)
    elif kind=='trees':
        for offset in [0,53]:
            ln(offset+21,14,offset+21,33)
            ln(offset+21,33,offset+5,50)
            ln(offset+21,33,offset+38,50)
            ln(offset+5,50,offset+5,67)
            ln(offset+38,50,offset+38,67)
            for a,b in [(offset+21,10),(offset+5,72),(offset+38,72)]:
                circle(p,x+a*scale,y+b*scale,6*scale,color)
    elif kind=='tag':
        polygon(p,[(x,y+18*scale),(x+56*scale,y+18*scale),(x+94*scale,y+51*scale),
                   (x+56*scale,y+84*scale),(x,y+84*scale)],PALE,color,3*scale)
        circle(p,x+69*scale,y+51*scale,6*scale,color)
        p.text('%',x+16*scale,y+24*scale,36*scale,color,True)
    elif kind=='decision':
        ln(0,49,23,49)
        ln(23,49,52,20)
        ln(23,49,52,79)
        circle(p,x+70*scale,y+18*scale,19*scale,PALE,color)
        circle(p,x+70*scale,y+80*scale,19*scale,HexColor('#FFF1E8'),ORANGE)
        ln(60,18,67,25)
        ln(67,25,81,9)
        p.text('?',x+63*scale,y+64*scale,24*scale,ORANGE,True)
    elif kind=='store':
        p.rect(x+7*scale,y+39*scale,78*scale,44*scale,PALE)
        polygon(p,[(x,y+36*scale),(x+15*scale,y+13*scale),(x+78*scale,y+13*scale),
                   (x+94*scale,y+36*scale)],color)
        p.rect(x+42*scale,y+52*scale,22*scale,31*scale,color)
        p.rect(x+17*scale,y+49*scale,15*scale,15*scale,color)


def draw_hero(p):
    white=HexColor('#FFFFFF'); light=HexColor('#CBE8DF'); mint=HexColor('#93D7BF')
    p.rect(0,0,WIDTH,381,INK)
    p.text('HKU  /  MAIB7002  /  GROUP Q',MARGIN,42,21,mint,True)
    p.paragraph('Should we discount<br/>tomorrow?',MARGIN,97,1080,
                size=82,leading=87,bold=True,color=white,max_height=180)
    p.paragraph('Daily decision support for perishable food',MARGIN,288,1030,
                size=29,color=light,max_height=44)
    p.text('Aashish Omprakash Pareek  |  Yuanyuan Yin  |  Lingyu Chen',MARGIN,341,20,white)
    # A basket of produce beside the business question, drawn as scalable vector shapes.
    x,y=1240,100
    circle(p,x+110,y+63,54,HexColor('#EAC061'))
    circle(p,x+191,y+73,57,HexColor('#E78354'))
    circle(p,x+39,y+94,42,HexColor('#97BA7B'))
    polygon(p,[(x+94,y+5),(x+135,y-17),(x+125,y+19)],mint)
    polygon(p,[(x+176,y+13),(x+221,y-5),(x+197,y+32)],HexColor('#ADC993'))
    polygon(p,[(x-9,y+114),(x+258,y+114),(x+230,y+208),(x+16,y+208)],HexColor('#E1D7BF'))
    for yy in [138,167,193]:
        p.line(x+17,y+yy,x+231,y+yy,INK,4)
    p.line(x+64,y+122,x+72,y+199,INK,3)
    p.line(x+153,y+122,x+153,y+199,INK,3)
    polygon(p,[(x+245,y+31),(x+312,y+31),(x+337,y+58),(x+312,y+85),(x+245,y+85)],mint)
    circle(p,x+316,y+58,4,INK)
    p.text('?',x+267,y+31,36,INK,True)
    p.line(x+317,y+58,x+293,y+114,light,2)


def draw_funnel(p,story):
    funnel=story['selection_funnel']
    p.text('01',MARGIN,542,23,TEAL,True)
    p.text('Why these five stores?',MARGIN+53,537,34,INK,True)
    p.paragraph('Use products with enough discount variation to study, then choose the stores with the most eligible products.',
                MARGIN,596,INNER,25,leading=33,max_height=67)
    starts=[MARGIN,490,900,1290]
    stages=[(f"{funnel['all_series']:,}",'public store-SKU histories'),
            (f"{funnel['after_all_three_filters']:,}",'eligible after filtering'),
            ('5','stores with most eligible series'),
            (str(funnel['selected_series_in_top_five']),'series in our benchmark')]
    for i,((value,label),xx) in enumerate(zip(stages,starts)):
        p.text(value,xx,650,57,TEAL,True)
        p.paragraph(label,xx,720,280,23,leading=30,max_height=66)
        if i<3:
            arrow(p,xx+285,696,starts[i+1]-37,RULE,3)
    p.line(MARGIN,793,WIDTH-MARGIN,793,RULE,1)
    p.paragraph('<b>Eligibility:</b> 15-85% discounted days, 6+ discount-state changes, and fewer than 20% zero-sales days.',
                MARGIN,813,INNER,22,leading=29,max_height=60)
    ids=' / '.join(str(r['store_id']) for r in funnel['top_five_stores_ranked'])
    p.paragraph(f'<b>Selected stores:</b> {ids}. One city; chosen retrospectively using the full training period.',
                MARGIN,853,INNER,20,leading=27,color=MUTED,max_height=54)


def draw_flow(p,selection):
    p.text('02',MARGIN,927,23,TEAL,True)
    p.paragraph('Forecast first. Then make a cautious decision.',MARGIN+53,917,INNER-55,34,bold=True,max_height=48)
    titles=['Read the history','Predict tomorrow','Compare discounts','Recommend or review']
    bodies=['Recent sales, past stockouts, calendar and a planned price rate.',
            'Random Forest averages trees (25%). CatBoost adds trees (75%), using MAE loss, richer history and store/SKU inputs.',
            'Only rates with historical support. Compare sales and price rate × sales.',
            'Require model agreement. If evidence or supply is uncertain, ask the manager to review.']
    icons=['history','trees','tag','decision']
    gap=46;w=(INNER-3*gap)/4
    for i in range(4):
        xx=MARGIN+i*(w+gap)
        icon(p,icons[i],xx+8,989,1.02)
        if i<3: arrow(p,xx+155,1032,xx+w+gap-27,RULE,3)
        p.paragraph(titles[i],xx,1095,w,25,bold=True,max_height=66)
        p.paragraph(bodies[i],xx,1164,w,23,leading=30,max_height=154)
    # This states the decision objective in ordinary language, before reporting model scores.
    p.rect(MARGIN,1327,INNER,72,PALE)
    p.paragraph('<b>The rule:</b> seek at least 5% more predicted sales while keeping at least 95% of full-price sales value. '
                'These are decision thresholds, not guarantees.',MARGIN+20,1341,INNER-40,22,leading=28,max_height=58)


def draw_results(p,rows,selected,baseline):
    lookup={r['model']:r for r in rows}
    original=lookup['Teammate blend 50/50']
    p.text('03',MARGIN,1449,23,TEAL,True)
    p.text('More accurate on the five-store benchmark',MARGIN+53,1444,34,INK,True)
    chart_rows=[('7-day sales average',baseline),('Original 50/50 blend',original),('Proposed 25/75 blend',selected)]
    label_x=MARGIN;bar_x=445;bar_w=586;value_x=1052
    top=1515
    for i,(name,row) in enumerate(chart_rows):
        yy=top+58*i
        col=[HexColor('#AFBEBD'),HexColor('#668D8A'),TEAL][i]
        p.text(name,label_x,yy+1,25,INK,i==2)
        p.rect(bar_x,yy+4,bar_w*(row['test_mae']/.35),32,col)
        p.text(f"{row['test_mae']:.4f}",value_x,yy-1,26,TEAL if i==2 else MUTED,i==2)
    p.paragraph('Final-period MAE: mean absolute error in the dataset sales scale; lower is better. '
                'Same 2,184 daily cases, 26 June-2 July 2024.',MARGIN,1695,1030,21,leading=28,color=MUTED,max_height=64)
    reduction=(1-selected['test_mae']/baseline['test_mae'])*100
    p.text(f'{reduction:.1f}%',1222,1507,72,TEAL,True)
    p.paragraph('lower prediction error<br/>than the simple baseline',1222,1601,WIDTH-MARGIN-1222,24,leading=32,max_height=104)
    p.paragraph('Selected using the lowest mean MAE across five chronological validation weeks.',
                1222,1692,WIDTH-MARGIN-1222,20,leading=27,color=MUTED,max_height=84)
    rmse=lookup['CatBoost history + IDs']['val_mae']
    mae=lookup['CatBoost history + IDs MAE']['val_mae']
    p.paragraph(f'<b>One technical experiment:</b> keep the extended features fixed and change CatBoost\'s training loss. '
                f'MAE loss reduced validation MAE from <b>{rmse:.4f} to {mae:.4f}</b> versus RMSE loss.',
                MARGIN,1789,INNER,22,leading=29,max_height=61)


def draw_case(p,example):
    # Every shown number comes from the selected real row and its saved what-if response.
    rates={round(c['discount'],2):c for c in example['candidates']}
    chosen=float(example['recommended_discount'])
    full=rates[1.0]; rec=rates[round(chosen,2)]
    off=(1-chosen)*100
    p.text('04',MARGIN,1890,23,TEAL,True)
    p.text('What would the manager actually see?',MARGIN+53,1885,34,INK,True)
    # Large price tag anchors the worked example visually.
    tag_x=MARGIN;tag_y=1959
    polygon(p,[(tag_x,tag_y),(tag_x+247,tag_y),(tag_x+297,tag_y+93),
               (tag_x+247,tag_y+186),(tag_x,tag_y+186)],TEAL)
    p.text(f'{off:.0f}%',tag_x+25,tag_y+28,67,HexColor('#FFFFFF'),True)
    p.text('OFF',tag_x+28,tag_y+112,34,HexColor('#FFFFFF'),True)
    circle(p,tag_x+261,tag_y+92,10,HexColor('#FFFFFF'))
    x=432
    p.paragraph(f'<b>Store {example["store_id"]}, SKU {example["product_id"]}, {str(example["dt"])[:10]}</b><br/>'
                f'Yesterday\'s observed sales: {float(example["sales_t"]):.3f}. Other context is held fixed.',
                x,1951,INNER-360,23,leading=30,max_height=68)
    p.paragraph(f'Full price: <b>{full["pred_sales"]:.3f}</b> predicted sales &nbsp; | &nbsp; '
                f'{off:.0f}% off: <b>{rec["pred_sales"]:.3f}</b> predicted sales',
                x,2034,INNER-360,25,leading=32,max_height=64)
    p.paragraph(f'<b>Sales-value proxy:</b> {chosen:.2f} × {rec["pred_sales"]:.3f} = {rec["pred_value"]:.3f}. '
                'The candidate passes support, value and model-agreement checks.',
                x,2084,INNER-360,22,leading=29,max_height=65)
    p.paragraph('An illustrative model scenario, not an observed response to changing the price.',
                MARGIN,2170,INNER,21,leading=27,color=MUTED,max_height=30)


def draw_poster(output,rows,selection,policy,selected,baseline,example):
    """Business-first editorial poster: pictures and decisions before score tables."""
    # A1 remains exactly one page. Geometry uses top-origin points.
    p=Poster(output)
    story=selection['story_evidence']
    draw_hero(p)
    p.paragraph('A category manager must choose tomorrow\'s discount. We use public sales histories '
                'to compare supported price options while limiting loss of predicted sales value.',
                MARGIN,430,INNER,28,leading=37,max_height=83)
    draw_funnel(p,story)
    draw_flow(p,selection)
    draw_results(p,rows,selected,baseline)
    draw_case(p,example)
    transfer=policy.get('transfer')
    p.line(MARGIN,2220,WIDTH-MARGIN,2220,RULE,1.2)
    if transfer:
        new,old,receipt=transfer['selected'],transfer['reference'],transfer['receipt']
        limits=(f"<b>Where evidence stops:</b> on {int(receipt['evaluation_rows'])} cases from {int(receipt['new_stores'])} new stores, "
                f"MAE was {float(new['mae']):.4f} vs. {float(old['mae']):.4f} for the original blend: no improvement; the models were not retuned. "
                "The main benchmark was already viewed; 96.9% of cases without recent stockouts still recommend discounts. Historical discounts are not causal evidence, and stockouts do not reveal inventory age or waste.")
    else:
        limits="<b>Where evidence stops:</b> the benchmark was already viewed. Historical discounts do not establish causal effects; stockouts do not reveal inventory age or waste."
    p.paragraph(limits,MARGIN,2240,INNER,19,leading=25,max_height=76)
    p.paragraph('<b>Public data:</b> Dingdong-Inc/FreshRetailNet-50K, CC BY 4.0. '
                '<b>Code + Docker:</b> github.com/Lisayinyy/MAIB_7002. '
                'Full experiments, limitations and contribution/AI-use statements accompany the notebook.',
                MARGIN,2323,INNER,13,leading=18,color=MUTED,max_height=25)
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
