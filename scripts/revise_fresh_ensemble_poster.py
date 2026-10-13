#!/usr/bin/env python3
"""Revise the supplied Fresh Ensemble poster, preserving its original header.

The original vector title, author line and fruit artwork are reused verbatim.
Everything below them is freshly drawn from the final project's saved evidence.
Older statistics are removed from the PDF stream, not hidden under rectangles.
Requires ReportLab and pypdf; see the output README for the verified versions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

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
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf"
ASSETS = ROOT / "assets/poster"
TMP = ROOT / "tmp/pdfs"
MM = 72 / 25.4
W, H = 594*MM, 841*MM
M, RIGHT = 22*MM, 572*MM
INK = "#163D29"
GREEN = "#0C5839"
LIME = "#D4ED62"
ORANGE = "#F27339"
ACCENT = "#087F4B"
MUTED = "#53685A"
CREAM = "#FFFDF3"
PALE_GREEN = "#EAF4D7"
PALE_ORANGE = "#FFF0DF"
RULE = "#D1D9CA"
WHITE = "#FFFFFF"
PIN = "a7e26dd979af88b4c14f98a7f029f6a64bffaedf"
NOTEBOOK_SHA256 = "a42326bed278c3b70c3469dd508ab563e6586dbdd84033117663c86a25033859"
REPO = "https://github.com/Lisayinyy/MAIB_7002/tree/pricing-ml-ap"
NOTEBOOK = f"https://github.com/Lisayinyy/MAIB_7002/blob/{PIN}/discount_sales_prediction.ipynb"
SOURCE_DATA = "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K"


def rows(name):
    with (ROOT / name).open(newline="") as f:
        return list(csv.DictReader(f))


def fonts():
    system = Path("/System/Library/Fonts/Supplemental")
    bundled = Path(reportlab.__file__).parent / "fonts"
    for name, filename, fallback in [("Fresh", "Arial.ttf", "Vera.ttf"),
                                      ("FreshBold", "Arial Bold.ttf", "VeraBd.ttf")]:
        file = system / filename
        pdfmetrics.registerFont(TTFont(name, str(file if file.exists() else bundled / fallback)))
    pdfmetrics.registerFontFamily("Fresh", normal="Fresh", bold="FreshBold", italic="Fresh", boldItalic="FreshBold")


def header(source: Path | None):
    """Keep content through the author line, dropping every obsolete body object."""
    path = ASSETS / "fresh_ensemble_original_header.pdf"
    if source is None:
        if not path.exists():
            raise FileNotFoundError("Provide --source with the user-supplied poster on first build")
        return path
    reader = PdfReader(source)
    assert len(reader.pages) == 1
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    page = writer.pages[0]
    stream = ContentStream(page.get_contents(), writer)
    author = "Lingyu Chen  |  Yuanyuan Yin  |  Aashish Omprakash Pareek"
    matches = [i for i, (args, op) in enumerate(stream.operations)
               if op == b"Tj" and str(args[0]) == author]
    assert len(matches) == 1, "Expected exactly one author line in this supplied template"
    stop = next(i for i in range(matches[0]+1, len(stream.operations))
                if stream.operations[i][1] == b"ET") + 1
    original_ops = stream.operations
    stream.operations = original_ops[:stop]
    depth = sum(1 if op == b"q" else -1 if op == b"Q" else 0 for _, op in stream.operations)
    assert depth == 0
    page.replace_contents(stream)
    check = page.extract_text()
    assert "FRESH FOOD." in check and author in check
    assert "8.56%" not in check and "Forecast tomorrow" not in check
    ASSETS.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        writer.write(f)
    (ASSETS / "fresh_ensemble_template_source.json").write_text(json.dumps({
        "source_filename": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "preserved": "Original title, fruit artwork, color palette, question and author order",
        "content_operations_kept": stop,
        "content_operations_original": len(original_ops),
    }, indent=2)+"\n")
    return path


class Layout:
    def __init__(self, file):
        self.c = canvas.Canvas(str(file), pagesize=(W, H), pageCompression=1)
        self.boxes = []

    def text(self, s, x, y, size=25, color=INK, bold=False, align="left"):
        font = "FreshBold" if bold else "Fresh"
        s = str(s)
        width = pdfmetrics.stringWidth(s, font, size)
        left = x if align == "left" else x - width if align == "right" else x - width/2
        assert 0 <= left <= left+width <= W+.1, s
        assert y+size < H, s
        self.c.setFillColor(HexColor(color)); self.c.setFont(font, size)
        fn = self.c.drawString if align == "left" else self.c.drawRightString if align == "right" else self.c.drawCentredString
        fn(x, H-y-.85*size, s)
        self.boxes.append(dict(text=s,x=left,y=y,w=width,h=size))

    def para(self, s, x, y, w, size=25, leading=None, color=INK, max_h=100):
        p = Paragraph(s, ParagraphStyle("p", fontName="Fresh", fontSize=size,
                                       leading=leading or size*1.22, textColor=HexColor(color)))
        _, h = p.wrap(w, H)
        assert h <= max_h+.1, (s, h, max_h)
        assert y+h < H-10, s
        p.drawOn(self.c, x, H-y-h)
        self.boxes.append(dict(text=s,x=x,y=y,w=w,h=h))

    def rect(self, x, y, w, h, color, radius=0):
        self.c.setFillColor(HexColor(color))
        if radius:
            self.c.roundRect(x,H-y-h,w,h,radius,stroke=0,fill=1)
        else:
            self.c.rect(x,H-y-h,w,h,stroke=0,fill=1)

    def line(self, x1, y1, x2, y2, color=RULE, width=1.2):
        self.c.setStrokeColor(HexColor(color)); self.c.setLineWidth(width)
        self.c.line(x1,H-y1,x2,H-y2)

    def arrow(self, x1, y1, x2, y2, color=ACCENT):
        from math import atan2, cos, sin
        self.line(x1,y1,x2,y2,color,2.5)
        a=atan2(y2-y1,x2-x1)
        for sign in (-1,1):
            self.line(x2,y2,x2-8*cos(a+sign*.55),y2-8*sin(a+sign*.55),color,2.5)

    def section(self, n, title, x, y, size=32, color=INK):
        r=14
        self.c.setFillColor(HexColor(ORANGE))
        self.c.circle(x+r,H-y-16,r,stroke=0,fill=1)
        self.text(n,x+r,y+5,size=21,color=WHITE,bold=True,align="center")
        self.text(title,x+41,y,size=size,bold=True,color=color)

    def donut(self, cx, cy, radius, percent, color=ORANGE):
        self.c.setFillColor(HexColor(PALE_GREEN))
        self.c.circle(cx, H-cy, radius, stroke=0, fill=1)
        self.c.setFillColor(HexColor(color))
        self.c.wedge(cx-radius, H-cy-radius, cx+radius, H-cy+radius,
                     startAng=90, extent=-percent*3.6, stroke=0, fill=1)
        self.c.setFillColor(HexColor(CREAM))
        self.c.circle(cx, H-cy, radius*.76, stroke=0, fill=1)
        self.text(f"{percent:.1f}%", cx, cy-21, size=34, bold=True, align="center")
        self.text("discount advised", cx, cy+18, size=14.5, color=MUTED, align="center")

    def qr(self, x, y, size):
        q=QrCodeWidget(REPO,barFillColor=HexColor(GREEN),barBorder=3)
        x0,y0,x1,y1=q.getBounds()
        d=Drawing(size,size,transform=[size/(x1-x0),0,0,size/(y1-y0),0,0])
        d.add(q); renderPDF.draw(d,self.c,x,H-y-size)
        self.c.linkURL(REPO,(x,H-y-size,x+size,H-y),relative=0,thickness=0)


def build(source=None):
    for folder in [OUT,TMP]: folder.mkdir(parents=True,exist_ok=True)
    base=header(source)
    fonts()
    scores={r[""]: {k:float(v) for k,v in r.items() if k} for r in rows("results/test_summary.csv")}
    weight=rows("results/ensemble_weight_sweep.csv")
    predictions=rows("results/test_predictions.csv")
    case=rows("results/visuals/one_product_discount_decision.csv")
    chosen=next(r for r in case if r["chosen"]=="True")
    reference=next(r for r in case if float(r["discount"])==1)
    blend=scores["RF + CatBoost (50/50)"]
    baseline=scores["Baseline (7-day mean)"]
    gain=100*(1-blend["mae"]/baseline["mae"])
    bands=[("Full/light (>0.95)",lambda x:x>.95),
           ("Moderate (0.80, 0.95]",lambda x:.8<x<=.95),
           ("Deep (<=0.80)",lambda x:x<=.8)]
    errors=[]
    for name,keep in bands:
        group=[r for r in predictions if keep(float(r["discount_next"]))]
        mae=sum(abs(float(r["target_sales"])-float(r["pred_ensemble"])) for r in group)/len(group)
        errors.append(dict(band=name,n=len(group),mae=mae))
    assert sum(r["n"] for r in errors)==2184
    assert abs(sum(r["n"]*r["mae"] for r in errors)/2184-blend["mae"])<1e-10
    assert float(chosen["discount"])==.95
    ratio=errors[2]["mae"]/errors[0]["mae"]
    layer=TMP/"fresh_ensemble_revision_body.pdf"
    p=Layout(layer)
    # The poster follows the notebook's experimental sequence, including the
    # surprising gap between forecasting accuracy and discount decisions.
    wx=next(r for r in rows("results/knn_stage1_with_weather_summary.csv")
            if r["weighting"]=="equal" and int(r["k"])==25)
    no_wx=next(r for r in rows("results/knn_stage1_summary.csv")
               if r["weighting"]=="equal" and int(r["k"])==25)
    policy=rows("results/visuals/recommendation_distribution.csv")
    shares={r["strategy"]:float(r["discount_share_percent"]) for r in policy}
    lx,rx,colw=M,310*MM,262*MM

    p.rect(M,201*MM,RIGHT-M,46*MM,LIME,radius=8)
    p.text("Can a better forecast guide tomorrow's discount?",M+26,208*MM,size=35,bold=True)
    p.para("A Dingdong category manager must balance unsold perishables against value lost to markdowns. We first predict sales, then test what happens when the forecast chooses the price.",
           M+26,224*MM,RIGHT-M-52,size=25,leading=30,max_h=64)

    p.section("1","Choose histories with price changes",lx,260*MM,size=29)
    p.para("We chose FreshRetailNet for its daily store-product sales, discounts and stockout records. We needed products with regular sales and repeated discount changes.",
           lx,279*MM,colw,size=24,leading=29,max_h=94)
    for x,value,label in [(lx,"50,000","public histories"),(123*MM,"16,684","pass the filters"),(228*MM,"312","in five stores")]:
        p.text(value,x,320*MM,size=37,bold=True,color=ACCENT)
        p.text(label,x,337*MM,size=19,color=MUTED)
    p.arrow(91*MM,329*MM,112*MM,329*MM)
    p.arrow(198*MM,329*MM,217*MM,329*MM)
    p.para("Keep histories with 15-85% discounted days (rate &lt; 0.95), 6+ discount-state changes and &lt;20% zero-sales days. Select the five stores with the most qualifying histories.",
           lx,353*MM,colw,size=22,leading=27,max_h=111)
    p.text("122 products, one city; 25,896 training/validation rows.",lx,396*MM,size=20,bold=True,color=ACCENT)

    p.section("2","Test our assumptions first",rx,260*MM,size=29)
    p.para("We began with kNN: predict from similar past days. Adding weather looked useful, but validation showed that fewer inputs worked better.",
           rx,279*MM,colw,size=24,leading=29,max_h=94)
    p.text("Mean absolute error (MAE)",rx,320*MM,size=22,bold=True)
    for i,(label,rr,color) in enumerate([("With weather",wx,"#A5B7A0"),("Weather removed",no_wx,ACCENT)]):
        y=(337+i*17)*MM
        p.text(label,rx,y,size=22)
        p.rect(401*MM,y+3,111*MM*float(rr["mae"])/.34,18,color)
        p.text(f"{float(rr['mae']):.4f}",RIGHT,y,size=24,bold=i==1,align="right")
    p.text("kNN: k=25, equal weights; five validation weeks.",rx,373*MM,size=19,color=MUTED)
    p.para("Decision: keep 10 inputs covering sales history, stockouts, discounts and calendar information.",
           rx,387*MM,colw,size=22,leading=26,max_h=55)

    p.line(M,412*MM,RIGHT,412*MM)
    p.section("3","Build from a simple baseline to a combined forecast",M,423*MM,size=31)
    p.text("Predict next-day normalized sales; select settings using five time-ordered validation weeks.",M,442*MM,size=23,color=MUTED)

    gap=13*MM; cw=(colw-gap)/2
    p.rect(lx,462*MM,cw,40*MM,PALE_GREEN,radius=6)
    p.rect(lx+cw+gap,462*MM,cw,40*MM,PALE_ORANGE,radius=6)
    rf_x=lx+cw/2; cb_x=lx+cw+gap+cw/2
    p.text("Random Forest",rf_x,468*MM,size=25,bold=True,color=ACCENT,align="center")
    p.text("300 parallel trees",rf_x,481*MM,size=22,align="center")
    p.text("Average their predictions",rf_x,491*MM,size=19,align="center")
    p.text("CatBoost",cb_x,468*MM,size=25,bold=True,color="#D15C19",align="center")
    p.text("500 sequential trees",cb_x,481*MM,size=22,align="center")
    p.text("Each corrects prior errors",cb_x,491*MM,size=19,align="center")
    avgx=lx+colw/2
    p.arrow(rf_x,504*MM,avgx-32*MM,517*MM)
    p.arrow(cb_x,504*MM,avgx+32*MM,517*MM,color=ORANGE)
    p.rect(lx+36*MM,518*MM,colw-72*MM,16*MM,GREEN,radius=5)
    p.text("0.5 RF + 0.5 CatBoost",avgx,521*MM,size=28,bold=True,color=WHITE,align="center")
    p.para("Averaging helped slightly. We chose 50/50 for simplicity on a flat validation curve; CatBoost alone is almost tied on the final week.",
           lx,542*MM,colw,size=22,leading=27,max_h=85)

    model_rows=[("7-day average","Baseline (7-day mean)","#A5B7A0"),
                ("kNN","kNN (k=25)","#B4CBB7"),
                ("Random Forest","Random forest",ACCENT),
                ("CatBoost","CatBoost","#61AAA6"),
                ("50/50 ensemble","RF + CatBoost (50/50)",ORANGE)]
    p.text("Final-week MAE (lower is better)",rx,459*MM,size=22,bold=True)
    bx,bw=416*MM,112*MM
    for i,(name,key,color) in enumerate(model_rows):
        y=(476+i*13)*MM
        v=scores[key]["mae"]
        p.text(name,rx,y,size=22,bold=i==4)
        p.rect(bx,y+4,bw*v/.32,17,color)
        p.text(f"{v:.4f}",RIGHT,y,size=23,bold=i==4,align="right")
    p.text(f"{gain:.1f}% lower MAE",rx,548*MM,size=34,bold=True,color=ACCENT)
    p.text(f"vs. the 7-day average  |  RMSE {blend['rmse']:.4f}",rx,564*MM,size=20,color=MUTED)
    p.text("Validation: 22 May - 25 Jun 2024. Final evaluation: 26 Jun - 2 Jul, 2,184 cases (a reused benchmark).",
           M,583*MM,size=20,color=MUTED)
    p.line(M,598*MM,RIGHT,598*MM)

    p.section("4","Then the model wants to discount",M,610*MM,size=29)
    p.section("5","Put stock rules before price",rx,610*MM,size=29)
    p.para("We varied tomorrow's price rate over eight candidates and maximized predicted sales x rate. Even with historical support, it chose discounts almost every day.",
           M,629*MM,colw,size=23,leading=28,max_h=91)
    p.para("We therefore tested rules: stockout today means full price; no stockout in a week requires a discount. Otherwise, the model compares eligible rates.",
           rx,629*MM,colw,size=23,leading=28,max_h=91)
    p.donut(52*MM,697*MM,27*MM,shares["Supported model only"])
    p.para("Model + historical support<br/><br/>Require 3+ training days within 0.025 of a candidate rate. This alone did not make recommendations selective.",
           89*MM,671*MM,193*MM,size=21,leading=26,max_h=145)
    p.donut(340*MM,697*MM,27*MM,shares["With stock rules"],color=ACCENT)
    p.para("Model + support + stock rules<br/><br/>The rules change the recommendation rate. They assume a stock policy; stockout records do not reveal inventory age.",
           377*MM,671*MM,195*MM,size=21,leading=26,max_h=145)
    p.text("Same 2,184 cases. Zero unit cost assumed. These are recommendation rates, not measured business improvements.",
           M,733*MM,size=19,color=MUTED)

    p.rect(M,744*MM,RIGHT-M,44*MM,GREEN,radius=7)
    p.section("6","One product: 5% off",M+22,751*MM,size=28,color=WHITE)
    p.text("Store 18 / Product 11 / 26 Jun 2024",M+22,767*MM,size=20,color=WHITE)
    p.text("Highest predicted value among eligible rates.",M+22,778*MM,size=17,color="#D4E3CE")
    p.line(295*MM,750*MM,295*MM,782*MM,color="#6B967B")
    options=[(344,1.0,"Full price"),(432,.95,"5% off"),(523,.8,"20% off")]
    for xm,rate,label in options:
        r=next(r for r in case if float(r["discount"])==rate)
        color=LIME if rate==.95 else WHITE
        p.text(label,xm*MM,751*MM,size=22,bold=rate==.95,color=color,align="center")
        p.text(f"{float(r['pred_value']):.3f}",xm*MM,763*MM,size=32,bold=True,color=color,align="center")
    p.text("Predicted value = sales x rate (5% off means a rate of 0.95).",433*MM,779*MM,size=16,color="#D4E3CE",align="center")

    p.text("Next: test the pricing decision in a real store.",M,794*MM,size=27,bold=True)
    p.para("Sales are normalized and stock-limited; discounts are observational. No measured revenue or waste gain.<br/>Five stores, retrospective selection and a reused evaluation week limit generalization.",
           M,808*MM,482*MM,size=18,leading=21,max_h=44)
    p.text("FreshRetailNet-50K / Dingdong / CC BY 4.0. Final notebook: pricing-ml-ap @ a7e26dd. Add inventory/cost data before a pilot.",
           M,826*MM,size=12,color=MUTED)
    p.qr(535*MM,791*MM,36*MM)
    p.text("CODE + NOTEBOOK",553*MM,829*MM,size=9.5,bold=True,color=GREEN,align="center")
    p.c.linkURL(SOURCE_DATA,(M,H-834*MM,240*MM,H-823*MM),relative=0,thickness=0)
    p.c.linkURL(NOTEBOOK,(M,H-248*MM,RIGHT,H-201*MM),relative=0,thickness=0)
    p.c.save()

    result=OUT/"MAIB7002_GroupQ_A1_Fresh_Ensemble_revised.pdf"
    original=PdfReader(base); revised=PdfReader(layer)
    page=original.pages[0]
    page.merge_page(revised.pages[0])
    writer=PdfWriter(); writer.add_page(page)
    writer.add_metadata({"/Title":"Fresh Food. Smarter Discounts. | From forecasts to discount decisions",
                         "/Author":"Lingyu Chen; Yuanyuan Yin; Aashish Omprakash Pareek",
                         "/Subject":"A1 portrait, 594 x 841 mm; final 50/50 Random Forest and CatBoost evidence"})
    with result.open("wb") as f: writer.write(f)
    text=PdfReader(result).pages[0].extract_text()
    for old in ("8.56%","0.2873","200 parallel trees","400 sequential trees","Lowest mean validation MAE: 50/50"):
        assert old not in text, f"Obsolete source text remains in PDF: {old}"
    (TMP/"fresh_ensemble_revision_layout.json").write_text(json.dumps(p.boxes,indent=2))
    files=["results/test_summary.csv","results/ensemble_weight_sweep.csv","results/test_predictions.csv",
           "results/visuals/one_product_discount_decision.csv", "results/knn_stage1_with_weather_summary.csv",
           "results/knn_stage1_summary.csv", "results/visuals/recommendation_distribution.csv"]
    receipt={"final_notebook_commit":PIN,"final_notebook_sha256":NOTEBOOK_SHA256,"final_notebook_url":NOTEBOOK,
             "story_cells":{"goal_and_selection":[0,2,3,4],"feature_experiment":[14,16,18,20],"forecast_and_blend":[38,40,42,44,46],"discount_decision":[54,58,60,62]},"source_template":json.loads((ASSETS/"fresh_ensemble_template_source.json").read_text()),
             "pdf_sha256":hashlib.sha256(result.read_bytes()).hexdigest(),"gain_percent":gain,"test_mae":blend["mae"],
             "test_rmse":blend["rmse"],"observed_discount_errors":errors,"deep_light_mae_ratio":ratio,
             "story":"Business trade-off -> data selection -> feature experiment -> forecast comparison -> over-discounting discovery -> assumed stock rules -> worked recommendation",
             "knn_weather_validation":{"with_weather":float(wx["mae"]),"without_weather":float(no_wx["mae"]),"k":25,"weighting":"equal"},
             "policy_discount_share_percent":shares,
             "selection":"Equal weights retained for simplicity; 60/40 has slightly lower mean validation MAE",
             "source_hashes":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
             "preservation":"Original title, fruit artwork, author order and palette; original body physically removed",
             "business_claim":"Predictive benchmark and model scenarios; no causal revenue/profit/waste gain shown"}
    (OUT/"fresh_ensemble_revision_evidence.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(result)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,help="User-provided PDF, required only for initial header extraction")
    args=parser.parse_args()
    build(args.source)
