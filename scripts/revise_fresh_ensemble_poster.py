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
    layer=TMP/"fresh_ensemble_revision_body.pdf"
    p=Layout(layer)
    p.rect(M,201*MM,RIGHT-M,47*MM,LIME,radius=8)
    p.text(f"{gain:.1f}%",M+30,209*MM,size=79,bold=True)
    p.text("lower prediction error",184*MM,209*MM,size=36,bold=True)
    p.text("RF + CatBoost (50/50) vs. 7-day average",184*MM,226*MM,size=27)
    p.text(f"Final-period MAE: {blend['mae']:.4f} vs. {baseline['mae']:.4f}  |  2,184 store-product-days",M+30,237*MM,size=22.5)

    # Data story: keep the original two-column rhythm and add the selection logic.
    lx,rx,colw=M,312*MM,261*MM
    p.section("1","Forecast tomorrow's sales",lx,261*MM)
    p.para("Help a Dingdong category manager compare daily discounts. Public FreshRetailNet histories contain sales, price rates, stockouts and calendar inputs.",
           lx,283*MM,colw,size=26,max_h=105)
    for x,value,label in [(lx,"50,000","public histories"),(123*MM,"16,684","eligible histories"),(223*MM,"312","in five stores")]:
        p.text(value,x,326*MM,size=38,bold=True,color=ACCENT)
        p.text(label,x,343*MM,size=19,color=MUTED)
    p.arrow(91*MM,334*MM,112*MM,334*MM)
    p.arrow(194*MM,334*MM,212*MM,334*MM)
    p.para("Keep 15-85% discounted days (rate &lt; 0.95), 6+ discount-state changes and &lt;20% zero-sales days; choose the five stores with most eligible histories.",
           lx,358*MM,colw,size=22,leading=27,max_h=88)
    p.text("312 histories / 122 products / one city",lx,393*MM,size=23,bold=True)
    p.text("Fit: 4 Apr - 25 Jun 2024; 25,896 rows",lx,407*MM,size=22)
    p.text("Validation: 5 expanding weekly folds",lx,420*MM,size=22)
    p.text("Benchmark: 26 Jun - 2 Jul; rolling one day ahead",lx,433*MM,size=20,color=MUTED)

    p.section("2","Combine two learners",rx,261*MM)
    p.para("10 history, calendar and promotion inputs.<br/>RF: squared-error splits; CatBoost: RMSE loss.<br/>Target: normalized observed sales, not units.",
           rx,283*MM,RIGHT-rx,size=25,leading=31,max_h=100)
    gap=15*MM; cw=(RIGHT-rx-gap)/2
    p.rect(rx,327*MM,cw,46*MM,PALE_GREEN,radius=6)
    p.rect(rx+cw+gap,327*MM,cw,46*MM,PALE_ORANGE,radius=6)
    rf_x=rx+cw/2; cb_x=rx+cw+gap+cw/2
    p.text("Random Forest",rf_x,333*MM,size=27,bold=True,color=ACCENT,align="center")
    p.text("300 parallel trees",rf_x,348*MM,size=24,align="center")
    p.text("Bootstrap, then average",rf_x,360*MM,size=21,align="center")
    p.text("CatBoost",cb_x,333*MM,size=27,bold=True,color="#D15C19",align="center")
    p.text("500 sequential trees",cb_x,348*MM,size=24,align="center")
    p.text("Reduce prior errors",cb_x,360*MM,size=21,align="center")
    avgx=(rx+RIGHT)/2; avw=98*MM
    p.arrow(rf_x,376*MM,avgx-30*MM,394*MM)
    p.arrow(cb_x,376*MM,avgx+30*MM,394*MM,color=ORANGE)
    p.text("50%",rf_x-12,385*MM,size=23,bold=True,color=ACCENT)
    p.text("50%",cb_x-16,385*MM,size=23,bold=True,color="#D15C19")
    p.rect(avgx-avw/2,394*MM,avw,15*MM,GREEN,radius=5)
    p.text("Average",avgx,396*MM,size=29,bold=True,color=WHITE,align="center")
    p.text("0.5 RF + 0.5 CatBoost = forecast",avgx,416*MM,size=25,bold=True,align="center")
    p.text("RF: depth 10, leaf 3  |  CB: depth 5, learning rate 0.03",rx,432*MM,size=21,color=MUTED)
    p.line(M,451*MM,RIGHT,451*MM)

    p.section("3","How much does prediction improve?",M,463*MM)
    p.text("MAE = mean absolute error; lower is better. Same final-period cases.",M,482*MM,size=24,color=MUTED)
    model_rows=[("7-day average","Baseline (7-day mean)","#A5B7A0"),
                ("kNN","kNN (k=25)","#B4CBB7"),
                ("Random Forest","Random forest",ACCENT),
                ("CatBoost","CatBoost","#61AAA6"),
                ("RF + CatBoost (50/50)","RF + CatBoost (50/50)",ORANGE)]
    bx,bw=229*MM,240*MM
    for i,(name,key,color) in enumerate(model_rows):
        y=(502+i*12)*MM
        v=scores[key]["mae"]
        p.text(name,M,y,size=26,bold=i==4)
        p.rect(bx,y+4,bw*v/.32,19,color)
        p.text(f"{v:.4f}",RIGHT-20,y,size=27,bold=i==4,align="right")
    p.text(f"Ensemble beats the baseline in all 5 validation weeks. Final RMSE (root mean square error): {blend['rmse']:.4f}.",
           M,566*MM,size=22,bold=True,color=ACCENT)
    p.line(M,582*MM,RIGHT,582*MM)

    p.section("4","Why a 50/50 blend?",M,594*MM,size=31)
    p.section("5","Where errors remain",rx,594*MM,size=31)
    p.text("RF / CatBoost (%)",M,615*MM,size=23,bold=True)
    p.text("Validation MAE",M+colw,615*MM,size=23,bold=True,align="right")
    p.line(M,627*MM,M+colw,627*MM)
    wanted=(0,.4,.5,.6,1)
    for i,w in enumerate(wanted):
        r=next(r for r in weight if float(r["rf_weight"])==w)
        y=(632+i*11)*MM
        if w==.5:p.rect(M-5,y-1,colw+10,29,PALE_GREEN,radius=3)
        p.text(f"{int(w*100)} / {100-int(w*100)}",M+10,y,size=24,bold=w==.5,color=ACCENT if w==.5 else INK)
        p.text(f"{float(r['mae']):.5f}",M+colw-7,y,size=24,bold=w==.5,color=ACCENT if w==.5 else INK,align="right")
    p.para("60/40 is marginally lower. We retain 50/50 for simplicity on a nearly flat validation curve.",
           M,691*MM,colw,size=21,leading=26,color=ACCENT,max_h=58)

    p.text("Observed rate",rx,615*MM,size=23,bold=True)
    p.text("MAE",497*MM,615*MM,size=23,bold=True,align="right")
    p.text("Cases",RIGHT,615*MM,size=23,bold=True,align="right")
    p.line(rx,627*MM,RIGHT,627*MM)
    for i,r in enumerate(errors):
        y=(636+i*17)*MM
        p.text(r["band"],rx,y,size=23)
        p.text(f"{r['mae']:.4f}",497*MM,y,size=24,align="right")
        p.text(f"{r['n']:,}",RIGHT,y,size=24,align="right")
    ratio=errors[2]["mae"]/errors[0]["mae"]
    p.text(f"Deep-discount MAE is {ratio:.1f}x full/light-rate MAE.",rx,691*MM,size=22,bold=True,color="#D15C19")

    # New business decision example, using the template's dark-green conclusion band.
    p.rect(M,713*MM,RIGHT-M,70*MM,GREEN,radius=7)
    p.section("6","From a forecast to a discount",M+22,721*MM,size=29,color=WHITE)
    p.text("5% off",M+24,740*MM,size=47,bold=True,color=LIME)
    p.text("Store 18 / Product 11 / 26 Jun 2024",M+220,741*MM,size=21,color=WHITE,bold=True)
    p.text(f"Predicted sales: {float(reference['pred_sales']):.2f} at full price -> {float(chosen['pred_sales']):.2f}",M+220,753*MM,size=23,color=WHITE)
    p.text(f"Value proxy: 0.95 x {float(chosen['pred_sales']):.2f} = {float(chosen['pred_value']):.2f}, vs. {float(reference['pred_value']):.2f} at full price",M+220,766*MM,size=21,color=WHITE)
    p.line(329*MM,731*MM,329*MM,776*MM,color="#6B967B")
    p.para("Compare eight price rates; require 3+ training days within 0.025 of each candidate rate. Apply assumed stock rules, then maximize <b>price rate x forecast sales.</b>",
           342*MM,725*MM,216*MM,size=23,leading=28,color=WHITE,max_h=122)
    p.text("Illustrative scenario; no observed uplift. Zero unit cost assumed.",M+24,775*MM,size=16.5,color="#D4E3CE")

    p.text("Better forecasts. Pricing impact still to test.",M,791*MM,size=28,bold=True)
    p.para("Discounts are observational; sales are stock-limited. No measured profit or waste benefit.<br/>Retrospective cohort and previously viewed benchmark. Next: inventory/cost data and a controlled pilot.",
           M,805*MM,476*MM,size=19,leading=24,max_h=51)
    p.text("FreshRetailNet-50K / Dingdong / CC BY 4.0. Normalized sales. Final notebook: pricing-ml-ap @ a7e26dd.",
           M,824*MM,size=13,color=MUTED)
    p.qr(531*MM,786*MM,39*MM)
    p.text("CODE + NOTEBOOK",550.5*MM,827*MM,size=11,bold=True,color=GREEN,align="center")
    p.c.linkURL(SOURCE_DATA,(M,H-831*MM,240*MM,H-821*MM),relative=0,thickness=0)
    p.c.linkURL(NOTEBOOK,(M,H-249*MM,RIGHT,H-201*MM),relative=0,thickness=0)
    p.c.save()

    result=OUT/"MAIB7002_GroupQ_A1_Fresh_Ensemble_revised.pdf"
    original=PdfReader(base); revised=PdfReader(layer)
    page=original.pages[0]
    page.merge_page(revised.pages[0])
    writer=PdfWriter(); writer.add_page(page)
    writer.add_metadata({"/Title":"Fresh Food. Smarter Discounts. | Final project revision",
                         "/Author":"Lingyu Chen; Yuanyuan Yin; Aashish Omprakash Pareek",
                         "/Subject":"A1 portrait, 594 x 841 mm; final 50/50 Random Forest and CatBoost evidence"})
    with result.open("wb") as f: writer.write(f)
    text=PdfReader(result).pages[0].extract_text()
    for old in ("8.56%","0.2873","200 parallel trees","400 sequential trees","Lowest mean validation MAE: 50/50"):
        assert old not in text, f"Obsolete source text remains in PDF: {old}"
    (TMP/"fresh_ensemble_revision_layout.json").write_text(json.dumps(p.boxes,indent=2))
    files=["results/test_summary.csv","results/ensemble_weight_sweep.csv","results/test_predictions.csv",
           "results/visuals/one_product_discount_decision.csv"]
    receipt={"final_notebook_commit":PIN,"source_template":json.loads((ASSETS/"fresh_ensemble_template_source.json").read_text()),
             "pdf_sha256":hashlib.sha256(result.read_bytes()).hexdigest(),"gain_percent":gain,"test_mae":blend["mae"],
             "test_rmse":blend["rmse"],"observed_discount_errors":errors,"deep_light_mae_ratio":ratio,
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
