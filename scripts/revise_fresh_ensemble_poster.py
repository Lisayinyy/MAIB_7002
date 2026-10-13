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
    predictions=rows("results/test_predictions.csv")
    case=rows("results/visuals/one_product_discount_decision.csv")
    chosen=next(r for r in case if r["chosen"]=="True")
    blend=scores["RF + CatBoost (50/50)"]
    baseline=scores["Baseline (7-day mean)"]
    gain=100*(1-blend["mae"]/baseline["mae"])
    band_specs=[("<5% off",lambda x:x>.95),
                ("5% to <20%",lambda x:.8<x<=.95),
                ("20%+ off",lambda x:x<=.8)]
    errors=[]
    for name,keep in band_specs:
        group=[r for r in predictions if keep(float(r["discount_next"]))]
        errors.append(dict(band=name,n=len(group),
                           mae=sum(abs(float(r["target_sales"])-float(r["pred_ensemble"])) for r in group)/len(group)))
    assert sum(r["n"] for r in errors)==2184
    assert abs(sum(r["n"]*r["mae"] for r in errors)/2184-blend["mae"])<1e-10
    assert float(chosen["discount"])==.95
    response=rows("results/visuals/discount_response_and_support.csv")
    policy=rows("results/visuals/recommendation_distribution.csv")
    shares={r["strategy"]:float(r["discount_share_percent"]) for r in policy}
    deepest=next(r for r in response if float(r["percent_off"])==40)
    layer=TMP/"fresh_ensemble_revision_body.pdf"
    p=Layout(layer)

    # Reflow figure content into poster-sized vector plots; do not shrink the
    # complete 16:9 figure pages. Three evidence panels carry the main story.
    p.rect(M,201*MM,RIGHT-M,31*MM,LIME,radius=8)
    p.text("Help a category manager choose tomorrow's discount.",M+22,207*MM,size=32,bold=True)
    p.text("Unsold perishables create waste; unnecessary markdowns give away value. Can forecasts help?",
           M+22,222*MM,size=22)
    p.text("WHY THESE DATA?",M,241*MM,size=18,bold=True,color=ACCENT)
    p.para("FreshRetailNet records daily sales, discounts and stockouts. We select regularly selling products with repeated price changes.",
           M,251*MM,222*MM,size=21,leading=25,max_h=80)
    for xm,value,label in [(289,"50,000","public histories"),(414,"16,684","eligible histories"),(540,"312","in five stores")]:
        p.text(value,xm*MM,239*MM,size=34,bold=True,color=ACCENT,align="center")
        p.text(label,xm*MM,253*MM,size=18,color=MUTED,align="center")
    p.arrow(331*MM,247*MM,363*MM,247*MM)
    p.arrow(462*MM,247*MM,499*MM,247*MM)
    p.para("Filters: 15-85% discounted days (rate &lt;0.95), 6+ discount switches, &lt;20% zero-sales days. Then take the five stores with the most eligible histories.",
           264*MM,264*MM,308*MM,size=17,leading=21,max_h=45)
    p.line(M,282*MM,RIGHT,282*MM)

    # FIGURE 02: forecast improvement and remaining absolute error by depth.
    p.section("1","A better forecast - with difficult cases still to solve",M,291*MM,size=32)
    p.text("Predict next-day normalized sales; choose settings with five time-ordered validation weeks.",M,307*MM,size=22,color=MUTED)
    p.text("Mean absolute error (MAE)",M,325*MM,size=23,bold=True)
    p.text("Ensemble MAE by observed discount",322*MM,325*MM,size=23,bold=True)
    p.text("Lower is better",283*MM,326*MM,size=17,color=MUTED,align="right")
    model_rows=[("7-day average","Baseline (7-day mean)","#B4C5AE"),
                ("kNN","kNN (k=25)","#B4C5AE"),
                ("Random Forest","Random forest","#6D9C80"),
                ("CatBoost","CatBoost",ACCENT),
                ("50/50 ensemble","RF + CatBoost (50/50)",ORANGE)]
    bx,bw=120*MM,118*MM
    for i,(name,key,color) in enumerate(model_rows):
        y=(340+i*11)*MM; v=scores[key]["mae"]
        p.text(name,M,y,size=21,bold=i==4)
        p.rect(bx,y+4,bw*v/.34,15,color)
        p.text(f"{v:.4f}",283*MM,y,size=22,bold=i==4,align="right")
    for tick in [0,.1,.2,.3]:
        tx=bx+bw*tick/.34
        p.text(f"{tick:.1f}",tx,396*MM,size=15,color=MUTED,align="center")
    p.line(bx,393*MM,bx+bw,393*MM)
    # These bars use the original figure's absolute-error diagnostic, omitting
    # a second baseline series so the small panel stays readable at A1 size.
    gx,gw,gy,gh=336*MM,228*MM,388*MM,47*MM
    for v in [0,.3,.6]:
        yy=gy-gh*v/.6
        p.line(gx,yy,gx+gw,yy,color=RULE,width=.8)
        p.text(f"{v:.1f}",gx-8,yy-7,size=15,color=MUTED,align="right")
    for i,r in enumerate(errors):
        xx=gx+(i+.5)*gw/3; barh=gh*r["mae"]/.6
        p.rect(xx-16*MM,gy-barh,32*MM,barh,ORANGE if i==2 else ACCENT)
        p.text(f"{r['mae']:.3f}",xx,gy-barh-24,size=21,bold=True,align="center")
        p.text(r["band"],xx,393*MM,size=18,bold=i==2,align="center")
        p.text(f"n = {r['n']:,}",xx,403*MM,size=15,color=MUTED,align="center")
    p.rect(M,416*MM,RIGHT-M,23*MM,PALE_GREEN,radius=6)
    p.text(f"{gain:.1f}% lower MAE",M+18,422*MM,size=31,bold=True,color=ACCENT)
    p.para("vs. the 7-day average. Equal-weight RF + CatBoost; CatBoost alone is nearly tied.\nDeep-discount days have larger absolute errors and higher average sales.",
           157*MM,421*MM,404*MM,size=19,leading=24,max_h=49)

    # FIGURE 03: predicted response and historical coverage are different things.
    p.section("2","Bigger predicted gains, thinner historical evidence",M,451*MM,size=32)
    p.text("Same 2,184 cases, eight discounts. Value proxy = predicted sales x price rate; zero cost assumed.",M,467*MM,size=22,color=MUTED)
    p.text("Predicted response vs. full price",M,484*MM,size=23,bold=True)
    p.text("Histories supporting each candidate",322*MM,484*MM,size=23,bold=True)
    for xx,label,color in [(M,"Sales",ACCENT),(91*MM,"Value proxy",ORANGE)]:
        p.line(xx,500*MM,xx+15*MM,500*MM,color=color,width=3)
        p.text(label,xx+19*MM,496.5*MM,size=18,color=color)
    p.text("% of 312 store-product histories",322*MM,497*MM,size=18,color=MUTED)
    x0,y0,pw,ph=46*MM,568*MM,232*MM,57*MM
    for v in [1,1.5,2,2.5]:
        yy=y0-ph*(v-.8)/1.7
        p.line(x0,yy,x0+pw,yy,width=.8)
        p.text(f"{v:.1f}x",x0-9,yy-8,size=15,color=MUTED,align="right")
    for key,color in [("sales_vs_no_discount",ACCENT),("value_vs_no_discount",ORANGE)]:
        pts=[(x0+pw*float(r["percent_off"])/40,y0-ph*(float(r[key])-.8)/1.7) for r in response]
        for (xa,ya),(xb,yb) in zip(pts,pts[1:]): p.line(xa,ya,xb,yb,color=color,width=3)
        for xx,yy in pts:
            p.c.setFillColor(HexColor(color));p.c.circle(xx,H-yy,4,fill=1,stroke=0)
        xx,yy=pts[-1]
        p.text(f"{float(deepest[key]):.2f}x",xx-9,yy-25,size=20,color=color,bold=True,align="right")
    sx,sy,sw,sh=344*MM,568*MM,217*MM,57*MM
    for v in [0,50,100]:
        yy=sy-sh*v/100
        p.line(sx,yy,sx+sw,yy,width=.8)
        p.text(str(v),sx-20,yy-8,size=15,color=MUTED,align="right")
    for r in response:
        xx=sx+sw*float(r["percent_off"])/40; bh=sh*float(r["support_percent"])/100
        p.rect(xx-4*MM,sy-bh,8*MM,bh,ORANGE if float(r["percent_off"])==40 else ACCENT)
        if float(r["percent_off"])==40:
            p.text("4.8%",xx,sy-bh-24,size=19,bold=True,color="#B24B1F",align="center")
    for start,width in [(x0,pw),(sx,sw)]:
        for v in [0,10,20,30,40]:
            p.text(f"{v}%",start+width*v/40,573*MM,size=16,color=MUTED,align="center")
        p.text("Discount (% off)",start+width/2,584*MM,size=17,color=MUTED,align="center")
    p.rect(M,597*MM,RIGHT-M,24*MM,PALE_ORANGE,radius=6)
    p.text("40% off",M+18,603*MM,size=30,bold=True,color="#B24B1F")
    p.para("2.32x predicted sales, but only 15 / 312 histories (4.8%) support this rate.\nSupport requires 3+ training days within 0.025 of the candidate price rate.",
           104*MM,602*MM,454*MM,size=20,leading=25,max_h=52)

    # FIGURE 04: explain the policy and show the entire recommendation mix.
    p.section("3","Add explicit stock rules before choosing the price",M,634*MM,size=32)
    p.text("Model + support still recommends discounts almost every day; rules change when and how deeply.",M,650*MM,size=22,color=MUTED)
    rule_rows=[("1  Stockout today >=1 hour","Full price tomorrow",PALE_GREEN),
               ("2  Otherwise, no stockout in 7 days","Use a supported discount, if available",PALE_ORANGE),
               ("3  Otherwise","Choose the highest predicted value",PALE_GREEN)]
    for i,(condition,decision,color) in enumerate(rule_rows):
        yy=(668+i*23)*MM
        p.rect(M,yy,255*MM,20*MM,color,radius=5)
        p.text(condition,M+13,yy+7,size=19,bold=True)
        p.text(decision,M+13,yy+32,size=20)
    p.text("Discount recommendations",313*MM,668*MM,size=21,bold=True)
    p.text(f"{shares['Supported model only']:.1f}%",313*MM,680*MM,size=34,bold=True,color=ORANGE)
    p.arrow(365*MM,687*MM,390*MM,687*MM,color=MUTED)
    p.text(f"{shares['With stock rules']:.1f}%",405*MM,680*MM,size=34,bold=True,color=ACCENT)
    p.text("with stock rules",455*MM,685*MM,size=18,color=MUTED)
    sx,sw=373*MM,198*MM
    pc={"Full price":"#D5DFC4","Moderate discount":ACCENT,"Deep discount":ORANGE}
    for j,(strategy,label) in enumerate([("Supported model only","Model + support"),("With stock rules","+ stock rules")]):
        yy=(706+j*18)*MM
        p.text(label,313*MM,yy+2,size=18,bold=j==1)
        left=sx
        for r in [r for r in policy if r["strategy"]==strategy]:
            portion=float(r["percent"]); width=sw*portion/100
            p.rect(left,yy,width,11*MM,pc[r["band"]])
            if portion>=10:
                p.text(f"{portion:.1f}%",left+width/2,yy+7,size=18,bold=True,color=INK if r["band"]=="Full price" else WHITE,align="center")
            left+=width
    for xm,label,color in [(313,"Full price",pc["Full price"]),(389,"0% < off < 20%",ACCENT),(482,"20%+ off",ORANGE)]:
        p.rect(xm*MM,742*MM,5*MM,5*MM,color)
        p.text(label,(xm+8)*MM,741*MM,size=17,color=MUTED)
    p.text("Same 2,184 cases. Rules are assumptions; stockouts do not reveal inventory age. Revenue or waste gains remain untested.",M,754*MM,size=18,color=MUTED)

    # Compact closing example from the original figure 05.
    p.rect(M,769*MM,RIGHT-M,29*MM,GREEN,radius=7)
    p.text("ONE PRODUCT, ONE DECISION: 5% OFF",M+18,775*MM,size=26,bold=True,color=LIME)
    p.text("Store 18 / Product 11 / 26 June",M+18,789*MM,size=18,color=WHITE)
    p.text("Predicted value",358*MM,775*MM,size=17,color="#D4E3CE",align="center")
    p.text("Full price  1.121",358*MM,786*MM,size=22,color=WHITE,align="center")
    p.text("Best eligible rate",502*MM,775*MM,size=17,color="#D4E3CE",align="center")
    p.text("5% off  1.198",502*MM,786*MM,size=23,bold=True,color=LIME,align="center")

    p.text("Next: validate decisions in a controlled store pilot.",M,804*MM,size=24,bold=True)
    p.para("Observational, normalized sales; value is a proxy, not currency. Five stores in one city; retrospective selection.\n26 Jun - 2 Jul 2024: 2,184 cases, a reused benchmark. Add inventory and cost data before a pilot.",
           M,815*MM,493*MM,size=16,leading=20,max_h=41)
    p.text("Source: Dingdong FreshRetailNet-50K (CC BY 4.0). Final notebook: pricing-ml-ap @ a7e26dd. Model scenarios do not prove causal uplift.",
           M,832*MM,size=10.5,color=MUTED)
    p.qr(538*MM,803*MM,33*MM)
    p.c.linkURL(SOURCE_DATA,(M,H-840*MM,230*MM,H-831*MM),relative=0,thickness=0)
    p.c.linkURL(NOTEBOOK,(M,H-232*MM,RIGHT,H-201*MM),relative=0,thickness=0)
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
    files=["results/test_summary.csv","results/test_predictions.csv",
           "results/visuals/one_product_discount_decision.csv",
           "results/visuals/discount_response_and_support.csv",
           "results/visuals/recommendation_distribution.csv"]
    receipt={"final_notebook_commit":PIN,"final_notebook_sha256":NOTEBOOK_SHA256,"final_notebook_url":NOTEBOOK,
             "story_cells":{"goal_and_selection":[0,2,3,4],"forecast_and_errors":[38,40,42,44,46,47,48],"discount_decision":[54,58,60,61,62]},
             "source_template":json.loads((ASSETS/"fresh_ensemble_template_source.json").read_text()),
             "pdf_sha256":hashlib.sha256(result.read_bytes()).hexdigest(),"gain_percent":gain,"test_mae":blend["mae"],
             "test_rmse":blend["rmse"],"observed_discount_errors":errors,
             "story":"Business question and cohort -> forecasting progress and remaining errors -> predicted response versus support -> stock rules and recommendation mix -> one supported 5%-off decision",
             "integrated_key_figures":[2,3,4],"compact_bookends":[1,5],"omitted_optional_figure":6,
             "response_ratios_denominator":"Mean prediction across the same 2184 cases at each candidate",
             "historical_support_denominator":312,"forty_percent_off":deepest,
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
