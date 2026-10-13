#!/usr/bin/env python3
"""Build the group's three-part A1 poster from pinned experiment evidence.

Problem / Data Cleaning -> Models -> Results & Implications.
Actual experiment charts and all text are vector objects. Decorative produce art is generated,
not experimental evidence. No QR code or original PDF body is imported.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import reportlab
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader, PdfWriter, Transformation

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf'
ASSETS=ROOT/'assets/poster'
TMP=ROOT/'tmp/pdfs'
MM=72/25.4
W,H=594*MM,841*MM
M,RIGHT=22*MM,572*MM
INK='#163D29';GREEN='#0C5839';LIME='#D4ED62';ORANGE='#F27339'
ACCENT='#087F4B';MUTED='#53685A';CREAM='#FFFDF3';PALE_GREEN='#EAF4D7'
PALE_ORANGE='#FFF0DF';RULE='#D1D9CA';WHITE='#FFFFFF'
PIN='a7e26dd979af88b4c14f98a7f029f6a64bffaedf'
NOTEBOOK_SHA256='a42326bed278c3b70c3469dd508ab563e6586dbdd84033117663c86a25033859'
NOTEBOOK=f'https://github.com/Lisayinyy/MAIB_7002/blob/{PIN}/discount_sales_prediction.ipynb'
SOURCE_DATA='https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K'

def rows(name):
    with (ROOT/name).open(newline='') as f: return list(csv.DictReader(f))

def fonts():
    system=Path('/System/Library/Fonts/Supplemental')
    bundled=Path(reportlab.__file__).parent/'fonts'
    for name,filename,fallback in [('Fresh','Arial.ttf','Vera.ttf'),('FreshBold','Arial Bold.ttf','VeraBd.ttf')]:
        file=system/filename
        pdfmetrics.registerFont(TTFont(name,str(file if file.exists() else bundled/fallback)))
    pdfmetrics.registerFontFamily('Fresh',normal='Fresh',bold='FreshBold',italic='Fresh',boldItalic='FreshBold')

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


def build():
    OUT.mkdir(parents=True,exist_ok=True);TMP.mkdir(parents=True,exist_ok=True)
    fonts()
    scores={r['']:{k:float(v) for k,v in r.items() if k} for r in rows('results/test_summary.csv')}
    response=rows('results/visuals/discount_response_and_support.csv')
    policy=rows('results/visuals/recommendation_distribution.csv')
    case=rows('results/visuals/one_product_discount_decision.csv')
    chosen=next(r for r in case if r['chosen']=='True')
    assert float(chosen['discount'])==.95
    blend=scores['RF + CatBoost (50/50)'];baseline=scores['Baseline (7-day mean)']
    gain=100*(1-blend['mae']/baseline['mae'])
    shares={r['strategy']:float(r['discount_share_percent']) for r in policy}
    result=OUT/'MAIB7002_GroupQ_A1_Fresh_Ensemble_revised.pdf'
    layer=TMP/'fresh_ensemble_empirical_body.pdf'
    p=Layout(layer)
    p.c.setTitle('Fresh Food. Smarter Discounts. | Forecasting and daily markdown decisions')
    p.c.setAuthor('Lingyu Chen; Yuanyuan Yin; Aashish Omprakash Pareek')
    p.c.setSubject('A1 portrait 594 x 841 mm; empirical notebook plots; Phase One and Phase Two; no QR')
    p.rect(0,0,W,H,CREAM)
    chart_slots=[]
    def chart(name,x,y,w=265):
        file=ASSETS/'experiments'/f'{name}.pdf'
        r=PdfReader(file);pg=r.pages[0]
        height=w*float(pg.mediabox.height)/float(pg.mediabox.width)
        chart_slots.append(dict(path=str(file),name=name,x=x,y=y,w=w,h=height))

    # Compact green identity; larger question text occupies less vertical space.
    p.rect(0,0,W,119*MM,GREEN)
    p.c.setFillColor(HexColor('#236A45'))
    p.c.ellipse(385*MM,H-113*MM,580*MM,H-7*MM,fill=1,stroke=0)
    p.c.setStrokeColor(HexColor('#689753'));p.c.setLineWidth(.8)
    p.c.ellipse(376*MM,H-116*MM,588*MM,H-4*MM,fill=0,stroke=1)
    hero=ASSETS/'fresh_produce_hero_v3.png'
    p.c.drawImage(ImageReader(str(hero)),399*MM,H-117*MM,width=171*MM,height=114*MM,mask='auto')
    p.text('HKU MAIB7002  /  GROUP Q',M,14*MM,size=24,bold=True,color=WHITE)
    p.text('FRESH FOOD.',M,32*MM,size=70,bold=True,color=WHITE)
    p.text('SMARTER',M,63*MM,size=59,bold=True,color=LIME)
    p.text('DISCOUNTS.',M,88*MM,size=59,bold=True,color=LIME)
    p.text('Lingyu Chen  |  Yuanyuan Yin  |  Aashish Omprakash Pareek',M,127*MM,size=21,color=MUTED)

    # 1: concise problem and compact data preparation, as requested.
    p.section('1','Problem',M,145*MM,size=32)
    p.text('Data Cleaning',311*MM,145*MM,size=32,bold=True)
    p.para('Discount tomorrow?<br/>How much is enough?',M,164*MM,264*MM,size=40,leading=47,max_h=95)
    p.para('Fresh-food managers must sell perishables before they spoil, while protecting value from unnecessary markdowns.',
           M,205*MM,264*MM,size=24,leading=29,max_h=90)
    p.text('Our goal: recommend a supported daily discount.',M,242*MM,size=22,bold=True,color=ACCENT)
    p.para('FreshRetailNet records daily sales, discounts and stockouts. We choose regular sellers with repeated price changes.',
           311*MM,164*MM,261*MM,size=21,leading=26,max_h=58)
    for xm,value,label in [(345,'50,000','histories'),(442,'16,684','eligible'),(543,'312','histories / 5 stores')]:
        p.text(value,xm*MM,187*MM,size=28,bold=True,color=ACCENT,align='center')
        p.text(label,xm*MM,200*MM,size=15.5,color=MUTED,align='center')
    p.arrow(378*MM,194*MM,407*MM,194*MM)
    p.arrow(478*MM,194*MM,510*MM,194*MM)
    p.para('15-85% days at rate &lt;0.95; 6+ state switches; &lt;20% zero-sales days. Take the five stores with the most eligible histories.',
           311*MM,213*MM,261*MM,size=16,leading=20,max_h=45)
    p.para('Check duplicates, gaps and missing values; cap rates &gt;1 and mask zero-price days. Build past-only inputs; drop incomplete history rows.',
           311*MM,232*MM,261*MM,size=17.5,leading=21,max_h=46)
    p.text('25,896 train / validation rows + 2,184 final-week cases',311*MM,251*MM,size=17,bold=True,color=ACCENT)
    p.line(M,265*MM,RIGHT,265*MM)

    # 2: four actual experimental plots, plus the baseline and final comparison.
    p.section('2','Models',M,277*MM,size=34)
    p.text("Phase One: predict tomorrow's sales",157*MM,279*MM,size=25,bold=True)
    p.text('10 decision-time inputs; five expanding validation weeks; MAE (mean absolute error) selects settings.',M,296*MM,size=21,color=MUTED)
    p.text('7-day mean baseline  /  kNN  /  Random Forest  /  CatBoost  /  RF + CatBoost',M,309*MM,size=20,bold=True,color=ACCENT)
    p.text('kNN: choose the neighbourhood',M,327*MM,size=25,bold=True)
    p.text('Random Forest: limit complexity',307*MM,327*MM,size=25,bold=True)
    chart('01_knn_validation',22,338)
    chart('02_rf_validation_grid',307,338)
    p.para('25 neighbours, standardized inputs, equal weights. The validation curve is flat near k=25.',M,422*MM,265*MM,size=19,leading=23,max_h=47)
    p.para('300 trees, depth 10, minimum leaf size 3. The whole tuning grid spans less than 0.003 MAE.',307*MM,422*MM,265*MM,size=19,leading=23,max_h=47)
    p.text('CatBoost: stop before overfitting',M,448*MM,size=25,bold=True)
    p.text('RF + CatBoost: choose simple weights',307*MM,448*MM,size=25,bold=True)
    chart('03_catboost_learning',22,459)
    chart('04_ensemble_weights',307,459)
    p.para('500 trees, depth 5, learning rate 0.03. On one fold, validation error turns up as training error keeps falling.',M,543*MM,265*MM,size=18.5,leading=22.5,max_h=48)
    p.para('50/50 is a simple choice on a flat curve; 60/40 has the lowest tested MAE. Similar errors limit the extra gain.',307*MM,543*MM,265*MM,size=18.5,leading=22.5,max_h=48)
    p.rect(M,568*MM,550*MM,32*MM,PALE_GREEN,radius=7)
    score_order=[('7-day mean','Baseline (7-day mean)'),('kNN','kNN (k=25)'),('Random Forest','Random forest'),('CatBoost','CatBoost'),('50/50 ensemble','RF + CatBoost (50/50)')]
    for i,(label,key) in enumerate(score_order):
        xx=(63+i*88)*MM
        p.text(label,xx,573*MM,size=17,bold=i==4,align='center')
        p.text(f"{scores[key]['mae']:.4f}",xx,583*MM,size=28,bold=True,color=ACCENT if i<4 else '#B24B1F',align='center')
    p.text('Final-week MAE',528*MM,573*MM,size=17,bold=True,align='center')
    p.text(f'{gain:.1f}% lower',528*MM,582*MM,size=27,bold=True,color=ACCENT,align='center')
    p.text('vs. the 7-day mean',528*MM,594*MM,size=12.5,color=MUTED,align='center')

    # 3: Phase Two turns a forecast into a specific business decision.
    p.section('3','Results & Implications',M,613*MM,size=33)
    p.text("Phase Two: choose tomorrow's discount",278*MM,615*MM,size=24,bold=True,color=ACCENT)
    p.text('Compare eight discounts (0-40% off). Value proxy = predicted sales x price rate; zero cost assumed.',M,631*MM,size=20,color=MUTED)
    p.text('Whether: discounting becomes selective',M,647*MM,size=24,bold=True)
    p.text('How much: 5% off (price rate 0.95)',307*MM,647*MM,size=24,bold=True)
    chart('05_policy_recommendations',22,658)
    chart('06_case_discount_depth',307,658)
    p.para('Discount advice falls from <b>98.3% to 55.0%</b>. Stockout today (1+ hour): full price. No stockout in seven days: supported discount, if available. Otherwise, compare values.',M,742*MM,265*MM,size=18,leading=22,max_h=67)
    p.para('Store 18 / Product 11 / 26 Jun: <b>5% off</b> predicts sales <b>1.261</b> and value <b>1.198</b>. A deeper 20% discount predicts more sales, but less value (1.136).',307*MM,742*MM,265*MM,size=18,leading=22,max_h=67)
    p.text('Support: 3+ training days within 0.025; cost and gain hurdle = 0.',M,771*MM,size=15.5,color=MUTED)
    p.text('This product has no support for 25%, 30% or 40% off.',307*MM,771*MM,size=15.5,color=MUTED)
    p.rect(M,786*MM,550*MM,28*MM,GREEN,radius=7)
    p.text('MANAGER TAKEAWAY',M+18,791*MM,size=19,bold=True,color=LIME)
    p.para('Review stock first, then choose a supported markdown that balances predicted sales and value. Set cost and gain thresholds before a controlled store pilot.',
           146*MM,791*MM,415*MM,size=21,leading=25,color=WHITE,max_h=53)
    p.text('26 Jun - 2 Jul 2024: reused benchmark. Five stores in one city; retrospective selection. Stock rules are assumptions about policy.',M,821*MM,size=13,color=MUTED)
    p.text('Normalized sales and model scenarios do not prove revenue or waste gains. Source: FreshRetailNet-50K (CC BY 4.0); final notebook a7e26dd.',M,831*MM,size=12,color=MUTED)
    p.c.save()
    writer=PdfWriter();writer.add_page(PdfReader(layer).pages[0]);page=writer.pages[0]
    for slot in chart_slots:
        ch=PdfReader(slot['path']).pages[0]
        scale=slot['w']*MM/float(ch.mediabox.width)
        tx=slot['x']*MM;ty=H-(slot['y']+slot['h'])*MM
        page.merge_transformed_page(ch,Transformation().scale(scale).translate(tx,ty))
    writer.add_metadata({'/Title':'Fresh Food. Smarter Discounts. | From prediction to daily discount decisions','/Author':'Lingyu Chen; Yuanyuan Yin; Aashish Omprakash Pareek','/Subject':'A1; empirical notebook charts, two project phases, no QR'})
    with result.open('wb') as f:writer.write(f)
    r=PdfReader(result)
    assert len(r.pages)==1
    text=r.pages[0].extract_text()
    assert 'QR' not in text
    for word in ['Data Cleaning','CatBoost','0.2892','7.9%','Results & Implications','Phase Two','1.261','1.198']:assert word in text
    (TMP/'fresh_ensemble_revision_layout.json').write_text(json.dumps(p.boxes,indent=2))
    (TMP/'fresh_ensemble_revision_chart_slots.json').write_text(json.dumps(chart_slots,indent=2))
    files=['results/test_summary.csv','results/visuals/recommendation_distribution.csv','results/visuals/one_product_discount_decision.csv','assets/poster/fresh_produce_hero_v3.png','assets/poster/fresh_produce_hero_v3_prompt.md']
    receipt={'final_notebook_commit':PIN,'final_notebook_sha256':NOTEBOOK_SHA256,'final_notebook_url':NOTEBOOK,
             'pdf_sha256':hashlib.sha256(result.read_bytes()).hexdigest(),
             'sections':['Problem / Data Cleaning','Models: Phase One sales prediction','Results & Implications: Phase Two price choice'],
             'model_methods':['7-day mean baseline','kNN','Random Forest','CatBoost','50/50 RF + CatBoost'],
             'model_section_mm':[277,600],'header':'Compact green identity; existing decorative produce asset retained',
             'no_qr_code':True,'plot_provenance':json.loads((ASSETS/'experiments/evidence.json').read_text()),'model_diagrams_removed':True,'chart_slots':[{k:v for k,v in slot.items() if k!='path'} for slot in chart_slots],'gain_percent':gain,'test_mae':blend['mae'],'test_rmse':blend['rmse'],
             'policy_discount_share_percent':shares,'cleaning_claim':'Checks and rules, not invented per-rule changed-row counts',
             'selection':'Equal weights retained for simplicity; 60/40 has slightly lower mean validation MAE',
             'business_claim':'Predictive benchmark and model scenarios; no causal revenue/profit/waste gain shown',
             'source_hashes':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}
    (OUT/'fresh_ensemble_revision_evidence.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(result)

if __name__=='__main__':build()
