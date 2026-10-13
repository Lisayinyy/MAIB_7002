#!/usr/bin/env python3
"""Build the group's three-part A1 poster from pinned experiment evidence.

Problem / Data Cleaning -> Models -> Results & Implications.
Charts and mechanisms are vector objects. Decorative produce art is generated,
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
from pypdf import PdfReader

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


def dot(p,x,y,r,color=ACCENT):
    p.c.setFillColor(HexColor(color));p.c.circle(x,H-y,r,stroke=0,fill=1)

def tree(p,x,y,s=1,color=ACCENT):
    # A mechanism schematic, not a tree fitted to a particular product.
    for a,b in [((0,0),(-13,19)),((0,0),(13,19)),((-13,19),(-21,36)),((-13,19),(-5,36)),((13,19),(5,36)),((13,19),(21,36))]:
        p.line(x+a[0]*s,y+a[1]*s,x+b[0]*s,y+b[1]*s,color=color,width=1.7)
    for dx,dy in [(0,0),(-13,19),(13,19),(-21,36),(-5,36),(5,36),(21,36)]:dot(p,x+dx*s,y+dy*s,3*s,color)

def model_card(p,x,y,n,title,explanation,settings,kind,color=ACCENT):
    width,height=265*MM,71*MM
    p.rect(x,y,width,height,WHITE,radius=9)
    p.rect(x,y,3*MM,height,color)
    p.text(n,x+15,y+17,size=17,bold=True,color=color)
    p.text(title,x+44,y+12,size=29,bold=True)
    p.para(explanation,x+18,y+57,165*MM,size=21,leading=26,max_h=82)
    p.text(settings,x+18,y+height-27,size=16.5,color=MUTED)
    cx=x+214*MM
    if kind=='mean':
        vals=[17,31,24,39,21,32,25]
        for i,v in enumerate(vals):p.rect(cx-68+i*18,y+126-v,11,v,'#9AB7A0')
        p.line(cx-72,y+99,cx+73,y+99,color=ORANGE,width=2.5)
        p.text('7 days -> average',cx,y+141,size=16,color=MUTED,align='center')
    elif kind=='knn':
        pts=[(-60,-3),(-40,-28),(-31,18),(-9,-15),(3,12),(23,-22),(28,22),(57,8),(-62,32),(61,-30),(-12,42),(41,46)]
        for i,(dx,dy) in enumerate(pts):dot(p,cx+dx,y+97+dy,4.5,ORANGE if i in [2,3,4,5,6] else '#B2C7B6')
        p.c.setStrokeColor(HexColor(ACCENT));p.c.setLineWidth(2);p.c.circle(cx,H-y-100,43,stroke=1,fill=0)
        dot(p,cx,y+100,6,ACCENT)
        p.text('Similar past days',cx,y+151,size=16,color=MUTED,align='center')
    elif kind=='rf':
        for xx in [-55,0,55]:tree(p,cx+xx,y+64,.85,color)
        for xx in [-55,0,55]:p.arrow(cx+xx,y+104,cx,y+133,color=color)
        p.rect(cx-42,y+140,84,22,PALE_GREEN,radius=3)
        p.text('Average',cx,y+143,size=15,bold=True,align='center')
    elif kind=='cb':
        for j,xx in enumerate([-56,0,56]):
            tree(p,cx+xx,y+78,.65,color)
            if j<2:p.arrow(cx+xx+17,y+94,cx+xx+39,y+94,color=color)
        p.text('Correct step by step',cx,y+136,size=16,color=MUTED,align='center')

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
    p=Layout(result)
    p.c.setTitle('Fresh Food. Smarter Discounts. | Problem, Models, Results')
    p.c.setAuthor('Lingyu Chen; Yuanyuan Yin; Aashish Omprakash Pareek')
    p.c.setSubject('A1 portrait 594 x 841 mm; five forecasting approaches; no QR code')
    p.rect(0,0,W,H,CREAM)

    # New compact header: retain identity, replace the original produce art.
    p.rect(0,0,W,136*MM,GREEN)
    p.c.setFillColor(HexColor('#236A45'))
    p.c.ellipse(381*MM,H-128*MM,582*MM,H-8*MM,fill=1,stroke=0)
    p.c.setStrokeColor(HexColor('#689753'));p.c.setLineWidth(.8)
    p.c.ellipse(371*MM,H-131*MM,588*MM,H-5*MM,fill=0,stroke=1)
    hero=ASSETS/'fresh_produce_hero_v3.png'
    p.c.drawImage(ImageReader(str(hero)),379*MM,H-134*MM,width=196*MM,height=196/1.5*MM,mask='auto')
    p.text('HKU MAIB7002  /  GROUP Q',M,17*MM,size=24,bold=True,color=WHITE)
    p.text('FRESH FOOD.',M,39*MM,size=77,bold=True,color=WHITE)
    p.text('SMARTER',M,72*MM,size=65,bold=True,color=LIME)
    p.text('DISCOUNTS.',M,98*MM,size=65,bold=True,color=LIME)
    p.text('Lingyu Chen  |  Yuanyuan Yin  |  Aashish Omprakash Pareek',M,143*MM,size=21,color=MUTED)

    # PART 1: problem on the left, data cleaning on the right.
    p.section('1','Problem',M,163*MM,size=34)
    p.text('Data Cleaning',311*MM,163*MM,size=34,bold=True)
    p.para('Should we discount it tomorrow?<br/>If so, by how much?',M,182*MM,264*MM,size=30,leading=36,max_h=80)
    p.para('Fresh-food managers balance unsold perishables against value lost to markdowns. We forecast next-day sales, then compare supported prices to recommend a daily discount.',
           M,214*MM,253*MM,size=23,leading=29,max_h=122)
    for x,title,body,color in [(M,'Too little discount','Unsold food',PALE_GREEN),(158*MM,'Too much discount','Value given away',PALE_ORANGE)]:
        p.rect(x,262*MM,125*MM,28*MM,color,radius=6)
        p.text(title,x+16,268*MM,size=21,bold=True)
        p.text(body,x+16,280*MM,size=20,color=MUTED)
    p.text('Goal: recommend a specific, historically supported discount.',M,300*MM,size=19,bold=True,color=ACCENT)
    p.para('FreshRetailNet-50K provides daily sales, discounts and stockouts. Select regular sellers with repeated price changes.',
           311*MM,182*MM,261*MM,size=20,leading=25,max_h=77)
    p.text('15-85% days at rate <0.95; 6+ state switches; <20% zero-sales days.',311*MM,202*MM,size=14.5,color=MUTED)
    for xm,value,label in [(345,'50,000','histories'),(442,'16,684','eligible'),(543,'312','histories / 5 stores')]:
        p.text(value,xm*MM,211*MM,size=29,bold=True,color=ACCENT,align='center')
        p.text(label,xm*MM,224*MM,size=16,color=MUTED,align='center')
    p.arrow(378*MM,218*MM,407*MM,218*MM)
    p.arrow(478*MM,218*MM,510*MM,218*MM)
    p.text('Take the five stores with the most eligible histories.',311*MM,233*MM,size=14,color=MUTED)
    checks=[('Check daily records','Unique keys; no missing fields or date gaps.'),
            ('Handle invalid price rates','Cap rates above 1; mask zero-price days.'),
            ('Build past-only features','Exclude rows without complete history inputs.')]
    for i,(title,desc) in enumerate(checks):
        yy=(239+i*19)*MM
        dot(p,315*MM,yy+10,5,ACCENT)
        p.text(title,322*MM,yy,size=20,bold=True)
        p.text(desc,322*MM,yy+27,size=17.5,color=MUTED)
    p.text('25,896 train / validation rows + 2,184 final-week rows',311*MM,300*MM,size=18,bold=True,color=ACCENT)
    p.line(M,314*MM,RIGHT,314*MM)

    # PART 2 is deliberately the largest panel.
    p.section('2','Models',M,326*MM,size=36)
    p.text('Five approaches, from a simple reference to a combined forecast.',135*MM,329*MM,size=22,color=MUTED)
    p.rect(M,349*MM,550*MM,35*MM,PALE_GREEN,radius=7)
    p.text('10 INPUTS',M+18,355*MM,size=18,bold=True,color=ACCENT)
    p.text('Sales history, stockouts, discounts, activity and calendar',M+18,366*MM,size=21)
    p.arrow(319*MM,368*MM,350*MM,368*MM)
    p.text('TARGET',368*MM,355*MM,size=18,bold=True,color=ACCENT)
    p.text('Next-day normalized sales',368*MM,366*MM,size=23,bold=True)
    p.text("Tomorrow's candidate price rate is an input; the model does not directly predict the discount.",M,391*MM,size=19,color=MUTED)
    model_card(p,M,405*MM,'01','7-day mean',
               "Use the previous week's average sales as tomorrow's forecast. This sets a simple benchmark.",
               'Baseline | no fitted model','mean')
    model_card(p,307*MM,405*MM,'02','kNN',
               'Find similar historical days and average their sales. Feature scaling makes distances comparable.',
               '25 neighbours | standardized inputs | equal weights','knn')
    model_card(p,M,487*MM,'03','Random Forest',
               'Fit many trees on resampled data, then average their predictions to reduce variability.',
               '300 trees | depth 10 | minimum leaf size 3','rf')
    model_card(p,307*MM,487*MM,'04','CatBoost',
               'Add trees sequentially. Each new tree targets prediction errors left by the earlier trees.',
               '500 trees | depth 5 | learning rate 0.03','cb',ORANGE)
    p.rect(M,569*MM,550*MM,43*MM,GREEN,radius=8)
    p.text('05  RF + CatBoost',M+19,576*MM,size=28,bold=True,color=WHITE)
    p.para('Average two strong forecasts. Equal weights keep the model simple; the validation curve is almost flat near 50/50.',
           M+19,590*MM,262*MM,size=19,leading=24,color=WHITE,max_h=50)
    for xx,label,color in [(339,'0.5 RF',PALE_GREEN),(423,'0.5 CatBoost',PALE_ORANGE)]:
        p.rect(xx*MM,579*MM,74*MM,16*MM,color,radius=4)
        p.text(label,(xx+37)*MM,583*MM,size=23,bold=True,align='center')
    p.text('+',418*MM,582*MM,size=25,bold=True,color=WHITE,align='center')
    p.arrow(497*MM,587*MM,514*MM,587*MM,color=LIME)
    p.text('Sales',541*MM,580*MM,size=23,bold=True,color=LIME,align='center')
    p.text('forecast',541*MM,591*MM,size=19,color=WHITE,align='center')
    p.text('EVALUATION',M,622*MM,size=17,bold=True,color=ACCENT)
    p.text('Five expanding validation weeks choose settings',94*MM,621*MM,size=20)
    p.arrow(329*MM,625*MM,354*MM,625*MM)
    p.text('Final week: 26 Jun - 2 Jul 2024',367*MM,621*MM,size=20)
    p.line(M,638*MM,RIGHT,638*MM)

    # PART 3: result first, then what it means for a manager.
    p.section('3','Results & Implications',M,650*MM,size=34)
    p.text('FORECAST RESULTS',M,672*MM,size=19,bold=True,color=ACCENT)
    p.text('MAE',245*MM,672*MM,size=17,bold=True,align='right')
    p.text('RMSE',286*MM,672*MM,size=17,bold=True,align='right')
    model_rows=[('7-day mean','Baseline (7-day mean)','#B4C5AE'),
                ('kNN','kNN (k=25)','#B4C5AE'),
                ('Random Forest','Random forest','#6D9C80'),
                ('CatBoost','CatBoost',ACCENT),
                ('RF + CatBoost','RF + CatBoost (50/50)',ORANGE)]
    bx,bw=105*MM,96*MM
    for i,(name,key,color) in enumerate(model_rows):
        yy=(686+i*11)*MM;rr=scores[key]
        p.text(name,M,yy,size=19,bold=i==4)
        p.rect(bx,yy+4,bw*rr['mae']/.34,15,color)
        p.text(f"{rr['mae']:.4f}",245*MM,yy,size=20,bold=i==4,align='right')
        p.text(f"{rr['rmse']:.4f}",286*MM,yy,size=19,bold=i==4,align='right')
    p.text('MAE = mean absolute error; both metrics: lower is better.',M,745*MM,size=16,color=MUTED)
    p.text(f'{gain:.1f}% lower MAE',M,757*MM,size=32,bold=True,color=ACCENT)
    p.text('vs. the baseline; CatBoost alone is nearly tied.',M,773*MM,size=18,color=MUTED)

    p.text('FROM FORECAST TO DECISION',311*MM,672*MM,size=19,bold=True,color=ACCENT)
    p.text('98.3%',311*MM,688*MM,size=32,bold=True,color=ORANGE)
    p.arrow(367*MM,695*MM,400*MM,695*MM,color=MUTED)
    p.text('55.0%',413*MM,688*MM,size=32,bold=True,color=ACCENT)
    p.text('of cases recommend a discount',311*MM,704*MM,size=19)
    p.text('Model + support',311*MM,717*MM,size=18,color=MUTED)
    p.text('+ stock rules',413*MM,717*MM,size=18,color=MUTED)
    p.para('Compare supported rates using predicted sales x price rate (zero cost). Stockout rules constrain the choice. One eligible product-day selects <b>5% off</b>.',
           311*MM,733*MM,260*MM,size=20,leading=25,max_h=94)
    p.text('Example: Store 18 / Product 11 / 26 June 2024',311*MM,764*MM,size=15,color=MUTED)
    p.text('Recommendations remain untested in real stores.',311*MM,773*MM,size=17,color=MUTED)
    p.rect(M,792*MM,550*MM,25*MM,PALE_GREEN,radius=6)
    p.text('IMPLICATION',M+17,798*MM,size=18,bold=True,color=ACCENT)
    p.para('Use forecasts as decision support. Validate pricing effects in a controlled store pilot, with inventory and cost data.',
           103*MM,797*MM,455*MM,size=21,leading=26,max_h=55)
    p.text('Limits: five stores; retrospective selection; reused benchmark. Value is a proxy, not currency. No causal sales, revenue or waste gain shown.',M,824*MM,size=13.5,color=MUTED)
    p.text('Source: Dingdong FreshRetailNet-50K, CC BY 4.0. Final notebook: pricing-ml-ap @ a7e26dd. Decorative produce artwork generated with AI.',M,833*MM,size=11,color=MUTED)
    p.c.linkURL(SOURCE_DATA,(M,H-838*MM,250*MM,H-830*MM),relative=0,thickness=0)
    p.c.save()
    r=PdfReader(result)
    assert len(r.pages)==1
    text=r.pages[0].extract_text()
    assert 'QR' not in text
    for word in ['Data Cleaning','CatBoost','0.2892','7.9%','Results & Implications']:assert word in text
    (TMP/'fresh_ensemble_revision_layout.json').write_text(json.dumps(p.boxes,indent=2))
    files=['results/test_summary.csv','results/visuals/recommendation_distribution.csv','results/visuals/one_product_discount_decision.csv','assets/poster/fresh_produce_hero_v3.png','assets/poster/fresh_produce_hero_v3_prompt.md']
    receipt={'final_notebook_commit':PIN,'final_notebook_sha256':NOTEBOOK_SHA256,'final_notebook_url':NOTEBOOK,
             'pdf_sha256':hashlib.sha256(result.read_bytes()).hexdigest(),
             'sections':['Problem / Data Cleaning','Models','Results & Implications'],
             'model_methods':['7-day mean baseline','kNN','Random Forest','CatBoost','50/50 RF + CatBoost'],
             'model_section_mm':[326,638],'header':'New AI-generated decorative produce art; compact green title area',
             'no_qr_code':True,'gain_percent':gain,'test_mae':blend['mae'],'test_rmse':blend['rmse'],
             'policy_discount_share_percent':shares,'cleaning_claim':'Checks and rules, not invented per-rule changed-row counts',
             'selection':'Equal weights retained for simplicity; 60/40 has slightly lower mean validation MAE',
             'business_claim':'Predictive benchmark and model scenarios; no causal revenue/profit/waste gain shown',
             'source_hashes':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}
    (OUT/'fresh_ensemble_revision_evidence.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(result)

if __name__=='__main__':build()
