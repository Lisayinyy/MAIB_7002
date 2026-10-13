#!/usr/bin/env python3
"""Make six evidence-led, reusable poster figures from the final notebook.

Uses saved predictions/experiments; no fitting or counterfactual experiment is run.
uv run --no-project --with matplotlib==3.11.2 --with pandas==3.0.6 \
  --with pillow --with pypdf==6.20.1 python scripts/build_key_findings_visuals.py
"""
from __future__ import annotations
import argparse, hashlib, io, json, re, subprocess, urllib.request
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[1]
PIN='a7e26dd979af88b4c14f98a7f029f6a64bffaedf'
NB_HASH='a42326bed278c3b70c3469dd508ab563e6586dbdd84033117663c86a25033859'
EVIDENCE='2d811ebc6262684c7ec991bafca4f2f46d123eb2'
NB_URL=f'https://github.com/Lisayinyy/MAIB_7002/blob/{PIN}/discount_sales_prediction.ipynb'
GREEN='#0C5839'; LIGHT='#E7F0DC'; LIME='#D4ED62'; ORANGE='#F27339'
CREAM='#FFFDF3'; INK='#163D29'; MUTED='#647567'; GRAY='#A7B4A6'; TEAL='#559F9A'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':15,'axes.titlesize':20,
    'axes.labelsize':15,'xtick.labelsize':13,'ytick.labelsize':14,'text.color':INK,
    'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED,
    'axes.edgecolor':'#D5DFD0','axes.facecolor':CREAM,'figure.facecolor':CREAM,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
    'svg.fonttype':'none','savefig.facecolor':CREAM})

class Evidence:
    def __init__(self,args):self.args=args;self.hashes={}
    def raw(self,path):
        b=(subprocess.check_output(['git','--git-dir='+self.args.git_dir,'show',self.args.evidence_commit+':'+path])
           if self.args.git_dir else (ROOT/path).read_bytes())
        self.hashes[path]=hashlib.sha256(b).hexdigest();return b
    def frame(self,path,**kw):return pd.read_csv(io.BytesIO(self.raw(path)),**kw)


def page(number,title,subtitle,core=True):
    f=plt.figure(figsize=(16,9),layout=None)
    f.text(.055,.95,f'{number:02d}  /  '+('POSTER MAIN FIGURE' if core else 'SUPPLEMENTARY EXPERIMENT'),
           fontsize=12,weight='bold',color=GREEN)
    f.text(.055,.885,title,fontsize=29,weight='bold')
    f.text(.055,.835,subtitle,fontsize=16,color=MUTED)
    return f


def note(f,takeaway,source):
    f.text(.055,.072,takeaway,fontsize=16,weight='bold',color=GREEN)
    f.text(.055,.032,source,fontsize=10,color=MUTED)
    f.text(.945,.032,'Final notebook / a7e26dd',ha='right',fontsize=10,color=MUTED,url=NB_URL)


def clean(ax,grid='x'):
    ax.spines[['top','right']].set_visible(False)
    ax.set_axisbelow(True);ax.grid(axis=grid,color='#DCE3D7',alpha=.8,lw=.8)
    ax.tick_params(length=0,pad=8)


def box(ax,x,y,w,h,label,fc=LIGHT,fontsize=16,color=INK,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.012',
                              linewidth=0,facecolor=fc,transform=ax.transAxes))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fontsize,color=color,
            weight='bold' if bold else 'normal',transform=ax.transAxes,linespacing=1.45)


def arrow(ax,start,end,color=GREEN):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=15,
                color=color,lw=2,transform=ax.transAxes,connectionstyle='arc3'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--git-dir');ap.add_argument('--evidence-commit',default=EVIDENCE)
    ap.add_argument('--notebook',type=Path);ap.add_argument('--output',type=Path,default=ROOT/'output/visuals/key_findings')
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    e=Evidence(args)
    nb=args.notebook.read_bytes() if args.notebook else urllib.request.urlopen(
        f'https://raw.githubusercontent.com/Lisayinyy/MAIB_7002/{PIN}/discount_sales_prediction.ipynb',timeout=45).read()
    assert hashlib.sha256(nb).hexdigest()==NB_HASH
    notebook=json.loads(nb)
    sel=e.frame('results/visuals/sample_selection.csv')
    scores=e.frame('results/test_summary.csv',index_col=0)
    pred=e.frame('results/test_predictions.csv')
    response=e.frame('results/visuals/discount_response_and_support.csv')
    policy=e.frame('results/visuals/recommendation_distribution.csv')
    case=e.frame('results/visuals/one_product_discount_decision.csv')
    wx=e.frame('results/knn_stage1_with_weather_summary.csv')
    nx=e.frame('results/knn_stage1_summary.csv')
    curves=e.frame('results/catboost_learning_curves.csv',header=[0,1],index_col=0).dropna(how='all')
    curves.index=pd.to_numeric(curves.index)
    assert len(pred)==2184 and response.total_histories.eq(312).all()
    bands=[]
    for name,mask in [('<5% off',pred.discount_next>.95),('5% to <20% off',(pred.discount_next>.8)&(pred.discount_next<=.95)),('20%+ off',pred.discount_next<=.8)]:
        g=pred.loc[mask];bands.append({'discount_band':name,'n':len(g),'mean_sales':g.target_sales.mean(),
            'baseline_mae':abs(g.target_sales-g.pred_baseline).mean(),'ensemble_mae':abs(g.target_sales-g.pred_ensemble).mean()})
    bands=pd.DataFrame(bands);bands.to_csv(out/'error_by_discount.csv',index=False)
    # Counts come from the final notebook's stored Table 5, not invented examples.
    output_text='\n'.join(''.join(o.get('data',{}).get('text/plain',[])) for o in notebook['cells'][61]['outputs'])
    counts={key:int(re.search(r'^'+re.escape(key)+r'\s+(\d+)',output_text,re.M).group(1))
            for key in ['sold out today','no stockout for a week','stock turning over']}
    assert sum(counts.values())==2184
    figs=[];specs=[]

    f=page(1,'Choose histories with enough discount variation',
           'FreshRetailNet links daily sales, discounts and stockouts for each store-product pair.')
    ax=f.add_axes([.055,.35,.89,.42]);ax.axis('off')
    values=[int(sel.histories.iloc[i]) for i in [0,3,4]]
    labels=['PUBLIC DATA','PASS THE FILTERS','WORKING COHORT']
    subs=['50,000 store-product histories','16,684 eligible histories','312 histories in five stores']
    for i,(v,l,s) in enumerate(zip(values,labels,subs)):
        x=i*.355;w=.285
        box(ax,x,.29,w,.59,'',fc=LIME if i==2 else LIGHT)
        ax.text(x+.03,.73,l,fontsize=13,weight='bold',transform=ax.transAxes,color=GREEN)
        ax.text(x+.03,.52,f'{v:,}',fontsize=42,weight='bold',transform=ax.transAxes)
        ax.text(x+.03,.38,s.split(' ',1)[1],fontsize=15,transform=ax.transAxes)
        if i<2:arrow(ax,(x+.292,.59),(x+.341,.59))
    f.text(.065,.35,'Require repeated price changes',fontsize=20,weight='bold')
    f.text(.065,.28,'15-85% discounted days; at least six discount-state switches;\nzero sales on fewer than 20% of days.',fontsize=17,linespacing=1.6)
    f.text(.63,.35,'Select the top five stores',fontsize=20,weight='bold')
    f.text(.63,.28,'Rank stores by eligible histories.\nAll five happen to be in one city.',fontsize=17,linespacing=1.6)
    f.text(.065,.165,'312 histories  |  122 products  |  25,896 training/validation rows  |  2,184 final-week cases',fontsize=18,weight='bold',color=GREEN)
    note(f,'This cohort supports comparison; it does not represent every store or product.',
         'Discounted day: price rate <0.95. Retrospective selection across the 90-day training file. Diagram stages are not area-scaled.')
    figs.append(('01_sample_selection',f));specs.append({'figure':1,'cells':[3,4,5],'purpose':'Why this dataset and five-store cohort'})

    f=page(2,'Forecasts improved; deep-discount days remained difficult',
           'Same 2,184 final-week cases. MAE is mean absolute error in normalized observed sales.')
    a=f.add_axes([.18,.26,.33,.48]);clean(a)
    keys=['Baseline (7-day mean)','kNN (k=25)','Random forest','CatBoost','RF + CatBoost (50/50)']
    labels=['7-day average','kNN','Random Forest','CatBoost','50/50 ensemble']
    vals=[scores.loc[k,'mae'] for k in keys]
    a.barh(range(5),vals,color=[GRAY,'#BAC9B4',GREEN,TEAL,ORANGE],height=.58)
    a.set_yticks(range(5),labels);a.invert_yaxis();a.set_xlim(0,.365);a.set_xlabel('Final-week MAE (lower is better)')
    for i,v in enumerate(vals):a.text(v+.009,i,f'{v:.4f}',va='center',fontsize=16,weight='bold' if i==4 else 'normal')
    a.set_title('Overall model comparison',loc='left',pad=20)
    b=f.add_axes([.60,.30,.34,.44]);clean(b,grid='y');x=np.arange(3)
    b.bar(x-.18,bands.baseline_mae,.34,color=GRAY,label='7-day average')
    b.bar(x+.18,bands.ensemble_mae,.34,color=ORANGE,label='50/50 ensemble')
    b.set_xticks(x,[f'{r.discount_band}\nn={r.n:,}' for r in bands.itertuples()]);b.set_ylim(0,.69);b.set_ylabel('Final-week MAE')
    b.set_title('Errors by observed discount',loc='left',pad=20)
    for i,row in bands.iterrows():
        for dx,col in [(-.18,'baseline_mae'),(.18,'ensemble_mae')]:b.text(i+dx,row[col]+.018,f'{row[col]:.3f}',ha='center',fontsize=12)
    b.legend(loc='upper left',frameon=False,fontsize=12,ncols=2,bbox_to_anchor=(-.02,1.015))
    gain=100*(1-vals[-1]/vals[0])
    f.text(.18,.165,f'{gain:.1f}% lower MAE than baseline',fontsize=22,weight='bold',color=GREEN)
    f.text(.60,.165,'Higher absolute errors also accompany higher sales.',fontsize=14,color=MUTED)
    note(f,'50% Random Forest + 50% CatBoost; CatBoost alone is almost tied.',
         'Final period: 26 Jun-2 Jul 2024, reused during development. Deep-discount MAE is not a scale-free reliability or causal measure.')
    figs.append(('02_forecast_results_and_errors',f));specs.append({'figure':2,'cells':[40,44,45,46,47,48],'purpose':'Forecast accuracy and the difficult discount segment'})

    f=page(3,'The largest predicted gains have the weakest support',
           'Hold each case\'s history fixed and vary the proposed discount. These are model scenarios.')
    a=f.add_axes([.09,.47,.65,.29]);b=f.add_axes([.09,.21,.65,.19],sharex=a)
    clean(a,'y');clean(b,'y');x=response.percent_off.to_numpy()
    a.plot(x,response.sales_vs_no_discount,'o-',lw=3,color=GREEN,label='Predicted sales')
    a.plot(x,response.value_vs_no_discount,'s-',lw=3,color=ORANGE,label='Predicted value = sales x rate')
    a.axhline(1,color=MUTED,lw=1,ls='--');a.set_ylim(.92,2.58);a.set_ylabel('Ratio to full-price forecast')
    a.legend(frameon=False,loc='upper left',fontsize=13);a.tick_params(labelbottom=False)
    a.text(40.8,response.sales_vs_no_discount.iloc[-1],f'{response.sales_vs_no_discount.iloc[-1]:.2f}x',color=GREEN,va='center',weight='bold')
    a.text(40.8,response.value_vs_no_discount.iloc[-1],f'{response.value_vs_no_discount.iloc[-1]:.2f}x',color=ORANGE,va='center',weight='bold')
    b.bar(x,response.support_percent,width=3.5,color=[GREEN]*7+[ORANGE])
    b.set_ylim(0,118);b.set_xlim(-2,45);b.set_xticks(x,[f'{v:.0f}%' for v in x]);b.set_xlabel('Proposed discount (% off full price)');b.set_ylabel('Histories with\nsupport (%)')
    for xx,v in zip(x,response.support_percent):b.text(xx,v+3,f'{v:.1f}',ha='center',fontsize=11)
    f.text(.79,.68,'At 40% off',fontsize=24,weight='bold')
    f.text(.79,.59,'2.32x',fontsize=40,weight='bold',color=GREEN)
    f.text(.79,.55,'predicted sales',fontsize=17)
    f.text(.79,.44,'15 / 312',fontsize=33,weight='bold',color=ORANGE)
    f.text(.79,.39,'histories support\nthis rate (4.8%)',fontsize=17,linespacing=1.5)
    f.text(.79,.23,'Support requires 3+\ntraining days within\n0.025 of the price rate.',fontsize=14,linespacing=1.6,color=MUTED)
    note(f,'A strong predicted response is not evidence that a new discount will cause the same gain.',
         'Ratios compare means over the same 2,184 cases; support denominator is 312 histories. Zero unit cost. Value is a normalized proxy.')
    figs.append(('03_discount_response_and_support',f));specs.append({'figure':3,'cells':[54,57,58],'purpose':'Why a pricing model needs historical support checks'})

    f=page(4,'Stock rules change both whether and how much to discount',
           'Restrict to supported rates, apply the assumed rules, then compare eligible predicted values.')
    ax=f.add_axes([.055,.44,.89,.34]);ax.axis('off')
    states=[('Stockout today >=1 hour','Full price',counts['sold out today'],GRAY),
            ('Else: no stockout for 7 days','Discount required; choose depth',counts['no stockout for a week'],ORANGE),
            ('Otherwise','Model chooses the eligible rate',counts['stock turning over'],GREEN)]
    for i,(condition,decision,n,color) in enumerate(states):
        y=.7-i*.32
        box(ax,.015,y,.34,.25,condition,fontsize=16)
        arrow(ax,(.365,y+.125),(.435,y+.125),color)
        box(ax,.44,y,.37,.25,decision,fc=color,color='white' if color==GREEN else INK,fontsize=16,bold=True)
        ax.text(.86,y+.135,f'{n/2184:.1%}',fontsize=27,weight='bold',va='center',transform=ax.transAxes,color=color if color!=GRAY else INK)
        ax.text(.86,y+.01,f'{n:,} cases',fontsize=12,transform=ax.transAxes,color=MUTED)
    a=f.add_axes([.25,.19,.68,.19]);a.set_xlim(0,100);a.set_ylim(-.55,1.55);a.axis('off')
    colors={'Full price':GRAY,'Moderate discount':TEAL,'Deep discount':ORANGE}
    for j,strategy in enumerate(['Supported model only','With stock rules']):
        y=1-j;offset=0
        a.text(-2,y,'Supported model only' if j==0 else 'With stock rules',ha='right',va='center',fontsize=15,weight='bold')
        for band in colors:
            r=policy[(policy.strategy==strategy)&(policy.band==band)].iloc[0];v=float(r.percent)
            a.barh(y,v,left=offset,color=colors[band],height=.48)
            if v>7:a.text(offset+v/2,y,f'{v:.1f}%',ha='center',va='center',fontsize=15,color=INK,weight='bold')
            else:a.text(offset+v/2,y+.36,f'{v:.1f}%',ha='center',fontsize=11)
            offset+=v
        assert abs(offset-100)<1e-6
    for j,(s,col) in enumerate(colors.items()):
        f.text(.26+j*.235,.158,'\u25a0 '+{'Full price':'Full price','Moderate discount':'>0% to <20% off','Deep discount':'20%+ off'}[s],fontsize=14,color=col)
    note(f,'Any discount: 98.3% of cases becomes 55.0%. This is changed behavior, not proven business gain.',
         'Same 2,184 cases, zero assumed unit cost. Stockout history does not establish inventory age or freshness. Notebook Table 5 supplies counts.')
    figs.append(('04_stock_rules_and_recommendations',f));specs.append({'figure':4,'cells':[54,60,61,62],'purpose':'Decision rules and their effect on recommendation composition','state_counts':counts})

    f=page(5,'For one product-day, predicted value peaks at 5% off',
           'Store 18 / Product 11 / 26 Jun 2024. Five candidates pass the historical-support check.')
    a=f.add_axes([.10,.25,.57,.47]);clean(a,'y')
    good=case[case.historically_supported.astype(str).str.lower().eq('true')]
    a.plot(good.percent_off,good.pred_value,'o-',color=GREEN,lw=3,ms=9)
    ref=float(good.loc[good.discount.eq(1),'pred_value'].iloc[0]);a.axhline(ref,color=MUTED,ls='--',lw=1)
    a.set_xlim(-1.8,22);a.set_ylim(1.105,1.218);a.set_xticks(good.percent_off,[f'{x:.0f}%' for x in good.percent_off])
    a.set_ylabel('Predicted value proxy (normalized)');a.set_xlabel('Candidate discount (% off full price)')
    selected=good[good.chosen.astype(str).str.lower().eq('true')].iloc[0]
    a.scatter([selected.percent_off],[selected.pred_value],s=300,color=ORANGE,zorder=5)
    for r in good.itertuples():a.annotate(f'{r.pred_value:.3f}',(r.percent_off,r.pred_value),xytext=(0,14),textcoords='offset points',ha='center',fontsize=16,weight='bold' if r.chosen else 'normal')
    a.text(21.7,ref-.001,'Full-price\nreference',ha='right',va='top',fontsize=12,color=MUTED)
    f.text(.73,.69,'5% off',fontsize=45,weight='bold',color=ORANGE)
    f.text(.73,.63,'Selected among eligible rates',fontsize=17,weight='bold')
    f.text(.73,.52,'Price rate 0.95\nx predicted sales 1.261\n= predicted value 1.198',fontsize=19,linespacing=1.6)
    f.text(.73,.31,'25%, 30% and 40% off\nare excluded: no training days\nwithin the support window.',fontsize=15,color=MUTED,linespacing=1.7)
    f.text(.10,.155,'Support days: '+ '   |   '.join(f'{int(r.percent_off)}% off: {int(r.support_days)}' for r in good.itertuples()),fontsize=15)
    note(f,'The model supplies a concrete rate; its predicted gain still needs a real-world test.',
         'Zero unit cost. Deterministic first eligible worked case, not selected for a large gain. Proxy values are not currency or realized revenue.')
    figs.append(('05_one_product_discount_choice',f));specs.append({'figure':5,'cells':[54,57,61],'purpose':'Concrete illustration using the final scenario code','case':'Store 18, product 11, 2024-06-26'})

    f=page(6,'More features and more trees did not always help',
           'Use this in the notebook or as a supporting figure when explaining the modelling choices.',core=False)
    a=f.add_axes([.095,.28,.37,.45]);clean(a,'y');w=wx[wx.weighting.eq('equal')];n=nx[nx.weighting.eq('equal')]
    a.plot(w.k,w.mae,'o-',color=GRAY,lw=3,label='With weather (13 inputs)')
    a.plot(n.k,n.mae,'o-',color=GREEN,lw=3,label='Without weather (10 inputs)')
    a.set_ylim(.3,.37);a.set_xlabel('Number of neighbours, k');a.set_ylabel('Mean validation MAE')
    a.set_title('Removing weather improved kNN',loc='left',pad=18);a.legend(frameon=False,fontsize=12)
    a.axvline(25,color=ORANGE,ls=':',lw=2)
    m0=float(w.loc[w.k.eq(25),'mae'].iloc[0]);m1=float(n.loc[n.k.eq(25),'mae'].iloc[0])
    f.text(.095,.17,f'At k=25: {m0:.4f} -> {m1:.4f}',fontsize=20,weight='bold',color=GREEN)
    b=f.add_axes([.59,.28,.35,.45]);clean(b,'y');iteration=curves.index.to_numpy()+1
    b.plot(iteration,curves[('lr=0.03','train')],color=TEAL,lw=2,label='Training')
    b.plot(iteration,curves[('lr=0.03','validation')],color=ORANGE,lw=2,label='Validation')
    b.axvline(500,color=GREEN,ls='--',lw=1.6,label='Chosen 500 trees')
    b.set_xlabel('Boosting trees (zoom: 100-3,000)');b.set_ylabel('MAE');b.set_ylim(.23,.34);b.set_xlim(100,3000)
    b.set_title('Training improves; validation levels off',loc='left',pad=18);b.legend(frameon=False,fontsize=12)
    f.text(.59,.17,'Learning rate 0.03; final validation fold',fontsize=17,weight='bold',color=GREEN)
    note(f,'Choose complexity using later dates, not how well the model fits its training data.',
         'Left: same five time-ordered folds, equal feature weights. Right: recorded CatBoost learning curve; final settings selected across five folds.')
    figs.append(('06_optional_model_complexity',f));specs.append({'figure':6,'cells':[14,18,20,34,36],'purpose':'Supplementary modelling lesson, not a primary business figure'})

    # PNGs for quick placement; SVGs preserve editable vector text and marks.
    pdf=ROOT/'output/pdf/FreshRetail_key_findings_visuals.pdf';pdf.parent.mkdir(parents=True,exist_ok=True)
    with PdfPages(pdf,metadata={'Title':'FreshRetail: key findings to visualize','Author':'MAIB7002 Group Q','Subject':'Five main figures and one supplementary figure; final pricing-ml-ap notebook'}) as book:
        for name,f in figs:
            for txt in f.texts:
                if '\n' in txt.get_text():
                    txt.set_verticalalignment('top')
            svg=out/(name+'.svg')
            f.savefig(svg)
            svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
            f.savefig(out/(name+'.png'),dpi=300)
            book.savefig(f)
            plt.close(f)
    thumbnails=[Image.open(out/(name+'.png')).convert('RGB').resize((1400,788),Image.Resampling.LANCZOS) for name,_ in figs]
    contact=Image.new('RGB',(2824,2412),'#D7E1CF')
    for i,im in enumerate(thumbnails):contact.paste(im,(8+(i%2)*1408,8+(i//2)*800))
    contact.save(out/'all_key_findings.png')
    receipt={'final_notebook_commit':PIN,'notebook_sha256':NB_HASH,'notebook_url':NB_URL,
        'evidence_commit':args.evidence_commit if args.git_dir else 'local repository files; see exact hashes',
        'source_hashes':e.hashes,'figures':specs,'actual_training_performed':False,
        'derived_error_by_discount':bands.to_dict('records'),
        'versions':{'matplotlib':matplotlib.__version__,'pandas':pd.__version__},
        'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest()}
    (out/'evidence.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'pdf':str(pdf),'figures':len(figs),'directory':str(out)},indent=2))

if __name__=='__main__':main()
