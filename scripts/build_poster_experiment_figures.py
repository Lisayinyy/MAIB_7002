#!/usr/bin/env python3
"""Reflow empirical notebook plots for A1; no models are trained.
Final cells19/25/35/39/61 and one saved PhaseTwo case, not model cartoons.
"""
from pathlib import Path
import hashlib,json,base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'assets/poster/experiments'
CREAM='#FFFDF3';INK='#163D29';GREEN='#087F4B';ORANGE='#F27339';MUTED='#53685A';GRAY='#A6B3A3';LIGHT='#D6DFC8'
PIN='a7e26dd979af88b4c14f98a7f029f6a64bffaedf';NB_HASH='a42326bed278c3b70c3469dd508ab563e6586dbdd84033117663c86a25033859'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.labelsize':13,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':11.5,'text.color':INK,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,'axes.facecolor':CREAM,'figure.facecolor':CREAM,'axes.edgecolor':'#D3DDCB','axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':CREAM})
hashes={}
def frame(name,**kw):
 p=ROOT/name;hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p,**kw)
def fig():
 f,a=plt.subplots(figsize=(7.5,2.3));f.subplots_adjust(left=.14,right=.965,bottom=.27,top=.91)
 a.set_axisbelow(True);a.grid(axis='y',color='#DDE4D7',lw=.7);a.tick_params(length=0,pad=6);return f,a
def save(f,name):
 f.savefig(OUT/(name+'.pdf'));f.savefig(OUT/(name+'.png'),dpi=260);f.savefig(OUT/(name+'.svg'))
 p=OUT/(name+'.svg');p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n');plt.close(f)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 npath=Path('/tmp/poster-final-notebook-a7e26.ipynb')
 if npath.exists():
  nb=npath.read_bytes();assert hashlib.sha256(nb).hexdigest()==NB_HASH;data=json.loads(nb)
  for cell in [19,25,35,39,57,61]:
   for o in data['cells'][cell].get('outputs',[]):
    img=o.get('data',{}).get('image/png')
    if img:(OUT/f'notebook_cell_{cell}_original.png').write_bytes(base64.b64decode(''.join(img)));break
 k=frame('results/knn_stage1_summary.csv');base=float(k[k.k==0].mae.iloc[0]);k=k[k.weighting=='equal'].sort_values('k')
 f,a=fig();a.plot(k.k,k.mae,color=GREEN,marker='o',lw=2.5,ms=5);a.axhline(base,color=GRAY,ls='--',lw=1.8)
 a.scatter([25],[float(k[k.k==25].mae.iloc[0])],color=ORANGE,s=58,zorder=4)
 a.annotate('k = 25',xy=(25,float(k[k.k==25].mae.iloc[0])),xytext=(6,14),textcoords='offset points',fontsize=12,color=ORANGE,weight='bold')
 a.text(95,base+.0007,'7-day baseline',ha='right',va='bottom',fontsize=11.5,color=MUTED)
 a.set_xscale('log');a.set_xticks([3,5,10,25,60,100],['3','5','10','25','60','100']);a.minorticks_off();a.set_xlabel('Number of neighbours (k)');a.set_ylabel('Validation MAE');a.set_yticks([.31,.33,.35]);a.set_ylim(.307,.352)
 save(f,'01_knn_validation')
 rf=frame('results/rf_tuning_summary.csv',dtype={'max_depth':str});grid=rf.pivot(index='max_depth',columns='min_leaf',values='mae').reindex(['10','15','20','No limit'])
 f,a=fig();a.grid(False);a.imshow(grid.to_numpy(),cmap='YlGn',aspect='auto',vmin=.3024,vmax=.3053);a.set_xticks(range(4),[str(x) for x in grid.columns]);a.set_yticks(range(4),grid.index);a.set_xlabel('Minimum rows per leaf');a.set_ylabel('Maximum depth')
 for i in range(4):
  for j in range(4):
   value=grid.iloc[i,j];a.text(j,i,f'{value:.4f}',ha='center',va='center',fontsize=13,color=INK if value<.304 else 'white')
 a.add_patch(Rectangle((-.5,-.5),1,1,fill=False,edgecolor=ORANGE,lw=3));save(f,'02_rf_validation_grid')
 lc=frame('results/catboost_learning_curves.csv',header=[0,1],index_col=0)['lr=0.03'];trees=lc.index.to_numpy()+1;keep=trees>=100
 f,a=fig();a.plot(trees[keep],lc.train.to_numpy()[keep],color=GRAY,lw=1.9,label='Training');a.plot(trees[keep],lc.validation.to_numpy()[keep],color=GREEN,lw=2.4,label='Validation')
 a.axvline(500,color=ORANGE,lw=1.5,ls='--');a.text(540,.343,'Locked: 500',fontsize=11.5,color=ORANGE)
 a.set_xlabel('Trees added (learning rate 0.03)');a.set_ylabel('MAE');a.set_xticks([100,1000,2000,3000]);a.set_ylim(.235,.355);a.set_yticks([.24,.28,.32]);a.legend(frameon=False,loc='upper right',ncol=2,handlelength=1.6,columnspacing=1.2);save(f,'03_catboost_learning')
 sw=frame('results/ensemble_weight_sweep.csv');f,a=fig();a.axvspan(40,70,color=LIGHT,alpha=.5);a.plot(sw.rf_weight*100,sw.mae,color=GREEN,lw=2.5,marker='o',ms=4)
 v50=float(sw[sw.rf_weight==.5].mae.iloc[0]);best=sw.loc[sw.mae.idxmin()];a.scatter([50],[v50],s=65,color=ORANGE,zorder=4)
 a.annotate('50/50 chosen',xy=(50,v50),xytext=(47,.3026),textcoords='data',fontsize=12,color=ORANGE,weight='bold',arrowprops={'arrowstyle':'-','color':ORANGE})
 a.text(60,.30108,'Grid minimum: 60/40',ha='center',va='top',fontsize=10.5,color=MUTED);a.set_xticks([0,25,50,75,100]);a.set_xlabel('Random Forest weight (%)');a.set_ylabel('Validation MAE');a.set_yticks([.301,.302,.303]);a.set_ylim(.30078,.3037);save(f,'04_ensemble_weights')
 policy=frame('results/visuals/recommendation_distribution.csv');f,a=fig();f.subplots_adjust(left=.25,right=.965,bottom=.27,top=.77)
 cats=['Full price','Moderate discount','Deep discount'];cols=[LIGHT,GREEN,ORANGE]
 for i,(strategy,label) in enumerate([('Supported model only','Model + support'),('With stock rules','+ stock rules')]):
  left=0
  for cat,col in zip(cats,cols):
   rr=policy[(policy.strategy==strategy)&(policy.band==cat)].iloc[0];v=float(rr.percent);a.barh(i,v,left=left,color=col,height=.47,label=cat if i==0 else None)
   if v>10:a.text(left+v/2,i,f'{v:.1f}%',ha='center',va='center',color=INK if cat=='Full price' else 'white',fontsize=12,weight='bold')
   left+=v
 a.set_yticks([0,1],['Model + support','+ stock rules']);a.invert_yaxis();a.set_xlim(0,100);a.set_xticks([0,50,100]);a.set_xlabel('Recommendation mix (% of 2,184 cases)')
 a.legend(['Full price','0% < off < 20%','20%+ off'],frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.45,1.01),fontsize=10.5,handlelength=1.1,columnspacing=.9);save(f,'05_policy_recommendations')
 case=frame('results/visuals/one_product_discount_decision.csv');case=case[case.eligible].sort_values('percent_off');chosen=case[case.chosen].iloc[0];assert float(chosen.percent_off)==5
 f,a=fig();a.plot(case.percent_off,case.pred_sales,color=GREEN,marker='o',lw=2.3,label='Predicted sales');a.plot(case.percent_off,case.pred_value,color=ORANGE,marker='o',lw=2.3,label='Value proxy');a.scatter([chosen.percent_off],[chosen.pred_value],color=ORANGE,marker='*',s=165,zorder=5)
 a.set_xticks([0,5,10,15,20],['0%','5%','10%','15%','20%']);a.set_xlabel('Supported discount (% off)');a.set_ylabel('Normalized scale');a.set_ylim(1.04,1.51);a.set_yticks([1.1,1.3,1.5]);a.legend(frameon=False,loc='upper left',ncol=2,fontsize=11.5,handlelength=1.5,columnspacing=1.1);save(f,'06_case_discount_depth')
 evidence={'notebook_commit':PIN,'notebook_sha256':NB_HASH,'source_hashes':hashes,'charts':{
 '01_knn_validation':{'notebook_cell':19,'subset':'equal-weight kNN and7-daybaseline'},'02_rf_validation_grid':{'notebook_cell':25,'note':'Numeric heatmap reproduced exactly'},
 '03_catboost_learning':{'notebook_cell':35,'subset':'lr=.03, trees100-3000; zero-based logged iteration +1 displayed as tree count'},'04_ensemble_weights':{'notebook_cell':39,'chosen_rf_weight':.5,'minimum_rf_weight':float(best.rf_weight)},
 '05_policy_recommendations':{'notebook_cell':61,'subset':'Two default policies, same2184cases'},'06_case_discount_depth':{'source':'Saved case from same final ensemble, alongside PhaseTwo','store':18,'product':11,'date':'2024-06-26','chosen_percent_off':5,'pred_sales':float(chosen.pred_sales),'pred_value':float(chosen.pred_value)}},
 'presentation':'Actual saved experiments reflowed with larger fonts for A1; no mechanism cartoons; no retraining or invented data'}
 (OUT/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n');print(OUT)
if __name__=='__main__':main()
