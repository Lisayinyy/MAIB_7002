#!/usr/bin/env python3
"""Independent read-only baseline verification for the teammate's cohort."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def scores(y, pred):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    err = y-pred
    return {'n':len(y),'MAE':float(np.abs(err).mean()),
            'WAPE_percent':float(np.abs(err).sum()/np.abs(y).sum()*100),
            'MSE':float((err**2).mean()), 'RMSE':float(np.sqrt((err**2).mean()))}


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',default=str(root/'data/raw/train.parquet'))
    parser.add_argument('--eval',default=str(root/'data/raw/eval.parquet'))
    parser.add_argument('--cohort',default=str(root/'data/metadata/teammate_selected_series.csv'))
    parser.add_argument('--output',default=str(root/'outputs/teammate_protocol/baseline_verification.json'))
    args=parser.parse_args()
    keys=['store_id','product_id']
    columns=keys+['dt','sale_amount','discount']
    cohort=pd.read_csv(args.cohort)[keys]
    assert len(cohort)==312 and not cohort.duplicated().any()
    frames=[]
    for split,path in [('train',args.train),('eval',args.eval)]:
        frame=pd.read_parquet(path,columns=columns).merge(cohort,on=keys,how='inner',validate='many_to_one')
        frame['split']=split
        frames.append(frame)
    df=pd.concat(frames,ignore_index=True)
    df['dt']=pd.to_datetime(df.dt)
    df=df.sort_values(keys+['dt']).reset_index(drop=True)
    assert not df.duplicated(keys+['dt']).any()
    assert df.groupby(keys).dt.diff().dropna().eq(pd.Timedelta(days=1)).all()
    grp=df.groupby(keys,sort=False)
    df['seven_day_avg']=grp.sale_amount.transform(lambda x:x.shift(1).rolling(7,min_periods=7).mean())
    df['same_weekday']=grp.sale_amount.shift(7)
    assert df.groupby(keys).size().eq(97).all()
    test=df[df.dt.between('2024-06-26','2024-07-02')].copy()
    assert len(test)==2184 and test.groupby(keys).size().eq(7).all()
    assert test[['seven_day_avg','same_weekday']].notna().all().all()
    test['discount_band']=pd.cut(test.discount,[-np.inf,.8,.95,np.inf],right=True,labels=['deep','moderate','none_or_light'])
    bands=test.groupby('discount_band',observed=True).size().to_dict()
    folds=[('2024-05-22','2024-05-28'),('2024-05-29','2024-06-04'),('2024-06-05','2024-06-11'),('2024-06-12','2024-06-18'),('2024-06-19','2024-06-25')]
    result={'definition':'Daily one-step-ahead evaluation; earlier realized sales within each validation/test week update the subsequent day baseline. Features use only previous dates.','rows':{'train_raw':int((df.split=='train').sum()),'eval_raw':int((df.split=='eval').sum()),'model_ready_train':int(((df.split=='train')&df.seven_day_avg.notna()).sum()),'test':len(test)},'discount_band_definition':{'deep':'discount <= 0.80','moderate':'0.80 < discount <= 0.95','none_or_light':'discount > 0.95'},'discount_band_counts':bands,'test':{},'test_by_discount':{},'cv_folds':[],'cv_mean_of_folds':{},'cv_pooled':{},'report_comparison':{}}
    for model in ['seven_day_avg','same_weekday']:
        result['test'][model]=scores(test.sale_amount,test[model])
        result['test_by_discount'][model]={str(k):scores(sub.sale_amount,sub[model]) for k,sub in test.groupby('discount_band',observed=True)}
    for start,end in folds:
        sub=df[df.dt.between(start,end)]
        assert len(sub)==2184
        result['cv_folds'].append({'start':start,'end':end,**{m:scores(sub.sale_amount,sub[m]) for m in ['seven_day_avg','same_weekday']}})
    pooled=df[df.dt.between(folds[0][0],folds[-1][1])]
    for model in ['seven_day_avg','same_weekday']:
        result['cv_mean_of_folds'][model]={k:float(np.mean([fold[model][k] for fold in result['cv_folds']])) for k in ['MAE','WAPE_percent','MSE','RMSE']}
        result['cv_pooled'][model]=scores(pooled.sale_amount,pooled[model])
    target={'seven_day_avg':{'MAE':.314,'WAPE_percent':32.4,'MSE':.271,'RMSE':.521},'same_weekday':{'MAE':.400,'WAPE_percent':41.3,'MSE':.381,'RMSE':.617}}
    for model,expected in target.items():
        result['report_comparison']['test_'+model]={k:{'actual':result['test'][model][k],'reported':v,'matches_reported_rounding':round(result['test'][model][k],1 if k=='WAPE_percent' else 3)==v} for k,v in expected.items()}
    expected_cv={'MAE':.328,'WAPE_percent':34.5,'MSE':.270}
    result['report_comparison']['cv_seven_day_avg']={k:{'actual_fold_mean':result['cv_mean_of_folds']['seven_day_avg'][k],'actual_pooled':result['cv_pooled']['seven_day_avg'][k],'reported':v,'fold_mean_matches_reported_rounding':round(result['cv_mean_of_folds']['seven_day_avg'][k],1 if k=='WAPE_percent' else 3)==v} for k,v in expected_cv.items()}
    result['report_comparison']['discount_band_match']=(bands=={'deep':241,'moderate':757,'none_or_light':1186})
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
