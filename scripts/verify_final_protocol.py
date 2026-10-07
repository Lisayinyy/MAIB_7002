#!/usr/bin/env python3
"""Independently verify this project's executed final protocol.

Run: python scripts/verify_final_protocol.py [--root /path/to/group_project]
Reads original public observations and saved outputs. It only writes
outputs/final_protocol/independent_verification.json; no models are retrained.
The feature reference uses explicit per-row historical slices rather than the
notebook's groupby shift/rolling/expanding implementation.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, ast, hashlib, json
import numpy as np
import pandas as pd

ROW_KEYS=['store_id','product_id','dt']

def independent_features(root):
    cohort=pd.read_csv(root/'data/metadata/teammate_selected_series.csv')[['store_id','product_id']]
    assert len(cohort)==312 and not cohort.duplicated().any()
    cols=['store_id','product_id','dt','sale_amount','discount','holiday_flag','activity_flag','stock_hour6_22_cnt']
    data=pd.concat([pd.read_parquet(root/'data/raw'/f,columns=cols).merge(cohort,how='inner',on=['store_id','product_id'],validate='many_to_one') for f in ['train.parquet','eval.parquet']],ignore_index=True)
    data['dt']=pd.to_datetime(data.dt)
    assert not data.duplicated(ROW_KEYS).any()
    assert len(data)==312*97
    output=[]
    # Direct per-row historical slices, independent of production groupby shift/rolling implementation.
    for (store,product),g in data.groupby(['store_id','product_id']):
        g=g.sort_values('dt').reset_index(drop=True)
        assert len(g)==97 and g.dt.diff().iloc[1:].eq(pd.Timedelta(days=1)).all()
        for pos in range(7,len(g)):
            r=g.iloc[pos].to_dict();history=g.iloc[:pos]
            r.update(weekday=g.iloc[pos]['dt'].dayofweek,
                sales_lag1=float(history.sale_amount.iloc[-1]),
                sales_lag7=float(history.sale_amount.iloc[-7]),
                sales_mean7=float(history.sale_amount.iloc[-7:].mean()),
                discount_lag1=float(history.discount.iloc[-1]),
                stockout_lag1=float(history.stock_hour6_22_cnt.iloc[-1]),
                series_mean=float(history.sale_amount.mean()))
            output.append(r)
    ref=pd.DataFrame(output).sort_values(ROW_KEYS).reset_index(drop=True)
    assert len(ref.loc[ref.dt<='2024-06-25'])==25896
    assert len(ref.loc[ref.dt>='2024-06-26'])==2184
    return ref

def metrics(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    assert y.shape==p.shape and np.isfinite(p).all()
    errors=y-p
    return {'MAE':float(np.mean(np.abs(errors))), 'MSE':float(np.mean(errors**2)),
            'RMSE':float(np.sqrt(np.mean(errors**2))), 'WAPE_percent':float(np.sum(np.abs(errors))/np.sum(np.abs(y))*100)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    ROOT=args.root.resolve()
    OUT=ROOT/'outputs/final_protocol'
    assert (OUT/'run_manifest.json').exists(), 'Run the complete notebook before verification.'
    expected_hashes={'train':'6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d',
                     'eval':'1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e'}
    actual_hashes={}
    for split,expected in expected_hashes.items():
        with (ROOT/f'data/raw/{split}.parquet').open('rb') as handle:
            actual_hashes[split]=hashlib.file_digest(handle,'sha256').hexdigest()
        assert actual_hashes[split]==expected, f'{split} source observations changed'
    K=['store_id','product_id','dt'];SERIES=K[:2]
    F=['discount','weekday','holiday_flag','activity_flag','sales_lag1','sales_lag7','sales_mean7','discount_lag1','stockout_lag1','series_mean']
    MODEL=['Ridge Regression','Decision Tree Regressor','Random Forest Regressor','HistGradientBoostingRegressor','CatBoost Regressor']
    BASE={'7-day average baseline':'sales_mean7','Same-weekday baseline':'sales_lag7'}
    ref=independent_features(ROOT).sort_values(K).reset_index(drop=True)
    feature=pd.read_parquet(OUT/'feature_frame.parquet').sort_values(K).reset_index(drop=True)
    pd.testing.assert_frame_equal(ref[K],feature[K],check_dtype=False)
    assert np.allclose(ref[F+['sale_amount','stock_hour6_22_cnt']],feature[F+['sale_amount','stock_hour6_22_cnt']],rtol=1e-12,atol=1e-12)
    selected=pd.read_csv(OUT/'selected_series.csv').sort_values(SERIES).reset_index(drop=True)
    original=pd.read_csv(ROOT/'data/metadata/teammate_selected_series.csv').sort_values(SERIES).reset_index(drop=True)
    pd.testing.assert_frame_equal(selected[SERIES],original[SERIES],check_dtype=False)
    nb=json.loads((ROOT/'01_freshretail_project.ipynb').read_text())
    code='\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    node=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='make_features')
    ns={'pd':pd,'np':np,'KEYS':SERIES,'ROW_KEYS':K}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'feature_function','exec'),ns)
    daily=pd.read_parquet(OUT/'selected_daily_observations.parquet');cut=pd.Timestamp('2024-06-26')
    changed=daily.copy();changed.loc[changed.dt.ge(cut),['sale_amount','stock_hour6_22_cnt']]=1234567.
    a=ns['make_features'](daily);b=ns['make_features'](changed)
    pd.testing.assert_frame_equal(a.loc[a.dt.le(cut),F],b.loc[b.dt.le(cut),F])
    folds=pd.read_csv(OUT/'folds.csv')
    starts=pd.date_range('2024-05-22',periods=5,freq='7D')
    assert folds.n_train.tolist()==[14976,17160,19344,21528,23712]
    assert folds.n_valid.eq(2184).all()
    for r,start in zip(folds.itertuples(),starts):
     assert pd.Timestamp(r.train_start)==pd.Timestamp('2024-04-04')
     assert pd.Timestamp(r.train_end)==start-pd.Timedelta(days=1)
     assert pd.Timestamp(r.valid_start)==start and pd.Timestamp(r.valid_end)==start+pd.Timedelta(days=6)
    report={'feature_rows':len(feature),'feature_columns_checked':F,'feature_row_identity_match':True,'feature_numeric_match':True,'causality_future_outcome_mutation_pass':True,'cohort_id_match_reconstruction':True,'fold_dates_and_counts_pass':True}
    basecv=pd.read_csv(OUT/'baseline_validation_folds.csv')
    for r in basecv.itertuples():
     start=starts[r.fold-1];v=ref.loc[ref.dt.between(start,start+pd.Timedelta(days=6))]
     m=metrics(v.sale_amount,v[BASE[r.Model]])
     for metric,val in m.items():assert np.isclose(getattr(r,'WAPE_pct' if metric=='WAPE_percent' else metric),val,rtol=1e-10,atol=1e-12)
    report['baseline_cv_metrics_recomputed']=True
    print('Features, cohort, causality, folds and baseline CV: PASS',flush=True)
    # Audit final prediction keys, outcomes, metrics, model choice and application example.
    pred=pd.read_csv(OUT/'test_predictions.csv',parse_dates=['dt']).sort_values(K).reset_index(drop=True)
    testref=ref.loc[ref.dt.ge('2024-06-26')].reset_index(drop=True)
    assert len(pred)==2184 and not pred.duplicated(K).any()
    pd.testing.assert_frame_equal(pred[K],testref[K],check_dtype=False)
    for col in ['sale_amount','discount','activity_flag','stock_hour6_22_cnt']:assert np.allclose(pred[col],testref[col],rtol=1e-12,atol=1e-12)
    for col,reference in BASE.items():assert np.allclose(pred[col],testref[reference],rtol=1e-12,atol=1e-12)
    all_scores=pd.read_csv(OUT/'all_test_scores.csv');assert len(all_scores)==12
    metric_checks=[]
    for r in all_scores.itertuples():
     col=r.Model if r.feature_set=='baseline' else f'{r.feature_set}::{r.Model}'
     assert pred[col].ge(0).all()
     m=metrics(pred.sale_amount,pred[col])
     for metric,val in m.items():assert np.isclose(getattr(r,'WAPE_pct' if metric=='WAPE_percent' else metric),val,rtol=1e-10,atol=1e-12)
     assert r.n==2184
     metric_checks.append({'feature_set':r.feature_set,'Model':r.Model,**m})
    comparison=pd.read_csv(OUT/'model_comparison_test.csv')
    assert comparison.Model.tolist()==list(BASE)+MODEL
    assert np.allclose(comparison.MAE_improvement_vs_7day_pct,100*(1-comparison.MAE/comparison.MAE.iloc[0]))
    trials=pd.read_csv(OUT/'validation_trials.csv');assert len(trials)==150
    assert trials.n.eq(2184).all()
    for row in trials.itertuples():assert row.n_train==folds.loc[folds.fold.eq(row.fold),'n_train'].iloc[0]
    assert trials.groupby(['feature_set','Model','candidate']).fold.nunique().eq(5).all()
    group=trials.groupby(['feature_set','Model','candidate','parameters'],sort=False).MAE.mean().reset_index()
    expected=group.sort_values(['MAE','candidate'],kind='stable').drop_duplicates(['feature_set','Model'])
    chosen=pd.read_csv(OUT/'chosen_parameters.csv')
    joined=chosen.merge(expected,on=['feature_set','Model'],suffixes=('_saved','_audit'),validate='one_to_one')
    assert len(joined)==10 and joined.candidate_saved.eq(joined.candidate_audit).all()
    assert np.allclose(joined.MAE_saved,joined.MAE_audit,rtol=1e-12,atol=1e-12)
    for r in joined.itertuples():assert json.loads(r.parameters_saved)==json.loads(r.parameters_audit)
    selection=json.loads((OUT/'selection_before_test.json').read_text())
    selected_name=expected.loc[expected.feature_set.eq('full')].sort_values('MAE').iloc[0].Model
    assert selection['selected_by_validation']==selected_name
    assert comparison.loc[comparison.selected_by_validation,'Model'].tolist()==[selected_name]
    for r in chosen.itertuples():assert selection['parameters'][f'{r.feature_set}::{r.Model}']==json.loads(r.parameters)
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    assert manifest['selection']==selection and manifest['n_train']==25896 and manifest['n_test']==2184
    # Fingerprint uses the original date-first saved order, exactly as manifest documents.
    original_pred=pd.read_csv(OUT/'test_predictions.csv',float_precision='round_trip')
    fingerprint=hashlib.sha256(original_pred[K+['sale_amount']].to_csv(index=False).encode()).hexdigest()
    assert fingerprint==manifest['test_row_target_sha256']
    ab=pd.read_csv(OUT/'promotion_ablation_test.csv')
    for r in ab.itertuples():
     full=all_scores.loc[all_scores.Model.eq(r.Model)&all_scores.feature_set.eq('full'),'MAE'].iloc[0]
     reduced=all_scores.loc[all_scores.Model.eq(r.Model)&all_scores.feature_set.eq('without_target_promotion'),'MAE'].iloc[0]
     assert np.isclose(r.full,full) and np.isclose(r.without_target_promotion,reduced)
     assert np.isclose(r.test_MAE_improvement_with_promotion_pct,100*(1-full/reduced))
     for fs,field in [('full','CV_full_MAE'),('without_target_promotion','CV_without_promotion_MAE')]:
      cv=expected.loc[expected.Model.eq(r.Model)&expected.feature_set.eq(fs),'MAE'].iloc[0]
      assert np.isclose(getattr(r,field),cv)
    # Recompute every diagnostic group and metric from saved row predictions.
    slices=pd.read_csv(OUT/'test_slice_scores.csv')
    for r in slices.itertuples():
     if r.dimension=='discount_band':mask=pred.discount_band.eq(r.slice)
     elif r.dimension=='date':mask=pred.dt.eq(pd.Timestamp(r.slice))
     elif r.dimension=='store':mask=pred.store_id.eq(int(r.slice))
     elif r.dimension=='stockout':mask=pred.stock_hour6_22_cnt.eq(0) if r.slice=='No recorded stockout' else pred.stock_hour6_22_cnt.gt(0)
     else:raise AssertionError(r.dimension)
     part=pred.loc[mask];col=r.Model if r.Model in BASE else f'full::{r.Model}'
     assert r.n==len(part)
     for metric,val in metrics(part.sale_amount,part[col]).items():assert np.isclose(getattr(r,'WAPE_pct' if metric=='WAPE_percent' else metric),val,rtol=1e-10,atol=1e-12)
    # The loaded selected model must produce its saved predictions on independently rebuilt features.
    import joblib
    bundle=joblib.load(OUT/'selected_forecast_model.joblib')
    assert bundle['selected_name']==selected_name and bundle['features']==F
    p=np.maximum(0,np.asarray(bundle['model'].predict(testref[F]),dtype=float))
    assert np.allclose(p,pred['full::'+selected_name],rtol=1e-10,atol=1e-12)
    meta=json.loads((OUT/'scenario_metadata.json').read_text());sc=pd.read_csv(OUT/'illustrative_discount_scenarios.csv')
    ctx=testref.loc[testref.store_id.eq(meta['store_id'])&testref.product_id.eq(meta['product_id'])&testref.dt.eq(pd.Timestamp(meta['target_date']))].iloc[0]
    h=daily.loc[daily.store_id.eq(meta['store_id'])&daily.product_id.eq(meta['product_id'])&daily.activity_flag.eq(meta['activity_flag'])&daily.dt.lt(pd.Timestamp(meta['target_date']))]
    support=h.discount.round(2).value_counts()
    assert set(sc.discount)==set(support.loc[support.ge(3)].index)
    assert np.allclose(sc.supported_training_days,sc.discount.map(support))
    sx=pd.DataFrame([ctx[F].to_dict()]*len(sc));sx['discount']=sc.discount
    sp=np.maximum(0,np.asarray(bundle['model'].predict(sx[F]),dtype=float))
    assert np.allclose(sp,sc.predicted_sales,rtol=1e-10,atol=1e-12)
    assert np.allclose(sc.sales_value_proxy,sc.discount*sc.predicted_sales)
    rp=sc.loc[sc.discount.eq(meta['reference_discount']),'sales_value_proxy'].iloc[0]
    assert sc.feasible.eq(sc.sales_value_proxy.ge((1-meta['max_proxy_loss'])*rp-1e-12)).all()
    w=sc.loc[sc.feasible].sort_values(['predicted_sales','discount'],ascending=[False,False]).iloc[0]
    assert sc.recommended.sum()==1 and sc.loc[sc.recommended,'discount'].iloc[0]==w.discount
    codecells=[c for c in nb['cells'] if c['cell_type']=='code']
    assert all(c.get('execution_count') is not None for c in codecells)
    assert not [o for c in codecells for o in c.get('outputs',[]) if o.get('output_type')=='error']
    report.update(test_row_identity_and_targets_match=True,test_metrics_independently_recomputed=metric_checks,
        validation_selection_recomputed=True,selected_model=selected_name,ablation_recomputed=True,
        diagnostic_metric_rows_recomputed=len(slices),saved_selected_model_predictions_match=True,
        scenario_support_and_constraints_pass=True,notebook_executed_code_cells=len(codecells),notebook_errors=0)
    report.update(status='PASS', verified_utc=datetime.now(timezone.utc).isoformat(), source_sha256=actual_hashes,
        notebook_sha256=hashlib.sha256((ROOT/'01_freshretail_project.ipynb').read_bytes()).hexdigest())
    (OUT/'independent_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
