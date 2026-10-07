"""Post-hoc ensemble exploration; preserve the frozen submission experiment."""
from pathlib import Path
from datetime import datetime, timezone
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '4'
import argparse
import json
import hashlib


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='Project root containing outputs/final_protocol.')
    parser.add_argument('--output-dir', type=Path,
                        help='Output directory (relative paths are relative to --root).')
    parser.add_argument('--quiet', action='store_true',
                        help='Hide fold progress and the detailed terminal table.')
    args = parser.parse_args(argv)

    import numpy as np
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from catboost import CatBoostRegressor
    from threadpoolctl import threadpool_limits

    ROOT = args.root.resolve()
    SOURCE = ROOT / 'outputs/final_protocol'
    OUT = args.output_dir or Path('outputs/ensemble_exploration')
    OUT = (OUT if OUT.is_absolute() else ROOT / OUT).resolve()
    if OUT == SOURCE or SOURCE in OUT.parents or OUT in SOURCE.parents:
        raise ValueError('Ensemble outputs must be separate from the main experiment outputs.')
    for filename in ['locked_protocol.json', 'chosen_parameters.csv',
                     'feature_frame.parquet', 'test_predictions.csv']:
        if not (SOURCE / filename).is_file():
            raise FileNotFoundError(f'Run the main notebook experiment first: missing {SOURCE / filename}')
    OUT.mkdir(parents=True, exist_ok=True)

    def emit(*values, **kwargs):
        if not args.quiet:
            print(*values, **kwargs)

    protocol = json.loads((SOURCE/'locked_protocol.json').read_text())
    features = protocol['features']['full']
    names = ['Ridge Regression','Decision Tree Regressor','Random Forest Regressor',
             'HistGradientBoostingRegressor','CatBoost Regressor']
    short = ['Ridge','Decision Tree','Random Forest','HistGBR','CatBoost']
    choice = pd.read_csv(SOURCE/'chosen_parameters.csv')
    choice = choice.loc[choice.feature_set.eq('full')].set_index('Model')
    params = {name:json.loads(choice.loc[name,'parameters']) for name in names}

    def build(name):
        p = params[name]
        if name == names[0]: return make_pipeline(StandardScaler(),Ridge(**p))
        if name == names[1]: return DecisionTreeRegressor(**p,random_state=7002)
        if name == names[2]: return RandomForestRegressor(**p,max_features=.8,random_state=7002,n_jobs=4)
        if name == names[3]: return HistGradientBoostingRegressor(**p,l2_regularization=1.,early_stopping=False,random_state=7002)
        return CatBoostRegressor(**p,loss_function='RMSE',random_seed=7002,has_time=True,
            thread_count=4,verbose=False,allow_writing_files=False)

    def score(y,p):
        e=np.asarray(p)-np.asarray(y)
        return {'MAE':float(np.abs(e).mean()),'MSE':float((e**2).mean()),
                'RMSE':float(np.sqrt((e**2).mean())),
                'WAPE_pct':float(100*np.abs(e).sum()/np.abs(y).sum())}

    # Freeze this limited candidate set before creating any new ensemble scores.
    methods = {}
    def add(name,weights):
        w=np.asarray(weights,dtype=float)
        assert np.all(w>=0) and np.isclose(w.sum(),1)
        if not any(np.allclose(w,old) for old in methods.values()): methods[name]=w

    for i,name in enumerate(short): add(name,np.eye(5)[i])
    add('RF + CatBoost (50/50)',[0,0,.5,0,.5])
    add('RF + HistGBR + CatBoost (equal)',[0,0,1/3,1/3,1/3])
    add('All five models (equal)',np.ones(5)/5)
    for rf in range(5):
        for hist in range(5-rf):
            cat=4-rf-hist
            label=' + '.join(f'{n} {v*25}%' for n,v in [('RF',rf),('HistGBR',hist),('CatBoost',cat)] if v)
            add(label,[0,0,rf/4,hist/4,cat/4])
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'features':features,'base_parameters':params,
        'methods':{name:w.tolist() for name,w in methods.items()},'weight_column_order':short,
        'selection':'lowest mean MAE over the existing five temporal validation folds; fixed insertion order resolves exact ties',
        'status':'post-hoc exploratory extension; frozen main experiment and original model selection unchanged',
        'caveats':['Base-model settings were already tuned on these five folds; ensemble CV is additional tuning evidence, not an independent nested evaluation.',
                   'The team already inspected the final week before proposing ensembles; final-week results are descriptive and not a fresh holdout.',
                   'No candidate or weight is chosen using final-week scores.']}
    (OUT/'exploration_plan.json').write_text(json.dumps(plan,indent=2)+'\n')

    f=pd.read_parquet(SOURCE/'feature_frame.parquet').sort_values(['dt','store_id','product_id'])
    train=f.loc[f.source_split.eq('train')]
    oof=[]
    fit_count=0
    for fold in protocol['folds']:
        start,end=pd.Timestamp(fold['valid_start']),pd.Timestamp(fold['valid_end'])
        tr,va=train.loc[train.dt.lt(start)],train.loc[train.dt.between(start,end)]
        assert len(tr)==fold['n_train'] and len(va)==2184 and tr.dt.max()<va.dt.min()
        part=va[['store_id','product_id','dt','sale_amount','discount','sales_mean7','sales_lag7']].copy()
        part['fold']=fold['fold']
        for name,alias in zip(names,short):
            model=build(name)
            with threadpool_limits(limits=4):
                model.fit(tr[features],tr.sale_amount)
                fit_count += 1
                part[alias]=np.maximum(0,model.predict(va[features]))
        oof.append(part)
        emit(f'Completed fold {fold["fold"]}/5',flush=True)
    oof=pd.concat(oof,ignore_index=True)
    assert len(oof)==10920 and not oof.duplicated(['store_id','product_id','dt']).any()
    oof.to_csv(OUT/'validation_predictions.csv',index=False)
    errors=oof[short].subtract(oof.sale_amount,axis=0)
    errors.corr().to_csv(OUT/'validation_error_correlations.csv')
    foldrows=[]
    for method,w in methods.items():
        for fold,rows in oof.groupby('fold'):
            foldrows.append({'Method':method,'fold':int(fold),**score(rows.sale_amount,rows[short].to_numpy()@w)})
    cvfold=pd.DataFrame(foldrows)
    cvmean=cvfold.groupby('Method',sort=False)[['MAE','MSE','RMSE','WAPE_pct']].mean().reindex(methods)
    cvmean['MAE_std']=cvfold.groupby('Method').MAE.std().reindex(cvmean.index)
    cvmean['weights']=cvmean.index.map(lambda x:json.dumps(methods[x].tolist()))
    winner=cvmean.MAE.idxmin()
    weights=methods[winner]
    cvfold.to_csv(OUT/'validation_fold_scores.csv',index=False)
    cvmean.to_csv(OUT/'validation_scores.csv')
    selected={'method':winner,'weights':dict(zip(short,weights.tolist())),
              'CV_mean_MAE':float(cvmean.loc[winner,'MAE']),'selection_uses_final_week':False}
    (OUT/'selection_before_final_week.json').write_text(json.dumps(selected,indent=2)+'\n')
    emit('Selected by validation:',selected,flush=True)

    # Only now read the existing final-week predictions and apply frozen weights.
    test=pd.read_csv(SOURCE/'test_predictions.csv')
    assert len(test)==2184 and not test.duplicated(['store_id','product_id','dt']).any()
    matrix=test[[f'full::{name}' for name in names]].to_numpy()
    result=[]
    for method,w in methods.items():
        result.append({'Method':method,'CV_MAE':float(cvmean.loc[method,'MAE']),
            'selected_by_validation':method==winner,**score(test.sale_amount,matrix@w)})
    scores=pd.DataFrame(result)
    scores.to_csv(OUT/'final_week_scores.csv',index=False)
    out=test[['store_id','product_id','dt','sale_amount','discount']].copy()
    out['selected_ensemble_prediction']=matrix@weights
    out.to_csv(OUT/'selected_final_week_predictions.csv',index=False)
    bands=pd.cut(out.discount,[-np.inf,.8,.95,np.inf],labels=['deep','moderate','none_or_light'],right=True)
    bandrows=[]
    for band,rows in out.groupby(bands,observed=True):
        idx=rows.index
        for method in ['Random Forest','CatBoost',winner]:
            bandrows.append({'band':str(band),'Method':method,'n':len(rows),
                **score(rows.sale_amount,matrix[idx]@methods[method])})
    pd.DataFrame(bandrows).to_csv(OUT/'discount_band_scores.csv',index=False)

    # Compare each reconstructed single-model CV mean with the locked experiment.
    for name,alias in zip(names,short):
        assert np.isclose(cvmean.loc[alias,'MAE'],choice.loc[name,'MAE'],rtol=1e-9,atol=1e-12)
    assert np.allclose(out.selected_ensemble_prediction,matrix@weights)
    # Refresh verification from this run; do not preserve old release/hash claims.
    assert fit_count == len(protocol['folds']) * len(names) == 25
    saved_oof = pd.read_csv(OUT/'validation_predictions.csv', float_precision='round_trip')
    saved_selected = pd.read_csv(OUT/'selected_final_week_predictions.csv', float_precision='round_trip')
    saved_cv = pd.read_csv(OUT/'validation_scores.csv').set_index('Method')
    saved_selection = json.loads((OUT/'selection_before_final_week.json').read_text())
    assert len(saved_oof) == len(oof) and len(saved_selected) == len(test)
    assert np.allclose(saved_oof[short], oof[short], rtol=1e-12, atol=1e-12)
    selected_max_difference = float(np.max(np.abs(saved_selected.selected_ensemble_prediction - matrix@weights)))
    assert selected_max_difference < 1e-12
    assert saved_cv.MAE.idxmin() == winner == saved_selection['method']
    assert np.allclose(list(saved_selection['weights'].values()), weights)
    notebook_path = ROOT/'01_freshretail_project.ipynb'
    notebook_cells = json.loads(notebook_path.read_text()).get('cells', []) if notebook_path.exists() else []
    supplement = [cell for cell in notebook_cells
                  if cell.get('metadata', {}).get('freshretail_section') == 'supplementary_ensembles_v1']
    cell_source = lambda cell: ''.join(cell.get('source', []))
    verification = {
        'verified_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'current ensemble run; no comparison with an earlier notebook release',
        'base_model_fit_count': fit_count,
        'validation_folds': len(protocol['folds']),
        'validation_rows': len(saved_oof), 'final_week_rows': len(saved_selected),
        'candidate_count': len(methods),
        'single_model_CV_reproduction_pass': True,
        'validation_prediction_readback_pass': True,
        'selection_matches_validation_minimum': True,
        'selected_method': winner,
        'selected_prediction_max_absolute_difference': selected_max_difference,
        'weights_sum_to_one': bool(np.isclose(weights.sum(), 1)),
        'notebook_supplement_source_check': {
            'section_present': len(supplement) == 3,
            'runner_invocation_present': any(cell.get('cell_type') == 'code' and
                'scripts/explore_ensembles.py' in cell_source(cell) for cell in supplement),
            'current_notebook_cell_count': len(notebook_cells),
            'source_sha256': hashlib.sha256('\n'.join(cell_source(cell) for cell in notebook_cells).encode()).hexdigest(),
            'scope': 'source integration only; notebook execution is verified by the notebook runner'}
    }
    (OUT/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    baseline_mae=float(np.abs(test.sale_amount-test['7-day average baseline']).mean())
    summary={'selected':selected,'selected_final_week':scores.loc[scores.Method.eq(winner)].iloc[0].to_dict(),
        'final_week_baseline_MAE':baseline_mae,'single_model_CV_reproduction_pass':True,
        'prediction_weights_sum_to_one':bool(np.isclose(weights.sum(),1)),
        'n_oof':len(oof),'n_test':len(test),'caveats':plan['caveats']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    visible=['Random Forest','HistGBR','CatBoost','RF + CatBoost (50/50)',
             'RF + HistGBR + CatBoost (equal)','All five models (equal)']
    if winner not in visible:visible.append(winner)
    report=scores.set_index('Method').loc[visible].reset_index()
    body='# Exploratory model combinations\n\n'+report.to_markdown(index=False,floatfmt='.6f')+'\n\n'
    body+='Selected by mean validation MAE: **'+winner+'**.\n\n'
    body+='The original main experiment and its validation-selected Random Forest are unchanged. '
    body+='This is a post-hoc extension after the final week had already been viewed. Base-model parameters '
    body+='and ensemble weights reuse the same CV folds, so CV gains include selection optimism. '
    body+='Weights were frozen before applying them to final-week predictions. No forecast gain establishes a causal pricing gain.\n'
    (OUT/'RESULTS.md').write_text(body)
    emit(report.to_string(index=False),flush=True)
    print('Exploration complete:',OUT,flush=True)


if __name__ == '__main__':
    main()
