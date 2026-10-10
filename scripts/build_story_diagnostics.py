#!/usr/bin/env python3
"""Explain policy sensitivity using saved scenarios; never tune or refit a model.

Each row changes one rule while holding every forecast and test-day context fixed.
The original rule remains the delivered default. There are no observed policy
outcomes with which to declare a better threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import pandas as pd
from finalproject_pricingml import config as C, v2

OUT = ROOT / "results/story_diagnostics"
SCENARIOS = [
    ("Saved policy", {}),
    ("Sales gain >=10%", {"min_sales_gain":.10}),
    ("Sales gain >=20%", {"min_sales_gain":.20}),
    ("Value floor >=100%", {"min_value_share":1.}),
    ("Value floor >=105%", {"min_value_share":1.05}),
    ("Support >=5 days", {"min_support_days":5}),
    ("Support >=10 days", {"min_support_days":10}),
    ("Milder within 5%", {"near_optimal_sales_tolerance":.05}),
    ("Review at >=4h stockout", {"stockout_hours_review_threshold":4}),
    ("Review at >=8h stockout", {"stockout_hours_review_threshold":8}),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs():
    return {str(p.relative_to(ROOT)):digest(p) for p in [
        ROOT/"results/v2/features.parquet",ROOT/"results/v2/discount_response.csv",
        ROOT/"results/v2/recommendations.csv",ROOT/"results/v2/experiment_plan.json",
        ROOT/"src/finalproject_pricingml/v2.py"]}


def verify():
    receipt=json.loads((OUT/"receipt.json").read_text())
    assert receipt["inputs_sha256"]==inputs(),"Scenario sources changed; regenerate diagnostics"
    assert receipt["script_sha256"]==digest(__file__)
    for name,hash_value in receipt["outputs_sha256"].items():
        assert digest(OUT/name)==hash_value,name
    table=pd.read_csv(OUT/"policy_sensitivity.csv")
    for col in ["discount_share","full_price_share","review_share","discount_share_nonstockout"]:
        assert table[col].between(0,1).all()
    assert np.allclose(table.discount_share+table.full_price_share+table.review_share,1)
    assert table.iloc[0].discount_count==1179 and table.iloc[0].review_count==1005
    print(json.dumps({"status":"passed","scenarios":len(table),"model_fits":0}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args()
    if args.verify_only:
        verify();return
    OUT.mkdir(parents=True,exist_ok=True)
    hashes=inputs()
    base=json.loads((ROOT/"results/v2/experiment_plan.json").read_text())["decision_parameters"]
    plan={"created_utc":datetime.now(timezone.utc).isoformat(),"base_parameters":base,
          "scenarios":[{"name":name,"changed_parameters":changes} for name,changes in SCENARIOS],
          "purpose":"Describe rule sensitivity without selecting a policy from hypothetical gains.",
          "default_policy_changed":False,"model_fits":0,"observed_policy_outcomes":False,
          "inputs_sha256":hashes,"script_sha256":digest(__file__)}
    (OUT/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    feat=pd.read_parquet(ROOT/"results/v2/features.parquet")
    frame=feat[feat.split.eq("test")].sort_values(C.ROW_KEY).set_index(C.ROW_KEY)
    curves=pd.read_csv(ROOT/"results/v2/discount_response.csv",parse_dates=["dt"])
    groups=list(curves.groupby(C.ROW_KEY,sort=True))
    saved=pd.read_csv(ROOT/"results/v2/recommendations.csv",parse_dates=["dt"]).set_index(C.ROW_KEY).sort_index()
    rows=[];all_decisions=[]
    for label,changes in SCENARIOS:
        settings=base|changes
        kwargs={k:settings[k] for k in ["min_sales_gain","min_value_share",
                "near_optimal_sales_tolerance","stockout_hours_review_threshold"]}
        decisions=[]
        for key,curve in groups:
            curve=curve.copy()
            curve["supported"]=curve.support_days.ge(settings["min_support_days"])
            hours=float(frame.loc[key,"stockout_hours_t"])
            result=v2.choose_action(curve,hours,**kwargs)
            decisions.append(dict(zip(C.ROW_KEY,key))|{"scenario":label,"previous_stockout_hours":hours}|result)
        decisions=pd.DataFrame(decisions).set_index(C.ROW_KEY).sort_index()
        assert decisions.index.equals(saved.index)
        if not changes:
            assert decisions.status.equals(saved.status)
            assert np.allclose(decisions.recommended_rate,saved.recommended_rate,equal_nan=True)
        discount=decisions.status.eq("recommend_discount")
        full=decisions.status.eq("keep_full_price")
        no_stockout=decisions.previous_stockout_hours.lt(1)
        rows.append({"scenario":label,"changed_parameters":json.dumps(changes),"rows":len(decisions),
                     "discount_count":int(discount.sum()),"full_price_count":int(full.sum()),
                     "review_count":int((~(discount|full)).sum()),
                     "discount_share":float(discount.mean()),"full_price_share":float(full.mean()),
                     "review_share":float((~(discount|full)).mean()),
                     "discount_share_nonstockout":float(discount[no_stockout].mean()),
                     "mean_recommended_rate":float(decisions.loc[discount,"recommended_rate"].mean())})
        all_decisions.append(decisions.reset_index())
        print(f"{label}: {discount.sum()}/{len(decisions)} discounts",flush=True)
    pd.DataFrame(rows).to_csv(OUT/"policy_sensitivity.csv",index=False)
    pd.concat(all_decisions,ignore_index=True).to_csv(OUT/"policy_decisions.csv",index=False)
    assert inputs()==hashes,"Canonical files changed during diagnostic computation"
    receipt={"completed_utc":datetime.now(timezone.utc).isoformat(),"status":"passed",
             "inputs_sha256":hashes,"script_sha256":digest(__file__),"model_fits":0,
             "original_policy_replayed_exactly":True,"canonical_results_unchanged":True,
             "outputs_sha256":{p.name:digest(p) for p in [OUT/"plan.json",OUT/"policy_sensitivity.csv",OUT/"policy_decisions.csv"]}}
    (OUT/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    verify()


if __name__=="__main__":
    main()
