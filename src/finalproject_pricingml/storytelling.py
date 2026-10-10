"""Small, inspectable figures and worked calculations for the project narrative.

These helpers explain the frozen experiment. They never select a new model,
change its parameters, or write into the canonical result directory. Historical
tuning figures are attributed to the teammate's saved experiments.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch

from . import config as C, models

GREEN, TEAL, AMBER, GRAY, INK = "#167262", "#66a698", "#cf8538", "#9aa5a0", "#233e36"


def _clean(ax, *, axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color="#e5ece7", linewidth=.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)


def training_eda(feat):
    """Price association and support, using feature dates before validation starts."""
    initial = feat[feat.dt < min(v[0] for v in C.VAL_FOLDS.values())].copy()
    means = initial.groupby(C.KEY)[C.TARGET].transform("mean")
    initial["relative_sales"] = initial[C.TARGET] / means.replace(0, np.nan)
    labels = ["Deep: rate <= 0.80", "Moderate: 0.80 < rate <= 0.95", "Near/full: rate > 0.95"]
    initial["band"] = pd.cut(initial.discount_next, [0, .8, .95, 1.], labels=labels,
                             include_lowest=True)
    table = initial.groupby("band", observed=True).agg(
        rows=(C.TARGET, "size"), mean_relative_sales=("relative_sales", "mean"),
        stockout_share=("target_stockout_hours", lambda x: x.gt(0).mean()),
    ).reindex(labels).reset_index()
    table["row_share"] = table.rows / len(initial)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    short = ["Deep", "Moderate", "Near/full price"]
    colors = [AMBER, TEAL, GREEN]
    axes[0].bar(short, table.mean_relative_sales, color=colors, width=.62)
    axes[0].axhline(1, color=GRAY, linestyle="--", linewidth=1)
    axes[0].set_ylabel("Sales / this series' initial-period mean")
    axes[0].set_title("Do discounted days look different?", loc="left", fontsize=12)
    axes[1].bar(short, table.rows, color=colors, width=.62)
    axes[1].set_ylabel("Recorded store-product-days")
    axes[1].set_title("How much evidence supports each band?", loc="left", fontsize=12)
    for i, row in table.iterrows():
        axes[1].text(i, row.rows, f"{int(row.rows):,}", ha="center", va="bottom", fontsize=10)
    for ax in axes:
        _clean(ax); ax.margins(y=.2)
    fig.suptitle(f"Initial training history only: {initial.dt.min():%d %b} - {initial.dt.max():%d %b %Y}",
                 fontsize=11, color=INK)
    fig.tight_layout()
    return fig, table


def split_timeline(feat, plan):
    """Expanding training windows, with a separately labelled reused benchmark."""
    first = feat.dt.min()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    rows = []
    for i, (fold, dates) in enumerate(plan["folds"].items()):
        start, end = map(pd.Timestamp, dates)
        train = feat[feat.dt < start]
        check = feat[feat.dt.between(start, end)]
        ax.barh(i, (start-first).days, left=mdates.date2num(first), color=TEAL, height=.6)
        ax.barh(i, (end-start).days+1, left=mdates.date2num(start), color=GREEN, height=.6)
        rows.append({"Round": f"Validation {fold}", "Training ends": str((start-pd.Timedelta(days=1)).date()),
                     "Training rows": len(train), "Scored rows": len(check)})
    final_start, final_end = pd.Timestamp(plan["test_start"]), feat.dt.max()
    ax.barh(5, (final_start-first).days, left=mdates.date2num(first), color=TEAL, height=.6)
    ax.barh(5, (final_end-final_start).days+1, left=mdates.date2num(final_start), color=AMBER, height=.6)
    ax.set_yticks(range(6), [f"Validation {i}" for i in range(1, 6)]+["Reused benchmark"])
    ax.invert_yaxis(); ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_title("Train on earlier dates; predict one day ahead on later dates", loc="left", fontsize=13)
    for color, label in [(TEAL,"Fit on earlier dates"),(GREEN,"Validation week"),(AMBER,"Previously viewed final week")]:
        ax.plot([], [], linewidth=7, color=color, label=label)
    ax.legend(loc="upper left", bbox_to_anchor=(0,-.13), ncol=3, frameon=False, fontsize=9)
    _clean(ax, axis="x"); fig.tight_layout()
    return fig, pd.DataFrame(rows)


def feature_walkthrough(feat, example):
    """Recompute actual lag/rolling inputs from one series' earlier recorded targets."""
    row = example.iloc[0]
    history = feat[(feat.store_id == row.store_id) & (feat.product_id == row.product_id)
                   & (feat.dt < row["dt"])].sort_values("dt")
    assert len(history) >= 14
    actual = {
        "sales_t": history[C.TARGET].iloc[-1],
        "sales_lag7": history[C.TARGET].iloc[-7],
        "sales_mean7": history[C.TARGET].tail(7).mean(),
        "sales_mean3": history[C.TARGET].tail(3).mean(),
        "sales_mean14": history[C.TARGET].tail(14).mean(),
        "stockout_hours_t": history.target_stockout_hours.iloc[-1],
    }
    checks = pd.DataFrame([{"Feature": k, "Recomputed from earlier days": v,
                            "Saved model input": row[k], "Matches": bool(np.isclose(v, row[k]))}
                           for k,v in actual.items()])
    assert checks.Matches.all()
    shown = history.tail(7)[["dt", C.TARGET, "discount_next", "target_stockout_hours"]].rename(columns={
        "dt":"Historical date", C.TARGET:"Recorded sales", "discount_next":"Recorded price rate",
        "target_stockout_hours":"Recorded stockout hours"})
    return shown.reset_index(drop=True), checks


def teammate_knn(root):
    root = Path(root) / "results"
    with_weather = pd.read_csv(root/"knn_stage1_with_weather_summary.csv")
    no_weather = pd.read_csv(root/"knn_stage1_summary.csv")
    w = with_weather[(with_weather.weighting=="equal") & (with_weather.k>0)]
    n = no_weather[(no_weather.weighting=="equal") & (no_weather.k>0)]
    fig, ax = plt.subplots(figsize=(9,3.6))
    ax.plot(w.k, w.mae, "o-", color=GRAY, label="13 inputs: including weather")
    ax.plot(n.k, n.mae, "o-", color=GREEN, label="10 inputs: without weather")
    baseline = float(no_weather.loc[no_weather.k.eq(0), "mae"].iloc[0])
    ax.axhline(baseline, color=AMBER, linestyle="--", label="7-day baseline")
    locked = n[n.k.eq(25)].iloc[0]
    ax.scatter([25], [locked.mae], s=110, facecolors="none", edgecolors=INK, linewidths=2, zorder=5)
    ax.set(xlabel="Number of neighbours (k)", ylabel="Mean validation MAE",
           title="kNN: weather removal matters more than fine-tuning k")
    ax.legend(frameon=False, fontsize=9); _clean(ax); fig.tight_layout()
    table = pd.DataFrame([
        {"Recorded design":"k=25, equal weights, with weather", "Validation MAE":w.set_index("k").loc[25,"mae"]},
        {"Recorded design":"k=25, equal weights, without weather", "Validation MAE":locked.mae},
        {"Recorded design":"7-day baseline", "Validation MAE":baseline},
    ])
    return fig, table


def teammate_rf(root):
    root = Path(root)/"results"
    grid = pd.read_csv(root/"rf_tuning_summary.csv", dtype={"max_depth":str})
    order = [x for x in ["10","15","20","No limit"] if x in grid.max_depth.unique()]
    pivot = grid.pivot(index="max_depth",columns="min_leaf",values="mae").reindex(order)
    fig, ax = plt.subplots(figsize=(7.8,3.5))
    im = ax.imshow(pivot, cmap="YlGnBu_r", aspect="auto")
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            ax.text(j,i,f"{pivot.iloc[i,j]:.4f}",ha="center",va="center",fontsize=11,
                    color="white" if pivot.iloc[i,j] < pivot.to_numpy().mean() else INK)
    ax.set_xticks(range(len(pivot.columns)),pivot.columns)
    ax.set_yticks(range(len(pivot.index)),pivot.index)
    ax.set(xlabel="Minimum training rows per leaf",ylabel="Maximum tree depth",
           title="Random Forest: a modest gap across 16 settings")
    fig.colorbar(im,ax=ax,label="Five-week validation MAE",fraction=.04,pad=.04)
    fig.tight_layout()
    return fig, grid.sort_values("mae").head(4)[["max_depth","min_leaf","mae","folds_beating_baseline"]]


def teammate_catboost(root):
    root = Path(root)/"results"
    curves = pd.read_csv(root/"catboost_learning_curves.csv", header=[0,1], index_col=0)
    grid = pd.read_csv(root/"catboost_round1b_summary.csv")
    fig, axes = plt.subplots(1,2,figsize=(10,3.8))
    for rate, color in zip(curves.columns.get_level_values(0).unique(),[GRAY,GREEN,AMBER]):
        axes[0].plot(curves.index+1, curves[(rate,"validation")],color=color,label=rate)
    slow = curves.columns.get_level_values(0).unique()[1]
    axes[1].plot(curves.index+1,curves[(slow,"train")],label="Training MAE",color=TEAL)
    axes[1].plot(curves.index+1,curves[(slow,"validation")],label="Validation MAE",color=GREEN)
    axes[0].set(title="Learning-rate trade-off",xlabel="Boosting iteration",ylabel="Validation MAE")
    axes[1].set(title=f"Training improves; later dates can worsen ({slow})",xlabel="Boosting iteration",ylabel="MAE")
    for ax in axes:
        ax.legend(frameon=False,fontsize=9); _clean(ax)
    fig.suptitle("Teammate's cached learning-curve experiment (diagnostic, not a new fit)",fontsize=11)
    fig.tight_layout()
    return fig, grid.sort_values("mae").head(5)[["depth","learning_rate","iterations","mae"]]


def blend_diagnostics(root):
    root = Path(root)
    old = pd.read_csv(root/"results/ensemble_weight_sweep.csv")
    old_oof = pd.read_csv(root/"results/oof_predictions.csv")
    oof = pd.read_csv(root/"results/v2/validation_predictions.csv")
    rows=[]
    for w in [0.,.25,.5,.75,1.]:
        prediction = w*oof["Teammate RF"]+(1-w)*oof["CatBoost history + IDs MAE"]
        error = (prediction-oof[C.TARGET]).abs().groupby(oof.val_fold).mean().mean()
        rows.append({"RF weight":w,"CatBoost weight":1-w,"Validation MAE":error})
    table=pd.DataFrame(rows)
    old_corr=(old_oof.rf-old_oof.actual).corr(old_oof.catboost-old_oof.actual)
    new_corr=(oof["Teammate RF"]-oof[C.TARGET]).corr(oof["CatBoost history + IDs MAE"]-oof[C.TARGET])
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    axes[0].plot(old.rf_weight,old.mae,"o-",color=GRAY)
    pick=old[old.rf_weight.eq(.5)].iloc[0]
    axes[0].scatter([.5],[pick.mae],color=GREEN,s=70,zorder=5)
    axes[0].set_title(f"Teammate blend: similar errors (r={old_corr:.2f})",fontsize=11)
    axes[1].plot(table["RF weight"],table["Validation MAE"],"o-",color=GREEN)
    axes[1].scatter([.25],[table.loc[table["RF weight"].eq(.25),"Validation MAE"].iloc[0]],
                    s=140,facecolors="none",edgecolors=AMBER,linewidths=2,zorder=5)
    axes[1].set_title(f"Extended blend: declared grid picks 25% RF (r={new_corr:.2f})",fontsize=11)
    for ax in axes:
        ax.set(xlabel="RF weight (remaining weight goes to CatBoost)",ylabel="Five-week validation MAE")
        ax.set_xticks([0,.25,.5,.75,1]);_clean(ax)
    fig.tight_layout()
    table.attrs.update(teammate_error_correlation=float(old_corr),extended_error_correlation=float(new_corr))
    return fig,table


def error_diagnostics(feat, predictions, winner):
    error=(predictions[winner]-predictions[C.TARGET]).abs()
    frame=predictions.copy();frame["absolute_error"]=error
    frame["band"]=pd.cut(frame.discount_next,[0,.8,.95,1.],labels=["Deep","Moderate","Near/full"],include_lowest=True)
    table=frame.groupby("band",observed=True).agg(rows=(C.TARGET,"size"),mae=("absolute_error","mean"),
                actual_mean=(C.TARGET,"mean"),forecast_mean=(winner,"mean")).reset_index()
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    axes[0].scatter(frame[C.TARGET],frame[winner],s=9,alpha=.25,color=GREEN,rasterized=True)
    lim=max(frame[C.TARGET].max(),frame[winner].max())
    axes[0].plot([0,lim],[0,lim],"--",color=AMBER,linewidth=1)
    axes[0].set(xlabel="Recorded normalized sales",ylabel="Predicted normalized sales",title="Where forecasts miss (all benchmark cases)")
    axes[1].bar(table.band.astype(str),table.mae,color=[AMBER,TEAL,GREEN])
    axes[1].set(ylabel="MAE",title="Does a good average hide difficult days?")
    for i,row in table.iterrows():
        axes[1].text(i,row.mae,f"n={int(row.rows):,}",ha="center",va="bottom",fontsize=9)
    axes[1].margins(y=.18)
    for ax in axes:_clean(ax)
    fig.tight_layout()
    return fig,table


def policy_sensitivity(root):
    table=pd.read_csv(Path(root)/"results/story_diagnostics/policy_sensitivity.csv")
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),gridspec_kw={"width_ratios":[1.45,1]})
    positions=np.arange(len(table))
    left=np.zeros(len(table))
    for col,label,color in [("discount_share","Recommend discount",GREEN),
                            ("full_price_share","Keep full price",TEAL),
                            ("review_share","Review / abstain",GRAY)]:
        vals=100*table[col].to_numpy()
        axes[0].barh(positions,vals,left=left,color=color,label=label,height=.66);left+=vals
    axes[0].set_yticks(positions,table["scenario"]);axes[0].invert_yaxis()
    axes[0].set(xlabel="Share of all 2,184 scenarios (%)",xlim=(0,100),title="Rules change the decisions")
    axes[0].legend(frameon=False,fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.12),ncol=1)
    axes[1].barh(positions,100*table["discount_share_nonstockout"],color=AMBER,height=.66)
    axes[1].set_yticks(positions,[""]*len(table));axes[1].invert_yaxis()
    axes[1].set(xlabel="Discount share (%)",xlim=(0,100),title="Same no-stockout subgroup")
    for ax in axes:_clean(ax,axis="x")
    fig.suptitle("Sensitivity of model-based scenarios, not observed business outcomes",fontsize=11)
    fig.tight_layout()
    return fig,table


def decision_flow():
    labels=[("01  CHECK SUPPLY","Recent stockout?\nAsk for supply review."),
            ("02  CHECK HISTORY","Is this candidate price\nsupported by past days?"),
            ("03  COMPARE MODELS","Do sales-gain and value\nchecks pass in all models?"),
            ("04  CHOOSE / DEFER","Pick a mild near-best discount\nor explain why review is needed.")]
    fig,ax=plt.subplots(figsize=(10,2.4));ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")
    for i,(title,body) in enumerate(labels):
        x=.01+i*.25
        ax.add_patch(FancyBboxPatch((x,.22),.225,.65,boxstyle="round,pad=.006,rounding_size=.02",
                                   facecolor="#edf5f0",edgecolor="#c9ded4"))
        ax.text(x+.1125,.69,title,ha="center",va="center",fontsize=10,color=GREEN,fontweight="bold")
        ax.text(x+.1125,.43,body,ha="center",va="center",fontsize=9,color=INK,linespacing=1.6)
        if i<3:ax.annotate("",xy=(x+.25,.55),xytext=(x+.227,.55),arrowprops={"arrowstyle":"->","color":GRAY})
    fig.tight_layout()
    return fig,pd.DataFrame(labels,columns=["Step","Reason"])


def live_knn_example(feat,example):
    """Recompute a real kNN forecast, show its nearest rows and average all 25 targets."""
    training=feat[feat.dt<C.TEST_START].copy().reset_index(drop=True)
    model=models.make_knn(25,n_jobs=4).fit(training[C.FEATURES],training[C.TARGET])
    transformed=model[:-1].transform(example[C.FEATURES])
    distances,indices=model[-1].kneighbors(transformed)
    neighbors=training.iloc[indices[0]][C.ROW_KEY+[C.TARGET]].copy()
    neighbors["Standardized Euclidean distance"]=distances[0]
    forecast=float(neighbors[C.TARGET].mean())
    direct=float(model.predict(example[C.FEATURES])[0])
    assert np.isclose(forecast,direct,atol=1e-12)
    calculation=pd.DataFrame({"Calculation":["Neighbours used","Sum of their observed targets","Sum / 25","Pipeline prediction"],
                              "Value":[25,neighbors[C.TARGET].sum(),forecast,direct]})
    return neighbors.head(5).reset_index(drop=True),calculation
