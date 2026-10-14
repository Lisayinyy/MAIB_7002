"""Series selection, cleaning and feature building.

One row of the feature table = one store-product-day, where the row's date is day t+1
(the day we predict). Columns ending in _t hold values from day t, the evening the
decision is made.
"""

import pandas as pd

from . import config as C



from pathlib import Path
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile

# --- Raw data: the two public parquet files, pinned to one revision ---------------------------
# The notebook calls ensure_raw_data() once at setup. It never downloads or extracts when valid
# files are already present, so a second run costs one hash check and nothing else.
RAW_REVISION = "08c1fab7f9257bc73679d415d65d644165d351d4"
RAW_URL = "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/{revision}/data/{name}"
RAW_SHA256 = {
    "train.parquet": "6706832db892bbae4969c19d87e07975d2543d2ba7d7d4756360654785de5a3d",
    "eval.parquet": "1b118840664280c6b88bffc84c80ee1f54c05d911e354b7599e5da10995e960e",
}
RAW_ZIP = "freshretail_raw.zip"  # the same two files, zipped (66 MB), shipped inside the hand-in folder


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(raw_dir):
    return all((raw_dir / name).is_file() and _sha256(raw_dir / name) == digest for name, digest in RAW_SHA256.items())


def _place(source, dest):
    """Move a verified file into place, replacing any stale file or dangling link."""
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    shutil.move(str(source), str(dest))


def ensure_raw_data(raw_dir=None, download=True):
    """Make sure data/raw holds the two raw parquet files, byte-identical to the pinned public revision.

    Tried in order, stopping at the first that succeeds:
      1. the files are already present and their SHA-256 matches: nothing is done;
      2. data/raw/freshretail_raw.zip is present: extract it once and verify;
      3. download the two files from the pinned Hugging Face revision and verify.
    Returns a one-line description of what happened. Raises if none of the three succeeds.
    """
    raw_dir = Path(raw_dir or C.RAW)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if _verified(raw_dir):
        return "Raw data already present and verified (SHA-256 matches the pinned revision); nothing downloaded or extracted."
    zip_path = raw_dir / RAW_ZIP
    if zip_path.is_file():
        with tempfile.TemporaryDirectory(dir=raw_dir) as tmp:
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(tmp)
            for name, digest in RAW_SHA256.items():
                extracted = Path(tmp) / name
                if extracted.is_file() and _sha256(extracted) == digest:
                    _place(extracted, raw_dir / name)
        if _verified(raw_dir):
            return f"Raw data extracted once from {RAW_ZIP} and verified against the pinned revision."
    if download:
        for name, digest in RAW_SHA256.items():
            dest = raw_dir / name
            if dest.is_file() and _sha256(dest) == digest:
                continue
            part = raw_dir / (name + ".part")
            urllib.request.urlretrieve(RAW_URL.format(revision=RAW_REVISION, name=name), part)
            if _sha256(part) != digest:
                part.unlink()
                raise RuntimeError(f"{name}: downloaded file failed SHA-256 verification")
            _place(part, dest)
        return "Raw data downloaded once (about 110 MB) from the pinned Hugging Face revision and verified."
    raise FileNotFoundError(f"Raw data not found in {raw_dir} and download is disabled")


def select_series(raw=None):
    """Score every store-product series for discount variation and pick the top stores.

    Returns (series, stores, selected): per-series statistics with a ``usable`` flag,
    per-store counts, and the usable series in the top ``TOP_N_STORES`` stores.
    """
    raw = raw or C.RAW / "train.parquet"
    cols = ["city_id", "store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt", "discount"]
    df = pd.read_parquet(raw, columns=cols).sort_values(["store_id", "product_id", "dt"])
    df["disc_day"] = df.discount < C.DISCOUNT_DAY

    g = df.groupby(C.KEY)
    series = g.agg(
        city_id=("city_id", "first"),
        disc_share=("disc_day", "mean"),
        disc_std=("discount", "std"),
        disc_min=("discount", "min"),
        mean_sales=("sale_amount", "mean"),
        zero_sales_share=("sale_amount", lambda x: (x == 0).mean()),
        stockout_day_share=("stock_hour6_22_cnt", lambda x: (x > 0).mean()),
    )
    series["switches"] = g.disc_day.apply(lambda x: (x != x.shift()).sum() - 1)
    series["usable"] = (
        series.disc_share.between(C.MIN_SHARE, C.MAX_SHARE)
        & (series.switches >= C.MIN_SWITCHES)
        & (series.zero_sales_share < C.MAX_ZERO_SALES)
    )

    stores = series.reset_index().groupby("store_id").agg(
        city_id=("city_id", "first"), n_series=("product_id", "size"), n_usable=("usable", "sum")
    ).sort_values("n_usable", ascending=False)
    top = stores.head(C.TOP_N_STORES)
    selected = series[series.usable & series.index.get_level_values("store_id").isin(top.index)]
    return series, stores, selected


def build_features(selected_keys, raw_dir=None):
    """Clean the selected series and build the feature table with split labels.

    ``selected_keys`` is any DataFrame with store_id and product_id columns (or index levels).
    Reads train.parquet and eval.parquet from ``raw_dir``.
    """
    raw_dir = raw_dir or C.RAW
    keys = selected_keys.reset_index()[C.KEY].drop_duplicates()
    cols = C.ROW_KEY + ["sale_amount", "stock_hour6_22_cnt", "discount", "holiday_flag", "activity_flag",
                        "precpt", "avg_temperature", "avg_humidity"]
    df = pd.concat([pd.read_parquet(raw_dir / f"{name}.parquet", columns=cols) for name in ["train", "eval"]])
    df = df.merge(keys, on=C.KEY)
    df["dt"] = pd.to_datetime(df.dt)
    df = df.sort_values(C.ROW_KEY).reset_index(drop=True)

    # --- Cleaning ---
    assert not df.duplicated(C.ROW_KEY).any(), "duplicate store-product-day rows"
    assert df.groupby(C.KEY).dt.agg(lambda x: x.diff().dt.days.dropna().eq(1).all()).all(), "gaps in dates"
    assert df.notna().all().all(), "missing values"

    df["discount"] = df.discount.clip(upper=1.0)  # a few rows sit slightly above 1.0
    bad_day = df.discount == 0  # discount of 0 = given away / data artefact: not real demand
    df.loc[bad_day, ["sale_amount", "discount"]] = float("nan")  # blanked so they never feed a lag or a target

    # --- Features ---
    g = df.groupby(C.KEY)
    feat = df[C.ROW_KEY].copy()
    feat["sales_t"] = g.sale_amount.shift(1)
    feat["sales_lag7"] = g.sale_amount.shift(7)  # same weekday as the day we predict
    feat["sales_mean7"] = g.sale_amount.transform(lambda x: x.shift(1).rolling(7, min_periods=5).mean())
    feat["sales_mean_to_date"] = g.sale_amount.transform(lambda x: x.shift(1).expanding(min_periods=5).mean())
    feat["stockout_hours_t"] = g.stock_hour6_22_cnt.shift(1)
    feat["discount_t"] = g.discount.shift(1)
    feat["discount_next"] = df.discount  # tomorrow's discount: the manager's decision
    feat["activity_t"] = g.activity_flag.shift(1)
    feat["holiday_next"] = df.holiday_flag
    feat["weekday_next"] = df.dt.dt.dayofweek  # 0 = Monday
    feat["precpt_t"] = g.precpt.shift(1)
    feat["temperature_t"] = g.avg_temperature.shift(1)
    feat["humidity_t"] = g.avg_humidity.shift(1)
    feat[C.TARGET] = df.sale_amount
    feat["target_stockout_hours"] = df.stock_hour6_22_cnt  # NOT a feature: only for the stockout sensitivity check
    # Stock history for the phase 2 guardrails (not a model feature): days with at least one stockout hour
    # among the 7 days up to and including day t (fewer at the very start of a series).
    feat["stockout_days7"] = g.stock_hour6_22_cnt.transform(lambda x: x.shift(1).gt(0).where(x.shift(1).notna()).rolling(7, min_periods=1).sum())
    feat = feat.dropna().reset_index(drop=True)

    # --- Splits ---
    feat["split"] = "train"
    feat.loc[feat.dt >= C.TEST_START, "split"] = "test"
    feat["val_fold"] = 0
    for k, (start, end) in C.VAL_FOLDS.items():
        feat.loc[feat.dt.between(start, end), "val_fold"] = k
    return feat


def load_features(path=None, rebuild=False):
    """Load the feature table, building it from the raw parquet files if it does not exist yet."""
    path = path or C.PROCESSED / "features.parquet"
    if rebuild or not path.exists() or "stockout_days7" not in pd.read_parquet(path).columns:
        _, _, selected = select_series()
        path.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(path.parent / "selected_series.csv")
        build_features(selected).to_parquet(path, index=False)
    return pd.read_parquet(path)


def split_summary(feat):
    """Rows and date range of each split and validation fold."""
    return feat.groupby(["split", "val_fold"]).agg(rows=("dt", "size"), first_day=("dt", "min"), last_day=("dt", "max"))
