"""Paths, column names, splits and plot colours shared by every step of the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"

# --- Columns ----------------------------------------------------------------------------------
KEY = ["store_id", "product_id"]
ROW_KEY = KEY + ["dt"]
TARGET = "target_sales"

# Known on the evening of day t; the target is sales on day t+1 (the row's date).
FEATURES = ["sales_t", "sales_lag7", "sales_mean7", "sales_mean_to_date", "stockout_hours_t",
            "discount_t", "discount_next", "activity_t", "holiday_next", "weekday_next"]
WEATHER = ["precpt_t", "temperature_t", "humidity_t"]  # tried and dropped: validation error was lower without

# --- Series selection -------------------------------------------------------------------------
DISCOUNT_DAY = 0.95  # a day counts as "discounted" when discount < 0.95 (more than 5% off)
MIN_SHARE, MAX_SHARE = 0.15, 0.85  # share of discounted days: drops never/always discounted series
MIN_SWITCHES = 6  # times the series flips between discounted and not discounted
MAX_ZERO_SALES = 0.20  # share of days with zero sales
TOP_N_STORES = 5

# --- Splits -----------------------------------------------------------------------------------
# Validation folds (one week each). For fold k, train on all rows dated before the fold starts.
VAL_FOLDS = {
    1: ("2024-05-22", "2024-05-28"),
    2: ("2024-05-29", "2024-06-04"),
    3: ("2024-06-05", "2024-06-11"),
    4: ("2024-06-12", "2024-06-18"),
    5: ("2024-06-19", "2024-06-25"),
}
FOLD_LABELS = {1: "May 22–28", 2: "May 29–Jun 4", 3: "Jun 5–11", 4: "Jun 12–18", 5: "Jun 19–25"}
TEST_START = "2024-06-26"  # eval.parquet: the unseen week

# Tomorrow's discount, grouped for the test-week breakdowns.
DISCOUNT_BAND_EDGES = [0, 0.8, 0.95, 1.0]
DISCOUNT_BAND_LABELS = ["deep (0.80 or lower)", "moderate (0.80-0.95]", "none (above 0.95)"]

# --- Plot style -------------------------------------------------------------------------------
COLOURS = {
    "baseline": "#898781",
    "knn": "#2a78d6",
    "rf": "#eb6834",
    "catboost": "#1baf7a",
    "ensemble": "#eda100",
    "extra": "#8b5cf6",
}
INK, MUTED, GRID, SURFACE, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb", "#c3c2b7"
