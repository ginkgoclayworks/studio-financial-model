#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 13:46:50 2025

@author: harshadghodke
"""


# ### Calculate the probability of an economic recession from live data


"""
GCWS Economic Headwind Nowcast — DOWNTURN_PROB_PER_MONTH

Sources
- Google Trends (via pytrends) for MA: pottery/ceramics demand proxy (optional, runs without API).
- BLS API (optional): Massachusetts unemployment rate (series LASST250000000000003).
- FRED API (optional): University of Michigan Consumer Sentiment (UMCSENT).

Behavior
- If BLS/FRED API keys are present in env, pull live data. Otherwise fall back to defaults.
- Normalize indicators to 0..1, build composite EHI, map to downturn probability via thresholds.
- Optional 3-mo moving average smoothing to reduce month-to-month noise.
- Saves a CSV snapshot and prints a concise report.

Env vars
  export BLS_API_KEY="..."
  export FRED_API_KEY="..."

Install
  pip install pytrends pandas numpy requests python-dateutil
"""

import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path



# ------------------------------
# Config
# ------------------------------
REGION = "US-MA"
MONTHS_LOOKBACK = 24

TRENDS_TERMS = [
    "pottery class",
    "ceramics class",
    "pottery studio",
    "ceramics studio",
]

BLS_SERIES_ID_MA_UNEMP = "LASST250000000000003"  # MA unemployment rate (seasonally adjusted, percent)

WEIGHTS = {
    "google_trends": 0.40,
    "unemployment":  0.40,
    "sentiment":     0.20,
}

# Optional smoothing to stabilize signals
USE_SMOOTHING = os.getenv("GCWS_NC_SMOOTH", "1") == "1"
SMOOTH_WINDOW = int(os.getenv("GCWS_NC_SMOOTH_WIN", "3"))  # months

THRESHOLDS = [
    (0.70, 0.05),  # healthy
    (0.40, 0.15),  # neutral
    (0.00, 0.30),  # weak
]

# Default fallbacks when API not available
FALLBACK_UNEMPLOYMENT = 5.0     # percent
FALLBACK_SENTIMENT    = 80.0    # UMich headline, index level (typical range ~50-110)


CACHE_PATH = Path(os.getenv("GCWS_NC_CACHE", "~/.gcws/nowcast_cache.json")).expanduser()
CACHE_TTL_SEC = int(os.getenv("GCWS_NC_TTL_SEC", "21600"))  # 6 hours

def _cache_load():
    try:
        d = json.loads(CACHE_PATH.read_text())
        if time.time() - float(d.get("ts", 0)) <= CACHE_TTL_SEC:
            return d
    except Exception:
        pass
    return None

def _cache_save(result: dict):
    try:
        out = {
            "ts": time.time(),
            "prob": float(result.get("prob", float("nan"))),
            "ehi": float(result.get("ehi", float("nan"))),
            "parts": result.get("parts", {}),
            "as_of": result.get("as_of"),
        }
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(out))
    except Exception:
        pass


# ------------------------------
# Helpers
# ------------------------------
def normalize_series(s: pd.Series, invert: bool = False) -> pd.Series:
    s = s.astype(float).copy()
    if s.count() < 2 or s.max() == s.min():
        return pd.Series(np.nan, index=s.index)
    out = (s - s.min()) / (s.max() - s.min())
    return (1.0 - out) if invert else out

def monthly_from_weekly(df_weekly: pd.DataFrame, value_col: str, date_col: str = "date") -> pd.DataFrame:
    df = df_weekly.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    m = df.groupby("month")[value_col].mean().reset_index()
    return m.rename(columns={"month": "date"})

def trim_last_n_months(df: pd.DataFrame, date_col: str, n: int) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    cutoff = (pd.Timestamp.today().to_period("M").to_timestamp() - pd.offsets.MonthEnd(0)) - relativedelta(months=(n-1))
    return df[df[date_col] >= cutoff].sort_values(date_col)

def safe_weighted_sum(parts: dict, weights: dict) -> float:
    used = {k: v for k, v in parts.items() if pd.notna(v)}
    if not used:
        return float("nan")
    total_w = sum(weights.get(k, 0) for k in used.keys())
    if total_w <= 0:
        return np.mean(list(used.values()))
    return sum(weights[k] / total_w * used[k] for k in used.keys())

def ehi_to_prob(ehi: float) -> float:
    if pd.isna(ehi):
        return float("nan")
    for thr, p in THRESHOLDS:
        if ehi >= thr:
            return p
    return THRESHOLDS[-1][1]

# ------------------------------
# Data fetchers
# ------------------------------
def fetch_google_trends(region: str, terms: list) -> pd.DataFrame:
    """
    Fetch Google Trends for a list of terms in `region`, average them,
    convert weekly series to monthly, drop the current (partial) month,
    optionally smooth, and trim to MONTHS_LOOKBACK months.

    Returns a DataFrame with columns: ["date", "google_trends"]
    where 'date' is the month start (Timestamp), and 'google_trends' is float.
    """
    import time
    import pandas as pd
    import numpy as np

    try:
        from pytrends.request import TrendReq
    except Exception:
        return pd.DataFrame(columns=["date", "google_trends"])

    if not terms:
        return pd.DataFrame(columns=["date", "google_trends"])

    # One session; lower rate to avoid 429s
    pytrends = TrendReq(hl="en-US", tz=360, timeout=(10, 25), retries=2, backoff_factor=0.4)

    frames = []
    for term in terms:
        # Tiny retry loop per term (network hiccups happen)
        for attempt in range(3):
            try:
                pytrends.build_payload([term], timeframe="today 5-y", geo=region)
                data = pytrends.interest_over_time()
                if data is None or data.empty:
                    break
                # Some pytrends versions include 'isPartial' column; ignore it
                if "isPartial" in data.columns:
                    data = data.drop(columns=["isPartial"])
                ser = (data[term]
                       .rename("value")
                       .reset_index()
                       .rename(columns={"date": "date"}))  # weekly timestamps
                frames.append(ser[["date", "value"]])
                break
            except Exception:
                # brief backoff, then retry this term
                time.sleep(0.5 * (attempt + 1))
                if attempt == 2:
                    # give up on this term after 3 tries
                    pass
        # polite spacing between terms
        time.sleep(0.3)

    if not frames:
        return pd.DataFrame(columns=["date", "google_trends"])

    # Average all requested terms at the weekly level
    weekly = (pd.concat(frames, ignore_index=True)
                .groupby("date", as_index=False)["value"].mean())

    # Convert weekly → monthly using your helper
    monthly = monthly_from_weekly(weekly, value_col="value", date_col="date")
    monthly = monthly.rename(columns={"value": "google_trends"})

    # Ensure Timestamp, month-start alignment
    monthly["date"] = pd.to_datetime(monthly["date"]).dt.to_period("M").dt.to_timestamp()

    # Drop the current (partial) month to avoid intra-month wobble
    current_month_start = pd.Timestamp.today().to_period("M").to_timestamp()
    monthly = monthly[monthly["date"] < current_month_start]

    # Optional smoothing (same behavior as your other feeds)
    if USE_SMOOTHING:
        monthly["google_trends"] = (monthly["google_trends"]
                                    .rolling(SMOOTH_WINDOW, min_periods=1)
                                    .mean())

    # Trim lookback window (most recent N months, inclusive)
    monthly = trim_last_n_months(monthly, "date", MONTHS_LOOKBACK)

    # Guarantee dtype and column order
    monthly = (monthly[["date", "google_trends"]]
               .assign(google_trends=lambda d: d["google_trends"].astype(float)))

    return monthly

def fetch_bls_unemployment(series_id: str) -> pd.DataFrame:
    import requests
    url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    headers = {"Content-Type": "application/json"}
    start_year = (datetime.today() - relativedelta(months=MONTHS_LOOKBACK+4)).year
    body = {
        "seriesid": [series_id],
        "startyear": str(start_year),
        "endyear": str(datetime.today().year),
    }
    api_key = os.environ.get("BLS_API_KEY")
    if api_key:
        body["registrationKey"] = api_key
    try:
        r = requests.post(url, headers=headers, data=json.dumps(body), timeout=20)
        r.raise_for_status()
        j = r.json()
        series = j["Results"]["series"][0]["data"]
        rows = []
        for d in series:
            if not d["period"].startswith("M"):  # skip annual
                continue
            y = int(d["year"]); m = int(d["period"][1:])
            val = float(d["value"])
            dt = pd.Timestamp(year=y, month=m, day=1)
            rows.append({"date": dt, "unemployment": val})
        df = pd.DataFrame(rows).sort_values("date")
        df = trim_last_n_months(df, "date", MONTHS_LOOKBACK)
        if USE_SMOOTHING:
            df["unemployment"] = df["unemployment"].rolling(SMOOTH_WINDOW, min_periods=1).mean()
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "unemployment"])

def fetch_fred_umcsent() -> pd.DataFrame:
    import requests
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        return pd.DataFrame(columns=["date", "sentiment"])
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "UMCSENT",
        "api_key": api_key,
        "file_type": "json",
        "observation_start": (datetime.today() - relativedelta(months=MONTHS_LOOKBACK+4)).strftime("%Y-%m-01"),
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        obs = r.json()["observations"]
        rows = [{"date": pd.Timestamp(o["date"]), "sentiment": float(o["value"])}
                for o in obs if o.get("value") not in (".", None)]
        df = pd.DataFrame(rows).dropna().sort_values("date")
        df = trim_last_n_months(df, "date", MONTHS_LOOKBACK)
        if USE_SMOOTHING:
            df["sentiment"] = df["sentiment"].rolling(SMOOTH_WINDOW, min_periods=1).mean()
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "sentiment"])
    
def classify_trend(series: pd.Series, window: int = 3):
    """
    Classify trend using recent slope (mean of last `window` diffs).
    Returns (label, slope).
    """
    s = series.dropna()
    if len(s) < 2:
        return "neutral", 0.0
    one_mo = s.diff().iloc[-1]
    three_mo = s.diff().tail(window).mean()
    slope = three_mo if not np.isnan(three_mo) else one_mo

    # thresholds are small because EHI is 0..1
    if slope >= 0.02:
        label = "improving"
    elif slope <= -0.02:
        label = "worsening"
    else:
        label = "neutral"
    return label, float(slope)

# ------------------------------
# Main
# ------------------------------
# ------------------------------
# Core computation (no print, no I/O)
# ------------------------------
def compute_nowcast():
    """Return dict with keys: as_of, parts, ehi, prob, trend_label, trend_slope, df.
    Debounces by reusing a cached value within CACHE_TTL_SEC to prevent second-by-second swings.
    """

    # 0) Try cache first (debounce). If you want to bypass, set CACHE_TTL_SEC=0 or delete the cache file.
    cached = _cache_load()
    if cached is not None and np.isfinite(cached.get("prob", np.nan)):
        # Reuse immediately to prevent intra-day jitter; still compute live below only if you want FORCE_REFRESH behavior.
        return {
            "as_of": cached.get("as_of"),
            "parts": cached.get("parts", {}),
            "ehi": float(cached.get("ehi")) if np.isfinite(cached.get("ehi", np.nan)) else np.nan,
            "prob": float(cached.get("prob")),
            "trend_label": None,
            "trend_slope": np.nan,
            "df": pd.DataFrame(),  # optional; you can store a small history in cache if you want
        }

    gt   = fetch_google_trends(REGION, TRENDS_TERMS)
    bls  = fetch_bls_unemployment(BLS_SERIES_ID_MA_UNEMP)
    fred = fetch_fred_umcsent()

    parts = {}
    dfs = []

    if not gt.empty:
        gt["gt_norm"] = normalize_series(gt["google_trends"], invert=False)
        parts["google_trends"] = gt["gt_norm"].iloc[-1]
        dfs.append(gt[["date", "google_trends", "gt_norm"]])
    else:
        parts["google_trends"] = np.nan

    if not bls.empty:
        bls["unemp_norm"] = normalize_series(bls["unemployment"], invert=True)  # higher unemployment is worse
        parts["unemployment"] = bls["unemp_norm"].iloc[-1]
        dfs.append(bls[["date", "unemployment", "unemp_norm"]])
    else:
        parts["unemployment"] = np.nan

    if not fred.empty:
        fred["sent_norm"]  = normalize_series(fred["sentiment"], invert=False)
        parts["sentiment"] = fred["sent_norm"].iloc[-1]
        dfs.append(fred[["date", "sentiment", "sent_norm"]])
    else:
        parts["sentiment"] = np.nan

    # 1) Require at least 2 signals; otherwise reuse cache if available
    _available = [k for k in ("google_trends","unemployment","sentiment") if pd.notna(parts.get(k))]
    if len(_available) < 2:
        cached = _cache_load()
        if cached and np.isfinite(cached.get("prob", np.nan)):
            return {
                "df": pd.DataFrame(),
                "parts": parts,
                "ehi": np.nan,
                "prob": float(cached["prob"]),
                "as_of": cached.get("as_of"),
                "trend_label": None,
                "trend_slope": np.nan,
            }
        # else: continue and fall back via anchors below

    # 2) Merge timeline (outer join) and ensure required columns
    merged = None
    for d in dfs:
        merged = d if merged is None else pd.merge(merged, d, on="date", how="outer")
    if merged is None:
        today = pd.Timestamp.today().to_period("M").to_timestamp()
        merged = pd.DataFrame({"date": [today]})
    merged = merged.sort_values("date").reset_index(drop=True)

    for col in ("google_trends","unemployment","sentiment","gt_norm","unemp_norm","sent_norm"):
        if col not in merged.columns:
            merged[col] = np.nan

    # 3) Fallback anchors (so normalization has two points)
    latest_idx = merged.index[-1]
    if pd.isna(parts.get("unemployment")):
        merged.loc[latest_idx, "unemployment"] = FALLBACK_UNEMPLOYMENT
        tmp = pd.Series([FALLBACK_UNEMPLOYMENT - 0.5, FALLBACK_UNEMPLOYMENT])
        parts["unemployment"] = normalize_series(tmp, invert=True).iloc[-1]

    if pd.isna(parts.get("sentiment")):
        merged.loc[latest_idx, "sentiment"] = FALLBACK_SENTIMENT
        tmp = pd.Series([FALLBACK_SENTIMENT - 2, FALLBACK_SENTIMENT])
        parts["sentiment"] = normalize_series(tmp, invert=False).iloc[-1]

    # 4) Composite + mapping
    ehi  = safe_weighted_sum(parts, WEIGHTS)
    prob = ehi_to_prob(ehi)

    # 5) EHI history + simple trend label
    def row_ehi(row):
        parts_row = {}
        if pd.notna(row.get("gt_norm", np.nan)):
            parts_row["google_trends"]  = row["gt_norm"]
        if pd.notna(row.get("unemp_norm", np.nan)):
            parts_row["unemployment"]   = row["unemp_norm"]
        if pd.notna(row.get("sent_norm", np.nan)):
            parts_row["sentiment"]      = row["sent_norm"]
        return safe_weighted_sum(parts_row, WEIGHTS)

    merged["EHI"] = merged.apply(row_ehi, axis=1)
    trend_label, trend_slope = classify_trend(merged["EHI"], window=3)
    merged["EHI_change_1m"] = merged["EHI"].diff()
    merged["EHI_change_3m"] = merged["EHI"].diff().rolling(3, min_periods=1).mean()

    # 6) Persist fresh result to cache (debounce subsequent calls)
    as_of_str = pd.Timestamp.today().strftime("%Y-%m-%d")
    try:
        _cache_save({
            "prob": float(prob) if not pd.isna(prob) else np.nan,
            "ehi":  float(ehi)  if not pd.isna(ehi)  else np.nan,
            "parts": parts,
            "as_of": as_of_str,
        })
    except Exception:
        pass

    # Optional: quick transparency print
    used = ", ".join(_available) if _available else "none"
    print(f"[nowcast] signals_used={used}  EHI={ehi:.3f}  prob={prob:.3f}")

    return {
        "as_of": as_of_str,
        "parts": parts,
        "ehi": float(ehi) if not pd.isna(ehi) else np.nan,
        "prob": float(prob) if not pd.isna(prob) else np.nan,
        "trend_label": trend_label,
        "trend_slope": float(trend_slope) if not pd.isna(trend_slope) else np.nan,
        "df": merged,
    }


def run_nowcast(save_csv: bool = True):
    result = compute_nowcast()

    # Console report
    print("\n=== GCWS Downturn Nowcast ===")
    print(f"As of: {result['as_of']}\n")
    def pv(x): return "NA" if pd.isna(x) else f"{x:.3f}"
    print("Normalized indicators (0..1, higher = healthier):")
    print(f"  google_trends : {pv(result['parts'].get('google_trends', np.nan))}")
    print(f"  unemployment  : {pv(result['parts'].get('unemployment', np.nan))}")
    print(f"  sentiment     : {pv(result['parts'].get('sentiment', np.nan))}")
    print(f"Economic Health Index (EHI): {pv(result['ehi'])}")
    val = 'NA' if pd.isna(result['prob']) else f"{result['prob']:.2f}"
    print(f"Suggested DOWNTURN_PROB_PER_MONTH: {val}")

    if save_csv:
        out_path = os.path.join(os.getcwd(), "downturn_nowcast.csv")
        result["df"].to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")

    return result


def main():
    run_nowcast(save_csv=True)


if __name__ == "__main__":
    main()
