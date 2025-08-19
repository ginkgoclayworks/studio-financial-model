#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 22:29:13 2025

@author: harshadghodke
"""

# app.py
import io, json, re, shutil, zipfile
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns  # for heatmaps

# Your adapter: must expose run_original_once(script_path, overrides_dict)
from final_batch_adapter import run_original_once

SCRIPT = "modular_simulator.py"   # your core simulator

# ---------- small helpers ----------
def slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9._-]+", "", s)
    return s[:80]

def sanitize_text(s: str) -> str:
    return re.sub(r"[\u2010-\u2015\u2212\u00AD\u200B\uFEFF\u202F]", "-", s)

def build_overrides(env: dict, strat: dict) -> dict:
    """Use ONLY globals your simulator already knows about."""
    ov = {}
    for k, v in env.items():
        if k not in {"name", "grant_month", "grant_amount"}:
            ov[k] = v
    for k, v in strat.items():
        if k != "name":
            ov[k] = v
    # singletons for sweeps
    if "RENT" in strat:
        ov["RENT_SCENARIOS"] = np.array([float(strat["RENT"])], dtype=float)
    if "OWNER_DRAW" in strat:
        ov["OWNER_DRAW_SCENARIOS"] = [float(strat["OWNER_DRAW"])]
    # pass grant + capex timing through SCENARIO_CONFIGS
    ov["SCENARIO_CONFIGS"] = [{
        "name": strat["name"],
        "capex_timing": "all" if ("all_upfront" in strat["name"]) else "staged",
        "grant_amount": env.get("grant_amount", 0.0),
        "grant_month": env.get("grant_month", None),
    }]
    return ov

# ---- capture all plt.show() calls from your modular_simulator without touching it
class FigureCapture:
    def __init__(self, title_suffix: str = ""):
        self.title_suffix = title_suffix
        self._orig_show = None
        self.images: List[Tuple[str, bytes]] = []
        self.manifest = []

    def __enter__(self):
        matplotlib.use("Agg", force=True)
        self._orig_show = plt.show
        counter = {"i": 0}

        def _title_for(fig):
            parts = []
            if fig._suptitle:
                txt = fig._suptitle.get_text()
                if txt:
                    parts.append(txt)
            for ax in fig.axes:
                t = getattr(ax, "get_title", lambda: "")()
                if t:
                    parts.append(t)
            return " | ".join(parts).strip()

        def _ensure_suffix(fig):
            if not self.title_suffix:
                return
            current = fig._suptitle.get_text() if fig._suptitle else ""
            if current:
                if self.title_suffix not in current:
                    fig._suptitle.set_text(f"{current} — {self.title_suffix}")
            else:
                fig.suptitle(self.title_suffix)

        def _show(*args, **kwargs):
            counter["i"] += 1
            fig = plt.gcf()
            _ensure_suffix(fig)
            buf = io.BytesIO()
            fig.savefig(buf, dpi=200, bbox_inches="tight", format="png")
            buf.seek(0)
            fname = f"fig_{counter['i']:02d}.png"
            self.images.append((fname, buf.read()))
            self.manifest.append({"file": fname, "title": _title_for(fig)})
            plt.close(fig)

        plt.show = _show
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig_show:
            plt.show = self._orig_show

# ---------- caching ----------
@st.cache_data(show_spinner=False)
def run_cell_cached(env: dict, strat: dict, seed: int):
    ov = build_overrides(env, strat)
    ov["RANDOM_SEED"] = seed
    title_suffix = f"{env['name']} | {strat['name']}"
    with FigureCapture(title_suffix) as cap:
        res = run_original_once(SCRIPT, ov)
    df_cell, eff = (res if isinstance(res, tuple) else (res, None))
    return df_cell, eff, cap.images, cap.manifest

# ---------- metrics ----------
def first_insolvency_month(series: pd.Series) -> float:
    s = series.dropna()
    idx = s.index[s.values < 0]
    return float(idx.min()) if len(idx) else np.nan

def first_sustained_ge_zero(series: pd.Series, k: int = 3) -> float:
    s = series.dropna().sort_index()
    ok = (s >= 0).astype(int).rolling(k, min_periods=k).sum() == k
    idx = ok[ok].index
    return float(idx.min()) if len(idx) else np.nan

def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def summarize_cell(df: pd.DataFrame) -> pd.DataFrame:
    # expects columns: environment, strategy, simulation_id, month, cash_balance, dscr, (maybe cfads)
    keys = ["environment", "strategy", "simulation_id"]
    T = int(df["month"].max())
    cash_col = pick_col(df, ["cash_balance","cash","ending_cash"])
    cf_col   = pick_col(df, ["cfads","operating_cash_flow","op_cf","net_cash_flow","cash_flow"])
    if cash_col is None:
        raise RuntimeError("cash balance column not found")
    if cf_col is None:
        df["_fallback_cf"] = (df.sort_values(keys+["month"])
                                .groupby(keys)[cash_col].diff().fillna(0.0))
        cf_col = "_fallback_cf"

    grp = df.sort_values(keys+["month"]).groupby(keys)
    t_ins = grp.apply(lambda g: first_insolvency_month(g.set_index("month")[cash_col])).reset_index(name="t_insolvency")
    t_be  = grp.apply(lambda g: first_sustained_ge_zero(g.set_index("month")[cf_col], k=3)).reset_index(name="t_breakeven")
    timings = t_ins.merge(t_be, on=keys, how="outer")

    # survival by “ever negative cash”
    ever_neg = (grp[cash_col].min().reset_index(name="min_cash")
                    .assign(neg=lambda d: d["min_cash"] < 0)
                    .groupby(["environment","strategy"])["neg"].mean()
                    .reset_index(name="prob_insolvent_by_T"))
    surv = ever_neg.assign(survival_prob=lambda d: 1.0 - d["prob_insolvent_by_T"])[["environment","strategy","survival_prob"]]

    # end-of-horizon cash + DSCR 12
    last = df[df["month"] == T]
    cash_q = (last.groupby(["environment","strategy"])[cash_col]
                  .quantile([0.10,0.50,0.90]).unstack().reset_index()
                  .rename(columns={0.10:"cash_q10",0.50:"cash_med",0.90:"cash_q90"}))
    m12 = 12 if T >= 12 else T
    dscr_q = (df[df["month"] == m12]
                .groupby(["environment","strategy"])["dscr"]
                .quantile([0.10,0.50,0.90]).unstack().reset_index()
                .rename(columns={0.10:"dscr_q10",0.50:"dscr_med",0.90:"dscr_q90"}))

    # timing medians
    def med(s): 
        s = s.replace([np.inf,-np.inf], np.nan).dropna()
        return float(s.median()) if len(s) else np.nan
    timing_summary = (timings.groupby(["environment","strategy"])
                      .agg(median_time_to_insolvency_months=("t_insolvency", med),
                           median_time_to_breakeven_months=("t_breakeven", med))
                      .reset_index())

    matrix_row = (surv.merge(cash_q, on=["environment","strategy"], how="left")
                      .merge(dscr_q, on=["environment","strategy"], how="left")
                      .merge(timing_summary, on=["environment","strategy"], how="left"))
    return matrix_row, timings

# ---------- UI ----------
st.set_page_config(page_title="GCWS Simulator", layout="wide")

st.title("Ginkgo Clayworks — Scenario Explorer")

# Presets (match your code)
SCENARIOS = [
    {"name":"Baseline", "DOWNTURN_PROB_PER_MONTH":0.05, "DOWNTURN_JOIN_MULT":1.0, "DOWNTURN_CHURN_MULT":1.0,
     "MARKET_POOLS_INFLOW":{"community_studio":4,"home_studio":2,"no_access":3}, "grant_amount":0.0, "grant_month":None},
    {"name":"Recession", "DOWNTURN_PROB_PER_MONTH":0.18, "DOWNTURN_JOIN_MULT":0.65, "DOWNTURN_CHURN_MULT":1.50,
     "MARKET_POOLS_INFLOW":{"community_studio":2,"home_studio":1,"no_access":1}, "grant_amount":0.0, "grant_month":None},
    {"name":"SlowRecovery_Grant25k_M4", "DOWNTURN_PROB_PER_MONTH":0.10, "DOWNTURN_JOIN_MULT":0.85, "DOWNTURN_CHURN_MULT":1.20,
     "MARKET_POOLS_INFLOW":{"community_studio":3,"home_studio":1,"no_access":2}, "grant_amount":25_000, "grant_month":4},
    {"name":"Boom", "DOWNTURN_PROB_PER_MONTH":0.02, "DOWNTURN_JOIN_MULT":1.20, "DOWNTURN_CHURN_MULT":0.85,
     "MARKET_POOLS_INFLOW":{"community_studio":6,"home_studio":3,"no_access":4}, "grant_amount":0.0, "grant_month":None},
    {"name":"GreatDepression", "DOWNTURN_PROB_PER_MONTH":0.28, "DOWNTURN_JOIN_MULT":0.45, "DOWNTURN_CHURN_MULT":1.90,
     "MARKET_POOLS_INFLOW":{"community_studio":1,"home_studio":0,"no_access":1}, "grant_amount":0.0, "grant_month":None},
]
STRATEGIES = [
    {"name":"I_all_upfront_Base", "RENT":3500, "OWNER_DRAW":1000},
    {"name":"II_staged_Base",     "RENT":3500, "OWNER_DRAW":1000},
]

# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    scen_names = [s["name"] for s in SCENARIOS]
    strat_names = [s["name"] for s in STRATEGIES]
    scen_sel = st.selectbox("Scenario", scen_names, index=0)
    strat_sel = st.selectbox("Strategy", strat_names, index=0)
    seed = st.number_input("Random seed", value=42, step=1)
    st.caption("Adjust specific params below (optional):")
    env = json.loads(json.dumps(next(s for s in SCENARIOS if s["name"] == scen_sel)))
    strat = json.loads(json.dumps(next(s for s in STRATEGIES if s["name"] == strat_sel)))
    # quick tweaks
    strat["RENT"] = st.slider("Rent ($/mo)", 2000, 6000, int(strat["RENT"]), 100)
    strat["OWNER_DRAW"] = st.slider("Owner draw ($/mo)", 0, 3000, int(strat["OWNER_DRAW"]), 100)
    env["DOWNTURN_PROB_PER_MONTH"] = st.slider("Downturn prob / mo", 0.0, 0.5, float(env["DOWNTURN_PROB_PER_MONTH"]), 0.01)
    run_btn = st.button("Run simulation", type="primary")

# Tabs
tab_run, tab_matrix = st.tabs(["Single run", "Matrix heatmaps"])

# ---- Single run
with tab_run:
    if run_btn:
        with st.spinner("Running simulator…"):
            df_cell, eff, images, manifest = run_cell_cached(env, strat, seed)
            row, timings = summarize_cell(df_cell)

        st.subheader(f"KPIs — {env['name']} | {strat['name']}")
        kpi = row.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Survival prob @ horizon", f"{kpi['survival_prob']:.2f}")
        col2.metric("Cash p10 → p50 ($k)", f"{kpi['cash_q10']/1e3:,.0f} → {kpi['cash_med']/1e3:,.0f}")
        col3.metric("DSCR @ M12 (p50)", f"{kpi['dscr_med']:.2f}")
        col4.metric("Breakeven (median, months)", f"{kpi['median_time_to_breakeven_months']:.0f}")

        st.markdown("#### Captured charts")
        for fname, data in images:
            st.image(data, caption=fname, use_column_width=True)

        # Download bundle (PNGs + manifest) for the cell
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            for fname, data in images:
                zf.writestr(fname, data)
        st.download_button("Download plots (zip)", data=buf.getvalue(),
                           file_name=f"{slug(env['name'])}__{slug(strat['name'])}_plots.zip")

        st.markdown("#### Raw results (first 250 rows)")
        st.dataframe(df_cell.head(250))

# ---- Matrix heatmaps
with tab_matrix:
    st.caption("Runs all presets in SCENARIOS × STRATEGIES with independent seeds.")
    if st.button("Build matrix"):
        with st.spinner("Running matrix…"):
            pieces = []
            for i, E in enumerate(SCENARIOS):
                for j, S in enumerate(STRATEGIES):
                    df_cell, eff, _imgs, _man = run_cell_cached(E, S, 42 + 1000*(i*len(STRATEGIES)+j))
                    row, _ = summarize_cell(df_cell)
                    pieces.append(row)
            matrix = pd.concat(pieces, ignore_index=True)

        st.markdown("##### Survival probability")
        pv = (matrix.pivot(index="environment", columns="strategy", values="survival_prob")
                    .reindex(index=sorted(matrix["environment"].unique()),
                             columns=sorted(matrix["strategy"].unique())))
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(pv, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax,
                    cbar_kws={"label": "probability"})
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("Survival probability (pre‑grant)")
        st.pyplot(fig)

        st.markdown("##### Median cash at horizon ($k)")
        pv = (matrix.pivot(index="environment", columns="strategy", values="cash_med")/1000.0)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(pv, annot=True, fmt=".0f", cmap="YlOrBr", ax=ax, cbar_kws={"label":"$ thousands"})
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("Median cash at horizon ($k)")
        st.pyplot(fig)

        st.markdown("##### Median time to breakeven (months)")
        pv = matrix.pivot(index="environment", columns="strategy", values="median_time_to_breakeven_months")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(pv, annot=True, fmt=".0f", cmap="Greens", ax=ax, cbar_kws={"label":"months"})
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("Median time to breakeven (months)")
        st.pyplot(fig)

        st.markdown("##### Median time to insolvency (months)")
        pv = matrix.pivot(index="environment", columns="strategy", values="median_time_to_insolvency_months")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(pv, annot=True, fmt=".0f", cmap="Reds", ax=ax, cbar_kws={"label":"months"})
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("Median time to insolvency (months)")
        st.pyplot(fig)

        st.download_button("Download matrix CSV",
                           data=matrix.to_csv(index=False).encode("utf-8"),
                           file_name="matrix_summary.csv", mime="text/csv")