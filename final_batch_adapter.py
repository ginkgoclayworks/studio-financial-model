# batch_adapter.py
from __future__ import annotations
import json
import os
import runpy
from types import SimpleNamespace
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

# Try to import the modular entrypoint (preferred path).
try:
    from modular_simulator import run_from_cfg as _run_modular
except Exception:
    _run_modular = None  # we'll gracefully fallback to run_path


# -----------------------------
# Allowed external -> internal knobs
# -----------------------------
ALLOWED_OVERRIDES: Dict[str, str] = {
    # core batch knobs
    "N_SIMULATIONS": "N_SIMULATIONS",
    "MONTHS": "MONTHS",
    "RANDOM_SEED": "RANDOM_SEED",

    # economics
    "RENT": "RENT_SCENARIOS",                 # wrapped as np.array([RENT])
    "OWNER_DRAW": "OWNER_DRAW_SCENARIOS",     # wrapped as [OWNER_DRAW]

    # events
    "EVENTS_ENABLED": "EVENTS_ENABLED",
    "TICKET_PRICE": "TICKET_PRICE",
    "EVENTS_MAX_PER_MONTH": "EVENTS_MAX_PER_MONTH",
    "BASE_EVENTS_PER_MONTH_LAMBDA": "BASE_EVENTS_PER_MONTH_LAMBDA",
    "EVENT_STAFF_RATE_PER_HOUR": "EVENT_STAFF_RATE_PER_HOUR",
    "EVENT_HOURS_PER_EVENT": "EVENT_HOURS_PER_EVENT",
    "ATTENDEES_PER_EVENT_RANGE": "ATTENDEES_PER_EVENT_RANGE",      # NEW
    "EVENT_MUG_COST_RANGE": "EVENT_MUG_COST_RANGE",                # NEW
    "EVENT_CONSUMABLES_PER_PERSON": "EVENT_CONSUMABLES_PER_PERSON", # NEW

    # classes
    "CLASSES_ENABLED": "CLASSES_ENABLED",
    "CLASS_COHORTS_PER_MONTH": "CLASS_COHORTS_PER_MONTH",
    "CLASS_CAP_PER_COHORT": "CLASS_CAP_PER_COHORT",
    "CLASS_PRICE": "CLASS_PRICE",
    "CLASS_CONV_RATE": "CLASS_CONV_RATE",
    "CLASS_CONV_LAG_MO": "CLASS_CONV_LAG_MO",

    # loans
    "RUNWAY_MONTHS": "RUNWAY_MONTHS",
    "LOAN_CONTINGENCY_PCT": "LOAN_CONTINGENCY_PCT",
    "EXTRA_BUFFER": "EXTRA_BUFFER",
    "LOAN_504_ANNUAL_RATE": "LOAN_504_ANNUAL_RATE",
    "LOAN_504_TERM_YEARS": "LOAN_504_TERM_YEARS",
    "LOAN_7A_ANNUAL_RATE": "LOAN_7A_ANNUAL_RATE",
    "LOAN_7A_TERM_YEARS": "LOAN_7A_TERM_YEARS",

    # membership/capacity
    "MAX_MEMBERS": "MAX_MEMBERS",
    "PRICE": "PRICE",
}


def _row_to_overrides(row: pd.Series) -> dict:
    ov: Dict[str, Any] = {}
    for external_key, internal_key in ALLOWED_OVERRIDES.items():
        if external_key in row and pd.notna(row[external_key]):
            val = row[external_key]
            if external_key == "RENT":
                ov[internal_key] = np.array([float(val)], dtype=float)
            elif external_key == "OWNER_DRAW":
                ov[internal_key] = [float(val)]
            elif external_key in {
                "N_SIMULATIONS", "MONTHS", "RANDOM_SEED", "EVENTS_MAX_PER_MONTH",
                "RUNWAY_MONTHS", "LOAN_504_TERM_YEARS", "LOAN_7A_TERM_YEARS", "MAX_MEMBERS"
            }:
                ov[internal_key] = int(val)
            elif external_key in {"EVENTS_ENABLED", "CLASSES_ENABLED"}:
                ov[internal_key] = bool(val)
            elif external_key in {"ATTENDEES_PER_EVENT_RANGE", "EVENT_MUG_COST_RANGE"}:
                v = row[external_key]
                if isinstance(v, str):
                    try:
                        v = json.loads(v)
                    except Exception:
                        # fallback: comma-separated "8,10,12"
                        v = [float(x) for x in str(v).split(",")]
                ov[internal_key] = v
            elif external_key == "EVENT_CONSUMABLES_PER_PERSON":
                ov[internal_key] = float(row[external_key])
            else:
                ov[internal_key] = float(val) if isinstance(val, (int, float, np.integer, np.floating)) else val
    return ov


def _suppress_plots():
    # Force a headless backend for batch runs
    os.environ["MPLBACKEND"] = "Agg"
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
    except Exception:
        pass


def _prefer_modular_run(overrides: Optional[dict]) -> Optional[Tuple[pd.DataFrame, SimpleNamespace]]:
    """
    If modular_simulator.run_from_cfg is available, use it.
    Returns (df, eff) or None if unavailable.
    """
    if _run_modular is None:
        return None

    cfg = dict(overrides or {})
    # Normalize singleton RENT/OWNER_DRAW if caller provided plain values
    if "RENT" in cfg and "RENT_SCENARIOS" not in cfg:
        cfg["RENT_SCENARIOS"] = np.array([float(cfg.pop("RENT"))], dtype=float)
    if "OWNER_DRAW" in cfg and "OWNER_DRAW_SCENARIOS" not in cfg:
        cfg["OWNER_DRAW_SCENARIOS"] = [float(cfg.pop("OWNER_DRAW"))]

    _suppress_plots()
    art = _run_modular(cfg)

    # Expect the modular runner to return a dict-like with at least results_df
    if isinstance(art, dict) and "results_df" in art and art["results_df"] is not None:
        df = art["results_df"]
        # Capture a few effective knobs if present; otherwise synthesize from cfg
        eff_keys = [
            "MONTHS", "N_SIMULATIONS", "PRICE", "PRICE_ELASTICITY",
            "RENT_SCENARIOS", "OWNER_DRAW_SCENARIOS",
            "DOWNTURN_PROB_PER_MONTH", "CAPACITY_SOFT_CAP", "RANDOM_SEED",
        ]
        eff_src = art.get("effective_config", art)
        eff = {k: eff_src.get(k, cfg.get(k, None)) for k in eff_keys}
        return df, SimpleNamespace(**eff)

    # Could not extract df; treat as modular unavailable
    return None


def run_original_once(script_path: str, overrides: dict | None = None):
    """
    Run one simulation either via the modular entrypoint (preferred) or by executing the script
    with injected globals (legacy path). Returns (df, eff).
    """
    # 1) Prefer the modular entrypoint if present
    modular_res = _prefer_modular_run(overrides)
    if modular_res is not None:
        return modular_res

    # 2) Legacy: run the script file with injected globals
    g = {"__name__": "__main__", "__file__": script_path}
    if overrides:
        # Normalize RENT / OWNER_DRAW if provided
        ov = dict(overrides)
        if "RENT" in ov and "RENT_SCENARIOS" not in ov:
            ov["RENT_SCENARIOS"] = np.array([float(ov.pop("RENT"))], dtype=float)
        if "OWNER_DRAW" in ov and "OWNER_DRAW_SCENARIOS" not in ov:
            ov["OWNER_DRAW_SCENARIOS"] = [float(ov.pop("OWNER_DRAW"))]
        g.update(ov)

    _suppress_plots()
    ns = runpy.run_path(script_path, init_globals=g)

    # Try to find results DataFrame by common names
    df = None
    for key in ("RESULTS_DF", "results_df", "RESULTS", "df"):
        if key in ns and ns[key] is not None:
            df = ns[key]
            break

    if df is None:
        available = ", ".join(sorted(k for k in ns.keys() if k.isidentifier()))
        raise RuntimeError(
            "Simulator did not expose a results DataFrame. "
            "Looked for RESULTS_DF / results_df / RESULTS / df. "
            f"Available names: {available}"
        )

    # Snapshot a few effective globals for tooling
    keys = [
        "MONTHS", "N_SIMULATIONS", "PRICE", "PRICE_ELASTICITY",
        "RENT_SCENARIOS", "OWNER_DRAW_SCENARIOS",
        "DOWNTURN_PROB_PER_MONTH", "CAPACITY_SOFT_CAP", "RANDOM_SEED",
    ]
    eff = {k: ns.get(k, None) for k in keys}
    return df, SimpleNamespace(**eff)


# -----------------------------
# Lender summary helpers
# -----------------------------
def lender_summary_from_results(results_df: pd.DataFrame) -> pd.DataFrame:
    def first_be(group: pd.DataFrame):
        be_m = group.loc[group["cumulative_op_profit"] >= 0, "month"].min()
        return be_m if pd.notna(be_m) else np.nan

    be_df = (results_df
             .groupby(["scenario", "rent", "owner_draw", "simulation_id"])
             .apply(first_be)
             .reset_index(name="op_break_even_month"))

    be_median = (be_df
                 .groupby(["scenario", "rent", "owner_draw"])["op_break_even_month"]
                 .median()
                 .reset_index(name="median_be_month"))

    # Loan sizing (month 1 median)
    loan_med = (results_df[results_df["month"] == 1]
                .groupby(["scenario","rent","owner_draw"])
                [["loan_principal_total","loan_principal_504","loan_principal_7a",
                  "loan_payment_total","loan_payment_504","loan_payment_7a"]]
                .median()
                .reset_index())

    def med_at(m, col):
        dfm = (results_df[results_df["month"] == m]
               .groupby(["scenario","rent","owner_draw"])[col]
               .median())
        return dfm.rename(f"{col}@{m}")

    out = (loan_med
           .merge(be_median, on=["scenario","rent","owner_draw"], how="left")
           .merge(med_at(12, "dscr"), on=["scenario","rent","owner_draw"], how="left")
           .merge(med_at(24, "dscr"), on=["scenario","rent","owner_draw"], how="left")
           .merge(med_at(12, "dscr_cash"), on=["scenario","rent","owner_draw"], how="left")
           .merge(med_at(24, "dscr_cash"), on=["scenario","rent","owner_draw"], how="left"))

    for col in ["dscr@12", "dscr@24", "dscr_cash@12", "dscr_cash@24"]:
        if col in out.columns:
            out[col] = out[col].round(2)

    # Min cash (median across sims) and cash@T
    min_cash = (results_df
                .groupby(["scenario","rent","owner_draw","simulation_id"])["cash_balance"]
                .min()
                .groupby(level=[0,1,2]).median()
                .reset_index(name="median_min_cash"))

    mmax = int(results_df["month"].max())
    cashT = (results_df[results_df["month"] == mmax]
             .groupby(["scenario","rent","owner_draw"])["cash_balance"]
             .median()
             .reset_index(name=f"median_cash@M{mmax}"))

    insol = (results_df
             .groupby(["scenario","rent","owner_draw","simulation_id"])["insolvent_before_grant"]
             .max()
             .groupby(level=[0,1,2]).mean()
             .reset_index(name="pct_sims_insolvent_pregrant"))

    out = (out
           .merge(min_cash, on=["scenario","rent","owner_draw"], how="left")
           .merge(cashT, on=["scenario","rent","owner_draw"], how="left")
           .merge(insol, on=["scenario","rent","owner_draw"], how="left"))

    return out


def run_batch(script_path: str, scenarios: pd.DataFrame) -> pd.DataFrame:
    """
    Run a batch defined by a DataFrame of scenario rows.
    Each row is mapped to an overrides dict via _row_to_overrides.
    Returns a concatenated lender summary table.
    """
    lender_rows = []
    for i, row in scenarios.reset_index(drop=True).iterrows():
        ov = _row_to_overrides(row)

        # Ensure singleton wrappers if caller passed plain numbers
        if "RENT_SCENARIOS" not in ov and "RENT" in row and pd.notna(row["RENT"]):
            ov["RENT_SCENARIOS"] = np.array([float(row["RENT"])], dtype=float)
        if "OWNER_DRAW_SCENARIOS" not in ov and "OWNER_DRAW" in row and pd.notna(row["OWNER_DRAW"]):
            ov["OWNER_DRAW_SCENARIOS"] = [float(row["OWNER_DRAW"])]

        res = run_original_once(script_path, ov)
        df, _eff = res if isinstance(res, tuple) else (res, None)

        summ = lender_summary_from_results(df)
        summ.insert(0, "scenario_id", i)
        lender_rows.append(summ)

    return pd.concat(lender_rows, ignore_index=True) if lender_rows else pd.DataFrame()