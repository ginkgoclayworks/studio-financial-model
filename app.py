# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Tue Aug 19 07:59:18 2025

# @author: harshadghodke


# Streamlit app for GCWS simulator
# """

import io, json, re, zipfile
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns  # for heatmaps

# Your adapter: must expose run_original_once(script_path, overrides_dict)
from final_batch_adapter import run_original_once



# put this near the top of app.py (after imports)
PARAM_SPECS = {
    # -------- Environment (scenario) --------
    "DOWNTURN_PROB_PER_MONTH": {"type": "float", "min": 0.0, "max": 0.5, "step": 0.01, "label": "Downturn prob / mo"},
    "DOWNTURN_JOIN_MULT":     {"type": "float", "min": 0.2, "max": 1.5, "step": 0.01, "label": "Join multiplier in downturn"},
    "DOWNTURN_CHURN_MULT":    {"type": "float", "min": 0.5, "max": 3.0, "step": 0.05, "label": "Churn multiplier in downturn"},
    "MARKET_POOLS_INFLOW": {"type": "market_inflow", "label": "Market inflow"},
    "grant_amount":           {"type": "int",   "min": 0, "max": 100_000, "step": 1000, "label": "Grant amount"},
    "grant_month":            {"type": "int",   "min": -1, "max": 36, "step": 1, "label": "Grant month (None=-1)"},
    # Optional future levers:
    "PRICE_ELASTICITY":       {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "label": "Price elasticity ε"},
    "REF_PRICE":              {"type": "int",   "min": 50, "max": 500, "step": 10, "label": "Reference price"},
    "WOM_RATE":               {"type": "float", "min": 0.0, "max": 0.2, "step": 0.005, "label": "Word-of-mouth rate"},
    "MARKETING_SPEND":        {"type": "int",   "min": 0, "max": 20_000, "step": 500, "label": "Marketing spend / mo"},
    "CAC":                    {"type": "int",   "min": 50, "max": 2000, "step": 10, "label": "CAC ($/lead)"},
    "LEAD_TO_JOIN_RATE":      {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "label": "Lead→Join conversion"},
    "MAX_ONBOARD_PER_MONTH":  {"type": "int",   "min": 1, "max": 200, "step": 1, "label": "Max onboarding / mo"},
    # -------- Strategy --------
    "RENT":                   {"type": "int",   "min": 1000, "max": 10_000, "step": 50, "label": "Rent ($/mo)"},
    "OWNER_DRAW":             {"type": "int",   "min": 0, "max": 5000, "step": 50, "label": "Owner draw ($/mo)"},
    "BASE_EVENTS_PER_MONTH_LAMBDA": {"type": "float", "min": 0.0, "max": 20.0, "step": 0.5, "label": "Events λ"},
    "EVENTS_MAX_PER_MONTH":   {"type": "int",   "min": 0, "max": 20, "step": 1, "label": "Events max / mo"},
    "TICKET_PRICE":           {"type": "int",   "min": 0, "max": 500, "step": 5, "label": "Ticket price"},
    "CLASSES_ENABLED":        {"type": "bool",  "label": "Classes enabled"},
    "CLASS_COHORTS_PER_MONTH": {"type": "int",  "min": 0, "max": 12, "step": 1, "label": "Class cohorts / mo"},
    "CLASS_CAP_PER_COHORT":   {"type": "int",   "min": 1, "max": 30, "step": 1, "label": "Class cap / cohort"},
    "CLASS_PRICE":            {"type": "int",   "min": 0, "max": 1000, "step": 10, "label": "Class price"},
    "CLASS_CONV_RATE":        {"type": "float", "min": 0.0, "max": 1.0, "step": 0.01, "label": "Class→Member conv"},
    "CLASS_CONV_LAG_MO":      {"type": "int",   "min": 0, "max": 12, "step": 1, "label": "Class conv lag (mo)"},
        # In PARAM_SPECS (near the other Strategy/Income items)
    "MEMBER_CAP": {
        "type": "int", "min": -1, "max": 500, "step": 1, "label": "Member cap (hard limit)"
    },
    "EXPANSION_THRESHOLD": {
        "type": "int", "min": 0, "max": 200, "step": 1, "label": "Expansion threshold (members)"
    },
}

ENV_SPEC_META = {
    "DOWNTURN_PROB_PER_MONTH": {
        "desc": "Chance the local economy is rough this month. 0 = great conditions; higher = harder to acquire/retain members.",
        "rec": (0.05, 0.25)
    },
    "DOWNTURN_JOIN_MULT": {
        "desc": "How much downturns reduce new joins. 1.0 = no change; 0.6 means 40% fewer joins in a bad month.",
        "rec": (0.6, 1.1)
    },
    "DOWNTURN_CHURN_MULT": {
        "desc": "How much downturns increase member churn. 1.0 = no change; 1.5 = 50% higher churn when the economy dips.",
        "rec": (1.0, 1.8)
    },
    "MARKET_POOLS_INFLOW": {
        "desc": "Rough monthly counts of potential joiners by pool: community studio users, home studio users, and people with no access.",
        "rec": (0, 10)
    },
    "grant_amount": {
        "desc": "One-time grant dollars injected into cash.",
        "rec": (0, 50000)
    },
    "grant_month": {
        "desc": "When the grant arrives (−1 = no grant). Month 0 is the start of operations.",
        "rec": (0, 12)
    },
    "PRICE_ELASTICITY": {
        "desc": "Sensitivity of joins to price. 0 = insensitive; higher values mean price changes move demand more.",
        "rec": (0.1, 0.6)
    },
    "REF_PRICE": {
        "desc": "Anchor price used for modeling price sensitivity (does not force your actual price; it shapes elasticity math).",
        "rec": (120, 220)
    },
    "WOM_RATE": {
        "desc": "Word-of-mouth strength. Approx fraction of members who generate a qualified lead each month.",
        "rec": (0.01, 0.06)
    },
    "MARKETING_SPEND": {
        "desc": "Monthly paid marketing budget (ads, sponsorships, flyers).",
        "rec": (0, 3000)
    },
    "CAC": {
        "desc": "Cost to acquire one qualified lead through paid channels.",
        "rec": (75, 250)
    },
    "LEAD_TO_JOIN_RATE": {
        "desc": "Share of qualified leads that become paying members.",
        "rec": (0.10, 0.35)
    },
    "MAX_ONBOARD_PER_MONTH": {
        "desc": "Operational limit on new member onboarding per month (paperwork, training, shelf setup).",
        "rec": (6, 20)
    },
    "MEMBER_CAP": {
        "desc": "Hard cap on total active members (0 or blank = let the simulator’s internal station bottlenecks govern).",
        "rec": (70, 110)
    },
    "EXPANSION_THRESHOLD": {
        "desc": "Member count at which you trigger an expansion (e.g., add wheels/racks/kiln).",
        "rec": (18, 30)
    },
}

for key, meta in ENV_SPEC_META.items():
    if key not in PARAM_SPECS:
        raise KeyError(f"PARAM_SPECS missing base spec for {key} (needs type/min/max/etc).")
    PARAM_SPECS[key].update(meta)    

    
STRAT_SPEC_META = {
    "RENT": {
        "desc": "Monthly base rent for the space (excludes utilities and insurance).",
        "rec": (2500, 5500)
    },
    "OWNER_DRAW": {
        "desc": "Cash you pay yourself each month from the business.",
        "rec": (0, 1500)
    },
    "BASE_EVENTS_PER_MONTH_LAMBDA": {
        "desc": "Average number of public events per month (Poisson-like). 0 disables events.",
        "rec": (0, 4)
    },
    "EVENTS_MAX_PER_MONTH": {
        "desc": "Upper bound on events you’re willing to host in any month (ops limit).",
        "rec": (0, 6)
    },
    "TICKET_PRICE": {
        "desc": "Ticket price for public events (make-a-mug, paint-your-own, etc.).",
        "rec": (55, 110)
    },
    "CLASSES_ENABLED": {
        "desc": "Enable/disable course cohorts (recurring multi-week classes).",
        "rec": (0, 1)
    },
    "CLASS_COHORTS_PER_MONTH": {
        "desc": "How many new class groups you start each month.",
        "rec": (0, 4)
    },
    "CLASS_CAP_PER_COHORT": {
        "desc": "Seats per cohort (max class size you can teach well).",
        "rec": (6, 14)
    },
    "CLASS_PRICE": {
        "desc": "Tuition per seat for a full cohort (entire multi-week course).",
        "rec": (200, 600)
    },
    "CLASS_CONV_RATE": {
        "desc": "Share of class participants who become ongoing members afterwards.",
        "rec": (0.05, 0.25)
    },
    "CLASS_CONV_LAG_MO": {
        "desc": "Delay between finishing a class and joining as a member (in months).",
        "rec": (0, 2)
    },
}

for key, meta in STRAT_SPEC_META.items():
    if key not in PARAM_SPECS:
        raise KeyError(f"PARAM_SPECS missing base spec for {key} (needs type/min/max/etc).")
    PARAM_SPECS[key].update(meta)
    

SCRIPT = "modular_simulator.py"   # your core simulator

# --- Group definitions ---

GROUPS = {
    # Scenario-owned
    "Income_env":   ["WOM_RATE", "LEAD_TO_JOIN_RATE", "MAX_ONBOARD_PER_MONTH"],
    "Expenses_env": ["MARKETING_SPEND", "CAC"],
    "Macro_env":    ["DOWNTURN_PROB_PER_MONTH", "DOWNTURN_JOIN_MULT",
                     "DOWNTURN_CHURN_MULT", "MARKET_POOLS_INFLOW",
                     "grant_amount", "grant_month"],
    "Capacity_env": ["MEMBER_CAP", "EXPANSION_THRESHOLD"],

    # Strategy-owned
    "Income_strat": ["REF_PRICE", "PRICE_ELASTICITY",
                     "CLASSES_ENABLED", "CLASS_COHORTS_PER_MONTH",
                     "CLASS_CAP_PER_COHORT", "CLASS_PRICE",
                     "CLASS_CONV_RATE", "CLASS_CONV_LAG_MO"],
    "Expenses_strat": ["RENT", "OWNER_DRAW"],
}


def compute_kpis_from_cell(df_cell: pd.DataFrame) -> dict:
    """
    Compute lender-style KPIs from a single cell's simulation dataframe.
    Assumes columns: month, simulation_id, cash; optionally dscr, active_members.
    """
    out = {}
    if df_cell.empty:
        return out

    # Horizon row per simulation
    last_month = int(df_cell["month"].max())
    end = df_cell[df_cell["month"] == last_month]

    # Survival: share of sims whose min cash never dipped below 0
    if {"simulation_id", "cash"}.issubset(df_cell.columns):
        min_cash_by_sim = df_cell.groupby("simulation_id")["cash"].min()
        out["survival_prob"] = float((min_cash_by_sim >= 0).mean())
        out["cash_q10"] = float(end["cash"].quantile(0.10))
        out["cash_med"] = float(end["cash"].quantile(0.50))
        out["cash_q90"] = float(end["cash"].quantile(0.90))

    # DSCR at horizon (if present)
    if "dscr" in end.columns:
        out["dscr_q10"] = float(end["dscr"].quantile(0.10))
        out["dscr_med"] = float(end["dscr"].quantile(0.50))
        out["dscr_q90"] = float(end["dscr"].quantile(0.90))

    # Members at horizon (optional, handy for display)
    if "active_members" in end.columns:
        out["members_med"] = float(end["active_members"].median())

    return out

def _subset(d, keys):
    return {k: d[k] for k in keys if k in d}

def _update_from(src, dst, keys):
    for k in keys:
        if k in src:
            dst[k] = src[k]

def _default_from_spec_for_push(key: str, spec: dict):
    t = (spec or {}).get("type")
    if key == "grant_month":          # sentinel: -1 means “None”
        return -1
    if t == "bool":  return False
    if t == "int":   return int(spec.get("min", 0))
    if t == "float": return float(spec.get("min", 0.0))
    if t == "market_inflow":
        return {"community_studio": 0, "home_studio": 0, "no_access": 0}
    if t == "json":  return {}
    return ""

def _clamp_num_for_push(val, lo, hi, typ):
    try:
        if typ == int:
            return int(min(max(val, lo), hi))
        else:
            return float(min(max(val, lo), hi))
    except Exception:
        return lo

def _canon(o):
    def _default(x):
        import numpy as np
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        raise TypeError
    return json.dumps(o, sort_keys=True, separators=(",", ":"), default=_default)

def _make_cache_key(env: dict, strat: dict, seed: int) -> str:
    return f"v4|{_canon(env)}|{_canon(strat)}|{seed}"

def _push_preset_to_widgets(preset: dict, *, prefix: str, keys: list):
    """
    Push preset values into st.session_state for all widgets in `keys`
    using the given `prefix` (must match render_param_controls’ wid_key).
    Handles special types and fills defaults when preset lacks a key.
    """
    for k in keys:
        spec = PARAM_SPECS.get(k, {})
        raw = preset.get(k, None)

        # Fill defaults when missing/None
        if raw is None:
            raw = _default_from_spec_for_push(k, spec)

        t = spec.get("type")

        if t == "bool":
            st.session_state[f"{prefix}_{k}"] = bool(raw)

        elif t == "int":
            lo = int(spec.get("min", 0)); hi = int(spec.get("max", 100))
            val = _clamp_num_for_push(raw, lo, hi, int)
            st.session_state[f"{prefix}_{k}"] = int(val)

        elif t == "float":
            lo = float(spec.get("min", 0.0)); hi = float(spec.get("max", 1.0))
            val = _clamp_num_for_push(raw, lo, hi, float)
            st.session_state[f"{prefix}_{k}"] = float(val)

        elif t == "market_inflow":
            cur = raw if isinstance(raw, dict) else {}
            cur = {
                "community_studio": int(max(0, cur.get("community_studio", 0))),
                "home_studio":      int(max(0, cur.get("home_studio", 0))),
                "no_access":        int(max(0, cur.get("no_access", 0))),
            }
            # set the compound value (if your code reads it)
            st.session_state[f"{prefix}_{k}"] = cur
            # set the three sub‑slider keys that are actually rendered
            base = f"{prefix}_{k}"
            st.session_state[f"{base}_c"] = cur["community_studio"]
            st.session_state[f"{base}_h"] = cur["home_studio"]
            st.session_state[f"{base}_n"] = cur["no_access"]

        else:
            # text/json fallback
            st.session_state[f"{prefix}_{k}"] = raw

            
# ---------- small helpers ----------
def _normalize_market_inflow(d: dict) -> dict:
    # Keep only the three known pools; coerce to non-negative ints.
    pools = {
        "community_studio": d.get("community_studio", 0),
        "home_studio":      d.get("home_studio", 0),
        "no_access":        d.get("no_access", 0),
    }
    out = {}
    for k, v in pools.items():
        try:
            out[k] = max(0, int(v))
        except Exception:
            out[k] = 0
    return out

def slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9._-]+", "", s)
    return s[:80]

def build_overrides(env: dict, strat: dict) -> dict:
    """Assemble the overrides payload for modular_simulator.py.

    Rules:
      - Copy env/strat values except meta fields.
      - Add singletons for sweepable values (RENT/OWNER_DRAW) so the core code
        that expects *_SCENARIOS keeps working.
      - Pass grant + capex timing through SCENARIO_CONFIGS.
      - UI capacity mapping: MEMBER_CAP (UI) -> HARD_CAP (sim). Only if > 0.
      - Forward EXPANSION_THRESHOLD when provided (>= 0).
      - Remove UI-only keys and any None/NaN values.
    """
    import math
    ov: dict = {}

    META_SKIP = {"name", "grant_month", "grant_amount"}  # kept only in SCENARIO_CONFIGS

    def _merge_clean(src: dict):
        for k, v in (src or {}).items():
            if k in META_SKIP:
                continue
            # Drop Nones/NaNs so we don't clobber core defaults
            if v is None:
                continue
            if isinstance(v, float) and math.isnan(v):
                continue
            ov[k] = v

    _merge_clean(env)
    _merge_clean(strat)

    # ----- Singletons for sweeps (keep core code paths intact)
    if "RENT" in strat:
        try:
            ov["RENT_SCENARIOS"] = np.array([float(strat["RENT"])], dtype=float)
        except Exception:
            pass
    if "OWNER_DRAW" in strat:
        try:
            ov["OWNER_DRAW_SCENARIOS"] = [float(strat["OWNER_DRAW"])]
        except Exception:
            pass
    # ----- Alias UI keys to simulator keys -----
    if "WOM_RATE" in ov and "WOM_Q" not in ov:
        ov["WOM_Q"] = float(ov.pop("WOM_RATE"))
    

    # ----- Scenario configs: grant + capex timing
    gm = env.get("grant_month", None)
    # support UI convention: -1 means "no grant"
    if isinstance(gm, (int, np.integer)) and gm < 0:
        gm = None
    sc_name = env.get("name", "Scenario")
    capex_timing = "all" if ("all_upfront" in str(strat.get("name", ""))) else "staged"
    ov["SCENARIO_CONFIGS"] = [{
        "name": sc_name,
        "capex_timing": capex_timing,
        "grant_amount": float(env.get("grant_amount", 0.0) or 0.0),
        "grant_month": gm,
    }]

    # ----- Capacity mapping (UI → simulator)
    # UI contract: MEMBER_CAP > 0 => enforce hard cap; 0/None => no override (use internal station bottlenecks)
    cap_raw = env.get("MEMBER_CAP", None)
    try:
        cap_val = int(cap_raw) if cap_raw is not None else None
    except (TypeError, ValueError):
        cap_val = None

    if cap_val is not None and cap_val > 0:
        ov["MAX_MEMBERS"] = cap_val
    else:
        ov.pop("MAX_MEMBERS", None)  # ensure we don't accidentally pin capacity

    # Don't leak UI-only field
    ov.pop("MEMBER_CAP", None)

    # ----- Expansion threshold (optional)
    thr_raw = env.get("EXPANSION_THRESHOLD", None)
    
    try:
        thr_val = int(thr_raw) if thr_raw is not None else None
    except (TypeError, ValueError):
        thr_val = None

    if thr_val is not None and thr_val >= 0:
        ov["EXPANSION_THRESHOLD"] = thr_val
    else:
        ov.pop("EXPANSION_THRESHOLD", None)

    return ov

def _normalize_env(env: dict) -> dict:
    env = dict(env)

    # normalize grant month
    gm = env.get("grant_month", None)
    if isinstance(gm, (int, np.integer)) and gm < 0:
        env["grant_month"] = None

    # normalize market inflow (dict of pools → ints)
    if isinstance(env.get("MARKET_POOLS_INFLOW"), dict):
        env["MARKET_POOLS_INFLOW"] = _normalize_market_inflow(env["MARKET_POOLS_INFLOW"])

    # ensure numeric types for macro knobs if present
    for k in ["DOWNTURN_PROB_PER_MONTH", "WOM_Q", "AWARENESS_RAMP_MONTHS"]:
        if k in env and env[k] is not None:
            env[k] = float(env[k])

    return env

def _help_text(spec: dict) -> str:
    """Build hover tooltip from spec['desc'] and spec['rec']."""
    if not spec:
        return ""
    desc = spec.get("desc")
    rec  = spec.get("rec")
    parts = []
    if desc:
        parts.append(str(desc))
    if rec and isinstance(rec, (list, tuple)) and len(rec) == 2:
        parts.append(f"Recommended range: {rec[0]}–{rec[1]}")
    return "  \n".join(parts)  # newlines become multi-line tooltips


def _hint_if_out_of_rec(val, spec):
    """Show a small caption if current value is outside recommended range."""
    try:
        show = st.session_state.get("_show_hints", True)
        if not show or not spec:
            return
        rec = spec.get("rec")
        if not (isinstance(rec, (list, tuple)) and len(rec) == 2):
            return
        lo, hi = float(rec[0]), float(rec[1])
        fv = float(val)
        if fv < lo or fv > hi:
            st.caption(f"⚠️ Outside the typical range ({lo}–{hi}). Sanity-check this choice.")
    except Exception:
        pass

def render_param_controls(title: str, params: dict, *, group_keys: Optional[List[str]] = None, prefix: str = "") -> dict:
    """
    Render Streamlit inputs for keys in `params` or explicit `group_keys`.
    If a key is missing or is None, use a sensible default from PARAM_SPECS.
    """
    def _default_from_spec(spec, key=None):
        t = spec.get("type") if spec else None
        # special sentinel: grant_month uses -1 to mean "None"
        if key == "grant_month":
            return -1
        if t == "bool":  return False
        if t == "int":   return int(spec.get("min", 0))
        if t == "float": return float(spec.get("min", 0.0))
        if t == "market_inflow":
            return {"community_studio": 0, "home_studio": 0, "no_access": 0}
        if t == "json":  return {}
        return ""

    def _clamp_num(val, lo, hi):
        try:
            return min(max(val, lo), hi)
        except Exception:
            return lo

    out = dict(params)
    st.markdown(f"**{title}**")

    keys = group_keys if group_keys else list(params.keys())
    for k in keys:
        spec = PARAM_SPECS.get(k)
        label = spec["label"] if spec and "label" in spec else k
        wid_key = f"{prefix}_{k}"

        # get current value; if missing or None, choose a default
        v = params.get(k, None)
        state_key = f"{prefix}_{k}"
        if state_key in st.session_state:
            v = st.session_state[state_key]
        else:
            v = params.get(k, _default_from_spec(spec, key=k))
            
        if v is None:
            v = _default_from_spec(spec, key=k)

        t = spec["type"] if spec else None

        if t == "bool":
            out[k] = st.checkbox(label, value=bool(v), key=wid_key, help=_help_text(spec))

        elif t == "int":
            lo = int(spec.get("min", 0)); hi = int(spec.get("max", 100))
            step = int(spec.get("step", 1)) or 1
            try:
                v_int = int(v) if v is not None else lo
            except Exception:
                v_int = lo
            v_int = _clamp_num(v_int, lo, hi)
        
            try:
                out[k] = int(st.slider(
                    label, min_value=lo, max_value=hi, step=step, value=int(v_int),
                    key=wid_key, help=_help_text(spec)
                ))
                _hint_if_out_of_rec(out[k], spec)
            except Exception:
                # use a distinct key to avoid duplicate registration
                out[k] = int(st.number_input(
                    label, min_value=lo, max_value=hi, step=step, value=int(v_int),
                    key=f"{wid_key}__ni", help=_help_text(spec)
                ))
                _hint_if_out_of_rec(out[k], spec)
        
        elif t == "float":
            lo = float(spec.get("min", 0.0)); hi = float(spec.get("max", 1.0))
            step = float(spec.get("step", 0.01)) or 0.01
            try:
                v_f = float(v) if v is not None else lo
            except Exception:
                v_f = lo
            v_f = _clamp_num(v_f, lo, hi)
        
            try:
                out[k] = float(st.slider(
                    label, min_value=lo, max_value=hi, step=step, value=float(v_f),
                    key=wid_key, help=_help_text(spec)
                ))
                _hint_if_out_of_rec(out[k], spec)
            except Exception:
                out[k] = float(st.number_input(
                    label, min_value=lo, max_value=hi, step=step, value=float(v_f),
                    key=f"{wid_key}__ni", help=_help_text(spec)
                ))
                _hint_if_out_of_rec(out[k], spec)

        elif t == "market_inflow":
            base = f"{wid_key}"
            cur = _normalize_market_inflow(v if isinstance(v, dict) else {})
            c_def = st.session_state.get(f"{base}_c", cur["community_studio"])
            h_def = st.session_state.get(f"{base}_h", cur["home_studio"])
            n_def = st.session_state.get(f"{base}_n", cur["no_access"])
        
            help_txt = _help_text(spec)
            c = st.slider("Community studio inflow", 0, 50, int(c_def), key=f"{base}_c", help=help_txt)
            h = st.slider("Home studio inflow",      0, 50, int(h_def), key=f"{base}_h", help=help_txt)
            n = st.slider("No access inflow",        0, 50, int(n_def), key=f"{base}_n", help=help_txt)
        
            out[k] = {"community_studio": c, "home_studio": h, "no_access": n}
            _hint_if_out_of_rec(c, spec); _hint_if_out_of_rec(h, spec); _hint_if_out_of_rec(n, spec)
        
            # Keep a synced parent copy too (optional)
            st.session_state[base] = out[k]

    # (no inner expanders; called inside parent expander)
    return out

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
            has_any_title = any(ax.get_title() for ax in fig.get_axes())
            if not has_any_title:
                fig.suptitle(self.title_suffix)

        def _show(*args, **kwargs):
            counter["i"] += 1
            fig = plt.gcf()
        
            _ensure_suffix(fig)  # your existing helper
        
            # --- NEW: de-clash titles and improve spacing ---
            has_suptitle  = bool(fig._suptitle and fig._suptitle.get_text())
            has_ax_titles = any(ax.get_title() for ax in fig.get_axes())
        
            if has_suptitle and has_ax_titles:
                # Give the suptitle its own headroom and shrink it a touch.
                fig._suptitle.set_y(0.98)
                try:
                    fig._suptitle.set_fontsize(max(fig._suptitle.get_fontsize() - 2, 10))
                except Exception:
                    pass
                # Leave extra space at the top for the axis title(s).
                fig.tight_layout(rect=[0, 0, 1, 0.94])
            else:
                fig.tight_layout()
        
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
def run_cell_cached(env: dict, strat: dict, seed: int, cache_key: Optional[str] = None):
    # make sure cache_key participates in the hash even if unused
    if cache_key is None:
        cache_key = _make_cache_key(env, strat, seed)

    # ✅ build overrides BEFORE using ov
    ov = build_overrides(env, strat)
    ov["RANDOM_SEED"] = seed

    title_suffix = f"{env['name']} | {strat['name']}"
    with FigureCapture(title_suffix) as cap:
        res = run_original_once(SCRIPT, ov)

    df_cell, eff = (res if isinstance(res, tuple) else (res, None))

    df_cell = df_cell.copy()
    df_cell["environment"] = env["name"]
    df_cell["strategy"]    = strat["name"]
    if "simulation_id" not in df_cell.columns:
        df_cell["simulation_id"] = 0
    return df_cell, eff, cap.images, cap.manifest


# ---------- column detection & timings (robust) ----------
def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _first_cash_negative(g: pd.DataFrame, month_col: str, cash_col: str) -> float:
    g = g.sort_values(month_col)
    m = g.loc[g[cash_col] < 0, month_col]
    return float(m.iloc[0]) if not m.empty else np.nan

def _first_sustained_ge_zero(g: pd.DataFrame, month_col: str, cf_col: str, k: int = 3) -> float:
    g = g.sort_values(month_col)
    ok = (g[cf_col] >= 0).astype(int).rolling(k, min_periods=k).sum() == k
    m = g.loc[ok.values, month_col]
    return float(m.iloc[0]) if not m.empty else np.nan

def summarize_cell(df: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    
    # columns expected: environment, strategy, simulation_id, month, cash_balance, dscr, (maybe cfads)
    env_col = "environment"
    strat_col = "strategy"
    sim_col = "simulation_id"

    # Month column: allow a few aliases just in case
    month_col = "month" if "month" in df.columns else (
        "Month" if "Month" in df.columns else "t"
    )
    if month_col not in df.columns:
        raise RuntimeError("Could not find a month column among ['month','Month','t'].")

    # pick columns (fall back to Δcash for CF)
    cash_col = pick_col(df, ["cash_balance","cash","ending_cash"])
    cf_col   = pick_col(df, ["cfads","operating_cash_flow","op_cf","net_cash_flow","cash_flow"])

    if cash_col is None:
        raise RuntimeError("cash balance column not found in results.")
    if cf_col is None:
        df = df.sort_values([env_col, strat_col, sim_col, month_col]).copy()
        df["_fallback_cf"] = df.groupby([env_col, strat_col, sim_col])[cash_col].diff().fillna(0.0)
        cf_col = "_fallback_cf"

    # ---- timings (build explicitly to avoid pandas .apply quirks) ----
    def _first_cash_negative(g: pd.DataFrame) -> float:
        s = g.set_index(month_col)[cash_col]
        idx = s.index[s.values < 0]
        return float(idx.min()) if len(idx) else np.nan

    def _first_sustained_ge_zero(g: pd.DataFrame, k: int = 3) -> float:
        s = g.set_index(month_col)[cf_col].sort_index()
        ok = (s >= 0).astype(int).rolling(k, min_periods=k).sum() == k
        idx = ok[ok].index
        return float(idx.min()) if len(idx) else np.nan

    rows = []
    for (env, strat, sim), g in df.sort_values([env_col, strat_col, sim_col, month_col]) \
                                  .groupby([env_col, strat_col, sim_col]):
        rows.append({
            env_col: env,
            strat_col: strat,
            sim_col: sim,
            "t_insolvency": _first_cash_negative(g),
            "t_breakeven":  _first_sustained_ge_zero(g, k=3),
            "min_cash": float(g[cash_col].min()),
        })
    timings = pd.DataFrame(rows, columns=[env_col, strat_col, sim_col, "t_insolvency", "t_breakeven", "min_cash"])
    if timings.empty:
        # ensure columns exist even if no groups
        timings = pd.DataFrame(columns=[env_col, strat_col, sim_col, "t_insolvency", "t_breakeven", "min_cash"])

    # ---- survival via ever-negative cash ----
    surv = (timings.assign(neg=lambda d: d["min_cash"] < 0)
                    .groupby([env_col, strat_col])["neg"].mean()
                    .reset_index(name="prob_insolvent_by_T"))
    surv["survival_prob"] = 1.0 - surv["prob_insolvent_by_T"]

    # ---- end-of-horizon cash & DSCR@M12 (or T) ----
    T = int(df[month_col].max())
    last = df[df[month_col] == T]
    cash_q = (last.groupby([env_col, strat_col])[cash_col]
                  .quantile([0.10, 0.50, 0.90]).unstack().reset_index()
                  .rename(columns={0.10:"cash_q10", 0.50:"cash_med", 0.90:"cash_q90"}))

    m12 = 12 if T >= 12 else T
    # DSCR may be missing in very early months, handle gracefully
    if "dscr" in df.columns:
        dscr_q = (df[df[month_col] == m12]
                    .groupby([env_col, strat_col])["dscr"]
                    .quantile([0.10, 0.50, 0.90]).unstack().reset_index()
                    .rename(columns={0.10:"dscr_q10", 0.50:"dscr_med", 0.90:"dscr_q90"}))
    else:
        dscr_q = pd.DataFrame({env_col: [], strat_col: [], "dscr_q10": [], "dscr_med": [], "dscr_q90": []})

    # ---- timing medians (guard against all-NaN) ----
    def _med_or_nan(s: pd.Series) -> float:
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        return float(s.median()) if len(s) else np.nan

    tim_summary = (timings.groupby([env_col, strat_col]).agg(
        median_time_to_insolvency_months=("t_insolvency", _med_or_nan),
        median_time_to_breakeven_months=("t_breakeven", _med_or_nan),
    ).reset_index())

    # ---- assemble one-row matrix for this cell ----
    matrix_row = (surv[[env_col, strat_col, "survival_prob"]]
                    .merge(cash_q, on=[env_col, strat_col], how="left")
                    .merge(dscr_q, on=[env_col, strat_col], how="left")
                    .merge(tim_summary, on=[env_col, strat_col], how="left"))

    # return a dict-like row (first row) and the timings table
    return matrix_row.iloc[0].to_dict(), timings

# ---------- UI ----------
st.set_page_config(page_title="GCWS Simulator", layout="wide")
st.title("Ginkgo Clayworks — Scenario Explorer")

# Presets (match your code)
SCENARIOS = [
    {
        "name": "Baseline",
        "DOWNTURN_PROB_PER_MONTH": 0.05,
        "DOWNTURN_JOIN_MULT": 1.00,
        "DOWNTURN_CHURN_MULT": 1.00,
        "MARKET_POOLS_INFLOW": {"community_studio": 4, "home_studio": 2, "no_access": 3},
        "grant_amount": 0.0, "grant_month": None,

        # Growth-ish levers (sane defaults)
        "WOM_RATE": 0.03,
        "LEAD_TO_JOIN_RATE": 0.20,
        "MAX_ONBOARD_PER_MONTH": 10,

        # Capacity levers
        "MEMBER_CAP": 92,              # soft cap line in your plots is ~92
        "EXPANSION_THRESHOLD": 20,     # tweakable
    },
    {
        "name": "Recession",
        "DOWNTURN_PROB_PER_MONTH": 0.18,
        "DOWNTURN_JOIN_MULT": 0.65,
        "DOWNTURN_CHURN_MULT": 1.50,
        "MARKET_POOLS_INFLOW": {"community_studio": 2, "home_studio": 1, "no_access": 1},
        "grant_amount": 0.0, "grant_month": None,
        "WOM_RATE": 0.02,
        "LEAD_TO_JOIN_RATE": 0.15,
        "MAX_ONBOARD_PER_MONTH": 8,
        "MEMBER_CAP": 86,
        "EXPANSION_THRESHOLD": 25,
    },
    {
        "name": "SlowRecovery_Grant25k_M4",
        "DOWNTURN_PROB_PER_MONTH": 0.10,
        "DOWNTURN_JOIN_MULT": 0.85,
        "DOWNTURN_CHURN_MULT": 1.20,
        "MARKET_POOLS_INFLOW": {"community_studio": 3, "home_studio": 1, "no_access": 2},
        "grant_amount": 25000, "grant_month": 4,
        "WOM_RATE": 0.025,
        "LEAD_TO_JOIN_RATE": 0.18,
        "MAX_ONBOARD_PER_MONTH": 9,
        "MEMBER_CAP": 94,
        "EXPANSION_THRESHOLD": 22,
    },
    {
        "name": "Boom",
        "DOWNTURN_PROB_PER_MONTH": 0.02,
        "DOWNTURN_JOIN_MULT": 1.20,
        "DOWNTURN_CHURN_MULT": 0.85,
        "MARKET_POOLS_INFLOW": {"community_studio": 6, "home_studio": 3, "no_access": 4},
        "grant_amount": 0.0, "grant_month": None,
        "WOM_RATE": 0.04,
        "LEAD_TO_JOIN_RATE": 0.25,
        "MAX_ONBOARD_PER_MONTH": 12,
        "MEMBER_CAP": 100,
        "EXPANSION_THRESHOLD": 18,
    },
    {
        "name": "GreatDepression",
        "DOWNTURN_PROB_PER_MONTH": 0.28,
        "DOWNTURN_JOIN_MULT": 0.45,
        "DOWNTURN_CHURN_MULT": 1.90,
        "MARKET_POOLS_INFLOW": {"community_studio": 1, "home_studio": 0, "no_access": 1},
        "grant_amount": 0.0, "grant_month": None,
        "WOM_RATE": 0.01,
        "LEAD_TO_JOIN_RATE": 0.10,
        "MAX_ONBOARD_PER_MONTH": 6,

        "MEMBER_CAP": 81,
        "EXPANSION_THRESHOLD": 30,
    },
]
STRATEGIES = [
    {"name":"I_all_upfront_Base", "RENT":3500, "OWNER_DRAW":1000},
    {"name":"II_staged_Base",     "RENT":3500, "OWNER_DRAW":1000},
]


# Sidebar controls
with st.sidebar:
    st.header("Configuration")
    st.caption("Hover over any label for a short explanation.")
    st.session_state["_show_hints"] = st.toggle(
        "Show hints", value=True,
        help="If on, the app shows a small note when a value is outside its typical range."
    )

    scen_names  = [s["name"] for s in SCENARIOS]
    strat_names = [s["name"] for s in STRATEGIES]

    scen_sel  = st.selectbox("Scenario preset", scen_names, index=0)
    strat_sel = st.selectbox("Strategy preset", strat_names, index=0)
    seed      = 42

    # After scen_sel / strat_sel are created:
    if "last_scen_sel" not in st.session_state:
        st.session_state["last_scen_sel"] = scen_sel
    if "last_strat_sel" not in st.session_state:
        st.session_state["last_strat_sel"] = strat_sel
    
    # Deep copies of the chosen presets
    env  = json.loads(json.dumps(next(s for s in SCENARIOS  if s["name"] == scen_sel)))
    strat = json.loads(json.dumps(next(s for s in STRATEGIES if s["name"] == strat_sel)))
    
    # If preset changed, push values into widgets for ALL groups you render
    if scen_sel != st.session_state["last_scen_sel"]:
        _push_preset_to_widgets(env,   prefix="env_income", keys=GROUPS["Income_env"])
        _push_preset_to_widgets(env,   prefix="env_exp",    keys=GROUPS["Expenses_env"])
        _push_preset_to_widgets(env,   prefix="env_macro",  keys=GROUPS["Macro_env"])
        _push_preset_to_widgets(env,   prefix="env_cap",    keys=GROUPS["Capacity_env"])
        st.session_state["last_scen_sel"] = scen_sel
    
    if strat_sel != st.session_state["last_strat_sel"]:
        _push_preset_to_widgets(strat, prefix="strat_income", keys=GROUPS["Income_strat"])
        _push_preset_to_widgets(strat, prefix="strat_exp",    keys=GROUPS["Expenses_strat"])
        # (no strat_macro — you removed that panel)
        st.session_state["last_strat_sel"] = strat_sel
        
    # render all known fields dynamically
    # --- Grouped controls ---
    with st.expander("Income", expanded=True):
        env_income = render_param_controls(
            "Income — Scenario (market/capacity inputs)",
            _subset(env, GROUPS["Income_env"]),
            group_keys=GROUPS["Income_env"], prefix="env_income"
        )
        strat_income = render_param_controls(
            "Income — Strategy (pricing, classes)",
            _subset(strat, GROUPS["Income_strat"]),
            group_keys=GROUPS["Income_strat"], prefix="strat_income"
        )
    
    with st.expander("Expenses", expanded=True):
        env_exp = render_param_controls(
            "Expenses — Scenario",
            _subset(env, GROUPS["Expenses_env"]),
            group_keys=GROUPS["Expenses_env"], prefix="env_exp"
        )
        strat_exp = render_param_controls(
            "Expenses — Strategy",
            _subset(strat, GROUPS["Expenses_strat"]),
            group_keys=GROUPS["Expenses_strat"], prefix="strat_exp"
        )
    
    with st.expander("Macro", expanded=True):
        env_macro = render_param_controls(
            "Macro — Scenario",
            _subset(env, GROUPS["Macro_env"]),
            group_keys=GROUPS["Macro_env"], prefix="env_macro"
        )
        # remove the Macro — Strategy block (there are no strategy-owned macro keys)
    
    with st.expander("Capacity (scenario)", expanded=True):
        env_cap = render_param_controls(
            "Capacity — Scenario",
            _subset(env, GROUPS["Capacity_env"]),
            group_keys=GROUPS["Capacity_env"], prefix="env_cap"
        )
    
    # Merge edits back
    for part in (env_income, env_exp, env_macro, env_cap):
        _update_from(part, env, part.keys())
    for part in (strat_income, strat_exp):
        _update_from(part, strat, part.keys())
    
    # --- Events (discrete) lives with Income ---
    with st.expander("Events (discrete)", expanded=True):
        events_fixed_ui   = st.selectbox("Events per month (fixed)", [0, 1, 2, 3, 4], index=0)
        ticket_choice_ui  = st.selectbox("Event ticket price ($)",  [50, 75, 100, 125], index=2)
    
    # Map discrete choices directly (0 == disabled)
    strat["BASE_EVENTS_PER_MONTH_LAMBDA"] = float(events_fixed_ui)
    strat["EVENTS_MAX_PER_MONTH"]         = int(events_fixed_ui)
    strat["TICKET_PRICE"]                 = int(ticket_choice_ui)

    # Preset save/load
    st.markdown("---")
    colA, colB = st.columns(2)
    if colA.button("Save preset (download)"):
        blob = json.dumps({"env": env, "strat": strat}, indent=2).encode("utf-8")
        st.download_button("Download JSON", data=blob, file_name="gcws_preset.json", mime="application/json", key="dl_preset")
    uploaded = colB.file_uploader("Load preset JSON", type=["json"], accept_multiple_files=False)
    if uploaded is not None:
        try:
            preset = json.loads(uploaded.read())
            env.update(preset.get("env", {}))
            strat.update(preset.get("strat", {}))
            st.success("Preset loaded. Scroll up and press Apply in each expander if needed.")
        except Exception as e:
            st.error(f"Invalid preset: {e}")

    run_btn = st.button("Run simulation", type="primary")

# Tabs
tab_run, tab_matrix = st.tabs(["Single run", "Matrix heatmaps"])

# ---- Single run
with tab_run:
    if run_btn:
        with st.spinner("Running simulator…"):
            env_norm = _normalize_env(env)
            cache_key = _make_cache_key(env_norm, strat, seed)
            df_cell, eff, images, manifest = run_cell_cached(env_norm, strat, seed, cache_key)
            
        st.subheader(f"KPIs — {env['name']} | {strat['name']}")

        # Core cash/dscr from cell
        kpi_cell = compute_kpis_from_cell(df_cell)
        
        # Timing/breakeven from summarize_cell
        row_dict, _tim = summarize_cell(df_cell)
        
        surv        = kpi_cell.get("survival_prob", np.nan)
        cash_q10    = kpi_cell.get("cash_q10", np.nan)
        cash_med    = kpi_cell.get("cash_med", np.nan)
        dscr_med    = kpi_cell.get("dscr_med", np.nan)
        t_breakeven = row_dict.get("median_time_to_breakeven_months", np.nan)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Survival prob @ horizon", f"{surv:.2f}" if np.isfinite(surv) else "NA")
        col2.metric("Cash p10 → p50 ($k)",
                    f"{(cash_q10/1e3):,.0f} → {(cash_med/1e3):,.0f}"
                    if np.isfinite(cash_q10) and np.isfinite(cash_med) else "NA")
        col3.metric("DSCR @ M12 (p50)", f"{dscr_med:.2f}" if np.isfinite(dscr_med) else "NA")
        col4.metric("Breakeven (median, months)",
                    f"{t_breakeven:.0f}" if np.isfinite(t_breakeven) else "NA")

        st.markdown("#### Captured charts")
        for fname, data in images:
            st.image(data, caption=fname, use_container_width=True)

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
            rows = []
            for i, E in enumerate(SCENARIOS):
                for j, S in enumerate(STRATEGIES):
                    # Copy the preset and apply the same discrete Events mapping
                    S2 = json.loads(json.dumps(S))  # deep copy
                    efixed = int(events_fixed_ui)   # 0..4 from UI; 0 == disabled
                    S2["BASE_EVENTS_PER_MONTH_LAMBDA"] = float(efixed)
                    S2["EVENTS_MAX_PER_MONTH"]         = int(efixed)
                    S2["TICKET_PRICE"]                 = int(ticket_choice_ui)
            
                    E_norm = _normalize_env(E)
                    seed_ = 42 + 1000*(i*len(STRATEGIES)+j)
                    df_cell, eff, _imgs, _man = run_cell_cached(E_norm, S2, seed_, _make_cache_key(E_norm, S2, seed_))
                    row_dict, _ = summarize_cell(df_cell)
                    row_dict["environment"] = E["name"]
                    row_dict["strategy"] = S2["name"]
                    rows.append(row_dict)
            matrix = pd.DataFrame(rows)

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


