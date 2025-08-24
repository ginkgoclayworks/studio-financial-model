#!/usr/bin/env python
# coding: utf-8




# In[11]:


"""
Ceramics Studio Simulation — Segmented Ramp + WOM + Tenure Churn + Capacity (+ DSCR & Min-Cash)
===============================================================================================

OVERVIEW
- Monthly Monte Carlo sim of a members-only pottery studio.
- Joins come from 3 market pools (community-studio, home-studio, no-access), plus referrals.
- Joins are shaped by seasonality, awareness ramp, word-of-mouth (WOM), price elasticity, capacity damping, and downturns.
- Churn is tenure-based (higher early, lower later) with optional uplift when the studio is crowded.
- We track revenues (membership, clay, firing, events, workshops), costs (fixed/variable), cash, loan, grants.

KEY DIALS TO TUNE
- Acquisition: MARKET_POOLS, POOL_BASE_INTENT, AWARENESS_*, WOM_Q, WOM_SATURATION, REFERRAL_RATE_PER_MEMBER, REFERRAL_CONV,
  PRICE, PRICE_ELASTICITY, DOWNTURN_*.
- Switching windows: CLASS_TERM_MONTHS, CS_UNLOCK_FRACTION_PER_TERM.
- Capacity feel: STATIONS[*]{capacity, alpha, kappa}, CAPACITY_DAMPING_BETA, OPEN_HOURS_PER_WEEK.
- Retention: ARCHETYPE_MONTHLY_CHURN, tenure multipliers (early, steady, late), UTILIZATION_CHURN_UPLIFT.
- Cash survivability: RENT_SCENARIOS, OWNER_DRAW_SCENARIOS, RUNWAY_MONTHS, LOAN_*, grants & timing.

MONTHLY FLOW (t = 0..T-1)
1) Update eligible switchers (community-studio):
   if t>0 and t % CLASS_TERM_MONTHS == 0:
       unlock = floor(P_CS_remaining * CS_UNLOCK_FRACTION_PER_TERM)
       cs_eligible += unlock; P_CS_remaining -= unlock

2) Build common join multiplier:
   F_t = S_t * J_t * A_t * W_t * D_t * ε_t * M_price
   where:
     S_t  = seasonality[t % 12]
     J_t  = DOWNTURN_JOIN_MULT if downturn else 1
     A_t  = linear ramp from AWARENESS_RAMP_START_MULT → AWARENESS_RAMP_END_MULT over AWARENESS_RAMP_MONTHS
     W_t  = 1 + WOM_Q * (N_t / WOM_SATURATION)
     D_t  = max(0, 1 - (N_t / SoftCap) ** CAPACITY_DAMPING_BETA)
     ε_t  ~ LogNormal(mean=-(σ^2)/2, sigma=ADOPTION_SIGMA)  # mean-one noise


3) Organic joins (binomial by pool):
   p_pool = BASE_INTENT_pool * F_t
   J_no_access ~ Binomial(P_no_access_remaining, p_no_access)
   J_home      ~ Binomial(P_home_remaining, p_home)
   J_cs        ~ Binomial(cs_eligible,       p_cs)
   Decrement each pool by its joins.

4) Referrals (small K-factor):
   J_ref ~ Poisson(REFERRAL_RATE_PER_MEMBER * N_t * REFERRAL_CONV)
   Cap by remaining supply; allocate (no-access → cs_eligible → home); decrement those pools.

5) Onboarding cap:
   J_tot = J_no_access + J_home + J_cs + J_ref
   If J_tot > MAX_ONBOARDINGS_PER_MONTH:
       Roll back overflow proportionally across sources and return people to the correct pools.

6) Create members:
   For each join, draw archetype by MEMBER_ARCHETYPES[*]["prob"]; store start_month, monthly_fee, clay_bags.

7) Tenure-based churn (hazard per member):
   base = ARCHETYPE_MONTHLY_CHURN[arch]
   h(tenure) ≈ 1.8*base (months ≤2), base (3–6), 0.7*base (≥7)
   p_leave = h * (DOWNTURN_CHURN_MULT if downturn else 1) * (1 + UTILIZATION_CHURN_UPLIFT * max(0, N_t/SoftCap - 1))
   Keep member if rand() > p_leave.

8) Ops & cash:
   Revenues = membership fees + clay + firing (stepped fee) + events (Poisson with seasonality) + workshops (probabilistic, net).
   Variable costs = clay COGS + water + electricity (firings_this_month(N_t), kiln #2 if on).
   Fixed OpEx = rent + insurance + glaze + seasonal heating.
   Operating profit = Revenues - (Fixed + Variable).
   Cash OpEx = Fixed + Variable + loan_payment + owner_draw (optionally tapered).
   Cash_t+1 = Cash_t + (Revenues - Cash OpEx); add grants on grant_month.
   Stage-II CapEx triggers when N_t ≥ EXPANSION_TRIGGER_MEMBERS (staged scenario).

CAPACITY (soft cap)
SoftCap = min over stations s of:
  (alpha_s * capacity_s * OPEN_HOURS_PER_WEEK) /
  (kappa_s * sum_arch(prob_arch * sessions_per_week_arch * hours_arch * usage_share_arch,s))

FINANCE
Loan principal at t=0 includes CapEx (depending on scenario), contingency, and runway months of burn (rent + OpEx + owner draw).
Monthly loan payment = standard amortization with LOAN_ANNUAL_RATE and LOAN_TERM_YEARS.
We also compute **monthly DSCR** ≈ Operating Profit / Debt Service.

OUTPUTS (per scenario × rent × draw, aggregated over simulations)
- Membership trajectory (median + 10–90%).
- Cash balance bands with grant markers.
- Median months to operating break-even (cumulative op profit ≥ 0).
- % insolvent before grant.
- Median final cash at month T.
- **Median minimum cash** across the horizon (stress indicator).
- **DSCR** summaries.
- Diagnostics: net adds distribution; optional joins-by-source vs departures plot.
"""


# In[19]:

from __future__ import annotations
from typing import Optional  # put this at the top of your script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from numpy.random import default_rng, SeedSequence
# near the top

import matplotlib
matplotlib.use("Agg")  # before importing pyplot
import json
from pathlib import Path
from datetime import datetime
import matplotlib as mpl
from collections import OrderedDict

from types import SimpleNamespace
from contextlib import contextmanager
import copy
import inspect

mpl.rcParams['font.family'] = 'Noto Sans'  # or another installed font with U+2011

_MISSING = object()

def _is_constant_name(name: str) -> bool:
    """Heuristic: ALL_CAPS names are treated as config constants."""
    return name.isupper() and not name.startswith("_")

def get_default_cfg() -> dict:
    """
    Snapshot the current module constants (ALL_CAPS) as defaults.
    This captures the values you’ve already defined in this file.
    """
    g = globals()
    out = {}
    for k, v in list(g.items()):
        if _is_constant_name(k):
            # deepcopy so callers can't mutate module state by reference
            try:
                out[k] = copy.deepcopy(v)
            except Exception:
                out[k] = v
    return out

@contextmanager
def override_globals(new_vals: dict):
    """
    Temporarily override module globals with values from new_vals, then restore.
    This lets the existing code keep using bare names (MONTHS, PRICE, …)
    while making the whole run controlled by a cfg dict.
    """
    old = {}
    g = globals()
    try:
        for k, v in new_vals.items():
            old[k] = g.get(k, _MISSING)
            g[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is _MISSING:
                g.pop(k, None)
            else:
                g[k] = v

def resolve_cfg(user_cfg: dict | None = None) -> dict:
    """
    Merge user overrides onto the current file’s defaults.
    Later, you can also validate types / ranges here if you like.
    """
    defaults = get_default_cfg()
    merged = dict(defaults)
    if user_cfg:
        merged.update(user_cfg)
    return merged

# -- Downturn probability sourcing -------------------------------------------
def _get_downturn_prob(cfg):
    """Return (prob, source). Never raises; clamps to [0,1].
    Precedence: explicit cfg override > live nowcast > fallback 0.15
    """
    # 1) explicit override
    if cfg and "DOWNTURN_PROB_PER_MONTH" in cfg:
        try:
            p = float(cfg["DOWNTURN_PROB_PER_MONTH"])
            return max(0.0, min(1.0, p)), "static(cfg)"
        except Exception:
            pass

    # 2) live nowcast
    try:
        from nowcast import run_nowcast
        nc = run_nowcast(save_csv=False)          # prints the banner once
        p = float(nc.get("prob", 0.15))
        if not (p == p):                           # NaN guard
            p = 0.15
        return max(0.0, min(1.0, p)), "nowcast"
    except Exception as e:
        print(f"[nowcast] error: {e.__class__.__name__}: {e} → using 0.15")
        return 0.15, "fallback(0.15)"

def apply_workshops(stream, cfg, T):
    """Monthly workshops: revenue, variable cost, optional conversion to members."""
    if not bool(cfg.get("WORKSHOPS_ENABLED", False)):
        return

    wpm   = float(cfg.get("WORKSHOPS_PER_MONTH", 0.0))
    avg_n = int(cfg.get("WORKSHOP_AVG_ATTENDANCE", 0))
    fee   = float(cfg.get("WORKSHOP_FEE", 0.0))
    var_c = float(cfg.get("WORKSHOP_COST_PER_EVENT", 0.0))
    conv  = float(cfg.get("WORKSHOP_CONV_RATE", 0.0))
    lag   = int(cfg.get("WORKSHOP_CONV_LAG_MO", 1))

    # Expected counts per month (you can add Poisson noise if desired)
    events_pm = wpm
    attendees_pm = int(round(events_pm * avg_n))
    gross_rev_pm = attendees_pm * fee
    var_cost_pm = events_pm * var_c
    net_rev_pm = gross_rev_pm - var_cost_pm
    conv_joins_pm = int(round(attendees_pm * conv))

    for t in range(T):
        # revenue
        stream["workshop_revenue"][t] += net_rev_pm
        # conversions (bounded by horizon)
        conv_t = min(T-1, t + lag)
        stream["joins_from_workshops"][conv_t] += conv_joins_pm

# =============================================================================
# Tunable Parameters
# =============================================================================

# -------------------------------------------------------------------------
# Simulation Settings
# -------------------------------------------------------------------------
MONTHS = 60
N_SIMULATIONS = 100
RANDOM_SEED = 42

# -------------------------------------------------------------------------
# Financing & Loans
# -------------------------------------------------------------------------
RUNWAY_MONTHS = 12
LOAN_CONTINGENCY_PCT = 0.08
EXTRA_BUFFER = 0.0  # cushion for sensitivity / sizing sweeps

# --- Aggregate loan (legacy / fallback) ---
LOAN_ANNUAL_RATE = 0.08
LOAN_TERM_YEARS = 5

# --- Split loans ---
LOAN_504_ANNUAL_RATE = 0.070   # effective blended proxy for SBA 504 (equipment)
LOAN_504_TERM_YEARS  = 5
LOAN_7A_ANNUAL_RATE  = 0.115   # SBA 7(a), 5-year term mid-case
LOAN_7A_TERM_YEARS   = 5

DSCR_CASH_TARGET = 1.25  # put with other constants
                        
# Owner draw taper
OWNER_DRAW_START_MONTH = 1
OWNER_DRAW_END_MONTH   = 12  # set None for indefinite
# Owner stipend controls
OWNER_STIPEND_MONTHS = 12  # how many months of stipend to reserve & actually pay
# (OPTIONAL) If you want calendar limits too, you can keep using OWNER_DRAW_START_MONTH/END
# and we’ll make END default to OWNER_STIPEND_MONTHS unless you override it.

# Staffing expansion trigger
STAFF_EXPANSION_THRESHOLD = 50
STAFF_COST_PER_MONTH = 2500.0

# -------------------------------------------------------------------------
# Community-studio switch windows & onboarding cap (were undefined)
# -------------------------------------------------------------------------
CLASS_TERM_MONTHS = 3                # e.g., 12-week terms → switch window every 3 months
CS_UNLOCK_FRACTION_PER_TERM = 0.25   # share of remaining CS pool that becomes eligible each window
MAX_ONBOARDINGS_PER_MONTH = None     # set an int (e.g., 6/10/12) to hard-cap monthly new members


# -------------------------------------------------------------------------
# Rent & Draw Scenarios
# -------------------------------------------------------------------------
RENT_SCENARIOS = np.arange(2500, 5001, 1000)
OWNER_DRAW_SCENARIOS = [0, 1000, 2000, 3000]

# -------------------------------------------------------------------------
# Capital Expenditures (triangular distributions)
# -------------------------------------------------------------------------
STAGE_I_CAPEX = {
    'kiln_1': (3500, 3800, 4200),
    'wheels_1_to_4': (900, 1200, 1500),
    'handbuilding_tables': (800, 1000, 1200),
    'slab_roller': (1000, 1200, 1400),
    'wire_racks_initial': (130, 140, 160),
    'glaze_table': (400, 500, 600),
    'clay_storage': (250, 300, 375),
    'industrial_sink_and_trap': (900, 1000, 1100),
    'ventilation': (1300, 1500, 1700),
    'electrical_setup': (2500, 3000, 3500),
    'misc_tools': (800, 1000, 1200),
}
STAGE_II_CAPEX = {
    'kiln_2': (6000, 6500, 7000),
    'wheels_5_to_8': (900, 1200, 1500),
    'wire_racks_expansion': (130, 140, 160),
    'spray_booth': (2000, 2500, 3000),
    'pugmill': (3900, 5000, 6500),
    'photo_booth': (250, 300, 350),
}

# -------------------------------------------------------------------------
# Operating Expenses (recurring)
# -------------------------------------------------------------------------
INSURANCE_COST = 75
GLAZE_COST_PER_MONTH = 833.33
HEATING_COST_WINTER = 450
HEATING_COST_SUMMER = 30

# Utilities
COST_PER_KWH = 0.2182
KWH_PER_FIRING_KMT1027 = 75
KWH_PER_FIRING_KMT1427 = 110
WATER_COST_PER_GALLON = 0.02
GALLONS_PER_BAG_CLAY = 1
WHOLESALE_CLAY_COST_PER_BAG = 16.75

# -------------------------------------------------------------------------
# Electricity & Kiln Scheduling
# -------------------------------------------------------------------------
DYNAMIC_FIRINGS = True
BASE_FIRINGS_PER_MONTH = 10
REFERENCE_MEMBERS_FOR_BASE_FIRINGS = 12
MIN_FIRINGS_PER_MONTH = 4
MAX_FIRINGS_PER_MONTH = 12
KILN2_AFTER_EXPANSION_SCENARIO_II = True
KILN2_AFTER_THRESHOLD_SCENARIO_I = True



# -------------------------------------------------------------------------
# --- Beginner Classes (opt-in offering) ---
# -------------------------------------------------------------------------

CLASSES_ENABLED = False
CLASS_COHORTS_PER_MONTH = 2
CLASS_CAP_PER_COHORT = 10
CLASS_FILL_MEAN = 0.85          # avg fill percentage (0–1)
CLASS_PRICE = 600             # per student
CLASS_COST_PER_STUDENT = 40.0   # materials/admin
CLASS_INSTR_RATE_PER_HR = 30.0
CLASS_HOURS_PER_COHORT = 3.0
CLASS_CONV_RATE = 0.12          # fraction converting to members
CLASS_CONV_LAG_MO = 1           # months after class end
CLASS_EARLY_CHURN_MULT = 0.8    # first 3–6 months lower churn for converts


# -------------------------------------------------------------------------
# Revenue: Memberships
# -------------------------------------------------------------------------
MEMBER_ARCHETYPES = {
    "Hobbyist":          {"prob": 0.35, "monthly_fee": 175, "clay_bags": (0.25, 0.5, 1)},
    "Committed Artist":  {"prob": 0.40, "monthly_fee": 185, "clay_bags": (1, 1.5, 2)},
    "Production Potter": {"prob": 0.10, "monthly_fee": 200, "clay_bags": (2, 2.5, 3)},
    "Seasonal User":     {"prob": 0.15, "monthly_fee": 150, "clay_bags": (0.25, 0.5, 1)},
}

# -------------------------------------------------------------------------
# Revenue: Events (paint-a-pot / sip-&-paint)
# -------------------------------------------------------------------------
EVENTS_ENABLED = True
EVENTS_MAX_PER_MONTH = 4
EVENT_MUG_COST_RANGE = (4.50, 7.50)   # Bisque Imports stoneware mugs
EVENT_CONSUMABLES_PER_PERSON = 2.50   # glaze, brushes, wipes, packaging
EVENT_STAFF_RATE_PER_HOUR = 22.0       # set >0 to include staff costs
EVENT_HOURS_PER_EVENT = 2.0
ATTENDEES_PER_EVENT_RANGE = [8, 10, 12]
TICKET_PRICE = 75

# Seasonality
SEASONALITY_WEIGHTS = np.array([1.1, 1.2, 1.3, 1.4, 1.3, 0.9, 0.8, 0.85, 1.3, 1.4, 1.2, 1.0])
NORMALIZE_SEASONALITY = True
BASE_EVENTS_PER_MONTH_LAMBDA = 3
SEASONALITY_WEIGHTS_NORM = (
    SEASONALITY_WEIGHTS / SEASONALITY_WEIGHTS.mean()
    if NORMALIZE_SEASONALITY else SEASONALITY_WEIGHTS
)

# -------------------------------------------------------------------------
# Revenue: Add-ons
# -------------------------------------------------------------------------
RETAIL_CLAY_PRICE_PER_BAG = 25
EXPANSION_TRIGGER_MEMBERS = 20

# Designated Studios
DESIGNATED_STUDIO_COUNT = 2
DESIGNATED_STUDIO_PRICE = 300.0
DESIGNATED_STUDIO_BASE_OCCUPANCY = 0.3

# -------------------------------------------------------------------------
# Membership Dynamics (churn, adoption, capacity)
# -------------------------------------------------------------------------
ARCHETYPE_MONTHLY_CHURN = {
    "Hobbyist":          0.049 * 0.95,
    "Committed Artist":  0.049 * 0.80,
    "Production Potter": 0.049 * 0.65,
    "Seasonal User":     0.049 * 1.90,
}
MIN_STAY = 1
MAX_STAY = 48

# Downturn regime
DOWNTURN_JOIN_MULT  = 0.65
DOWNTURN_CHURN_MULT = 1.50

# Market pools
MARKET_POOLS = {"community_studio": 70, "home_studio": 50, "no_access": 20}
MARKET_POOLS_INFLOW = {"community_studio": 4, "home_studio": 2, "no_access": 3}
POOL_BASE_INTENT = {"community_studio": 0.10, "home_studio": 0.010, "no_access": 0.040}

# Word-of-mouth
WOM_Q = 0.60
WOM_SATURATION = 60
ADOPTION_SIGMA = 0.20
AWARENESS_RAMP_MONTHS = 4
AWARENESS_RAMP_START_MULT = 0.5
AWARENESS_RAMP_END_MULT = 1.0

# Capacity & utilization
OPEN_HOURS_PER_WEEK = 16 * 7  # 112 hours
STATIONS = {
    "wheels":       {"capacity": 8, "alpha": 0.80, "kappa": 2},
    "handbuilding": {"capacity": 6, "alpha": 0.50, "kappa": 3.0},
    "glaze":        {"capacity": 6, "alpha": 0.55, "kappa": 2.6},
}
SESSIONS_PER_WEEK = {"Hobbyist": 1.0, "Committed Artist": 1.5,
                     "Production Potter": 3.5, "Seasonal User": 0.75}
SESSION_HOURS = {"Hobbyist": 1.7, "Committed Artist": 2.75,
                 "Production Potter": 3.8, "Seasonal User": 2.0}
USAGE_SHARE = {
    "Hobbyist": {"wheels": 0.50, "handbuilding": 0.35, "glaze": 0.15},
    "Committed Artist": {"wheels": 0.45, "handbuilding": 0.35, "glaze": 0.20},
    "Production Potter": {"wheels": 0.60, "handbuilding": 0.25, "glaze": 0.15},
    "Seasonal User": {"wheels": 0.40, "handbuilding": 0.45, "glaze": 0.15},
}
CAPACITY_DAMPING_BETA = 4

# -------------------------------------------------------------------------
# Pricing & Referrals
# -------------------------------------------------------------------------
PRICE = 175
JOIN_PRICE_ELASTICITY = -0.6
CHURN_PRICE_ELASTICITY = 0.3
BASELINE_JOIN_RATE = 0.013
REFERRAL_RATE_PER_MEMBER = 0.06
REFERRAL_CONV = 0.22
MAX_MEMBERS = 77
UTILIZATION_CHURN_UPLIFT = 0.25

# -------------------------------------------------------------------------
# Taxation & Entity Setup (Massachusetts)
# -------------------------------------------------------------------------
ENTITY_TYPE = "sole_prop"

# Individual / SE
MA_PERSONAL_INCOME_TAX_RATE = 0.05
SE_EARNINGS_FACTOR = 0.9235
SE_SOC_SEC_RATE = 0.124
SE_MEDICARE_RATE = 0.029
SE_SOC_SEC_WAGE_BASE = 168_600

# Payroll (S-corp)
SCORP_OWNER_SALARY_PER_MONTH = 4000.0
EMPLOYEE_PAYROLL_TAX_RATE = 0.0765
EMPLOYER_PAYROLL_TAX_RATE = 0.0765

# Corporate (C-corp)
FED_CORP_TAX_RATE = 0.21
MA_CORP_TAX_RATE  = 0.08

# Sales & property taxes
MA_SALES_TAX_RATE = 0.0625
PERSONAL_PROPERTY_TAX_ANNUAL = 0.0
PERSONAL_PROPERTY_TAX_BILL_MONTH = 3

# Remittance cadence
ESTIMATED_TAX_REMIT_FREQUENCY_MONTHS = 3
SALES_TAX_REMIT_FREQUENCY_MONTHS = 3

# -------------------------------------------------------------------------
# Maintenance & Marketing
# -------------------------------------------------------------------------
MAINTENANCE_BASE_COST = 200.0
MAINTENANCE_RANDOM_STD = 150.0

MARKETING_COST_BASE = 300.0
MARKETING_RAMP_MONTHS = 12
MARKETING_RAMP_MULTIPLIER = 2.0

# -------------------------------------------------------------------------
# Scenarios (with grants)
# -------------------------------------------------------------------------
SCENARIO_CONFIGS = [
    {"name": "I_all_upfront",               "capex_timing": "all",    "grant_amount": 0.0,    "grant_month": None},
    {"name": "II_staged",                   "capex_timing": "staged", "grant_amount": 0.0,    "grant_month": None},
    {"name": "III_all_upfront_grant25k_m4", "capex_timing": "all",    "grant_amount": 25_000, "grant_month": 4},
    {"name": "IV_staged_grant15k_m9",       "capex_timing": "staged", "grant_amount": 15_000, "grant_month": 9},
]


def _to_serializable(x):
    try:
        import numpy as np
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
    except Exception:
        pass
    return x

# ---- EFFECTIVE CONFIG ECHO (place near the end, after overrides/globals are set) ----

def _g(name, default=None):
    return globals().get(name, default)

def _ser(x):
    # ensure numpy / arrays are JSONable
    try:
        return _to_serializable(x)
    except Exception:
        return x

EFFECTIVE_CONFIG = OrderedDict({
    "MONTHS": _g("MONTHS"),
    "N_SIMULATIONS": _g("N_SIMULATIONS"),
    "PRICE": _g("PRICE"),
    "JOIN_PRICE_ELASTICITY": _g("JOIN_PRICE_ELASTICITY"),
    "CHURN_PRICE_ELASTICITY": _g("CHURN_PRICE_ELASTICITY"),
    "RENT_SCENARIOS": _ser(_g("RENT_SCENARIOS")),
    "OWNER_DRAW_SCENARIOS": _ser(_g("OWNER_DRAW_SCENARIOS")),
    # Macro / growth levers
    "DOWNTURN_PROB_PER_MONTH": _g("DOWNTURN_PROB_PER_MONTH"),
    "DOWNTURN_JOIN_MULT": _g("DOWNTURN_JOIN_MULT"),
    "DOWNTURN_CHURN_MULT": _g("DOWNTURN_CHURN_MULT"),
    "MARKET_POOLS_INFLOW": _ser(_g("MARKET_POOLS_INFLOW")),
    "WOM_Q": _g("WOM_Q"),
    "AWARENESS_RAMP_MONTHS": _g("AWARENESS_RAMP_MONTHS"),
    # Capacity / limits (use one naming scheme consistently)
    "HARD_CAP": _g("HARD_CAP"),
    "CAPACITY_SOFT_CAP": _g("CAPACITY_SOFT_CAP"),
})

# Write once, after fully assembled
try:
    out_dir = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S_effective")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "effective_config.json", "w") as f:
        json.dump(EFFECTIVE_CONFIG, f, indent=2, default=_to_serializable)
except Exception:
    pass

print("[EFFECTIVE_CONFIG]", json.dumps(EFFECTIVE_CONFIG, default=_to_serializable))
# =============================================================================
# Helpers
# =============================================================================

def seasonal_churn_mult(month_idx: int) -> float:
    """
    Multiplier for churn probability based on month of year.
    month_idx is 0-based in the sim; convert to human month (1–12).
    """
    m = (month_idx % 12) + 1  # 1..12

    # Example profile — tweak as you see fit:
    if m in (6, 7, 8):       # Jun–Aug: travel/moves
        return 1.25          # +25% churn
    elif m in (1, 2):        # Jan–Feb: post-holiday dropouts
        return 1.15
    elif m in (11, 12):      # Nov–Dec: holidays, gift memberships expiring
        return 1.10
    else:
        return 1.0
    
    
def in_owner_draw_window(month_idx: int) -> bool:
    """
    Returns True if owner draw should be paid in the given 0-based month index.
    Interprets OWNER_DRAW_START_MONTH/OWNER_DRAW_END_MONTH as 1-based, inclusive.
    If OWNER_DRAW_END_MONTH is None, the window is [START, ∞).
    """
    m = month_idx + 1  # convert to 1-based for human-friendly comparison
    start = int(OWNER_DRAW_START_MONTH)
    end = OWNER_DRAW_END_MONTH
    if end is None:
        return m >= start
    return start <= m <= int(end)


def sample_capex(capex_dict, rng):
    return sum(rng.triangular(low, mode, high) for (low, mode, high) in capex_dict.values())

def draw_adopters(remaining_pool, monthly_intent, rng):
    """
    Stochastic adoption draw from a pool with intent rate.
    We use a Binomial draw clipped by remaining_pool for realism.
    """
    intent = np.clip(monthly_intent, 0.0, 1.0)
    return int(rng.binomial(n=int(max(0, remaining_pool)), p=float(intent)))

def calculate_monthly_payment(principal, annual_rate, years):
    if annual_rate == 0:
        return principal / (years * 12)
    r = annual_rate / 12
    n = years * 12
    return principal * (r * (1 + r)**n) / ((1 + r)**n - 1)

def compute_firing_fee(clay_lbs):
    if clay_lbs <= 20: return clay_lbs * 3
    elif clay_lbs <= 40: return 20 * 3 + (clay_lbs - 20) * 4
    else: return 20 * 3 + 20 * 4 + (clay_lbs - 40) * 5

def firings_this_month(n_active_members):
    if not DYNAMIC_FIRINGS:
        return BASE_FIRINGS_PER_MONTH
    raw = BASE_FIRINGS_PER_MONTH * (n_active_members / max(1, REFERENCE_MEMBERS_FOR_BASE_FIRINGS))
    return int(np.clip(round(raw), MIN_FIRINGS_PER_MONTH, MAX_FIRINGS_PER_MONTH))

def compute_membership_soft_cap():
    H = OPEN_HOURS_PER_WEEK
    caps = {}
    for s, cfg in STATIONS.items():
        denom = 0.0
        for arch, arch_cfg in MEMBER_ARCHETYPES.items():
            mix = arch_cfg["prob"]
            s_per_wk = SESSIONS_PER_WEEK[arch]
            dur = SESSION_HOURS[arch]
            share = USAGE_SHARE[arch][s]
            denom += mix * s_per_wk * dur * share
        caps[s] = (cfg["alpha"] * cfg["capacity"] * H) / (cfg["kappa"] * denom)
    return min(caps.values()), caps

def awareness_multiplier(month_idx):
    """Smooth ramp from START to END over AWARENESS_RAMP_MONTHS."""
    if AWARENESS_RAMP_MONTHS <= 0: 
        return AWARENESS_RAMP_END_MULT
    t = min(1.0, month_idx / AWARENESS_RAMP_MONTHS)
    return AWARENESS_RAMP_START_MULT + t * (AWARENESS_RAMP_END_MULT - AWARENESS_RAMP_START_MULT)

def wom_multiplier(current_members):
    """Simple Bass-style imitation term, saturating with membership level."""
    if WOM_SATURATION <= 0:
        return 1.0
    # 1 + q * (adopters / K); bounded >= 1
    return 1.0 + WOM_Q * (current_members / WOM_SATURATION)

def compute_cs_unlock_share(month_idx, remaining_cs_pool):
    """
    Every CLASS_TERM_MONTHS, let a fraction of the *remaining* community-studio pool
    become eligible to switch.
    """
    if CLASS_TERM_MONTHS <= 0 or remaining_cs_pool <= 0:
        return 0
    return (
    int(np.floor(remaining_cs_pool * CS_UNLOCK_FRACTION_PER_TERM))
    if ((month_idx % CLASS_TERM_MONTHS) == 0 and (month_idx > 0))
    else 0
    )

def month_churn_prob(arch, tenure_mo):
    """Piecewise hazard: higher early churn, stickier later."""
    base = ARCHETYPE_MONTHLY_CHURN[arch]
    if tenure_mo <= 2:
        return min(0.99, base * 1.8)   # onboarding risk
    if tenure_mo <= 6:
        return base                    # steady state
    return base * 0.7                  # long-stay sticky

# =============================================================================
# Simulation
# =============================================================================
def _core_simulation_and_reports():
    """
    The original script body goes here, unmodified:
    - MEMBERSHIP_SOFT_CAP, PER_STATION_CAPS = compute_membership_soft_cap()
    - the loops over RENT_SCENARIOS, OWNER_DRAW_SCENARIOS, SCENARIO_CONFIGS…
    - building rows -> results_df
    - plots
    - summary_table / owner_takehome_table
    - exports
    Return anything you want programmatic access to.
    """

    MEMBERSHIP_SOFT_CAP, PER_STATION_CAPS = compute_membership_soft_cap()
    print(f"Soft membership cap (multi-station bottleneck): ~{MEMBERSHIP_SOFT_CAP:.1f} members")
    for s, cap in PER_STATION_CAPS.items():
        print(f"  Station cap via {s:12s}: ~{cap:.1f}")
    
    rows = []
    
    for fixed_rent in RENT_SCENARIOS:
        for owner_draw in OWNER_DRAW_SCENARIOS:
            for scen_cfg in SCENARIO_CONFIGS:
                scen_name = scen_cfg["name"]
                
                # Deterministic, per-path RNG: (seed, rent, draw, scenario_index, sim)
                scen_index = next(i for i, s in enumerate(SCENARIO_CONFIGS) if s["name"] == scen_name)            
                
                for sim in range(N_SIMULATIONS):
                    ss = SeedSequence([RANDOM_SEED, int(fixed_rent), int(owner_draw), int(scen_index), int(sim)])
                    rng = default_rng(ss)
                    # CapEx
                    capex_I_cost = sample_capex(STAGE_I_CAPEX, rng)
                    capex_II_cost = sample_capex(STAGE_II_CAPEX, rng)
    
                    # Runway
                    avg_monthly_heat = (HEATING_COST_WINTER + HEATING_COST_SUMMER) / 2
                    runway_costs = (INSURANCE_COST + GLAZE_COST_PER_MONTH + avg_monthly_heat + fixed_rent + owner_draw) * RUNWAY_MONTHS
    
                    # Loan principal sizing
                   # ----- Split loan sizing: 504 (CapEx) + 7(a) (runway/draw/buffer) -----
                    total_capex_for_loan = (capex_I_cost + capex_II_cost) if ("all_upfront" in scen_name) else capex_I_cost
                    loan_504_principal = total_capex_for_loan * (1 + LOAN_CONTINGENCY_PCT)
                    loan_7a_principal  = runway_costs + EXTRA_BUFFER
    
                    monthly_loan_payment_504 = calculate_monthly_payment(loan_504_principal, LOAN_504_ANNUAL_RATE, LOAN_504_TERM_YEARS)
                    monthly_loan_payment_7a  = calculate_monthly_payment(loan_7a_principal,  LOAN_7A_ANNUAL_RATE,  LOAN_7A_TERM_YEARS)
                    monthly_loan_payment     = monthly_loan_payment_504 + monthly_loan_payment_7a
    
                    loan_principal_total = loan_504_principal + loan_7a_principal
                    sized_runway_costs = runway_costs  # keep for reporting
                    
    
                    # Tracking
                    cash_balance = 0.0
                    cumulative_op_profit = 0.0
                    cumulative_after_capex = 0.0
                    expansion_triggered = False
                    active_members = []
                    
                    # --- Market pool state for this simulation ---
                    remaining_pool = MARKET_POOLS.copy()
    
                    # Community-studio: track an "eligible to switch" sub-pool.
                    cs_eligible = 0
    
                    insolvent_before_grant = False
                    grant_month = scen_cfg["grant_month"]
                    grant_amount = scen_cfg["grant_amount"]
                    
                    # >>> BEGIN classes: per-simulation state
                    pending_class_conversions = {}   # {target_month: count}
                    # >>> END classes
                    
                    # >>> BEGIN workshops: per-simulation state
                    stream = {}
                    stream["workshop_revenue"] = np.zeros(MONTHS)
                    stream["joins_from_workshops"] = np.zeros(MONTHS, dtype=int)
                    # Precompute monthly workshops using UI-configured knobs
                    apply_workshops(stream, globals(), MONTHS)
                    
                    # >>> END workshops
                    
                    # --- Tax/state trackers (reset each simulation) ---
                    se_ss_wage_base_used_ytd = 0.0        # for SE Social Security cap (sole/partnership)
                    se_tax_payable_accum = 0.0            # accrued SE tax (sole/partnership)
                    state_tax_payable_accum = 0.0         # accrued MA personal income tax (pass-through)
                    corp_tax_payable_accum = 0.0          # accrued corporate income tax (C-corp)
                    sales_tax_payable_accum = 0.0         # accrued sales tax to remit
                    tax_payments_this_month = 0.0         # cash paid this month for taxes
    
                    # >>> BEGIN events: capture config-driven knobs once per simulation
                    _g = globals()
                    events_enabled       = bool(_g.get("EVENTS_ENABLED", True))
                    events_max_per_month = int(_g.get("EVENTS_MAX_PER_MONTH", 4))
                    base_lambda          = float(_g.get("BASE_EVENTS_PER_MONTH_LAMBDA", 3.0))
                    ticket_price         = float(_g.get("TICKET_PRICE", 75.0))
                    attendees_range      = list(_g.get("ATTENDEES_PER_EVENT_RANGE", [8, 10, 12]))
                    mug_cost_range       = tuple(_g.get("EVENT_MUG_COST_RANGE", (4.5, 7.5)))
                    consumables_pp       = float(_g.get("EVENT_CONSUMABLES_PER_PERSON", 2.5))
                    staff_rate_hr        = float(_g.get("EVENT_STAFF_RATE_PER_HOUR", 22.0))
                    hours_per_event      = float(_g.get("EVENT_HOURS_PER_EVENT", 2.0))
                    # >>> END events
                    
                    _g = globals()
                    
                    # --- Pricing elasticity setup (fixed baseline) ---
                    price = float(globals().get("PRICE", 165.0))
                    reference_price = float(globals().get("REFERENCE_PRICE", price))  # default to current price if missing
                    join_eps  = float(globals().get("JOIN_PRICE_ELASTICITY", -0.6))  # negative
                    churn_eps = float(globals().get("CHURN_PRICE_ELASTICITY",  0.3))  # positive

                    def _pmult(p, pref, eps):
                        if pref <= 0:
                            return 1.0
                        m = (max(p, 1e-9) / pref) ** eps
                        return float(np.clip(m, 0.25, 4.0))  # safety caps

                    price_mult_joins = _pmult(price, reference_price, join_eps)
                    price_mult_churn = _pmult(price, reference_price, churn_eps)
                    
                    def _pmult(p, pref, eps):
                        if pref <= 0:
                            return 1.0
                        # guard against absurd magnitudes
                        m = (max(p, 1e-9) / pref) ** eps
                        return float(np.clip(m, 0.25, 4.0))  # cap impact; tweak bounds if you like
                    
                    price_mult_joins = _pmult(price, reference_price, join_eps)
                    price_mult_churn = _pmult(price, reference_price, churn_eps)
                    
                    for month in range(MONTHS):
                        
                        # Reset SE wage base every January
                        if (month % 12) == 0:
                            se_ss_wage_base_used_ytd = 0.0
                            
                        grant_received = 0
                        seasonal = SEASONALITY_WEIGHTS_NORM[month % 12]
                        is_downturn = (rng.random() < DOWNTURN_PROB_PER_MONTH)
                        join_mult  = DOWNTURN_JOIN_MULT  if is_downturn else 1.0
                        churn_mult = DOWNTURN_CHURN_MULT if is_downturn else 1.0
                        
                        # ----- Beginner classes (optional) -----
                        revenue_classes_gross = 0.0
                        classes_cost = 0.0
                        class_students_this_month = 0
                        revenue_classes = 0
                        
                        if CLASSES_ENABLED:
                            # stochastic fill around mean
                            for _ in range(int(CLASS_COHORTS_PER_MONTH)):
                                fill = rng.normal(CLASS_FILL_MEAN, 0.08)
                                fill = float(np.clip(fill, 0.0, 1.0))
                                seats = int(round(CLASS_CAP_PER_COHORT * fill))
                                class_students_this_month += seats
                                revenue_classes_gross += seats * CLASS_PRICE
                                classes_cost += (seats * CLASS_COST_PER_STUDENT) + (CLASS_INSTR_RATE_PER_HR * CLASS_HOURS_PER_COHORT)
                        
                            # schedule conversion of a fraction of students to members after a lag
                            # keep a small queue keyed by target month
                            if month == 0:
                                pending_class_conversions = {}
                            target_m = month + int(CLASS_CONV_LAG_MO)
                            converts = int(round(class_students_this_month * CLASS_CONV_RATE))
                            if converts > 0:
                                pending_class_conversions[target_m] = pending_class_conversions.get(target_m, 0) + converts
                             # Net class revenue (flow into total_revenue)
                            revenue_classes = max(0.0, revenue_classes_gross - classes_cost)
    
                        class_joins_now = 0
                        if CLASSES_ENABLED and 'pending_class_conversions' in locals():
                            class_joins_now = int(pending_class_conversions.pop(month, 0))
                            # gate by available supply and MAX_ONBOARDINGS_PER_MONTH later
                        
                        # Replenish pools each month  <-- ADD THESE LINES
                        for _k, _v in MARKET_POOLS_INFLOW.items():
                            remaining_pool[_k] += int(_v)
    
                        # ----- Segment-based ramped adoption -----
                        cap_ratio = len(active_members) / max(1.0, MEMBERSHIP_SOFT_CAP)
                        capacity_damping = max(0.0, 1.0 - cap_ratio**CAPACITY_DAMPING_BETA)
    
                        # Seasonality & downturn still matter
                        seasonal_mult = seasonal
                        downturn_join_mult = join_mult
                        awareness_mult = awareness_multiplier(month)
                        wom_mult = wom_multiplier(len(active_members))
    
                        # Unlock a new tranche of community-studio members at term boundaries
                        unlock_now = compute_cs_unlock_share(month, remaining_pool["community_studio"])
                        unlock_now = min(unlock_now, remaining_pool["community_studio"])
                        remaining_pool["community_studio"] -= unlock_now
                        cs_eligible += unlock_now
    
                        # Build per-pool effective monthly intent
                        # Add small lognormal noise to keep trajectories from looking too synthetic
                        noise = rng.lognormal(mean=-(ADOPTION_SIGMA**2)/2, sigma=ADOPTION_SIGMA)
                        
                        # --- price elasticity multiplier (cheaper than reference => higher intent)
                        price_mult = price_mult_joins
                        
                        intent_common_mult = (
                            seasonal_mult * downturn_join_mult * awareness_mult *
                            wom_mult * capacity_damping * noise * price_mult
                        )
                    
    
                        pool_intents = {
                            "no_access":        POOL_BASE_INTENT["no_access"]        * intent_common_mult,
                            "home_studio":      POOL_BASE_INTENT["home_studio"]      * intent_common_mult,
                            "community_studio": POOL_BASE_INTENT["community_studio"] * intent_common_mult,  # applies only to cs_eligible
                        }
    
                        # Draw adopters from each pool
                        joins_no_access   = draw_adopters(remaining_pool["no_access"],      pool_intents["no_access"], rng)
                        joins_home        = draw_adopters(remaining_pool["home_studio"],    pool_intents["home_studio"], rng)
                        joins_comm_studio = draw_adopters(cs_eligible,                      pool_intents["community_studio"], rng)
    
                        # Update pools
                        remaining_pool["no_access"]   -= joins_no_access
                        remaining_pool["home_studio"] -= joins_home
                        cs_eligible                   -= joins_comm_studio
    
                        # Total joins this month (respect onboarding ops cap, if any)
                        joins = (
                            joins_no_access + joins_home + joins_comm_studio
                            + int(stream.get("joins_from_workshops", np.zeros(MONTHS))[month])
                        )
                        
                        # --- referral loop (Poisson) ---
                        referral_joins = rng.poisson(REFERRAL_RATE_PER_MEMBER * len(active_members) * REFERRAL_CONV)
                        remaining_supply = remaining_pool["no_access"] + remaining_pool["home_studio"] + cs_eligible
                        referral_joins = int(min(referral_joins, remaining_supply))
    
                        # Baseline joins (capacity-aware trickle). Allocate like referrals, respecting remaining supply.
                        cap_ratio = len(active_members) / max(1.0, MEMBERSHIP_SOFT_CAP)
                        baseline_capacity_factor = max(0.0, 1.0 - cap_ratio**CAPACITY_DAMPING_BETA)
                        baseline_demand = int(rng.poisson(BASELINE_JOIN_RATE * MEMBERSHIP_SOFT_CAP * baseline_capacity_factor))
    
                        remaining_supply = remaining_pool["no_access"] + remaining_pool["home_studio"] + cs_eligible
                        baseline_demand = min(baseline_demand, int(remaining_supply))
    
                        bn_no_access = min(baseline_demand, remaining_pool["no_access"])
                        remaining_pool["no_access"] -= bn_no_access
                        spill = baseline_demand - bn_no_access
    
                        bn_cs = min(spill, cs_eligible)
                        cs_eligible -= bn_cs
                        spill -= bn_cs
    
                        bn_home = min(spill, remaining_pool["home_studio"])
                        remaining_pool["home_studio"] -= bn_home
    
                        baseline_joins = bn_no_access + bn_cs + bn_home
                        joins += baseline_joins
                        
                        # apportion referrals
                        ref_no_access = min(referral_joins, remaining_pool["no_access"])
                        remaining_pool["no_access"] -= ref_no_access
                        spill = referral_joins - ref_no_access
    
                        ref_cs = min(spill, cs_eligible)
                        cs_eligible -= ref_cs
                        spill -= ref_cs
    
                        ref_home = min(spill, remaining_pool["home_studio"])
                        remaining_pool["home_studio"] -= ref_home
    
                        # final join count including class conversions
                        joins = (
                            class_joins_now
                            + (joins_no_access + joins_home + joins_comm_studio)
                            + baseline_joins
                            + referral_joins
                        )
                        joins += int(stream["joins_from_workshops"][month])
    
                     # If onboarding capped, roll back proportionally across ALL sources (incl baseline & referrals)
                        if MAX_ONBOARDINGS_PER_MONTH is not None and joins > MAX_ONBOARDINGS_PER_MONTH:
                            overflow = joins - MAX_ONBOARDINGS_PER_MONTH
                            
                            # Let classes soak overflow first (class converts don’t return to pools)
                            rb_classes = min(overflow, class_joins_now)
                            class_joins_now -= rb_classes
                            overflow -= rb_classes
    
                            # totals by source (organic + baseline + referral per pool)
                            take_no_access_total = (joins_no_access + bn_no_access + ref_no_access)
                            take_home_total      = (joins_home      + bn_home      + ref_home)
                            take_cs_total        = (joins_comm_studio + bn_cs      + ref_cs)
    
                            total_drawn = max(1, take_no_access_total + take_home_total + take_cs_total)
    
                            rb_no_access = int(round(overflow * (take_no_access_total / total_drawn)))
                            rb_home      = int(round(overflow * (take_home_total      / total_drawn)))
                            rb_cs        = overflow - rb_no_access - rb_home
    
                            # Return to pools
                            remaining_pool["no_access"]   += rb_no_access
                            remaining_pool["home_studio"] += rb_home
                            cs_eligible                   += rb_cs
    
                            # Reduce per-source takes (prefer rolling back referrals first, then baseline, then organic)
                            # --- no_access
                            give_from_ref = min(rb_no_access, ref_no_access); ref_no_access -= give_from_ref; rb_no_access -= give_from_ref
                            give_from_base = min(rb_no_access, bn_no_access); bn_no_access -= give_from_base; rb_no_access -= give_from_base
                            joins_no_access -= rb_no_access  # whatever remains comes from organic
    
                            # --- home
                            give_from_ref = min(rb_home, ref_home); ref_home -= give_from_ref; rb_home -= give_from_ref
                            give_from_base = min(rb_home, bn_home); bn_home -= give_from_base; rb_home -= give_from_base
                            joins_home -= rb_home
    
                            # --- community studio
                            give_from_ref = min(rb_cs, ref_cs); ref_cs -= give_from_ref; rb_cs -= give_from_ref
                            give_from_base = min(rb_cs, bn_cs); bn_cs -= give_from_base; rb_cs -= give_from_base
                            joins_comm_studio -= rb_cs
    
                            # Recompute aggregates post-rollback
                            baseline_joins = bn_no_access + bn_home + bn_cs
                            referral_joins = ref_no_access + ref_home + ref_cs
                            joins = (
                                class_joins_now
                                + (joins_no_access + joins_home + joins_comm_studio)
                                + baseline_joins
                                + referral_joins
                            )
                            # Safety
                            joins = min(joins, MAX_ONBOARDINGS_PER_MONTH)
    
                        # Create new members (keep archetype mix; NO geometric duration now)
                        # Cap joins so we never exceed MAX_MEMBERS
                        # Cap joins so we never exceed MAX_MEMBERS
                        joins = min(joins, MAX_MEMBERS - len(active_members))
                        
                        # Tag class converts for provenance (first N new members this month)
                        n_from_class = int(locals().get("class_joins_now", 0) or 0)
                        
                        for i in range(int(joins)):
                            archetype = rng.choice(
                                list(MEMBER_ARCHETYPES.keys()),
                                p=[v["prob"] for v in MEMBER_ARCHETYPES.values()]
                            )
                            active_members.append({
                                "type": archetype,
                                "start_month": month,
                                "monthly_fee": MEMBER_ARCHETYPES[archetype]["monthly_fee"],
                                "clay_bags": MEMBER_ARCHETYPES[archetype]["clay_bags"],
                                # NEW: record source to enable later analytics/retention tweaks
                                "src": "class" if i < n_from_class else "other",
                            })
    
                        # Tenure-based churn with utilization uplift near/over capacity (+ seasonality)
                        before = len(active_members)
                        kept = []
                        util_over = max(0.0, (len(active_members) / max(1.0, MEMBERSHIP_SOFT_CAP)) - 1.0)
    
                        # seasonal multiplier for this calendar month (0-based month in the sim)
                        scm = seasonal_churn_mult(month)
    
                        for m in active_members:
                            tenure = month - m["start_month"]
    
                            p_leave = month_churn_prob(m["type"], tenure_mo=tenure)
                            p_leave *= churn_mult                         # downturn regime
                            p_leave *= price_mult_churn 
                            p_leave *= (1.0 + UTILIZATION_CHURN_UPLIFT * util_over)  # crowding
                            p_leave *= scm                                # 🔸 seasonality
                            p_leave = float(np.clip(p_leave, 0.0, 0.99))
    
                            if rng.random() > p_leave:
                                kept.append(m)
    
                        active_members = kept
                        departures = before - len(active_members)
                        net_adds = joins - departures
    
                        # Revenues — membership, clay, firing, events
                        revenue_membership = sum(m["monthly_fee"] for m in active_members)
                        revenue_clay = 0.0  # <-- now NET (pre-tax)
                        revenue_firing = 0.0
                        total_clay_lbs = 0.0
                        
                       # Designated artist studios (stochastic monthly occupancy)
                        ds_occupied = int(rng.binomial(DESIGNATED_STUDIO_COUNT, DESIGNATED_STUDIO_BASE_OCCUPANCY)) if DESIGNATED_STUDIO_COUNT > 0 else 0
                        revenue_designated_studios = ds_occupied * DESIGNATED_STUDIO_PRICE
                     
                        for m in active_members:  
                            bags = rng.choice(m["clay_bags"])
                            revenue_clay += bags * RETAIL_CLAY_PRICE_PER_BAG
                            clay_lbs = bags * 25
                            total_clay_lbs += clay_lbs
                            revenue_firing += compute_firing_fee(clay_lbs)
    
                        # ----- Events: gross revenue and explicit COGS (mugs + consumables + optional labor) -----
                        revenue_events_gross = 0.0
                        events_cost_materials = 0.0
                        events_cost_labor = 0.0

                        events_this_month = 0
                        if events_enabled:
                            # seasonality: keep your existing normalization
                            seasonal = SEASONALITY_WEIGHTS_NORM[month % 12]
                            # stochastic event count with hard cap
                            lam = max(0.0, base_lambda * seasonal)
                            events_this_month = int(np.clip(rng.poisson(lam), 0, events_max_per_month))

                            for _ in range(events_this_month):
                                attendees = int(rng.choice(attendees_range))
                                # revenue
                                event_gross = attendees * ticket_price
                                revenue_events_gross += event_gross
                                # materials (mugs + consumables)
                                mugs_cost = attendees * rng.uniform(*mug_cost_range)
                                consumables_cost = attendees * consumables_pp
                                events_cost_materials += (mugs_cost + consumables_cost)
                                # labor (optional)
                                if staff_rate_hr > 0 and hours_per_event > 0:
                                    events_cost_labor += staff_rate_hr * hours_per_event

                        revenue_events = max(0.0, revenue_events_gross - events_cost_materials - events_cost_labor)
                        
                        # Variable costs
                        variable_clay_cost = (total_clay_lbs / 25) * WHOLESALE_CLAY_COST_PER_BAG
                        water_cost = total_clay_lbs / 25 * GALLONS_PER_BAG_CLAY * WATER_COST_PER_GALLON
    
                        # Electricity
                        if "I_all_upfront" in scen_name:
                            kiln2_on = (len(active_members) >= EXPANSION_TRIGGER_MEMBERS) if KILN2_AFTER_THRESHOLD_SCENARIO_I else True
                        else:
                            kiln2_on = expansion_triggered if KILN2_AFTER_EXPANSION_SCENARIO_II else True
    
                        firings = firings_this_month(len(active_members))
                        kwh_per_firing = KWH_PER_FIRING_KMT1027 + (KWH_PER_FIRING_KMT1427 if kiln2_on else 0)
                        electricity_cost = firings * kwh_per_firing * COST_PER_KWH
    
                        # Heating
                        monthly_heating_cost = HEATING_COST_WINTER if month % 12 in [10, 11, 0, 1, 2, 3] else HEATING_COST_SUMMER
    
                        #Staff cost after expansion
                        staff_cost = STAFF_COST_PER_MONTH if len(active_members) >= STAFF_EXPANSION_THRESHOLD else 0.0
                        
                        #Maintenance
                        maintenance_cost = MAINTENANCE_BASE_COST + max(0, rng.normal(0, MAINTENANCE_RANDOM_STD))
                        
                        #Marketing
                        if month < MARKETING_RAMP_MONTHS:
                            marketing_cost = MARKETING_COST_BASE * MARKETING_RAMP_MULTIPLIER
                        else:
                            marketing_cost = MARKETING_COST_BASE
                        
                       # ---------- S-corp owner salary (expense) & employer payroll taxes ----------
                        owner_salary_expense = 0.0
                        employer_payroll_tax = 0.0
                        if ENTITY_TYPE == "s_corp":
                            owner_salary_expense = SCORP_OWNER_SALARY_PER_MONTH
                            employer_payroll_tax = owner_salary_expense * EMPLOYER_PAYROLL_TAX_RATE
    
                        # Employee-side FICA withheld from wages (also remitted in cash by the business)
                        employee_withholding = 0.0
                        if ENTITY_TYPE == "s_corp":
                            employee_withholding = owner_salary_expense * EMPLOYEE_PAYROLL_TAX_RATE
    
                        # ---------- OpEx (pre-tax) ----------
                        fixed_opex_profit = fixed_rent + INSURANCE_COST + GLAZE_COST_PER_MONTH + monthly_heating_cost
                        total_opex_profit = (
                            fixed_opex_profit
                            + variable_clay_cost
                            + water_cost
                            + electricity_cost
                            + staff_cost
                            + marketing_cost
                            + maintenance_cost
                            + owner_salary_expense
                            + employer_payroll_tax
                        )
    
                        in_draw_window = in_owner_draw_window(month)  # existing calendar gate (start/end months)
                        within_stipend_quota = (month < OWNER_STIPEND_MONTHS)  # stipend only for first N months
                        owner_draw_now = owner_draw if (in_draw_window and within_stipend_quota) else 0.0
                        fixed_opex_cash = fixed_opex_profit + monthly_loan_payment + owner_draw_now
    
                        # Cash OpEx (pre-tax)
                        total_opex_cash = (
                            fixed_opex_cash
                            + variable_clay_cost
                            + water_cost
                            + electricity_cost
                            + staff_cost
                            + maintenance_cost
                            + marketing_cost
                            + owner_salary_expense
                            + employer_payroll_tax
                        )
                        
                        if ENTITY_TYPE == "s_corp":
                            total_opex_cash += employee_withholding
                            
                        total_revenue = (
                            revenue_membership + revenue_clay + revenue_firing + revenue_events
                            + stream["workshop_revenue"][month]
                            + revenue_designated_studios
                            + (0.0 if not CLASSES_ENABLED else revenue_classes)
                        )
    
                        # ---------- Operating profit (pre-tax) ----------
                        op_profit = total_revenue - total_opex_profit
    
                        # ---------- Pass-through vs corporate tax accrual ----------
                        se_tax_this_month = 0.0
                        state_income_tax_this_month = 0.0
                        corp_tax_this_month = 0.0
    
                        if ENTITY_TYPE in ("sole_prop", "partnership"):
                            se_earnings = max(0.0, op_profit) * SE_EARNINGS_FACTOR
                            ss_base_remaining = max(0.0, SE_SOC_SEC_WAGE_BASE - se_ss_wage_base_used_ytd)
                            ss_taxable_now = min(se_earnings, ss_base_remaining)
                            se_tax_ss = ss_taxable_now * SE_SOC_SEC_RATE
                            se_ss_wage_base_used_ytd += ss_taxable_now
    
                            se_tax_medicare = se_earnings * SE_MEDICARE_RATE
                            se_tax_this_month = se_tax_ss + se_tax_medicare
                            se_tax_payable_accum += se_tax_this_month
    
                            half_se_deduction = 0.5 * se_tax_this_month
                            ma_taxable_income = max(0.0, op_profit - half_se_deduction)
                            state_income_tax_this_month = ma_taxable_income * MA_PERSONAL_INCOME_TAX_RATE
                            state_tax_payable_accum += state_income_tax_this_month
    
                        elif ENTITY_TYPE == "s_corp":
                            # Owner salary + employer payroll tax already included in OpEx above.
                            ma_taxable_income = max(0.0, op_profit)
                            state_income_tax_this_month = ma_taxable_income * MA_PERSONAL_INCOME_TAX_RATE
                            state_tax_payable_accum += state_income_tax_this_month
    
                        elif ENTITY_TYPE == "c_corp":
                            corp_taxable_income = max(0.0, op_profit)
                            corp_tax_this_month = corp_taxable_income * (FED_CORP_TAX_RATE + MA_CORP_TAX_RATE)
                            corp_tax_payable_accum += corp_tax_this_month
    
                        # ---------- Annual personal property tax (cash only unless you prefer accrual) ----------
                        property_tax_this_month = 0.0
                        if PERSONAL_PROPERTY_TAX_ANNUAL > 0 and ((month + 1) % 12 == (PERSONAL_PROPERTY_TAX_BILL_MONTH % 12)):
                            property_tax_this_month = PERSONAL_PROPERTY_TAX_ANNUAL
    
                        # ---------- Quarterly remittances (cash) ----------
                        tax_payments_this_month = 0.0
                        if ((month + 1) % ESTIMATED_TAX_REMIT_FREQUENCY_MONTHS) == 0:
                            if ENTITY_TYPE in ("sole_prop", "partnership", "s_corp"):
                                tax_payments_this_month += se_tax_payable_accum
                                tax_payments_this_month += state_tax_payable_accum
                                se_tax_payable_accum = 0.0
                                state_tax_payable_accum = 0.0
                            if ENTITY_TYPE == "c_corp":
                                tax_payments_this_month += corp_tax_payable_accum
                                corp_tax_payable_accum = 0.0
    
                        # Sales tax (retail clay here) — collected and remitted (cash only)
                        sales_tax_collected = revenue_clay * MA_SALES_TAX_RATE
                        sales_tax_payable_accum += sales_tax_collected
                        sales_tax_remitted = 0.0
                        if ((month + 1) % SALES_TAX_REMIT_FREQUENCY_MONTHS) == 0:
                            sales_tax_remitted = sales_tax_payable_accum
                            tax_payments_this_month += sales_tax_remitted
                            sales_tax_payable_accum = 0.0
    
                        # ---------- Cash view ----------
                        total_opex_cash += property_tax_this_month
                        total_opex_cash += tax_payments_this_month
                        net_cash_flow = total_revenue - total_opex_cash + sales_tax_collected
    
                        # Accrual after-tax profit (income/self-employment/corp taxes only)
                        tax_cost = se_tax_this_month + state_income_tax_this_month + corp_tax_this_month
                        op_profit_after_tax = op_profit - tax_cost
    
                        # DSCR based on pre-tax operating profit (keep if you want to track it)
                        dscr = (op_profit / monthly_loan_payment) if monthly_loan_payment > 0 else np.nan
                        
                        # --- CFADS / Cash-DSCR (lender standard) ---
                        # total_opex_cash currently INCLUDES:
                        #   • cash operating costs (fixed + variable + payroll items)
                        #   • cash tax remittances (quarterly, etc.)
                        #   • owner_draw_now
                        #   • monthly_loan_payment
                        # We want CFADS to EXCLUDE owner draws and EXCLUDE debt service,
                        # but INCLUDE cash taxes and all other cash opex.
                        
                        opex_cash_excl_debt_and_draws = (
                            total_opex_cash
                            - monthly_loan_payment    # remove debt service
                            - owner_draw_now          # remove distributions
                        )
                        
                        cfads = total_revenue - opex_cash_excl_debt_and_draws
                        
                        dscr_cash = (cfads / monthly_loan_payment) if monthly_loan_payment > 0 else np.nan
    
                        # Month 0 loan & capex (use split principals; cash = total proceeds − upfront CapEx spend)
                        if month == 0:
                            if "all_upfront" in scen_name:
                                upfront_capex = (capex_I_cost + capex_II_cost)
                            else:
                                upfront_capex = capex_I_cost
                            cash_balance += loan_principal_total - upfront_capex
                            cumulative_after_capex -= upfront_capex
    
                        # Staged expansion capex at trigger
                        if ("staged" in scen_name) and (not expansion_triggered) and (len(active_members) >= EXPANSION_TRIGGER_MEMBERS):
                            cash_balance -= capex_II_cost
                            cumulative_after_capex -= capex_II_cost
                            expansion_triggered = True
    
                        # Apply monthly results
                        cash_balance += net_cash_flow
                        cumulative_op_profit += op_profit
                        cumulative_after_capex += op_profit
    
                        # Insolvency before grant
                        # FIX
                        pre_grant_or_no_grant = (grant_month is None) or (month < grant_month)
                        if pre_grant_or_no_grant and (cash_balance < 0) and (not insolvent_before_grant):
                            insolvent_before_grant = True
    
                        # Grant (keep 0-based)
                        if (grant_month is not None) and (month == grant_month):
                            cash_balance += grant_amount
                            grant_received = grant_amount
    
                        # Store row
                        rows.append({
                            "simulation_id": sim,
                            "scenario": scen_name,
                            "rent": fixed_rent,
                            "owner_draw": owner_draw,
                            "month": month + 1,
                            "active_members": len(active_members),
                            "joins": joins,
                            "departures": departures,
                            "net_adds": net_adds,
                            "cash_balance": cash_balance,
                            "net_cash_flow": net_cash_flow,
                            "cumulative_op_profit": cumulative_op_profit,
                            "cumulative_profit_after_capex": cumulative_after_capex,
                            "revenue_membership": revenue_membership,
                            "revenue_firing": revenue_firing,
                            "revenue_clay": revenue_clay,
                            "revenue_events": revenue_events,
                            "revenue_workshops_net": net_ws,
                            "revenue_designated_studios": revenue_designated_studios,
                            "designated_studio_occupied": ds_occupied,
                            "grant_received": grant_received,
                            "insolvent_before_grant": insolvent_before_grant,
                            "grant_month": grant_month,
                            "grant_amount": grant_amount,
                            "is_downturn": is_downturn,                        
                            "loan_payment_total": monthly_loan_payment,
                            "loan_payment_504": monthly_loan_payment_504,
                            "loan_payment_7a": monthly_loan_payment_7a,
                            "loan_principal_total": loan_principal_total,
                            "loan_principal_504": loan_504_principal,
                            "loan_principal_7a": loan_7a_principal,
                            "capex_I_cost": capex_I_cost,
                            "capex_II_cost": capex_II_cost,
                            "runway_costs": sized_runway_costs,
                            "dscr": dscr,
                            "dscr_cash": dscr_cash,
                            "staff_cost": staff_cost,
                            "maintenance_cost": maintenance_cost,
                            "marketing_cost": marketing_cost,
                            "op_profit_after_tax": op_profit_after_tax,
                            "sales_tax_collected": sales_tax_collected,
                            "sales_tax_remitted": sales_tax_remitted,
                            "se_tax_accrued": se_tax_this_month,
                            "state_income_tax_accrued": state_income_tax_this_month,
                            "corp_tax_accrued": corp_tax_this_month,
                            "tax_payments_made": tax_payments_this_month,
                            "property_tax": property_tax_this_month,
                            "owner_salary_expense": owner_salary_expense,
                            "employer_payroll_tax": employer_payroll_tax,
                            "entity_type": ENTITY_TYPE,
                            "owner_draw_paid": owner_draw_now,
                            "employee_withholding": employee_withholding,
                            "workshop_attendees": workshop_attendees,
                            "workshop_gross": gross_ws,
                            "workshop_cost": cost_ws,
                            "events_this_month": events_this_month,
                            "revenue_events_gross": revenue_events_gross,
                            "events_cost_materials": events_cost_materials,
                            "events_cost_labor": events_cost_labor,
                            "revenue_classes_gross": revenue_classes_gross,
                            "classes_cost": classes_cost,
                            "revenue_classes": revenue_classes,
                            "class_students": class_students_this_month,
                            "cfads":cfads,
                            "dscr_cash_breach_1_00": (dscr_cash < 1.00) if np.isfinite(dscr_cash) else False,
                            "dscr_cash_breach_1_25": (dscr_cash < DSCR_CASH_TARGET) if np.isfinite(dscr_cash) else False,
                        })
    
    # ---- Build DataFrame ----
    results_df = pd.DataFrame(rows)
    print("Built results_df with shape:", results_df.shape)
    
    # =============================================================================
    # Dashboard Plots
    # =============================================================================
    sns.set_context("talk")
    
    # Global membership (median + band) with cap
    g = results_df.groupby("month")["active_members"]
    med = g.median(); p10 = g.quantile(0.10); p90 = g.quantile(0.90)
    plt.figure(figsize=(10,6))
    plt.plot(med.index, med.values, linewidth=2, label="Median")
    plt.fill_between(med.index, p10.values, p90.values, alpha=0.12, label="10–90% range")
    plt.axhline(MEMBERSHIP_SOFT_CAP, linestyle="--", linewidth=1.5, label=f"Soft cap ≈ {MEMBERSHIP_SOFT_CAP:.0f}")
    # Optional hard cap (if MAX_MEMBERS is set)
    try:
        plt.axhline(MAX_MEMBERS, linestyle=":", linewidth=1.5, color="orange", label=f"Hard cap = {int(MAX_MEMBERS)}")
    except Exception:
        pass
    plt.title("Projected Membership Over Time — Capacity-aware")
    plt.xlabel("Month"); plt.ylabel("Active Members")
    plt.legend(); plt.tight_layout(); plt.show()
    
    # Cash balance overlays per (scenario, rent)
    for scen in results_df["scenario"].unique():
        for rent_val in sorted(results_df["rent"].unique()):
            df_rent = results_df[(results_df["scenario"] == scen) & (results_df["rent"] == rent_val)]
            if df_rent.empty:
                continue
    
            fig, ax = plt.subplots(figsize=(10, 6))
            for od, df_od in df_rent.groupby("owner_draw"):
                grouped = df_od.groupby("month")["cash_balance"]
                median = grouped.median()
                p10 = grouped.quantile(0.1)
                p90 = grouped.quantile(0.9)
    
                ax.plot(median.index, median.values, label=f"Draw ${od:,.0f}/mo", linewidth=2)
                ax.fill_between(median.index, p10.values, p90.values, alpha=0.10)
    
            # Grant markers
            grant_info = df_rent[["grant_month", "grant_amount"]].drop_duplicates()
            for _, row in grant_info.iterrows():
                gm, ga = row["grant_month"], row["grant_amount"]
                if pd.notna(gm) and gm is not None:
                    ax.axvline(gm+1, color="green", linestyle="--", linewidth=1.5)
                    ymin, ymax = ax.get_ylim()
                    ax.text(gm+1, ymin + 0.05*(ymax - ymin), f"Grant ${ga:,.0f}", color="green", rotation=90, va="bottom")
    
            ax.axhline(0, color="black", linestyle=":", linewidth=1)
            ax.set_title(f"Cash Balance Over Time — {scen} | Rent ${rent_val:,.0f}/mo")
            ax.set_xlabel("Month"); ax.set_ylabel("Cash Balance ($)")
            ax.legend(loc="best"); plt.tight_layout(); plt.show()
    
    # Operating break-even heatmaps
    def first_break_even(group):
        be_month = group.loc[group["cumulative_op_profit"] >= 0, "month"].min()
        return be_month if pd.notna(be_month) else np.nan
    
    be_df = (
        results_df
        .groupby(["scenario", "rent", "owner_draw", "simulation_id"])
        .apply(first_break_even)
        .reset_index(name="op_break_even_month")
    )
    
    for scen in be_df["scenario"].unique():
        pivot = be_df[be_df["scenario"] == scen].pivot_table(
            index="owner_draw", columns="rent", values="op_break_even_month", aggfunc="median"
        )
        plt.figure(figsize=(8,6))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={"label": "Months"})
        plt.title(f"Median Months to Operating Break-Even — {scen}")
        plt.xlabel("Monthly Rent ($)"); plt.ylabel("Owner Draw ($/mo)")
        plt.tight_layout(); plt.show()
    
    # Revenue vs OpEx (cash) small-multiples, including workshops
    for scen in results_df["scenario"].unique():
        for rent_val in sorted(results_df["rent"].unique()):
            df_rent = results_df[(results_df["scenario"] == scen) & (results_df["rent"] == rent_val)]
            if df_rent.empty:
                continue
    
            owner_draws = sorted(df_rent["owner_draw"].unique())
            ncols = len(owner_draws); nrows = 1
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 3.6*nrows), sharey=True, sharex=True)
            axes = np.atleast_2d(axes)
    
            for j, od in enumerate(owner_draws):
                ax = axes[0, j]
                df_sub = df_rent[df_rent["owner_draw"] == od].copy()
    
                rev_components = [
                    "revenue_membership",
                    "revenue_firing",
                    "revenue_clay",
                    "revenue_events",
                    "revenue_workshops_net",
                    "revenue_designated_studios",
                    "revenue_classes",  # NEW: classes (net)
                ]
                df_sub["total_revenue"] = df_sub[rev_components].sum(axis=1)
                df_sub["total_opex_cash"] = df_sub["total_revenue"] - df_sub["net_cash_flow"]

                g_rev = df_sub.groupby("month")[rev_components].median()
                g_ops = df_sub.groupby("month")[["total_revenue", "total_opex_cash", "net_cash_flow"]].median()

                ax.stackplot(
                    g_rev.index,
                    g_rev["revenue_membership"],
                    g_rev["revenue_firing"],
                    g_rev["revenue_clay"],
                    g_rev["revenue_events"],
                    g_rev["revenue_workshops_net"],
                    g_rev["revenue_designated_studios"],
                    g_rev["revenue_classes"],  # NEW
                    labels=[
                        "Membership", "Firing Fees", "Clay", "Events",
                        "Workshops (net)", "Designated Studios", "Classes (net)"  # NEW
                    ],
                    alpha=0.9,
                )
                
                ax.plot(g_ops.index, g_ops["total_opex_cash"], linewidth=2.0, label="Total OpEx (cash)")
                ax.plot(g_ops.index, g_ops["net_cash_flow"], linestyle="--", linewidth=1.5, label="Net Cash Flow")
    
                ax.set_title(f"Draw ${od:,.0f}/mo", fontsize=11)
                if j == 0: ax.set_ylabel("Dollars ($)")
                ax.set_xlabel("Month")
    
           # Collect and de‑duplicate legend entries from all facets
            for ax in fig.axes:
                # keep existing text but shrink font
                ax.set_xlabel(ax.get_xlabel(), fontsize=10)
                ax.set_ylabel(ax.get_ylabel(), fontsize=10)
                ax.tick_params(axis="both", which="major", labelsize=9)
                ax.tick_params(axis="both", which="minor", labelsize=8)  # if you have minor ticks
                
            handles_all, labels_all = [], []
            for ax_ in axes.flat:
                h_, l_ = ax_.get_legend_handles_labels()
                handles_all.extend(h_); labels_all.extend(l_)
            from collections import OrderedDict
            by_label = OrderedDict()
            for h_, l_ in zip(handles_all, labels_all):
                if l_ not in by_label:
                    by_label[l_] = h_
            # ↓ Make axis labels and tick labels smaller on all axes in this figure
            
            fig.legend(
                list(by_label.values()),
                list(by_label.keys()),
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),   # push below x-axis
                ncol=min(4, len(by_label)),    # spread entries
                frameon=False,
                fontsize = 10
            )
            
            # Title + spacing that leaves headroom (top) and legend space (bottom)
            fig.suptitle(f"Revenue vs OpEx — {scen} | Rent ${rent_val:,.0f}/mo", y=0.98, fontsize=14)
            fig.tight_layout(rect=[0, 0.08, 1, 0.95])  # leave more room at bottom
            plt.show()
    
    
    # --- Spaghetti + band for one configuration ---
    scenario_pick   = results_df["scenario"].unique()[0]
    rent_pick       = sorted(results_df["rent"].unique())[0]
    owner_draw_pick = sorted(results_df["owner_draw"].unique())[0]
    
    cfg = results_df[
        (results_df["scenario"] == scenario_pick) &
        (results_df["rent"] == rent_pick) &
        (results_df["owner_draw"] == owner_draw_pick)
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # median band
    g = cfg.groupby("month")["active_members"]
    med = g.median(); p10 = g.quantile(0.10); p90 = g.quantile(0.90)
    ax.plot(med.index, med.values, linewidth=2, label="Median")
    ax.fill_between(med.index, p10.values, p90.values, alpha=0.12, label="10–90%")
    ax.axhline(MEMBERSHIP_SOFT_CAP, linestyle="--", linewidth=1.5, label=f"Soft cap ≈ {MEMBERSHIP_SOFT_CAP:.0f}")
    try:
        ax.axhline(MAX_MEMBERS, linestyle=":", linewidth=1.5, color="orange", label=f"Hard cap = {int(MAX_MEMBERS)}")
    except Exception:
        pass
    
    # random paths
    _from = cfg["simulation_id"].unique()
    local_rng = default_rng(SeedSequence([RANDOM_SEED, 999, int(rent_pick), int(owner_draw_pick)]))
    for sim_id in local_rng.choice(_from, size=min(25, len(_from)), replace=False):
        path = cfg[cfg["simulation_id"] == sim_id].sort_values("month")
        ax.plot(path["month"], path["active_members"], linewidth=0.8, alpha=0.35)
    
    ax.set_title(f"Membership Paths — {scenario_pick} | Rent ${rent_pick:,.0f}/mo | Draw ${owner_draw_pick:,.0f}/mo")
    ax.set_xlabel("Month"); ax.set_ylabel("Active Members")
    ax.legend(loc="best"); plt.tight_layout(); plt.show()
            
            
    # Net adds histograms + probability of negative months (global simplified view)
    vals = results_df["net_adds"].values
    plt.figure(figsize=(10,6))
    plt.hist(vals, bins=range(int(vals.min())-1, int(vals.max())+2), alpha=0.5)
    plt.axvline(0, color="black", linestyle=":")
    plt.title("Distribution of Monthly Net Adds — All configurations")
    plt.xlabel("Net Adds (Joins − Departures) per Month"); plt.ylabel("Frequency")
    plt.tight_layout(); plt.show()
    
    neg_prob = (results_df["net_adds"] < 0).mean()
    print(f"Probability of a negative net-adds month (global): {neg_prob:.1%}")
    
    # Optional: Workshop timing diagnostic — fraction of months with workshops (by scenario)
    diag = (results_df.assign(ws=(results_df["revenue_workshops_net"]>0))
            .groupby(["scenario","simulation_id"])["ws"].mean().groupby("scenario").describe())
    print("\nWorkshop timing diagnostic (share of months with a workshop):")
    print(diag)
    
    # =============================================================================
    # Summary Table
    # =============================================================================
    insolvent_summary = (
        results_df
        .groupby(["scenario", "rent", "owner_draw", "simulation_id"])
        .agg({"insolvent_before_grant":"max"})
        .groupby(level=[0,1,2]).mean()
        .rename(columns={"insolvent_before_grant": "pct_insolvent_before_grant"})
    )
    
    be_df = (
        results_df
        .groupby(["scenario", "rent", "owner_draw", "simulation_id"])
        .apply(first_break_even)
        .reset_index(name="op_break_even_month")
    )
    
    beop_summary = (
        be_df
        .groupby(["scenario", "rent", "owner_draw"])["op_break_even_month"]
        .median()
        .to_frame("median_op_break_even_month")
    )
    
    final_cash_summary = (
        results_df[results_df["month"] == MONTHS]
        .groupby(["scenario", "rent", "owner_draw"])["cash_balance"]
        .median()
        .to_frame("median_final_cash_m60")
    )
    
    # Median minimum cash across the horizon (stress indicator)
    min_cash_summary = (
        results_df.groupby(["scenario","rent","owner_draw","simulation_id"])["cash_balance"].min()
        .groupby(level=[0,1,2]).median()
        .to_frame("median_min_cash")
    )
    
    # Median CFADS months 12 & 24
    cfads_12 = (results_df[results_df["month"]==12]
                .groupby(["scenario","rent","owner_draw"])["cfads"].median()
                .rename("median_cfads_m12"))
    cfads_24 = (results_df[results_df["month"]==24]
                .groupby(["scenario","rent","owner_draw"])["cfads"].median()
                .rename("median_cfads_m24"))
    
    # % months breaching cash-DSCR<1.25
    breach_rate = (results_df.groupby(["scenario","rent","owner_draw"])["dscr_cash_breach_1_25"]
                   .mean().rename("%_months_below_1_25"))
    
    
    summary_table = (insolvent_summary
                     .join(beop_summary, how="outer")
                     .join(final_cash_summary, how="outer")
                     .join(min_cash_summary, how="outer")
                     .join(cfads_12, on=["scenario","rent","owner_draw"])
                     .join(cfads_24, on=["scenario","rent","owner_draw"])
                     .join(breach_rate, on=["scenario","rent","owner_draw"]))
    
    
    
    # Add median monthly revenue from designated studios
    ds_rev_summary = (
        results_df
        .groupby(["scenario", "rent", "owner_draw", "simulation_id"])["revenue_designated_studios"]
        .median()
        .groupby(level=[0, 1, 2]).median()
        .to_frame("median_monthly_ds_revenue")
    )
    summary_table = summary_table.join(
        ds_rev_summary,
        on=["scenario", "rent", "owner_draw"]
    )
    
    summary_table = summary_table.reset_index()
    print("\n=== Summary (by scenario, rent, owner_draw) ===")
    print(summary_table.to_string(index=False))
    
    # Optional: export CSVs for lender materials
    try:
        results_df.to_csv("gcws_results_detailed.csv", index=False)
        summary_table.to_csv("gcws_summary_table.csv", index=False)
        print("\nSaved: gcws_results_detailed.csv and gcws_summary_table.csv")
    except Exception as e:
        print("CSV export skipped:", e)
    
        
        
    # =============================================================================
    # Owner take-home summary (salary + draws − personal taxes − employee FICA)
    # =============================================================================
    def summarize_owner_takehome(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
    
        # employee FICA withheld from S‑corp wages (owner’s side)
        out["employee_withholding"] = np.where(
            out["entity_type"] == "s_corp",
            out["owner_salary_expense"] * EMPLOYEE_PAYROLL_TAX_RATE,
            0.0
        )
    
        # net wage paid to owner after employee FICA (only relevant for s_corp)
        out["wage_net_to_owner"] = out["owner_salary_expense"] - out["employee_withholding"]
    
        # pass‑through personal taxes we already accrued in the model
        # (sole prop / partnership: SE tax + MA personal; s‑corp: MA personal on pass-through)
        out["personal_taxes_owner"] = np.where(
            out["entity_type"].isin(["sole_prop", "partnership", "s_corp"]),
            out["se_tax_accrued"] + out["state_income_tax_accrued"],
            0.0
        )
    
        # make sure the column exists even if you forgot to add it
        if "owner_draw_paid" not in out.columns:
            out["owner_draw_paid"] = 0.0  # fallback
    
        # monthly owner take-home cash
        out["owner_takehome_cash"] = (
            out["wage_net_to_owner"] + out["owner_draw_paid"] - out["personal_taxes_owner"]
        )
    
        # aggregate to annual per simulation (12 months) and then take scenario medians
        annual = (
            out.groupby(["scenario", "rent", "owner_draw", "simulation_id"], as_index=False)
               .agg(owner_takehome_annual=("owner_takehome_cash", "sum"))
        )
        med = (annual
               .groupby(["scenario", "rent", "owner_draw"], as_index=False)["owner_takehome_annual"]
               .median()
               .rename(columns={"owner_takehome_annual": "median_owner_takehome_annual"}))
    
        return med
    
    owner_takehome_table = summarize_owner_takehome(results_df)
    print("\n=== Median Owner Take‑Home (annual, by scenario/rent/draw) ===")
    print(owner_takehome_table.to_string(index=False))
    
    # =============================================================================
    # Optional: Sweep EXTRA_BUFFER to keep median min-cash >= 0 (ask sizing helper)
    # =============================================================================
    def sweep_extra_buffer(buffers=(0, 5000, 10000, 15000, 20000)):
        """
        Coarse sensitivity on EXTRA_BUFFER added to the loan principal.
        Returns a DataFrame with median min-cash by (scenario, rent, owner_draw) for each buffer.
        This does not alter baseline results unless you manually set EXTRA_BUFFER and re-run.
        """
        import pandas as pd
        global EXTRA_BUFFER
        baseline_extra = EXTRA_BUFFER
        out = []
    
        for b in buffers:
            EXTRA_BUFFER = float(b)
    
            # --- Minimal re-run to capture min cash per simulation ---
    
            rows_min = []
            MEMBERSHIP_SOFT_CAP, _ = compute_membership_soft_cap()
    
            for fixed_rent in RENT_SCENARIOS:
                for owner_draw in OWNER_DRAW_SCENARIOS:
                    for scen_cfg in SCENARIO_CONFIGS:
                        scen_name = scen_cfg["name"]
                        for sim in range(N_SIMULATIONS):
                            scen_index = next(i for i, s in enumerate(SCENARIO_CONFIGS) if s["name"] == scen_name)
                            ss = SeedSequence([RANDOM_SEED, int(fixed_rent), int(owner_draw), int(scen_index), int(sim)])
                            rng = default_rng(ss)
                            
                            # --- Pricing elasticity setup (fixed baseline) ---
                            price = float(globals().get("PRICE", 165.0))
                            reference_price = float(globals().get("REFERENCE_PRICE", price))  # default to current price if missing
                            join_eps  = float(globals().get("JOIN_PRICE_ELASTICITY", -0.6))  # negative
                            churn_eps = float(globals().get("CHURN_PRICE_ELASTICITY",  0.3))  # positive
                            def _pmult(p, pref, eps):
                                if pref <= 0: 
                                    return 1.0
                                m = (max(p, 1e-9) / pref) ** eps
                                return float(np.clip(m, 0.25, 4.0))  # safety caps
                            price_mult_joins = _pmult(price, reference_price, join_eps)
                            price_mult_churn = _pmult(price, reference_price, churn_eps)
                            # --- CapEx and loan sizing (aligned with main sim) ---
                            capex_I_cost = sample_capex(STAGE_I_CAPEX, rng)
                            capex_II_cost = sample_capex(STAGE_II_CAPEX, rng)
    
                            # Runway (INCLUDES owner draw, matching main sim)
                            avg_monthly_heat = (HEATING_COST_WINTER + HEATING_COST_SUMMER) / 2
                            runway_costs = (
                                INSURANCE_COST + GLAZE_COST_PER_MONTH + avg_monthly_heat + fixed_rent + owner_draw
                            ) * RUNWAY_MONTHS
    
                            # Split loans (504 = CapEx + contingency; 7(a) = runway + EXTRA_BUFFER)
                            total_capex_for_loan = (capex_I_cost + capex_II_cost) if ("all_upfront" in scen_name) else capex_I_cost
                            loan_504_principal = total_capex_for_loan * (1 + LOAN_CONTINGENCY_PCT)
                            loan_7a_principal  = runway_costs + EXTRA_BUFFER
    
                            # Monthly debt service (define these; they’re used later)
                            monthly_loan_payment_504 = calculate_monthly_payment(
                                loan_504_principal, LOAN_504_ANNUAL_RATE, LOAN_504_TERM_YEARS
                            )
                            monthly_loan_payment_7a = calculate_monthly_payment(
                                loan_7a_principal, LOAN_7A_ANNUAL_RATE, LOAN_7A_TERM_YEARS
                            )
                            monthly_loan_payment = monthly_loan_payment_504 + monthly_loan_payment_7a
    
                            loan_principal_total = loan_504_principal + loan_7a_principal
    
                            cash_balance = 0.0
                            active_members = []
                            remaining_pool = MARKET_POOLS.copy()
                            cs_eligible = 0
                            expansion_triggered = False
                            grant_month = scen_cfg["grant_month"]; grant_amount = scen_cfg["grant_amount"]
                            min_cash = float("inf")
    
                            for month in range(MONTHS):
                                seasonal = SEASONALITY_WEIGHTS_NORM[month % 12]
                                
                                is_downturn = (rng.random() < DOWNTURN_PROB_PER_MONTH)
                                churn_mult = DOWNTURN_CHURN_MULT if is_downturn else 1.0
                                
    
                                # Replenish pools each month
                                for _k, _v in MARKET_POOLS_INFLOW.items():
                                    remaining_pool[_k] += int(_v)
    
                                # Unlock CS tranche
                                unlock_now = compute_cs_unlock_share(month, remaining_pool["community_studio"])
                                unlock_now = min(unlock_now, remaining_pool["community_studio"])
                                remaining_pool["community_studio"] -= unlock_now
                                cs_eligible += unlock_now
    
                                # Joins (coarse – same intent structure)
                                noise = rng.lognormal(mean=-(ADOPTION_SIGMA**2)/2, sigma=ADOPTION_SIGMA)
                                price_mult = price_mult_joins
                                cap_ratio = len(active_members) / max(1.0, MEMBERSHIP_SOFT_CAP)
                                capacity_damping = max(0.0, 1.0 - cap_ratio**CAPACITY_DAMPING_BETA)
                                intent_common_mult = (seasonal * (DOWNTURN_JOIN_MULT if is_downturn else 1.0) *
                                                      awareness_multiplier(month) * wom_multiplier(len(active_members)) *
                                                      capacity_damping * noise * price_mult)
                                pool_intents = {
                                    "no_access":        POOL_BASE_INTENT["no_access"]        * intent_common_mult,
                                    "home_studio":      POOL_BASE_INTENT["home_studio"]      * intent_common_mult,
                                    "community_studio": POOL_BASE_INTENT["community_studio"] * intent_common_mult,
                                }
                                joins_no_access   = draw_adopters(remaining_pool["no_access"],   pool_intents["no_access"],rng)
                                joins_home        = draw_adopters(remaining_pool["home_studio"], pool_intents["home_studio"],rng)
                                joins_comm_studio = draw_adopters(cs_eligible,                   pool_intents["community_studio"],rng)
                                remaining_pool["no_access"] -= joins_no_access
                                remaining_pool["home_studio"] -= joins_home
                                cs_eligible -= joins_comm_studio
                                joins = joins_no_access + joins_home + joins_comm_studio
                                # Cap onboarding simply
                                if MAX_ONBOARDINGS_PER_MONTH is not None and joins > MAX_ONBOARDINGS_PER_MONTH:
                                    joins = MAX_ONBOARDINGS_PER_MONTH
    
                                # Add members
                                n_from_class = int(locals().get("class_joins_now", 0) or 0)
                                for i in range(int(joins)):
                                    arch = rng.choice(
                                        list(MEMBER_ARCHETYPES.keys()),
                                        p=[v["prob"] for v in MEMBER_ARCHETYPES.values()]
                                    )
                                    active_members.append({
                                        "type": arch,
                                        "start_month": month,
                                        "monthly_fee": MEMBER_ARCHETYPES[arch]["monthly_fee"],
                                        "clay_bags": MEMBER_ARCHETYPES[arch]["clay_bags"],
                                        "src": "class" if i < n_from_class else "other",
                                    })
    
                                # Churn
                                kept = []
                                util_over = max(0.0, (len(active_members) / max(1.0, MEMBERSHIP_SOFT_CAP)) - 1.0)
                                for m in active_members:
                                    p_leave = month_churn_prob(m["type"], tenure_mo=(month - m["start_month"])) * churn_mult * price_mult_churn
                                    p_leave = np.clip(p_leave * (1.0 + UTILIZATION_CHURN_UPLIFT * util_over), 0.0, 0.99)
                                    if rng.random() > p_leave: kept.append(m)
                                active_members = kept
    
                                # Revenues (simplified calc consistent with main code)
                                # Stage-II CapEx spend when expansion triggers (match main sim)
                                if ("staged" in scen_name) and (not expansion_triggered) and (len(active_members) >= EXPANSION_TRIGGER_MEMBERS):
                                    cash_balance -= capex_II_cost
                                    expansion_triggered = True
                                revenue_membership = sum(m["monthly_fee"] for m in active_members)
                                revenue_clay = 0.0; revenue_firing = 0.0; total_clay_lbs = 0.0
                                for m in active_members:
                                    bags = rng.choice(m["clay_bags"]); revenue_clay += bags * RETAIL_CLAY_PRICE_PER_BAG
                                    clay_lbs = bags * 25; total_clay_lbs += clay_lbs
                                    revenue_firing += (20*3 + max(0, min(20, clay_lbs-20))*4 + max(0, clay_lbs-40)*5) if clay_lbs>20 else clay_lbs*3
                                
                                # Events: net = gross − materials − optional labor
                                revenue_events_gross = 0.0
                                events_cost_materials = 0.0
                                events_cost_labor = 0.0
    
                                events_this_month = int(np.clip(rng.poisson(BASE_EVENTS_PER_MONTH_LAMBDA * seasonal), 0, EVENTS_MAX_PER_MONTH))
                                for _ in range(events_this_month):
                                    attendees = int(rng.choice(ATTENDEES_PER_EVENT_RANGE))
                                    revenue_events_gross += attendees * TICKET_PRICE
    
                                    mug_unit_cost = float(rng.uniform(*EVENT_MUG_COST_RANGE))
                                    events_cost_materials += attendees * (mug_unit_cost + EVENT_CONSUMABLES_PER_PERSON)
                                    events_cost_labor += EVENT_STAFF_RATE_PER_HOUR * EVENT_HOURS_PER_EVENT
    
                                revenue_events = max(0.0, revenue_events_gross - events_cost_materials - events_cost_labor)
                                
                                net_ws = 0.0
                                
                                # Designated artist studios (stochastic monthly occupancy)
                                ds_occupied = int(rng.binomial(DESIGNATED_STUDIO_COUNT, DESIGNATED_STUDIO_BASE_OCCUPANCY)) if DESIGNATED_STUDIO_COUNT > 0 else 0
                                revenue_designated_studios = ds_occupied * DESIGNATED_STUDIO_PRICE
                                # Workshops revenue — use precomputed values from stream
                                net_ws = stream["workshop_revenue"][month]
    
                                total_revenue = (revenue_membership + revenue_clay + revenue_firing + revenue_events + net_ws + revenue_designated_studios + (0.0 if not CLASSES_ENABLED else revenue_classes))
    
                                variable_clay_cost = (total_clay_lbs / 25) * WHOLESALE_CLAY_COST_PER_BAG
                                water_cost = (total_clay_lbs / 25) * GALLONS_PER_BAG_CLAY * WATER_COST_PER_GALLON
                                
                                firings = max(MIN_FIRINGS_PER_MONTH, min(MAX_FIRINGS_PER_MONTH, round(
                                    BASE_FIRINGS_PER_MONTH * (len(active_members) / max(1, REFERENCE_MEMBERS_FOR_BASE_FIRINGS))
                                )))
    
                                # Match main sim’s kiln-2 behavior
                                if "I_all_upfront" in scen_name:
                                    kiln2_on = (len(active_members) >= EXPANSION_TRIGGER_MEMBERS) if KILN2_AFTER_THRESHOLD_SCENARIO_I else True
                                else:
                                    kiln2_on = expansion_triggered if KILN2_AFTER_EXPANSION_SCENARIO_II else True
    
                                kwh_per_firing = KWH_PER_FIRING_KMT1027 + (KWH_PER_FIRING_KMT1427 if kiln2_on else 0)
                                electricity_cost = firings * kwh_per_firing * COST_PER_KWH
                                
                                
                                monthly_heating_cost = HEATING_COST_WINTER if month % 12 in [10,11,0,1,2,3] else HEATING_COST_SUMMER
    
                                fixed_opex_profit = fixed_rent + INSURANCE_COST + GLAZE_COST_PER_MONTH + monthly_heating_cost
    
                                # Owner draw gating (same as main sim)
                                in_draw_window = in_owner_draw_window(month)
                                within_stipend_quota = (month < OWNER_STIPEND_MONTHS)
                                owner_draw_now = owner_draw if (in_draw_window and within_stipend_quota) else 0.0
    
                                # Staff after threshold
                                staff_cost = STAFF_COST_PER_MONTH if len(active_members) >= STAFF_EXPANSION_THRESHOLD else 0.0
    
                                # Maintenance (randomized, never negative)
                                maintenance_cost = MAINTENANCE_BASE_COST + max(0, rng.normal(0, MAINTENANCE_RANDOM_STD))
    
                                # Marketing ramp
                                marketing_cost = MARKETING_COST_BASE * (MARKETING_RAMP_MULTIPLIER if month < MARKETING_RAMP_MONTHS else 1.0)
    
                                # S-corp owner salary + payroll taxes (match main sim)
                                owner_salary_expense = 0.0
                                employer_payroll_tax = 0.0
                                employee_withholding = 0.0
                                if ENTITY_TYPE == "s_corp":
                                    owner_salary_expense = SCORP_OWNER_SALARY_PER_MONTH
                                    employer_payroll_tax = owner_salary_expense * EMPLOYER_PAYROLL_TAX_RATE
                                    employee_withholding = owner_salary_expense * EMPLOYEE_PAYROLL_TAX_RATE
    
                                fixed_opex_cash = fixed_opex_profit + monthly_loan_payment + owner_draw_now
                                total_opex_cash = (
                                    fixed_opex_cash
                                    + variable_clay_cost
                                    + water_cost
                                    + electricity_cost
                                    + staff_cost
                                    + maintenance_cost
                                    + marketing_cost
                                    + owner_salary_expense
                                    + employer_payroll_tax
                                    + employee_withholding
                                )
                                
                                # --- Lightweight taxes & remittances (quarterly) + sales tax + property tax ---
                                if month == 0:
                                    # accumulators that persist across months within this sim
                                    se_ss_wage_base_used_ytd = 0.0
                                    se_tax_payable_accum = 0.0
                                    state_tax_payable_accum = 0.0
                                    corp_tax_payable_accum = 0.0
                                    sales_tax_payable_accum = 0.0
                                    
                                # Reset SE wage base every January (match main sim)
                                if (month % 12) == 0 and month > 0:
                                    se_ss_wage_base_used_ytd = 0.0
    
                                # Approximate operating profit (pre-debt, pre-draw; matches main sim’s tax base)
                                op_profit_approx = (
                                    total_revenue
                                    - (fixed_opex_profit
                                       + variable_clay_cost + water_cost + electricity_cost
                                       + staff_cost + maintenance_cost + marketing_cost
                                       + owner_salary_expense + employer_payroll_tax)
                                )
    
                                se_tax_this_month = 0.0
                                state_income_tax_this_month = 0.0
                                corp_tax_this_month = 0.0
    
                                if ENTITY_TYPE in ("sole_prop", "partnership"):
                                    se_earnings = max(0.0, op_profit_approx) * SE_EARNINGS_FACTOR
                                    ss_base_remaining = max(0.0, SE_SOC_SEC_WAGE_BASE - se_ss_wage_base_used_ytd)
                                    ss_taxable_now = min(se_earnings, ss_base_remaining)
                                    se_tax_ss = ss_taxable_now * SE_SOC_SEC_RATE
                                    se_ss_wage_base_used_ytd += ss_taxable_now
                                    se_tax_medicare = se_earnings * SE_MEDICARE_RATE
                                    se_tax_this_month = se_tax_ss + se_tax_medicare
                                    se_tax_payable_accum += se_tax_this_month
    
                                    half_se_deduction = 0.5 * se_tax_this_month
                                    ma_taxable_income = max(0.0, op_profit_approx - half_se_deduction)
                                    state_income_tax_this_month = ma_taxable_income * MA_PERSONAL_INCOME_TAX_RATE
                                    state_tax_payable_accum += state_income_tax_this_month
    
                                elif ENTITY_TYPE == "s_corp":
                                    ma_taxable_income = max(0.0, op_profit_approx)
                                    state_income_tax_this_month = ma_taxable_income * MA_PERSONAL_INCOME_TAX_RATE
                                    state_tax_payable_accum += state_income_tax_this_month
    
                                elif ENTITY_TYPE == "c_corp":
                                    corp_taxable_income = max(0.0, op_profit_approx)
                                    corp_tax_this_month = corp_taxable_income * (FED_CORP_TAX_RATE + MA_CORP_TAX_RATE)
                                    corp_tax_payable_accum += corp_tax_this_month
    
                                # Sales tax on clay retail (cash collected → remit quarterly)
                                sales_tax_collected = revenue_clay * MA_SALES_TAX_RATE
                                sales_tax_payable_accum += sales_tax_collected
    
                                # Quarterly remittances (cash)
                                tax_payments_this_month = 0.0
                                if ((month + 1) % ESTIMATED_TAX_REMIT_FREQUENCY_MONTHS) == 0:
                                    if ENTITY_TYPE in ("sole_prop", "partnership", "s_corp"):
                                        tax_payments_this_month += se_tax_payable_accum
                                        tax_payments_this_month += state_tax_payable_accum
                                        se_tax_payable_accum = 0.0
                                        state_tax_payable_accum = 0.0
                                    if ENTITY_TYPE == "c_corp":
                                        tax_payments_this_month += corp_tax_payable_accum
                                        corp_tax_payable_accum = 0.0
    
                                if ((month + 1) % SALES_TAX_REMIT_FREQUENCY_MONTHS) == 0:
                                    tax_payments_this_month += sales_tax_payable_accum
                                    sales_tax_payable_accum = 0.0
    
                                # Annual personal property tax (cash only)
                                if PERSONAL_PROPERTY_TAX_ANNUAL > 0 and ((month + 1) % 12 == (PERSONAL_PROPERTY_TAX_BILL_MONTH % 12)):
                                    tax_payments_this_month += PERSONAL_PROPERTY_TAX_ANNUAL
    
                                # Add tax remittances to cash OpEx
                                total_opex_cash += tax_payments_this_month
    
                                # Month 0 loan proceeds / CapEx (match main sim; use total of 504 + 7(a))
                                if month == 0:
                                    upfront_capex = (capex_I_cost + capex_II_cost) if ("all_upfront" in scen_name) else capex_I_cost
                                    cash_balance += loan_principal_total - upfront_capex
                                cash_balance += (total_revenue - total_opex_cash)
                                if (grant_month is not None) and (month == grant_month):
                                    cash_balance += grant_amount
                                min_cash = min(min_cash, cash_balance)
    
                            rows_min.append({"scenario": scen_name, "rent": fixed_rent, "owner_draw": owner_draw, "min_cash": min_cash})
    
            df = pd.DataFrame(rows_min)
            agg = df.groupby(["scenario","rent","owner_draw"])["min_cash"].median().reset_index()
            agg["extra_buffer"] = b
            out.append(agg)
    
        EXTRA_BUFFER = baseline_extra
        return pd.concat(out, ignore_index=True)
    
    # =============================================================================
    # Business-plan dashboard (plots + lender summary)
    # =============================================================================
    def plot_business_plan_dashboard(
        results_df: pd.DataFrame,
        scenario: str,
        rent: float,
        owner_draw: float,
        membership_soft_cap: float,
        max_members: Optional[int] = None,  # ✅ fixed type hint
        dscr_target: float = 1.25
    ):
        """
        Creates a dashboard of 8 charts + a lender summary for a given configuration.
        Assumes results_df has columns created by the simulation above.
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
    
        sns.set_context("talk")
        sns.set_style("whitegrid")
    
        cfg = results_df.query(
            "scenario == @scenario and rent == @rent and owner_draw == @owner_draw"
        ).copy()
        if cfg.empty:
            print("No rows match the requested configuration.")
            return
    
        # --- Common aggregates
        months = np.arange(1, cfg["month"].max() + 1)
        # Revenue components and totals
        # Revenue components and totals
        rev_cols = [
            "revenue_membership",
            "revenue_firing",
            "revenue_clay",
            "revenue_events",
            "revenue_workshops_net",
            "revenue_designated_studios",
            "revenue_classes",  # NEW: classes (net)
        ]
        cfg["rev_total"] = cfg[rev_cols].sum(axis=1)
        cfg["opex_cash"] = cfg["rev_total"] - cfg["net_cash_flow"]
    
        # Median bands helper
        def band(series):
            g = series.groupby(cfg["month"])
            return g.median(), g.quantile(0.10), g.quantile(0.90)
    
        # --- 1) Cash runway (median + 10–90%)
        med, p10, p90 = band(cfg["cash_balance"])
        plt.figure(figsize=(10, 6))
        plt.plot(months, med.reindex(months).values, linewidth=2, label="Median")
        plt.fill_between(months, p10.reindex(months).values, p90.reindex(months).values, alpha=0.12, label="10–90%")
        # Grant markers
        grant_info = cfg[["grant_month", "grant_amount"]].drop_duplicates()
        for _, row in grant_info.iterrows():
            gm, ga = row["grant_month"], row["grant_amount"]
            if pd.notna(gm) and gm is not None and ga and ga > 0:
                plt.axvline(gm + 1, color="green", linestyle="--", linewidth=1.5)
                ymin, ymax = plt.ylim()
                plt.text(gm + 1, ymin + 0.05 * (ymax - ymin), f"Grant ${ga:,.0f}", color="green", rotation=90, va="bottom")
        plt.axhline(0, color="black", linestyle=":", linewidth=1)
        plt.title(f"Cash Balance — {scenario} | Rent ${rent:,.0f} | Draw ${owner_draw:,.0f}")
        plt.xlabel("Month"); plt.ylabel("Cash ($)")
        plt.legend(); plt.tight_layout(); plt.show()
    
        # --- 2) Cash-at-risk curve (P[cash<0] by month)
        car = cfg.groupby("month")["cash_balance"].apply(lambda s: (s < 0).mean())
        plt.figure(figsize=(10, 4.5))
        plt.plot(car.index, car.values, linewidth=2)
        plt.ylim(0, 1)
        plt.title("Cash-at-Risk: Probability(Cash < $0) by Month")
        plt.xlabel("Month"); plt.ylabel("Probability")
        plt.tight_layout(); plt.show()
    
        # --- 3) Operating break-even ECDF (P(BE <= t))
        be_by_sim = (
            cfg.sort_values(["simulation_id", "month"])
               .groupby("simulation_id")
               .apply(lambda d: d.loc[d["cumulative_op_profit"] >= 0, "month"].min())
               .dropna()
        )
        ecdf = pd.Series({t: (be_by_sim <= t).mean() for t in months})
        plt.figure(figsize=(10, 4.5))
        plt.plot(ecdf.index, ecdf.values, linewidth=2)
        if not be_by_sim.empty:
            be_median = be_by_sim.median()
            plt.axvline(be_median, linestyle="--", linewidth=1.5, color="gray")
            plt.text(be_median, 0.05, f"Median BE ≈ M{int(be_median)}", rotation=90, va="bottom", ha="right")
        plt.ylim(0, 1)
        plt.title("Operating Break-even ECDF")
        plt.xlabel("Month"); plt.ylabel("P(BE reached by month)")
        plt.tight_layout(); plt.show()
    
        # --- 4) DSCRs over time (Op‑Profit DSCR and Cash‑DSCR) with target
        dscr_op = cfg["dscr"].replace([np.inf, -np.inf], np.nan)
        dscr_cash_series = cfg["dscr_cash"].replace([np.inf, -np.inf], np.nan)
    
        op_med, op_p10, op_p90 = band(dscr_op)
        cash_med, cash_p10, cash_p90 = band(dscr_cash_series)
    
        plt.figure(figsize=(10, 4.8))
        # Op‑Profit DSCR band
        plt.plot(months, op_med.reindex(months).values, linewidth=2, label="Op‑Profit DSCR (median)")
        plt.fill_between(months, op_p10.reindex(months).values, op_p90.reindex(months).values, alpha=0.10, label="Op‑Profit DSCR (10–90%)")
        # Cash‑DSCR band
        plt.plot(months, cash_med.reindex(months).values, linewidth=2, linestyle="--", label="Cash‑DSCR (median)")
        plt.fill_between(
        months,
        cash_p10.reindex(months).values,
        cash_p90.reindex(months).values,
        alpha=0.10,
        label="Cash‑DSCR (10–90%)"
    )
        
        
        # Target line
        plt.axhline(dscr_target, linestyle="--", linewidth=1.5, label=f"Target {dscr_target:.2f}×")
        plt.title("DSCR Over Time — Operating vs Cash")
        plt.xlabel("Month"); plt.ylabel("DSCR (×)")
        plt.legend(); plt.tight_layout(); plt.show()
    
        # --- 5) Membership vs caps
        m_med, m_p10, m_p90 = band(cfg["active_members"])
        plt.figure(figsize=(10, 4.5))
        plt.plot(months, m_med.reindex(months).values, linewidth=2, label="Median")
        plt.fill_between(months, m_p10.reindex(months).values, m_p90.reindex(months).values, alpha=0.12, label="10–90%")
        plt.axhline(membership_soft_cap, linestyle="--", linewidth=1.5, label=f"Soft cap ≈ {membership_soft_cap:.0f}")
        if max_members is not None:
            plt.axhline(max_members, linestyle=":", linewidth=1.5, color="orange", label=f"Hard cap = {max_members}")
        plt.title("Active Members — Capacity View")
        plt.xlabel("Month"); plt.ylabel("Members")
        plt.legend(); plt.tight_layout(); plt.show()
    
        # --- 6) Revenue mix (median) vs OpEx (cash) & Net Cash Flow
        g_rev = cfg.groupby("month")[rev_cols].median()
        g_ops = cfg.groupby("month")[["rev_total", "opex_cash", "net_cash_flow"]].median()
        plt.figure(figsize=(11, 5.5))
        plt.stackplot(
            g_rev.index,
            g_rev["revenue_membership"],
            g_rev["revenue_firing"],
            g_rev["revenue_clay"],
            g_rev["revenue_events"],
            g_rev["revenue_workshops_net"],
            g_rev["revenue_designated_studios"],
            g_rev["revenue_classes"],  # NEW
            labels=[
                "Membership", "Firing", "Clay", "Events",
                "Workshops (net)", "Designated studios", "Classes (net)"  # NEW
            ],
            alpha=0.9,
        )
        
        plt.plot(g_ops.index, g_ops["opex_cash"], linewidth=2.0, label="Total OpEx (cash)")
        plt.plot(g_ops.index, g_ops["net_cash_flow"], linestyle="--", linewidth=1.5, label="Net cash flow")
        plt.title("Revenue Composition vs Cash OpEx")
        plt.xlabel("Month"); plt.ylabel("Dollars ($)")
        plt.legend(loc="upper center", ncol=3)
        plt.tight_layout(); plt.show()
    
        # --- 7) Unit metrics: Revenue per Member & Net CF per Member (median)
        unit = cfg.copy()
        unit["rev_per_member"] = unit["rev_total"] / unit["active_members"].clip(lower=1)
        unit["ncf_per_member"] = unit["net_cash_flow"] / unit["active_members"].clip(lower=1)
        up = unit.groupby("month")[["rev_per_member", "ncf_per_member"]].median()
        plt.figure(figsize=(10, 4.5))
        plt.plot(up.index, up["rev_per_member"], linewidth=2, label="Revenue / Member / Month")
        plt.plot(up.index, up["ncf_per_member"], linewidth=2, linestyle="--", label="Net Cash Flow / Member / Month")
        plt.title("Unit Economics (Median)")
        plt.xlabel("Month"); plt.ylabel("$/Member/Month")
        plt.legend(); plt.tight_layout(); plt.show()
    
        # --- 8) Stress lens: Min cash distribution across simulations
        min_cash_by_sim = cfg.groupby("simulation_id")["cash_balance"].min()
        plt.figure(figsize=(10, 4.5))
        bins = max(10, min(40, int(np.sqrt(len(min_cash_by_sim)))))
        plt.hist(min_cash_by_sim.values, bins=bins, alpha=0.6)
        plt.axvline(min_cash_by_sim.median(), color="black", linestyle="--", linewidth=1.5, label=f"Median = ${min_cash_by_sim.median():,.0f}")
        plt.title("Minimum Cash Across Simulations")
        plt.xlabel("Minimum Cash over 60 Months ($)"); plt.ylabel("Count")
        plt.legend(); plt.tight_layout(); plt.show()
    
        # --- Lender summary (concise)
        # Key stats at months 12 and 24
        def pct(x): return f"{100*x:.0f}%"
        car_12 = (cfg.loc[cfg["month"] == 12, "cash_balance"] < 0).mean() if (cfg["month"] == 12).any() else np.nan
        car_24 = (cfg.loc[cfg["month"] == 24, "cash_balance"] < 0).mean() if (cfg["month"] == 24).any() else np.nan
        dscr_12 = cfg.loc[cfg["month"] == 12, "dscr"].median() if (cfg["month"] == 12).any() else np.nan
        dscr_24 = cfg.loc[cfg["month"] == 24, "dscr"].median() if (cfg["month"] == 24).any() else np.nan
        dscr_cash_12 = cfg.loc[cfg["month"] == 12, "dscr_cash"].median() if (cfg["month"] == 12).any() else np.nan
        dscr_cash_24 = cfg.loc[cfg["month"] == 24, "dscr_cash"].median() if (cfg["month"] == 24).any() else np.nan
        be_m = be_by_sim.median() if not be_by_sim.empty else np.nan
        insol_before_grant = cfg.groupby("simulation_id")["insolvent_before_grant"].max().mean()
    
        # Owner take-home (if table exists)
        try:
            oth = summarize_owner_takehome(results_df)
            oth_row = oth.query("scenario == @scenario and rent == @rent and owner_draw == @owner_draw")
            owner_takehome_median = float(oth_row["median_owner_takehome_annual"].iloc[0]) if not oth_row.empty else np.nan
        except Exception:
            owner_takehome_median = np.nan
    
        lender_summary = pd.DataFrame({
            "Scenario": [scenario],
            "Rent ($/mo)": [f"{rent:,.0f}"],
            "Owner draw ($/mo)": [f"{owner_draw:,.0f}"],
            "Median BE month": [None if pd.isna(be_m) else int(be_m)],
            "CAR @12": [None if pd.isna(car_12) else pct(car_12)],
            "CAR @24": [None if pd.isna(car_24) else pct(car_24)],
            "DSCR @12 (Op‑Profit, p50)": [None if pd.isna(dscr_12) else f"{dscr_12:.2f}x"],
            "DSCR @24 (Op‑Profit, p50)": [None if pd.isna(dscr_24) else f"{dscr_24:.2f}x"],
            "DSCR @12 (Cash, p50)": [None if pd.isna(dscr_cash_12) else f"{dscr_cash_12:.2f}x"],
            "DSCR @24 (Cash, p50)": [None if pd.isna(dscr_cash_24) else f"{dscr_cash_24:.2f}x"],
            "Median min cash ($)": [f"{min_cash_by_sim.median():,.0f}"],
            "Median cash @M60 ($)": [f"{cfg.loc[cfg['month']==cfg['month'].max(), 'cash_balance'].median():,.0f}"],
            "% sims insolvent pre-grant": [pct(insol_before_grant)],
            "Owner take-home (annual, median)": [None if pd.isna(owner_takehome_median) else f"${owner_takehome_median:,.0f}"],
        })
        print("\n=== Lender Summary (focus configuration) ===")
        print(lender_summary.to_string(index=False))
    
    
    # ---- Example call (choose one configuration to render) ----
    # Update these three to the combo you want to present:
    scenario_focus   = "II_staged"
    rent_focus       = 3500
    owner_draw_focus = 0
    
    # Try to pass MAX_MEMBERS if defined, else None
    try:
        max_members_focus = int(MAX_MEMBERS)
    except Exception:
        max_members_focus = None
    
    plot_business_plan_dashboard(
        results_df=results_df,
        scenario=scenario_focus,
        rent=rent_focus,
        owner_draw=owner_draw_focus,
        membership_soft_cap=MEMBERSHIP_SOFT_CAP,
        max_members=max_members_focus,
        dscr_target=1.25
    )
    # At the very end, return the artifacts you care about:
    return {
        "results_df": results_df,
        "summary_table": summary_table,
        "owner_takehome_table": owner_takehome_table
    }


def run_from_cfg(cfg: dict | None = None):
    """
    Public entrypoint for batch/adapter use.
    - Merges cfg with defaults captured from this file
    - Temporarily overrides module constants
    - Runs your original code path unchanged
    - Restores globals afterward
    Returns a dict with DataFrames, and your existing side‑effects (plots/CSVs/prints) still happen.
    """
    merged = resolve_cfg(cfg)

    # Inject downturn prob (unless caller pinned it)
    p, p_src = _get_downturn_prob(merged)
    merged.setdefault("DOWNTURN_PROB_PER_MONTH", p)
    print(f"[nowcast] DOWNTURN_PROB_PER_MONTH = {merged['DOWNTURN_PROB_PER_MONTH']:.3f}  (source={p_src})")

    with override_globals(merged):
        # If any helpers previously read globals, they still will—now pointed at merged.
        artifacts = _core_simulation_and_reports()
    
    # OPTIONAL: stamp the value/source into returned tables for traceability
    try:
        artifacts["summary_table"]["downturn_prob_per_month"] = merged["DOWNTURN_PROB_PER_MONTH"]
        artifacts["summary_table"]["downturn_source"] = p_src
    except Exception:
        pass


    return artifacts

if __name__ == "__main__":
    # Running the file directly behaves exactly as before:
    run_from_cfg({})
