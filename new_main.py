#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 16:43:44 2025

@author: harshadghodke
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def main():
    """
    Scenario × Strategy matrix runner for the GCWS simulator.
    Exports per-cell plots (zipped), raw results, summaries, and heatmaps.
    """
    # ------------------------ Imports ------------------------
    import json, re, shutil
    from datetime import datetime
    from pathlib import Path
    from contextlib import contextmanager
    from typing import List, Optional

    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns  # required by heatmap_from

    # IMPORTANT: adapter must expose run_original_once(script_path, overrides_dict)
    from final_batch_adapter import run_original_once

    # ------------------------ Config ------------------------
    SCRIPT = "modular_simulator.py"   # core simulator file

    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_BASE = Path("runs")
    OUTDIR = OUTPUT_BASE / f"{RUN_ID}_matrix"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"[info] Writing outputs to: {OUTDIR.resolve()}")

    # ------------------------ Helpers ------------------------
    def parquet_engine_available() -> bool:
        try:
            import pyarrow  # noqa: F401
            return True
        except Exception:
            try:
                import fastparquet  # noqa: F401
                return True
            except Exception:
                return False

    HAVE_PARQUET = parquet_engine_available()
    if not HAVE_PARQUET:
        print("[warn] No parquet engine found (pyarrow/fastparquet). Will skip .parquet and write CSV only.")

    def to_parquet_safe(df: pd.DataFrame, path: Path):
        if HAVE_PARQUET:
            try:
                df.to_parquet(path, index=False)
                print(f"[info] Saved parquet: {path}")
                return True
            except Exception as e:
                print(f"[warn] Parquet save failed ({e}). Skipping parquet for {path.name}.")
        return False

    def slug(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"\s+", "-", s)
        s = re.sub(r"[^a-z0-9._-]+", "", s)
        return s[:80]

    def sanitize_text(s: str) -> str:
        # Replace NB hyphen and other odd dashes/soft hyphens with a plain hyphen
        return re.sub(r"[\u2010-\u2015\u2212\u00AD\u200B\uFEFF\u202F]", "-", s)

    def build_overrides(env: dict, strat: dict) -> dict:
        """Build init_globals for one (environment, strategy) cell."""
        ov = {}
        # Environment globals
        for k, v in env.items():
            if k not in {"name", "grant_amount", "grant_month"}:
                ov[k] = v
        # Strategy globals
        for k, v in strat.items():
            if k != "name":
                ov[k] = v
        # Ensure singletons for rent/draw sweeps
        if "RENT" in strat:
            ov["RENT_SCENARIOS"] = np.array([float(strat["RENT"])], dtype=float)
        if "OWNER_DRAW" in strat:
            ov["OWNER_DRAW_SCENARIOS"] = [float(strat["OWNER_DRAW"])]
        # Single scenario entry inside the simulator
        ov["SCENARIO_CONFIGS"] = [{
            "name": strat["name"],
            "capex_timing": "all" if ("all_upfront" in strat["name"]) else "staged",
            "grant_amount": env.get("grant_amount", 0.0),
            "grant_month": env.get("grant_month", None),
        }]
        return ov


    @contextmanager
    def capture_matplotlib(outdir, *, title_suffix=None, filename_prefix=None):
        """
        Redirects plt.show() to save numbered PNGs into `outdir`.
        - If `title_suffix` is provided, it's appended to the figure's suptitle
          (or used as title if none exists):  "<existing> — <title_suffix>".
        - If `filename_prefix` is provided, files are saved as
          "<filename_prefix>__fig_XX.png".
        Also writes a manifest.json (filename + final title).
        """
        outdir.mkdir(parents=True, exist_ok=True)
        matplotlib.use("Agg", force=True)
        orig_show = plt.show
        manifest = []
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
            if not title_suffix:
                return
            current = fig._suptitle.get_text() if fig._suptitle else ""
            # Grab the first Axes title if no suptitle exists
            ax_titles = [ax.get_title() for ax in fig.axes if hasattr(ax, "get_title")]
            base_title = current or (ax_titles[0] if ax_titles else "")
            # Combine into one line
            combined = f"{base_title} — {title_suffix}" if base_title else title_suffix
            fig.suptitle(combined)
            # Clear axes titles so they don’t clash
            for ax in fig.axes:
                ax.set_title("")
    
        def _show(*args, **kwargs):
            counter["i"] += 1
            fig = plt.gcf()
    
            # Ensure scenario/strategy is in the title
            _ensure_suffix(fig)
    
            # Build filename
            base = f"fig_{counter['i']:02d}.png"
            if filename_prefix:
                base = f"{filename_prefix}__{base}"
            fpath = outdir / base
    
            fig.savefig(fpath, dpi=200, bbox_inches="tight")
    
            manifest.append({"file": base, "title": _title_for(fig)})
            plt.close(fig)
    
        plt.show = _show
        try:
            yield manifest
        finally:
            try:
                (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
            except Exception:
                pass
            plt.show = orig_show
            

    # --- Seaborn heatmap ---
    def heatmap_from(
        df, value_col, title, out_fname,
        row_order=None, col_order=None,
        *, value_transform=None, fmt=".0f",
        cmap="YlOrBr", vmin=None, vmax=None, cbar_label=None
    ):
        # Pivot and order
        idx  = row_order or sorted(df["environment"].unique())
        cols = col_order or sorted(df["strategy"].unique())
        pivot = (df.pivot(index="environment", columns="strategy", values=value_col)
                   .reindex(index=idx, columns=cols))
        if value_transform is not None:
            pivot = pivot.applymap(lambda x: np.nan if not np.isfinite(x) else value_transform(x))

        # Figure size scales with matrix size
        W = max(1.5 * len(cols) + 3, 6)
        H = max(0.9 * len(idx)  + 2, 4.5)
        fig, ax = plt.subplots(figsize=(W, H))

        sns.heatmap(
            pivot, annot=True, fmt=fmt,
            cmap=cmap, vmin=vmin, vmax=vmax,
            linewidths=0.5, linecolor="white",
            cbar=True, cbar_kws={"label": cbar_label} if cbar_label else None,
            annot_kws={"size": 10}
        )
        ax.set_title(sanitize_text(title), fontsize=14, pad=12)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_xticklabels([sanitize_text(c) for c in cols], rotation=30, ha="right")
        ax.set_yticklabels([sanitize_text(r) for r in idx], rotation=0)
        fig.tight_layout()

        outpath = OUTDIR / out_fname
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[info] Saved heatmap: {outpath}")
        return fig
    
    # ------------------------ Lender stress test helper ------------------------
    def make_lender_stress_test(results, matrix, timings, T, outdir: Path):
        """
        Build a lender-facing table by merging lender-relevant risk metrics
        onto your existing summary `matrix`.
        Requires:
          - results: long per-month DataFrame with columns
                     ['environment','strategy','simulation_id','month','cash_balance','dscr', ...]
          - matrix : your base summary (survival_prob, cash/DSCR quantiles, timing medians already merged)
          - timings: DataFrame with per-simulation 't_insolvency' and 't_breakeven'
          - T: final horizon month (int)
        Produces: lender_stress_test_table.csv
        """
        import numpy as np

    
        base_keys = ["environment", "strategy"]
        keys_sim  = ["environment", "strategy", "simulation_id"]
    
        # --- Column pickers (robust to naming drift) ---
        def _pick_col(df, candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None
    
        cash_col = _pick_col(results, ["cash_balance","cash","ending_cash"])
        dscr_col = _pick_col(results, ["dscr","DSCR"])
        if cash_col is None or dscr_col is None:
            raise RuntimeError("Missing required columns for lender table (cash_balance/dscr).")
    
       # --- 1) Covenant breach probabilities across the horizon ---
        grp = (results
               .sort_values(keys_sim + ["month"])
               .groupby(keys_sim)[dscr_col])
        
        dscr_flags = (grp.agg(
            any_below_1_25=lambda s: bool((s < 1.25).any()),
            any_below_1_00=lambda s: bool((s < 1.00).any()),
            pct_months_below_1_25=lambda s: float((s < 1.25).mean()),
            pct_months_below_1_00=lambda s: float((s < 1.00).mean()),
        ).reset_index())
        
        covenant_prob = (dscr_flags
            .groupby(base_keys, as_index=False)
            .agg(
                prob_any_dscr_below_1_25=("any_below_1_25", "mean"),
                prob_any_dscr_below_1_00=("any_below_1_00", "mean"),
                median_pct_months_below_1_25=("pct_months_below_1_25", "median"),
                median_pct_months_below_1_00=("pct_months_below_1_00", "median"),
            ))
    
        # --- 2) Min‑cash cushion distribution (per sim -> p10 across sims) ---
        min_cash_per_sim = (results.groupby(keys_sim)[cash_col].min()
                            .reset_index(name="min_cash"))
        min_cash_p10 = (min_cash_per_sim.groupby(base_keys)["min_cash"]
                        .quantile(0.10).reset_index(name="min_cash_p10"))
    
        # --- 3) Timing quantiles from `timings` (already has t_insolvency / t_breakeven) ---
        def _q(s, q):
            s = s.dropna()
            return float(np.nan) if s.empty else float(np.nanquantile(s, q))
    
        timing_q = (timings.groupby(base_keys)
                    .agg(
                        t_insolvency_p10=("t_insolvency", lambda s: _q(s, 0.10)),
                        t_insolvency_p50=("t_insolvency", lambda s: _q(s, 0.50)),
                        t_insolvency_p90=("t_insolvency", lambda s: _q(s, 0.90)),
                        t_breakeven_p10=("t_breakeven",   lambda s: _q(s, 0.10)),
                        t_breakeven_p50=("t_breakeven",   lambda s: _q(s, 0.50)),
                        t_breakeven_p90=("t_breakeven",   lambda s: _q(s, 0.90)),
                    )
                    .reset_index())
    
        # --- 4) DSCR medians at 12/24/36 months (if available) ---
        dscr_horizons = []
        for m in (12, 24, 36):
            if results["month"].max() >= m:
                dscr_m = (results[results["month"] == m]
                          .groupby(base_keys)[dscr_col]
                          .median()
                          .reset_index(name=f"dscr_m{m}_med"))
                dscr_horizons.append(dscr_m)
    
        # --- Assemble lender table by merging onto your existing `matrix` ---
        lender_table = (matrix
            .merge(covenant_prob, on=base_keys, how="left")
            .merge(min_cash_p10,  on=base_keys, how="left")
            .merge(timing_q,      on=base_keys, how="left"))
        # Conservative headline columns for lenders
        lender_table["cash_conservative"] = lender_table["cash_q10"]
        lender_table["dscr_conservative"] = lender_table["dscr_q10"]
        lender_table["runway_p10_months"] = lender_table["t_insolvency_p10"]  # downside runway
        lender_table["breakeven_p90_months"] = lender_table["t_breakeven_p90"] # slow-case breakeven
    
        for ds in dscr_horizons:
            lender_table = lender_table.merge(ds, on=base_keys, how="left")
    
        # --- Persist ---
        lender_csv = outdir / "lender_stress_test_table.csv"
        lender_table.to_csv(lender_csv, index=False)
        print(f"[info] Saved lender stress test table: {lender_csv}")
    
        return lender_table

    # ------------------------ Define SCENARIOS (external) ------------------------
    SCENARIOS = [
        {
            "name": "Baseline",
            "DOWNTURN_PROB_PER_MONTH": 0.05,
            "DOWNTURN_JOIN_MULT": 1.0,
            "DOWNTURN_CHURN_MULT": 1.0,
            "MARKET_POOLS_INFLOW": {"community_studio": 4, "home_studio": 2, "no_access": 3},
            "grant_amount": 0.0,
            "grant_month": None,
        },
        {
            "name": "Recession",
            "DOWNTURN_PROB_PER_MONTH": 0.18,
            "DOWNTURN_JOIN_MULT": 0.65,
            "DOWNTURN_CHURN_MULT": 1.50,
            "MARKET_POOLS_INFLOW": {"community_studio": 2, "home_studio": 1, "no_access": 1},
            "grant_amount": 0.0,
            "grant_month": None,
        },
        {
            "name": "SlowRecovery_Grant25k_M4",
            "DOWNTURN_PROB_PER_MONTH": 0.10,
            "DOWNTURN_JOIN_MULT": 0.85,
            "DOWNTURN_CHURN_MULT": 1.20,
            "MARKET_POOLS_INFLOW": {"community_studio": 3, "home_studio": 1, "no_access": 2},
            "grant_amount": 25_000,
            "grant_month": 4,
        },
        {
            "name": "Boom",
            "DOWNTURN_PROB_PER_MONTH": 0.02,
            "DOWNTURN_JOIN_MULT": 1.20,
            "DOWNTURN_CHURN_MULT": 0.85,
            "MARKET_POOLS_INFLOW": {"community_studio": 6, "home_studio": 3, "no_access": 4},
            "grant_amount": 0.0,
            "grant_month": None,
        },
        {
            "name": "GreatDepression",
            "DOWNTURN_PROB_PER_MONTH": 0.35,      # persistent, very high shock rate
            "DOWNTURN_JOIN_MULT": 0.35,           # joins cut in half
            "DOWNTURN_CHURN_MULT": 2.5,          # churn nearly doubles
            "MARKET_POOLS_INFLOW": {"community_studio": 1, "home_studio": 0, "no_access": 1},  # dried-up inflow
            "grant_amount": 0.0,
            "grant_month": None,
        }
    ]
    (OUTDIR / "scenarios.json").write_text(json.dumps(SCENARIOS, indent=2))

    # ------------------------ Define STRATEGIES (internal) ------------------------
    STRATEGIES = [
        {"name": "I_all_upfront_Base", "RENT": 3500, "OWNER_DRAW": 1000},
        {"name": "II_staged_Base",     "RENT": 3500, "OWNER_DRAW": 1000},
    ]
    (OUTDIR / "strategies.json").write_text(json.dumps(STRATEGIES, indent=2))

    # ------------------------ Run matrix ------------------------
    all_results = []
    BASE_SEED = 42
    n_strats = len(STRATEGIES)

    plots_root = OUTDIR / "plots"
    zips_root  = OUTDIR / "cell_artifacts"
    plots_root.mkdir(parents=True, exist_ok=True)
    zips_root.mkdir(parents=True, exist_ok=True)

    for i, env in enumerate(SCENARIOS):
        for j, strat in enumerate(STRATEGIES):
            ov = build_overrides(env, strat)
            ov["RANDOM_SEED"] = BASE_SEED + 1000 * (i * n_strats + j)

            cell_slug     = f"{slug(env['name'])}__{slug(strat['name'])}"
            cell_plot_dir = plots_root / cell_slug
            
            with capture_matplotlib(
                cell_plot_dir,
                title_suffix=f"{env['name']} | {strat['name']}",
                filename_prefix=cell_slug
            ) as manifest:
                res = run_original_once(SCRIPT, ov)

            # Zip plots (even if empty)
            shutil.make_archive(str(zips_root / cell_slug), "zip", root_dir=cell_plot_dir)

            # Small README for the folder
            (cell_plot_dir / "README.txt").write_text(
                f"Environment: {env['name']}\nStrategy: {strat['name']}\nFigures captured: {len(manifest)}\n"
            )

            # Label/stash results
            df_cell, eff = (res if isinstance(res, tuple) else (res, None))
            if df_cell is None:
                raise RuntimeError(
                    f"No DataFrame returned for env={env['name']} strat={strat['name']}."
                    " Ensure RESULTS_DF is set or adapter finds results_df."
                )
            df_cell = df_cell.copy()
            df_cell["environment"] = env["name"]
            df_cell["strategy"]    = strat["name"]
            all_results.append(df_cell)

    if not all_results:
        raise RuntimeError("No results generated. Check SCENARIOS/STRATEGIES and adapter.")

    results = pd.concat(all_results, ignore_index=True)

    # ------------------------ Save raw results ------------------------
    raw_csv = OUTDIR / "results_raw.csv"
    results.to_csv(raw_csv, index=False)
    print(f"[info] Saved CSV: {raw_csv}")
    to_parquet_safe(results, OUTDIR / "results_raw.parquet")

    # Per-cell partitions
    cell_dir = OUTDIR / "cells"
    cell_dir.mkdir(exist_ok=True)
    for (env_name, strat_name), g in results.groupby(["environment", "strategy"], sort=False):
        sub = g.sort_values(["simulation_id", "month"])
        fn_csv = cell_dir / f"{slug(env_name)}__{slug(strat_name)}.csv"
        sub.to_csv(fn_csv, index=False)
        to_parquet_safe(sub, cell_dir / f"{slug(env_name)}__{slug(strat_name)}.parquet")
    print(f"[info] Wrote per-cell distributions to: {cell_dir}")

    # ------------------------ Summaries (base) ------------------------
    T = int(results["month"].max())
    base_keys = ["environment", "strategy"]

    # survival = (results.groupby(base_keys + ["simulation_id"])["insolvent_before_grant"].max()
    #                  .groupby(base_keys)
    #                  .apply(lambda s: 1.0 - float(s.mean()))
    #                  .reset_index(name="survival_prob"))

    ever_neg = (
        results.groupby(base_keys + ["simulation_id"])["cash_balance"]
               .min()
               .lt(0)  # True if ever below 0
               .groupby(level=[0,1])
               .mean()
               .reset_index(name="prob_insolvent_by_T")
    )
    
    survival = ever_neg.assign(survival_prob=lambda d: 1.0 - d["prob_insolvent_by_T"])[["environment","strategy","survival_prob"]]

    last   = results[results["month"] == T]
    cash_q = (last.groupby(base_keys)["cash_balance"]
                   .quantile([0.10, 0.50, 0.90])
                   .unstack()
                   .reset_index()
                   .rename(columns={0.10: "cash_q10", 0.50: "cash_med", 0.90: "cash_q90"}))

    m_dscr = 12 if (results["month"].max() >= 12) else T
    dscr_q = (results[results["month"] == m_dscr]
                  .groupby(base_keys)["dscr"]
                  .quantile([0.10, 0.50, 0.90])
                  .unstack()
                  .reset_index()
                  .rename(columns={0.10: "dscr_q10", 0.50: "dscr_med", 0.90: "dscr_q90"}))

    matrix = (survival
              .merge(cash_q, on=base_keys, how="outer")
              .merge(dscr_q, on=base_keys, how="outer"))

    # ------------------------ Lender timing metrics ------------------------
    CF_CANDIDATES   = ["cfads", "operating_cash_flow", "op_cf", "net_cash_flow", "cash_flow"]
    CASH_CANDIDATES = ["cash_balance", "cash", "ending_cash"]

    def _first_month_ge_zero(s: pd.Series) -> float:
        idx = s.index[s.values >= 0]
        return float(idx.min()) if len(idx) else np.nan

    def _first_month_cash_negative(s: pd.Series) -> float:
        idx = s.index[s.values < 0]
        return float(idx.min()) if len(idx) else np.nan
    
    def first_sustained_ge_zero(s, k=3):
        # first month where *k consecutive* months are >= 0
        s = s.sort_index()
        ok = (s >= 0).astype(int).rolling(k, min_periods=k).sum() == k
        idx = ok[ok].index
        return float(idx.min()) if len(idx) else np.nan

    def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    keys_sim = ["environment", "strategy", "simulation_id"]

    cash_col = _pick_col(results, CASH_CANDIDATES)
    cf_col   = _pick_col(results, CF_CANDIDATES)

    if cash_col is None:
        raise RuntimeError(f"Could not find a cash balance column ({CASH_CANDIDATES}) in results.")
    if cf_col is None:
        results["_fallback_cf"] = (results
            .sort_values(keys_sim + ["month"])
            .groupby(keys_sim)[cash_col].diff().fillna(0.0))
        cf_col = "_fallback_cf"

    grp = results.sort_values(keys_sim + ["month"]).groupby(keys_sim)

    first_insolvency = (grp.apply(lambda g: _first_month_cash_negative(g.set_index("month")[cash_col]))
                          .reset_index(name="t_insolvency"))

    first_breakeven  = (grp.apply(lambda g: first_sustained_ge_zero(g.set_index("month")[cf_col]))
                          .reset_index(name="t_breakeven"))

    timings = first_insolvency.merge(first_breakeven, on=keys_sim, how="outer")

    def _median_among_valid(s: pd.Series) -> float:
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        return float(s.median()) if len(s) else np.nan

    tim_summary = (timings.groupby(base_keys).agg(
        median_time_to_insolvency_months=("t_insolvency", _median_among_valid),
        share_insolvent_by_T=("t_insolvency", lambda s: float(np.mean(~s.isna()))),
        median_time_to_breakeven_months=("t_breakeven", _median_among_valid)
    ).reset_index())

    # Merge timings into the base matrix
    matrix = matrix.merge(tim_summary, on=base_keys, how="left")
    
    # Build and save the lender stress test table
    lender_table = make_lender_stress_test(results, matrix, timings, T, OUTDIR)

    # ------------------------ Save summary & heatmaps ------------------------
    summary_csv = OUTDIR / "matrix_summary.csv"
    matrix.to_csv(summary_csv, index=False)
    print(f"[info] Saved summary matrix: {summary_csv}")
    to_parquet_safe(matrix, OUTDIR / "matrix_summary.parquet")

    pd.set_option("display.max_columns", 999)
    print("\n=== Scenario × Strategy matrix (medians, 10–90% bands, timings) ===")
    print(matrix.sort_values(base_keys).to_string(index=False))

    envs   = sorted(matrix["environment"].unique())
    strats = sorted(matrix["strategy"].unique())

    # Survival prob (0–1 scale)
    heatmap_from(
        matrix, "survival_prob",
        f"Survival probability (pre‑grant) at M{T}",
        f"heatmap_survival_M{T}.png",
        row_order=envs, col_order=strats,
        fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar_label="probability"
    )

    # Median cash (in $k)
    heatmap_from(
        matrix, "cash_med",
        f"Median cash at M{T} ($k)",
        f"heatmap_cash_med_M{T}.png",
        row_order=envs, col_order=strats,
        value_transform=lambda x: x/1000.0,
        fmt=".0f", cmap="YlOrBr", cbar_label="$ thousands"
    )
    # 3) Median time to insolvency (months)
    heatmap_from(
        matrix, "median_time_to_insolvency_months",
        f"Median time to insolvency (months) — horizon M{T}",
        f"heatmap_t_insolvency_M{T}.png",
        row_order=envs, col_order=strats,
        fmt=".0f", cmap="Reds", cbar_label="months"
    )
    
    # 4) Median time to breakeven (months)
    heatmap_from(
        matrix, "median_time_to_breakeven_months",
        f"Median time to breakeven (months) — horizon M{T}",
        f"heatmap_t_breakeven_M{T}.png",
        row_order=envs, col_order=strats,
        fmt=".0f", cmap="Greens", cbar_label="months"
    )

    # ------------------------ README ------------------------
    readme = f"""Ginkgo Clayworks Studio — Scenario × Strategy matrix
Run ID: {RUN_ID}
Script: {SCRIPT}
Rows = scenarios (external environment)
Cols = strategies (internal choices)

Artifacts:
- results_raw.csv (+ .parquet if engine available)
- matrix_summary.csv (+ .parquet if engine available)
- heatmap_survival_M{T}.png
- heatmap_cash_med_M{T}.png
- plots/<env>__<strategy>/ (PNGs + manifest.json) and cell_artifacts/*.zip
- cells/*.csv (+ .parquet if engine available)

Reproduce:
- See scenarios.json and strategies.json for the exact inputs.
"""
    (OUTDIR / "README.txt").write_text(readme)
    print(f"[info] Wrote README: {OUTDIR / 'README.txt'}")

if __name__ == "__main__":
    main()