Ginkgo Clayworks Studio — Scenario × Strategy matrix
Run ID: 20250818_220840
Script: modular_simulator.py
Rows = scenarios (external environment)
Cols = strategies (internal choices)

Artifacts:
- results_raw.csv (+ .parquet if engine available)
- matrix_summary.csv (+ .parquet if engine available)
- heatmap_survival_M60.png
- heatmap_cash_med_M60.png
- plots/<env>__<strategy>/ (PNGs + manifest.json) and cell_artifacts/*.zip
- cells/*.csv (+ .parquet if engine available)

Reproduce:
- See scenarios.json and strategies.json for the exact inputs.
