"""Export MLflow sweep runs (params and metrics) to CSV reports."""

import mlflow
from pathlib import Path

Path("reports").mkdir(exist_ok=True)

EXPERIMENTS = ["climbing_backend_sweep", "climbing_heuristic_tuning"]

for exp_name in EXPERIMENTS:
    runs = mlflow.search_runs(experiment_names=[exp_name])
    if runs.empty:
        print(f"No runs found for {exp_name} — did you run the sweep script first?")
        continue
    cols = [c for c in runs.columns if c.startswith(("params.", "metrics.")) or c == "run_id"]
    out = runs[cols].sort_values(by=cols[-1] if cols else "run_id")
    csv_path = f"reports/{exp_name}.csv"
    out.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(out)} runs)")