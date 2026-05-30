"""
Regenerate every paper figure and LaTeX table from results/tables/all_runs.csv.
Run after any new experiment completes to refresh figures.

Usage:
    python scripts/make_paper_figures.py [--wide] [--results_dir results]
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Regenerate all paper figures")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--wide", action="store_true", help="Double-column figure width")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    csv_path = str(results_dir / "tables" / "all_runs.csv")
    figures_dir = str(results_dir / "figures")
    tables_dir = str(results_dir / "tables")
    conf_path = "data/labels/confidence.json"

    from src.viz.plots import (
        fig_failure_repro, fig_exp1_sasnet, fig_exp2_ablation,
        fig_exp3_loro, fig_master_table, fig_pr_curves, fig_confidence_strata,
    )

    fns = [
        (fig_failure_repro, {"csv_path": csv_path}),
        (fig_exp1_sasnet,   {"csv_path": csv_path}),
        (fig_exp2_ablation, {"csv_path": csv_path}),
        (fig_exp3_loro,     {"csv_path": csv_path}),
        (fig_master_table,  {"csv_path": csv_path}),
        (fig_pr_curves,     {"csv_path": csv_path}),
    ]

    if Path(conf_path).exists():
        fns.append((fig_confidence_strata, {"confidence_json": conf_path}))
    else:
        print(f"Skipping fig_confidence_strata: {conf_path} not found")

    for fn, kwargs in fns:
        try:
            fn(output_dir=figures_dir, wide=args.wide, **kwargs)
            print(f"Generated: {fn.__name__}")
        except FileNotFoundError as e:
            print(f"Skipped {fn.__name__}: {e}")
        except Exception as e:
            print(f"Error in {fn.__name__}: {e}")

    # Generate LaTeX table
    _write_latex_tables(csv_path, tables_dir)
    # Update RESULTS.md
    _write_results_md(csv_path, "docs/RESULTS.md")

    print("\nAll figures and tables regenerated.")


def _write_latex_tables(csv_path: str, tables_dir: str):
    import pandas as pd
    if not Path(csv_path).exists():
        return
    df = pd.read_csv(csv_path)
    cols = ["run_id", "backbone_config", "metric_hc_iou", "metric_precision",
            "metric_recall", "metric_f1", "metric_fpr_control_old_dhaka"]
    avail = [c for c in cols if c in df.columns]
    latex = df[avail].round(3).to_latex(index=False, na_rep="--")
    out = Path(tables_dir) / "master_table.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex)
    print(f"LaTeX table written: {out}")


def _write_results_md(csv_path: str, md_path: str):
    import pandas as pd
    if not Path(csv_path).exists():
        return
    df = pd.read_csv(csv_path)
    cols = ["run_id", "backbone_config", "metric_hc_iou", "metric_precision",
            "metric_recall", "metric_f1", "metric_fpr_control_old_dhaka"]
    avail = [c for c in cols if c in df.columns]
    table = df[avail].round(3).to_markdown(index=False)
    content = f"# Results\n\n_Auto-updated by scripts/make_paper_figures.py_\n\n{table}\n"
    Path(md_path).parent.mkdir(parents=True, exist_ok=True)
    Path(md_path).write_text(content)
    print(f"RESULTS.md updated: {md_path}")


if __name__ == "__main__":
    main()
