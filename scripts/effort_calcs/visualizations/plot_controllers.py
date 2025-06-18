#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")


def extract_controller_and_spring(experiment_name: str) -> tuple[str, str]:
    base = experiment_name.replace("controllers_", "")
    if "_springless" in base:
        return base.replace("_springless", ""), "springless"
    elif "_spring" in base:
        return base.replace("_spring", ""), "spring"
    else:
        return base, "unknown"


def load_data(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"No file at {str(file_path)}")

    with file_path.open("r") as f:
        data = json.load(f)

    include_metrics = {"Delta_H_y_u", "Delta_H_u", "Delta_H_y"}
    rows = []
    for entry in data:
        controller, spring = extract_controller_and_spring(entry["experiment_name"])
        for key, value in entry.items():
            if key != "experiment_name" and key in include_metrics:
                rows.append({
                    "controller": controller,
                    "spring": spring,
                    "metric": key,
                    "value": value,
                })
    return rows


def main():
    # Define all JSON files to include here
    result_files = [
        "experiment_controllers_springless.json",
        "experiment_controllers_spring.json",
    ]
    base = "/home/andrew/Documents/RoboTUM_Workspace/IsaacDynamics/report_results/"

    all_rows = []
    for file in result_files:
        all_rows.extend(load_data(Path(base + file)))

    df = pd.DataFrame(all_rows)

    # Plot using FacetGrid to split by metric
    g = sns.FacetGrid(df, col="metric", sharey=False, height=6, aspect=1.2)
    g.map_dataframe(sns.barplot, x="controller", y="value", hue="spring", dodge=True, ci=None)
    g.add_legend(title="Spring Type")
    g.set_axis_labels("Controller", "ΔH (Bits)")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.axhline(0, color="black", linestyle="--")

    plt.tight_layout()
    output_path = Path("report_results/entropy_metric_comparison_all.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    main()
