#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import matplotlib.pyplot as plt
import re
from pathlib import Path

import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")


def extract_controller_spring_setpoint(experiment_name: str):
    base = experiment_name.replace("controllers_", "").replace(".json", "")
    match = re.match(r"(.*)_(springless|spring)_(\d+)", base)
    if match:
        controller, spring, setpoint = match.groups()
        return controller, spring, int(setpoint)
    return base, "unknown", None


def load_data(file_path: Path):
    with file_path.open("r") as f:
        data = json.load(f)

    include_metrics = {"Delta_H_y_u", "Delta_H_u", "Delta_H_y"}
    rows = []
    for entry in data:
        controller, spring, setpoint = extract_controller_spring_setpoint(entry["experiment_name"])
        row = {
            "controller": controller,
            "spring": spring,
            "setpoint": setpoint,
            "spring_setpoint": f"{spring}_{setpoint}",
        }
        for key in include_metrics:
            row[key] = entry.get(key, 0.0)
        rows.append(row)

    return rows


# Load and prepare data
result_files = [
    "experiment_controllers_springless_90.json",
    "experiment_controllers_spring_90.json",
    "experiment_controllers_spring_60.json",
    "experiment_controllers_spring_45.json",
    "experiment_controllers_spring_30.json",
    "experiment_controllers_spring_0.json",
]
base_path = Path("/home/andrew/Documents/RoboTUM_Workspace/IsaacDynamics/report_results")

all_rows = []
for file_name in result_files:
    all_rows.extend(load_data(base_path / file_name))
df = pd.DataFrame(all_rows)


# Compute efficiency
def compute_efficiency(row):
    return row["Delta_H_y_u"] / row["Delta_H_u"] if row["Delta_H_u"] < 0 else 0.0


df["Efficiency"] = df.apply(compute_efficiency, axis=1)

# Melt for plotting
melted_df = pd.melt(
    df,
    id_vars=["controller", "spring_setpoint"],
    value_vars=["Delta_H_y_u", "Delta_H_u", "Delta_H_y", "Efficiency"],
    var_name="metric",
    value_name="value",
)

# Plot setup
metrics = ["Delta_H_y_u", "Delta_H_u", "Delta_H_y", "Efficiency"]
spring_setpoint_labels = sorted(df["spring_setpoint"].unique())
palette = sns.color_palette("tab10", n_colors=len(spring_setpoint_labels))
label_colors = {label: palette[i] for i, label in enumerate(spring_setpoint_labels)}
label_hatches = {label: "" if "springless" not in label else "//" for label in spring_setpoint_labels}
label_indices = {label: i for i, label in enumerate(spring_setpoint_labels)}

fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=False)
axes = axes.flatten()
bar_width = 0.8 / len(spring_setpoint_labels)

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    sub_df = melted_df[melted_df["metric"] == metric]

    # Exclude random and unactuated only in Efficiency plot
    if metric == "Efficiency":
        sub_df = sub_df[~sub_df["controller"].isin(["random", "unactuated"])]

    controllers = sorted(sub_df["controller"].unique())

    for i, controller in enumerate(controllers):
        for label in spring_setpoint_labels:
            match = sub_df[(sub_df["controller"] == controller) & (sub_df["spring_setpoint"] == label)]
            if not match.empty:
                x_pos = i + (label_indices[label] - len(spring_setpoint_labels) / 2) * bar_width + bar_width / 2
                ax.bar(
                    x_pos,
                    match["value"].values[0],
                    width=bar_width,
                    color=label_colors[label],
                    edgecolor="black",
                    hatch=label_hatches[label],
                    label=label,
                )

    ax.set_title(metric)
    ax.set_xlabel("Controller")
    ax.set_ylabel("ΔH (Bits)" if metric != "Efficiency" else "Ratio")
    ax.set_xticks(range(len(controllers)))
    ax.set_xticklabels(controllers)
    ax.axhline(0, color="black", linestyle="--")

# Legend
handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(
    by_label.values(),
    by_label.keys(),
    title="Spring + Setpoint",
    loc="lower center",
    ncol=len(by_label),
    frameon=False,
)

plt.tight_layout(rect=[0, 0.08, 1, 1])
output_path = Path("report_results/entropy_metric_efficiency_filtered.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path)
plt.show()
