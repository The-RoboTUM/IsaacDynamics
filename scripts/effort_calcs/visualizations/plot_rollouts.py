#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import matplotlib.pyplot as plt
import re
from pathlib import Path

import seaborn as sns

sns.set(style="whitegrid")

from isaaclab_dynamics.utils.io import setup_parser


def extract_episode_num(name: str) -> int:
    match = re.search(r"rollouts_(\d+)", name)
    return int(match.group(1)) if match else -1


def extract_mode(name: str) -> str:
    return "full" if "full" in name else "simplified" if "simplified" in name else "unknown"


def main(args_cli):
    path = Path(args_cli.results_file_name)
    if not path.exists():
        raise FileNotFoundError(f"No file at {args_cli.results_file_name}")

    with path.open("r") as f:
        data = json.load(f)

    # Organize data by mode
    grouped = {"full": [], "simplified": []}
    for entry in data:
        mode = extract_mode(entry["experiment_name"])
        if mode in grouped:
            grouped[mode].append((extract_episode_num(entry["experiment_name"]), entry))

    plt.figure(figsize=(10, 6))

    # Define consistent colors for each metric
    color_map = {
        "Delta_H_y_u": sns.color_palette()[0],
        "Delta_H_u": sns.color_palette()[1],
        "Delta_H_y": sns.color_palette()[2],
    }

    for metric_key, marker in zip(["Delta_H_y_u", "Delta_H_u", "Delta_H_y"], ["o", "x", "^"]):
        for mode in ["full", "simplified"]:
            entries = grouped[mode]
            if not entries:
                continue
            entries.sort(key=lambda x: x[0])
            episode_counts = [e[0] for e in entries]
            metric_values = [e[1][metric_key] for e in entries]
            linestyle = "-" if mode == "full" else "--"
            label = f"{metric_key.replace('_', '|') if metric_key=='Delta_H_y_u' else metric_key} ({mode})"
            plt.plot(
                episode_counts,
                metric_values,
                label=label,
                marker=marker,
                linestyle=linestyle,
                color=color_map[metric_key],
            )

    plt.xlabel("Number of Episodes")
    plt.ylabel("Control Effort (Bits)")
    plt.title("Control Effort vs. Number of Episodes (Full vs. Simplified)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("report_results/control_effort_vs_episodes.png")

    if args_cli.show_plots:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    parser = setup_parser()
    args_cli, _ = parser.parse_known_args()
    main(args_cli)
