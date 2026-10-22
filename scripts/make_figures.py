#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.9", "numpy>=2.0"]
# ///
"""Vector figures for the paper from data/*.csv. Usage: scripts/make_figures.py [--data data] [--out figures]"""
import argparse
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL = {"Gemma": "#1b6ca8", "Qwen": "#d9822b", "Classic": "#555555"}


def rows(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def kappa(data):
    k = {}
    for r in rows(data / "2_agreement.csv"):
        if r["Metric"] == "Cohen's kappa":
            k[r["Level"]] = (float(r["Gemma"]), float(r["Qwen"]))
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    data, out = pathlib.Path(a.data), pathlib.Path(a.out)
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.3), gridspec_kw={"width_ratios": [1, 1.15]})
    k = kappa(data)
    levels = ["Detector (node)", "Corridor", "Network (overall)"]
    x = np.arange(len(levels))
    for i, m in enumerate(("Gemma", "Qwen")):
        vals = [k[l][i] for l in levels]
        bars = ax1.bar(x + (i - 0.5) * 0.36, vals, 0.34, color=COL[m], label=m)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    ax1.set_xticks(x, ["Node", "Corridor", "Network"])
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Cohen's $\\kappa$ vs. reference")
    ax1.legend(frameon=False, loc="upper left", fontsize=7)
    ax1.set_title("(a) Verdict agreement", fontsize=8)

    for m in ("Gemma", "Qwen"):
        s = np.sort([float(r["seconds"]) for r in rows(data / f"3_sessions_{m}.csv")]) / 60
        ax2.step(s, np.arange(1, len(s) + 1) / len(s), where="post", color=COL[m], label=m)
    ex = {r[""]: r for r in rows(data / "4_execution_time.csv")}
    classic = float(ex["Classic (deterministic)"]["mean"]) / 60
    ax2.axvline(classic, color=COL["Classic"], ls=":", lw=1)
    ax2.text(classic * 1.1, 0.08, "classic\n(mean)", fontsize=7, color=COL["Classic"])
    ax2.axvline(60, color="k", ls="--", lw=0.8)
    ax2.text(54, 0.5, "hourly slot", fontsize=7, rotation=90, ha="right")
    ax2.set_xscale("log")
    ax2.set_xlabel("Session time (min, log scale)")
    ax2.set_ylabel("Share of sessions")
    ax2.set_title("(b) Session time", fontsize=8)
    ax2.legend(frameon=False, loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "agreement-time.pdf", transparent=True)
    print("wrote", out / "agreement-time.pdf")


if __name__ == "__main__":
    main()
