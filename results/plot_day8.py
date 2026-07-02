"""Day-8 charts: frontier extraction, categorization regime crossover, ablation."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = os.path.dirname(os.path.abspath(__file__))


def _rows(name):
    return list(csv.DictReader(open(os.path.join(RESULTS, name), encoding="utf-8")))


def _f(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan


# ---------------------------------------------- 1) extraction frontier ---
fc = _rows("frontier_comparison.csv")
ext = [r for r in fc if r["component"] == "extraction"]
labels = ["regex\n(Day-1)", "rules_smart\n(champion, $0)", "LLM zero-shot\n(Opus 4.8, paid)"]
fields = ["amount_acc", "date_acc", "merchant_acc"]
fnames = ["amount", "date", "merchant"]
x = np.arange(len(fields)); w = 0.26
colors = ["#b0b0b0", "#2b8cbe", "#e6550d"]
fig, ax = plt.subplots(figsize=(8.4, 4.6))
for i, r in enumerate(ext):
    vals = [_f(r[f]) for f in fields]
    b = ax.bar(x + (i - 1) * w, vals, w, label=labels[i], color=colors[i])
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(fnames)
ax.set_ylim(0, 1.12); ax.set_ylabel("field accuracy")
ax.set_title("Receipt extraction — fresh 50 SROIE receipts (offset>=100, unseen)\n"
             "LLM wins accuracy; rules_smart wins ~10,000x on cost + latency", fontsize=10)
ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "frontier_extraction.png"), dpi=130)
plt.close(fig)

# ------------------------------- 2) categorization regime crossover ------
cat = [r for r in fc if r["component"] == "categorization"]
methods = ["keyword\n(Day-1)", "tfidf_linsvc\n(champion, $0)", "LLM zero-shot\n(Opus 4.8)"]
in_d = [_f(r["f1_in_dist"]) for r in cat]
nov = [_f(r["f1_novel"]) for r in cat]
x = np.arange(len(methods)); w = 0.38
fig, ax = plt.subplots(figsize=(8.2, 4.8))
b1 = ax.bar(x - w / 2, in_d, w, label="in-distribution merchants (n=60)", color="#2b8cbe")
b2 = ax.bar(x + w / 2, nov, w, label="NOVEL merchants, never trained on (n=40)", color="#e6550d")
for b, vals in ((b1, in_d), (b2, nov)):
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
ax.axhline(0.3408, ls="--", c="#888", lw=1)
ax.text(2.35, 0.355, "keyword novel floor", fontsize=7, color="#666")
ax.set_xticks(x); ax.set_xticklabels(methods)
ax.set_ylim(0, 1.12); ax.set_ylabel("macro-F1 (labels present in regime)")
ax.set_title("Expense categorization — where the $0 model matches the LLM, and where it FAILS\n"
             "champion ties LLM in-distribution but collapses below keyword on unseen brands", fontsize=10)
ax.legend(fontsize=8, loc="center left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "frontier_categorize_regime.png"), dpi=130)
plt.close(fig)

# ----------------------------------------------------- 3) ablation -------
ab = _rows("ablation.csv")
stages = [r["stage"].split("  ", 1)[-1] for r in ab]
overall = [_f(r["macro_f1"]) for r in ab]
ind = [_f(r["f1_in_dist"]) for r in ab]
nvl = [_f(r["f1_novel"]) for r in ab]
xs = np.arange(len(stages))
fig, ax = plt.subplots(figsize=(9.2, 4.8))
ax.plot(xs, ind, "-o", label="in-distribution", color="#2b8cbe")
ax.plot(xs, overall, "-o", label="overall (60/40 mix)", color="#31a354")
ax.plot(xs, nvl, "-o", label="novel merchants", color="#e6550d")
for xi, v in zip(xs, overall):
    ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color="#31a354")
ax.set_xticks(xs); ax.set_xticklabels(stages, rotation=18, ha="right", fontsize=8)
ax.set_ylim(0, 1.12); ax.set_ylabel("macro-F1")
ax.set_title("Categorizer ablation on the fresh held-out 100\n"
             "features perfect the in-distribution fit; NOTHING lifts novel merchants past ~0.29 (OOV ceiling)",
             fontsize=10)
ax.legend(fontsize=8, loc="center right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "ablation.png"), dpi=130)
plt.close(fig)

print("wrote frontier_extraction.png, frontier_categorize_regime.png, ablation.png")
