"""Day-7 Phase-5 (b): active-learning loop for the expense categorizer.

Day-6 finding (carried forward): the categorizer saturates the in-distribution set
(OOF macro-F1 ~0.985) but collapses on realistic HARD merchant strings — rows with
NO literal category keyword (`no_kw`) or a keyword for the WRONG class (`overlap`).
The dominant failure was `model_failure`, and the Day-6 report predicted the fix is
DATA, not hyperparameters: label the hard cases and retrain.

This is that experiment, done as a proper active-learning loop and measured against
a random-sampling control:

  * SEED   = a small stratified label budget from the easy in-distribution data
             (mimics "we only labelled the obvious transactions first").
  * POOL   = remaining in-dist rows + a 242-row HARD pool (transactions_pool.csv),
             labels hidden.
  * TEST   = fixed held-out set = stratified hard rows (the metric that matters) +
             stratified in-dist rows (regression guard). Never trained on.
  * Two arms, identical seed:
        active  -> query the 32 LOWEST-confidence pool rows each round
                   (uncertainty = 1 - max softmax over LinearSVC margins)
        random  -> query 32 random pool rows each round
  * 6 rounds. Each round: fit TF-IDF(word+char)+LinearSVC champion, measure
    macro-F1 (overall), hard-subset accuracy, in-dist accuracy, and the fraction
    of the queried batch that was actually HARD.

Genuine insight to test: does uncertainty sampling preferentially pull the hard
no_kw/overlap rows, lifting hard-case accuracy faster (fewer labels) than random —
closing the Day-6 gap via data as predicted?

Outputs
-------
    results/phase5_active_learning.csv          (per-round curves, both arms)
    results/phase5_active_learning.png          (hard-acc + macro-F1 vs labels)
    results/samples/phase5_al_queried_hard.json (hard rows AL pulled early)
    results/phase5_robustness.csv               (track=active_learning summary rows)
    results/metrics.json                        (appended)
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.categorization.train import _build_pipeline  # champion pipeline factory
from results.phase5_io import upsert_track

EVAL = os.path.join(ROOT, "data", "eval")
RESULTS = os.path.join(ROOT, "results")
SAMPLES = os.path.join(RESULTS, "samples")
os.makedirs(SAMPLES, exist_ok=True)

SEED = 20260701
N_ROUNDS = 6
BATCH = 32
SEED_BUDGET = 80


def load_rows(path, hard):
    out = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        out.append({"description": r["description"], "category": r["category"],
                    "difficulty": r.get("difficulty", "easy" if not hard else "no_kw"),
                    "hard": hard})
    return out


def stratified_take(rows, frac, rng, key="category"):
    """Return (taken, rest) with `frac` of each class taken."""
    by = defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    taken, rest = [], []
    for _, items in by.items():
        idx = rng.permutation(len(items))
        k = max(1, int(round(len(items) * frac)))
        taken += [items[i] for i in idx[:k]]
        rest += [items[i] for i in idx[k:]]
    return taken, rest


def macro_f1(pipe, rows):
    from sklearn.metrics import f1_score
    X = [r["description"] for r in rows]
    y = [r["category"] for r in rows]
    pred = pipe.predict(X)
    return f1_score(y, pred, average="macro", zero_division=0)


def accuracy(pipe, rows):
    if not rows:
        return float("nan")
    X = [r["description"] for r in rows]
    y = [r["category"] for r in rows]
    pred = pipe.predict(X)
    return float(np.mean([a == b for a, b in zip(pred, y)]))


def uncertainty(pipe, rows):
    """1 - max softmax over LinearSVC decision margins (higher = less confident)."""
    X = [r["description"] for r in rows]
    scores = np.atleast_2d(pipe.decision_function(X))
    ex = np.exp(scores - scores.max(axis=1, keepdims=True))
    soft = ex / ex.sum(axis=1, keepdims=True)
    return 1.0 - soft.max(axis=1)


def run_arm(arm, seed_rows, pool_rows, test_hard, test_indist, rng):
    labeled = list(seed_rows)
    pool = list(pool_rows)
    curve = []
    early_hard_queried = []
    for rnd in range(N_ROUNDS + 1):
        pipe = _build_pipeline()
        pipe.fit([r["description"] for r in labeled], [r["category"] for r in labeled])
        row = {
            "arm": arm, "round": rnd, "n_labeled": len(labeled),
            "macro_f1_all": round(macro_f1(pipe, test_hard + test_indist), 4),
            "hard_acc": round(accuracy(pipe, test_hard), 4),
            "indist_acc": round(accuracy(pipe, test_indist), 4),
        }
        # select next batch (skip on the final measurement round)
        if rnd < N_ROUNDS and pool:
            if arm == "active":
                u = uncertainty(pipe, pool)
                order = np.argsort(-u)  # most uncertain first
            else:
                order = rng.permutation(len(pool))
            pick = order[:BATCH]
            batch = [pool[i] for i in pick]
            row["batch_hard_frac"] = round(np.mean([r["hard"] for r in batch]), 3)
            if rnd < 2 and arm == "active":
                early_hard_queried += [{"round": rnd, "description": r["description"],
                                        "category": r["category"], "difficulty": r["difficulty"]}
                                       for r in batch if r["hard"]]
            labeled += batch
            pool = [pool[i] for i in range(len(pool)) if i not in set(pick.tolist())]
        else:
            row["batch_hard_frac"] = ""
        curve.append(row)
    return curve, early_hard_queried


def main():
    rng = np.random.RandomState(SEED)
    indist = load_rows(os.path.join(EVAL, "transactions.csv"), hard=False)
    hard = load_rows(os.path.join(EVAL, "transactions_pool.csv"), hard=True)

    # fixed held-out TEST (never trained on)
    test_hard, hard_rest = stratified_take(hard, 0.30, rng)     # ~72 hard test rows
    test_indist, indist_rest = stratified_take(indist, 0.12, rng)  # ~72 in-dist test rows

    # seed = small stratified budget from in-dist rest; pool = the remainder + hard rest
    seed_rows, indist_pool = stratified_take(indist_rest, SEED_BUDGET / max(1, len(indist_rest)), rng)
    pool_rows = indist_pool + hard_rest
    rng.shuffle(pool_rows)

    print(f"[day7-AL] seed={len(seed_rows)}  pool={len(pool_rows)} "
          f"(hard_in_pool={sum(r['hard'] for r in pool_rows)})  "
          f"test_hard={len(test_hard)}  test_indist={len(test_indist)}")

    curves = []
    early = []
    for arm in ("active", "random"):
        arm_rng = np.random.RandomState(SEED + (0 if arm == "active" else 7))
        c, e = run_arm(arm, seed_rows, pool_rows, test_hard, test_indist, arm_rng)
        curves += c
        early += e
        last = c[-1]
        print(f"  {arm:<7} final: n={last['n_labeled']} macroF1={last['macro_f1_all']} "
              f"hard_acc={last['hard_acc']} indist_acc={last['indist_acc']}")

    # write per-round curve CSV
    curve_path = os.path.join(RESULTS, "phase5_active_learning.csv")
    with open(curve_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "round", "n_labeled", "macro_f1_all",
                                          "hard_acc", "indist_acc", "batch_hard_frac"])
        w.writeheader()
        w.writerows(curves)

    # chart: hard-acc + macro-F1 vs n_labeled, both arms
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for arm, style in (("active", "-o"), ("random", "--s")):
        c = [r for r in curves if r["arm"] == arm]
        xs = [r["n_labeled"] for r in c]
        ax1.plot(xs, [r["hard_acc"] for r in c], style, label=arm)
        ax2.plot(xs, [r["macro_f1_all"] for r in c], style, label=arm)
    ax1.set_title("Hard-case accuracy (no_kw / overlap)")
    ax2.set_title("Overall macro-F1 (hard + in-dist test)")
    for ax in (ax1, ax2):
        ax.set_xlabel("# labeled examples"); ax.grid(alpha=0.3); ax.legend()
    ax1.set_ylabel("accuracy"); ax2.set_ylabel("macro-F1")
    fig.suptitle("Day-7 active learning vs random sampling — categorizer")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "phase5_active_learning.png"), dpi=110)

    json.dump(early[:25], open(os.path.join(SAMPLES, "phase5_al_queried_hard.json"), "w"), indent=2)

    # headline deltas
    act = [r for r in curves if r["arm"] == "active"]
    rnd_ = [r for r in curves if r["arm"] == "random"]
    seed_hard = act[0]["hard_acc"]
    final_hard_act = act[-1]["hard_acc"]
    final_hard_rnd = rnd_[-1]["hard_acc"]
    avg_batch_hard_act = np.mean([r["batch_hard_frac"] for r in act if r["batch_hard_frac"] != ""])
    avg_batch_hard_rnd = np.mean([r["batch_hard_frac"] for r in rnd_ if r["batch_hard_frac"] != ""])
    print(f"\n[day7-AL] hard_acc seed={seed_hard} -> active={final_hard_act} vs random={final_hard_rnd}")
    print(f"[day7-AL] mean hard-fraction of queried batch: active={avg_batch_hard_act:.2f} "
          f"random={avg_batch_hard_rnd:.2f}")

    # combined long CSV
    summary = [
        {"variant": "seed_only", "metric": "hard_acc", "value": seed_hard, "note": f"n={act[0]['n_labeled']}"},
        {"variant": "active_final", "metric": "hard_acc", "value": final_hard_act, "note": f"n={act[-1]['n_labeled']}"},
        {"variant": "random_final", "metric": "hard_acc", "value": final_hard_rnd, "note": f"n={rnd_[-1]['n_labeled']}"},
        {"variant": "active_final", "metric": "macro_f1_all", "value": act[-1]["macro_f1_all"], "note": ""},
        {"variant": "random_final", "metric": "macro_f1_all", "value": rnd_[-1]["macro_f1_all"], "note": ""},
        {"variant": "active", "metric": "mean_batch_hard_frac", "value": round(float(avg_batch_hard_act), 3), "note": "uncertainty pulls hard rows"},
        {"variant": "random", "metric": "mean_batch_hard_frac", "value": round(float(avg_batch_hard_rnd), 3), "note": "baseline"},
    ]
    upsert_track("active_learning", summary)

    # metrics.json
    mpath = os.path.join(RESULTS, "metrics.json")
    metrics = json.load(open(mpath)) if os.path.exists(mpath) else []
    if not isinstance(metrics, list):
        metrics = [metrics]
    entry = next((m for m in metrics if m.get("day") == 7), None)
    if entry is None:
        entry = {"day": 7, "generated": "2026-07-01", "phase": "Phase 5 - robustness + reach"}
        metrics.append(entry)
    entry["active_learning"] = {
        "seed_budget": len(seed_rows), "rounds": N_ROUNDS, "batch": BATCH,
        "hard_acc_seed": seed_hard, "hard_acc_active_final": final_hard_act,
        "hard_acc_random_final": final_hard_rnd,
        "mean_batch_hard_frac_active": round(float(avg_batch_hard_act), 3),
        "mean_batch_hard_frac_random": round(float(avg_batch_hard_rnd), 3),
        "macro_f1_active_final": act[-1]["macro_f1_all"],
        "macro_f1_random_final": rnd_[-1]["macro_f1_all"],
    }
    json.dump(metrics, open(mpath, "w"), indent=2)
    print(f"[day7-AL] wrote {curve_path}, chart, samples, combined CSV, metrics.json")


if __name__ == "__main__":
    main()
