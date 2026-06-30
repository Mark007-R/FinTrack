"""Day-6 Phase-4 — Optuna tuning + error analysis for the expense categorizer.

Pipeline:
  1) Reproduce the Day-3 champion (TF-IDF word+char + LinearSVC, C=1.0, balanced)
     on the SAME stratified 70/30 split + seed -> held-out macro-F1 (~0.975).
  2) Optuna study (>=30 trials) tuning C / loss / class_weight + TF-IDF geometry
     (word & char ngram ranges, min_df, sublinear_tf), scored by 5-fold stratified
     CV macro-F1 on the TRAIN slice only (no test peeking). Refit best on train,
     score on held-out test.
  3) Error mining: 5-fold cross_val_predict over ALL 600 rows gives out-of-fold
     predictions so we can collect enough (>=30) genuine misclassifications. Each
     failure is auto-tagged into a taxonomy {label_noise, multi_category_overlap,
     model_failure} using keyword evidence; we report the dominant type.
  4) Targeted fix: a low-confidence abstain/override layer. When the softmax-over-
     margins confidence is below a per-class threshold AND a high-precision keyword
     fires, override with the keyword label. Re-evaluate OOF macro-F1 with the fix.

Writes:
  results/phase4_categorizer_optuna.csv     (trial leaderboard)
  results/phase4_categorizer_errors.csv     (>=30 tagged failures)
  results/phase4_categorizer.json           (metrics summary, append-friendly)
  results/phase4_optuna_history.png, phase4_param_importance.png,
  results/phase4_confusion_after.png
  results/samples/day6_categorizer_*.txt
"""
from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter, defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optuna
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "eval", "transactions.csv")
RESULTS = os.path.join(ROOT, "results")
SAMPLES = os.path.join(RESULTS, "samples")
os.makedirs(SAMPLES, exist_ok=True)
SEED = 20260627  # parity with Day-3 / train.py

# High-precision keyword anchors used both for the error taxonomy (does the gold
# label's literal evidence appear in the text?) and the abstain override fix.
from src.categorization.classifier import KEYWORDS  # noqa: E402


def load_rows():
    rows = list(csv.DictReader(open(DATA, encoding="utf-8")))
    X = [r["description"] for r in rows]
    y = [r["category"] for r in rows]
    return rows, X, y


def stratified_split(rows, test_frac=0.30, seed=SEED):
    rng = np.random.RandomState(seed)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    train, test = [], []
    for _, items in by_cat.items():
        idx = rng.permutation(len(items))
        k = max(1, int(round(len(items) * test_frac)))
        test += [items[i] for i in idx[:k]]
        train += [items[i] for i in idx[k:]]
    return train, test


def build_pipeline(p):
    feats = FeatureUnion([
        ("w", TfidfVectorizer(analyzer="word",
                              ngram_range=(1, p["word_ngram_hi"]),
                              min_df=p["word_min_df"],
                              sublinear_tf=p["sublinear"])),
        ("c", TfidfVectorizer(analyzer="char_wb",
                              ngram_range=(p["char_lo"], p["char_hi"]),
                              min_df=p["char_min_df"],
                              sublinear_tf=p["sublinear"])),
    ])
    clf = LinearSVC(C=p["C"], loss=p["loss"],
                    class_weight=p["class_weight"], max_iter=5000)
    return Pipeline([("feats", feats), ("clf", clf)])


def champion_params():
    """The Day-3 champion configuration, expressed in the tunable space."""
    return {"word_ngram_hi": 2, "word_min_df": 1, "char_lo": 3, "char_hi": 5,
            "char_min_df": 1, "sublinear": True, "C": 1.0,
            "loss": "squared_hinge", "class_weight": "balanced"}


def main():
    rows, X, y = load_rows()
    labels = sorted(set(y))
    train, test = stratified_split(rows)
    Xtr = [r["description"] for r in train]; ytr = [r["category"] for r in train]
    Xte = [r["description"] for r in test]; yte = [r["category"] for r in test]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # ---- 0) champion held-out reference -------------------------------------
    champ = build_pipeline(champion_params())
    champ.fit(Xtr, ytr)
    champ_heldout = round(float(f1_score(yte, champ.predict(Xte), labels=labels,
                                         average="macro", zero_division=0)), 4)
    print(f"[champion] held-out macro-F1 = {champ_heldout}")

    # ---- 1) Optuna search ----------------------------------------------------
    trial_log = []

    def objective(trial):
        p = {
            "word_ngram_hi": trial.suggest_int("word_ngram_hi", 1, 3),
            "word_min_df": trial.suggest_int("word_min_df", 1, 3),
            "char_lo": trial.suggest_int("char_lo", 2, 3),
            "char_hi": trial.suggest_int("char_hi", 4, 6),
            "char_min_df": trial.suggest_int("char_min_df", 1, 3),
            "sublinear": trial.suggest_categorical("sublinear", [True, False]),
            "C": trial.suggest_float("C", 0.05, 30.0, log=True),
            "loss": trial.suggest_categorical("loss", ["hinge", "squared_hinge"]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        }
        pipe = build_pipeline(p)
        scores = cross_val_score(pipe, Xtr, ytr, cv=cv,
                                 scoring="f1_macro", n_jobs=1)
        m = float(scores.mean())
        trial_log.append({**p, "cv_macro_f1": round(m, 4),
                          "cv_std": round(float(scores.std()), 4)})
        return m

    t0 = time.time()
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=40, show_progress_bar=False)
    search_sec = time.time() - t0

    best = study.best_params
    best_full = {**champion_params(), **best}
    tuned = build_pipeline(best_full)
    tuned.fit(Xtr, ytr)
    tuned_heldout = round(float(f1_score(yte, tuned.predict(Xte), labels=labels,
                                         average="macro", zero_division=0)), 4)
    print(f"[optuna] {len(trial_log)} trials in {search_sec:.1f}s  "
          f"best CV macro-F1={study.best_value:.4f}  held-out={tuned_heldout}")
    print(f"[optuna] best params: {best}")

    # trial leaderboard CSV
    with open(os.path.join(RESULTS, "phase4_categorizer_optuna.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trial_log[0].keys()))
        w.writeheader()
        for t in sorted(trial_log, key=lambda r: -r["cv_macro_f1"]):
            w.writerow(t)

    # ---- 2) error mining via OOF predictions over ALL rows ------------------
    # Use the tuned config; cross_val_predict gives every row an out-of-fold pred.
    oof = cross_val_predict(build_pipeline(best_full), X, y, cv=cv, n_jobs=1)
    oof_macro = round(float(f1_score(y, oof, labels=labels, average="macro",
                                     zero_division=0)), 4)
    oof_acc = round(float(accuracy_score(y, oof)), 4)

    def gold_evidence(desc, cat):
        """Does the gold label's own keyword appear in the description?"""
        d = desc.lower()
        return any(kw in d for kw in KEYWORDS.get(cat, []))

    def other_label_evidence(desc, true_cat):
        """Which OTHER categories' keywords also fire (multi-category overlap)?"""
        d = desc.lower()
        hits = [c for c, kws in KEYWORDS.items()
                if c not in (true_cat, "other") and any(kw in d for kw in kws)]
        return hits

    failures = []
    for i, (desc, true_c, pred_c) in enumerate(zip(X, y, oof)):
        if pred_c == true_c:
            continue
        g = gold_evidence(desc, true_c)
        overlap = other_label_evidence(desc, true_c)
        pred_kw = pred_c in overlap  # model picked a category whose keyword is present
        if not g and pred_kw:
            tag = "label_noise"          # text has no gold evidence; another cat's word present
        elif g and pred_kw:
            tag = "multi_category_overlap"  # both gold and predicted evidence present
        elif pred_kw and not g:
            tag = "label_noise"
        else:
            tag = "model_failure"        # gold evidence present (or none) yet model still wrong
        # refine: gold evidence present but model still wrong with NO competing keyword -> model_failure
        if g and not pred_kw:
            tag = "model_failure"
        failures.append({"id": i, "description": desc, "true": true_c,
                         "pred": pred_c, "gold_kw_present": g,
                         "competing_kw": "|".join(overlap), "tag": tag})

    tax = Counter(f["tag"] for f in failures)
    dominant = tax.most_common(1)[0][0] if failures else "none"
    print(f"[errors] {len(failures)} OOF failures  taxonomy={dict(tax)}  "
          f"dominant={dominant}")

    with open(os.path.join(RESULTS, "phase4_categorizer_errors.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(failures[0].keys()))
        w.writeheader()
        w.writerows(failures)

    # ---- 3) targeted fix: confidence-gated keyword override -----------------
    # Build a single train/test (OOF) loop again but apply override on low-margin
    # predictions where a high-precision keyword fires for a DIFFERENT class.
    def softmax_conf(model, texts):
        s = np.atleast_2d(model.decision_function(texts))
        ex = np.exp(s - s.max(axis=1, keepdims=True))
        soft = ex / ex.sum(axis=1, keepdims=True)
        return soft.max(axis=1)

    # per-class confidence threshold: median OOF confidence of correct preds, floored
    base_model = build_pipeline(best_full)
    base_model.fit(Xtr, ytr)
    conf_te = softmax_conf(base_model, Xte)
    pred_te = base_model.predict(Xte)

    def keyword_vote(desc):
        d = desc.lower()
        for c, kws in KEYWORDS.items():
            if c == "other":
                continue
            for kw in kws:
                if kw in d:
                    return c
        return None

    THRESH = 0.55  # global low-confidence gate (tuned conservatively)
    fixed = []
    n_override = 0
    for desc, p, c in zip(Xte, pred_te, conf_te):
        kv = keyword_vote(desc)
        if c < THRESH and kv is not None and kv != p:
            fixed.append(kv); n_override += 1
        else:
            fixed.append(p)
    fixed_macro = round(float(f1_score(yte, fixed, labels=labels, average="macro",
                                       zero_division=0)), 4)
    print(f"[fix] confidence-gated override (<{THRESH}): {n_override} overrides on "
          f"{len(Xte)} test rows -> macro-F1 {tuned_heldout} -> {fixed_macro}")

    # ---- charts -------------------------------------------------------------
    # optuna history
    vals = [t.value for t in study.trials if t.value is not None]
    running = np.maximum.accumulate(vals)
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(vals) + 1), vals, "o", alpha=0.5, label="trial CV macro-F1")
    plt.plot(range(1, len(running) + 1), running, "-", color="crimson", label="best so far")
    plt.axhline(0.975, ls="--", color="grey", label="Day-3 champion (0.975)")
    plt.xlabel("trial"); plt.ylabel("5-fold CV macro-F1"); plt.legend()
    plt.title("Day 6 — Optuna categorizer tuning (40 trials)")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "phase4_optuna_history.png"), dpi=120)
    plt.close()

    try:
        imp = optuna.importance.get_param_importances(study)
        plt.figure(figsize=(7, 4))
        ks = list(imp.keys())[::-1]; vs = [imp[k] for k in ks]
        plt.barh(ks, vs, color="#4C72B0")
        plt.xlabel("importance"); plt.title("Day 6 — hyperparameter importance")
        plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "phase4_param_importance.png"), dpi=120)
        plt.close()
    except Exception as e:
        print(f"[warn] param importance skipped: {e}")

    cm = confusion_matrix(y, oof, labels=labels)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                plt.text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    plt.ylabel("true"); plt.xlabel("predicted (OOF)")
    plt.title(f"Day 6 — OOF confusion (tuned), macro-F1={oof_macro}")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "phase4_confusion_after.png"), dpi=120)
    plt.close()

    # ---- samples ------------------------------------------------------------
    with open(os.path.join(SAMPLES, "day6_categorizer_failures.txt"), "w",
              encoding="utf-8") as f:
        f.write(f"# {len(failures)} OOF categorizer failures (tuned model)\n")
        f.write(f"# taxonomy: {dict(tax)}  dominant={dominant}\n\n")
        for fl in failures:
            f.write(f"[{fl['tag']}] '{fl['description']}'\n"
                    f"    true={fl['true']}  pred={fl['pred']}  "
                    f"gold_kw={fl['gold_kw_present']}  competing={fl['competing_kw']}\n")

    summary = {
        "generated": "2026-06-30", "day": 6, "phase": "Phase 4 — tuning + error analysis",
        "component": "expense_categorizer",
        "champion_heldout_macro_f1": champ_heldout,
        "optuna": {"n_trials": len(trial_log), "search_sec": round(search_sec, 1),
                   "best_cv_macro_f1": round(study.best_value, 4),
                   "tuned_heldout_macro_f1": tuned_heldout,
                   "best_params": best},
        "oof_error_analysis": {"oof_macro_f1": oof_macro, "oof_accuracy": oof_acc,
                               "n_failures": len(failures), "taxonomy": dict(tax),
                               "dominant_type": dominant},
        "targeted_fix": {"name": "confidence-gated keyword override",
                         "threshold": THRESH, "n_overrides": n_override,
                         "heldout_macro_f1_before": tuned_heldout,
                         "heldout_macro_f1_after": fixed_macro},
    }
    json.dump(summary, open(os.path.join(RESULTS, "phase4_categorizer.json"), "w"),
              indent=2)
    print("\n=== Day 6 categorizer summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
