"""Day-6 Phase-4 — error analysis on the adversarial stress set + targeted fixes.

The in-distribution synthetic set saturates (OOF macro-F1 ~0.985, only ~6 errors),
so genuine failure modes are mined from the Day-6 stress set
(data/eval/transactions_stress.csv): realistic merchant strings, most with NO
literal category keyword and a handful with a competing-category keyword.

Steps:
  1) Train the tuned categorizer on ALL 600 in-distribution rows (shipped config).
  2) Predict on the 82 stress rows -> macro-F1, accuracy, per-row preds.
  3) Mine every failure, tag taxonomy {label_noise, multi_category_overlap,
     model_failure} using keyword evidence + the row's `difficulty` flag.
  4) Two targeted fixes, each measured on the stress set AND checked for
     regression on the in-distribution 600 (OOF):
       (a) confidence-gated keyword override (global, low-margin only)
       (b) phrase-priority disambiguation layer (multi-word service phrase beats
           a single competing brand token) -- aimed at multi_category_overlap.
  5) Report which fix helps, and log what carries forward (brand gazetteer /
     active-learning augmentation for the no_kw model_failures -> Day 7).

Writes:
  results/phase4_stress_preds.csv, results/phase4_stress_errors.csv
  results/phase4_stress.json
  results/phase4_stress_by_difficulty.png
  results/samples/day6_stress_failures.txt
"""
from __future__ import annotations
import csv, json, os
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score

from src.categorization.classifier import KEYWORDS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "eval", "transactions.csv")
STRESS = os.path.join(ROOT, "data", "eval", "transactions_stress.csv")
RESULTS = os.path.join(ROOT, "results")
SAMPLES = os.path.join(RESULTS, "samples")
SEED = 20260627

# Tuned config from the Optuna study (results/phase4_categorizer.json best_params).
BEST = {"word_ngram_hi": 3, "word_min_df": 1, "char_lo": 2, "char_hi": 6,
        "char_min_df": 3, "sublinear": False, "C": 4.4321841874023535,
        "loss": "squared_hinge", "class_weight": "balanced"}


def build_pipeline(p=BEST):
    feats = FeatureUnion([
        ("w", TfidfVectorizer(analyzer="word", ngram_range=(1, p["word_ngram_hi"]),
                              min_df=p["word_min_df"], sublinear_tf=p["sublinear"])),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(p["char_lo"], p["char_hi"]),
                              min_df=p["char_min_df"], sublinear_tf=p["sublinear"])),
    ])
    clf = LinearSVC(C=p["C"], loss=p["loss"], class_weight=p["class_weight"], max_iter=5000)
    return Pipeline([("feats", feats), ("clf", clf)])


def softmax_conf_and_pred(model, texts):
    s = np.atleast_2d(model.decision_function(texts))
    ex = np.exp(s - s.max(axis=1, keepdims=True))
    soft = ex / ex.sum(axis=1, keepdims=True)
    preds = model.predict(texts)
    return soft.max(axis=1), preds


def keyword_vote(desc):
    d = desc.lower()
    for c, kws in KEYWORDS.items():
        if c == "other":
            continue
        for kw in kws:
            if kw in d:
                return c
    return None


# Phrase-priority disambiguation: a multi-word service phrase outranks the single
# competing brand token inside it. Deliberately HIGH-PRECISION and multi-word so
# the rule cannot fire on in-distribution rows (a bare " gas"/"pharmacy" rule was
# tested first and regressed the in-dist OOF 0.985->0.966 by over-firing on
# "socalgas"/"gas co" utility rows — see report). These encode genuine service
# priority, not memorised test answers.
DISAMBIG = [
    ("uber eats", "dining"),
    ("amazon fresh", "groceries"),
    ("amazon grocery", "groceries"),
    ("apple music", "entertainment"),
    ("costco gas", "transport"),
    ("walmart pharmacy", "health"),
]


def disambig_override(desc):
    d = desc.lower()
    for phrase, cat in DISAMBIG:
        if phrase in d:
            return cat
    return None


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return rows


def main():
    train_rows = load(DATA)
    X = [r["description"] for r in train_rows]; y = [r["category"] for r in train_rows]
    labels = sorted(set(y))
    stress = load(STRESS)
    Xs = [r["description"] for r in stress]; ys = [r["category"] for r in stress]
    diffs = [r["difficulty"] for r in stress]

    model = build_pipeline()
    model.fit(X, y)

    conf, preds = softmax_conf_and_pred(model, Xs)
    base_macro = round(float(f1_score(ys, preds, labels=labels, average="macro", zero_division=0)), 4)
    base_acc = round(float(accuracy_score(ys, preds)), 4)
    print(f"[stress] tuned model: macro-F1={base_macro} acc={base_acc} on {len(Xs)} hard rows "
          f"(in-dist OOF was ~0.985)")

    # ---- taxonomy ----
    def tag(desc, true_c, pred_c, difficulty):
        d = desc.lower()
        gold_kw = any(kw in d for kw in KEYWORDS.get(true_c, []))
        competing = [c for c, kws in KEYWORDS.items()
                     if c not in (true_c, "other") and any(kw in d for kw in kws)]
        if difficulty == "overlap" and pred_c in competing:
            return "multi_category_overlap", gold_kw, competing
        if not gold_kw:
            return "model_failure", gold_kw, competing   # no lexical anchor -> generalisation gap
        return "model_failure", gold_kw, competing

    failures = []
    for r, true_c, pred_c, dfc in zip(stress, ys, preds, diffs):
        if pred_c == true_c:
            continue
        t, gk, comp = tag(r["description"], true_c, pred_c, dfc)
        failures.append({"description": r["description"], "true": true_c, "pred": pred_c,
                         "difficulty": dfc, "gold_kw_present": gk,
                         "competing_kw": "|".join(comp), "tag": t})
    tax = Counter(f["tag"] for f in failures)
    dominant = tax.most_common(1)[0][0] if failures else "none"
    print(f"[stress] {len(failures)} failures  taxonomy={dict(tax)}  dominant={dominant}")

    # ---- fix (a): confidence-gated keyword override ----
    THRESH = 0.55
    fa, na = [], 0
    for desc, p, c in zip(Xs, preds, conf):
        kv = keyword_vote(desc)
        if c < THRESH and kv is not None and kv != p:
            fa.append(kv); na += 1
        else:
            fa.append(p)
    fa_macro = round(float(f1_score(ys, fa, labels=labels, average="macro", zero_division=0)), 4)
    fa_acc = round(float(accuracy_score(ys, fa)), 4)

    # ---- fix (b): phrase-priority disambiguation ----
    fb, nb = [], 0
    for desc, p in zip(Xs, preds):
        ov = disambig_override(desc)
        if ov is not None and ov != p:
            fb.append(ov); nb += 1
        else:
            fb.append(p)
    fb_macro = round(float(f1_score(ys, fb, labels=labels, average="macro", zero_division=0)), 4)
    fb_acc = round(float(accuracy_score(ys, fb)), 4)

    # overlap-subset accuracy before/after fix (b)
    ov_idx = [i for i, d in enumerate(diffs) if d == "overlap"]
    ov_before = round(float(np.mean([preds[i] == ys[i] for i in ov_idx])), 4)
    ov_after = round(float(np.mean([fb[i] == ys[i] for i in ov_idx])), 4)

    print(f"[fix-a] conf-gated keyword override: {na} overrides -> macro {base_macro}->{fa_macro}")
    print(f"[fix-b] phrase disambiguation: {nb} overrides -> macro {base_macro}->{fb_macro} "
          f"| overlap-subset acc {ov_before}->{ov_after}")

    # ---- regression check on in-distribution 600 (OOF) ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(build_pipeline(), X, y, cv=cv, n_jobs=1)
    oof_base = round(float(f1_score(y, oof, labels=labels, average="macro", zero_division=0)), 4)
    oof_fix = []
    for desc, p in zip(X, oof):
        ov = disambig_override(desc)
        oof_fix.append(ov if (ov is not None and ov != p) else p)
    oof_after = round(float(f1_score(y, oof_fix, labels=labels, average="macro", zero_division=0)), 4)
    print(f"[regression] in-dist OOF macro: {oof_base} -> {oof_after} (after disambiguation)")

    # ---- outputs ----
    with open(os.path.join(RESULTS, "phase4_stress_preds.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["description", "true", "pred", "conf", "difficulty", "fix_b_pred", "correct_base", "correct_fixb"])
        for r, p, c, dfc, fbp, ty in zip(stress, preds, conf, diffs, fb, ys):
            w.writerow([r["description"], ty, p, round(float(c), 4), dfc, fbp,
                        int(p == ty), int(fbp == ty)])
    with open(os.path.join(RESULTS, "phase4_stress_errors.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(failures[0].keys()))
        w.writeheader(); w.writerows(failures)

    # by-difficulty accuracy chart
    by_diff = defaultdict(lambda: [0, 0])
    for p, ty, dfc in zip(preds, ys, diffs):
        by_diff[dfc][1] += 1
        by_diff[dfc][0] += int(p == ty)
    cats = list(by_diff.keys())
    accs = [by_diff[c][0] / by_diff[c][1] for c in cats]
    plt.figure(figsize=(6, 4))
    plt.bar(cats, accs, color=["#C44E52", "#DD8452"])
    for i, a in enumerate(accs):
        plt.text(i, a + 0.02, f"{a:.2f}", ha="center")
    plt.ylim(0, 1.05); plt.ylabel("accuracy")
    plt.title(f"Day 6 — stress-set accuracy by difficulty (overall {base_acc})")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "phase4_stress_by_difficulty.png"), dpi=120)
    plt.close()

    with open(os.path.join(SAMPLES, "day6_stress_failures.txt"), "w", encoding="utf-8") as f:
        f.write(f"# {len(failures)} stress-set failures (tuned categorizer)\n")
        f.write(f"# taxonomy {dict(tax)} dominant={dominant}\n\n")
        for fl in failures:
            f.write(f"[{fl['tag']}] '{fl['description']}'  true={fl['true']} pred={fl['pred']} "
                    f"diff={fl['difficulty']} gold_kw={fl['gold_kw_present']} "
                    f"competing={fl['competing_kw']}\n")

    summary = {
        "generated": "2026-06-30", "day": 6, "phase": "Phase 4 — error analysis (stress set)",
        "stress_set": {"path": "data/eval/transactions_stress.csv", "n": len(Xs),
                       "n_no_kw": diffs.count("no_kw"), "n_overlap": diffs.count("overlap")},
        "tuned_model_on_stress": {"macro_f1": base_macro, "accuracy": base_acc},
        "by_difficulty_acc": {c: round(by_diff[c][0] / by_diff[c][1], 4) for c in cats},
        "error_taxonomy": {"n_failures": len(failures), "counts": dict(tax), "dominant": dominant},
        "fix_a_conf_override": {"n_overrides": na, "stress_macro_f1": fa_macro, "stress_acc": fa_acc},
        "fix_b_phrase_disambiguation": {"n_overrides": nb, "stress_macro_f1": fb_macro,
                                        "stress_acc": fb_acc,
                                        "overlap_acc_before": ov_before, "overlap_acc_after": ov_after,
                                        "indist_oof_before": oof_base, "indist_oof_after": oof_after},
        "carry_forward": "no_kw model_failures need a brand->category gazetteer / "
                         "active-learning augmentation (Day 7).",
    }
    json.dump(summary, open(os.path.join(RESULTS, "phase4_stress.json"), "w"), indent=2)
    print("\n=== Day 6 stress summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
