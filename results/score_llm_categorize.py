"""Score the blind zero-shot LLM (Claude) categorizations on the 60-row test subset and
build an apples-to-apples sub-table where EVERY method is scored on the same 60 rows.

LLM predictions (samples/llm_cat_predictions.json) were produced blind to labels.
"""
import csv
import json
import os

from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SAMPLES = os.path.join(RESULTS, "samples")

truth = json.load(open(os.path.join(SAMPLES, "phase2b_subset_truth.json")))
labels = truth["labels"]
sub = truth["subset"]
llm = json.load(open(os.path.join(SAMPLES, "llm_cat_predictions.json")))["predictions"]

y_true = [r["true"] for r in sub]
methods = ["keyword", "tfidf_lgbm", "tfidf_linsvc", "sbert_lgbm", "distilbert"]
table = []
for m in methods:
    y_pred = [r[m] for r in sub]
    macro = round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4)
    acc = round(sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true), 4)
    table.append({"method": m, "n_subset": len(sub), "macro_f1": macro, "accuracy": acc})

# LLM row
y_llm = [llm[str(r["id"])] for r in sub]
macro = round(float(f1_score(y_true, y_llm, labels=labels, average="macro", zero_division=0)), 4)
acc = round(sum(1 for a, b in zip(y_true, y_llm) if a == b) / len(y_true), 4)
table.append({"method": "llm_zeroshot (Claude)", "n_subset": len(sub), "macro_f1": macro, "accuracy": acc})

with open(os.path.join(RESULTS, "phase2b_llm_subset.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["method", "n_subset", "macro_f1", "accuracy"])
    w.writeheader(); w.writerows(table)

print("Comparable sub-table on the SAME 60 test rows:")
for r in table:
    print(f"  {r['method']:24s} macro-F1={r['macro_f1']:.4f} acc={r['accuracy']:.4f}")

# show LLM disagreements vs truth
print("\nLLM disagreements vs ground truth:")
for r in sub:
    p = llm[str(r["id"])]
    if p != r["true"]:
        print(f"  id {r['id']}: pred={p} true={r['true']}")
json.dump(table, open(os.path.join(SAMPLES, "phase2b_llm_scored.json"), "w"), indent=2)
print("\nSaved phase2b_llm_subset.csv")
