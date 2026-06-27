"""Day-3 Phase-2b: expense-categorizer bake-off (FinTrack upgrade).

Same labeled transaction eval as Day 1 (data/eval/transactions.csv, 600 rows, 10 classes).
Stratified 70/30 split. Trained models learn on train; ALL methods are scored on the SAME
held-out test split:

  1. keyword       -- the Day-1 keyword baseline (run_baselines.KEYWORDS), no training
  2. tfidf_lgbm    -- TF-IDF (word 1-2grams) + LightGBM multiclass
  3. sbert_lgbm    -- all-MiniLM-L6-v2 embeddings + LightGBM head
  4. distilbert    -- distilbert-base-uncased fine-tuned (CPU, 4 epochs)

A 60-row stratified subset of the test split is dumped (descriptions only) for a blind
Claude zero-shot head-to-head scored later by score_llm_categorize.py; every method's
macro-F1 is also recomputed on that exact 60-row subset so the LLM row is comparable.

Outputs:
    results/phase2b_categorize.csv               (full test leaderboard + per-class F1)
    results/phase2b_categorize_macroF1.png
    results/samples/_llm_cat_blind_input.json    (60 descriptions, no labels)
    results/samples/phase2b_subset_truth.json    (hidden labels + every model's subset preds)
    results/samples/phase2b_samples.json         (per-method misclassification samples)
    results/metrics.json                         (appended)
"""
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "results"))
sys.path.insert(0, ROOT)
from run_baselines import KEYWORDS, keyword_predict  # exact Day-1 baseline

EVAL = os.path.join(ROOT, "data", "eval")
RESULTS = os.path.join(ROOT, "results")
SAMPLES = os.path.join(RESULTS, "samples")
os.makedirs(SAMPLES, exist_ok=True)
SEED = 20260627


def load():
    rows = list(csv.DictReader(open(os.path.join(EVAL, "transactions.csv"), encoding="utf-8")))
    return rows


def stratified_split(rows, test_frac=0.30, seed=SEED):
    from collections import defaultdict
    rng = np.random.RandomState(seed)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    train, test = [], []
    for cat, items in by_cat.items():
        idx = rng.permutation(len(items))
        k = max(1, int(round(len(items) * test_frac)))
        test += [items[i] for i in idx[:k]]
        train += [items[i] for i in idx[k:]]
    return train, test


def per_class_f1(y_true, y_pred, labels):
    from sklearn.metrics import f1_score
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)
    return round(float(macro), 4), round(float(acc), 4), {c: round(float(v), 4) for c, v in zip(labels, f1s)}


# ----------------------- models -----------------------
def _tfidf_union():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    # word + char_wb: char n-grams are essential for short, high-cardinality merchant
    # strings where test merchants are largely unseen in train (OOV at the word level).
    return FeatureUnion([
        ("w", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ])


def fit_tfidf_lgbm(train_desc, y_train):
    import lightgbm as lgb
    vec = _tfidf_union()
    Xtr = vec.fit_transform(train_desc)
    # min_child_samples=5 is critical: with the default 20, LightGBM cannot split the
    # ultra-sparse TF-IDF matrix and collapses (macro-F1 ~0.12, fails to fit train).
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                             min_child_samples=5, class_weight="balanced",
                             random_state=SEED, verbose=-1)
    clf.fit(Xtr, y_train)
    return lambda descs: list(clf.predict(vec.transform(descs)))


def fit_tfidf_linsvc(train_desc, y_train):
    from sklearn.svm import LinearSVC
    vec = _tfidf_union()
    Xtr = vec.fit_transform(train_desc)
    clf = LinearSVC(class_weight="balanced", C=1.0)
    clf.fit(Xtr, y_train)
    return lambda descs: list(clf.predict(vec.transform(descs)))


def fit_sbert_lgbm(train_desc, y_train):
    from sentence_transformers import SentenceTransformer
    import lightgbm as lgb
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    Xtr = enc.encode(train_desc, normalize_embeddings=True, show_progress_bar=False)
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                             class_weight="balanced", random_state=SEED, verbose=-1)
    clf.fit(Xtr, y_train)
    return lambda descs: list(clf.predict(enc.encode(descs, normalize_embeddings=True, show_progress_bar=False)))


def fit_distilbert(train_desc, y_train, labels):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    lab2id = {c: i for i, c in enumerate(labels)}
    tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    enc = tok(train_desc, truncation=True, padding="max_length", max_length=32, return_tensors="pt")
    yt = torch.tensor([lab2id[y] for y in y_train])
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], yt)
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(labels))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    # class weights to help rare classes
    from collections import Counter
    cnt = Counter(yt.tolist())
    w = torch.tensor([len(yt) / (len(labels) * cnt.get(i, 1)) for i in range(len(labels))], dtype=torch.float)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    for epoch in range(4):
        for ids, mask, yy in dl:
            opt.zero_grad()
            out = model(input_ids=ids, attention_mask=mask)
            loss = lossf(out.logits, yy)
            loss.backward()
            opt.step()
    model.eval()
    id2lab = {i: c for c, i in lab2id.items()}

    def predict(descs):
        e = tok(list(descs), truncation=True, padding="max_length", max_length=32, return_tensors="pt")
        with torch.no_grad():
            logits = model(input_ids=e["input_ids"], attention_mask=e["attention_mask"]).logits
        return [id2lab[int(i)] for i in logits.argmax(1)]
    return predict


def main():
    rows = load()
    labels = sorted(set(r["category"] for r in rows))
    train, test = stratified_split(rows)
    tr_desc = [r["description"] for r in train]; y_tr = [r["category"] for r in train]
    te_desc = [r["description"] for r in test]; y_te = [r["category"] for r in test]
    print(f"train={len(train)} test={len(test)} classes={len(labels)}")

    methods = {}
    methods["keyword (baseline)"] = (keyword_predict_batch := lambda descs: [keyword_predict(d) for d in descs], 0.0)

    timings = {}
    preds = {}
    # keyword
    t0 = time.time(); preds["keyword (baseline)"] = [keyword_predict(d) for d in te_desc]; timings["keyword (baseline)"] = time.time() - t0

    for name, fitter in [("tfidf_lgbm", fit_tfidf_lgbm), ("tfidf_linsvc", fit_tfidf_linsvc),
                          ("sbert_lgbm", fit_sbert_lgbm)]:
        t0 = time.time(); fn = fitter(tr_desc, y_tr); preds[name] = fn(te_desc); timings[name] = time.time() - t0
        print(f"  fit+pred {name} in {timings[name]:.1f}s")

    t0 = time.time(); fn_db = fit_distilbert(tr_desc, y_tr, labels); preds["distilbert"] = fn_db(te_desc); timings["distilbert"] = time.time() - t0
    print(f"  fit+pred distilbert in {timings['distilbert']:.1f}s")

    # ---- full-test leaderboard ----
    table = []
    perclass = {}
    for name in ["keyword (baseline)", "tfidf_lgbm", "tfidf_linsvc", "sbert_lgbm", "distilbert"]:
        macro, acc, pcf1 = per_class_f1(y_te, preds[name], labels)
        table.append({"method": name, "n_test": len(y_te), "macro_f1": macro, "accuracy": acc,
                      "fit_pred_sec": round(timings[name], 2)})
        perclass[name] = pcf1
        print(f"{name:20s} macro-F1={macro:.4f} acc={acc:.4f}")

    # write CSV (leaderboard + per-class)
    cols = ["method", "n_test", "macro_f1", "accuracy", "fit_pred_sec"] + labels
    with open(os.path.join(RESULTS, "phase2b_categorize.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in table:
            r = dict(row); r.update(perclass[row["method"]]); w.writerow(r)

    # ---- 60-row stratified LLM subset of the test set ----
    from collections import defaultdict
    rng = np.random.RandomState(SEED + 1)
    by_cat = defaultdict(list)
    for r in test:
        by_cat[r["category"]].append(r)
    subset = []
    per = max(1, 60 // len(labels))
    for cat, items in by_cat.items():
        idx = rng.permutation(len(items))[:per]
        subset += [items[i] for i in idx]
    # pad up toward 60 from remaining test rows
    remaining = [r for r in test if r not in subset]
    rng.shuffle(remaining)
    subset += remaining[:max(0, 60 - len(subset))]
    sub_ids = [r["id"] for r in subset]
    sub_desc = [r["description"] for r in subset]
    sub_true = [r["category"] for r in subset]

    json.dump([{"id": r["id"], "description": r["description"]} for r in subset],
              open(os.path.join(SAMPLES, "_llm_cat_blind_input.json"), "w"), indent=2)
    # hidden truth + every model's predictions on the subset (for a comparable sub-table)
    sub_preds = {name: dict(zip([r["id"] for r in test], preds[name])) for name in preds}
    truth_blob = {"labels": labels,
                  "subset": [{"id": r["id"], "true": r["category"],
                              "keyword": sub_preds["keyword (baseline)"][r["id"]],
                              "tfidf_lgbm": sub_preds["tfidf_lgbm"][r["id"]],
                              "tfidf_linsvc": sub_preds["tfidf_linsvc"][r["id"]],
                              "sbert_lgbm": sub_preds["sbert_lgbm"][r["id"]],
                              "distilbert": sub_preds["distilbert"][r["id"]]} for r in subset]}
    json.dump(truth_blob, open(os.path.join(SAMPLES, "phase2b_subset_truth.json"), "w"), indent=2)

    # ---- misclassification samples (where keyword fails but a model gets it) ----
    msamples = []
    kw_test = dict(zip([r["id"] for r in test], preds["keyword (baseline)"]))
    tf_test = dict(zip([r["id"] for r in test], preds["tfidf_lgbm"]))
    for r in test:
        if kw_test[r["id"]] != r["category"] and tf_test[r["id"]] == r["category"]:
            msamples.append({"description": r["description"], "true": r["category"],
                             "keyword_pred": kw_test[r["id"]], "tfidf_lgbm_pred": tf_test[r["id"]]})
        if len(msamples) >= 12:
            break
    json.dump(msamples, open(os.path.join(SAMPLES, "phase2b_samples.json"), "w"), indent=2)

    # ---- chart: macro-F1 + the two keyword-blindspot classes ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [t["method"] for t in table]
    macro = [t["macro_f1"] for t in table]
    util = [perclass[n]["utilities"] for n in names]
    other = [perclass[n]["other"] for n in names]
    x = np.arange(len(names)); wd = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(x - wd, macro, wd, label="macro-F1")
    ax.bar(x, util, wd, label="utilities F1 (kw blindspot)")
    ax.bar(x + wd, other, wd, label="other F1")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1")
    ax.set_title("FinTrack Day 3 — expense categorizer: macro-F1 + keyword blind-spot classes (test n=%d)" % len(y_te))
    for i, v in enumerate(macro):
        ax.text(x[i] - wd, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "phase2b_categorize_macroF1.png"), dpi=130)

    # ---- append metrics.json ----
    from datetime import datetime
    mpath = os.path.join(RESULTS, "metrics.json")
    blob = json.load(open(mpath)) if os.path.exists(mpath) else []
    if isinstance(blob, dict):
        blob = [blob]
    blob.append({"generated": datetime.now().strftime("%Y-%m-%d"), "day": 3,
                 "phase": "Phase 2b - expense categorizer comparison",
                 "split": {"train": len(train), "test": len(test), "seed": SEED},
                 "leaderboard": table, "per_class_f1": perclass})
    json.dump(blob, open(mpath, "w"), indent=2)
    print("\nSaved phase2b_categorize.csv / .png / subset / samples / metrics.json")


if __name__ == "__main__":
    main()
