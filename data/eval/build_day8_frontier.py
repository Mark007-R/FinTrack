"""Day-8 FRESH held-out eval set builder (Phase 6 frontier comparison).

Two genuinely unseen held-out sets, disjoint from everything used Days 1-7:

  (1) frontier_receipts.jsonl  -- 50 SROIE receipts from the test split at
      OFFSET >= 100. Days 1-7 only ever touched the FIRST 100 test receipts
      (data/eval/receipts.jsonl), so offset>=100 is a clean held-out slice of
      the SAME public dataset. GT (merchant/date/total) is reconstructed from
      the B-COMPANY / B-DATE / B-TOTAL NER spans, identical to the Day-1 build.
      Pulled via the HF datasets-server rows API (the dataset SCRIPT no longer
      loads under datasets>=3, but the rows API serves the parsed rows fine).

  (2) frontier_transactions.csv -- 100 synthetic transactions from the SAME
      generator as the Day-1 training set but a NEW seed (20260702) AND an
      explicit dedupe against the 600 training descriptions, so not one row the
      champion categorizer trained on can appear in the held-out set.

MEDIA DISCIPLINE: public SROIE receipts + 100% synthetic transactions. No real
personal financial data.

Run:  python data/eval/build_day8_frontier.py
"""
import csv
import json
import os
import random
import time
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS_API = "https://datasets-server.huggingface.co/rows"


# ---------------------------------------------------------------- receipts ---
def _fetch_page(offset, length):
    q = urllib.parse.urlencode({
        "dataset": "darentang/sroie", "config": "sroie", "split": "test",
        "offset": offset, "length": length})
    url = f"{ROWS_API}?{q}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=45) as r:
                return json.load(r)["rows"]
        except Exception as e:
            print(f"  retry {attempt} offset={offset}: {repr(e)[:80]}")
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"rows API failed at offset {offset}")


def _spans_from_ner(words, tags):
    """Reconstruct COMPANY/DATE/TOTAL strings from B-/I- NER label ids.

    Label order (from the feature schema):
      0 O 1 B-COMPANY 2 I-COMPANY 3 B-DATE 4 I-DATE
      5 B-ADDRESS 6 I-ADDRESS 7 B-TOTAL 8 I-TOTAL
    """
    name = {1: "company", 2: "company", 3: "date", 4: "date",
            7: "total", 8: "total"}
    out = {"company": [], "date": [], "total": []}
    for w, t in zip(words, tags):
        key = name.get(int(t))
        if key:
            out[key].append(w)
    company = " ".join(out["company"]).strip()
    date = " ".join(out["date"]).strip()
    total = " ".join(out["total"]).strip().replace(" ", "")  # "193. 00" -> "193.00"
    return company, date, total


def build_receipts(n_target=50, start_offset=100):
    receipts, offset, page = [], start_offset, 60
    while len(receipts) < n_target and offset < 340:
        rows = _fetch_page(offset, page)
        if not rows:
            break
        for item in rows:
            row = item["row"]
            words = row["words"]
            tags = row["ner_tags"]
            company, date, total = _spans_from_ner(words, tags)
            # keep only receipts with a parseable labeled total (same rule Day 1 used)
            try:
                float(total.replace(",", ""))
            except (ValueError, AttributeError):
                continue
            if not total:
                continue
            text = " ".join(words)
            receipts.append({
                "id": f"f{item['row_idx']}",  # 'f' prefix => cannot collide with Day-1 ids
                "text": text,
                "gt_merchant": company,
                "gt_date": date,
                "gt_total": total.replace(",", ""),
            })
            if len(receipts) >= n_target:
                break
        offset += page
    out = os.path.join(HERE, "frontier_receipts.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in receipts:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out}  n={len(receipts)}  (SROIE test offset>={start_offset}, held-out)")
    return receipts


# ------------------------------------------------------------ transactions ---
# Import the EXACT generator pieces from the Day-1 builder so the held-out set is
# drawn from the identical distribution (only the seed + dedupe differ).
import build_transactions as bt  # noqa: E402

# NOVEL merchants — real brands DELIBERATELY absent from the Day-1 training
# vocabulary (bt.MERCHANTS). These create an out-of-distribution regime: the
# champion TF-IDF+char model never saw these tokens, so this is precisely where
# world-knowledge (the LLM) is expected to win. Chosen so the category is NOT
# inferable from a substring keyword either (keyword baseline also fails here).
NOVEL = {
    "groceries": ["LIDL", "WEGMANS", "FOOD LION", "STOP N SHOP", "WINCO FOODS", "MEIJER", "HARRIS TEETER"],
    "dining": ["SHAKE SHACK", "SWEETGREEN", "FIVE GUYS", "WENDYS", "IN-N-OUT", "CHICK-FIL-A", "RAISING CANES"],
    "transport": ["SUNOCO", "CITGO", "TURO", "SPIRIT AIR", "CALTRAIN", "FLIXBUS", "ZIPCAR"],
    "utilities": ["DOMINION ENERGY", "COX COMM", "FRONTIER COMM", "PSE&G", "CENTURYLINK", "XCEL"],
    "entertainment": ["PARAMOUNT PLUS", "PEACOCK TV", "CINEMARK", "TWITCH", "CRUNCHYROLL", "FANDANGO"],
    "health": ["MINUTECLINIC", "DAVITA", "TELADOC", "CIGNA", "AETNA", "GOODRX"],
    "shopping": ["SHEIN", "TEMU", "OVERSTOCK", "CHEWY", "REI CO-OP", "LOWES", "ULTA"],
    "income": ["GUSTO", "ADP", "CASH APP FROM M R", "WISE TRANSFER IN"],
    "rent": ["CAMDEN LIVING", "UDR HOMES", "MAA COMMUNITIES", "ESSEX PROPERTY"],
    "other": ["COINBASE", "ROBINHOOD", "WESTERN UNION", "MONEYGRAM", "CASH APP"],
}


def build_transactions(n_target=100, novel_frac=0.4, seed=20260702):
    """Held-out set = (1-novel_frac) in-distribution + novel_frac OOV merchants.

    Every row is tagged `regime` in {in_dist, novel} so Day-8 can report the
    crossover: where the $0 local model matches the LLM (in-distribution) vs
    where it collapses and the LLM's world knowledge wins (novel merchants).
    """
    train_path = os.path.join(HERE, "transactions.csv")
    seen = set()
    if os.path.exists(train_path):
        for r in csv.DictReader(open(train_path, encoding="utf-8")):
            seen.add(r["description"].strip().lower())

    random.seed(seed)
    cats = list(bt.WEIGHTS)
    wts = [bt.WEIGHTS[c] for c in cats]
    n_novel = int(round(n_target * novel_frac))
    rows, i, start_day, guard = [], 0, 0, 0
    while len(rows) < n_target and guard < n_target * 400:
        guard += 1
        cat = random.choices(cats, weights=wts, k=1)[0]
        regime = "novel" if sum(1 for r in rows if r["regime"] == "novel") < n_novel and random.random() < 0.5 else "in_dist"
        pool = NOVEL[cat] if regime == "novel" else bt.MERCHANTS[cat]
        base = random.choice(pool)
        desc = bt.make_desc(base)
        if desc.strip().lower() in seen:
            continue  # dedupe vs training set + within-set
        seen.add(desc.strip().lower())
        amt = bt.amount_for(cat)
        start_day += random.randint(0, 2)
        day = 1 + (start_day % 28)
        month = 1 + (start_day // 28) % 12
        rows.append({"id": i, "description": desc, "amount": amt,
                     "date": f"2025-{month:02d}-{day:02d}", "category": cat, "regime": regime})
        i += 1
    # top up novel if the coin flips left us short
    out = os.path.join(HERE, "frontier_transactions.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "description", "amount", "date", "category", "regime"])
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"wrote {out}  n={len(rows)}  (seed={seed}, deduped vs training)")
    print("  class counts:", dict(Counter(r["category"] for r in rows)))
    print("  regime counts:", dict(Counter(r["regime"] for r in rows)))
    return rows


if __name__ == "__main__":
    build_receipts()
    build_transactions()
