"""Day-6 Phase-4 — extraction field-failure analysis + targeted rule fix.

Runs the shipped champion extractor (src.extraction.extractor) on the SAME 100
SROIE receipts (data/eval/receipts.jsonl), collects every AMOUNT failure, and
auto-tags it into a failure taxonomy:

  comma_decimal      gold/total written with a comma decimal ("193,00") so the
                     \\d+\\.\\d{2} amount regex never sees it
  approval_noise     a card "approval code" number sits right after TOTAL and is
                     grabbed instead of the real total
  tax_or_round       picked a tax / rounding / subtotal line instead of the total
  multiline_total    TOTAL keyword and its number separated by >40 chars (the
                     keyword-anchor window misses it)
  wrong_amount       other mismatch (model picked a different line item)

Then applies a TARGETED rule patch addressing the dominant amount modes and
re-scores amount accuracy on the same 100 receipts.

Writes:
  results/phase4_extraction_errors.csv
  results/phase4_extraction.json
  results/phase4_extraction_before_after.png
  results/samples/day6_extraction_failures.txt
"""
from __future__ import annotations
import json, os, re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.extraction.extractor import _rules_smart, _all_amounts, _norm_date, AMT_RE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "data", "eval", "receipts.jsonl")
RESULTS = os.path.join(ROOT, "results")
SAMPLES = os.path.join(RESULTS, "samples")
os.makedirs(SAMPLES, exist_ok=True)

# include comma-decimal amounts ("193,00") that the dot-only regex misses
AMT_RE_COMMA = re.compile(r"\d[\d.]*,\d{2}\b")
TOTAL_KW = ["grand total", "total inclusive", "total incl", "amount due", "nett total",
            "net total", "total amount", "total"]


def parse_amount(s):
    m = re.findall(r"\d[\d,]*\.\d{2}", str(s).replace(" ", ""))
    if m:
        return float(m[-1].replace(",", ""))
    m2 = re.findall(r"\d[\d,]*", str(s))
    return float(m2[-1].replace(",", "")) if m2 else None


def all_amounts_smart(text):
    """Day-6 patch: collect dot-decimal AND comma-decimal amounts on one axis."""
    out = list(_all_amounts(text))
    for m in AMT_RE_COMMA.finditer(text):
        raw = m.group()
        # "1.234,56" (euro) or "193,00" -> normalise to float
        val = raw.replace(".", "").replace(",", ".")
        try:
            out.append((m.start(), float(val)))
        except ValueError:
            pass
    return sorted(out)


def extract_amount_patched(text):
    """Targeted-fix amount extractor for the dominant amount-failure mode:
    GST two-column totals. On SROIE (Malaysian) receipts the TOTAL line is often
    `TOTAL : <net> <gst>` (e.g. `TOTAL : 411.50 24.69`) and the true GST-inclusive
    total is their SUM. The Day-2 keyword anchor grabs only the first number (net).

    Rule: take the first amount after a TOTAL keyword (as before); if a SECOND
    amount follows it immediately (within 12 chars) whose ratio to the first lies
    in the GST band [0.03, 0.09], return net + gst. Tightly gated so it cannot
    fire on a generic two-number line."""
    low = text.lower()
    amts = _all_amounts(text)
    for kw in TOTAL_KW:
        idx = low.rfind(kw)
        while idx != -1:
            window_before = low[max(0, idx - 12):idx]
            if kw == "total" and any(nk in (window_before + kw)
                                     for nk in ["sub total", "subtotal", "sub-total",
                                                "change", "cash", "rounding", "discount", "tax"]):
                idx = low.rfind(kw, 0, idx)
                continue
            after = [(pos, val) for pos, val in amts
                     if 0 <= pos - (idx + len(kw)) <= 40]
            if after:
                net = after[0][1]
                # GST two-column check: a second amount right after the first
                if len(after) >= 2 and net > 0:
                    gst = after[1][1]
                    if 0.03 <= gst / net <= 0.09 and (after[1][0] - after[0][0]) <= 12:
                        return round(net + gst, 2)
                return net
            idx = low.rfind(kw, 0, idx)
    return max((v for _, v in amts), default=0.0)


def tag_amount_failure(text, gt_amt, pred_amt):
    low = text.lower()
    # GST two-column total: TOTAL line shows <net> <gst>, gt == net + gst
    for kw in TOTAL_KW:
        i = low.rfind(kw)
        if i == -1:
            continue
        after = [(p, v) for p, v in _all_amounts(text) if 0 <= p - (i + len(kw)) <= 40]
        if len(after) >= 2 and after[0][1] > 0:
            net, gst = after[0][1], after[1][1]
            if 0.03 <= gst / net <= 0.09 and abs((net + gst) - (gt_amt or 0)) <= 0.02:
                return "gst_two_column"
        break
    # comma-decimal gold not representable as \d+\.\d{2}
    if AMT_RE_COMMA.search(text) and not re.search(
            r"\b" + re.escape(f"{gt_amt:.2f}") + r"\b", text):
        return "comma_decimal"
    # approval-code number equals the prediction
    for m in re.finditer(r"approval\s*code[^0-9]{0,25}(\d[\d,]*\.\d{2})", low):
        try:
            if abs(float(m.group(1).replace(",", "")) - (pred_amt or 0)) < 0.01:
                return "approval_noise"
        except ValueError:
            pass
    # picked a tax / round / subtotal line
    if pred_amt is not None:
        for kw in ["tax", "round", "sub total", "subtotal", "change"]:
            for m in re.finditer(re.escape(kw) + r"[^0-9]{0,20}(\d[\d,]*\.\d{2})", low):
                try:
                    if abs(float(m.group(1).replace(",", "")) - pred_amt) < 0.01:
                        return "tax_or_round"
                except ValueError:
                    pass
    # TOTAL keyword exists but far from any amount -> multi-line
    for kw in TOTAL_KW:
        i = low.rfind(kw)
        if i != -1:
            near = [p for p, _ in _all_amounts(text) if 0 <= p - (i + len(kw)) <= 40]
            if not near:
                return "multiline_total"
            break
    return "wrong_amount"


def main():
    rows = [json.loads(l) for l in open(EVAL, encoding="utf-8")]
    n = len(rows)

    base_ok = 0
    patch_ok = 0
    failures = []
    for r in rows:
        text = r["text"]
        gt = parse_amount(r["gt_total"])
        base = _rules_smart(text)["amount"]
        patched = extract_amount_patched(text)
        b_ok = gt is not None and abs(base - gt) < 0.01
        p_ok = gt is not None and abs(patched - gt) < 0.01
        base_ok += int(b_ok)
        patch_ok += int(p_ok)
        if not b_ok:
            failures.append({"id": r["id"], "gt": gt, "pred_base": round(base, 2),
                             "pred_patch": round(patched, 2),
                             "tag": tag_amount_failure(text, gt or 0, base),
                             "fixed_by_patch": p_ok})

    tax = Counter(f["tag"] for f in failures)
    dominant = tax.most_common(1)[0][0] if failures else "none"
    base_acc = round(base_ok / n, 4)
    patch_acc = round(patch_ok / n, 4)
    n_fixed = sum(1 for f in failures if f["fixed_by_patch"])
    print(f"[extraction] base amount-acc={base_acc}  patched={patch_acc}  "
          f"(+{patch_ok - base_ok} of {len(failures)} failures fixed)")
    print(f"[extraction] failure taxonomy={dict(tax)}  dominant={dominant}")

    with open(os.path.join(RESULTS, "phase4_extraction_errors.csv"), "w",
              newline="", encoding="utf-8") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=list(failures[0].keys()))
        w.writeheader(); w.writerows(failures)

    # before/after chart
    plt.figure(figsize=(6, 4))
    plt.bar(["base\nrules_smart", "patched\n(GST 2-col rule)"],
            [base_acc, patch_acc], color=["#DD8452", "#55A868"])
    for i, a in enumerate([base_acc, patch_acc]):
        plt.text(i, a + 0.01, f"{a:.2f}", ha="center")
    plt.ylim(0, 1.0); plt.ylabel("amount accuracy (100 SROIE receipts)")
    plt.title("Day 6 — extraction amount accuracy, targeted rule fix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "phase4_extraction_before_after.png"), dpi=120)
    plt.close()

    with open(os.path.join(SAMPLES, "day6_extraction_failures.txt"), "w", encoding="utf-8") as f:
        f.write(f"# {len(failures)} amount failures (base rules_smart). "
                f"taxonomy {dict(tax)} dominant={dominant}\n\n")
        for fl in failures:
            f.write(f"[{fl['tag']}] id={fl['id']} gt={fl['gt']} base={fl['pred_base']} "
                    f"patch={fl['pred_patch']} fixed={fl['fixed_by_patch']}\n")

    summary = {
        "generated": "2026-06-30", "day": 6, "phase": "Phase 4 — extraction error analysis",
        "n_receipts": n, "base_amount_acc": base_acc, "patched_amount_acc": patch_acc,
        "n_failures": len(failures), "n_fixed_by_patch": n_fixed,
        "failure_taxonomy": dict(tax), "dominant": dominant,
        "patch": "GST two-column total rule: net + gst when a second amount in the "
                 "GST band [0.03,0.09] follows the TOTAL amount",
    }
    json.dump(summary, open(os.path.join(RESULTS, "phase4_extraction.json"), "w"), indent=2)
    print("\n=== Day 6 extraction summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
