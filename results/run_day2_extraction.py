"""Day-2 Phase-2a: receipt-extraction bake-off (FinTrack upgrade).

Compares field-level extraction (amount / date / merchant) on the SAME 100
SROIE receipts used for the Day-1 baseline:

  1. regex          -- the real find_bill_details (extract_bill.py:43); first \\d+\\.\\d{2}
  2. rules_basic    -- "pdfplumber+rules" naive tier: largest amount + first date
  3. rules_smart    -- keyword-anchored TOTAL detection + merchant + Date-anchored date
  4. donut          -- naver-clova-ix/donut-base-finetuned-cord-v2 (OCR-free deep model)

The eval input is OCR text (what poppler/pdfplumber yields from a PDF). Donut is an
image model, so each receipt's OCR text is rendered to a clean image and fed to Donut
-- flagged as an approximation (real photos differ). LayoutLMv3 needs a SROIE-finetuned
token-classification head + per-token boxes, which we do not have offline; it is documented
as deferred in the report rather than faked.

Outputs:
    results/phase2a_extraction.csv
    results/phase2a_extraction_field_accuracy.png
    results/samples/phase2a_extraction_samples.json
    results/metrics.json   (appended)
"""
import json
import os
import re
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from extract_bill import find_bill_details  # the real regex under audit

EVAL = os.path.join(ROOT, "data", "eval")
RESULTS = os.path.join(ROOT, "results")
SAMPLES = os.path.join(RESULTS, "samples")
os.makedirs(SAMPLES, exist_ok=True)

DONUT_N = int(os.environ.get("DONUT_N", "50"))  # cap deep-model rows for CPU runtime


# ----------------------- shared helpers -----------------------
def parse_amount(s):
    m = re.findall(r"\d[\d,]*\.\d{2}", str(s).replace(" ", ""))
    if not m:
        m2 = re.findall(r"\d[\d,]*", str(s))
        return float(m2[-1].replace(",", "")) if m2 else None
    return float(m[-1].replace(",", ""))


def norm_date(s):
    s = str(s).strip()
    for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y",
                "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(s.split()[0] if " " in s else s, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    m = re.search(r"\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}", s)
    if m:
        for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(m.group(), fmt).strftime("%Y-%m-%d")
            except Exception:
                continue
    return None


def norm_merchant(s):
    if not s:
        return ""
    s = re.sub(r"[^A-Za-z0-9 ]", " ", str(s)).upper()
    s = re.sub(r"\s+", " ", s).strip()
    return s


MERCH_STOP = {"SDN", "BHD", "ROC", "NO", "PTE", "LTD", "CO", "ENTERPRISE", "TRADING",
              "TAX", "INVOICE", "RECEIPT", "THE", "AND"}


def merchant_match(pred, gt):
    """Token-overlap match: >=50% of gt's meaningful tokens appear in pred."""
    p, g = norm_merchant(pred), norm_merchant(gt)
    if not p or not g:
        return False
    gt_tok = [t for t in g.split() if t not in MERCH_STOP and len(t) > 1]
    if not gt_tok:
        gt_tok = g.split()
    pset = set(p.split())
    hit = sum(1 for t in gt_tok if t in pset)
    return hit / len(gt_tok) >= 0.5


AMT_RE = re.compile(r"\d[\d,]*\.\d{2}")


def all_amounts(text):
    out = []
    for m in AMT_RE.finditer(text):
        try:
            out.append((m.start(), float(m.group().replace(",", ""))))
        except ValueError:
            pass
    return out


# ----------------------- extractors -----------------------
def x_regex(text):
    amt, date = find_bill_details(text)
    return {"amount": abs(amt), "date": date if date != "Not found" else None, "merchant": None}


def x_rules_basic(text):
    """pdfplumber+rules naive tier: largest monetary value + first parseable date."""
    amts = [a for _, a in all_amounts(text)]
    amount = max(amts) if amts else 0.0
    d = re.search(r"\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}", text)
    return {"amount": amount, "date": norm_date(d.group()) if d else None, "merchant": None}


TOTAL_KW = ["grand total", "total inclusive", "total incl", "amount due", "nett total",
            "net total", "total amount", "total"]
NEG_KW = ["sub total", "subtotal", "sub-total", "change", "cash", "rounding", "discount", "tax"]


def x_rules_smart(text):
    """Keyword-anchored extraction: amount near a TOTAL keyword (not subtotal/tax/change),
    merchant from the opening company line, date near a 'Date' keyword."""
    low = text.lower()
    amts = all_amounts(text)
    amount = 0.0
    # find amount following the strongest total keyword, avoiding negative keywords nearby
    for kw in TOTAL_KW:
        idx = low.rfind(kw)
        while idx != -1:
            window_before = low[max(0, idx - 12):idx]
            if any(nk in (window_before + kw) for nk in NEG_KW) and kw == "total":
                idx = low.rfind(kw, 0, idx)
                continue
            after = [(pos, val) for pos, val in amts if 0 <= pos - (idx + len(kw)) <= 40]
            if after:
                amount = after[0][1]
                break
            idx = low.rfind(kw, 0, idx)
        if amount:
            break
    if not amount and amts:
        amount = max(a for _, a in amts)

    # date: prefer one right after a 'date' label, else first in doc
    date = None
    dm = re.search(r"date[^0-9]{0,8}(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})", low)
    if dm:
        date = norm_date(dm.group(1))
    if not date:
        d = re.search(r"\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}", text)
        date = norm_date(d.group()) if d else None

    # merchant: first ALL-CAPS-ish company chunk before ROC/TAX INVOICE/address noise
    merchant = None
    head = text[:200]
    head = re.split(r"ROC NO|TAX INVOICE|INVOICE NO|\bROC\b", head, flags=re.I)[0]
    caps = re.findall(r"\b[A-Z][A-Z&.\- ]{3,}(?:SDN BHD|BHD|ENTERPRISE|TRADING|PTE LTD|LTD|RESTAURANT|CAFE|MART|SUPERMARKET)?\b", head)
    caps = [c.strip() for c in caps if len(c.strip()) > 4 and "COPY" not in c]
    if caps:
        merchant = max(caps, key=len).strip()
    return {"amount": amount, "date": date, "merchant": merchant}


# ----------------------- Donut -----------------------
_donut = {}


def render_text_image(text, width=720):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 60:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    h = 30 + len(lines) * 22
    img = Image.new("RGB", (width, max(h, 80)), "white")
    d = ImageDraw.Draw(img)
    y = 15
    for ln in lines:
        d.text((15, y), ln, fill="black", font=font)
        y += 22
    return img


def x_donut(text):
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    if "model" not in _donut:
        name = "naver-clova-ix/donut-base-finetuned-cord-v2"
        _donut["proc"] = DonutProcessor.from_pretrained(name)
        _donut["model"] = VisionEncoderDecoderModel.from_pretrained(name)
        _donut["model"].eval()
    proc, model = _donut["proc"], _donut["model"]
    image = render_text_image(text).convert("RGB")
    pixel_values = proc(image, return_tensors="pt").pixel_values
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = proc.tokenizer(task_prompt, add_special_tokens=False,
                                       return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(pixel_values, decoder_input_ids=decoder_input_ids,
                             max_length=768, early_stopping=True, pad_token_id=proc.tokenizer.pad_token_id,
                             eos_token_id=proc.tokenizer.eos_token_id, use_cache=True,
                             num_beams=1, bad_words_ids=[[proc.tokenizer.unk_token_id]])
    seq = proc.batch_decode(out)[0]
    seq = seq.replace(proc.tokenizer.eos_token, "").replace(proc.tokenizer.pad_token, "")
    seq = re.sub(r"<.*?>", " ", seq)  # strip xml-ish tags into the raw json fallback
    js = {}
    try:
        js = proc.token2json(proc.batch_decode(out)[0])
    except Exception:
        js = {}
    amount = 0.0
    tot = js.get("total") if isinstance(js, dict) else None
    if isinstance(tot, dict):
        for k in ("total_price", "total", "cashprice", "menuqty_cnt"):
            if k in tot:
                pa = parse_amount(tot[k])
                if pa:
                    amount = pa
                    break
    if not amount:
        cand = all_amounts(seq)
        amount = max((v for _, v in cand), default=0.0)
    return {"amount": amount, "date": None, "merchant": None}


# ----------------------- evaluation -----------------------
EXTRACTORS = [
    ("regex (baseline)", x_regex, None),
    ("rules_basic (pdfplumber+rules)", x_rules_basic, None),
    ("rules_smart (keyword-anchored)", x_rules_smart, None),
    ("donut-cord-v2 (deep, rendered img)", x_donut, DONUT_N),
]


def run():
    rows = [json.loads(l) for l in open(os.path.join(EVAL, "receipts.jsonl"), encoding="utf-8")]
    table = []
    samples = {}
    for name, fn, limit in EXTRACTORS:
        subset = rows[:limit] if limit else rows
        n = len(subset)
        amt_ok = date_ok = merch_ok = both_ok = 0
        t0 = time.time()
        ex_samples = []
        for r in subset:
            try:
                pred = fn(r["text"])
            except Exception as e:
                pred = {"amount": 0.0, "date": None, "merchant": None, "error": str(e)}
            gt_amt = parse_amount(r["gt_total"])
            gt_date = norm_date(r["gt_date"])
            a_ok = gt_amt is not None and pred.get("amount") is not None and abs(pred["amount"] - gt_amt) < 0.01
            d_ok = gt_date is not None and pred.get("date") == gt_date
            m_ok = merchant_match(pred.get("merchant"), r["gt_merchant"])
            amt_ok += a_ok; date_ok += d_ok; merch_ok += m_ok; both_ok += (a_ok and d_ok)
            if len(ex_samples) < 8:
                ex_samples.append({"id": r["id"], "gt_amount": gt_amt, "pred_amount": pred.get("amount"),
                                   "amount_ok": a_ok, "gt_date": gt_date, "pred_date": pred.get("date"),
                                   "date_ok": d_ok, "gt_merchant": r["gt_merchant"],
                                   "pred_merchant": pred.get("merchant"), "merchant_ok": m_ok})
        dt = time.time() - t0
        row = {"method": name, "n": n,
               "amount_acc": round(amt_ok / n, 4), "date_acc": round(date_ok / n, 4),
               "merchant_acc": round(merch_ok / n, 4),
               "exact_match_amt_date": round(both_ok / n, 4),
               "sec_per_doc": round(dt / n, 4)}
        table.append(row)
        samples[name] = ex_samples
        print(f"{name:42s} n={n:3d}  amt={row['amount_acc']:.2f}  date={row['date_acc']:.2f}  "
              f"merch={row['merchant_acc']:.2f}  exact={row['exact_match_amt_date']:.2f}  {row['sec_per_doc']:.3f}s/doc")
    return table, samples


def main():
    table, samples = run()
    # CSV
    import csv
    cols = ["method", "n", "amount_acc", "date_acc", "merchant_acc", "exact_match_amt_date", "sec_per_doc"]
    with open(os.path.join(RESULTS, "phase2a_extraction.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    # samples
    with open(os.path.join(SAMPLES, "phase2a_extraction_samples.json"), "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    # chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    methods = [r["method"].split(" (")[0] for r in table]
    fields = ["amount_acc", "date_acc", "merchant_acc"]
    x = np.arange(len(methods)); w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, fld in enumerate(fields):
        ax.bar(x + (i - 1) * w, [r[fld] for r in table], w, label=fld.replace("_acc", ""))
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=12, ha="right", fontsize=8)
    ax.set_ylabel("field accuracy"); ax.set_ylim(0, 1)
    ax.set_title("FinTrack Day 2 — receipt extraction: field accuracy by method (SROIE, n=100/50)")
    ax.legend()
    for i, fld in enumerate(fields):
        for j, r in enumerate(table):
            ax.text(x[j] + (i - 1) * w, r[fld] + 0.01, f"{r[fld]:.2f}", ha="center", va="bottom", fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "phase2a_extraction_field_accuracy.png"), dpi=130)
    # append metrics.json
    mpath = os.path.join(RESULTS, "metrics.json")
    blob = json.load(open(mpath)) if os.path.exists(mpath) else []
    if isinstance(blob, dict):
        blob = [blob]
    blob.append({"generated": datetime.now().strftime("%Y-%m-%d"), "day": 2,
                 "phase": "Phase 2a - receipt extraction comparison", "results": table})
    json.dump(blob, open(mpath, "w"), indent=2)
    print("\nSaved phase2a_extraction.csv / .png / samples / metrics.json")


if __name__ == "__main__":
    main()
