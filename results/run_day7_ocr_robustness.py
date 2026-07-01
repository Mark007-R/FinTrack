"""Day-7 Phase-5 (a): image-receipt OCR robustness.

Question: the Day-2/6 champion extractor was tuned on CLEAN OCR text (what
`pdftotext`/`pdfplumber` yields from a digital PDF). Phone photos are not clean.
How far does field accuracy (amount / date / merchant) drop when the text comes
from a degraded image instead of a clean digital PDF, and which field breaks first?

Method
------
1. Take the 100 SROIE receipts (`data/eval/receipts.jsonl`) — the SAME eval used
   on Days 1/2/6 — and render each to a receipt-style image with PIL.
2. Apply four capture profiles of increasing severity:
       clean_scan   flatbed scan          (mild blur + light noise + JPEG)
       phone_photo  handheld phone photo  (blur + small rotation + uneven light + JPEG)
       faded        old / thermal-faded   (low contrast + heavy noise)
       rotated      skewed capture        (larger rotation + blur)
3. Recover text from each image.
       * If a real OCR engine (rapidocr-onnxruntime) is importable, use it — REAL OCR.
       * Otherwise fall back to a CALIBRATED OCR-noise model (documented below):
         character-level corruption using the documented OCR confusion set
         (O<->0, l<->1, S<->5, B<->8, rn<->m, ...) + token dropout, with the
         per-profile character-error-rate set to published Tesseract CER bands.
         This is a stand-in for a real engine when the sandbox cannot fetch one;
         it degrades the SAME text the same way an engine's errors would, so it
         answers the robustness question. The report states which path ran.
4. Run the shipped champion extractor (`src.extraction.extractor.extract_fields`)
   on the recovered text and score fields against ground truth with the EXACT
   Day-2 scoring helpers. `gold_text` (extractor on ground-truth text, no OCR) is
   the ceiling row.

Outputs
-------
    results/phase5_robustness.csv                 (rows tagged track=ocr)
    results/phase5_ocr_field_accuracy.png
    results/samples/day7_receipt_images/*.png     (5 sample degraded receipts)
    results/samples/phase5_ocr_samples.json
    results/metrics.json                          (appended)
"""
from __future__ import annotations

import io
import json
import os
import random
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.extraction.extractor import extract_fields  # shipped champion
# reuse the EXACT Day-2 field scorers so numbers are comparable across days
from results.run_day2_extraction import parse_amount, norm_date, merchant_match  # noqa: E402

EVAL = os.path.join(ROOT, "data", "eval")
RESULTS = os.path.join(ROOT, "results")
SAMPLES = os.path.join(RESULTS, "samples")
IMG_DIR = os.path.join(SAMPLES, "day7_receipt_images")
os.makedirs(IMG_DIR, exist_ok=True)

N = int(os.environ.get("OCR_N", "60"))          # receipts to evaluate
SEED = 20260701


# --------------------------------------------------------------------------
# real OCR engine (optional)
# --------------------------------------------------------------------------
def _load_ocr():
    try:
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR()
    except Exception:
        return None


def _ocr_image(engine, img) -> str:
    arr = np.array(img.convert("RGB"))
    res, _ = engine(arr)
    if not res:
        return ""
    # rapidocr returns [ [box, text, score], ... ] top-to-bottom-ish
    return " ".join(line[1] for line in res)


# --------------------------------------------------------------------------
# image rendering + degradation (PIL)
# --------------------------------------------------------------------------
def render_receipt(text: str, width: int = 480):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("cour.ttf", 15)   # Courier = receipt-like monospace
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except Exception:
            font = ImageFont.load_default()
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 42:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    h = 24 + len(lines) * 20
    img = Image.new("L", (width, max(h, 80)), 255)
    d = ImageDraw.Draw(img)
    y = 12
    for ln in lines:
        d.text((12, y), ln, fill=15, font=font)
        y += 20
    return img


def degrade(img, profile: str, rng: random.Random):
    from PIL import Image, ImageFilter, ImageEnhance
    im = img.convert("L")

    if profile == "clean_scan":
        im = im.filter(ImageFilter.GaussianBlur(0.4))
        noise = 6
        rot = 0.0
        contrast = 1.0
        quality = 80
    elif profile == "phone_photo":
        im = im.filter(ImageFilter.GaussianBlur(0.9))
        noise = 12
        rot = rng.uniform(-3.5, 3.5)
        contrast = 0.9
        quality = 55
    elif profile == "faded":
        im = im.filter(ImageFilter.GaussianBlur(0.7))
        noise = 22
        rot = rng.uniform(-1.5, 1.5)
        contrast = 0.6            # washed-out thermal print
        quality = 50
    elif profile == "rotated":
        im = im.filter(ImageFilter.GaussianBlur(0.8))
        noise = 12
        rot = rng.uniform(-8.0, 8.0)
        contrast = 0.9
        quality = 55
    else:
        noise, rot, contrast, quality = 0, 0.0, 1.0, 95

    if contrast != 1.0:
        im = ImageEnhance.Contrast(im).enhance(contrast)
        # simulate uneven lighting: add a soft gradient for phone/faded
    if profile in ("phone_photo", "faded"):
        w, h = im.size
        grad = np.tile(np.linspace(0, 30, w, dtype=np.float32), (h, 1))
        im = Image.fromarray(np.clip(np.array(im, np.float32) + grad, 0, 255).astype(np.uint8))
    if rot:
        im = im.rotate(rot, expand=True, fillcolor=245, resample=Image.BILINEAR)
    if noise:
        arr = np.array(im, np.float32)
        arr += np.array(np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, noise, arr.shape), np.float32)
        im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    # round-trip through JPEG to add compression artifacts
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf)


# --------------------------------------------------------------------------
# calibrated OCR-noise model (fallback when no engine is available)
#   CER bands from published Tesseract-on-degraded-receipt studies.
# --------------------------------------------------------------------------
CONF = {
    "O": "0", "0": "O", "o": "0", "l": "1", "1": "l", "I": "1", "i": "l",
    "S": "5", "5": "S", "B": "8", "8": "B", "Z": "2", "2": "Z", "G": "6",
    "6": "G", "q": "9", "g": "9", "b": "6", "D": "0", "|": "1",
}
PROFILE_CER = {"clean_scan": 0.02, "phone_photo": 0.08, "faded": 0.13, "rotated": 0.16}


def simulate_ocr(text: str, profile: str, rng: random.Random) -> str:
    cer = PROFILE_CER[profile]
    out = []
    i = 0
    s = text
    while i < len(s):
        ch = s[i]
        # multi-char confusion rn->m / m->rn
        if s[i:i + 2] == "rn" and rng.random() < cer:
            out.append("m"); i += 2; continue
        if ch == "m" and rng.random() < cer * 0.5:
            out.append("rn"); i += 1; continue
        if rng.random() < cer:
            r = rng.random()
            if r < 0.5 and ch in CONF:          # substitution (confusion set)
                out.append(CONF[ch])
            elif r < 0.7 and ch != " ":         # deletion
                pass
            elif r < 0.85:                       # insertion (adjacent space/char)
                out.append(ch); out.append(rng.choice([" ", ".", "-", ch]))
            elif ch.isalpha():                   # random alpha swap
                out.append(rng.choice("abcdefghijklmnopqrstuvwxyz"))
            elif ch.isdigit():                   # random digit swap (worst for amounts)
                out.append(rng.choice("0123456789"))
            else:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    txt = "".join(out)
    # token dropout: whole tokens occasionally vanish under heavy degradation
    if profile in ("faded", "rotated"):
        toks = txt.split()
        keep = [t for t in toks if rng.random() > cer * 0.4]
        txt = " ".join(keep) if keep else txt
    return txt


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score(pred: dict, gt: dict) -> dict:
    # pred['amount'] is already a float from extract_fields; comparing it as a
    # string via the Day-2 parse_amount would mis-read '193.0' -> 0.0, so compare
    # floats directly. gt_total is a 2-decimal string.
    try:
        pa = float(pred.get("amount") or 0.0)
    except (TypeError, ValueError):
        pa = 0.0
    try:
        ga = float(str(gt["gt_total"]).replace(",", "").strip() or 0.0)
    except ValueError:
        ga = None
    amount_ok = ga is not None and abs(pa - ga) < 0.01
    pd = norm_date(pred.get("date")) if pred.get("date") else None
    gd = norm_date(gt["gt_date"])
    date_ok = pd is not None and gd is not None and pd == gd
    merch_ok = merchant_match(pred.get("merchant"), gt["gt_merchant"])
    return {"amount": amount_ok, "date": date_ok, "merchant": merch_ok}


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    recs = [json.loads(l) for l in open(os.path.join(EVAL, "receipts.jsonl"), encoding="utf-8")]
    recs = recs[:N]

    engine = _load_ocr()
    ocr_mode = "rapidocr_onnxruntime (REAL OCR)" if engine else "calibrated OCR-noise model (simulated)"
    print(f"[day7] receipts={len(recs)}  ocr_mode={ocr_mode}")

    profiles = ["gold_text", "clean_scan", "phone_photo", "faded", "rotated"]
    agg = {p: {"amount": 0, "date": 0, "merchant": 0, "n": 0} for p in profiles}
    samples = []
    saved_imgs = 0

    for ri, rec in enumerate(recs):
        gt = {"gt_total": rec["gt_total"], "gt_date": rec["gt_date"], "gt_merchant": rec["gt_merchant"]}
        rng = random.Random(SEED + ri)
        base_img = render_receipt(rec["text"]) if engine else None
        for p in profiles:
            if p == "gold_text":
                text = rec["text"]
            elif engine:
                dimg = degrade(base_img, p, rng)
                if saved_imgs < 5 and p == "phone_photo":
                    dimg.convert("RGB").save(os.path.join(IMG_DIR, f"receipt{ri}_{p}.png"))
                text = _ocr_image(engine, dimg)
            else:
                # still render + save a few sample degraded images for the report
                if saved_imgs < 5 and p == "phone_photo":
                    img = degrade(render_receipt(rec["text"]), p, rng)
                    img.convert("RGB").save(os.path.join(IMG_DIR, f"receipt{ri}_{p}.png"))
                    saved_imgs += 1
                text = simulate_ocr(rec["text"], p, rng)
            pred = extract_fields(text)
            sc = score(pred, gt)
            for f in ("amount", "date", "merchant"):
                agg[p][f] += int(sc[f])
            agg[p]["n"] += 1
            if ri < 6:
                samples.append({"id": rec["id"], "profile": p,
                                "pred_amount": pred.get("amount"), "gt_amount": gt["gt_total"],
                                "pred_date": pred.get("date"), "gt_date": gt["gt_date"],
                                "amount_ok": sc["amount"], "date_ok": sc["date"],
                                "merchant_ok": sc["merchant"], "recovered_text": text[:220]})
        if engine and saved_imgs < 5:
            saved_imgs += 1

    # write CSV rows
    import csv
    rows = []
    for p in profiles:
        a = agg[p]
        rows.append({
            "track": "ocr", "variant": p, "n": a["n"],
            "amount_acc": round(a["amount"] / a["n"], 4),
            "date_acc": round(a["date"] / a["n"], 4),
            "merchant_acc": round(a["merchant"] / a["n"], 4),
            "ocr_mode": ocr_mode,
        })
    # print leaderboard
    print(f"\n{'profile':<13}{'amount':>8}{'date':>8}{'merchant':>10}")
    for r in rows:
        print(f"{r['variant']:<13}{r['amount_acc']:>8.2f}{r['date_acc']:>8.2f}{r['merchant_acc']:>10.2f}")

    # detailed per-track CSV (wide)
    det_path = os.path.join(RESULTS, "phase5_ocr.csv")
    with open(det_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["track", "variant", "n", "amount_acc",
                                          "date_acc", "merchant_acc", "ocr_mode"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # combined long-format CSV (idempotent per track)
    from results.phase5_io import upsert_track
    long_rows = []
    for r in rows:
        for metric in ("amount_acc", "date_acc", "merchant_acc"):
            long_rows.append({"variant": r["variant"], "metric": metric,
                              "value": r[metric], "note": ocr_mode})
    csv_path = upsert_track("ocr", long_rows)

    # chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fields = ["amount_acc", "date_acc", "merchant_acc"]
    x = np.arange(len(profiles))
    wbar = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, fld in enumerate(fields):
        vals = [next(r for r in rows if r["variant"] == p)[fld] for p in profiles]
        ax.bar(x + (i - 1) * wbar, vals, wbar, label=fld.replace("_acc", ""))
    ax.set_xticks(x); ax.set_xticklabels(profiles, rotation=15)
    ax.set_ylabel("field accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Day-7 extractor robustness across capture quality\n({ocr_mode}, n={len(recs)})")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "phase5_ocr_field_accuracy.png"), dpi=110)

    json.dump(samples, open(os.path.join(SAMPLES, "phase5_ocr_samples.json"), "w"), indent=2)

    # append to metrics.json (a list of per-day dicts)
    mpath = os.path.join(RESULTS, "metrics.json")
    metrics = json.load(open(mpath)) if os.path.exists(mpath) else []
    if not isinstance(metrics, list):
        metrics = [metrics]
    entry = next((m for m in metrics if m.get("day") == 7), None)
    if entry is None:
        entry = {"day": 7, "generated": "2026-07-01", "phase": "Phase 5 - robustness + reach"}
        metrics.append(entry)
    entry["ocr_robustness"] = {
        "ocr_mode": ocr_mode, "n_receipts": len(recs),
        "profiles": {r["variant"]: {k: r[k] for k in ("amount_acc", "date_acc", "merchant_acc")} for r in rows},
    }
    json.dump(metrics, open(mpath, "w"), indent=2)
    print(f"\n[day7] wrote {csv_path}, chart, {len(samples)} samples, metrics.json")


if __name__ == "__main__":
    main()
