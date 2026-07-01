"""Day-7 Phase-5 (b, part 2): multilingual / edge-receipt reach for the extractor.

The Day-2/6 champion extractor anchors the amount on English TOTAL keywords
(`total`, `amount due`, ...) and the date on an English `date` label. Real users
photograph non-English receipts. This measures the gap and closes part of it with
a small multilingual gazetteer (a "reach" add, not a re-architecture).

Synthetic receipts (public brand names + fabricated line items -> media-discipline
OK) in Spanish / French / German, each with a known total / date / merchant. German
is the hard case: its total label is `Gesamtbetrag`/`Summe`/`Betrag` and its date
label is `Datum` — neither contains the English anchor, so the shipped extractor
falls back to `max(amount)` and mis-reads the date.

  baseline    -> shipped `extract_fields` (English anchors only)
  +gazetteer  -> same logic + multilingual TOTAL / date labels

Regression guard: the multilingual tokens are re-checked on the 60 English SROIE
receipts to confirm they never fire on English (so shipping them is safe).

Outputs
-------
    results/phase5_multilingual.csv
    results/samples/phase5_multilingual_samples.json
    results/phase5_robustness.csv   (track=multilingual summary rows)
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from datetime import datetime

from src.extraction.extractor import extract_fields, _rules_smart, _all_amounts
from results.run_day2_extraction import norm_date, merchant_match
from results.phase5_io import upsert_track


def _norm_date_ml(s):
    """Day-7 finding: European receipts use dot-separated day-first dates
    (11.03.2024) that the shipped `_norm_date` never tries — it only parses
    '/' and '-'. Normalise dots and try day-first + ISO formats."""
    if not s:
        return None
    s = str(s).strip().split()[0]
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y-%m-%d",
                "%d.%m.%y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return None

RESULTS = os.path.join(ROOT, "results")
SAMPLES = os.path.join(RESULTS, "samples")
os.makedirs(SAMPLES, exist_ok=True)

# multilingual gazetteer (the reach add). English tokens omitted — they are already
# in the shipped extractor. These are lowercase substrings checked against low-text.
ML_TOTAL_KW = ["gesamtbetrag", "gesamtsumme", "gesamt", "summe", "betrag zu zahlen",
               "endbetrag", "importe total", "total a pagar", "montant total",
               "total ttc", "net a payer", "totale", "importo totale"]
ML_DATE_LABELS = ["fecha", "datum", "date"]

# --- synthetic multilingual receipts: (lang, merchant, gt_total, gt_date, text) ---
RECEIPTS = [
    ("de", "REWE MARKT GMBH", "23.47", "2024-03-11",
     "REWE MARKT GMBH Koenigstrasse 12 70173 Stuttgart Datum: 11.03.2024 "
     "Bio Milch 1.29 Broetchen 0.85 Kaffee 6.49 Zwischensumme 8.63 MwSt 7% 0.60 "
     "Gesamtbetrag 23.47 EC-Karte 23.47 Vielen Dank"),
    ("de", "MUELLER DROGERIE", "14.90", "2024-05-02",
     "MUELLER DROGERIE Marienplatz 8 Datum 02.05.2024 Shampoo 3.95 Zahnpasta 2.49 "
     "Seife 1.99 Netto 12.52 MwSt 19% 2.38 Summe 14.90 Bar 20.00 Rueckgeld 5.10"),
    ("de", "EDEKA SUEDWEST", "51.08", "2024-01-19",
     "EDEKA SUEDWEST Datum: 19.01.2024 Rindfleisch 12.90 Gemuese 8.44 Wein 9.99 "
     "Kaese 6.75 Zwischensumme 46.30 Pfand 1.00 Gesamtsumme 51.08 Kreditkarte 51.08"),
    ("de", "DM DROGERIEMARKT", "8.75", "2024-06-07",
     "DM-DROGERIE MARKT Datum 07.06.2024 Vitamine 5.95 Creme 2.80 Endbetrag 8.75 "
     "Kartenzahlung 8.75 Auf Wiedersehen"),
    ("de", "SATURN ELECTRO", "199.00", "2024-02-28",
     "SATURN ELECTRO Datum: 28.02.2024 USB Kabel 19.00 Kopfhoerer 180.00 "
     "Nettobetrag 167.23 MwSt 31.77 Gesamtbetrag 199.00 Visa 199.00"),
    ("es", "MERCADONA SA", "37.62", "2024-04-15",
     "MERCADONA S.A. Calle Mayor 5 Fecha: 15/04/2024 Pan 1.20 Leche 0.95 "
     "Fruta 4.50 Pescado 12.30 Base imponible 33.10 IVA 21% 4.52 Importe total 37.62 Tarjeta 37.62"),
    ("es", "EL CORTE INGLES", "89.99", "2024-03-03",
     "EL CORTE INGLES Fecha 03/03/2024 Camisa 39.99 Pantalon 50.00 "
     "Subtotal 74.37 IVA 15.62 Total a pagar 89.99 Visa 89.99 Gracias por su compra"),
    ("es", "CARREFOUR EXPRESS", "12.40", "2024-05-21",
     "CARREFOUR EXPRESS Fecha: 21/05/2024 Agua 0.60 Cafe 3.80 Bocadillo 4.50 "
     "Snacks 3.50 Importe total 12.40 Efectivo 15.00 Cambio 2.60"),
    ("fr", "CARREFOUR CITY", "26.83", "2024-02-09",
     "CARREFOUR CITY 15 Rue de Paris Date: 09/02/2024 Baguette 1.10 Fromage 6.40 "
     "Vin 8.90 Yaourt 3.20 Sous-total 19.60 TVA 5.5% 1.08 Montant total 26.83 CB 26.83"),
    ("fr", "FNAC DARTY", "349.00", "2024-06-18",
     "FNAC Date 18/06/2024 Casque audio 149.00 Disque dur 200.00 "
     "Total HT 290.83 TVA 58.17 Total TTC 349.00 Carte bancaire 349.00 Merci"),
    ("fr", "MONOPRIX SA", "18.75", "2024-01-27",
     "MONOPRIX Date: 27/01/2024 Salade 2.30 Poulet 7.95 Dessert 3.50 "
     "Boisson 2.00 Net a payer 18.75 Especes 20.00 Rendu 1.25"),
    ("it", "ESSELUNGA SPA", "42.10", "2024-04-04",
     "ESSELUNGA S.P.A. Data: 04/04/2024 Pasta 1.50 Pomodori 3.20 Formaggio 8.90 "
     "Vino 6.50 Imponibile 37.60 IVA 4.50 Totale 42.10 Carta 42.10 Grazie"),
]


EN_TOTAL_KW = ["grand total", "total inclusive", "total incl", "amount due",
               "nett total", "net total", "total amount", "total"]


def _norm_date_en(s):
    """Pre-Day-7 English-only date parser: '/' and '-' formats, NO dot support."""
    if not s:
        return None
    s = str(s).strip().split()[0]
    for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return None


def extract_english_only(text: str) -> dict:
    """The pre-Day-7 extractor behaviour: English TOTAL/date anchors only, no dot
    dates. Pinned here so the before/after stays reproducible after the gazetteer
    was merged into the shipped `extract_fields`."""
    low = text.lower()
    amts = _all_amounts(text)
    amount = 0.0
    for kw in EN_TOTAL_KW:
        idx = low.rfind(kw)
        if idx != -1:
            after = [(pos, val) for pos, val in amts if 0 <= pos - (idx + len(kw)) <= 40]
            if after:
                amount = after[0][1]
                break
    if not amount and amts:
        amount = max(a for _, a in amts)
    dm = re.search(r"date[^0-9]{0,8}(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})", low)
    date = _norm_date_en(dm.group(1)) if dm else None
    if not date:
        d = re.search(r"\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}", text)
        date = _norm_date_en(d.group()) if d else None
    return {"amount": amount, "date": date, "merchant": _rules_smart(text).get("merchant")}


def extract_ml(text: str) -> dict:
    """Shipped rules + multilingual TOTAL / date labels appended (the gazetteer)."""
    low = text.lower()
    amts = _all_amounts(text)
    amount = 0.0
    # try multilingual + english total labels (multi-word ones first)
    kws = ML_TOTAL_KW + ["grand total", "total amount", "amount due", "total"]
    for kw in kws:
        idx = low.rfind(kw)
        if idx != -1:
            after = [(pos, val) for pos, val in amts if 0 <= pos - (idx + len(kw)) <= 40]
            if after:
                amount = after[0][1]
                break
    if not amount and amts:
        amount = max(a for _, a in amts)

    date = None
    for lab in ML_DATE_LABELS:
        dm = re.search(lab + r"[^0-9]{0,8}(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})", low)
        if dm:
            date = _norm_date_ml(dm.group(1))
            if date:
                break
    if not date:
        d = re.search(r"\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}", text)
        date = _norm_date_ml(d.group()) if d else None

    # merchant: reuse shipped heuristic result
    merchant = _rules_smart(text).get("merchant")
    return {"amount": amount, "date": date, "merchant": merchant}


def score(pred, gt_total, gt_date, gt_merch):
    try:
        pa = float(pred.get("amount") or 0.0)
    except (TypeError, ValueError):
        pa = 0.0
    ga = float(gt_total)
    amount_ok = abs(pa - ga) < 0.01
    pd = norm_date(pred.get("date")) if pred.get("date") else None
    date_ok = pd is not None and pd == gt_date
    merch_ok = merchant_match(pred.get("merchant"), gt_merch)
    return amount_ok, date_ok, merch_ok


def main():
    agg = {"baseline": {"amount": 0, "date": 0, "merchant": 0},
           "+gazetteer": {"amount": 0, "date": 0, "merchant": 0}}
    n = len(RECEIPTS)
    samples = []
    for lang, merch, gt_total, gt_date, text in RECEIPTS:
        base = extract_english_only(text)   # pinned pre-Day-7 behaviour
        ml = extract_ml(text)               # + multilingual gazetteer (now shipped)
        ba, bd, bm = score(base, gt_total, gt_date, merch)
        ma, md, mm = score(ml, gt_total, gt_date, merch)
        for f, v in zip(("amount", "date", "merchant"), (ba, bd, bm)):
            agg["baseline"][f] += int(v)
        for f, v in zip(("amount", "date", "merchant"), (ma, md, mm)):
            agg["+gazetteer"][f] += int(v)
        samples.append({"lang": lang, "merchant": merch, "gt_total": gt_total,
                        "base_amount": base.get("amount"), "base_amount_ok": ba,
                        "ml_amount": ml.get("amount"), "ml_amount_ok": ma,
                        "base_date_ok": bd, "ml_date_ok": md})

    print(f"[day7-ML] {n} non-English receipts (de/es/fr/it)")
    print(f"{'variant':<12}{'amount':>8}{'date':>8}{'merchant':>10}")
    rows = []
    for v in ("baseline", "+gazetteer"):
        a = agg[v]
        print(f"{v:<12}{a['amount']/n:>8.2f}{a['date']/n:>8.2f}{a['merchant']/n:>10.2f}")
        rows.append({"variant": v, "amount_acc": round(a["amount"]/n, 3),
                     "date_acc": round(a["date"]/n, 3), "merchant_acc": round(a["merchant"]/n, 3)})

    # regression guard: do multilingual tokens fire on English SROIE receipts?
    recs = [json.loads(l) for l in open(os.path.join(ROOT, "data", "eval", "receipts.jsonl"),
                                        encoding="utf-8")][:60]
    fired = sum(1 for r in recs if any(kw in r["text"].lower() for kw in ML_TOTAL_KW))
    print(f"[day7-ML] regression guard: multilingual TOTAL tokens fired on {fired}/60 English receipts")

    # detailed CSV
    import csv
    with open(os.path.join(RESULTS, "phase5_multilingual.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "amount_acc", "date_acc", "merchant_acc"])
        w.writeheader(); w.writerows(rows)
    json.dump(samples, open(os.path.join(SAMPLES, "phase5_multilingual_samples.json"), "w"), indent=2)

    # combined long CSV
    long_rows = []
    for r in rows:
        for m in ("amount_acc", "date_acc", "merchant_acc"):
            long_rows.append({"variant": r["variant"], "metric": m, "value": r[m],
                              "note": f"n={n} non-english"})
    long_rows.append({"variant": "gazetteer_english_regression", "metric": "tokens_fired_on_english",
                      "value": fired, "note": "0 = safe to ship"})
    upsert_track("multilingual", long_rows)
    print("[day7-ML] wrote phase5_multilingual.csv, samples, combined CSV")


if __name__ == "__main__":
    main()
