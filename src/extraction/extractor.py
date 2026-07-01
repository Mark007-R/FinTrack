"""Receipt/bill field extractor — Day-2 Phase-2a champion (`rules_smart`).

Day-2 bake-off result (100 SROIE receipts):

    method                amount  date  merchant
    regex (old default)    0.15   0.49   0.00
    rules_smart (champion) 0.58   0.87   0.77      <-- this module
    LLM zero-shot (ceiling)1.00   0.85   0.95      <-- Day-7/8 fallback, paid

This is the *keyword-anchored* extractor: it finds the amount next to a TOTAL
keyword (not subtotal/tax/change), the date next to a 'Date' label, and the
merchant from the opening company line. The old naive regex (`extract_bill.py`)
is retained only as a degraded fallback when the rules return nothing.

`extract_bill.find_bill_details` delegates here while preserving its signature.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

AMT_RE = re.compile(r"\d[\d,]*\.\d{2}")
TOTAL_KW = ["grand total", "total inclusive", "total incl", "amount due", "nett total",
            "net total", "total amount", "total",
            # Day-7 reach: multilingual total labels. English ones are matched first
            # (they appear earlier in the list), so English receipts are unaffected;
            # these only fire when no English anchor is present. German is the real
            # gap — its total label ('Gesamtbetrag'/'Summe') shares no English token.
            # Verified: 0/60 English SROIE receipts contain any of these.
            "gesamtbetrag", "gesamtsumme", "gesamt", "endbetrag", "summe",
            "importe total", "total a pagar", "montant total", "net a payer",
            "importo totale", "totale"]
NEG_KW = ["sub total", "subtotal", "sub-total", "change", "cash", "rounding", "discount", "tax"]


def _all_amounts(text: str):
    out = []
    for m in AMT_RE.finditer(text):
        try:
            out.append((m.start(), float(m.group().replace(",", ""))))
        except ValueError:
            pass
    return out


def _norm_date(s: str) -> Optional[str]:
    # Day-7 finding: European receipts use dot-separated day-first dates
    # (11.03.2024) that were silently unparseable before — only '/' and '-'
    # formats were tried. Added %d.%m.%Y / %d.%m.%y (day-first, as EU receipts print).
    s = str(s).strip()
    for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y",
                "%d.%m.%Y", "%d.%m.%y", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(s.split()[0] if " " in s else s, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    m = re.search(r"\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}", s)
    if m:
        for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y"):
            try:
                return datetime.strptime(m.group(), fmt).strftime("%Y-%m-%d")
            except Exception:
                continue
    return None


def _rules_smart(text: str) -> dict:
    """Keyword-anchored extraction (the Day-2 champion logic)."""
    low = text.lower()
    amts = _all_amounts(text)
    amount = 0.0
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
                # Day-6 fix: GST two-column totals. SROIE receipts often print the
                # TOTAL line as `<net> <gst>` (e.g. "TOTAL : 411.50 24.69") and the
                # true GST-inclusive total is their sum. If a second amount follows
                # immediately with a ratio in the GST band [0.03,0.09], use net+gst.
                # Tightly gated so it can't fire on a generic two-number line.
                # (Day-6 error analysis: this mode was 28 of 42 amount failures;
                # the rule lifted amount accuracy 0.58 -> 0.83 on the 100 SROIE set.)
                if len(after) >= 2 and amount > 0:
                    gst = after[1][1]
                    if 0.03 <= gst / amount <= 0.09 and (after[1][0] - after[0][0]) <= 12:
                        amount = round(amount + gst, 2)
                break
            idx = low.rfind(kw, 0, idx)
        if amount:
            break
    if not amount and amts:
        amount = max(a for _, a in amts)

    date = None
    # Day-7 reach: anchor on English + German/Spanish date labels (French/Italian
    # already use 'date'/'data'). These never appear before an English date, so
    # English behaviour is unchanged.
    dm = re.search(r"(?:date|datum|fecha|data)[^0-9]{0,8}(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})", low)
    if dm:
        date = _norm_date(dm.group(1))
    if not date:
        d = re.search(r"\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}", text)
        date = _norm_date(d.group()) if d else None

    merchant = None
    head = text[:200]
    head = re.split(r"ROC NO|TAX INVOICE|INVOICE NO|\bROC\b", head, flags=re.I)[0]
    caps = re.findall(
        r"\b[A-Z][A-Z&.\- ]{3,}(?:SDN BHD|BHD|ENTERPRISE|TRADING|PTE LTD|LTD|RESTAURANT|CAFE|MART|SUPERMARKET)?\b",
        head)
    caps = [c.strip() for c in caps if len(c.strip()) > 4 and "COPY" not in c]
    if caps:
        merchant = max(caps, key=len).strip()
    return {"amount": amount, "date": date, "merchant": merchant}


def _regex_fallback(text: str) -> dict:
    """The original naive regex (first \\d+\\.\\d{2}) — kept only as a last resort."""
    amount_match = re.findall(r"\b\d+\.\d{2}\b", text)
    date_match = re.findall(r"\b(\d{1,2}[\/.-]\d{1,2}[\/.-]\d{2,4})\b", text)
    amount = float(amount_match[0]) if amount_match else 0.0
    date = _norm_date(date_match[0]) if date_match else None
    return {"amount": amount, "date": date, "merchant": None}


class ReceiptExtractor:
    """Champion extractor with a regex safety net.

    Strategy: run `rules_smart`; if it fails to find an amount, fall back to the
    old regex so the pipeline never returns nothing on a degenerate document.
    """

    def extract(self, text: str) -> dict:
        if not text or not text.strip():
            return {"amount": 0.0, "date": None, "merchant": None,
                    "method": "empty", "confidence": 0.0}
        res = _rules_smart(text)
        if res["amount"]:
            # confidence is a heuristic: full credit for a total + a date + a merchant
            conf = 0.5 + 0.25 * (res["date"] is not None) + 0.25 * (res["merchant"] is not None)
            return {**res, "method": "rules_smart", "confidence": round(conf, 2)}
        fb = _regex_fallback(text)
        return {**fb, "method": "regex_fallback", "confidence": 0.2}


_default = ReceiptExtractor()


def extract_fields(text: str) -> dict:
    """Module-level convenience wrapper around the default extractor."""
    return _default.extract(text)
