"""Receipt/bill extraction tests — the Day-2/5/6/7 champion (`rules_smart`).

Covers the champion path, the regex fallback, the documented edge cases
(subtotal/tax vs total, GST two-column totals, empty PDF, non-receipt text,
foreign-currency/EU dates), and the preserved `find_bill_details` signature.
"""
from __future__ import annotations

from src.extraction import extract_fields
from src.extraction.extractor import ReceiptExtractor
import extract_bill


def test_total_anchored_amount():
    text = "MINI MART\nItem A 3.50\nItem B 2.00\nTOTAL 5.50\n"
    res = extract_fields(text)
    assert res["amount"] == 5.50
    assert res["method"] == "rules_smart"


def test_prefers_total_over_subtotal_and_tax():
    # first decimal is the subtotal; a naive regex would grab 40.00
    text = "SHOP\nSubtotal 40.00\nTax 2.40\nTotal 42.40\n"
    res = extract_fields(text)
    assert res["amount"] == 42.40


def test_gst_two_column_total_is_summed():
    # SROIE-style: "TOTAL : <net> <gst>" where the inclusive total is net+gst
    text = "KEDAI\nTOTAL : 411.50 24.69\n"
    res = extract_fields(text)
    assert res["amount"] == 436.19  # 411.50 + 24.69 (gst ratio ~0.06, in-band)


def test_empty_text_returns_empty_method():
    # edge case: empty PDF / OCR produced nothing
    res = extract_fields("")
    assert res["amount"] == 0.0
    assert res["method"] == "empty"
    assert res["confidence"] == 0.0


def test_non_receipt_text_falls_back_gracefully():
    # a non-receipt document with no TOTAL keyword must not crash; either it
    # finds the max decimal via rules or drops to the regex fallback.
    text = "Dear customer, thank you for your enquiry dated 12/03/2024."
    res = extract_fields(text)
    assert set(res) >= {"amount", "date", "merchant", "method", "confidence"}
    assert res["method"] in {"rules_smart", "regex_fallback", "empty"}


def test_eu_dot_date_is_normalised():
    text = "LADEN\nDatum 11.03.2024\nGesamtbetrag 19.90\n"
    res = extract_fields(text)
    assert res["date"] == "2024-03-11"        # day-first EU dot format
    assert res["amount"] == 19.90             # German total label anchored


def test_confidence_in_unit_interval():
    text = "CAFE ROMA\nDate 01/02/2024\nTotal 12.00\n"
    res = extract_fields(text)
    assert 0.0 <= res["confidence"] <= 1.0
    assert res["confidence"] > 0.5            # total + date (+ maybe merchant)


def test_real_sroie_receipt_amount(eval_receipts):
    r = eval_receipts[0]
    res = extract_fields(r["text"])
    assert abs(res["amount"] - float(r["gt_total"])) < 0.01


def test_real_sroie_receipt_merchant(eval_receipts):
    r = eval_receipts[0]
    res = extract_fields(r["text"])
    # merchant should contain the core brand token from ground truth
    assert res["merchant"] is not None
    assert "OJC" in res["merchant"].upper()


def test_find_bill_details_signature_preserved():
    # Flask route contract: returns (signed_amount, date_string); amount negative
    amount, date = extract_bill.find_bill_details("STORE\nTotal 30.00\nDate 05/06/2024\n")
    assert amount == -30.00                    # signed outflow
    assert date == "2024-06-05"


def test_find_bill_details_never_raises_on_garbage():
    amount, date = extract_bill.find_bill_details("")
    assert amount == 0.0
    assert date == "Not found"


def test_extractor_class_matches_module_wrapper():
    text = "DELI\nTOTAL 7.25\n"
    assert ReceiptExtractor().extract(text)["amount"] == extract_fields(text)["amount"]
