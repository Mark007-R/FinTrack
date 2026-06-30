"""Day-6 adversarial stress set for the expense categorizer.

The Day-1 synthetic transactions are in-distribution and the champion saturates
them (OOF macro-F1 ~0.985, only ~6 errors in 600 rows) -- too easy to mine 30
genuine failures from. This set is deliberately HARD and realistic: real-world
merchant strings that (a) contain NO literal category keyword, or (b) contain a
keyword for the WRONG category (cross-category overlap). It is the eval where the
categorizer's true failure modes surface.

All rows are synthetic merchant strings (public brand names, no personal data) ->
respects the FinTrack media-discipline rule. Deterministic, no RNG.

Categories: groceries dining transport utilities rent entertainment health
shopping income other.

Writes data/eval/transactions_stress.csv  (description, category, difficulty).
"""
from __future__ import annotations
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "eval", "transactions_stress.csv")

# (description, gold_category, difficulty)
#   no_kw   = no literal keyword for the gold class
#   overlap = a keyword for a DIFFERENT class is present (cross-category trap)
ROWS = [
    # --- groceries: brand names with no "grocer/market/food" token, + overlaps
    ("TRADER JOES #455 SAN JOSE CA", "groceries", "no_kw"),
    ("WHOLE FOODS MKT 10212", "groceries", "no_kw"),
    ("AMAZON FRESH SEATTLE WA", "groceries", "overlap"),      # amazon -> shopping
    ("INSTACART*WEGMANS", "groceries", "no_kw"),
    ("TST* SPROUTS FARMERS", "groceries", "no_kw"),
    ("H-E-B #213 AUSTIN TX", "groceries", "no_kw"),
    ("PUBLIX SUPER MKT", "groceries", "no_kw"),
    ("TARGET T-1842 GROCERY", "groceries", "overlap"),        # target -> shopping
    ("COSTCO WHSE #1043", "groceries", "no_kw"),
    ("ALDI 65 BROOKLYN NY", "groceries", "no_kw"),
    # --- dining: restaurants & delivery, several overlapping with transport/shopping
    ("UBER EATS SAN FRANCISCO", "dining", "overlap"),         # uber -> transport
    ("CHIPOTLE 2487 DENVER CO", "dining", "no_kw"),
    ("TST* SHAKE SHACK", "dining", "no_kw"),
    ("PANERA BREAD #601", "dining", "no_kw"),
    ("CHICK-FIL-A #02199", "dining", "no_kw"),
    ("WENDYS 5512", "dining", "no_kw"),
    ("DOORDASH*WINGSTOP", "dining", "no_kw"),
    ("SQ *BLUE BOTTLE", "dining", "no_kw"),
    ("OLIVE GARDEN 0021", "dining", "no_kw"),
    ("SONIC DRIVE IN #443", "dining", "no_kw"),
    # --- transport: ride/fuel/transit, overlaps with groceries/dining
    ("COSTCO GAS #1043", "transport", "overlap"),             # costco -> groceries
    ("SHELL OIL 5742", "transport", "no_kw"),
    ("EXXONMOBIL 9921", "transport", "no_kw"),
    ("CHEVRON 0091234", "transport", "no_kw"),
    ("BP#8843 FUEL", "transport", "no_kw"),
    ("LYFT *RIDE WED 2PM", "transport", "no_kw"),
    ("MTA*NYCT PAYGO", "transport", "no_kw"),
    ("DELTA AIR 0062317", "transport", "no_kw"),
    ("SUNPASS TOLL", "transport", "no_kw"),
    ("76 GAS STATION", "transport", "no_kw"),
    # --- utilities: providers, no "utility" token, some brand overlaps
    ("PG&E WEB ONLINE", "utilities", "no_kw"),
    ("CON EDISON OF NY", "utilities", "no_kw"),
    ("XFINITY MOBILE", "utilities", "no_kw"),
    ("SPECTRUM 8002221", "utilities", "no_kw"),
    ("DUKE ENERGY", "utilities", "no_kw"),
    ("SOCALGAS PAYMENT", "utilities", "no_kw"),
    ("T-MOBILE POSTPAID", "utilities", "no_kw"),
    ("NATIONAL GRID", "utilities", "no_kw"),
    # --- rent: housing, some no_kw
    ("GREYSTAR RESIDENTIAL", "rent", "no_kw"),
    ("AVALON COMMUNITIES", "rent", "no_kw"),
    ("ZILLOW RENTAL PYMT", "rent", "no_kw"),
    ("EQUITY RESIDENTIAL", "rent", "no_kw"),
    ("CAMDEN PROPERTY TR", "rent", "no_kw"),
    ("WEWORK MEMBERSHIP", "rent", "no_kw"),
    # --- entertainment: streaming/events, overlaps w/ shopping (apple)
    ("APPLE MUSIC SUBSCRIPTION", "entertainment", "overlap"), # apple -> shopping
    ("AMC ONLINE 0123", "entertainment", "no_kw"),
    ("HBO MAX", "entertainment", "no_kw"),
    ("STEAMGAMES.COM", "entertainment", "no_kw"),
    ("TICKETMASTER EVENT", "entertainment", "no_kw"),
    ("PARAMOUNT PLUS", "entertainment", "no_kw"),
    ("PLAYSTATION NETWORK", "entertainment", "no_kw"),
    ("FANDANGO", "entertainment", "no_kw"),
    # --- health: pharmacies/clinics, overlap with groceries (walmart pharmacy)
    ("WALMART PHARMACY 10", "health", "overlap"),             # walmart -> groceries
    ("CVS/PHARMACY #4512", "health", "no_kw"),
    ("WALGREENS 6621", "health", "no_kw"),
    ("QUEST DIAGNOSTICS", "health", "no_kw"),
    ("LABCORP", "health", "no_kw"),
    ("ONE MEDICAL", "health", "no_kw"),
    ("EQUINOX FITNESS", "health", "no_kw"),
    ("KAISER PERMANENTE", "health", "no_kw"),
    # --- shopping: retail, overlap with groceries (target) & dining
    ("BEST BUY #00141", "shopping", "no_kw"),
    ("APPLE STORE R052", "shopping", "no_kw"),
    ("ZARA USA 0231", "shopping", "no_kw"),
    ("H&M 0512 NEW YORK", "shopping", "no_kw"),
    ("WAYFAIR ORDER", "shopping", "no_kw"),
    ("ETSY.COM", "shopping", "no_kw"),
    ("SEPHORA 0411", "shopping", "no_kw"),
    ("HOME DEPOT 6534", "shopping", "no_kw"),
    ("LOWES #01923", "shopping", "no_kw"),
    ("NORDSTROM 0042", "shopping", "no_kw"),
    # --- income: deposits, some no literal keyword
    ("ACME CORP DIR DEP", "income", "no_kw"),
    ("ZELLE FROM EMMA RENT", "income", "overlap"),            # rent -> rent
    ("VENMO CASHOUT", "income", "no_kw"),
    ("STRIPE TRANSFER", "income", "no_kw"),
    ("IRS TREAS 310 TAX REF", "income", "no_kw"),
    ("ROBINHOOD ACH", "income", "no_kw"),
    # --- other: misc that maps to none cleanly
    ("USPS PO 0512", "other", "no_kw"),
    ("NOTARY PUBLIC FEE", "other", "no_kw"),
    ("CHARITY: WATER DONATION", "other", "no_kw"),
    ("DMV VEHICLE REG", "other", "no_kw"),
    ("CITY PARKING TICKET", "other", "overlap"),              # parking -> transport
    ("ATM WITHDRAWAL 0231", "other", "no_kw"),
]


def main():
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "description", "category", "difficulty"])
        for i, (desc, cat, diff) in enumerate(ROWS):
            w.writerow([i, desc.lower(), cat, diff])
    from collections import Counter
    cats = Counter(r[1] for r in ROWS)
    diffs = Counter(r[2] for r in ROWS)
    print(f"wrote {OUT}: {len(ROWS)} rows")
    print(f"  by category: {dict(cats)}")
    print(f"  by difficulty: {dict(diffs)}")


if __name__ == "__main__":
    main()
