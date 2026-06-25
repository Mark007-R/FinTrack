"""Reproducible generator for the expense-categorization eval set.

MEDIA DISCIPLINE: 100% synthetic. No real personal financial data is used.
The descriptions imitate how raw bank/POS feeds actually look (store numbers,
POS prefixes, abbreviations, no spaces) so that a keyword baseline fails on
merchants whose category word is absent from the string -- e.g. "SHELL OIL"
(transport), "PG&E" (utilities), "CVS/PHARMACY" (health). Seeded for repro.

Run:  python data/eval/build_transactions.py   ->  data/eval/transactions.csv
"""
import csv
import os
import random

SEED = 20260625
random.seed(SEED)

# (merchant tokens, category). Deliberately mixes "obvious" strings (contain the
# category word) with "hard" strings (category word absent) per class.
MERCHANTS = {
    "groceries": ["WALMART SUPERCENTER", "TRADER JOE'S", "WHOLE FOODS MKT", "SAFEWAY",
                  "KROGER", "ALDI", "COSTCO WHSE", "PUBLIX", "TESCO", "GROCERY OUTLET",
                  "H-E-B", "SPROUTS FARMERS MKT"],
    "dining": ["MCDONALD'S", "STARBUCKS", "CHIPOTLE", "DOORDASH", "UBER EATS",
               "OLIVE GARDEN", "PIZZA HUT", "DUNKIN", "PANERA BREAD", "TST* CAFE ROUGE",
               "SQ *TAQUERIA", "GRUBHUB"],
    "transport": ["SHELL OIL", "UBER TRIP", "LYFT RIDE", "BP#", "CHEVRON", "EXXONMOBIL",
                  "MTA SUBWAY", "DELTA AIR", "AMTRAK", "PARKING METER", "76 GAS",
                  "ENTERPRISE RENT-A-CAR"],
    "utilities": ["PG&E", "COMCAST XFINITY", "AT&T", "VERIZON WIRELESS", "CON EDISON",
                  "NATIONAL GRID", "SPECTRUM", "DUKE ENERGY", "T-MOBILE", "WATER DIST",
                  "SCE&G", "CITY GAS CO"],
    "rent": ["RENT - APT 4B", "GREYSTAR PROPERTY", "AVALON COMMUNITIES", "ZILLOW RENTAL",
             "APARTMENT LEASE PMT", "LANDLORD ACH", "EQUITY RESIDENTIAL", "REALPAGE PMT"],
    "entertainment": ["NETFLIX", "SPOTIFY", "AMC THEATRES", "STEAM GAMES", "HULU",
                      "DISNEY PLUS", "PLAYSTATION NTWK", "REGAL CINEMAS", "HBO MAX",
                      "TICKETMASTER", "XBOX LIVE"],
    "health": ["CVS/PHARMACY", "WALGREENS", "QUEST DIAGNOSTICS", "KAISER PERMANENTE",
               "LABCORP", "ONE MEDICAL", "DR SMITH DDS", "RITE AID", "GNC LIVE WELL",
               "OPTUM RX"],
    "shopping": ["AMAZON.COM", "TARGET", "BEST BUY", "IKEA", "NIKE STORE", "MACY'S",
                 "HOME DEPOT", "ETSY", "EBAY", "ZARA", "APPLE STORE", "WAYFAIR"],
    "income": ["PAYROLL DEPOSIT", "DIRECT DEP - ACME CORP", "STRIPE PAYOUT", "VENMO CASHOUT",
               "ZELLE FROM J DOE", "IRS TAX REFUND", "INTEREST PAYMENT", "DIVIDEND VANGUARD"],
    "other": ["ATM WITHDRAWAL", "BANK FEE", "VENMO PAYMENT", "PAYPAL TRANSFER",
              "CHECK #1042", "WIRE TRANSFER", "USPS POSTAGE", "NOTARY SERVICE",
              "GOFUNDME DONATION", "MISC DEBIT"],
}

# class weights -> deliberate imbalance: rare classes income/rent/health under-sampled
WEIGHTS = {"groceries": 18, "dining": 16, "shopping": 14, "transport": 12,
           "utilities": 10, "entertainment": 9, "other": 8, "health": 6,
           "rent": 4, "income": 3}

POS_PREFIX = ["", "", "", "POS DEBIT ", "ACH ", "DEBIT CARD ", "PURCHASE ", "SQ *", "TST* "]
SUFFIX_CITY = ["", "", " #{n}", " #{n}", " SAN JOSE CA", " NEW YORK NY", " AUSTIN TX",
               " SEATTLE WA", " {n} ID", " STORE {n}"]


def make_desc(base):
    s = base
    if random.random() < 0.45:
        s = random.choice(POS_PREFIX) + s
    if random.random() < 0.5:
        suf = random.choice(SUFFIX_CITY).format(n=random.randint(100, 9999))
        s = s + suf
    if random.random() < 0.25:
        s = s.lower()
    elif random.random() < 0.2:
        s = s.title()
    return s.strip()


def amount_for(cat):
    rng = {"groceries": (8, 180), "dining": (5, 90), "transport": (3, 120),
           "utilities": (20, 300), "rent": (700, 3200), "entertainment": (5, 80),
           "health": (10, 450), "shopping": (8, 600), "income": (200, 5000),
           "other": (1, 500)}[cat]
    val = round(random.uniform(*rng), 2)
    return val if cat == "income" else -val


def main():
    cats = list(WEIGHTS)
    wts = [WEIGHTS[c] for c in cats]
    n = 600
    rows = []
    start_day = 0
    for i in range(n):
        cat = random.choices(cats, weights=wts, k=1)[0]
        base = random.choice(MERCHANTS[cat])
        desc = make_desc(base)
        amt = amount_for(cat)
        start_day += random.randint(0, 2)
        day = 1 + (start_day % 28)
        month = 1 + (start_day // 28) % 12
        rows.append({"id": i, "description": desc, "amount": amt,
                     "date": f"2025-{month:02d}-{day:02d}", "category": cat})
    out = os.path.join(os.path.dirname(__file__), "transactions.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "description", "amount", "date", "category"])
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print("wrote", out, "rows=", len(rows))
    print("class counts:", dict(Counter(r["category"] for r in rows)))


if __name__ == "__main__":
    main()
