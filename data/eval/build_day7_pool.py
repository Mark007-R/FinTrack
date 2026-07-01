"""Day-7 active-learning pool: a larger HARD, realistic transaction pool.

The Day-6 stress set (82 rows) proved the categorizer's real failure modes are
(a) merchant strings with NO literal category keyword and (b) cross-category
overlaps. To run a credible active-learning loop we need a bigger *unlabeled*
pool with the same difficulty. This expands a curated brand -> category gazetteer
(public brand names only -> respects the FinTrack media-discipline rule) with
realistic bank-statement suffixes (store ids, cities, POS prefixes) into ~300
distinct rows, deterministically (no RNG).

Each row is tagged `difficulty`:
    no_kw    -> gold class has no literal keyword in the string
    overlap  -> a keyword for a DIFFERENT class is present (cross-category trap)
    easy     -> the literal keyword is present (in-distribution style)

Writes data/eval/transactions_pool.csv  (id, description, category, difficulty).
"""
from __future__ import annotations
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "data", "eval", "transactions_pool.csv")

# brand -> (category, difficulty).  no_kw unless the literal class keyword is present.
BRANDS = {
    "groceries": [
        ("trader joes", "no_kw"), ("whole foods mkt", "no_kw"), ("sprouts farmers", "no_kw"),
        ("h-e-b", "no_kw"), ("publix super mkt", "no_kw"), ("aldi", "no_kw"),
        ("wegmans", "no_kw"), ("safeway", "no_kw"), ("kroger", "no_kw"), ("stop & shop", "no_kw"),
        ("amazon fresh", "overlap"), ("target grocery", "overlap"), ("costco whse", "no_kw"),
        ("instacart", "no_kw"), ("giant eagle", "no_kw"),
    ],
    "dining": [
        ("chipotle", "no_kw"), ("shake shack", "no_kw"), ("panera bread", "no_kw"),
        ("chick-fil-a", "no_kw"), ("wendys", "no_kw"), ("olive garden", "no_kw"),
        ("sonic drive in", "no_kw"), ("blue bottle", "no_kw"), ("five guys", "no_kw"),
        ("uber eats", "overlap"), ("doordash wingstop", "no_kw"), ("grubhub", "no_kw"),
        ("in-n-out", "no_kw"), ("dennys", "no_kw"), ("popeyes", "no_kw"),
    ],
    "transport": [
        ("shell oil", "no_kw"), ("exxonmobil", "no_kw"), ("chevron", "no_kw"),
        ("bp fuel", "no_kw"), ("lyft ride", "no_kw"), ("mta nyct paygo", "no_kw"),
        ("delta air", "no_kw"), ("sunpass toll", "no_kw"), ("76 gas station", "no_kw"),
        ("costco gas", "overlap"), ("united airlines", "no_kw"), ("amtrak", "no_kw"),
        ("hertz rental", "no_kw"), ("spothero parking", "no_kw"), ("valero", "no_kw"),
    ],
    "utilities": [
        ("pg&e web online", "no_kw"), ("con edison", "no_kw"), ("xfinity mobile", "no_kw"),
        ("spectrum", "no_kw"), ("duke energy", "no_kw"), ("socalgas payment", "no_kw"),
        ("t-mobile postpaid", "no_kw"), ("national grid", "no_kw"), ("at&t uverse", "no_kw"),
        ("verizon fios", "no_kw"), ("dominion energy", "no_kw"), ("centurylink", "no_kw"),
    ],
    "rent": [
        ("greystar residential", "no_kw"), ("avalon communities", "no_kw"),
        ("zillow rental pymt", "no_kw"), ("equity residential", "no_kw"),
        ("camden property tr", "no_kw"), ("wework membership", "no_kw"),
        ("essex apartment", "easy"), ("maa residential", "no_kw"), ("udr communities", "no_kw"),
    ],
    "entertainment": [
        ("apple music", "overlap"), ("amc online", "no_kw"), ("hbo max", "no_kw"),
        ("steamgames.com", "no_kw"), ("ticketmaster", "no_kw"), ("paramount plus", "no_kw"),
        ("playstation network", "no_kw"), ("fandango", "no_kw"), ("regal cinema", "easy"),
        ("hulu", "no_kw"), ("disney plus", "no_kw"), ("nintendo eshop", "no_kw"),
    ],
    "health": [
        ("walmart pharmacy", "overlap"), ("cvs/pharmacy", "no_kw"), ("walgreens", "no_kw"),
        ("quest diagnostics", "no_kw"), ("labcorp", "no_kw"), ("one medical", "no_kw"),
        ("equinox fitness", "no_kw"), ("kaiser permanente", "no_kw"), ("rite aid", "no_kw"),
        ("planet fitness", "no_kw"), ("teladoc", "no_kw"), ("goodrx", "no_kw"),
    ],
    "shopping": [
        ("best buy", "no_kw"), ("apple store", "no_kw"), ("zara usa", "no_kw"),
        ("h&m", "no_kw"), ("wayfair order", "no_kw"), ("etsy.com", "no_kw"),
        ("sephora", "no_kw"), ("home depot", "no_kw"), ("lowes", "no_kw"),
        ("nordstrom", "no_kw"), ("macys", "no_kw"), ("ikea", "no_kw"), ("ebay", "no_kw"),
    ],
    "income": [
        ("acme corp dir dep", "no_kw"), ("zelle from emma rent", "overlap"),
        ("venmo cashout", "no_kw"), ("stripe transfer", "no_kw"),
        ("irs treas 310 tax ref", "no_kw"), ("robinhood ach", "no_kw"),
        ("paypal transfer", "no_kw"), ("adp payroll", "easy"), ("gusto payroll", "easy"),
    ],
    "other": [
        ("usps po", "no_kw"), ("notary public fee", "no_kw"), ("charity water donation", "no_kw"),
        ("dmv vehicle reg", "no_kw"), ("city parking ticket", "overlap"),
        ("atm withdrawal", "no_kw"), ("western union", "no_kw"), ("coinbase", "no_kw"),
        ("gofundme", "no_kw"),
    ],
}

# realistic bank-statement decorations; combined deterministically per brand
SUFFIXES = ["", " #{n}", " {city} {st}", " {n}", " pos {n}", " store {n}", " online"]
CITIES = [("san jose", "ca"), ("austin", "tx"), ("seattle", "wa"), ("denver", "co"),
          ("brooklyn", "ny"), ("miami", "fl"), ("chicago", "il"), ("boston", "ma")]
POS_PREFIX = ["", "pos debit ", "tst* ", "sq *", "ach ", "ckcd "]


def main():
    rows = []
    rid = 0
    for cat, brands in BRANDS.items():
        for bi, (brand, diff) in enumerate(brands):
            # generate 2 decorated variants per brand (deterministic by index)
            for k in range(2):
                pref = POS_PREFIX[(bi + k) % len(POS_PREFIX)]
                suf = SUFFIXES[(bi + k) % len(SUFFIXES)]
                city, st = CITIES[(bi + k) % len(CITIES)]
                n = 1000 + (bi * 7 + k * 131) % 9000
                suf = suf.format(n=n, city=city, st=st)
                desc = f"{pref}{brand}{suf}".strip().lower()
                rows.append((rid, desc, cat, diff))
                rid += 1

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "description", "category", "difficulty"])
        w.writerows(rows)

    from collections import Counter
    print(f"wrote {OUT}: {len(rows)} rows")
    print(f"  by category:   {dict(Counter(r[2] for r in rows))}")
    print(f"  by difficulty: {dict(Counter(r[3] for r in rows))}")


if __name__ == "__main__":
    main()
