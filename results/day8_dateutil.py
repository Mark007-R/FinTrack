"""Day-8 date normalizer — a strict superset of the Day-2 scorer's norm_date.

The Day-2 `norm_date` (run_day2_extraction) cannot parse 2-digit-year dates like
'18-03-18' or 'Feb'-month strings like '28-Feb-2018'. On the fresh SROIE offset>=100
slice a meaningful minority of receipts print exactly those formats, so the Day-2
scorer maps their GROUND TRUTH to None and no prediction can ever score — dragging
date accuracy down for the champion AND the LLM alike.

norm_date8 adds %d-%m-%y / %d/%m/%y / %d.%m.%y / %d-%b-%Y (etc.) so the GT parses.
It is applied SYMMETRICALLY to both prediction and GT in every Day-8 comparison,
so the head-to-head ranking is unchanged; only the absolute date numbers get honest.
"""
import re
from datetime import datetime

_FMTS = ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y-%m-%d",
         "%d/%m/%y", "%d-%m-%y", "%d.%m.%Y", "%d.%m.%y",
         "%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%d-%B-%Y", "%d %b %y")


def norm_date8(s):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    head = s.split()[0] if (" " in s and re.match(r"\d", s)) else s
    for fmt in _FMTS:
        for cand in (s, head):
            try:
                return datetime.strptime(cand, fmt).strftime("%Y-%m-%d")
            except Exception:
                continue
    m = re.search(r"\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}", s)
    if m:
        for fmt in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y-%m-%d",
                    "%d/%m/%y", "%d-%m-%y", "%d.%m.%Y", "%d.%m.%y"):
            try:
                return datetime.strptime(m.group(), fmt).strftime("%Y-%m-%d")
            except Exception:
                continue
    return None
