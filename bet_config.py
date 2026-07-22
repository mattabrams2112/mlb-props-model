"""
Central staking / qualifying config — the ONE place bet tiers are defined.

Tiers:
  85+   → 1.0u ($8)  — tracked since the start
  80-84 → 0.5u ($4)  — added on EXPANSION_DATE (below)

Past days are never rewritten: an 80-84 play only counts as a tracked bet on
or after EXPANSION_DATE. Every play dated before that keeps the 85-only history,
so records for prior days are identical to what they were.

To change the go-live date or the tiers, edit the constants here — every page
(Tracker, Daily Results, Game View, dashboard) reads from this module.
"""

UNIT_DOLLARS   = 8.0          # $ per 1.0 unit
EXPANSION_DATE = '2026-07-21' # day 80-84 (0.5u) bets went live
TIER1_MIN      = 85           # 1.0u
TIER2_MIN      = 80           # 0.5u — only counts on/after EXPANSION_DATE

# 90+ ratings are boom-or-bust (they over-project and bust to 0 ~60% of the time),
# so from CAP_DATE forward they are NO LONGER tracked bets. Days before CAP_DATE
# keep their 90+ plays exactly as recorded — past records never change.
CAP_DATE       = '2026-07-22' # day 90+ was dropped from tracked bets
TIER_MAX       = 90           # ratings >= this are excluded on/after CAP_DATE


def units_for(rating) -> float:
    """Stake in units for a rating (0 if it isn't a bet at all)."""
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return 0.0
    if r >= TIER1_MIN:
        return 1.0
    if r >= TIER2_MIN:
        return 0.5
    return 0.0


def units_label(rating) -> str:
    """e.g. '1u' or '0.5u' — trailing zeros trimmed."""
    u = units_for(rating)
    return f'{u:g}u' if u else '—'


def bet_label(rating) -> str:
    """e.g. '$8 (1u)' or '$4 (0.5u)'."""
    u = units_for(rating)
    return f'${u * UNIT_DOLLARS:.0f} ({u:g}u)' if u else '—'


def qualifies(rating, date_str) -> bool:
    """
    Is this play a tracked bet for its date?
      85-89 → always
      80-84 → only on/after EXPANSION_DATE
      90+   → only BEFORE CAP_DATE (dropped as a bet from CAP_DATE forward)
    """
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return False
    d = str(date_str)[:10]
    # 90+ dropped from CAP_DATE onward; earlier days keep them unchanged
    if r >= TIER_MAX and d >= CAP_DATE:
        return False
    if r >= TIER1_MIN:
        return True
    return r >= TIER2_MIN and d >= EXPANSION_DATE


def qualifies_mask(df, rating_col: str = 'rating', date_col: str = 'date_str'):
    """
    Vectorized qualifier for a DataFrame — returns a boolean Series.
    85-89 any date, 80-84 on/after EXPANSION_DATE, 90+ only before CAP_DATE.
    """
    import pandas as pd
    r = pd.to_numeric(df[rating_col], errors='coerce')
    if date_col in df.columns:
        d = df[date_col].astype(str).str[:10]
    else:
        d = df['date'].astype(str).str[:10]
    base     = (r >= TIER1_MIN) | ((r >= TIER2_MIN) & (d >= EXPANSION_DATE))
    excluded = (r >= TIER_MAX) & (d >= CAP_DATE)
    return base & ~excluded
