#!/usr/bin/env python3
"""GTA VI launch-day Twitch viewers: hour-by-hour quantile forecast.

Single entrypoint, Python 3.13 standard library only, deterministic, no network.

Pipeline for each case
----------------------
1. Read the target's hour-1 print ``p`` and work in log space relative to it.
2. Comparables.  Each comparable launch's release-day curve is re-indexed so
   that its US-East midnight unlock sits on the same hour as the target's
   (04:00Z under US daylight time, 05:00Z otherwise), then divided by its own
   hour 1.  That keeps the staggered local-midnight shape (UK/Europe already in
   hour 1, the US-East jump, US-West, the evening peak) aligned.
   Launches that are not console midnight-local unlocks are dropped
   (single-global, PC-only, early-access, free-to-play, ...), as is any curve
   whose hour 1 was evidently pre-unlock (day max far above hour 1).
3. Size effect.  Smaller launches grow more over the day.  For every hour a
   weighted least-squares slope of log ratio on log(hour-1 viewers) is fitted,
   clamped to [-SLOPE_CAP, 0], and used to move each comparable's ratio to the
   target's hour-1 size (target size clamped to the comparables' range).
4. Density.  Per hour, the shifted log ratios become a weighted Gaussian kernel
   mixture with Silverman bandwidth (floor 0.05, cap KERNEL_SD).  The house
   starter forecast (lognormal at p, log sd 0.15*sqrt(h-1)) is added as one
   more component of weight PRIOR_STRENGTH, so it dominates when comparables
   are thin.  The five quantiles are read off the mixture CDF by bisection.
"""
import json
import math
import os
import re
from datetime import date, datetime, timedelta, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
HOUSE_Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
HOURS = range(2, 25)

# Fixed model constants.
KERNEL_SD = 0.20        # cap on the per-hour kernel bandwidth
MIN_BW = 0.05           # floor on the per-hour kernel bandwidth
PRIOR_STRENGTH = 3.0    # pseudo-count given to the house starter component
SLOPE_CAP = 0.5         # bound on |d log ratio / d log hour-1|
MIN_SLOPE_COMPS = 4     # comparables needed before fitting a size slope
MAX_DAY_RATIO = 15.0    # day max above 15x hour 1 => hour 1 was pre-unlock
MIN_BASE_VIEWERS = 50   # ignore comparables with a tiny hour 1
MIN_HOURS = 12          # comparables need at least this many usable hours
HOUSE_SD_PER_HOUR = 0.15

DROP_MARKERS = (
    "single global", "single simultaneous", "simultaneous across",
    "simultaneous global", "pc only", "pc single", "early-access-contaminated",
    "free-to-play", "f2p", "tier: weak", "not a launch",
    "target-history",
)


# --------------------------------------------------------------------------
# time helpers
# --------------------------------------------------------------------------
def to_utc(stamp):
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(timezone.utc)


def _nth_sunday(year, month, n):
    first = date(year, month, 1)
    first_sunday = first + timedelta(days=(6 - first.weekday()) % 7)
    return first_sunday + timedelta(weeks=n - 1)


def east_midnight_hour(day):
    """UTC hour (on ``day``) at which US-East local midnight falls."""
    dst = _nth_sunday(day.year, 3, 2) <= day < _nth_sunday(day.year, 11, 1)
    return 4 if dst else 5


# --------------------------------------------------------------------------
# comparables
# --------------------------------------------------------------------------
def trust(comp):
    """Weight given to one comparable launch (0 = drop it)."""
    desc = "%s %s" % (comp.get("unlock", ""), comp.get("name", ""))
    desc = desc.lower()
    for marker in DROP_MARKERS:
        if marker in desc:
            return 0.0
    has_pc = re.search(r"\bpc\b", str(comp.get("platforms", "")).lower()) is not None
    weight = 0.6 if has_pc else 1.0
    if "midnight" not in desc:
        weight *= 0.7  # unlock time unverified
    return weight


def hourly_series(comp):
    """{hours after day_start: viewers} for the positive, finite rows."""
    start = to_utc(comp["day_start"])
    series = {}
    for row in comp.get("hourly") or []:
        raw = row.get("viewers")
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val) and val > 0:
            k = round((to_utc(row["from"]) - start).total_seconds() / 3600.0)
            series[int(k)] = val
    return start, series


class Comparable:
    __slots__ = ("weight", "log_base", "ratios")

    def __init__(self, weight, log_base, ratios):
        self.weight = weight
        self.log_base = log_base   # log(hour-1 viewers)
        self.ratios = ratios       # {target hour: log(viewers / hour 1)}


def build_comparable(comp, target_unlock):
    weight = trust(comp)
    if weight <= 0:
        return None
    start, series = hourly_series(comp)
    lag = target_unlock - east_midnight_hour(start.date())
    base = series.get(-lag)
    if base is None or base < MIN_BASE_VIEWERS:
        return None
    ratios = {}
    for h in HOURS:
        val = series.get(h - 1 - lag)
        if val is not None:
            ratios[h] = math.log(val / base)
    if len(ratios) < MIN_HOURS or max(ratios.values()) > math.log(MAX_DAY_RATIO):
        return None
    return Comparable(weight, math.log(base), ratios)


# --------------------------------------------------------------------------
# size adjustment
# --------------------------------------------------------------------------
def size_slope(pairs):
    """Clamped weighted LS slope from (weight, x, y) triples."""
    total = sum(w for w, _, _ in pairs)
    if len(pairs) < MIN_SLOPE_COMPS or total <= 0:
        return 0.0
    mx = sum(w * x for w, x, _ in pairs) / total
    my = sum(w * y for w, _, y in pairs) / total
    sxx = sum(w * (x - mx) ** 2 for w, x, _ in pairs)
    if sxx <= 0:
        return 0.0
    slope = sum(w * (x - mx) * (y - my) for w, x, y in pairs) / sxx
    return min(0.0, max(-SLOPE_CAP, slope))


def adjust_for_size(comps, log_p):
    if len(comps) < MIN_SLOPE_COMPS:
        return comps
    lo = min(c.log_base for c in comps)
    hi = max(c.log_base for c in comps)
    target = min(max(log_p, lo), hi)
    slopes = {}
    for h in HOURS:
        slopes[h] = size_slope([(c.weight, c.log_base, c.ratios[h])
                                for c in comps if h in c.ratios])
    shifted = []
    for c in comps:
        moved = {h: r + slopes[h] * (target - c.log_base) for h, r in c.ratios.items()}
        shifted.append(Comparable(c.weight, c.log_base, moved))
    return shifted


# --------------------------------------------------------------------------
# per-hour density
# --------------------------------------------------------------------------
def weighted_quantile(sorted_pts, total, tau):
    acc = 0.0
    for w, x in sorted_pts:
        acc += w
        if acc >= tau * total:
            return x
    return sorted_pts[-1][1]


def silverman(pts):
    """Silverman bandwidth for weighted points [(w, x)], clamped."""
    total = sum(w for w, _ in pts)
    n_eff = total ** 2 / sum(w * w for w, _ in pts)
    mean = sum(w * x for w, x in pts) / total
    sd = math.sqrt(sum(w * (x - mean) ** 2 for w, x in pts) / total)
    ordered = sorted(pts, key=lambda t: t[1])
    iqr = (weighted_quantile(ordered, total, 0.75)
           - weighted_quantile(ordered, total, 0.25)) / 1.349
    spread = min(sd, iqr) if iqr > 0 else sd
    return min(max(0.9 * spread * n_eff ** -0.2, MIN_BW), KERNEL_SD)


def phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def mixture_quantile(kernels, tau):
    """Quantile of a Gaussian mixture [(weight, mean, sd)] via bisection."""
    total = sum(k[0] for k in kernels)
    lo = min(m - 8 * s for _, m, s in kernels)
    hi = max(m + 8 * s for _, m, s in kernels)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        cdf = sum(w * phi((mid - m) / s) for w, m, s in kernels) / total
        if cdf < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def hour_quantiles(log_p, kernels):
    out = []
    floor = 0.0
    for tau in LEVELS:
        q = math.exp(log_p + mixture_quantile(kernels, tau))
        q = round(max(min(max(q, 0.01), 1e9), floor), 2)
        q = max(q, floor)
        out.append(q)
        floor = q
    return out


# --------------------------------------------------------------------------
# case driver
# --------------------------------------------------------------------------
def hour1_viewers(case):
    p = None
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            p = row.get("viewers")
    try:
        p = float(p)
    except (TypeError, ValueError):
        return 1.0
    return p if math.isfinite(p) and p >= 1 else 1.0


def forecast_case(case):
    p = hour1_viewers(case)
    log_p = math.log(p)
    unlock = east_midnight_hour(to_utc(case["day_start"]).date())

    comps = []
    for raw in case.get("comparables") or []:
        try:
            c = build_comparable(raw, unlock)
        except (KeyError, TypeError, ValueError):
            continue
        if c is not None:
            comps.append(c)
    comps = adjust_for_size(comps, log_p)

    hours = []
    for h in HOURS:
        kernels = [(PRIOR_STRENGTH, 0.0, HOUSE_SD_PER_HOUR * math.sqrt(h - 1))]
        pts = [(c.weight, c.ratios[h]) for c in comps if h in c.ratios]
        if sum(w for w, _ in pts) > 0:
            bw = silverman(pts)
            kernels.extend((w, x, bw) for w, x in pts)
        hours.append({"hour": h, "q": hour_quantiles(log_p, kernels)})
    return {"hours": hours}


def house_forecast(case):
    """Starter fallback: lognormal around hour 1."""
    p = 1.0
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            try:
                p = max(1.0, float(row.get("viewers")))
            except (TypeError, ValueError):
                p = 1.0
    hours = []
    for h in HOURS:
        sd = HOUSE_SD_PER_HOUR * math.sqrt(h - 1)
        hours.append({"hour": h, "q": [round(p * math.exp(z * sd), 2) for z in HOUSE_Z]})
    return {"hours": hours}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), encoding="utf-8") as fh:
            case = json.load(fh)
        case_id = case.get("id") or name[:-3]
        try:
            result = forecast_case(case)
        except Exception:
            result = house_forecast(case)
        with open(os.path.join(out_dir, case_id + ".out"), "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))


if __name__ == "__main__":
    main()
