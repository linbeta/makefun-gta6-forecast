#!/usr/bin/env python3
"""GTA VI launch-day Twitch viewers: hour-by-hour quantile forecast.

Model (standard library only, deterministic, no network):
  * For every comparable launch, take its release-day hourly curve and divide
    by its own "hour 1", after aligning the curve so that its US-East midnight
    unlock lands on the same hour as the target's (04:00Z under US daylight
    time, 05:00Z otherwise).  This keeps the staggered local-midnight shape
    (UK/Europe already in hour 1, US East jump, US West, evening peak) in place.
  * Keep the launches that look like console midnight-local unlocks; drop
    single-global / PC-only / early-access / free-to-play launches and any
    curve whose hour 1 was clearly pre-unlock (day max far above hour 1).
  * Size effect: smaller launches grow more over the day.  One weighted slope
    of each comparable's mean log ratio on log(its hour 1) is fitted at
    runtime and shifts every comparable's ratios to the target's hour-1 size
    (capped to the comparables' size range; slope bounded to [-0.5, 0]).
  * Per hour, the log ratios form a weighted kernel mixture.  It is blended
    with the house baseline (lognormal around the hour-1 print with log sd
    0.15*sqrt(h-1)) with weight that grows as comparables run out, and the five
    quantiles are read off the mixture CDF.
"""
import json
import math
import os
import re
from datetime import date, datetime, timedelta, timezone

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Model constants (fixed by reasoning, not tuned per practice hour).
KERNEL_SD = 0.20          # log-scale smoothing around each comparable's ratio
PRIOR_STRENGTH = 3.0      # pseudo-count of the house baseline in the blend
SLOPE_CAP = 0.5           # |size slope| bound (log ratio per log hour-1 viewers)
MAX_DAY_RATIO = 15.0      # hour 1 below 1/15 of the day max => pre-unlock hour 1
EXCLUDE_WORDS = (
    "single global", "single simultaneous", "simultaneous across",
    "simultaneous global", "pc only", "pc single", "early-access-contaminated",
    "free-to-play", "f2p", "tier: weak", "not a launch",
    "target-history",
)


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def nth_sunday(year, month, n):
    d = date(year, month, 1)
    d += timedelta(days=(6 - d.weekday()) % 7)
    return d + timedelta(days=7 * (n - 1))


def us_east_unlock_offset(day):
    """Hours after 00:00Z of the release date at which US-East midnight falls."""
    start = nth_sunday(day.year, 3, 2)
    end = nth_sunday(day.year, 11, 1)
    return 4 if start <= day < end else 5


def comp_weight(comp):
    text = (str(comp.get("unlock", "")) + " " + str(comp.get("name", ""))).lower()
    if any(w in text for w in EXCLUDE_WORDS):
        return 0.0
    plats = str(comp.get("platforms", "")).lower()
    console_only = re.search(r"\bpc\b", plats) is None
    w = 1.0 if console_only else 0.6
    if "midnight" not in text:
        w *= 0.7  # unverified unlock: keep, but trust less
    return w


def comp_log_ratios(comp, target_offset):
    """Return {target_hour: log(ratio)} for hours 2..24, or None if unusable."""
    w = comp_weight(comp)
    if w <= 0:
        return None, 0.0
    day0 = parse_ts(comp["day_start"])
    shift = target_offset - us_east_unlock_offset(day0.date())
    series = {}
    for row in comp.get("hourly") or []:
        v = row.get("viewers")
        if v is None:
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(v) or v <= 0:
            continue
        k = int(round((parse_ts(row["from"]) - day0).total_seconds() / 3600.0))
        series[k] = v
    # target hour h (1-based, offset h-1) <- comparable offset (h-1) - shift
    base = series.get(0 - shift)
    if base is None or base < 50:
        return None, 0.0
    out = {}
    for h in range(2, 25):
        v = series.get(h - 1 - shift)
        if v is not None:
            out[h] = math.log(v / base)
    if len(out) < 12:
        return None, 0.0
    if max(out.values()) > math.log(MAX_DAY_RATIO):
        return None, 0.0
    out[0] = math.log(base)  # key 0 carries log(hour-1 viewers)
    return out, w


def norm_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def mixture_quantile(components, tau):
    """components: list of (weight, mean, sd) in log space."""
    total = sum(c[0] for c in components)
    lo = min(c[1] - 8 * c[2] for c in components)
    hi = max(c[1] + 8 * c[2] for c in components)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        cdf = sum(w * norm_cdf((mid - m) / s) for w, m, s in components) / total
        if cdf < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forecast_case(case):
    known = case.get("known") or []
    p = None
    for row in known:
        if row.get("hour") == 1:
            p = row.get("viewers")
    try:
        p = float(p)
    except (TypeError, ValueError):
        p = 1.0
    if not math.isfinite(p) or p < 1:
        p = 1.0
    logp = math.log(p)

    target_offset = us_east_unlock_offset(parse_ts(case["day_start"]).date())
    comps = []
    for comp in case.get("comparables") or []:
        try:
            lr, w = comp_log_ratios(comp, target_offset)
        except (KeyError, TypeError, ValueError):
            continue
        if lr:
            comps.append((lr, w))

    # Size effect: smaller hour-1 prints grow more over the day.  One weighted
    # least-squares slope of each comparable's mean log ratio on log(hour 1),
    # applied to every hour; the target's hour 1 is capped to the comps' range.
    slope = 0.0
    if len(comps) >= 4:
        xs = [(w, lr[0], sum(v for k, v in lr.items() if k > 0) / (len(lr) - 1))
              for lr, w in comps]
        sw = sum(w for w, _, _ in xs)
        mx = sum(w * x for w, x, _ in xs) / sw
        my = sum(w * y for w, _, y in xs) / sw
        sxx = sum(w * (x - mx) ** 2 for w, x, _ in xs)
        if sxx > 0:
            slope = sum(w * (x - mx) * (y - my) for w, x, y in xs) / sxx
        slope = min(0.0, max(-SLOPE_CAP, slope))
        lo_b = min(lr[0] for lr, _ in comps)
        hi_b = max(lr[0] for lr, _ in comps)
        tgt = min(max(logp, lo_b), hi_b)
        comps = [({k: (v + slope * (tgt - lr[0]) if k > 0 else v) for k, v in lr.items()}, w)
                 for lr, w in comps]

    rows = []
    for h in range(2, 25):
        house_sd = 0.15 * math.sqrt(h - 1)
        pts = [(w, lr[h]) for lr, w in comps if h in lr]
        n_eff = sum(w for w, _ in pts)
        mix = [(PRIOR_STRENGTH, 0.0, house_sd)]
        if n_eff > 0:
            scale = n_eff  # comparables share weight n_eff in total
            for w, x in pts:
                mix.append((w * scale / n_eff, x, KERNEL_SD))
        qs = []
        prev = 0.0
        for tau in LEVELS:
            q = math.exp(logp + mixture_quantile(mix, tau))
            q = min(max(q, 0.01), 1e9)
            q = max(q, prev)
            q = round(q, 2)
            if q < prev:
                q = prev
            prev = q
            qs.append(q)
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        path = os.path.join(in_dir, name)
        with open(path, encoding="utf-8") as f:
            case = json.load(f)
        cid = case.get("id") or name[:-3]
        try:
            result = forecast_case(case)
        except Exception:
            result = house_forecast(case)
        with open(os.path.join(out_dir, cid + ".out"), "w", encoding="utf-8") as f:
            json.dump(result, f, separators=(",", ":"))


def house_forecast(case):
    p = 1.0
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            try:
                p = max(1.0, float(row.get("viewers")))
            except (TypeError, ValueError):
                p = 1.0
    rows = []
    for h in range(2, 25):
        rows.append({"hour": h, "q": [round(p * math.exp(z * 0.15 * math.sqrt(h - 1)), 2) for z in Z]})
    return {"hours": rows}


if __name__ == "__main__":
    main()
