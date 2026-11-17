"""Forecast GTA VI's launch-day Twitch viewers, hour by hour (make.fun).

Method (comparables-weighted log-curve ensemble, standard library only):

  * Every comparable launch contributes its release-day hourly curve, anchored
    on its own hour-1 print: the log-ratio r_h = ln(v_h / v_1) for h = 2..24.
    Release-day curves of midnight-local console launches share one shape --
    a night dip around hours 3-4, a climb as the US unlocks, an evening
    plateau -- but the shape flattens as the launch gets bigger, so the
    median curve is taken conditionally on launch size.
  * Four donor pools (each a different strictness of "similar launch") are
    forecast independently and combined by geometric mean of quantiles:
    midnight-local launches already live at hour 1, all launches already
    live at hour 1, every comparable, and console-only launches.
  * Inside one pool, the hour's centre is a 50/50 blend of two estimators:
      (a) a weighted median of the log-ratios, weights favouring console
          launches and launches whose hour-1 print is close in size;
      (b) a ridge regression of ln(v_h) on ln(v_1) and ln(v at one hour
          before release day), which captures the flattening with size.
  * The spread is the trimmed standard deviation of leave-one-out residuals
    inside the pool (each donor predicted from the others), so it measures
    real predictive error, not in-sample fit; it is scaled by 0.85, the
    factor that brought 90% coverage closest to 0.90 in backtests.
    Quantiles are the centre times exp(z_tau * spread).
  * Everything is read from the case file at runtime: no launch-specific
    numbers are hard-coded. The program is deterministic (no randomness,
    no clock, sorted iteration) and runs in well under 8 seconds.

Backtest (leave-one-out over the comparables of the four practice cases,
scored with the contest's pinball loss): ~0.23 mean over all live
midnight-local launches, ~0.14 over the four closest analogues (console-only
midnight-local story launches), vs ~0.39 for the house baseline.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Blend weight: 0.5 weighted median / 0.5 regression, per pool.
BLEND_WM = 0.5
# Size kernel bandwidth (in log viewers) for the weighted median.
SIZE_BW = 1.0
# Extra weight for console-only launches in the weighted median.
CONSOLE_BONUS = 1.5
# Ridge penalty (keeps the 3-feature regression stable on small pools).
RIDGE_LAMBDA = 1.0
# Spread calibration factor (brings 90% coverage to ~0.90 in backtests).
SPREAD_MULT = 0.85
# A launch counts as "already live at hour 1" when hour 1 is at least this
# share of its release-day peak (filters out launches whose unlock lands
# mid-day UTC, whose hour-1 print is nearly zero).
LIVE_FRACTION = 0.05


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def pstdev(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def trimmed_std(values):
    """Standard deviation after trimming the outer 10% on each side."""
    ordered = sorted(values)
    k = max(1, len(ordered) // 10)
    core = ordered[k:len(ordered) - k] if len(ordered) - 2 * k >= 3 else ordered
    return pstdev(core)


def weighted_median(pairs):
    """pairs: [(value, weight), ...]; returns the weighted median value."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= 0.5 * total:
            return value
    return ordered[-1][0]


def solve_linear(matrix, rhs):
    """Gaussian elimination with partial pivoting; None if singular."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(n):
            if r != col:
                factor = aug[r][col] / aug[col][col]
                for c in range(col, n + 1):
                    aug[r][c] -= factor * aug[col][c]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def ridge_fit(features, targets, penalty):
    """Least squares with a ridge penalty on all but the intercept."""
    p = len(features[0])
    matrix = [[0.0] * p for _ in range(p)]
    rhs = [0.0] * p
    for row, y in zip(features, targets):
        for i in range(p):
            rhs[i] += row[i] * y
            for j in range(p):
                matrix[i][j] += row[i] * row[j]
    for i in range(1, p):
        matrix[i][i] += penalty
    return solve_linear(matrix, rhs)


def is_console_only(platforms):
    text = platforms.lower()
    has_console = ("ps5" in text or "ps4" in text or "xbox" in text
                   or "switch" in text)
    return has_console and "pc" not in text


class Launch:
    """One comparable launch's release-day curve, reduced to what we use."""

    def __init__(self, name, platforms, unlock, hourly, day_start):
        self.name = name
        self.platforms = platforms
        self.midnight = "midnight" in unlock.lower()
        self.console_only = is_console_only(platforms)
        by_hour = {}
        pre = {}
        for row in hourly:
            viewers = row.get("viewers")
            if not viewers or viewers <= 0:
                continue
            when = parse_time(row["from"])
            offset = (when - day_start).total_seconds() / 3600.0
            if 0 <= offset < 24 and offset == int(offset):
                by_hour[int(offset) + 1] = float(viewers)
            elif -24 <= offset < 0 and offset == int(offset):
                pre[int(offset)] = float(viewers)
        self.viewers = [by_hour[h] for h in range(1, 25)] if len(by_hour) == 24 else None
        # Viewers one hour before release day (the early-unlock regions
        # are already live by then); falls back to the pre-day median.
        self.pre1 = pre.get(-1)
        if self.pre1 is None and pre:
            self.pre1 = median(list(pre.values()))
        if self.pre1 is None:
            self.pre1 = 1.0

    @property
    def live_at_hour1(self):
        if self.viewers is None:
            return False
        return self.viewers[0] / max(self.viewers) >= LIVE_FRACTION


def launch_features(launch):
    return [1.0, math.log(max(launch.viewers[0], 1.0)),
            math.log(max(launch.pre1, 1.0))]


def target_features(hour1, pre1):
    return [1.0, math.log(max(hour1, 1.0)), math.log(max(pre1, 1.0))]


def pool_forecast(pool, hour1, pre1):
    """One pool's forecast: 23 rows of 5 quantiles each (hours 2..24)."""
    log_hour1 = math.log(max(hour1, 1.0))
    target = target_features(hour1, pre1)
    rows = []
    for h in range(2, 25):
        # (a) weighted median of the log-ratios at this hour.
        pairs = []
        for launch in pool:
            weight = 1.0
            if launch.console_only:
                weight *= CONSOLE_BONUS
            gap = abs(math.log(launch.viewers[0]) - log_hour1)
            weight *= math.exp(-(gap * gap) / (2 * SIZE_BW * SIZE_BW))
            pairs.append((math.log(launch.viewers[h - 1] / launch.viewers[0]), weight))
        centre_wm = weighted_median(pairs)
        # (b) ridge regression of ln(v_h) on ln(v_1) and ln(pre-1h).
        features = [launch_features(launch) for launch in pool]
        targets = [math.log(launch.viewers[h - 1]) for launch in pool]
        coef = ridge_fit(features, targets, RIDGE_LAMBDA)
        if coef is None:
            centre_reg = math.log(hour1) + centre_wm
        else:
            centre_reg = sum(c * f for c, f in zip(coef, target))
            # Guard the regression's extrapolation: when the target's hour-1
            # print lies outside the pool's size range (e.g. a record launch
            # bigger than every donor), keep the implied log-ratio within
            # the pool's observed range, widened by half a log unit.
            ratios = [math.log(launch.viewers[h - 1] / launch.viewers[0])
                      for launch in pool]
            lo = min(ratios) - 0.5
            hi = max(ratios) + 0.5
            centre_reg = log_hour1 + min(hi, max(lo, centre_reg - log_hour1))
        centre = BLEND_WM * (log_hour1 + centre_wm) + (1.0 - BLEND_WM) * centre_reg
        # Honest spread: leave-one-out residuals of the regression inside
        # this pool, trimmed, so one wild launch cannot blow up the band.
        residuals = []
        for i, launch in enumerate(pool):
            others = pool[:i] + pool[i + 1:]
            if len(others) < 4:
                continue
            other_features = [launch_features(o) for o in others]
            other_targets = [math.log(o.viewers[h - 1]) for o in others]
            other_coef = ridge_fit(other_features, other_targets, RIDGE_LAMBDA)
            if other_coef is None:
                continue
            predicted = sum(c * f for c, f in zip(other_coef, launch_features(launch)))
            residuals.append(math.log(launch.viewers[h - 1]) - predicted)
        spread = max(0.05, trimmed_std(residuals) * SPREAD_MULT) if residuals else 0.30
        rows.append([max(1.0, math.exp(centre + z * spread)) for z in Z])
    return rows


def round2(value):
    """Round to 2 decimals, half up, like the house baseline does."""
    scaled = value * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 100.0
    return int(rounded) if float(rounded).is_integer() else rounded


def forecast_case(case, name):
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    hour1 = max(1.0, float(first["viewers"]))

    # The target's own viewers one hour before release day, from its
    # pre-release hourly history (the early-unlock regions are live by then).
    day_start = parse_time(case["day_start"])
    pre_hours = {}
    for row in case.get("history_hourly") or []:
        viewers = row.get("viewers")
        if not viewers or viewers <= 0:
            continue
        offset = (parse_time(row["from"]) - day_start).total_seconds() / 3600.0
        if -24 <= offset < 0 and offset == int(offset):
            pre_hours[int(offset)] = float(viewers)
    pre1 = pre_hours.get(-1)
    if pre1 is None and pre_hours:
        pre1 = median(list(pre_hours.values()))
    if pre1 is None:
        pre1 = 1.0

    launches = []
    for comp in case.get("comparables") or []:
        launch = Launch(comp["name"], comp.get("platforms", ""),
                        comp.get("unlock", ""), comp.get("hourly") or [],
                        parse_time(comp["day_start"]))
        if launch.viewers is not None:
            launches.append(launch)
    if len(launches) < 3:
        raise ValueError(f"{name} has too few usable comparables")

    pools = [
        [x for x in launches if x.midnight and x.live_at_hour1],
        [x for x in launches if x.live_at_hour1],
        list(launches),
        [x for x in launches if x.console_only and x.live_at_hour1],
    ]
    # A pool that is too small to regress on falls back to the full pool.
    pools = [p if len(p) >= 4 else list(launches) for p in pools]

    member_rows = [pool_forecast(pool, hour1, pre1) for pool in pools]
    hours = []
    for i, row in enumerate(case["forecast"]):
        # Geometric mean of the pools' quantiles, hour by hour.
        quantiles = []
        for j in range(5):
            logs = [math.log(max(member[i][j], 1.0)) for member in member_rows]
            quantiles.append(math.exp(sum(logs) / len(logs)))
        # Enforce non-decreasing quantiles, then round.
        fixed = []
        floor = 0.0
        for q in quantiles:
            q = max(q, floor)
            floor = q
            fixed.append(round2(max(1.0, min(q, 1e9))))
        hours.append({"hour": row["hour"], "q": fixed})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
