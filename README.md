# GTA VI Twitch CCU forecast — make.fun `gta6-twitch-ccu-forecast-01`

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), from everything recorded up to the
end of hour 1. `main.py` (Python 3.13, standard library only) reads one case
file per run and writes five quantiles per hour for hours 2–24.

## Method

Comparables-weighted log-curve ensemble:

- Each comparable launch contributes its release-day hourly curve anchored on
  its own hour-1 print (log-ratios `ln(v_h / v_1)`).
- Four donor pools (midnight-local launches already live at hour 1; all
  launches already live at hour 1; every comparable; console-only launches)
  are forecast independently and combined by geometric mean of quantiles.
- Inside a pool, each hour's centre blends a size-weighted median of the
  log-ratios with a ridge regression of `ln(v_h)` on `ln(v_1)` and
  `ln(viewers one hour before release day)`.
- The spread is the trimmed standard deviation of leave-one-out residuals
  inside the pool (honest predictive error, not in-sample fit), scaled by
  0.85 — the factor that brought 90% coverage closest to 0.90 in backtests.

Leave-one-out backtest on the comparables of the four practice cases
(contest pinball loss): ~0.23 mean over all live midnight-local launches,
~0.14 over the four closest analogues (console-only midnight-local story
launches), vs ~0.39 for the house baseline.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic, no network, runs in a few seconds per case.
