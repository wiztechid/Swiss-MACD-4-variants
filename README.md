# WISS Live / Forward Test Dashboard

## What this version does
- Downloads the latest daily OHLCV for all 253 valid tickers from Yahoo Finance.
- Recalculates MACD 8-21-5, MA20, MA50, RSI14, and Volume MA20.
- Runs all 4 variants in parallel.
- Creates NEW BUY and NEW EXIT confirmation cards.
- Preserves the same backtest convention: signal at daily close, execution at the next trading day's Open.
- Maintains persistent virtual positions/trade history in `live_state.json`.
- Updates `live_signals.json`, which the dashboard refreshes every 60 seconds.
- Adds a Forward Test history table and Forward Compound chart per ticker/variant.

## Schedule
GitHub Actions is set to 17:00 WITA (09:00 UTC), Monday-Friday.
This is intentionally after IDX's regular daily close, so the Daily signal is confirmed.

## Signal / execution logic
NEW BUY appears on the close-confirmation day.
The virtual entry is executed using the next trading day's Open when the next scheduled update runs.

NEW EXIT works the same way:
- signal is confirmed at close;
- the virtual exit is booked at the next trading day's Open.

This keeps the live forward test apple-to-apple with the historical backtest.

## Online setup
1. Create a GitHub repository.
2. Upload all files/folders in this package, including `.github/workflows/update.yml`.
3. Repository Settings → Pages → deploy from `main` / root.
4. Repository Settings → Actions → General → Workflow permissions → allow Read and write permissions.
5. Run `Update WISS Forward Test` once manually from the Actions tab.
6. Open the GitHub Pages URL.

## Main files
- `index.html` — dashboard
- `update_live.py` — full live signal + state engine
- `live_state.json` — persistent forward portfolio/history
- `live_signals.json` — latest dashboard snapshot
- `requirements.txt` — Python dependencies
- `.github/workflows/update.yml` — 17:00 WITA scheduler

## Important
Yahoo Finance is suitable for this Daily forward-test workflow, but it is not an exchange-grade realtime feed.
