# etf-futures-rv

A reproducible, research-only framework for studying market-neutral relative value between SPY and an S&P 500 futures proxy. It does **not** connect to a brokerage, place orders, or use real money. A backtest is a historical simulation, not evidence that the strategy will be profitable in the future.

## Scope and quick start

The baseline uses public daily Yahoo Finance data:

- `SPY`: the SPDR S&P 500 ETF. Its **adjusted close** is used so historical splits and distributions are reflected in the return series.
- `ES=F`: Yahoo Finance's continuous E-mini S&P 500 futures series. This is the preferred free proxy. It is not an exchange-grade, contract-specific record and should not be treated as executable data.
- `^GSPC`: optional fallback cash-index proxy (`--fallback-index`). It is not a futures price and cannot represent roll mechanics, margin, or the tradable futures basis.

Create a virtual environment, install dependencies, then run the fixed baseline:

```bash
python -m pip install -r requirements.txt
python scripts/run_baseline.py
# If ES=F is unavailable from Yahoo Finance:
python scripts/run_baseline.py --fallback-index
python -m pytest
```

## Local research dashboard

The repository includes a Streamlit dashboard that runs on your Mac only. It has no deployment configuration, no brokerage integration, and no order-placement functionality.

```bash
python -m streamlit run app.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit (normally `http://127.0.0.1:8501`). The sidebar lets you explicitly select either `ES=F` or the `^GSPC` fallback; it never substitutes a source automatically. Data is cached for 15 minutes and **Refresh Data** clears that cache and makes a new Yahoo Finance request. If retrieval fails, the dashboard reports the error and does not display substitute data.

The dashboard pages cover the current research state, diagnostics, fixed strategy assumptions, an interactive date-range backtest, chronological and walk-forward validation, and data-quality/source disclosures. It is a local research interface only; displayed targets are never sent anywhere.

The command caches aligned source data under `data/raw/` and writes `outputs/baseline_report.json`, price/basis/signal, and equity/risk figures. These generated artifacts are excluded from Git because Yahoo Finance can revise history. The notebook equivalents are [01_exploratory_research.ipynb](notebooks/01_exploratory_research.ipynb) and [02_final_strategy_analysis.ipynb](notebooks/02_final_strategy_analysis.ipynb).

## Financial intuition and model

For an equity index futures contract, no-arbitrage cost of carry suggests

\[
F_t = S_t e^{(r-q)T},
\]

where \(r\) is the financing rate, \(q\) the dividend yield, and \(T\) time to expiry. SPY is approximately one-tenth of the S&P 500 index level, so the baseline normalizes the futures proxy by 10. It calculates the log residual

\[
b_t = \log(F_t/10) - \log(SPY_t e^{(r-q)T}).
\]

The residual is standardized only with its trailing 60 daily observations. A z-score at or above +2 enters **long SPY / short futures** because futures are rich relative to carry fair value; a z-score at or below -2 takes the reverse pair. The trade exits once the absolute z-score is at or below 0.5. These are fixed baseline choices, made for interpretability rather than optimized parameters.

Each active pair has equal and opposite dollar notionals (50% of equity per leg at 1.0 gross exposure). The configuration bounds targets to one unit, caps gross exposure, and rejects non-neutral weights. Production implementation would additionally require contract multiplier-aware sizing, liquidity caps, margin, borrow availability, and exchange-level risk controls.

## Data integrity and bias controls

- The data loader normalizes timestamps, removes duplicate dates, inner-joins contemporaneous observations, and drops incomplete rows. It never forward-fills a quote.
- Signal calculations use rolling trailing windows. The backtester delays every target by one full timestamp before it affects close-to-close P&L; the current close therefore cannot influence its own return.
- SPY adjusted prices mitigate split and dividend discontinuities. The futures proxy is not dividend-adjusted; the theoretical carry term explicitly includes a dividend-yield assumption.
- `ES=F` is a vendor-constructed continuous series. Its unknown roll method can introduce jumps and roll bias, and its daily bars omit intraday synchronization and bid/ask information. Direct settlement data for identified contracts and an explicit roll schedule are required before treating results as tradable research.
- There is no constituent selection, so ETF survivorship bias is not the usual issue; however, vendor history revisions and the future contract's continuous-roll construction remain material historical-data biases.

## Backtesting methodology

`etf_futures_rv/backtest.py` is timestamped portfolio accounting. For each date, it first marks P&L on the previously established position, then fills the **prior date's** target and charges the resulting turnover. This conservative sequencing prevents same-bar look-ahead.

Default costs per one-way entry or exit include ETF commission/slippage (1.0 bp), futures commission/slippage (1.0 bp), ETF half spread (0.5 bp), and futures half spread (0.25 bp). Active pairs also incur a simple annualized 4% financing charge on the ETF leg. These are assumptions, not observed trading costs; the `BacktestConfig` makes them easy to stress.

The evaluation splits observations chronologically: the first 70% is for research and the final 30% is out-of-sample. It also performs expanding-history walk-forward evaluation in 126-day test blocks after a 504-day initial research period. Neither process selects parameters. Any later threshold, lookback, carry, or cost variation must be treated as a new hypothesis and evaluated on an untouched holdout.

## Research outputs and metrics

Research utilities report level/return correlation, residual descriptive statistics, Augmented Dickey-Fuller stationarity, and Engle-Granger cointegration. Cointegration is diagnostic only: shared index exposure and continuous-contract rolls can make a low p-value economically misleading.

The report calculates total and annualized return, annualized volatility, Sharpe and Sortino ratios, maximum drawdown, Calmar ratio, win rate, profit factor, turnover, average gross exposure, and number of trades. Figures include normalized price series, carry basis, z-score/active targets, equity curve, drawdown, 63-day rolling Sharpe, and rolling volatility.

## Results and interpretation

No historical performance figure is committed as a project claim because the baseline uses a live public vendor download that can change and this workspace does not bundle market data. Running the command above produces a dated, reproducible artifact from the exact then-available observations and fixed assumptions. Report both the out-of-sample and walk-forward results, including costs, before forming any conclusion.

Even a favorable result may be explained by stale closes, ETF/futures timing mismatch, a continuous-contract roll artifact, unmodelled dividends or funding, or data snooping. The apparent dislocation may be too small or too short-lived to survive realistic execution. This project therefore supports hypothesis testing and risk analysis only; it makes no claim of economic validity or future profitability.

## Project layout

```text
etf_futures_rv/     typed implementation: data, research, strategy, backtest, metrics, validation, plots
data/               generated data policy and cache location
research/           research workflow notes
strategy/            baseline strategy notes
backtest/            accounting methodology notes
metrics/             metric definitions
visualization/       figure workflow notes
notebooks/           exploratory and final-analysis notebooks
tests/               unit and explicit no-look-ahead tests
scripts/             reproducible baseline runner
```

## Testing

Run `python -m pytest`. The suite covers timestamp alignment and missing data, dollar-neutral sizing, signal bounds, transaction-cost accounting, delayed execution, requested metrics, and two tests that alter a future price and verify that earlier signals and backtest results do not change.
