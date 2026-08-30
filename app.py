"""Local Streamlit dashboard for the ETF/futures relative-value research project.

Run locally with: ``streamlit run app.py``. This app is research-only and does
not contain broker, order-routing, or trading functionality.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from etf_futures_rv.backtest import BacktestResult, run_backtest
from etf_futures_rv.config import BacktestConfig, DataConfig, StrategyConfig
from etf_futures_rv.data import download_public_data
from etf_futures_rv.metrics import drawdown_series, performance_metrics
from etf_futures_rv.research import (
    cointegration_test,
    descriptive_statistics,
    normalized_price_series,
    stationarity_test,
    theoretical_fair_value,
)
from etf_futures_rv.strategy import dollar_neutral_notionals, generate_signal
from etf_futures_rv.validation import chronological_split, walk_forward_backtest


st.set_page_config(page_title="ETF Futures RV Research", page_icon="📈", layout="wide")


@st.cache_data(ttl=900, show_spinner=False)
def load_market_data(use_fallback: bool, refresh_nonce: int) -> tuple[pd.DataFrame, str]:
    """Load one explicit Yahoo Finance source and cache it for fifteen minutes."""
    del refresh_nonce  # Changes to this argument intentionally invalidate the cache key.
    prices = download_public_data(DataConfig(), use_fallback=use_fallback)
    downloaded_at = datetime.now().astimezone().isoformat(timespec="seconds")
    return prices, downloaded_at


def line_chart(frame: pd.DataFrame, title: str, yaxis_title: str) -> go.Figure:
    """Create a reusable interactive line chart without altering source values."""
    figure = go.Figure()
    for column in frame.columns:
        figure.add_trace(go.Scatter(x=frame.index, y=frame[column], name=str(column), mode="lines"))
    figure.update_layout(title=title, hovermode="x unified", margin=dict(l=20, r=20, t=45, b=20), legend_title_text="Series")
    figure.update_yaxes(title_text=yaxis_title)
    return figure


def signal_label(direction: float) -> str:
    """Translate established strategy direction values for display."""
    if direction > 0:
        return "Long SPY / Short futures"
    if direction < 0:
        return "Short SPY / Long futures"
    return "Flat"


def metric_value(value: float, kind: str = "number") -> str:
    """Format existing calculated metrics for a readable UI."""
    if pd.isna(value):
        return "N/A"
    if kind == "percent":
        return f"{value:.2%}"
    if kind == "currency":
        return f"${value:,.0f}"
    return f"{value:.2f}"


def mapping_frame(values: dict[str, object]) -> pd.DataFrame:
    """Render heterogeneous diagnostic values without Arrow type coercion warnings."""
    return pd.DataFrame({"Value": [str(value) for value in values.values()]}, index=values.keys())


METRIC_PRESENTATION: tuple[tuple[str, str, str], ...] = (
    ("total_return", "Total return", "percent"),
    ("annualized_return", "Annualized return", "percent"),
    ("annualized_volatility", "Annualized volatility", "percent"),
    ("sharpe_ratio", "Sharpe ratio", "number"),
    ("sortino_ratio", "Sortino ratio", "number"),
    ("maximum_drawdown", "Maximum drawdown", "percent"),
    ("calmar_ratio", "Calmar ratio", "number"),
    ("win_rate", "Win rate", "percent"),
    ("profit_factor", "Profit factor", "number"),
    ("turnover", "Turnover", "number"),
    ("average_gross_exposure", "Average gross exposure", "percent"),
    ("number_of_trades", "Number of trades", "number"),
)


def show_metrics(metrics: dict[str, float]) -> None:
    """Render all standard metrics returned by the package's metrics module."""
    for row_start in range(0, len(METRIC_PRESENTATION), 4):
        columns = st.columns(4)
        for column, (key, label, kind) in zip(columns, METRIC_PRESENTATION[row_start : row_start + 4], strict=True):
            column.metric(label, metric_value(metrics[key], kind))


def latest_signal_reason(zscore: float, direction: float, config: StrategyConfig) -> str:
    """Explain the already-generated target in plain English."""
    if pd.isna(zscore):
        return f"No target is available yet because {config.lookback_days} trailing observations are required."
    if direction > 0:
        return f"The carry residual z-score is {zscore:.2f}, at/above the +{config.entry_z:.2f} entry threshold: futures are rich versus fair value."
    if direction < 0:
        return f"The carry residual z-score is {zscore:.2f}, at/below the -{config.entry_z:.2f} entry threshold: futures are cheap versus fair value."
    return f"The carry residual z-score is {zscore:.2f}; it does not call for a new position, or an existing trade met the ±{config.exit_z:.2f} exit rule."


def show_dashboard(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    fair_value: pd.Series,
    backtest_config: BacktestConfig,
    source_name: str,
    downloaded_at: str,
) -> None:
    """Render current market, fair-value, and target-position status."""
    latest = prices.index[-1]
    latest_signal = signal.iloc[-1]
    etf_notional, futures_notional = dollar_neutral_notionals(
        backtest_config.initial_capital, float(latest_signal["etf_weight"]), backtest_config.gross_exposure
    )
    st.title("Dashboard")
    st.warning("Research-only dashboard — no brokerage connection, order routing, or real-money trading is available.")
    st.caption(f"Selected source: {source_name}. Latest market-data date: {latest:%Y-%m-%d}. Downloaded locally: {downloaded_at}.")

    rows = [
        ("Latest SPY", metric_value(float(prices.iloc[-1]["etf_close"]), "currency")),
        ("Latest futures proxy", f"{float(prices.iloc[-1]['futures_close']):,.2f}"),
        ("Carry fair value (SPY)", metric_value(float(fair_value.iloc[-1]), "currency")),
        ("Carry residual", f"{float(latest_signal['carry_residual']) * 10_000:.2f} bps"),
        ("Current z-score", metric_value(float(latest_signal["zscore"]))),
        ("Current strategy signal", signal_label(float(latest_signal["etf_weight"]))),
        ("SPY target notional", metric_value(etf_notional, "currency")),
        ("Futures target notional", metric_value(futures_notional, "currency")),
    ]
    for start in range(0, len(rows), 4):
        columns = st.columns(4)
        for column, (label, value) in zip(columns, rows[start : start + 4], strict=True):
            column.metric(label, value)

    st.info(latest_signal_reason(float(latest_signal["zscore"]), float(latest_signal["etf_weight"]), StrategyConfig()))
    st.caption(f"Target notionals use the baseline ${backtest_config.initial_capital:,.0f} reference capital and equal-and-opposite dollar legs; targets are not orders.")


def show_research(prices: pd.DataFrame, signal: pd.DataFrame, strategy_config: StrategyConfig) -> None:
    """Render existing research diagnostics and their plain-English interpretation."""
    st.title("Research")
    normalized = normalized_price_series(prices, strategy_config)
    st.plotly_chart(line_chart(normalized, "Normalized SPY, futures proxy, and carry fair value", "Rebased level (first date = 100)"), width="stretch")

    fair_value = theoretical_fair_value(prices, strategy_config)
    fair_value_frame = pd.DataFrame(
        {"SPY": prices["etf_close"], "Futures proxy / 10": prices["futures_close"] / strategy_config.futures_to_etf_scale, "Carry fair value": fair_value}
    )
    st.plotly_chart(line_chart(fair_value_frame, "Futures proxy versus carry fair value", "Price level"), width="stretch")
    st.plotly_chart(line_chart(signal[["carry_residual"]].mul(10_000), "Carry residual / basis", "Basis points"), width="stretch")
    st.plotly_chart(line_chart(signal[["zscore"]], "Trailing carry-residual z-score", "Standard deviations"), width="stretch")

    st.subheader("Summary statistics")
    statistics = descriptive_statistics(prices, strategy_config)
    st.dataframe(mapping_frame(statistics), width="stretch")
    st.caption("Price and return correlation measure co-movement. Residual mean and standard deviation describe the carry-adjusted deviation in basis points; neither demonstrates tradable alpha.")

    left, right = st.columns(2)
    with left:
        st.subheader("ADF stationarity test")
        try:
            adf = stationarity_test(signal["carry_residual"])
            st.dataframe(mapping_frame(adf), width="stretch")
            interpretation = "supports stationarity" if adf["stationary_at_5pct"] else "does not establish stationarity"
            st.info(f"The ADF p-value tests whether the residual behaves like a unit root. Here it {interpretation} at the 5% threshold. This is evidence about the sampled series, not a trading guarantee.")
        except ValueError as exc:
            st.warning(f"ADF unavailable: {exc}")
    with right:
        st.subheader("Engle-Granger cointegration test")
        try:
            coint = cointegration_test(prices)
            st.dataframe(mapping_frame(coint), width="stretch")
            interpretation = "supports cointegration" if coint["cointegrated_at_5pct"] else "does not establish cointegration"
            st.info(f"The test asks whether log SPY and the futures proxy share a stable long-run relationship. Here it {interpretation} at 5%. Continuous-futures roll construction can distort this result.")
        except ValueError as exc:
            st.warning(f"Cointegration unavailable: {exc}")


def show_strategy(signal: pd.DataFrame, strategy_config: StrategyConfig, backtest_config: BacktestConfig) -> None:
    """Show fixed baseline parameters and current target rationale."""
    st.title("Strategy")
    st.info("Parameters are displayed for transparency. This dashboard does not optimize them automatically.")
    left, right = st.columns(2)
    with left:
        st.subheader("Signal and carry parameters")
        st.dataframe(mapping_frame(asdict(strategy_config)), width="stretch")
    with right:
        st.subheader("Position, financing, and execution assumptions")
        st.dataframe(mapping_frame(asdict(backtest_config)), width="stretch")

    st.subheader("Mathematical logic")
    st.latex(r"F_t = S_t e^{(r-q)T}")
    st.latex(r"b_t = \log(F_t/10) - \log(SPY_t e^{(r-q)T})")
    st.markdown("The model computes a trailing z-score of $b_t$. A high positive z-score targets long SPY / short futures; a large negative z-score targets the reverse. Equal-and-opposite dollar legs maintain the intended market-neutral hedge. The backtest delays the target by one full bar.")

    latest = signal.iloc[-1]
    st.subheader("Current signal")
    st.metric("Target", signal_label(float(latest["etf_weight"])))
    st.write(latest_signal_reason(float(latest["zscore"]), float(latest["etf_weight"]), strategy_config))


def show_backtest(prices: pd.DataFrame, signal: pd.DataFrame, backtest_config: BacktestConfig) -> None:
    """Run the established accounting engine on an explicitly selected date range."""
    st.title("Backtest")
    st.caption("Uses the existing timestamped, one-bar-delayed backtester. Historical results do not predict future profitability.")
    selected_dates = st.date_input(
        "Backtest date range",
        value=(prices.index.min().date(), prices.index.max().date()),
        min_value=prices.index.min().date(),
        max_value=prices.index.max().date(),
    )
    initial_capital = st.number_input("Initial capital", min_value=10_000.0, value=float(backtest_config.initial_capital), step=50_000.0, format="%.0f")
    if not isinstance(selected_dates, tuple) or len(selected_dates) != 2:
        st.info("Select both a start and an end date to run the backtest.")
        return
    start, end = (pd.Timestamp(selected_dates[0]), pd.Timestamp(selected_dates[1]))
    selected_prices = prices.loc[start:end]
    if len(selected_prices) < 3:
        st.warning("Select at least three aligned observations for a meaningful backtest.")
        return
    selected_config = BacktestConfig(**{**asdict(backtest_config), "initial_capital": float(initial_capital)})
    try:
        result = run_backtest(selected_prices, signal.loc[selected_prices.index], selected_config)
        show_metrics(performance_metrics(result, selected_config.trading_days))
    except ValueError as exc:
        st.error(f"Backtest could not run: {exc}")
        return

    st.plotly_chart(line_chart(result.daily[["equity"]], "Equity curve", "Portfolio value"), width="stretch")
    st.plotly_chart(line_chart(drawdown_series(result.daily["equity"]).to_frame(), "Drawdown", "Drawdown"), width="stretch")
    st.subheader("Trade and signal history")
    history = signal.loc[selected_prices.index, ["carry_residual", "zscore", "etf_weight", "futures_weight"]].join(
        result.daily[["position", "executed_target", "turnover", "transaction_cost"]], how="left"
    )
    st.dataframe(history.sort_index(ascending=False), width="stretch", height=350)
    st.subheader("Executed trades")
    st.dataframe(result.trades, width="stretch", hide_index=True)


def show_validation(prices: pd.DataFrame, signal: pd.DataFrame, strategy_config: StrategyConfig, backtest_config: BacktestConfig) -> None:
    """Show fixed-rule chronological and expanding-history validation results."""
    st.title("Validation")
    st.warning("Validation is fixed-rule evaluation, not parameter optimization. In-sample performance must not be interpreted as out-of-sample evidence.")
    try:
        train, test = chronological_split(prices)
        in_sample = run_backtest(train, signal.loc[train.index], backtest_config)
        out_of_sample = run_backtest(test, signal.loc[test.index], backtest_config)
        metrics_frame = pd.DataFrame(
            {
                "In-sample (first 70%)": performance_metrics(in_sample, backtest_config.trading_days),
                "Out-of-sample (final 30%)": performance_metrics(out_of_sample, backtest_config.trading_days),
            }
        )
        st.subheader("Chronological split")
        st.caption(f"In-sample: {train.index.min():%Y-%m-%d} to {train.index.max():%Y-%m-%d} ({len(train):,} observations). Out-of-sample: {test.index.min():%Y-%m-%d} to {test.index.max():%Y-%m-%d} ({len(test):,} observations).")
        st.dataframe(metrics_frame, width="stretch")
    except ValueError as exc:
        st.error(f"Chronological validation unavailable: {exc}")
        return

    st.subheader("Expanding-history walk-forward")
    st.caption("Starts after 504 observations and evaluates 126-observation out-of-sample blocks. Parameters remain fixed; results are rebased into one research equity path.")
    try:
        walk_forward = walk_forward_backtest(prices, strategy_config, backtest_config)
        show_metrics(performance_metrics(walk_forward, backtest_config.trading_days))
        st.plotly_chart(line_chart(walk_forward.daily[["equity"]], "Walk-forward out-of-sample equity curve", "Portfolio value"), width="stretch")
    except ValueError as exc:
        st.warning(f"Walk-forward validation unavailable: {exc}")


def show_data(prices: pd.DataFrame, source_name: str, downloaded_at: str) -> None:
    """Display transparent source and aligned-panel quality information."""
    st.title("Data")
    st.subheader("Selected source")
    st.write(source_name)
    st.markdown("Data is downloaded locally from [Yahoo Finance](https://finance.yahoo.com/) through `yfinance`. No brokerage or trading API is used.")
    if source_name.startswith("ES=F"):
        st.warning("`ES=F` is Yahoo Finance's continuous futures proxy, not exchange-grade, contract-specific data. Its unknown roll construction, daily timestamps, and missing executable bid/ask information can materially bias research results.")
    else:
        st.warning("`^GSPC` is an explicitly selected cash-index fallback, not a futures contract. It cannot model futures roll, margin, or a tradable futures basis.")

    summary = mapping_frame(
        {
            "Aligned date range": f"{prices.index.min():%Y-%m-%d} to {prices.index.max():%Y-%m-%d}",
            "Aligned observations": len(prices),
            "Missing values in aligned panel": int(prices.isna().sum().sum()),
            "Duplicate timestamps in aligned panel": int(prices.index.duplicated().sum()),
            "Data freshness": f"{prices.index.max():%Y-%m-%d} (latest aligned market date)",
            "Downloaded locally": downloaded_at,
        }
    )
    st.dataframe(summary, width="stretch")
    st.info("The existing data module normalizes timestamps, keeps the latest duplicate before alignment, inner-joins contemporaneous prices, and drops incomplete rows. The figures above describe the cleaned aligned research panel, not unavailable provider-side raw records.")
    st.dataframe(prices.tail(20).sort_index(ascending=False), width="stretch")


def main() -> None:
    """Configure explicit source controls and render the chosen local dashboard page."""
    strategy_config = StrategyConfig()
    backtest_config = BacktestConfig()
    if "refresh_nonce" not in st.session_state:
        st.session_state.refresh_nonce = 0

    with st.sidebar:
        st.header("Local research controls")
        source_choice = st.radio(
            "Futures data source",
            options=("ES=F — Yahoo Finance continuous E-mini proxy", "^GSPC — explicit cash-index fallback"),
            index=0,
            help="The fallback is selected only when you choose it; the app never substitutes sources automatically.",
        )
        if st.button("Refresh Data", type="primary"):
            st.cache_data.clear()
            st.session_state.refresh_nonce += 1
        page = st.radio("Page", ("Dashboard", "Research", "Strategy", "Backtest", "Validation", "Data"))
        st.caption("Cached locally for up to 15 minutes. Refresh forces a new Yahoo Finance request.")

    use_fallback = source_choice.startswith("^GSPC")
    source_name = "^GSPC — Yahoo Finance cash-index fallback" if use_fallback else "ES=F — Yahoo Finance continuous E-mini proxy"
    with st.spinner("Loading the explicitly selected Yahoo Finance source…"):
        try:
            prices, downloaded_at = load_market_data(use_fallback, int(st.session_state.refresh_nonce))
        except Exception as exc:  # Provider/network errors must remain visible to the researcher.
            st.error(f"Data retrieval failed for {source_name}. No alternate source was used.")
            st.exception(exc)
            st.stop()
    try:
        signal = generate_signal(prices, strategy_config)
        fair_value = theoretical_fair_value(prices, strategy_config)
    except ValueError as exc:
        st.error(f"Research calculations could not run on the downloaded panel: {exc}")
        st.stop()

    if page == "Dashboard":
        show_dashboard(prices, signal, fair_value, backtest_config, source_name, downloaded_at)
    elif page == "Research":
        show_research(prices, signal, strategy_config)
    elif page == "Strategy":
        show_strategy(signal, strategy_config, backtest_config)
    elif page == "Backtest":
        show_backtest(prices, signal, backtest_config)
    elif page == "Validation":
        show_validation(prices, signal, strategy_config, backtest_config)
    else:
        show_data(prices, source_name, downloaded_at)


if __name__ == "__main__":
    main()
