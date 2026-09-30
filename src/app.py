"""Interactive mean-variance dashboard.

Run it from the root of the repository with:

    streamlit run src/app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import mean_variance

EXTRACT_PATH = config.DATA_DIR / "crsp_monthly_returns.csv"

st.set_page_config(page_title="Markowitz Portfolio Selection", layout="wide")


@st.cache_data
def load_returns():
    """Load the CRSP extract if it is there. Otherwise simulate returns, so the
    app still runs right after cloning the repository."""
    if EXTRACT_PATH.exists():
        df = pd.read_csv(EXTRACT_PATH, parse_dates=["date"], index_col="date")
        return df.drop(columns=["MKT", "RF"]), df["RF"].mean(), True

    rng = np.random.default_rng(0)
    names = [f"Stock {c}" for c in "ABCDEFGH"]
    market = rng.normal(0.008, 0.045, size=300)
    betas = rng.uniform(0.5, 1.5, size=len(names))
    noise = rng.normal(0.002, 0.06, size=(300, len(names)))
    dates = pd.date_range("2000-01-31", periods=300, freq="ME")
    simulated = pd.DataFrame(np.outer(market, betas) + noise, index=dates, columns=names)
    return simulated, 0.0015, False


@st.cache_data
def no_short_frontier(mu, Sigma, n_points=50):
    """The no-short frontier has no closed form, so it is solved point by point."""
    targets = np.linspace(mu.min(), mu.max(), n_points)
    weights = mean_variance.frontier(mu, Sigma, targets, allow_short=False)
    vols = np.sqrt([w @ Sigma @ w for w in weights])
    return targets, weights, vols


@st.cache_data
def constrained_risk_free_frontier(mu, Sigma, rf, targets, allow_short, allow_borrowing):
    """Like the no-short frontier, this is solved point by point. Targets that
    the constraints rule out get a volatility of NaN, which leaves a gap."""
    vols = np.full(len(targets), np.nan)
    for i, target in enumerate(targets):
        try:
            w = mean_variance.minimize_variance_with_risk_free(
                mu, Sigma, rf, target, allow_short, allow_borrowing
            )
            vols[i] = np.sqrt(w @ Sigma @ w)
        except RuntimeError:
            pass
    return vols


def two_asset_stats(w, mu_a, mu_b, sd_a, sd_b, rho):
    """Mean and volatility of a portfolio with weight w on asset a and 1 - w on asset b."""
    mean = w * mu_a + (1 - w) * mu_b
    var = w**2 * sd_a**2 + (1 - w) ** 2 * sd_b**2 + 2 * w * (1 - w) * rho * sd_a * sd_b
    return mean, np.sqrt(np.maximum(var, 0))


# The risk-free rate field shows two decimals of a percent. A rate that matches
# the mean of the minimum variance portfolio to that precision counts as equal.
RF_TOL = 0.5e-4


def risk_free_rate_input(default, key):
    rate = st.number_input(
        "Risk-free rate (annualized, %)", min_value=-20.0, max_value=100.0,
        value=round(100 * default, 2), step=0.25, format="%.2f", key=key,
    )
    return rate / 100


def format_sharpe(excess, vol):
    if vol < 1e-9:
        return "n/a" if abs(excess) < 1e-9 else ("∞" if excess > 0 else "-∞")
    return f"{excess / vol:.2f}"


def add_risk_free_frontier(fig, rf, slope, x_max, **line):
    """With a risk-free asset, the frontier is two lines that leave (0, rf) with
    opposite slopes. The upper one is the efficient part."""
    fig.add_scatter(
        x=[x_max, 0, x_max], y=[rf - slope * x_max, rf, rf + slope * x_max],
        mode="lines", name="Frontier with the risk-free asset", line=line,
    )


def explain_tangency(rf, mean_gmv, tangency, on_chart):
    """Say where the tangency portfolio went. `tangency` is its annualized
    (mean, volatility), or None if it does not exist."""
    if tangency is None:
        st.info(
            "The risk-free rate equals the expected return of the minimum variance "
            f"portfolio ({mean_gmv:.2%}), so there is no tangency portfolio. The two "
            "frontier lines are the asymptotes of the curve: they approach it and never "
            "touch it. Every frontier portfolio puts 100% of wealth in the risk-free "
            "asset and adds a long-short position in the risky assets that costs nothing."
        )
        return
    if rf > mean_gmv:
        st.info(
            "The risk-free rate is above the expected return of the minimum variance "
            f"portfolio ({mean_gmv:.2%}). The tangency portfolio is now on the lower, "
            "inefficient half of the curve. Efficient portfolios short it and put more "
            "than 100% of wealth in the risk-free asset, so the upper line never touches "
            "the curve."
        )
    if not on_chart:
        st.caption(
            f"The tangency portfolio is off the chart, at an expected return of "
            f"{tangency[0]:.0%} and a volatility of {tangency[1]:.0%}. It runs off to "
            "infinity as the risk-free rate approaches the expected return of the "
            "minimum variance portfolio."
        )


returns, rf, using_crsp = load_returns()

st.title("Portfolio Selection (Markowitz, 1952)")
if not using_crsp:
    st.warning(
        "Showing simulated returns because `_data/crsp_monthly_returns.csv` was "
        "not found. Run `doit` (or `python src/pull_crsp.py`) to download the data."
    )

tab_two, tab_many, tab_two_rf, tab_many_rf = st.tabs(
    ["Two assets", "Many assets", "Two assets + risk-free", "Many assets + risk-free"]
)

with tab_two:
    left, right = st.columns([1, 3])
    with left:
        a = st.selectbox("Asset 1", returns.columns, index=0)
        b = st.selectbox("Asset 2", returns.columns, index=1)
        sample_rho = float(returns[a].corr(returns[b]))
        rho = st.slider("Correlation", -1.0, 1.0, round(sample_rho, 2), 0.01)
        w_a = st.slider(f"Weight on {a}", -0.5, 1.5, 0.5, 0.01)
        st.caption(f"The correlation in the data is {sample_rho:.2f}.")

    mu_a, mu_b = 12 * returns[a].mean(), 12 * returns[b].mean()
    sd_a, sd_b = np.sqrt(12) * returns[a].std(), np.sqrt(12) * returns[b].std()

    grid = np.linspace(-0.5, 1.5, 201)
    mean_grid, sd_grid = two_asset_stats(grid, mu_a, mu_b, sd_a, sd_b, rho)
    mean_now, sd_now = two_asset_stats(w_a, mu_a, mu_b, sd_a, sd_b, rho)

    with right:
        col1, col2, col3 = st.columns(3)
        col1.metric("Expected return", f"{mean_now:.1%}")
        col2.metric("Volatility", f"{sd_now:.1%}")
        col3.metric(
            "Weighted average of the two volatilities",
            f"{abs(w_a) * sd_a + abs(1 - w_a) * sd_b:.1%}",
        )

        fig = go.Figure()
        fig.add_scatter(x=sd_grid, y=mean_grid, mode="lines", name="All mixes of the two")
        fig.add_scatter(
            x=[sd_a, sd_b], y=[mu_a, mu_b], mode="markers+text", text=[a, b],
            textposition="top center", marker=dict(size=10), name="Assets",
        )
        fig.add_scatter(
            x=[sd_now], y=[mean_now], mode="markers",
            marker=dict(size=16, symbol="star"), name="Your portfolio",
        )
        fig.update_layout(
            xaxis_title="Volatility (annualized)", yaxis_title="Expected return (annualized)",
            xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=480,
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "The expected return moves in a straight line as you change the weight. "
            "The volatility does not. Drag the correlation toward -1 and watch the curve bend."
        )

with tab_many:
    left, right = st.columns([1, 3])
    with left:
        chosen = st.multiselect("Assets", list(returns.columns), default=list(returns.columns))
        years = sorted(returns.index.year.unique())
        start, end = st.select_slider(
            "Estimation window", options=years, value=(years[0], years[-1])
        )
        no_shorts = st.checkbox("No short sales (weights cannot be negative)")
        st.caption(
            "Shorten or shift the estimation window and watch the tangency weights. "
            "The minimum variance weights move much less. With shorts allowed, the "
            "frontier comes from a formula. With no short sales there is no formula, "
            "so each point is solved numerically."
        )

    sample = returns.loc[str(start) : str(end), chosen]
    if len(chosen) < 2 or len(sample) <= len(chosen):
        st.info("Choose at least two assets and a window with more months than assets.")
    else:
        mu = sample.mean().values
        Sigma = sample.cov().values

        def stats(w):
            return 12 * w @ mu, np.sqrt(12 * w @ Sigma @ w)

        # Unconstrained frontier, from the closed form.
        A, B, _, _ = mean_variance.frontier_constants(mu, Sigma)
        top = max(mu.max(), B / A) * 1.6
        targets = np.linspace(min(mu.min(), 0), top, 200)
        vols = np.sqrt(mean_variance.frontier_variance(mu, Sigma, targets))

        w_gmv = mean_variance.minimize_variance(mu, Sigma, None, allow_short=not no_shorts)
        try:
            w_tan = mean_variance.max_sharpe_weights(mu, Sigma, rf, allow_short=not no_shorts)
        except RuntimeError:
            w_tan = None  # e.g. no asset beat the risk-free rate in this window

        with right:
            fig = go.Figure()
            fig.add_scatter(
                x=np.sqrt(12) * vols, y=12 * targets, mode="lines", name="Frontier, shorts allowed",
                line=dict(dash="dot" if no_shorts else "solid"),
            )
            if no_shorts:
                targets_ns, weights_ns, vols_ns = no_short_frontier(mu, Sigma)
                fig.add_scatter(
                    x=np.sqrt(12) * vols_ns, y=12 * targets_ns, mode="lines",
                    name="Frontier, no shorts", line=dict(width=4),
                )
            fig.add_scatter(
                x=np.sqrt(12) * sample.std(), y=12 * sample.mean(), mode="markers+text",
                text=chosen, textposition="top center", name="Assets",
                marker=dict(color="gray"),
            )
            mean_gmv, sd_gmv = stats(w_gmv)
            fig.add_scatter(
                x=[sd_gmv], y=[mean_gmv], mode="markers",
                marker=dict(size=14, symbol="diamond"), name="Minimum variance",
            )
            if w_tan is not None:
                mean_tan, sd_tan = stats(w_tan)
                fig.add_scatter(
                    x=[sd_tan], y=[mean_tan], mode="markers",
                    marker=dict(size=16, symbol="star"), name="Tangency",
                )
            fig.update_layout(
                xaxis_title="Volatility (annualized)", yaxis_title="Expected return (annualized)",
                xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=520,
                xaxis_range=[0, float(np.sqrt(12) * sample.std().max()) * 1.1],
            )
            st.plotly_chart(fig, width="stretch")

            weights = pd.DataFrame({"Minimum variance": w_gmv}, index=chosen)
            if w_tan is not None:
                weights.insert(0, "Tangency", w_tan)
            else:
                st.info("No tangency portfolio: no asset beat the risk-free rate in this window.")
            bars = go.Figure()
            for name in weights.columns:
                bars.add_bar(x=weights.index, y=weights[name], name=name)
            bars.update_layout(
                barmode="group", yaxis_tickformat=".0%", yaxis_title="Portfolio weight", height=320
            )
            st.plotly_chart(bars, width="stretch")

            if no_shorts:
                efficient = targets_ns >= weights_ns[np.argmin(vols_ns)] @ mu
                area = go.Figure()
                for i, name in enumerate(chosen):
                    area.add_scatter(
                        x=12 * targets_ns[efficient], y=weights_ns[efficient, i],
                        mode="lines", stackgroup="one", name=name,
                    )
                area.update_layout(
                    title="Composition of the efficient no-short frontier",
                    xaxis_title="Target expected return (annualized)",
                    yaxis_title="Portfolio weight",
                    xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=380,
                )
                st.plotly_chart(area, width="stretch")

with tab_two_rf:
    left, right = st.columns([1, 3])
    with left:
        a = st.selectbox("Asset 1", returns.columns, index=0, key="rf_asset_1")
        b = st.selectbox("Asset 2", returns.columns, index=1, key="rf_asset_2")
        sample_rho = float(returns[a].corr(returns[b]))
        rho = st.slider(
            "Correlation", -1.0, 1.0, round(sample_rho, 2), 0.01, key="rf_correlation"
        )
        rf_annual = risk_free_rate_input(12 * rf, key="rf_two_assets")
        w_a = st.slider(f"Weight on {a} in your risky mix", -0.5, 1.5, 0.5, 0.01, key="rf_w_a")
        w_f = st.slider(
            "Share of wealth in the risk-free asset", -1.0, 1.0, 0.0, 0.01, key="rf_w_f"
        )
        st.caption(
            f"The correlation in the data is {sample_rho:.2f}. A negative share in the "
            "risk-free asset means borrowing at the risk-free rate."
        )

    mu_a, mu_b = 12 * returns[a].mean(), 12 * returns[b].mean()
    sd_a, sd_b = np.sqrt(12) * returns[a].std(), np.sqrt(12) * returns[b].std()
    cov_ab = rho * sd_a * sd_b
    mu = np.array([mu_a, mu_b])
    Sigma = np.array([[sd_a**2, cov_ab], [cov_ab, sd_b**2]])

    # With a correlation of +1 or -1, Sigma is singular and some mix of the two
    # assets has no risk. Unless that mix earns the risk-free rate, it is an arbitrage.
    z, H, arbitrage = mean_variance.risk_free_frontier(mu, Sigma, rf_annual)
    singular = abs(rho) > 1 - 1e-9

    # Minimum variance mix. If every mix has the same variance, any of them will do.
    spread_var = sd_a**2 + sd_b**2 - 2 * cov_ab
    w_gmv = (sd_b**2 - cov_ab) / spread_var if spread_var > 1e-12 else 0.5
    mean_gmv, _ = two_asset_stats(w_gmv, mu_a, mu_b, sd_a, sd_b, rho)

    if arbitrage or H < 1e-14 or abs(mean_gmv - rf_annual) < RF_TOL:
        tangency = None
    else:
        tangency = two_asset_stats(z[0] / z.sum(), mu_a, mu_b, sd_a, sd_b, rho)

    mean_mix, sd_mix = two_asset_stats(w_a, mu_a, mu_b, sd_a, sd_b, rho)
    mean_now = w_f * rf_annual + (1 - w_f) * mean_mix
    sd_now = abs(1 - w_f) * sd_mix

    # Keep the chart on the assets. The tangency portfolio can be arbitrarily
    # far away, so it only stretches the axes when it is reasonably close.
    on_chart = tangency is not None and tangency[1] <= 2 * max(sd_a, sd_b)
    xs = [sd_a, sd_b, sd_now] + ([tangency[1]] if on_chart else [])
    ys = [0, mu_a, mu_b, rf_annual, mean_now] + ([tangency[0]] if on_chart else [])
    x_max = 1.15 * max(xs)
    pad = 0.15 * max(max(ys) - min(ys), 0.02)
    y_range = [min(ys) - pad, max(ys) + 2 * pad]

    with right:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Expected return", f"{mean_now:.1%}")
        col2.metric("Volatility", f"{sd_now:.1%}")
        col3.metric("Sharpe ratio of your risky mix", format_sharpe(mean_mix - rf_annual, sd_mix))
        col4.metric("Highest possible Sharpe ratio", "∞" if arbitrage else f"{np.sqrt(H):.2f}")

        grid = np.linspace(-6, 7, 1301)
        mean_grid, sd_grid = two_asset_stats(grid, mu_a, mu_b, sd_a, sd_b, rho)

        fig = go.Figure()
        fig.add_scatter(x=sd_grid, y=mean_grid, mode="lines", name="All mixes of the two")
        if arbitrage:
            fig.add_scatter(
                x=[0, 0], y=y_range, mode="lines", line=dict(width=4),
                name="Frontier with the risk-free asset",
            )
        else:
            add_risk_free_frontier(fig, rf_annual, np.sqrt(H), x_max)
        if sd_mix > 1e-9:
            fig.add_scatter(
                x=[0, x_max], y=[rf_annual, rf_annual + (mean_mix - rf_annual) / sd_mix * x_max],
                mode="lines", line=dict(dash="dash", width=1.5),
                name="Your capital allocation line",
            )
        fig.add_scatter(
            x=[sd_a, sd_b, 0], y=[mu_a, mu_b, rf_annual], mode="markers+text",
            text=[a, b, "Risk-free"], textposition=["top center", "top center", "middle right"],
            marker=dict(size=10), name="Assets",
        )
        if tangency is not None:
            fig.add_scatter(
                x=[tangency[1]], y=[tangency[0]], mode="markers",
                marker=dict(size=16, symbol="star"), name="Tangency",
            )
        fig.add_scatter(
            x=[sd_mix], y=[mean_mix], mode="markers",
            marker=dict(size=12, symbol="circle-open", line=dict(width=3)),
            name="Your risky mix",
        )
        fig.add_scatter(
            x=[sd_now], y=[mean_now], mode="markers",
            marker=dict(size=14, symbol="diamond"), name="Your portfolio",
        )
        fig.update_layout(
            xaxis_title="Volatility (annualized)", yaxis_title="Expected return (annualized)",
            xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=520,
            xaxis_range=[0, x_max], yaxis_range=y_range,
        )
        st.plotly_chart(fig, width="stretch")

        if arbitrage:
            if spread_var > 1e-12:
                riskless, _ = two_asset_stats(w_gmv, mu_a, mu_b, sd_a, sd_b, rho)
                detail = (
                    f"Put {w_gmv:.0%} in {a} and {1 - w_gmv:.0%} in {b} and the risk "
                    f"cancels exactly. That mix earns {riskless:.2%} for sure, and the "
                    f"risk-free rate is {rf_annual:.2%}."
                )
            else:
                detail = (
                    f"{a} and {b} now carry exactly the same risk but have different "
                    "expected returns, so buying one and shorting the other is a sure profit."
                )
            st.warning(
                f"With a correlation of {rho:.0f}, this is an arbitrage. {detail} Borrow at "
                "the lower of the two sure rates and invest at the higher, and any expected "
                "return can be had with no risk. The frontier is the vertical axis, and "
                "there is no tangency portfolio. Prices like these could not last."
            )
        elif H < 1e-14:
            st.info(
                "Both assets have an expected return equal to the risk-free rate. Risk "
                "earns nothing, so the frontier is the risk-free asset alone."
            )
        elif singular:
            st.info(
                f"With a correlation of {rho:.0f}, the two assets are the same bet, so "
                "together they span only one capital allocation line."
            )
        else:
            explain_tangency(rf_annual, mean_gmv, tangency, on_chart)
        st.caption(
            "The dashed line shows every portfolio you can build from your risky mix and "
            "the risk-free asset. Its slope is the Sharpe ratio of the mix. Change the "
            "weight until the dashed line is as steep as it can be. That mix is the "
            "tangency portfolio. Then change the risk-free rate and watch it move."
        )

with tab_many_rf:
    left, right = st.columns([1, 3])
    with left:
        chosen = st.multiselect(
            "Assets", list(returns.columns), default=list(returns.columns), key="rf_assets"
        )
        years = sorted(returns.index.year.unique())
        start, end = st.select_slider(
            "Estimation window", options=years, value=(years[0], years[-1]), key="rf_window"
        )
        rf_annual = risk_free_rate_input(12 * rf, key="rf_many_assets")
        target_annual = st.number_input(
            "Target expected return (annualized, %)", min_value=-50.0, max_value=150.0,
            value=10.0, step=0.5, format="%.2f", key="rf_target",
        ) / 100
        no_shorts = st.checkbox("No short sales of the risky assets", key="rf_no_shorts")
        no_borrowing = st.checkbox("No borrowing at the risk-free rate", key="rf_no_borrowing")

    sample = returns.loc[str(start) : str(end), chosen]
    if len(chosen) < 2 or len(sample) <= len(chosen):
        st.info("Choose at least two assets and a window with more months than assets.")
    elif np.linalg.cond(sample.cov().values) > 1e12:
        st.info("These returns are too close to collinear. Choose other assets or a longer window.")
    else:
        mu = sample.mean().values
        Sigma = sample.cov().values
        rf_monthly = rf_annual / 12
        constrained = no_shorts or no_borrowing

        def stats(w):
            """Annualized mean and volatility. What is not in the risky assets earns rf."""
            return 12 * (rf_monthly + w @ (mu - rf_monthly)), np.sqrt(12 * w @ Sigma @ w)

        A, B, _, _ = mean_variance.frontier_constants(mu, Sigma)
        mean_gmv = 12 * B / A
        z, H, _ = mean_variance.risk_free_frontier(mu, Sigma, rf_monthly)

        if no_shorts:
            try:
                w_tan = mean_variance.max_sharpe_weights(mu, Sigma, rf_monthly, allow_short=False)
            except RuntimeError:
                w_tan = None  # no asset beat the risk-free rate
        elif abs(mean_gmv - rf_annual) < RF_TOL:
            w_tan = None
        else:
            w_tan = z / z.sum()
        tangency = None if w_tan is None else stats(w_tan)

        try:
            w_target = mean_variance.minimize_variance_with_risk_free(
                mu, Sigma, rf_monthly, target_annual / 12, not no_shorts, not no_borrowing
            )
        except RuntimeError:
            w_target = None

        with left:
            st.caption(
                "The minimum variance portfolio of the risky assets has an expected "
                f"return of {mean_gmv:.2%}. Try a risk-free rate below it, equal to it, "
                "and above it. With no constraints, the frontier comes from a formula. "
                "With constraints, each point is solved numerically."
            )

        # Keep the chart on the assets. The tangency portfolio can be arbitrarily
        # far away, so it only stretches the axes when it is reasonably close.
        asset_vols = np.sqrt(12) * sample.std()
        on_chart = tangency is not None and tangency[1] <= 2 * asset_vols.max()
        x_max = 1.1 * max(asset_vols.max(), tangency[1] if on_chart else 0)
        ys = [0, 12 * mu.min(), 12 * mu.max(), rf_annual] + ([tangency[0]] if on_chart else [])
        span = max(ys) - min(ys)
        y_range = [min(ys) - 0.1 * span, max(ys) + 0.35 * span]

        with right:
            if no_shorts:
                best = "n/a" if tangency is None else format_sharpe(tangency[0] - rf_annual, tangency[1])
            else:
                best = f"{np.sqrt(12 * H):.2f}"
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Highest possible Sharpe ratio", best)
            col2.metric(
                "Expected return of the tangency portfolio",
                "None" if tangency is None else f"{tangency[0]:.1%}",
            )
            col3.metric(
                "Volatility of the target portfolio",
                "Infeasible" if w_target is None else f"{stats(w_target)[1]:.1%}",
            )
            col4.metric(
                "Target's weight on the risk-free asset",
                "Infeasible" if w_target is None else f"{1 - w_target.sum():.0%}",
            )

            fig = go.Figure()
            targets = np.linspace(y_range[0], y_range[1], 300) / 12
            vols = np.sqrt(mean_variance.frontier_variance(mu, Sigma, targets))
            fig.add_scatter(
                x=np.sqrt(12) * vols, y=12 * targets, mode="lines", name="Risky assets only",
            )
            add_risk_free_frontier(
                fig, rf_annual, np.sqrt(12 * H), x_max, dash="dot" if constrained else "solid"
            )
            if constrained:
                # The frontier has kinks at the risk-free rate and can end at the
                # most extreme asset, so make sure those targets are on the grid.
                targets_c = np.unique(np.concatenate(
                    [np.linspace(y_range[0], y_range[1], 60) / 12, [rf_monthly, mu.min(), mu.max()]]
                ))
                vols_c = constrained_risk_free_frontier(
                    mu, Sigma, rf_monthly, targets_c, not no_shorts, not no_borrowing
                )
                fig.add_scatter(
                    x=np.sqrt(12) * vols_c, y=12 * targets_c, mode="lines",
                    name="Frontier with your constraints", line=dict(width=4),
                )
            fig.add_scatter(
                x=list(asset_vols) + [0], y=list(12 * sample.mean()) + [rf_annual],
                mode="markers+text", text=chosen + ["Risk-free"],
                textposition=["top center"] * len(chosen) + ["middle right"],
                name="Assets", marker=dict(color="gray"),
            )
            if tangency is not None:
                fig.add_scatter(
                    x=[tangency[1]], y=[tangency[0]], mode="markers",
                    marker=dict(size=16, symbol="star"), name="Tangency",
                )
            if w_target is not None:
                mean_target, sd_target = stats(w_target)
                fig.add_scatter(
                    x=[sd_target], y=[mean_target], mode="markers",
                    marker=dict(size=14, symbol="diamond"), name="Target portfolio",
                )
            fig.update_layout(
                xaxis_title="Volatility (annualized)", yaxis_title="Expected return (annualized)",
                xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=520,
                xaxis_range=[0, x_max], yaxis_range=y_range,
            )
            st.plotly_chart(fig, width="stretch")

            if not no_shorts:
                explain_tangency(rf_annual, mean_gmv, tangency, on_chart)
            elif tangency is None:
                st.info(
                    "No asset has an expected return above the risk-free rate. Without "
                    "short sales, nothing can beat holding the risk-free asset alone, so "
                    "the efficient frontier is that single point and there is no "
                    "tangency portfolio."
                )
            elif not on_chart:
                st.caption(
                    f"The tangency portfolio is off the chart, at an expected return of "
                    f"{tangency[0]:.0%} and a volatility of {tangency[1]:.0%}."
                )
            if w_target is None:
                st.info(
                    f"No portfolio reaches an expected return of {target_annual:.2%} "
                    "under these constraints."
                )
            elif target_annual < rf_annual:
                st.caption(
                    "The target is below the risk-free rate, so the target portfolio is on "
                    "the lower, inefficient line. It has the least variance of any portfolio "
                    "with that expected return, but the risk-free asset alone pays more with "
                    "no risk."
                )

            weights = pd.DataFrame(index=chosen + ["Risk-free"])
            if w_tan is not None:
                weights["Tangency"] = np.append(w_tan, 0)
            if w_target is not None:
                weights["Target portfolio"] = np.append(w_target, 1 - w_target.sum())
            if len(weights.columns) > 0:
                bars = go.Figure()
                for name in weights.columns:
                    bars.add_bar(x=weights.index, y=weights[name], name=name)
                bars.update_layout(
                    barmode="group", yaxis_tickformat=".0%", yaxis_title="Portfolio weight",
                    height=320,
                )
                st.plotly_chart(bars, width="stretch")
                st.caption(
                    "Change the target and the risky weights of the target portfolio all "
                    "scale together. Only the split between the tangency portfolio and the "
                    "risk-free asset moves, unless a constraint gets in the way."
                )
