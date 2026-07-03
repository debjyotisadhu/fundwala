"""
Indian Mutual Fund Analysis Dashboard
======================================
A comprehensive Streamlit dashboard for analysing Indian mutual funds using
free, publicly available NAV data from mfapi.in (sourced from AMFI).

Run:
    pip install streamlit pandas numpy plotly requests
    streamlit run mf_dashboard.py
"""

import datetime as dt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fundwala",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    div[data-testid="stMetric"] {
        background: rgba(28, 131, 225, 0.06);
        border: 1px solid rgba(28, 131, 225, 0.15);
        border-radius: 10px;
        padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

API_BASE = "https://api.mfapi.in/mf"
RISK_FREE_RATE = 0.065  # ~ current Indian 10Y G-Sec / T-bill proxy
TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# Data layer (cached)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def search_schemes(query: str) -> list[dict]:
    """Search schemes by name via mfapi.in."""
    try:
        r = requests.get(f"{API_BASE}/search", params={"q": query}, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_nav_history(scheme_code: int) -> tuple[dict, pd.DataFrame]:
    """Fetch full NAV history for a scheme. Returns (meta, dataframe)."""
    r = requests.get(f"{API_BASE}/{scheme_code}", timeout=30)
    r.raise_for_status()
    payload = r.json()
    meta = payload.get("meta", {})
    df = pd.DataFrame(payload.get("data", []))
    if df.empty:
        return meta, df
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    df = df.set_index("date")
    return meta, df


# ---------------------------------------------------------------------------
# Analytics helpers
# ---------------------------------------------------------------------------
def nav_on_or_before(df: pd.DataFrame, date: pd.Timestamp) -> float | None:
    sub = df.loc[:date]
    return float(sub["nav"].iloc[-1]) if not sub.empty else None


def trailing_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Point-to-point trailing returns; annualised (CAGR) beyond 1 year."""
    end_date = df.index[-1]
    latest = float(df["nav"].iloc[-1])
    periods = {
        "1 Week": dt.timedelta(weeks=1),
        "1 Month": pd.DateOffset(months=1),
        "3 Months": pd.DateOffset(months=3),
        "6 Months": pd.DateOffset(months=6),
        "YTD": None,  # special-cased
        "1 Year": pd.DateOffset(years=1),
        "3 Years": pd.DateOffset(years=3),
        "5 Years": pd.DateOffset(years=5),
        "10 Years": pd.DateOffset(years=10),
        "Since Inception": None,  # special-cased
    }
    rows = []
    for label, offset in periods.items():
        if label == "YTD":
            start = pd.Timestamp(end_date.year, 1, 1)
        elif label == "Since Inception":
            start = df.index[0]
        else:
            start = end_date - offset
        if start < df.index[0] and label not in ("Since Inception",):
            rows.append((label, np.nan))
            continue
        start_nav = nav_on_or_before(df, start)
        if not start_nav:
            rows.append((label, np.nan))
            continue
        years = (end_date - max(start, df.index[0])).days / 365.25
        if years > 1.0:
            ret = (latest / start_nav) ** (1 / years) - 1
        else:
            ret = latest / start_nav - 1
        rows.append((label, ret * 100))
    out = pd.DataFrame(rows, columns=["Period", "Return (%)"])
    return out


def calendar_year_returns(df: pd.DataFrame) -> pd.DataFrame:
    yearly = df["nav"].resample("YE").last()
    rets = yearly.pct_change().dropna() * 100
    return pd.DataFrame({"Year": rets.index.year.astype(str), "Return (%)": rets.values})


def rolling_returns(df: pd.DataFrame, years: int) -> pd.Series:
    """Rolling CAGR computed on a calendar-day grid so windows are true N-year spans."""
    daily = df["nav"].resample("D").ffill()
    window = int(round(365.25 * years))
    past = daily.shift(window)
    cagr = (daily / past) ** (1 / years) - 1
    return (cagr.dropna() * 100)


def drawdown_series(df: pd.DataFrame) -> pd.Series:
    nav = df["nav"]
    peak = nav.cummax()
    return (nav / peak - 1) * 100


def risk_metrics(df: pd.DataFrame) -> dict:
    daily_ret = df["nav"].pct_change().dropna()
    if daily_ret.empty:
        return {}
    ann_vol = daily_ret.std() * np.sqrt(TRADING_DAYS)
    ann_ret = (1 + daily_ret.mean()) ** TRADING_DAYS - 1
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else np.nan
    downside = daily_ret[daily_ret < 0]
    dstd = downside.std() * np.sqrt(TRADING_DAYS)
    sortino = (ann_ret - RISK_FREE_RATE) / dstd if dstd and dstd > 0 else np.nan
    dd = drawdown_series(df)
    positive_days = (daily_ret > 0).mean() * 100
    return {
        "Annualised Return": ann_ret * 100,
        "Annualised Volatility": ann_vol * 100,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": dd.min(),
        "Positive Days (%)": positive_days,
        "Best Day (%)": daily_ret.max() * 100,
        "Worst Day (%)": daily_ret.min() * 100,
    }


def xirr(cashflows: list[tuple[pd.Timestamp, float]], guess: float = 0.12) -> float | None:
    """Newton-Raphson XIRR. Cashflows: negative = investment, positive = value."""
    if len(cashflows) < 2:
        return None
    t0 = cashflows[0][0]
    times = np.array([(d - t0).days / 365.25 for d, _ in cashflows])
    amounts = np.array([a for _, a in cashflows])

    def npv(rate):
        return np.sum(amounts / (1 + rate) ** times)

    def d_npv(rate):
        return np.sum(-times * amounts / (1 + rate) ** (times + 1))

    rate = guess
    for _ in range(100):
        f, fp = npv(rate), d_npv(rate)
        if abs(fp) < 1e-12:
            break
        new_rate = rate - f / fp
        if new_rate <= -0.999:
            new_rate = (rate - 0.999) / 2
        if abs(new_rate - rate) < 1e-8:
            return new_rate
        rate = new_rate
    return rate if abs(npv(rate)) < 1e-3 else None


def simulate_sip(df: pd.DataFrame, monthly_amount: float, start: pd.Timestamp,
                 end: pd.Timestamp, sip_day: int = 1) -> dict | None:
    """Simulate a monthly SIP using actual historical NAVs."""
    daily = df["nav"].resample("D").ffill()
    daily = daily.loc[start:end]
    if daily.empty:
        return None
    dates = pd.date_range(start, end, freq="MS") + pd.Timedelta(days=sip_day - 1)
    dates = [d for d in dates if daily.index[0] <= d <= daily.index[-1]]
    if not dates:
        return None
    units, invested, rows = 0.0, 0.0, []
    for d in dates:
        nav = float(daily.asof(d))
        u = monthly_amount / nav
        units += u
        invested += monthly_amount
        rows.append({"date": d, "nav": nav, "units": units,
                     "invested": invested, "value": units * nav})
    final_nav = float(daily.iloc[-1])
    final_value = units * final_nav
    cf = [(d, -monthly_amount) for d in dates] + [(daily.index[-1], final_value)]
    rate = xirr(cf)
    ledger = pd.DataFrame(rows).set_index("date")
    ledger["value"] = ledger["units"] * daily.reindex(ledger.index).values
    # portfolio value over time (daily)
    unit_ts = pd.Series(0.0, index=daily.index)
    for d in dates:
        unit_ts.loc[d:] += monthly_amount / float(daily.asof(d))
    value_ts = unit_ts * daily
    invested_ts = pd.Series(0.0, index=daily.index)
    for d in dates:
        invested_ts.loc[d:] += monthly_amount
    return {
        "invested": invested,
        "final_value": final_value,
        "units": units,
        "xirr": rate * 100 if rate is not None else None,
        "value_ts": value_ts,
        "invested_ts": invested_ts,
        "n_installments": len(dates),
    }


def simulate_lumpsum(df: pd.DataFrame, amount: float, start: pd.Timestamp,
                     end: pd.Timestamp) -> dict | None:
    daily = df["nav"].resample("D").ffill().loc[start:end]
    if daily.empty or len(daily) < 2:
        return None
    buy_nav, final_nav = float(daily.iloc[0]), float(daily.iloc[-1])
    units = amount / buy_nav
    final_value = units * final_nav
    years = (daily.index[-1] - daily.index[0]).days / 365.25
    cagr = ((final_value / amount) ** (1 / years) - 1) * 100 if years > 0 else None
    return {
        "final_value": final_value,
        "abs_return": (final_value / amount - 1) * 100,
        "cagr": cagr,
        "value_ts": units * daily,
        "years": years,
    }


def fmt_inr(x: float) -> str:
    """Indian-style comma formatting (lakhs/crores)."""
    x = round(x)
    s = f"{x:,}"
    # convert western grouping to Indian grouping
    parts = str(int(x))
    if len(parts) > 3:
        last3 = parts[-3:]
        rest = parts[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        s = ",".join(groups + [last3])
    return f"₹{s}"


PLOTLY_LAYOUT = dict(
    template="plotly_white",
    hovermode="x unified",
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)

# ---------------------------------------------------------------------------
# Sidebar — fund selection
# ---------------------------------------------------------------------------
st.sidebar.title("📊 MF Analyzer")
st.sidebar.caption("Data: AMFI via mfapi.in · NAVs only, no advice")

query = st.sidebar.text_input(
    "Search a mutual fund",
    placeholder="e.g. Parag Parikh Flexi Cap Direct Growth",
)

selected_scheme = None
if query and len(query) >= 3:
    with st.sidebar:
        with st.spinner("Searching…"):
            results = search_schemes(query)
    if results:
        options = {f'{r["schemeName"]}  ·  #{r["schemeCode"]}': r["schemeCode"] for r in results[:60]}
        choice = st.sidebar.selectbox("Select scheme", list(options.keys()))
        selected_scheme = options[choice]
    else:
        st.sidebar.warning("No schemes found — try different keywords.")
else:
    st.sidebar.info("Type at least 3 characters to search. Tip: add 'Direct' and 'Growth' to narrow results.")

st.sidebar.divider()
compare_query = st.sidebar.text_input("Add funds to compare (optional)", placeholder="Search another fund…")
if "compare_list" not in st.session_state:
    st.session_state.compare_list = {}  # code -> name

if compare_query and len(compare_query) >= 3:
    cmp_results = search_schemes(compare_query)
    if cmp_results:
        cmp_opts = {f'{r["schemeName"]}': r["schemeCode"] for r in cmp_results[:40]}
        cmp_choice = st.sidebar.selectbox("Pick fund to add", list(cmp_opts.keys()), key="cmp_sel")
        if st.sidebar.button("➕ Add to comparison"):
            code = cmp_opts[cmp_choice]
            if len(st.session_state.compare_list) < 5:
                st.session_state.compare_list[code] = cmp_choice
            else:
                st.sidebar.warning("Max 5 funds in comparison.")

if st.session_state.compare_list:
    st.sidebar.write("**Comparison basket:**")
    for code, name in list(st.session_state.compare_list.items()):
        c1, c2 = st.sidebar.columns([5, 1])
        c1.caption(name[:55])
        if c2.button("✕", key=f"rm_{code}"):
            del st.session_state.compare_list[code]
            st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Fund Wala")
st.caption("Indian Mutual Fund Analysis Dashboard")

if not selected_scheme:
    st.markdown(
        """
        Search for any Indian mutual fund in the sidebar to get:

        - **Overview** — NAV chart, fund details, latest NAV
        - **Performance** — trailing returns, calendar-year returns, rolling returns
        - **Risk** — volatility, Sharpe/Sortino, drawdowns
        - **Calculators** — historical SIP (with XIRR) and lumpsum simulations
        - **Compare** — up to 5 funds side by side

        *Covers all AMFI-listed schemes: equity, debt, hybrid, index funds and more.*
        """
    )
    st.stop()

with st.spinner("Fetching NAV history…"):
    try:
        meta, nav_df = fetch_nav_history(selected_scheme)
    except Exception as e:
        st.error(f"Could not fetch data for this scheme: {e}")
        st.stop()

if nav_df.empty:
    st.error("No NAV data available for this scheme.")
    st.stop()

scheme_name = meta.get("scheme_name", "Selected fund")
latest_nav = float(nav_df["nav"].iloc[-1])
latest_date = nav_df.index[-1]
prev_nav = float(nav_df["nav"].iloc[-2]) if len(nav_df) > 1 else latest_nav
day_chg = (latest_nav / prev_nav - 1) * 100

st.subheader(scheme_name)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest NAV", f"₹{latest_nav:,.4f}", f"{day_chg:+.2f}%")
c2.metric("As of", latest_date.strftime("%d %b %Y"))
c3.metric("Fund House", meta.get("fund_house", "—"))
c4.metric("Category", meta.get("scheme_category", "—"))

tab_ov, tab_perf, tab_risk, tab_calc, tab_cmp = st.tabs(
    ["📈 Overview", "🏆 Performance", "⚠️ Risk", "🧮 Calculators", "⚖️ Compare"]
)

# ------------------------------ Overview ----------------------------------
with tab_ov:
    range_map = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10, "Max": None}
    sel_range = st.radio("Range", list(range_map.keys()), horizontal=True, index=2)
    yrs = range_map[sel_range]
    plot_df = nav_df if yrs is None else nav_df.loc[nav_df.index[-1] - pd.DateOffset(years=yrs):]

    log_scale = st.toggle("Log scale", value=False)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["nav"], mode="lines",
                             name="NAV", line=dict(width=2)))
    fig.update_layout(title="NAV history", yaxis_title="NAV (₹)", **PLOTLY_LAYOUT)
    if log_scale:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Scheme details"):
        st.json({
            "Scheme code": meta.get("scheme_code"),
            "Scheme type": meta.get("scheme_type"),
            "Category": meta.get("scheme_category"),
            "ISIN (Growth)": meta.get("isin_growth"),
            "ISIN (Div. reinvestment)": meta.get("isin_div_reinvestment"),
            "First NAV date": nav_df.index[0].strftime("%d %b %Y"),
            "Total NAV records": len(nav_df),
        })

# ----------------------------- Performance --------------------------------
with tab_perf:
    left, right = st.columns([1, 1.4])

    with left:
        st.markdown("#### Trailing returns")
        tr = trailing_returns(nav_df).dropna()
        tr_disp = tr.copy()
        tr_disp["Return (%)"] = tr_disp["Return (%)"].map(lambda v: f"{v:.2f}%")
        st.dataframe(tr_disp, hide_index=True, use_container_width=True)
        st.caption("Returns beyond 1 year are annualised (CAGR).")

    with right:
        st.markdown("#### Calendar-year returns")
        cy = calendar_year_returns(nav_df)
        if not cy.empty:
            fig = px.bar(cy, x="Year", y="Return (%)",
                         color=cy["Return (%)"] > 0,
                         color_discrete_map={True: "#2E8B57", False: "#C0392B"})
            fig.update_layout(showlegend=False, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough history for calendar-year returns.")

    st.markdown("#### Rolling returns")
    span_years = (nav_df.index[-1] - nav_df.index[0]).days / 365.25
    available = [y for y in (1, 3, 5) if span_years > y + 0.5]
    if available:
        roll_choice = st.radio("Window", [f"{y}Y" for y in available], horizontal=True)
        y = int(roll_choice[:-1])
        rr = rolling_returns(nav_df, y)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rr.index, y=rr.values, mode="lines", name=f"{y}Y rolling CAGR"))
        fig.add_hline(y=rr.mean(), line_dash="dash", line_color="grey",
                      annotation_text=f"Avg {rr.mean():.1f}%")
        fig.update_layout(title=f"{y}-year rolling returns (CAGR %)",
                          yaxis_title="CAGR (%)", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Average", f"{rr.mean():.2f}%")
        s2.metric("Median", f"{rr.median():.2f}%")
        s3.metric("Best window", f"{rr.max():.2f}%")
        s4.metric("Worst window", f"{rr.min():.2f}%")
        neg = (rr < 0).mean() * 100
        st.caption(f"Windows with negative returns: **{neg:.1f}%** of all {y}-year holding periods.")
    else:
        st.info("Fund is too young for rolling-return analysis.")

# -------------------------------- Risk ------------------------------------
with tab_risk:
    lookbacks = {"1Y": 1, "3Y": 3, "5Y": 5, "Max": None}
    lb = st.radio("Risk lookback", list(lookbacks.keys()), horizontal=True, index=1)
    y = lookbacks[lb]
    risk_df = nav_df if y is None else nav_df.loc[nav_df.index[-1] - pd.DateOffset(years=y):]

    m = risk_metrics(risk_df)
    if m:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Annualised return", f"{m['Annualised Return']:.2f}%")
        r2.metric("Volatility (ann.)", f"{m['Annualised Volatility']:.2f}%")
        r3.metric("Sharpe ratio", f"{m['Sharpe Ratio']:.2f}")
        r4.metric("Sortino ratio", f"{m['Sortino Ratio']:.2f}")
        r5, r6, r7, r8 = st.columns(4)
        r5.metric("Max drawdown", f"{m['Max Drawdown']:.2f}%")
        r6.metric("Positive days", f"{m['Positive Days (%)']:.1f}%")
        r7.metric("Best day", f"{m['Best Day (%)']:+.2f}%")
        r8.metric("Worst day", f"{m['Worst Day (%)']:+.2f}%")
        st.caption(f"Risk-free rate assumed: {RISK_FREE_RATE:.1%}. Sharpe/Sortino use daily returns annualised over {TRADING_DAYS} trading days.")

    dd = drawdown_series(risk_df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy",
                             mode="lines", line=dict(color="#C0392B"), name="Drawdown"))
    fig.update_layout(title="Drawdown from peak (%)", yaxis_title="Drawdown (%)", **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Distribution of daily returns")
    daily_ret = risk_df["nav"].pct_change().dropna() * 100
    fig = px.histogram(daily_ret, nbins=80)
    fig.update_layout(showlegend=False, xaxis_title="Daily return (%)",
                      yaxis_title="Days", **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------- Calculators --------------------------------
with tab_calc:
    mode = st.radio("Mode", ["SIP (historical)", "Lumpsum (historical)"], horizontal=True)
    min_d, max_d = nav_df.index[0].date(), nav_df.index[-1].date()

    if mode.startswith("SIP"):
        c1, c2, c3 = st.columns(3)
        amt = c1.number_input("Monthly SIP amount (₹)", 500, 10_00_000, 10_000, step=500)
        default_start = max(min_d, (pd.Timestamp(max_d) - pd.DateOffset(years=5)).date())
        start = c2.date_input("Start date", default_start, min_value=min_d, max_value=max_d)
        end = c3.date_input("End date", max_d, min_value=min_d, max_value=max_d)

        if start >= end:
            st.warning("Start date must be before end date.")
        else:
            res = simulate_sip(nav_df, amt, pd.Timestamp(start), pd.Timestamp(end))
            if res:
                gain = res["final_value"] - res["invested"]
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Invested", fmt_inr(res["invested"]))
                k2.metric("Final value", fmt_inr(res["final_value"]),
                          f"{gain / res['invested'] * 100:+.1f}%")
                k3.metric("Gain", fmt_inr(gain))
                k4.metric("XIRR", f"{res['xirr']:.2f}%" if res["xirr"] is not None else "—")
                st.caption(f"{res['n_installments']} monthly installments · {res['units']:.3f} units accumulated")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res["value_ts"].index, y=res["value_ts"].values,
                                         name="Portfolio value", mode="lines", line=dict(width=2)))
                fig.add_trace(go.Scatter(x=res["invested_ts"].index, y=res["invested_ts"].values,
                                         name="Amount invested", mode="lines",
                                         line=dict(dash="dash", color="grey")))
                fig.update_layout(title="SIP growth (actual NAV history)",
                                  yaxis_title="₹", **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough NAV history in the selected window.")
    else:
        c1, c2, c3 = st.columns(3)
        amt = c1.number_input("Lumpsum amount (₹)", 1000, 10_00_00_000, 1_00_000, step=1000)
        default_start = max(min_d, (pd.Timestamp(max_d) - pd.DateOffset(years=5)).date())
        start = c2.date_input("Invest date", default_start, min_value=min_d, max_value=max_d)
        end = c3.date_input("Valuation date", max_d, min_value=min_d, max_value=max_d)

        if start >= end:
            st.warning("Invest date must be before valuation date.")
        else:
            res = simulate_lumpsum(nav_df, amt, pd.Timestamp(start), pd.Timestamp(end))
            if res:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Invested", fmt_inr(amt))
                k2.metric("Final value", fmt_inr(res["final_value"]))
                k3.metric("Absolute return", f"{res['abs_return']:+.1f}%")
                k4.metric("CAGR", f"{res['cagr']:.2f}%" if res["cagr"] else "—")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res["value_ts"].index, y=res["value_ts"].values,
                                         mode="lines", name="Value", line=dict(width=2)))
                fig.add_hline(y=amt, line_dash="dash", line_color="grey",
                              annotation_text="Invested amount")
                fig.update_layout(title="Lumpsum investment growth",
                                  yaxis_title="₹", **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough NAV history in the selected window.")

# ------------------------------- Compare ----------------------------------
with tab_cmp:
    basket = dict(st.session_state.compare_list)
    basket[selected_scheme] = scheme_name  # always include the primary fund
    if len(basket) < 2:
        st.info("Add funds to the comparison basket from the sidebar (up to 5).")
    else:
        with st.spinner("Loading comparison data…"):
            series, metas = {}, {}
            for code, name in basket.items():
                try:
                    m, d = fetch_nav_history(code)
                    if not d.empty:
                        series[name] = d
                        metas[name] = m
                except Exception:
                    st.warning(f"Could not load: {name}")

        yrs_opts = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10, "Common max": None}
        sel = st.radio("Comparison window", list(yrs_opts.keys()), horizontal=True, index=1)
        y = yrs_opts[sel]

        # Common start
        latest_end = min(d.index[-1] for d in series.values())
        common_start = max(d.index[0] for d in series.values())
        if y is not None:
            common_start = max(common_start, latest_end - pd.DateOffset(years=y))

        fig = go.Figure()
        for name, d in series.items():
            daily = d["nav"].resample("D").ffill().loc[common_start:latest_end]
            if daily.empty:
                continue
            growth = daily / daily.iloc[0] * 100
            fig.add_trace(go.Scatter(x=growth.index, y=growth.values,
                                     mode="lines", name=name[:45]))
        fig.update_layout(title=f"Growth of ₹100 since {common_start.strftime('%d %b %Y')}",
                          yaxis_title="Value of ₹100", **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        # Comparison table
        rows = []
        for name, d in series.items():
            tr = trailing_returns(d).set_index("Period")["Return (%)"]
            rm = risk_metrics(d.loc[d.index[-1] - pd.DateOffset(years=3):])
            rows.append({
                "Fund": name[:50],
                "1Y (%)": tr.get("1 Year", np.nan),
                "3Y CAGR (%)": tr.get("3 Years", np.nan),
                "5Y CAGR (%)": tr.get("5 Years", np.nan),
                "Vol 3Y (%)": rm.get("Annualised Volatility", np.nan),
                "Sharpe 3Y": rm.get("Sharpe Ratio", np.nan),
                "Max DD 3Y (%)": rm.get("Max Drawdown", np.nan),
            })
        cmp_df = pd.DataFrame(rows).set_index("Fund").round(2)
        st.dataframe(cmp_df, use_container_width=True)
        st.caption("Risk columns computed over the last 3 years of each fund's history.")

st.divider()
st.caption(
    "⚠️ Educational tool only - NAV data from mfapi.in (AMFI). Past performance does not "
    "guarantee future returns. This is not investment advice; consult a SEBI-registered adviser."
)
