"""
Exploratory analysis of Banxico's daily Currency-in-Circulation series.

Produces the figures and stationarity diagnostics referenced in part (a) of
the write-up: level/trend, growth rate, seasonal decomposition (weekly and
annual), day-of-week pattern, and ACF/PACF + ADF/KPSS tests on the series
used for modeling.

Run: python src/eda.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGDIR = ROOT / "figures"
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_series() -> tuple[pd.Series, pd.Series]:
    cal = pd.read_csv(PROCESSED / "cic_calendar_daily.csv", index_col=0, parse_dates=True)["value"]
    biz = pd.read_csv(PROCESSED / "cic_business_daily.csv", index_col=0, parse_dates=True)["value"]
    cal.index.freq = "D"
    biz.index.freq = "B"
    return cal, biz


def fig_full_history(cal: pd.Series):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(cal.index, cal.values / 1e6, lw=1)
    axes[0].set_title("Currency in Circulation, Mexico (SF43702) — level, calendar-daily")
    axes[0].set_ylabel("Trillion MXN")

    log_s = np.log(cal)
    axes[1].plot(cal.index, log_s.values, lw=1, color="darkorange")
    axes[1].set_title("Log level")
    axes[1].set_ylabel("log(MXN millions)")
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    fig.savefig(FIGDIR / "01_full_history_level_and_log.png")
    plt.close(fig)


def fig_yoy_growth(cal: pd.Series):
    yoy = cal.pct_change(365) * 100
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(yoy.index, yoy.values, lw=1, color="firebrick")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Currency in Circulation — year-over-year growth rate")
    ax.set_ylabel("% YoY")
    fig.tight_layout()
    fig.savefig(FIGDIR / "02_yoy_growth.png")
    plt.close(fig)


def fig_seasonal_decompose(biz: pd.Series):
    log_biz = np.log(biz)

    # Weekly seasonality (period=5 business days) on log level.
    stl_weekly = STL(log_biz, period=5, robust=True).fit()
    fig = stl_weekly.plot()
    fig.set_size_inches(11, 7)
    fig.suptitle("STL decomposition — log(level), weekly period (m=5)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "03_stl_weekly.png")
    plt.close(fig)

    # Annual seasonality: aggregate to month-end level, decompose with period=12.
    monthly = biz.resample("ME").last()
    log_monthly = np.log(monthly)
    stl_annual = STL(log_monthly, period=12, robust=True).fit()
    fig2 = stl_annual.plot()
    fig2.set_size_inches(11, 7)
    fig2.suptitle("STL decomposition — month-end log(level), annual period (m=12)", y=1.02)
    fig2.tight_layout()
    fig2.savefig(FIGDIR / "04_stl_annual_monthly.png")
    plt.close(fig2)

    return stl_weekly, stl_annual, monthly


def fig_month_of_year_seasonality(biz: pd.Series):
    """Illustrate the December cash-demand spike directly."""
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    mom_growth = biz.resample("ME").last().pct_change() * 100
    df = mom_growth.to_frame("pct")
    df["month"] = pd.Categorical(df.index.month_name().str.slice(0, 3),
                                  categories=order, ordered=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    df.boxplot(column="pct", by="month", ax=ax, grid=False)
    ax.set_title("Month-over-month growth by calendar month (all years)")
    ax.set_xlabel("")
    ax.set_ylabel("% MoM (month-end level)")
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(FIGDIR / "05_month_of_year_boxplot.png")
    plt.close(fig)


def fig_day_of_week(biz: pd.Series):
    log_ret = np.log(biz).diff().dropna()
    df = log_ret.to_frame("ret")
    df["dow"] = df.index.day_name().str.slice(0, 3)
    order = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    means = df.groupby("dow")["ret"].mean().reindex(order) * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(order, means.values, color="steelblue")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Average daily log-return by day of week (business days)")
    ax.set_ylabel("% avg daily change")
    fig.tight_layout()
    fig.savefig(FIGDIR / "06_day_of_week_pattern.png")
    plt.close(fig)


def fig_acf_pacf(biz: pd.Series):
    log_diff = np.log(biz).diff().dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(log_diff, lags=40, ax=axes[0])
    axes[0].set_title("ACF — Δlog(level)")
    plot_pacf(log_diff, lags=40, ax=axes[1], method="ywm")
    axes[1].set_title("PACF — Δlog(level)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "07_acf_pacf_diff_log.png")
    plt.close(fig)
    return log_diff


def stationarity_tests(biz: pd.Series) -> pd.DataFrame:
    log_s = np.log(biz)
    log_diff = log_s.diff().dropna()

    rows = []
    for name, series in [("level", biz), ("log(level)", log_s), ("Δlog(level)", log_diff)]:
        adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
        kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags="auto")
        rows.append({
            "series": name,
            "ADF stat": adf_stat, "ADF p-value": adf_p,
            "KPSS stat": kpss_stat, "KPSS p-value": kpss_p,
        })
    table = pd.DataFrame(rows).set_index("series")
    table.to_csv(FIGDIR / "stationarity_tests.csv")
    return table


def main():
    cal, biz = load_series()
    fig_full_history(cal)
    fig_yoy_growth(cal)
    stl_weekly, stl_annual, monthly = fig_seasonal_decompose(biz)
    fig_month_of_year_seasonality(biz)
    fig_day_of_week(biz)
    fig_acf_pacf(biz)
    table = stationarity_tests(biz)

    print("Stationarity tests (level / log-level / diff-log-level):")
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nFigures saved to {FIGDIR}")


if __name__ == "__main__":
    main()
