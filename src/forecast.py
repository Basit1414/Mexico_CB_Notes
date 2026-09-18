"""
Refit the selected SARIMA model on the full history and forecast the next
20 business ("working") days.

Run: python src/forecast.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGDIR = ROOT / "figures"
HORIZON = 20


def _parse_order_file() -> tuple[tuple, tuple]:
    lines = (FIGDIR / "selected_sarima_order.txt").read_text().splitlines()
    order = eval(lines[0].split("=", 1)[1])
    seasonal_order = eval(lines[1].split("=", 1)[1])
    return order, seasonal_order


def main():
    biz = pd.read_csv(PROCESSED / "cic_business_daily.csv", index_col=0, parse_dates=True)["value"]
    biz.index.freq = "B"
    log_biz = np.log(biz)

    order, seasonal_order = _parse_order_file()
    print(f"Refitting SARIMA{order}x{seasonal_order} on full history "
          f"({biz.index.min().date()} to {biz.index.max().date()}, n={len(biz)})...")

    res = SARIMAX(
        log_biz, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)

    future_index = pd.bdate_range(biz.index.max() + pd.Timedelta(days=1), periods=HORIZON)
    fc = res.get_forecast(steps=HORIZON)
    fc_log = fc.predicted_mean
    ci_log = fc.conf_int(alpha=0.05)

    forecast_level = np.exp(fc_log)
    ci_level = np.exp(ci_log)
    forecast_level.index = future_index
    ci_level.index = future_index

    out = pd.DataFrame({
        "forecast": forecast_level.values,
        "lower_95": ci_level.iloc[:, 0].values,
        "upper_95": ci_level.iloc[:, 1].values,
    }, index=future_index)
    out.index.name = "date"
    out.to_csv(PROCESSED / "forecast_next_20bd.csv")

    print("\nForecast for the next 20 working days:")
    print(out.to_string(float_format=lambda x: f"{x:,.0f}"))

    day1 = out["forecast"].iloc[0]
    day20 = out["forecast"].iloc[-1]
    last_actual = biz.iloc[-1]
    print(f"\nLast actual ({biz.index[-1].date()}): {last_actual:,.0f} MXN millions")
    print(f"Forecast day 1  ({out.index[0].date()}): {day1:,.0f}  "
          f"({(day1/last_actual - 1)*100:+.2f}% vs last actual)")
    print(f"Forecast day 20 ({out.index[-1].date()}): {day20:,.0f}  "
          f"({(day20/last_actual - 1)*100:+.2f}% vs last actual)")

    # --- plot: last ~120 business days of history + forecast + CI ---
    hist_window = biz.iloc[-120:]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hist_window.index, hist_window.values, label="Actual", color="steelblue")
    ax.plot(out.index, out["forecast"], label="SARIMA forecast", color="firebrick",
             marker="o", ms=3)
    ax.fill_between(out.index, out["lower_95"], out["upper_95"], color="firebrick",
                     alpha=0.15, label="95% CI")
    ax.axvline(biz.index[-1], color="gray", ls="--", lw=0.8)
    ax.set_title(f"Currency in Circulation — 20-business-day forecast (SARIMA{order}x{seasonal_order})")
    ax.set_ylabel("MXN millions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "08_forecast_20bd.png")
    plt.close(fig)
    print(f"\nSaved forecast table to {PROCESSED / 'forecast_next_20bd.csv'}")
    print(f"Saved forecast plot to {FIGDIR / '08_forecast_20bd.png'}")


if __name__ == "__main__":
    main()
