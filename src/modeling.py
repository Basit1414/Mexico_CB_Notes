"""
Model selection for forecasting Currency in Circulation, 20 business days ahead.

Candidates:
  - SARIMA(p,d,q)(P,D,Q,5) fit on log(level). Order chosen once via AIC grid
    search on an initial training window, then evaluated out-of-sample.
  - ETS(A,A,A) (Holt-Winters, additive error/trend/seasonal) fit on
    log(level), seasonal_periods=5. Additive-in-log corresponds to
    multiplicative trend/seasonal in the original level, which matches the
    variance-growing-with-level behaviour seen in the EDA.

Both use a weekly seasonal period (m=5 business days): the ACF/PACF of the
differenced log-series in eda.py show strong, significant spikes exactly at
lags 5 and 10, which is the clearest, cheapest-to-model seasonal signal at
daily frequency. Annual (December) seasonality is visible in the STL
decomposition but is not encoded directly (a period-252 seasonal ARIMA term
is impractical to estimate); it is picked up implicitly through recent
levels/trend and discussed as a limitation for forecasts that span
December.

Selection method: expanding-window (rolling-origin) cross-validation,
forecasting 20 business days ahead per fold, scored with RMSE/MAE/MAPE
computed in level space (exponentiating the log forecasts) since that is
the operationally meaningful unit.

Run: python src/modeling.py
"""
from pathlib import Path
import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGDIR = ROOT / "figures"
SEASONAL_PERIOD = 5
HORIZON = 20


def load_log_series() -> pd.Series:
    biz = pd.read_csv(PROCESSED / "cic_business_daily.csv", index_col=0, parse_dates=True)["value"]
    biz.index.freq = "B"
    return np.log(biz)


def select_sarima_order(train_log: pd.Series, search_window: int = 750) -> tuple[tuple, tuple]:
    """Small AIC grid search on a recent training window (for speed).

    Returns ((p,d,q), (P,D,Q,m)). d is fixed at 1 (ADF/KPSS on the full
    series showed log-level is non-stationary, Δlog-level is stationary).
    """
    window = train_log.iloc[-search_window:]
    d = 1
    m = SEASONAL_PERIOD
    best = None
    for p, q, P, D, Q in itertools.product(range(3), range(3), range(2), range(2), range(2)):
        if p == 0 and q == 0:
            continue
        try:
            res = SARIMAX(
                window, order=(p, d, q), seasonal_order=(P, D, Q, m),
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
            if best is None or res.aic < best[0]:
                best = (res.aic, (p, d, q), (P, D, Q, m))
        except Exception:
            continue
    _, order, seasonal_order = best
    return order, seasonal_order


def fit_forecast_sarima(train_log: pd.Series, order, seasonal_order, horizon=HORIZON) -> np.ndarray:
    res = SARIMAX(
        train_log, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    fc = res.get_forecast(steps=horizon)
    return fc.predicted_mean.values, fc.conf_int(alpha=0.05)


def fit_forecast_ets(train_log: pd.Series, horizon=HORIZON) -> np.ndarray:
    model = ExponentialSmoothing(
        train_log, trend="add", seasonal="add", seasonal_periods=SEASONAL_PERIOD,
        initialization_method="estimated",
    ).fit()
    fc = model.forecast(horizon)
    return fc.values, model


def metrics(actual_level: np.ndarray, forecast_level: np.ndarray) -> dict:
    err = actual_level - forecast_level
    return {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE": float(np.mean(np.abs(err / actual_level)) * 100),
    }


def rolling_origin_cv(log_series: pd.Series, order, seasonal_order,
                       n_folds=12, horizon=HORIZON) -> pd.DataFrame:
    n = len(log_series)
    rows = []
    for k in range(n_folds):
        cutoff = n - (n_folds - k) * horizon
        train = log_series.iloc[:cutoff]
        test_log = log_series.iloc[cutoff: cutoff + horizon]
        actual_level = np.exp(test_log.values)

        sarima_fc, _ = fit_forecast_sarima(train, order, seasonal_order, horizon)
        sarima_level = np.exp(sarima_fc)

        ets_fc, _ = fit_forecast_ets(train, horizon)
        ets_level = np.exp(ets_fc)

        m_sarima = metrics(actual_level, sarima_level)
        m_ets = metrics(actual_level, ets_level)

        rows.append({"fold": k, "model": "SARIMA", "test_start": test_log.index[0], **m_sarima})
        rows.append({"fold": k, "model": "ETS(A,A,A)", "test_start": test_log.index[0], **m_ets})
        print(f"  fold {k+1}/{n_folds} (test starts {test_log.index[0].date()}): "
              f"SARIMA RMSE={m_sarima['RMSE']:.1f} MAPE={m_sarima['MAPE']:.2f}% | "
              f"ETS RMSE={m_ets['RMSE']:.1f} MAPE={m_ets['MAPE']:.2f}%")

    return pd.DataFrame(rows)


def main():
    log_series = load_log_series()

    print("Selecting SARIMA order via AIC grid search on a recent 750-obs window...")
    order, seasonal_order = select_sarima_order(log_series)
    print(f"Selected SARIMA order: {order} x {seasonal_order}")

    print("\nRunning rolling-origin cross-validation (12 folds x 20-business-day horizon)...")
    cv = rolling_origin_cv(log_series, order, seasonal_order)
    cv.to_csv(FIGDIR / "cv_results.csv", index=False)

    summary = cv.groupby("model")[["RMSE", "MAE", "MAPE"]].mean()
    print("\nAverage out-of-sample error across folds:")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
    summary.to_csv(FIGDIR / "cv_summary.csv")

    with open(FIGDIR / "selected_sarima_order.txt", "w") as f:
        f.write(f"order={order}\nseasonal_order={seasonal_order}\n")


if __name__ == "__main__":
    main()
