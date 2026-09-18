"""
Load and clean Banco de Mexico's daily "Currency in Circulation" series.

Source: Banxico SIE, cuadro CF120 ("Base monetaria, circulante y depositos"),
series SF43702 ("Billetes y Monedas en Circulacion"). Raw export lives at
data/raw/CIC.xlsx.

Data treatment performed here (see README for rationale):
  1. Parse the SIE export's header block and extract the {date, value} rows.
  2. The raw series is calendar-daily: Banxico only revalues the balance on
     business days and carries it forward flat over weekends/holidays. We
     keep this raw calendar series for the full-history plot, but also
     produce a business-day ('B') version for modeling, since the
     forward-filled weekend points are not new information and would
     distort autocorrelation/seasonality estimates.
  3. Basic integrity checks: monotonic dates, no gaps in the calendar
     series, no missing values.
"""
from pathlib import Path

import pandas as pd

RAW_XLSX = Path(__file__).resolve().parents[1] / "data" / "raw" / "CIC.xlsx"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
SERIES_ID = "SF43702"
SERIES_LABEL = "Currency in Circulation (Billetes y Monedas en Circulacion)"


def load_raw(xlsx_path: Path = RAW_XLSX) -> pd.DataFrame:
    """Parse the Banxico SIE export into a clean {date, value} DataFrame."""
    raw = pd.read_excel(xlsx_path, header=None)

    # The SIE export has a fixed metadata block, then a header row
    # ("Date", <series id>) followed by the data rows. Locate it by content
    # rather than a hardcoded row number, in case Banxico's export changes.
    header_row = raw.index[raw[0].astype(str).str.strip() == "Date"][0]
    df = raw.iloc[header_row + 1 :, [0, 1]].copy()
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    return df


def to_calendar_series(df: pd.DataFrame) -> pd.Series:
    """Full calendar-daily series (millions of pesos), DatetimeIndex."""
    s = df.set_index("date")["value"]
    s.index = pd.DatetimeIndex(s.index, freq="D")
    return s.rename(SERIES_ID)


def to_business_day_series(calendar_series: pd.Series) -> pd.Series:
    """Business-day ('B') version used for modeling.

    Weekend rows are pure carry-forward duplicates of Friday's value in the
    raw export, so we simply drop Saturday/Sunday. Some Mexican public
    holidays that fall on a weekday will still show a flat carry-forward
    value (Banxico did not revalue that day) -- we leave those in place,
    since they are genuine "no new print" business days, not a data gap.
    """
    bday = calendar_series.asfreq("B")
    return bday.rename(calendar_series.name)


def validate(calendar_series: pd.Series) -> dict:
    """Integrity checks; returns a small report dict."""
    full_range = pd.date_range(calendar_series.index.min(), calendar_series.index.max(), freq="D")
    missing_days = full_range.difference(calendar_series.index)
    return {
        "start": calendar_series.index.min(),
        "end": calendar_series.index.max(),
        "n_obs": len(calendar_series),
        "n_missing_calendar_days": len(missing_days),
        "n_nulls": int(calendar_series.isna().sum()),
        "is_monotonic": bool(calendar_series.index.is_monotonic_increasing),
    }


def build_and_save() -> tuple[pd.Series, pd.Series]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    calendar_series = to_calendar_series(df)
    business_series = to_business_day_series(calendar_series)

    calendar_series.to_csv(PROCESSED_DIR / "cic_calendar_daily.csv", header=["value"])
    business_series.to_csv(PROCESSED_DIR / "cic_business_daily.csv", header=["value"])
    return calendar_series, business_series


if __name__ == "__main__":
    cal, biz = build_and_save()
    report = validate(cal)
    print("Calendar-daily series validation:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"\nBusiness-day series: {len(biz)} obs, "
          f"{biz.index.min().date()} to {biz.index.max().date()}")
    print(f"Saved to {PROCESSED_DIR}")
