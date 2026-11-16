"""
Data retrieval for "Do Markets Overreact to Earnings?" (FODS project)

pip install yfinance pandas pandas_datareader requests pyarrow

Layout produced:
  data/universe.csv            point-in-time S&P 500 tickers (2015-2024)
  data/prices/<TICKER>.parquet daily OHLCV (split/dividend adjusted)
  data/earnings_yf.parquet     earnings dates + EPS estimate/actual (Yahoo)
  data/earnings_av.parquet     same from Alpha Vantage (cross-check / fill gaps)
  data/ff_daily.parquet        Fama-French 3 factors + RF, daily, in decimals
  data/iv/<TICKER>.csv         implied vol history from DoltHub (2019+)
"""
import os, time, json
import pandas as pd
import requests
import yfinance as yf

START, END = "2015-01-01", "2024-12-31"
os.makedirs("data/prices", exist_ok=True)
os.makedirs("data/iv", exist_ok=True)


# ---------------------------------------------------------------- 1. Universe
def build_universe(constituents_csv: str) -> list[str]:
    """
    Download the historical-components CSV manually from
    github.com/fja05680/sp500 (or hanshof/sp500_constituents).
    Format: columns `date`, `tickers` (comma-separated list per date).
    Use EVERY ticker that was a member at any point in the window,
    not today's list -- otherwise you have survivorship bias.
    """
    df = pd.read_csv(constituents_csv, parse_dates=["date"])
    df = df[(df["date"] >= START) & (df["date"] <= END)]
    tickers = sorted({t.strip() for row in df["tickers"] for t in row.split(",")})
    tickers = [t.replace(".", "-") for t in tickers]  # BRK.B -> BRK-B for Yahoo
    pd.Series(tickers, name="ticker").to_csv("data/universe.csv", index=False)
    return tickers


# ---------------------------------------------------------------- 2. Prices
def fetch_prices(tickers: list[str]) -> list[str]:
    """Returns tickers that failed (usually delisted / renamed -- log these)."""
    failed = []
    for t in tickers + ["^GSPC", "SPY"]:
        path = f"data/prices/{t}.parquet"
        if os.path.exists(path):
            continue
        df = yf.download(t, start=START, end=END, auto_adjust=True, progress=False)
        if df.empty:
            failed.append(t)
            continue
        df.to_parquet(path)
        time.sleep(0.3)  # be polite; Yahoo rate-limits aggressively
    return failed


# ---------------------------------------------------------------- 3a. Earnings (Yahoo)
def fetch_earnings_yf(tickers: list[str]) -> pd.DataFrame:
    """
    Ticker.get_earnings_dates gives: Earnings Date (with time), EPS Estimate,
    Reported EPS, Surprise(%). limit=60 ~ 15 years of quarters.
    The timestamp's hour tells you pre-market vs after-close -> needed to set t=0.
    """
    rows = []
    for t in tickers:
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=60)
        except Exception as e:
            print("yf earnings failed", t, e)
            continue
        if ed is None or ed.empty:
            continue
        ed = ed.reset_index().rename(columns={"Earnings Date": "ann_ts"})
        ed["ticker"] = t
        rows.append(ed)
        time.sleep(0.5)
    out = pd.concat(rows, ignore_index=True)
    out["ann_ts"] = pd.to_datetime(out["ann_ts"], utc=True).dt.tz_convert("America/New_York")
    # After 16:00 ET -> the market first reacts the NEXT trading day
    out["amc"] = out["ann_ts"].dt.hour >= 16
    out = out[(out["ann_ts"] >= START) & (out["ann_ts"] <= END)]
    out.to_parquet("data/earnings_yf.parquet")
    return out


# ---------------------------------------------------------------- 3b. Earnings (Alpha Vantage)
def fetch_earnings_av(tickers: list[str], api_key: str) -> pd.DataFrame:
    """
    function=EARNINGS -> quarterlyEarnings: fiscalDateEnding, reportedDate,
    reportedEPS, estimatedEPS, surprise, surprisePercentage, reportTime.
    Free key = 25 requests/day, so run this over ~3 weeks (it resumes from cache),
    or request their free educational access.
    """
    cache = "data/av_cache"
    os.makedirs(cache, exist_ok=True)
    calls = 0
    for t in tickers:
        f = f"{cache}/{t}.json"
        if os.path.exists(f):
            continue
        if calls >= 25:
            print("Daily quota used; re-run tomorrow.")
            break
        r = requests.get("https://www.alphavantage.co/query",
                         params={"function": "EARNINGS", "symbol": t, "apikey": api_key},
                         timeout=30).json()
        calls += 1
        if "quarterlyEarnings" not in r:  # throttle msg comes back as HTTP 200
            print("AV issue", t, list(r)[:1])
            break
        json.dump(r, open(f, "w"))
        time.sleep(13)  # 5 calls/min
    rows = []
    for f in os.listdir(cache):
        r = json.load(open(f"{cache}/{f}"))
        q = pd.DataFrame(r["quarterlyEarnings"])
        q["ticker"] = r.get("symbol", f[:-5])
        rows.append(q)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        for c in ["reportedEPS", "estimatedEPS", "surprise", "surprisePercentage"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out.to_parquet("data/earnings_av.parquet")
    return out


# ---------------------------------------------------------------- 4. Fama-French
def fetch_ff() -> pd.DataFrame:
    from pandas_datareader import data as pdr
    ff = pdr.DataReader("F-F_Research_Data_Factors_daily", "famafrench", START, END)[0] / 100
    ff.to_parquet("data/ff_daily.parquet")  # Mkt-RF, SMB, HML, RF
    return ff


# ---------------------------------------------------------------- 5. Implied vol (DoltHub)
DOLT_API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"

def dolt_query(sql: str) -> pd.DataFrame:
    r = requests.get(DOLT_API, params={"q": sql}, timeout=120)
    r.raise_for_status()
    body = r.json()
    if body.get("query_execution_status") != "Success":
        raise RuntimeError(body.get("query_execution_message"))
    return pd.DataFrame(body["rows"])

def inspect_iv_schema():
    """Run this FIRST: confirm column names and how dense the dates are."""
    print(dolt_query("SELECT * FROM volatility_history WHERE act_symbol='AAPL' "
                     "ORDER BY date LIMIT 10"))

def fetch_iv(ticker: str, year: int):
    """Small per-symbol, per-year pulls only -- big queries time out on the API.
    For all ~500 names, `dolt clone` the repo instead (see notes)."""
    sym = ticker.replace("-", ".")
    df = dolt_query(f"SELECT * FROM volatility_history WHERE act_symbol='{sym}' "
                    f"AND date BETWEEN '{year}-01-01' AND '{year}-12-31' ORDER BY date")
    df.to_csv(f"data/iv/{ticker}_{year}.csv", index=False)
    return df


if __name__ == "__main__":
    tickers = build_universe("sp500_historical_components.csv")
    print(len(tickers), "tickers in point-in-time universe")
    print("price failures:", fetch_prices(tickers))
    fetch_earnings_yf(tickers)
    fetch_ff()
    # fetch_earnings_av(tickers, api_key=os.environ["AV_KEY"])
    # inspect_iv_schema()
