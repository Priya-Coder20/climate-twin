"""Live data pipeline: Open-Meteo (free, no API key) -> data/live/latest.csv
Run manually, or schedule daily (cron / GitHub Actions / `python live_pipeline.py --loop`)."""
import requests, pandas as pd, time, sys, os
 
# Replace with your own district centroid file: columns district,lat,lon
DISTRICTS = pd.read_csv("data/maharashtra_districts.csv")   # district,lat,lon
VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum"
URL = "https://api.open-meteo.com/v1/forecast"
 
def fetch_batch(batch):
    p = dict(latitude=",".join(map(str, batch.lat)), longitude=",".join(map(str, batch.lon)),
             daily=VARS, past_days=30, forecast_days=7, timezone="Asia/Kolkata")
    r = requests.get(URL, params=p, timeout=60); r.raise_for_status()
    js = r.json(); js = js if isinstance(js, list) else [js]
    rows = []
    for (_, d), item in zip(batch.iterrows(), js):
        df = pd.DataFrame(item["daily"]).rename(columns={
            "time": "date", "temperature_2m_max": "tmax",
            "temperature_2m_min": "tmin", "precipitation_sum": "rain"})
        df["district"] = d.district
        rows.append(df)
    return pd.concat(rows)
 
def run():
    out = []
    for i in range(0, len(DISTRICTS), 10):          # 10 locations per call
        out.append(fetch_batch(DISTRICTS.iloc[i:i+10])); time.sleep(1)
    df = pd.concat(out)
    df["date"] = pd.to_datetime(df["date"])
    # basic quality checks
    df = df.dropna(subset=["tmax", "tmin"])
    df = df[(df.tmax.between(-5, 55)) & (df.tmin.between(-5, 45)) & (df.rain.fillna(0) >= 0)]
    df["rain"] = df["rain"].fillna(0)
    os.makedirs("data/live", exist_ok=True)
    df.to_csv("data/live/latest.csv", index=False)
    print("saved", len(df), "rows up to", df.date.max().date())
 
if __name__ == "__main__":
    if "--loop" in sys.argv:
        while True: run(); time.sleep(6*3600)
    else: run()