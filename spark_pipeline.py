import pandas as pd
import os

df = pd.read_csv("data/visitor_data.csv")

visitor_total = (
    df.groupby("zone")["visitor_count"]
    .sum()
    .reset_index(name="total_visitors")
)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["minute_group"] = (df["timestamp"].dt.minute // 15) * 15

visitor_time = (
    df.groupby(
        ["zone", "hour", "minute_group"]
    )["visitor_count"]
    .mean()
    .reset_index(name="avg_visitors")
)

ml_dataset = df[["hour", "visitor_count"]]

os.makedirs("output/visitor_total", exist_ok=True)
os.makedirs("output/visitor_time", exist_ok=True)
os.makedirs("output/ml_visitor", exist_ok=True)

visitor_total.to_parquet(
    "output/visitor_total/data.parquet",
    index=False
)

visitor_time.to_parquet(
    "output/visitor_time/data.parquet",
    index=False
)

ml_dataset.to_parquet(
    "output/ml_visitor/data.parquet",
    index=False
)

print("PARQUET BERHASIL DISIMPAN")