import pandas as pd
import random
from datetime import datetime, timedelta
import os

os.makedirs("data", exist_ok=True)

zones = ["FoodCourt", "FashionArea", "Cinema"]

start_time = datetime.now()

data = []

for minute in range(180):
    current_time = start_time + timedelta(minutes=minute)

    for zone in zones:
        visitors = random.randint(10, 500)

        data.append([
            current_time.strftime("%Y-%m-%d %H:%M:%S"),
            zone,
            visitors
        ])

df = pd.DataFrame(
    data,
    columns=[
        "timestamp",
        "zone",
        "visitor_count"
    ]
)

df.to_csv(
    "data/visitor_data.csv",
    index=False
)

print("="*50)
print("DATA BERHASIL DIBUAT")
print("="*50)
print(df.head())
print("\nJumlah Data :", len(df))