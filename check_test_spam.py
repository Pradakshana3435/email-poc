import pandas as pd
from src.predict import predict

df = pd.read_csv("data/test.csv").fillna("")
res = [predict(s, b) for s, b in zip(df["subject"], df["body"])]

print(pd.Series([r["action"] for r in res]).value_counts())
print("\nSpam-related:")
for s, r in zip(df["subject"], res):
    if r["action"] == "spam" or str(r.get("reason", "")).startswith("possible spam"):
        print(f"{r['action']:<14} {r['confidence']:.2f}  {s}")