#verify_numbers.py
import pandas as pd
from sklearn.metrics import f1_score

CATS = ["Technical Support", "Payment Issue", "Account & Login", "Sales Inquiry",
        "Complaint", "Refund Request", "Job Application", "General Inquiry"]
ROUTED = ["auto_route", "route_with_flag"]

def load(p):
    return pd.read_csv(p, keep_default_na=False)

t = load("reports/final_test_predictions.csv")
h = load("reports/final_handwritten_predictions.csv")

print("=== SANITY ===")
for name, df in [("test", t), ("handwritten", h)]:
    print(name, len(df), "rows")
    print(" action:", df["action"].value_counts().to_dict())
    print(" pred values:", sorted(df["pred"].unique()))
    print(" true labels:", sorted(df["category"].unique()))
    print(" pred_priority:", df["pred_priority"].value_counts().to_dict())
print(" handwritten split:", h["split"].value_counts().to_dict())

print("\n=== HELD-OUT TEST ===")
ok = t["pred"] == t["category"]
routed = t["action"].isin(ROUTED)
print("accuracy (manual review counts wrong):", round(ok.mean(), 3))
print("macro F1, 9 labels:", round(f1_score(t["category"], t["pred"], average="macro"), 3))
print("macro F1, 8 real cats:", round(f1_score(t["category"], t["pred"], labels=CATS, average="macro"), 3))
print("routed:", routed.sum(), "| routed correct:", ok[routed].sum(), round(ok[routed].mean(), 3))
for a in ROUTED:
    m = t["action"] == a
    print(f"  {a}: {m.sum()} rows, {round(ok[m].mean(), 3)} correct")
print("manual review:", (~routed).sum(), round((~routed).mean(), 3))
print("by source:", ok.groupby(t["source"]).mean().round(3).to_dict())
pok = t["pred_priority"] == t["priority"]
print("priority acc all:", round(pok.mean(), 3), "| routed only:", round(pok[routed].mean(), 3))

print("\n=== HAND-WRITTEN (split == test) ===")
x = h[h["split"] == "test"].copy()
print("rows:", len(x))
inc = x[x["category"].isin(CATS)]
strict = inc["pred"] == inc["category"]
lenient = strict | ((inc["alt_label"] != "") & (inc["pred"] == inc["alt_label"]))
print(f"in-category n={len(inc)} strict {strict.sum()} ({strict.mean():.3f}) lenient {lenient.sum()} ({lenient.mean():.3f})")
for ty, g in inc.groupby("type"):
    print(f"  {ty}: strict {strict[g.index].sum()}/{len(g)}  lenient {lenient[g.index].sum()}/{len(g)}")
unk = x[x["category"] == "Unknown"]
print("Unknown caught:", (unk["pred"] == "Unknown").sum(), "/", len(unk))
print("real emails sent to Unknown:", (inc["pred"] == "Unknown").sum(), "/", len(inc))
xr = x[x["action"].isin(ROUTED)]
xok = xr["pred"] == xr["category"]
print("auto/flag split of routed:")
for a in ROUTED:
    m = xr["action"] == a
    print(f"  {a}: {xok[m].sum()}/{m.sum()} correct")
print("overall 9-class acc:", round((x["pred"] == x["category"]).mean(), 3))
print("recall per category:")
print((inc["pred"] == inc["category"]).groupby(inc["category"]).mean().round(3).to_string())