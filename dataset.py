import pandas as pd

def load(p): return pd.read_csv(p, keep_default_na=False)

tr, va, te = load("data/train.csv"), load("data/val.csv"), load("data/test.csv")
tr["split"], va["split"], te["split"] = "train", "val", "test"
al = pd.concat([tr, va, te], ignore_index=True)
fin = load("data/emails_final.csv")

print("rows: train", len(tr), "val", len(va), "test", len(te), "| total", len(al), "| emails_final", len(fin))
print("\ncategory x split"); print(pd.crosstab(al["category"], al["split"], margins=True))
print("\nsource x split"); print(pd.crosstab(al["source"], al["split"], margins=True))
print("\ncategory x source"); print(pd.crosstab(al["category"], al["source"], margins=True))
print("\npriority x split"); print(pd.crosstab(al["priority"], al["split"], normalize="columns").round(3))
print("\nwords per email:"); print(al["words"].describe().round(1))
print("\nsynthetic batches:"); print(al[al["source"] == "synthetic"].groupby("batch").size().to_dict())

key = lambda d: (d["subject"].str.strip().str.lower() + "||" + d["body"].str.strip().str.lower())
print("\nexact duplicates inside all data:", key(al).duplicated().sum())
print("train/test exact overlap:", len(set(key(tr)) & set(key(te))))
print("train/val exact overlap:", len(set(key(tr)) & set(key(va))))

h = load("data/test_handwritten.csv")
print("\nhand-written file:", len(h), "rows"); print(pd.crosstab(h["split"], h["category"]))
print(pd.crosstab(h[h["category"] != "Unknown"]["type"], h[h["category"] != "Unknown"]["split"]))
hk = set(key(h)); print("hand-written overlap with train+val+test:", len(hk & set(key(al))))