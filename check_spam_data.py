import pandas as pd

def rd(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False)

def norm(d):
    return (d.subject + " " + d.body).str.lower().str.split().str.join(" ")

spam = rd("data/spam/spam.csv")
hard = rd("data/spam/hard_ham.csv")
ham = pd.concat([rd(f"data/{n}.csv") for n in ("train", "val", "test")])

for name, d in [("spam", spam), ("hard_ham", hard)]:
    miss = {"subject", "body"} - set(d.columns)
    print(f"\n== {name}: {len(d)} rows, columns {list(d.columns)}")
    if miss:
        print("   MISSING COLUMNS:", miss)
        continue
    n = norm(d)
    print("   empty bodies:", (d.body.str.strip() == "").sum())
    print("   duplicates inside file:", n.duplicated().sum())
    print("   exact copies of normal emails:", n.isin(set(norm(ham))).sum())

print("\n== style check (a big gap means the model may learn style, not spam)")
print(f"{'':10}{'words(med)':>12}{'has url':>10}{'ALL CAPS':>10}{'has !':>8}{'messy':>8}")
for name, d in [("spam", spam), ("hard_ham", hard), ("normal", ham)]:
    t = d.subject + " " + d.body
    w = t.str.split().str.len()
    url = t.str.contains(r"https?://|www\.", regex=True).mean()
    caps = t.apply(lambda s: sum(c.isupper() for c in s) / max(1, sum(c.isalpha() for c in s)) > 0.3).mean()
    bang = t.str.contains("!").mean()
    messy = t.str.contains(r"\b(pls|plz|u r|kindly|asap)\b", case=False, regex=True).mean()
    print(f"{name:10}{w.median():>12.0f}{url:>10.0%}{caps:>10.0%}{bang:>8.0%}{messy:>8.0%}")