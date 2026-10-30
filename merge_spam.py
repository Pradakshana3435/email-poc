import pandas as pd, glob

def merge(target, pattern):
    parts = [pd.read_csv(target, dtype=str, keep_default_na=False)]
    for f in sorted(glob.glob(pattern)):
        d = pd.read_csv(f, dtype=str, keep_default_na=False)
        print("  adding", f, len(d), "rows", list(d.columns))
        parts.append(d[["subject", "body"]])
    d = pd.concat(parts, ignore_index=True)
    key = (d.subject + " " + d.body).str.lower().str.split().str.join(" ")
    d = d[~key.duplicated()]
    d.to_csv(target, index=False, encoding="utf-8")
    print(target, "->", len(d), "rows")

merge("data/spam/spam.csv", "data/spam/new_spam_*.csv")
merge("data/spam/hard_ham.csv", "data/spam/new_ham_*.csv")