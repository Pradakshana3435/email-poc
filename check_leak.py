import pandas as pd

for name in ["emails_final", "train", "val", "test"]:
    d = pd.read_csv(f"data/{name}.csv", keep_default_na=False)
    text = d["subject"] + " " + d["body"]
    leak = text.str.contains("Tech Online Store", case=False)
    ph = text.str.contains(r"<[a-z_]+>", regex=True)
    print(f"{name:13s} rows {len(d):4d} | 'Tech Online Store': {leak.sum():3d} | unfilled placeholders: {ph.sum():3d}")
    print("   by source:", d.loc[leak, "source"].value_counts().to_dict(), "|", d.loc[ph, "source"].value_counts().to_dict())