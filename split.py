import os, json
import numpy as np, pandas as pd

df = pd.read_csv("dataset.csv")
feat = [c for c in df.columns if c != "label"]
labels = sorted(df["label"].unique())
label_map = {l: i for i, l in enumerate(labels)}
os.makedirs("splits", exist_ok=True)

tr, va, te = [], [], []
for l in labels:
    idx = np.where(df["label"].values == l)[0]   # giu thu tu chup
    n = len(idx); a, b = int(n * 0.70), int(n * 0.85)
    tr += list(idx[:a]); va += list(idx[a:b]); te += list(idx[b:])

for name, ids in [("train", tr), ("val", va), ("test", te)]:
    part = df.iloc[ids]
    np.save(f"splits/X_{name}.npy", part[feat].values.astype("float32"))
    np.save(f"splits/y_{name}.npy", part["label"].map(label_map).values)
    part.to_csv(f"splits/{name}.csv", index=False)
    print(name, len(part))

json.dump(label_map, open("splits/label_map.json", "w"), ensure_ascii=False, indent=2)
print(label_map)