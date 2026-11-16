import pandas as pd, numpy as np

df = pd.read_csv("dataset_raw.csv")
feat = [c for c in df.columns if c != "label"]
n0 = len(df)

# 1. Bỏ dòng thiếu dữ liệu
df = df.dropna()
n_nan = n0 - len(df)

# 2. Bỏ dòng trùng lặp
b = len(df)
df = df.drop_duplicates()
n_dup = b - len(df)

# 3. Chuẩn hóa: lấy cổ tay (landmark 1) làm gốc, chia cho khoảng cách xa nhất
X = df[feat].values.reshape(-1, 21, 3)
X = X - X[:, [0], :]
scale = np.linalg.norm(X[:, :, :2], axis=2).max(axis=1)
mask = scale > 1e-6
n_bad = int((~mask).sum())
X, df, scale = X[mask], df[mask], scale[mask]
X = X / scale[:, None, None]

out = pd.DataFrame(X.reshape(len(X), 63), columns=feat)
out["label"] = df["label"].values
out.to_csv("dataset.csv", index=False)

print(f"Ban dau: {n0} | thieu: {n_nan} | trung: {n_dup} | hong: {n_bad} | con lai: {len(out)}")
print(out["label"].value_counts().sort_index())