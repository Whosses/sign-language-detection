import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

os.makedirs("eda_output", exist_ok=True)
df = pd.read_csv("dataset.csv")
feat = [c for c in df.columns if c != "label"]
use = [c for c in feat if df[c].std() > 1e-9]   # bo cot hang (co tay = 0)
labels = sorted(df["label"].unique())

print("Kich thuoc:", df.shape, "| thieu:", int(df.isna().sum().sum()))
print(df["label"].value_counts().sort_index())
df[use].describe().T.to_csv("eda_output/describe.csv")

# 1. so mau moi lop
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x="label", order=labels)
plt.title("So mau moi lop"); plt.tight_layout()
plt.savefig("eda_output/01_countplot.png", dpi=150); plt.close()

# 2. heatmap tuong quan
plt.figure(figsize=(11, 9))
sns.heatmap(df[use].corr(), cmap="coolwarm", center=0)
plt.title("Tuong quan giua cac feature"); plt.tight_layout()
plt.savefig("eda_output/02_heatmap.png", dpi=150); plt.close()

# 3. PCA 2D
Z = PCA(n_components=2).fit_transform(df[use])
plt.figure(figsize=(7, 6))
for l in labels:
    m = (df["label"] == l).values
    plt.scatter(Z[m, 0], Z[m, 1], label=l, s=25)
plt.legend(); plt.title("PCA 2D"); plt.tight_layout()
plt.savefig("eda_output/03_pca.png", dpi=150); plt.close()

# 4. t-SNE 2D
perp = max(5, min(30, len(df) // 4))
T = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(df[use])
plt.figure(figsize=(7, 6))
for l in labels:
    m = (df["label"] == l).values
    plt.scatter(T[m, 0], T[m, 1], label=l, s=25)
plt.legend(); plt.title("t-SNE 2D"); plt.tight_layout()
plt.savefig("eda_output/04_tsne.png", dpi=150); plt.close()

# 5. ve skeleton ban tay (1 mau/lop)
edges = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),
         (9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
fig, axes = plt.subplots(1, len(labels), figsize=(3 * len(labels), 3.2))
axes = np.atleast_1d(axes)
for ax, l in zip(axes, labels):
    p = df[df["label"] == l].iloc[0][feat].values.reshape(21, 3)
    for a, b in edges:
        ax.plot([p[a, 0], p[b, 0]], [-p[a, 1], -p[b, 1]], "b-")
    ax.scatter(p[:, 0], -p[:, 1], c="r", s=10)
    ax.set_title(l); ax.set_aspect("equal"); ax.axis("off")
plt.tight_layout(); plt.savefig("eda_output/05_skeleton.png", dpi=150); plt.close()
print("Xong. Mo thu muc eda_output/ de xem anh.")