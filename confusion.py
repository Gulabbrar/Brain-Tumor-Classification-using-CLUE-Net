import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cm_df = pd.read_csv("test_confusion_matrix.csv", index_col=0)
labels = ["glioma", "meningioma", "pituitary", "no_tumor"]

fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=300)
sns.heatmap(
    cm_df.values,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=True,
    xticklabels=labels,
    yticklabels=labels,
    linewidths=0.8,
    linecolor="white",
    annot_kws={"size": 12},
    ax=ax,
)
ax.set_title("CLUE-Net (Ours): Confusion Matrix", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("Predicted Class", fontsize=11)
ax.set_ylabel("True Class", fontsize=11)
ax.tick_params(axis="both", labelsize=10.5)

plt.tight_layout()
plt.savefig("cluenet_confusion_matrix.png", dpi=300, bbox_inches="tight")