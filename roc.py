import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import roc_curve, auc

# Configure paths to your held-out test prediction CSVs
MODELS = [
    ("CustomCNN",      "Results/CustomCNN/customcnn_test_predictions.csv",           ["glioma", "meningioma", "pituitary", "no_tumor"], "#7f7f7f", "--"),
    ("MobileNetV2",    "Results/MobileNetV2/mobilenetv2_test_predictions.csv",       ["glioma", "meningioma", "pituitary", "no_tumor"], "#8c564b", "-."),
    ("DenseNet121",    "Results/DenseNet121/densenet121_test_predictions.csv",       ["glioma", "meningioma", "pituitary", "no_tumor"], "#1f77b4", "-."),
    ("EfficientNetB0", "Results/EfficientNetB0/efficientnetb0_test_predictions.csv", ["glioma", "meningioma", "pituitary", "no_tumor"], "#2ca02c", ":"),
    ("InceptionV3",    "Results/InceptionV3/inceptionv3_test_predictions.csv",       ["glioma", "meningioma", "pituitary", "no_tumor"], "#9467bd", "--"),
    ("ResNet50",       "Results/ResNet50/resnet50_test_predictions.csv",             ["glioma", "meningioma", "pituitary", "no_tumor"], "#ff7f0e", "-"),
    ("Xception",       "Results/Xception/xception_test_predictions.csv",             ["glioma", "meningioma", "pituitary", "no_tumor"], "#17becf", "-"),
    ("CLUE-Net (Ours)","Results/CLUE_Net/test_predictions.csv",                      ["glioma", "meningioma", "pituitary", "notumor"],  "#d62728", "-"),
]

def compute_macro_roc(csv_path, classes):
    df = pd.read_csv(csv_path)
    y_true = pd.get_dummies(df["true_label"])[classes].values
    y_prob = df[[f"prob_{c}" for c in classes]].values

    fpr_dict, tpr_dict = {}, {}
    for i, c in enumerate(classes):
        fpr_dict[c], tpr_dict[c], _ = roc_curve(y_true[:, i], y_prob[:, i])

    all_fpr = np.unique(np.concatenate([fpr_dict[c] for c in classes]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in classes:
        mean_tpr += np.interp(all_fpr, fpr_dict[c], tpr_dict[c])
    mean_tpr /= len(classes)
    return all_fpr, mean_tpr, auc(all_fpr, mean_tpr)

fig, ax = plt.subplots(figsize=(7.5, 5.8), dpi=300)
axins = inset_axes(ax, width="48%", height="46%", loc="lower right", borderpad=2.2)

for name, path, classes, color, ls in MODELS:
    if not os.path.exists(path):
        print(f"Skipping missing file: {path}")
        continue
    fpr, tpr, macro_auc = compute_macro_roc(path, classes)
    lw = 2.4 if "CLUE-Net" in name else 1.4
    zorder = 10 if "CLUE-Net" in name else 2
    label = f"{name} (AUC = {macro_auc:.4f})"

    ax.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw, label=label, zorder=zorder)
    axins.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw, zorder=zorder)

# Main plot styling
ax.plot([0, 1], [0, 1], color="#bbbbbb", linestyle=":", linewidth=1.0, label="Chance (AUC = 0.5000)")
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])
ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=11, fontweight="bold")
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
ax.set_title("Macro-Averaged One-vs-Rest ROC Comparison Across All Models", fontsize=12, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.35)
ax.legend(loc="center right", bbox_to_anchor=(0.99, 0.68), fontsize=8.2, frameon=True)

# Zoomed top-left inset styling
axins.set_xlim([0.0, 0.12])
axins.set_ylim([0.82, 1.002])
axins.set_title("Top-Left Region Zoom (FPR ≤ 0.12)", fontsize=8.5, fontweight="bold")
axins.grid(True, linestyle="--", alpha=0.35)
axins.tick_params(labelsize=7.5)

plt.tight_layout()
plt.savefig("all_models_macro_roc.png", dpi=300, bbox_inches="tight")
print("Saved all_models_macro_roc.png")