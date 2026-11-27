

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import hashlib
import json
import math
import random
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve, auc, log_loss
)
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import binomtest

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE


def canonical_class(folder_name):
    """Map common folder-name variants to the four target labels."""
    s = re.sub(r"[^a-z]", "", folder_name.lower())
    if "glioma" in s:
        return "glioma"
    if "meningioma" in s:
        return "meningioma"
    if "pituitary" in s:
        return "pituitary"
    if any(x in s for x in ("notumor", "normal", "healthy", "no_tumor")):
        return "no_tumor"
    return None


def find_image_records(root):
    """Find images and infer labels from class folders; handles split folders."""
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    records = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        label = None
        class_folder = None
        for parent in p.parents:
            label = canonical_class(parent.name)
            if label:
                class_folder = parent
                break
            if parent == root.parent:
                break
        if label is None:
            continue

        rel_parts = p.relative_to(root).parts
        split = None
        for part in rel_parts[:-1]:
            part_l = part.lower()
            if part_l in {"train", "training"}:
                split = "train"
            elif part_l in {"val", "valid", "validation", "dev"}:
                split = "val"
            elif part_l in {"test", "testing"}:
                split = "test"

        # A possible domain/site label is a folder between the split root and
        # class directory, or between dataset root and class directory.
        domain = "unknown"
        try:
            idx = rel_parts.index(class_folder.name)
            if idx > 0:
                candidate = rel_parts[idx - 1]
                if candidate.lower() not in {
                    "train", "training", "val", "valid", "validation",
                    "dev", "test", "testing"
                }:
                    domain = candidate
        except ValueError:
            pass

        records.append({
            "path": str(p), "label": label, "split": split,
            "domain": domain, "filename": p.name
        })

    df = pd.DataFrame(records).drop_duplicates(subset=["path"])
    if df.empty:
        raise ValueError(
            "No labeled images found. Ensure images are inside class folders "
            "named with glioma, meningioma, pituitary, or no_tumor/normal."
        )
    found = set(df.label.unique())
    missing = set(CLASS_NAMES) - found
    if missing:
        raise ValueError(f"Could not find images for classes: {sorted(missing)}")
    return df


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def perform_eda(df, out_dir):
    out_dir = Path(out_dir)
    eda_dir = out_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(eda_dir / "image_manifest.csv", index=False)

    counts = df["label"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    counts.rename_axis("class").reset_index(name="image_count").to_csv(
        eda_dir / "class_counts.csv", index=False
    )
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                     palette="Set2", legend=False)
    ax.set_title("Image Count by Class", weight="bold")
    ax.set_xlabel("Tumor Class")
    ax.set_ylabel("Number of Images")
    ax.tick_params(axis="x", rotation=15)
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    savefig(eda_dir / "class_distribution.png")

    # File-size, dimensions, image mode, and corrupt-file checks.
    image_info = []
    corrupt = []
    for row in df.itertuples(index=False):
        try:
            with Image.open(row.path) as im:
                w, h = im.size
                mode = im.mode
                image_info.append({
                    "path": row.path, "label": row.label,
                    "width": w, "height": h, "aspect_ratio": w / max(h, 1),
                    "mode": mode, "file_size_kb": os.path.getsize(row.path) / 1024
                })
        except Exception as e:
            corrupt.append({"path": row.path, "error": str(e)})
    info = pd.DataFrame(image_info)
    info.to_csv(eda_dir / "image_properties.csv", index=False)
    pd.DataFrame(corrupt, columns=["path", "error"]).to_csv(
        eda_dir / "corrupt_images.csv", index=False
    )

    if not info.empty:
        for col, title, filename in [
            ("width", "Image Width Distribution", "width_distribution.png"),
            ("height", "Image Height Distribution", "height_distribution.png"),
            ("file_size_kb", "Image File Size Distribution (KB)", "filesize_distribution.png"),
            ("aspect_ratio", "Image Aspect-Ratio Distribution", "aspect_ratio_distribution.png")
        ]:
            plt.figure(figsize=(8, 5))
            sns.histplot(data=info, x=col, hue="label", bins=30,
                         element="step", stat="count", common_norm=False)
            plt.title(title, weight="bold")
            plt.xlabel(col.replace("_", " ").title())
            plt.ylabel("Image Count")
            savefig(eda_dir / filename)

    # Contact sheet: examples from each class.
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    for r, cls in enumerate(CLASS_NAMES):
        subset = df[df.label == cls].sample(
            min(5, (df.label == cls).sum()), random_state=SEED
        )
        for c in range(5):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(subset):
                row = subset.iloc[c]
                try:
                    im = Image.open(row.path).convert("RGB")
                    im.thumbnail((180, 180))
                    ax.imshow(im)
                    ax.set_title(cls.replace("_", " ").title(), fontsize=8)
                except Exception:
                    ax.set_title("Unreadable", fontsize=8)
    fig.suptitle("Representative MRI Images by Class", fontsize=14, weight="bold")
    savefig(eda_dir / "sample_images.png")

    # Exact byte-hash duplicates, reported for auditing (not silently deleted).
    hash_map = {}
    for row in df.itertuples(index=False):
        try:
            h = hashlib.md5(Path(row.path).read_bytes()).hexdigest()
            hash_map.setdefault(h, []).append(row.path)
        except Exception:
            pass
    duplicates = [{"md5": h, "paths": " | ".join(paths), "count": len(paths)}
                  for h, paths in hash_map.items() if len(paths) > 1]
    pd.DataFrame(duplicates, columns=["md5", "paths", "count"]).to_csv(
        eda_dir / "exact_duplicate_audit.csv", index=False
    )

    summary = {
        "total_images_found": int(len(df)),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "corrupt_images": int(len(corrupt)),
        "exact_duplicate_groups": int(len(duplicates)),
        "image_width_min": int(info.width.min()) if len(info) else None,
        "image_width_median": float(info.width.median()) if len(info) else None,
        "image_height_min": int(info.height.min()) if len(info) else None,
        "image_height_median": float(info.height.median()) if len(info) else None,
        "image_modes": info["mode"].value_counts().to_dict() if len(info) else {}
    }
    (eda_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print("EDA summary:", json.dumps(summary, indent=2))
    return df[~df.path.isin([item["path"] for item in corrupt])].copy()


def split_data(df, out_dir):
    """Preserve an existing split; otherwise create stratified 70/15/15 split."""
    df = df.copy()
    split_values = set(df.split.dropna().unique())
    if {"train", "val", "test"}.issubset(split_values):
        train = df[df.split == "train"].copy()
        val = df[df.split == "val"].copy()
        test = df[df.split == "test"].copy()
        print("Using the dataset's existing train/validation/test split.")
    else:
        # Avoid accidental mixed partial split structures.
        if split_values:
            print("Some split folder labels found but not all three; rebuilding a "
                  "stratified split across discovered images.")
        train, temp = train_test_split(
            df, test_size=0.30, random_state=SEED, stratify=df.label
        )
        val, test = train_test_split(
            temp, test_size=0.50, random_state=SEED, stratify=temp.label
        )
        train, val, test = train.copy(), val.copy(), test.copy()
        train["split"], val["split"], test["split"] = "train", "val", "test"

    # Exact duplicate hashes crossing splits are leakage warnings.
    split_frames = {"train": train, "val": val, "test": test}
    hashes = {}
    for split_name, part in split_frames.items():
        for path in part.path:
            try:
                h = hashlib.md5(Path(path).read_bytes()).hexdigest()
                hashes.setdefault(h, {}).setdefault(split_name, []).append(path)
            except Exception:
                pass
    leaks = []
    for h, split_dict in hashes.items():
        if len(split_dict) > 1:
            leaks.append({"md5": h, "splits": ",".join(sorted(split_dict)),
                          "paths": " | ".join(p for ps in split_dict.values() for p in ps)})
    leak_path = Path(out_dir) / "eda" / "cross_split_exact_duplicate_audit.csv"
    pd.DataFrame(leaks, columns=["md5", "splits", "paths"]).to_csv(leak_path, index=False)
    if leaks:
        print(f"WARNING: {len(leaks)} exact duplicate hash(es) cross split boundaries. "
              f"Review {leak_path}; results may be optimistic.")
    else:
        print("No cross-split exact byte duplicates detected (near-duplicates are not ruled out).")

    for name, part in split_frames.items():
        part.to_csv(Path(out_dir) / f"{name}_manifest.csv", index=False)
    split_counts = pd.concat([
        train.assign(split="train"), val.assign(split="val"), test.assign(split="test")
    ]).groupby(["split", "label"]).size().unstack(fill_value=0).reindex(
        columns=CLASS_NAMES, fill_value=0
    )
    split_counts.to_csv(Path(out_dir) / "split_class_counts.csv")
    ax = split_counts.plot(kind="bar", figsize=(10, 6), colormap="Set2")
    ax.set_title("Class Distribution Across Data Splits", weight="bold")
    ax.set_xlabel("Data Split")
    ax.set_ylabel("Number of Images")
    ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(Path(out_dir) / "eda" / "split_class_distribution.png")
    return train, val, test


def make_dataset(frame, training=False, batch_size=BATCH_SIZE):
    paths = frame.path.astype(str).values
    labels = frame.label.map({c: i for i, c in enumerate(CLASS_NAMES)}).values.astype("int32")
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path, label):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], antialias=True)
        image = tf.cast(image, tf.float32)  # keep [0,255]; EfficientNet expects this.
        return image, label

    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(min(len(frame), 3000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def augmentation_block(x):
    # Conservative, label-preserving augmentation for MRI images.
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.04)(x)
    x = layers.RandomZoom(0.08)(x)
    x = layers.RandomContrast(0.10)(x)
    return x


def build_custom_cnn(num_classes=4):
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="mri_image")
    x = augmentation_block(inp)
    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.15)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    out = layers.Dense(num_classes, activation="softmax", name="class_output")(x)
    return keras.Model(inp, out, name="CustomCNN")


def build_transfer_model(name, num_classes=4, pretrained=True):
    app_map = {
        "VGG16": keras.applications.VGG16,
        "ResNet50": keras.applications.ResNet50,
        "EfficientNetB0": keras.applications.EfficientNetB0,
        "DenseNet121": keras.applications.DenseNet121,
        "MobileNetV2": keras.applications.MobileNetV2,
        "InceptionV3": keras.applications.InceptionV3,
        "Xception": keras.applications.Xception,
    }
    constructor = app_map[name]
    base = constructor(include_top=False, weights="imagenet" if pretrained else None,
                       input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling=None)
    base.trainable = False
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="mri_image")
    x = augmentation_block(inp)
    # Application-specific preprocessing must match the backbone.
    if name == "VGG16":
        x = layers.Lambda(keras.applications.vgg16.preprocess_input, name="vgg_preprocess")(x)
    elif name == "ResNet50":
        x = layers.Lambda(keras.applications.resnet.preprocess_input, name="resnet_preprocess")(x)
    elif name == "DenseNet121":
        x = layers.Lambda(keras.applications.densenet.preprocess_input, name="densenet_preprocess")(x)
    elif name == "MobileNetV2":
        x = layers.Lambda(keras.applications.mobilenet_v2.preprocess_input, name="mobilenet_preprocess")(x)
    elif name == "InceptionV3":
        x = layers.Lambda(keras.applications.inception_v3.preprocess_input, name="inception_preprocess")(x)
    elif name == "Xception":
        x = layers.Lambda(keras.applications.xception.preprocess_input, name="xception_preprocess")(x)
    # EfficientNet includes rescaling internally and expects [0,255].
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    out = layers.Dense(num_classes, activation="softmax", name="class_output")(x)
    model = keras.Model(inp, out, name=name)
    model.backbone = base
    return model


@tf.custom_gradient
def reverse_gradient(x, coefficient):
    def grad(dy):
        return -coefficient * dy, tf.zeros_like(coefficient)
    return x, grad


def build_dann(num_classes=4, num_domains=2):
    """Domain-Adversarial Neural Network; requires real site/scanner/domain labels."""
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="mri_image")
    alpha = keras.Input(shape=(), dtype=tf.float32, name="grl_coefficient")
    base = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg"
    )
    base.trainable = False
    features = base(inp, training=False)
    features = layers.Dense(256, activation="relu")(features)
    features = layers.Dropout(0.3)(features)
    class_out = layers.Dense(num_classes, activation="softmax", name="class_output")(features)
    reversed_features = layers.Lambda(
        lambda z: reverse_gradient(z[0], z[1]), name="gradient_reversal"
    )([features, alpha])
    d = layers.Dense(128, activation="relu")(reversed_features)
    d = layers.Dropout(0.3)(d)
    domain_out = layers.Dense(num_domains, activation="softmax", name="domain_output")(d)
    model = keras.Model([inp, alpha], [class_out, domain_out], name="DANN")
    model.backbone = base
    return model


def get_class_weights(train_df):
    y = train_df.label.map({c: i for i, c in enumerate(CLASS_NAMES)}).values
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(CLASS_NAMES)), y=y)
    return {int(i): float(w) for i, w in enumerate(weights)}


def plot_training(history, model_name, out_dir):
    hist = history.history
    for metric, title in [("loss", "Loss"), ("accuracy", "Accuracy")]:
        train_key = metric if metric in hist else "class_output_" + metric
        val_key = "val_" + train_key
        if train_key not in hist:
            candidates = [k for k in hist if k.endswith(metric) and not k.startswith("val_")]
            if not candidates:
                continue
            train_key = candidates[0]
            val_key = "val_" + train_key
        plt.figure(figsize=(8, 5))
        plt.plot(hist[train_key], label="Training")
        if val_key in hist:
            plt.plot(hist[val_key], label="Validation")
        plt.title(f"{model_name}: {title} by Epoch", weight="bold")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.legend()
        savefig(Path(out_dir) / f"{model_name.lower()}_{metric}_curve.png")


def expected_calibration_error(y_true, probs, bins=15):
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    bin_data = []
    for i in range(bins):
        mask = (confidence > edges[i]) & (confidence <= edges[i + 1])
        if i == 0:
            mask |= confidence == 0
        if mask.any():
            acc = correct[mask].mean()
            conf = confidence[mask].mean()
            ece += mask.mean() * abs(acc - conf)
            bin_data.append((conf, acc, int(mask.sum())))
    return float(ece), bin_data


def multiclass_dice_specificity(y_true, y_pred, n_classes):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    dice, specificity = [], []
    total = cm.sum()
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        dice.append((2 * tp) / max(2 * tp + fp + fn, 1))
        specificity.append(tn / max(tn + fp, 1))
    return float(np.mean(dice)), float(np.mean(specificity)), dice, specificity


def plot_confusion(y_true, y_pred, model_name, out_dir):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)))
    plt.figure(figsize=(7.5, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={"size": 11}, linewidths=0.5, linecolor="white")
    plt.title(f"{model_name}: Confusion Matrix", weight="bold")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    savefig(Path(out_dir) / f"{model_name.lower()}_confusion_matrix.png")


def plot_roc(y_true, probs, model_name, out_dir):
    y_bin = label_binarize(y_true, classes=np.arange(len(CLASS_NAMES)))
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(CLASS_NAMES):
        if len(np.unique(y_bin[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        class_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{cls} (AUC={class_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Chance")
    plt.title(f"{model_name}: One-vs-Rest ROC Curves", weight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.legend(loc="lower right", fontsize=8)
    savefig(Path(out_dir) / f"{model_name.lower()}_roc_auc.png")


def plot_calibration(y_true, probs, model_name, out_dir):
    ece, bins = expected_calibration_error(y_true, probs)
    plt.figure(figsize=(6.5, 6))
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    if bins:
        plt.plot([b[0] for b in bins], [b[1] for b in bins], "o-", label="Model")
    plt.title(f"{model_name}: Calibration (ECE={ece:.4f})", weight="bold")
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Observed Accuracy")
    plt.legend()
    savefig(Path(out_dir) / f"{model_name.lower()}_calibration.png")


def evaluate_model(model, test_ds, model_name, out_dir, is_dann=False):
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    if is_dann:
        images = np.concatenate([x.numpy() for x, _ in test_ds], axis=0)
        probs, _ = model.predict([images, np.zeros(len(images), dtype=np.float32)], verbose=0)
    else:
        probs = model.predict(test_ds, verbose=0)
    y_pred = probs.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    dice, specificity, dice_per_class, specificity_per_class = multiclass_dice_specificity(
        y_true, y_pred, len(CLASS_NAMES)
    )
    try:
        auc_macro = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc_macro = float("nan")
    try:
        ce = log_loss(y_true, probs, labels=np.arange(len(CLASS_NAMES)))
    except ValueError:
        ce = float("nan")
    ece, _ = expected_calibration_error(y_true, probs)
    report = classification_report(
        y_true, y_pred, labels=np.arange(len(CLASS_NAMES)),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).T.to_csv(Path(out_dir) / f"{model_name.lower()}_classification_report.csv")
    pd.DataFrame({
        "true_label": [CLASS_NAMES[i] for i in y_true],
        "predicted_label": [CLASS_NAMES[i] for i in y_pred],
        "prob_" + CLASS_NAMES[0]: probs[:, 0],
        "prob_" + CLASS_NAMES[1]: probs[:, 1],
        "prob_" + CLASS_NAMES[2]: probs[:, 2],
        "prob_" + CLASS_NAMES[3]: probs[:, 3],
    }).to_csv(Path(out_dir) / f"{model_name.lower()}_test_predictions.csv", index=False)

    plot_confusion(y_true, y_pred, model_name, out_dir)
    plot_roc(y_true, probs, model_name, out_dir)
    plot_calibration(y_true, probs, model_name, out_dir)
    metrics = {
        "model": model_name, "accuracy": float(acc), "macro_precision": float(precision),
        "macro_recall_sensitivity": float(recall), "macro_f1": float(f1),
        "macro_specificity": float(specificity), "macro_dice": float(dice),
        "macro_roc_auc_ovr": float(auc_macro), "ece_15_bins": float(ece),
        "cross_entropy": float(ce), "n_test": int(len(y_true)),
    }
    for i, cls in enumerate(CLASS_NAMES):
        metrics[f"dice_{cls}"] = float(dice_per_class[i])
        metrics[f"specificity_{cls}"] = float(specificity_per_class[i])
    return metrics, y_true, y_pred, probs


def gradcam(model, image_path, model_name, out_dir):
    """Grad-CAM for CNN transfer models; uses the last 4D convolutional layer."""
    try:
        # Find last convolutional layer, including inside application backbone.
        conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, keras.Model):
                for inner in reversed(layer.layers):
                    if len(getattr(inner, "output_shape", ())) == 4:
                        conv_layer = inner
                        break
            if conv_layer is not None:
                break
            if len(getattr(layer, "output_shape", ())) == 4:
                conv_layer = layer
                break
        if conv_layer is None:
            print(f"Grad-CAM skipped for {model_name}: no 4-D feature layer found.")
            return

        image = keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        arr = keras.utils.img_to_array(image).astype("float32")
        # Model has internal preprocessing, so pass resized RGB pixels [0,255].
        conv_model = keras.Model(model.inputs[0], [conv_layer.output, model.output])
        with tf.GradientTape() as tape:
            conv_out, predictions = conv_model(tf.expand_dims(arr, axis=0), training=False)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
        grads = tape.gradient(loss, conv_out)
        if grads is None:
            print(f"Grad-CAM skipped for {model_name}: gradients unavailable.")
            return
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_out[0] * weights, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = tf.image.resize(heatmap[..., tf.newaxis], [IMG_SIZE, IMG_SIZE]).numpy().squeeze()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(arr.astype("uint8"))
        axes[0].set_title("Input MRI")
        axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM Heatmap")
        axes[2].imshow(arr.astype("uint8"))
        axes[2].imshow(heatmap, cmap="jet", alpha=0.40, vmin=0, vmax=1)
        axes[2].set_title(f"Overlay: {CLASS_NAMES[int(class_idx)]}")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(f"{model_name}: Explainable AI (Grad-CAM)", weight="bold")
        savefig(Path(out_dir) / f"{model_name.lower()}_gradcam.png")
    except Exception as e:
        print(f"Grad-CAM could not be generated for {model_name}: {e}")


def train_standard_model(model_name, train_df, val_df, test_ds, out_dir, epochs):
    model_dir = Path(out_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df, training=False)
    final_model_path = model_dir / f"{model_name.lower()}_final.keras"

    # If this model completed on a previous run, reuse it rather than retraining.
    if final_model_path.exists():
        print(f"Found completed model: {final_model_path}. Loading it instead of retraining.")
        model = keras.models.load_model(final_model_path)
        metrics, y_true, y_pred, probs = evaluate_model(
            model, test_ds, model_name, out_dir
        )
        return model, metrics, y_true, y_pred, probs

    model = build_custom_cnn() if model_name == "CustomCNN" else build_transfer_model(model_name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    initial_backup_dir = model_dir / f"{model_name.lower()}_backup_initial"
    callbacks = [
        # Restores model + optimizer state from the most recent completed epoch
        # if this run was interrupted.
        keras.callbacks.BackupAndRestore(
            backup_dir=str(initial_backup_dir),
            save_freq="epoch",
            delete_checkpoint=True
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                          patience=3, min_lr=1e-7, verbose=1),
        keras.callbacks.ModelCheckpoint(
            str(model_dir / f"{model_name.lower()}_best.keras"),
            monitor="val_loss", save_best_only=True, verbose=0
        ),
        keras.callbacks.CSVLogger(str(Path(out_dir) / f"{model_name.lower()}_training_log.csv"))
    ]
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        class_weight=get_class_weights(train_df), callbacks=callbacks, verbose=1
    )
    # Optional light fine-tuning for transfer learning, with a low learning rate.
    if model_name not in {"CustomCNN"} and hasattr(model, "backbone"):
        model.backbone.trainable = True
        # Keep BatchNorm statistics stable and only tune upper backbone layers.
        for layer in model.backbone.layers[:-25]:
            layer.trainable = False
        for layer in model.backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
        fine_backup_dir = model_dir / f"{model_name.lower()}_backup_finetune"
        fine_callbacks = [
            keras.callbacks.BackupAndRestore(
                backup_dir=str(fine_backup_dir),
                save_freq="epoch",
                delete_checkpoint=True
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                          restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                              patience=2, min_lr=1e-7, verbose=1),
            keras.callbacks.ModelCheckpoint(
                str(model_dir / f"{model_name.lower()}_best.keras"),
                monitor="val_loss", save_best_only=True, verbose=0
            ),
            keras.callbacks.CSVLogger(
                str(Path(out_dir) / f"{model_name.lower()}_finetune_log.csv")
            )
        ]
        history = model.fit(
            train_ds, validation_data=val_ds, epochs=max(5, epochs // 2),
            class_weight=get_class_weights(train_df), callbacks=fine_callbacks, verbose=1
        )
    model.save(final_model_path)
    plot_training(history, model_name, out_dir)
    metrics, y_true, y_pred, probs = evaluate_model(
        model, test_ds, model_name, out_dir
    )
    sample_path = test_df_sample_path(test_ds, None)
    # Grad-CAM needs a real image path; supplied later by caller.
    return model, metrics, y_true, y_pred, probs


def test_df_sample_path(test_ds, fallback):
    return fallback


def train_dann(train_df, val_df, test_ds, out_dir, epochs):
    """Train DANN only when at least two genuine domain labels are available."""
    domains = sorted(set(train_df.domain) - {"unknown"})
    if len(domains) < 2:
        print("DANN SKIPPED: at least two meaningful domain/site/scanner labels are "
              "required. The folder structure did not provide them. Do not create "
              "artificial domain labels just to make DANN run.")
        return None
    domain_to_id = {d: i for i, d in enumerate(domains)}
    tr = train_df[train_df.domain.isin(domains)].copy()
    va = val_df[val_df.domain.isin(domains)].copy()
    if len(tr) == 0 or len(va) == 0:
        print("DANN SKIPPED: train/validation split has no usable domain labels.")
        return None

    def dataset_with_domain(frame, training=False):
        ds = make_dataset(frame, training=training, batch_size=BATCH_SIZE)
        # Rebuild dataset with domain labels by matching path lookup.
        domain_lookup = dict(zip(frame.path, frame.domain.map(domain_to_id)))
        def add_domain(x, y):
            # Path is no longer present after image loading, so construct from arrays
            # is not possible here. Use a direct path-aware pipeline instead.
            return x, {"class_output": y}
        # Dedicated path-aware data pipeline.
        paths = frame.path.astype(str).values
        cls_ids = frame.label.map({c: i for i, c in enumerate(CLASS_NAMES)}).values.astype("int32")
        dom_ids = frame.domain.map(domain_to_id).values.astype("int32")
        raw = tf.data.Dataset.from_tensor_slices((paths, cls_ids, dom_ids))
        def load(path, cls, dom):
            image_bytes = tf.io.read_file(path)
            im = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
            im.set_shape([None, None, 3])
            im = tf.cast(tf.image.resize(im, [IMG_SIZE, IMG_SIZE]), tf.float32)
            return (im, tf.constant(1.0, tf.float32)), {
                "class_output": cls, "domain_output": dom
            }
        raw = raw.map(load, num_parallel_calls=AUTOTUNE)
        if training:
            raw = raw.shuffle(min(len(frame), 3000), seed=SEED)
        return raw.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    model = build_dann(num_domains=len(domains))
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss={"class_output": "sparse_categorical_crossentropy",
              "domain_output": "sparse_categorical_crossentropy"},
        loss_weights={"class_output": 1.0, "domain_output": 0.2},
        metrics={"class_output": ["accuracy"], "domain_output": ["accuracy"]}
    )
    # This version uses alpha=1; GRL reverses gradients. A production experiment
    # can schedule alpha over training progress.
    tr_ds, va_ds = dataset_with_domain(tr, True), dataset_with_domain(va, False)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_class_output_loss", patience=6,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            str(Path(out_dir) / "models" / "dann_best.keras"),
            monitor="val_class_output_loss", save_best_only=True
        ),
        keras.callbacks.CSVLogger(str(Path(out_dir) / "dann_training_log.csv"))
    ]
    model.fit(tr_ds, validation_data=va_ds, epochs=epochs, callbacks=callbacks, verbose=1)
    model.save(Path(out_dir) / "models" / "dann_final.keras")
    return model


def main():
    parser = argparse.ArgumentParser(description="Four-class brain MRI classification pipeline")
    parser.add_argument("--data_dir", required=True, help="Dataset root directory")
    parser.add_argument("--output_dir", default="brain_tumor_results", help="Output folder")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--models", nargs="+", default=[
        "CustomCNN", "VGG16", "ResNet50", "EfficientNetB0",
        "DenseNet121", "MobileNetV2", "InceptionV3", "Xception"
    ], choices=["CustomCNN", "VGG16", "ResNet50", "EfficientNetB0",
                "DenseNet121", "MobileNetV2", "InceptionV3", "Xception"])
    parser.add_argument("--include_dann", action="store_true",
                        help="Attempt DANN if real domain labels exist in folder hierarchy")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["eda", "models", "xai"]:
        (out_dir / sub).mkdir(exist_ok=True)

    df = find_image_records(args.data_dir)
    df = perform_eda(df, out_dir)
    train_df, val_df, test_df = split_data(df, out_dir)
    print(f"Split sizes: train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}")
    print("Train distribution:\n", train_df.label.value_counts())
    print("Validation distribution:\n", val_df.label.value_counts())
    print("Test distribution:\n", test_df.label.value_counts())

    test_ds = make_dataset(test_df, training=False)
    all_metrics = []
    predictions_for_stats = {}
    trained_models = {}

    for model_name in args.models:
        print(f"\n{'=' * 24} Training/resuming {model_name} {'=' * 24}")
        print("Resume behavior: unfinished models restore the latest completed epoch; "
              "models with a *_final.keras file are loaded without retraining.")
        try:
            model, metrics, y_true, y_pred, probs = train_standard_model(
                model_name, train_df, val_df, test_ds, out_dir, args.epochs
            )
            all_metrics.append(metrics)
            predictions_for_stats[model_name] = (y_true, y_pred, probs)
            trained_models[model_name] = model
            # Grad-CAM on a held-out test image, not a training image.
            gradcam(model, test_df.iloc[0].path, model_name, out_dir / "xai")
            print(json.dumps(metrics, indent=2))
            pd.DataFrame(all_metrics).to_csv(out_dir / "model_comparison.csv", index=False)
        except Exception as e:
            print(f"ERROR while training {model_name}: {e}")
            with open(out_dir / "errors.log", "a", encoding="utf-8") as f:
                f.write(f"{model_name}: {repr(e)}\n")

    if args.include_dann:
        try:
            dann_model = train_dann(train_df, val_df, test_ds, out_dir, args.epochs)
            if dann_model is not None:
                metrics, y_true, y_pred, probs = evaluate_model(
                    dann_model, test_ds, "DANN", out_dir, is_dann=True
                )
                all_metrics.append(metrics)
                predictions_for_stats["DANN"] = (y_true, y_pred, probs)
                gradcam(dann_model, test_df.iloc[0].path, "DANN", out_dir / "xai")
        except Exception as e:
            print(f"DANN failed: {e}")
            with open(out_dir / "errors.log", "a", encoding="utf-8") as f:
                f.write(f"DANN: {repr(e)}\n")

    if not all_metrics:
        raise RuntimeError("No models completed successfully. Check errors.log.")
    results = pd.DataFrame(all_metrics).sort_values("macro_f1", ascending=False)
    results.to_csv(out_dir / "model_comparison.csv", index=False)

    # Comparison figure; all metrics are test-set metrics.
    metric_cols = ["accuracy", "macro_recall_sensitivity", "macro_f1",
                   "macro_specificity", "macro_dice", "macro_roc_auc_ovr"]
    plot_df = results.set_index("model")[metric_cols]
    ax = plot_df.plot(kind="bar", figsize=(14, 7), ylim=(0, 1), colormap="Set2")
    ax.set_title("Held-Out Test Performance Comparison", weight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    savefig(out_dir / "model_comparison_metrics.png")

    # McNemar-style exact binomial test on paired correctness for top two models.
    # This is an approximate paired comparison (exact binomial test on discordant cases).
    stats = {"interpretation_note": (
        "p-values compare paired test-set correctness of the top two models using "
        "an exact binomial test on discordant predictions; they do not establish "
        "clinical significance and should be reported with study design context."
    )}
    if len(results) >= 2:
        m1, m2 = results.iloc[0]["model"], results.iloc[1]["model"]
        y1, p1, _ = predictions_for_stats[m1]
        y2, p2, _ = predictions_for_stats[m2]
        correct1, correct2 = (p1 == y1), (p2 == y2)
        n01 = int(np.sum(correct1 & ~correct2))
        n10 = int(np.sum(~correct1 & correct2))
        n_discordant = n01 + n10
        p_value = float(binomtest(min(n01, n10), n=n_discordant, p=0.5).pvalue) if n_discordant else 1.0
        stats["paired_test"] = {
            "model_1": m1, "model_2": m2,
            "model_1_only_correct": n01, "model_2_only_correct": n10,
            "discordant_cases": n_discordant, "exact_binomial_p_value": p_value
        }
    (out_dir / "paired_model_comparison_pvalue.json").write_text(json.dumps(stats, indent=2))

    # Final overview of class-specific metrics and run configuration.
    config = {
        "seed": SEED, "image_size": IMG_SIZE, "batch_size": BATCH_SIZE,
        "epochs_requested": args.epochs, "classes": CLASS_NAMES,
        "train_n": len(train_df), "validation_n": len(val_df), "test_n": len(test_df),
        "models_requested": args.models, "dann_requested": args.include_dann,
        "figure_dpi": 300
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    print("\nFinished. Results saved to:", out_dir)
    print(results[["model", "accuracy", "macro_recall_sensitivity", "macro_f1",
                   "macro_specificity", "macro_dice", "macro_roc_auc_ovr", "ece_15_bins"]].to_string(index=False))
    print("\nIMPORTANT: inspect errors.log, duplicate audits, and class counts before reporting results.")


if __name__ == "__main__":
    main()
