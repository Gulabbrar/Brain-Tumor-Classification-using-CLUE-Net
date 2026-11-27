# ================================================================
# VALIDATED POST-TRAINING XAI PIPELINE FOR BRAIN MRI CLASSIFICATION
# Loads existing .keras models only. Does NOT retrain or overwrite them.
#
# Supports: Grad-CAM, Grad-CAM++, LIME, Occlusion
# SHAP is intentionally excluded from this version until a defensible
# training-set background and superpixel-level feature design are added.
#
# Expected model input: RGB float32 images in [0, 255], 224 x 224.
# Class order MUST match the supplied training pipeline:
# 0 glioma, 1 meningioma, 2 pituitary, 3 no_tumor
# ================================================================

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as ndi
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore")

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
CLASS_ALIASES = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "notumor": "no_tumor",
    "normal": "no_tumor",
    "healthy": "no_tumor",
    "notumour": "no_tumor",
}
IMG_SIZE = 224
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_FILES = {
    "CustomCNN": "customcnn_final.keras",
    "VGG16": "vgg16_final.keras",
    "ResNet50": "resnet50_final.keras",
    "EfficientNetB0": "efficientnetb0_final.keras",
    "DenseNet121": "densenet121_final.keras",
    "MobileNetV2": "mobilenetv2_final.keras",
    "InceptionV3": "inceptionv3_final.keras",
    "Xception": "xception_final.keras",
}


def canonical_class(folder_name):
    key = "".join(ch for ch in folder_name.lower() if ch.isalnum())
    for alias, canonical in CLASS_ALIASES.items():
        if alias in key:
            return canonical
    return None


def discover_test_images(data_dir):
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")

    records = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
        }:
            continue
        if not any(part.lower() in {"test", "testing"} for part in path.parts):
            continue

        label = None
        for parent in path.parents:
            label = canonical_class(parent.name)
            if label:
                break
            if parent == root.parent:
                break
        if label:
            records.append({"path": str(path), "label": label})

    frame = pd.DataFrame(records).drop_duplicates("path")
    if frame.empty:
        raise RuntimeError(
            f"No labelled test images found under {root}. "
            "Expected Test/<class_name>/<image> or test/<class_name>/<image>."
        )

    missing = set(CLASS_NAMES) - set(frame.label.unique())
    if missing:
        raise RuntimeError(
            f"Test split is missing classes: {sorted(missing)}. "
            "Check folder names and the held-out test directory."
        )
    return frame


def load_image(path):
    image = keras.utils.load_img(
        path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb"
    )
    return keras.utils.img_to_array(image).astype(np.float32)


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def prediction_array(model, batch):
    """Standard four-class classifier inference; does not normalize inputs."""
    output = model(tf.convert_to_tensor(batch, dtype=tf.float32), training=False)
    if isinstance(output, dict):
        if "class_output" not in output:
            raise RuntimeError(f"Model output dictionary keys: {list(output.keys())}")
        output = output["class_output"]
    elif isinstance(output, (tuple, list)):
        output = output[0]
    result = np.asarray(output)
    if result.ndim != 2 or result.shape[-1] != len(CLASS_NAMES):
        raise RuntimeError(
            f"Expected (batch, 4) class probabilities; received {result.shape}. "
            "This script does not currently support DANN's two-input signature."
        )
    return result


def locate_last_conv(model):
    """Find the last 4-D convolutional layer, including nested application models."""
    candidates = []

    def visit(container, prefix=""):
        for layer in container.layers:
            if isinstance(layer, keras.Model):
                visit(layer, prefix + layer.name + "/")
            elif isinstance(layer, (
                keras.layers.Conv2D,
                keras.layers.DepthwiseConv2D,
                keras.layers.SeparableConv2D,
            )):
                try:
                    if len(layer.output.shape) == 4:
                        candidates.append((prefix + layer.name, layer))
                except Exception:
                    pass

    visit(model)
    if not candidates:
        raise RuntimeError("No 4-D convolutional layer found.")
    return candidates[-1]


def build_gradcam_graph(model, conv_layer):
    """
    Build a graph that preserves the original model's preprocessing and head.

    For transfer models, the nested backbone is evaluated separately to expose
    the chosen internal activation. The original outer-model layers before and
    after the backbone are reused in their original order.
    """
    backbone = getattr(model, "backbone", None)

    if backbone is None:
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output],
        )
        return ("flat", grad_model, None)

    # This is the architecture used in the supplied training pipeline:
    # input -> augmentation/preprocessing -> backbone -> pooling/head -> softmax.
    backbone_index = next(
        (i for i, layer in enumerate(model.layers) if layer is backbone), None
    )
    if backbone_index is None:
        raise RuntimeError(
            "A model.backbone attribute exists, but the backbone is not a "
            "top-level layer. Cannot safely reconstruct the classifier head."
        )

    try:
        pre_backbone = keras.Model(model.inputs, backbone.input)
        feature_model = keras.Model(
            backbone.inputs, [conv_layer.output, backbone.output]
        )
        head_layers = model.layers[backbone_index + 1:]
        if not head_layers:
            raise RuntimeError("No classifier head layers found after backbone.")

        # Reuse original layers and weights; this does not train or alter them.
        head_input = keras.Input(shape=tuple(backbone.output.shape[1:]))
        z = head_input
        for layer in head_layers:
            z = layer(z)
        head_model = keras.Model(head_input, z)
        return ("nested", pre_backbone, feature_model, head_model)
    except Exception as exc:
        raise RuntimeError(
            "Could not construct a connected Grad-CAM graph for this nested "
            f"backbone: {exc}"
        ) from exc


def gradcam_map(model, image, model_name, method="Grad-CAM"):
    conv_name, conv_layer = locate_last_conv(model)
    graph = build_gradcam_graph(model, conv_layer)
    x = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)

    if graph[0] == "flat":
        grad_model = graph[1]
        with tf.GradientTape() as tape:
            conv, pred = grad_model(x, training=False)
            class_id = tf.argmax(pred[0], output_type=tf.int32)
            score = pred[:, class_id]
    else:
        pre_backbone, feature_model, head_model = graph[1:]
        backbone_input = pre_backbone(x, training=False)
        with tf.GradientTape() as tape:
            conv, backbone_features = feature_model(backbone_input, training=False)
            pred = head_model(backbone_features, training=False)
            class_id = tf.argmax(pred[0], output_type=tf.int32)
            score = pred[:, class_id]

    grads = tape.gradient(score, conv)
    if grads is None:
        raise RuntimeError("Gradient is disconnected from the selected feature map.")

    if method == "Grad-CAM":
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.reduce_sum(weights * conv, axis=-1)

    elif method == "Grad-CAM++":
        # Common gradient-power approximation to Grad-CAM++ alpha weights.
        # This is NOT an exact numerical evaluation of second/third derivatives.
        grads2 = tf.square(grads)
        grads3 = grads2 * grads
        sum_activations = tf.reduce_sum(conv, axis=(1, 2), keepdims=True)
        denominator = 2.0 * grads2 + sum_activations * grads3
        alpha = grads2 / (denominator + 1e-8)
        alpha = tf.where(tf.math.is_finite(alpha), alpha, tf.zeros_like(alpha))
        weights = tf.reduce_sum(
            alpha * tf.nn.relu(grads), axis=(1, 2), keepdims=True
        )
        cam = tf.reduce_sum(weights * conv, axis=-1)
    else:
        raise ValueError(f"Unknown CAM method: {method}")

    cam = tf.nn.relu(cam)[0]
    maximum = tf.reduce_max(cam)
    if float(maximum.numpy()) <= 1e-10:
        raise RuntimeError("CAM is numerically empty for this image and target.")
    cam = cam / maximum
    cam = tf.image.resize(cam[..., None], [IMG_SIZE, IMG_SIZE]).numpy().squeeze()
    probabilities = prediction_array(model, image[None, ...])[0]
    predicted = int(np.argmax(probabilities))
    return cam, predicted, probabilities, conv_name


def save_cam_panel(image, cam, model_name, method, predicted, conv_name, out_path):
    base = np.clip(image / 255.0, 0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(base)
    axes[0].set_title("Original MRI")
    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(method)
    axes[2].imshow(base)
    axes[2].imshow(cam, cmap="jet", alpha=0.40, vmin=0, vmax=1)
    axes[2].set_title(f"Predicted: {CLASS_NAMES[predicted]}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{model_name} | {method} | layer: {conv_name}", fontweight="bold")
    save_fig(out_path)


def approximate_foreground_mask(image):
    """
    Suppress obvious black canvas only. This is NOT a brain or tumour mask.
    """
    gray = np.mean(image, axis=-1)
    threshold = max(6.0, float(np.percentile(gray, 55)) * 0.12)
    mask = gray > threshold
    mask = ndi.binary_closing(mask, structure=np.ones((9, 9)))
    mask = ndi.binary_fill_holes(mask)
    labels, count = ndi.label(mask)
    if count == 0:
        return np.ones(gray.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    result = labels == int(np.argmax(sizes))
    return result if result.mean() >= 0.08 else np.ones(gray.shape, dtype=bool)


def run_lime(model, image, model_name, out_path, num_samples=1000):
    try:
        from lime import lime_image
        from skimage.segmentation import slic
    except ImportError as exc:
        raise RuntimeError("Install LIME dependencies: pip install lime scikit-image") from exc

    foreground = approximate_foreground_mask(image)

    def segmentation_fn(rgb):
        segments = slic(
            rgb, n_segments=90, compactness=0.12, sigma=1.0,
            start_label=1, channel_axis=-1, mask=foreground
        )
        segments = np.asarray(segments, dtype=np.int32)
        segments[~foreground] = 0
        return segments

    def predict_fn(batch):
        return prediction_array(model, np.asarray(batch, dtype=np.float32))

    explainer = lime_image.LimeImageExplainer(random_state=SEED)
    explanation = explainer.explain_instance(
        image.astype(np.uint8), predict_fn, top_labels=1,
        hide_color=0, num_samples=num_samples, segmentation_fn=segmentation_fn
    )
    class_id = int(explanation.top_labels[0])
    _, positive = explanation.get_image_and_mask(
        class_id, positive_only=True, num_features=12, hide_rest=False
    )
    _, negative = explanation.get_image_and_mask(
        class_id, positive_only=False, negative_only=True,
        num_features=12, hide_rest=False
    )
    positive = (positive > 0) & foreground
    negative = (negative > 0) & foreground
    base = np.clip(image / 255.0, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(base)
    axes[0].set_title("Original MRI")
    axes[1].imshow(base)
    overlay = np.zeros((*foreground.shape, 4), dtype=np.float32)
    overlay[..., 1] = 1
    overlay[..., 3] = positive.astype(np.float32) * 0.48
    axes[1].imshow(overlay)
    axes[1].set_title("Positive evidence")
    axes[2].imshow(base)
    overlay = np.zeros((*foreground.shape, 4), dtype=np.float32)
    overlay[..., 0] = 1
    overlay[..., 3] = negative.astype(np.float32) * 0.48
    axes[2].imshow(overlay)
    axes[2].set_title("Negative evidence")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{model_name} | LIME | explained class: {CLASS_NAMES[class_id]}")
    save_fig(out_path)


def run_occlusion(model, image, model_name, out_path, patch=16, stride=8):
    """Probability-drop occlusion map for the model's predicted class."""
    base_prob = prediction_array(model, image[None, ...])[0]
    target = int(np.argmax(base_prob))
    foreground = approximate_foreground_mask(image)
    h, w = image.shape[:2]
    total = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    fill = np.mean(image[foreground], axis=0) if foreground.any() else np.mean(image, axis=(0, 1))
    batch, coords = [], []

    def evaluate(items, locations):
        outputs = prediction_array(model, np.asarray(items, dtype=np.float32))
        for (y, x, local_mask), probability in zip(locations, outputs):
            drop = float(base_prob[target] - probability[target])
            view = total[y:y+patch, x:x+patch]
            count_view = counts[y:y+patch, x:x+patch]
            view[local_mask] += drop
            count_view[local_mask] += 1

    for y in range(0, h - patch + 1, stride):
        for x in range(0, w - patch + 1, stride):
            local = foreground[y:y+patch, x:x+patch]
            if local.mean() < 0.35:
                continue
            altered = image.copy()
            region = altered[y:y+patch, x:x+patch]
            region[local] = fill
            batch.append(altered)
            coords.append((y, x, local.copy()))
            if len(batch) == 32:
                evaluate(batch, coords)
                batch, coords = [], []
    if batch:
        evaluate(batch, coords)

    sensitivity = total / np.maximum(counts, 1)
    sensitivity[~foreground] = 0
    scale = np.max(np.abs(sensitivity)) + 1e-8
    normalized = sensitivity / scale

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    base = np.clip(image / 255.0, 0, 1)
    axes[0].imshow(base)
    axes[0].set_title("Original MRI")
    axes[1].imshow(normalized, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title("Occlusion sensitivity")
    axes[2].imshow(base)
    axes[2].imshow(normalized, cmap="coolwarm", alpha=0.48, vmin=-1, vmax=1)
    axes[2].set_title(f"Target: {CLASS_NAMES[target]}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"{model_name} | Occlusion | patch={patch}, stride={stride} "
        "| positive = probability decrease"
    )
    save_fig(out_path)


def select_cases(test_frame, model_predictions, per_class=4):
    """
    Pick a reproducible mixture of correct and incorrect cases per true class.
    Selection is based on model predictions, not alphabetical file order.
    """
    frame = test_frame.copy()
    frame["predicted"] = model_predictions
    frame["correct"] = frame.label == frame.predicted
    selected = []
    for cls in CLASS_NAMES:
        subset = frame[frame.label == cls].sort_values("path")
        if subset.empty:
            continue
        correct = subset[subset.correct]
        incorrect = subset[~subset.correct]
        n_correct = min(len(correct), max(1, (per_class + 1) // 2))
        n_incorrect = min(len(incorrect), per_class - n_correct)
        chosen = pd.concat([correct.head(n_correct), incorrect.head(n_incorrect)])
        if len(chosen) < per_class:
            remainder = subset[~subset.path.isin(chosen.path)]
            chosen = pd.concat([chosen, remainder.head(per_class - len(chosen))])
        selected.extend(chosen.head(per_class).to_dict("records"))
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Post-training XAI; loads existing models and never retrains."
    )
    parser.add_argument("--data_dir", required=True,
                        help="Dataset root containing an explicit Test/test split")
    parser.add_argument("--models_dir", required=True,
                        help="Folder containing *_final.keras files")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--models", nargs="+",
                        default=["CustomCNN", "EfficientNetB0"],
                        choices=list(MODEL_FILES))
    parser.add_argument("--per_class", type=int, default=4)
    parser.add_argument("--methods", nargs="+",
                        default=["GradCAM", "GradCAM++", "LIME", "Occlusion"],
                        choices=["GradCAM", "GradCAM++", "LIME", "Occlusion"])
    parser.add_argument("--lime_samples", type=int, default=1000)
    args = parser.parse_args()

    if args.per_class < 1:
        raise ValueError("--per_class must be at least 1.")

    models_dir = Path(args.models_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else models_dir.parent / "xai_validated"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    test_frame = discover_test_images(args.data_dir)
    print(f"Found {len(test_frame)} held-out test images.")
    print("Class counts:", test_frame.label.value_counts().to_dict())

    summary, failures = [], []
    for model_name in args.models:
        model_path = models_dir / MODEL_FILES[model_name]
        if not model_path.exists():
            failures.append({
                "model": model_name, "case": "", "method": "load",
                "error": f"Model file not found: {model_path}"
            })
            print(f"SKIP {model_name}: missing {model_path}")
            continue

        print(f"\nLoading {model_name}: {model_path}")
        try:
            # safe_mode=False is needed only because trusted training models
            # contain Lambda preprocessing layers. Load only your own models.
            model = keras.models.load_model(
                model_path, compile=False, safe_mode=False
            )
        except Exception as exc:
            failures.append({
                "model": model_name, "case": "", "method": "load", "error": repr(exc)
            })
            print(f"LOAD FAILED: {exc}")
            continue

        # Predict the complete held-out test split before selecting explanation cases.
        all_probs = []
        for start in range(0, len(test_frame), 32):
            paths = test_frame.path.iloc[start:start+32].tolist()
            batch = np.stack([load_image(path) for path in paths])
            all_probs.append(prediction_array(model, batch))
        all_probs = np.concatenate(all_probs, axis=0)
        all_pred_ids = all_probs.argmax(axis=1)
        all_pred_names = [CLASS_NAMES[i] for i in all_pred_ids]
        cases = select_cases(test_frame, all_pred_names, args.per_class)

        model_dir = output_dir / model_name.lower()
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save full-test prediction table, not merely the XAI example subset.
        prediction_rows = []
        for i, row in enumerate(test_frame.itertuples(index=False)):
            prediction_rows.append({
                "image": row.path,
                "true_class": row.label,
                "predicted_class": all_pred_names[i],
                "confidence": float(np.max(all_probs[i])),
                "correct": bool(row.label == all_pred_names[i]),
            })
        pd.DataFrame(prediction_rows).to_csv(
            model_dir / "full_test_predictions.csv", index=False
        )

        test_accuracy = float(np.mean(
            test_frame.label.to_numpy() == np.asarray(all_pred_names)
        ))
        print(f"{model_name} full-test accuracy from loaded model: {test_accuracy:.4f}")

        for case_number, record in enumerate(cases, start=1):
            image = load_image(record["path"])
            true_label = record["label"]
            safe_stem = Path(record["path"]).stem[:50]
            prefix = f"{case_number:03d}_{true_label}_{safe_stem}"
            probs = prediction_array(model, image[None, ...])[0]
            predicted = int(np.argmax(probs))

            for method in args.methods:
                out_path = model_dir / f"{prefix}_{method.lower().replace('+', 'plus')}.png"
                try:
                    if method == "GradCAM":
                        cam, pred, _, conv_name = gradcam_map(
                            model, image, model_name, "Grad-CAM"
                        )
                        save_cam_panel(
                            image, cam, model_name, "Grad-CAM", pred, conv_name, out_path
                        )
                    elif method == "GradCAM++":
                        cam, pred, _, conv_name = gradcam_map(
                            model, image, model_name, "Grad-CAM++"
                        )
                        save_cam_panel(
                            image, cam, model_name, "Grad-CAM++", pred, conv_name, out_path
                        )
                    elif method == "LIME":
                        run_lime(
                            model, image, model_name, out_path,
                            num_samples=args.lime_samples
                        )
                    elif method == "Occlusion":
                        run_occlusion(model, image, model_name, out_path)

                    summary.append({
                        "model": model_name,
                        "image": record["path"],
                        "true_class": true_label,
                        "predicted_class": CLASS_NAMES[predicted],
                        "confidence": float(probs[predicted]),
                        "correct": bool(CLASS_NAMES[predicted] == true_label),
                        "method": method,
                        "status": "success",
                        "output": str(out_path),
                    })
                    print(f"  OK {method}: {prefix}")
                except Exception as exc:
                    failures.append({
                        "model": model_name, "case": record["path"],
                        "method": method, "error": repr(exc)
                    })
                    summary.append({
                        "model": model_name,
                        "image": record["path"],
                        "true_class": true_label,
                        "predicted_class": CLASS_NAMES[predicted],
                        "confidence": float(probs[predicted]),
                        "correct": bool(CLASS_NAMES[predicted] == true_label),
                        "method": method,
                        "status": "failed",
                        "output": "",
                    })
                    print(f"  FAILED {method}: {exc}")

        del model
        keras.backend.clear_session()

    pd.DataFrame(summary).to_csv(output_dir / "xai_method_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(output_dir / "xai_failures.csv", index=False)
    (output_dir / "run_configuration.json").write_text(json.dumps({
        "seed": SEED,
        "image_size": IMG_SIZE,
        "input_range": "[0,255]",
        "class_order": CLASS_NAMES,
        "models": args.models,
        "methods": args.methods,
        "per_class": args.per_class,
        "lime_samples": args.lime_samples,
        "retrained_models": False,
        "shap_included": False,
        "foreground_mask_is_tumour_mask": False,
    }, indent=2), encoding="utf-8")

    print("\nXAI processing finished.")
    print("Output directory:", output_dir)
    print("Review xai_method_summary.csv, xai_failures.csv, and each model's full_test_predictions.csv.")
    print("IMPORTANT: foreground masking is not brain/tumour segmentation.")
    print("IMPORTANT: Grad-CAM++ uses a gradient-power approximation; validate it and report this implementation detail.")
    print("IMPORTANT: this script generates explanations; it does not by itself establish clinical validity.")


if __name__ == "__main__":
    main()
