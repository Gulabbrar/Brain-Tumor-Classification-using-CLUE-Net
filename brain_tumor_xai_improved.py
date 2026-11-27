# ================================================================
# POST-TRAINING EXPLAINABLE AI (XAI) FOR BRAIN TUMOR CLASSIFICATION
# Loads existing .keras models; DOES NOT retrain.
# Methods: Grad-CAM, Grad-CAM++, LIME, SHAP, Occlusion Sensitivity
# ================================================================

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
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
    "DANN": "dann_final.keras",
}


def canonical_class(folder_name):
    s = "".join(ch for ch in folder_name.lower() if ch.isalnum())
    if "glioma" in s:
        return "glioma"
    if "meningioma" in s:
        return "meningioma"
    if "pituitary" in s:
        return "pituitary"
    if "notumor" in s or "normal" in s or "healthy" in s:
        return "no_tumor"
    return None


def discover_test_images(data_dir):
    root = Path(data_dir).expanduser().resolve()
    rows = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {
            ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
        }:
            continue
        parts = [x.lower() for x in p.parts]
        # Restrict to explicit test/testing split, avoiding train leakage.
        if not any(x in {"test", "testing"} for x in parts):
            continue
        label = None
        for parent in p.parents:
            label = canonical_class(parent.name)
            if label:
                break
            if parent == root.parent:
                break
        if label:
            rows.append({"path": str(p), "label": label})
    df = pd.DataFrame(rows).drop_duplicates("path")
    if df.empty:
        raise RuntimeError(
            f"No test images detected under {root}. Ensure the test folder "
            "contains class subfolders."
        )
    missing = set(CLASS_NAMES) - set(df.label.unique())
    if missing:
        print("WARNING: Missing test class(es):", sorted(missing))
    return df


def load_image(path):
    im = keras.utils.load_img(path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="rgb")
    arr = keras.utils.img_to_array(im).astype(np.float32)
    return arr



def get_head_foreground_mask(image):
    """
    Approximate visible head/MRI foreground mask.
    This is only used to suppress obvious black background; it is NOT
    a brain parenchyma mask, tumour mask, or clinical annotation.
    """
    gray = np.mean(image, axis=-1)
    threshold = max(6.0, float(np.percentile(gray, 55)) * 0.12)
    mask = gray > threshold
    mask = ndi.binary_closing(mask, structure=np.ones((9, 9)))
    mask = ndi.binary_fill_holes(mask)
    labels, n = ndi.label(mask)
    if n == 0:
        return np.ones(gray.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes))
    result = labels == largest
    # Avoid an overly aggressive mask if thresholding found only a small object.
    if result.mean() < 0.08:
        return np.ones(gray.shape, dtype=bool)
    return result


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def safe_model_predict(model, image_batch, model_name):
    # Standard classifiers output softmax class probabilities.
    # DANN has two inputs and two outputs; use the class output only.
    if model_name == "DANN":
        outputs = model([image_batch, tf.zeros((len(image_batch),), dtype=tf.float32)],
                        training=False)
        if isinstance(outputs, (tuple, list)):
            return outputs[0].numpy()
        if isinstance(outputs, dict):
            return outputs["class_output"].numpy()
    pred = model(image_batch, training=False)
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    if isinstance(pred, dict):
        pred = pred.get("class_output", list(pred.values())[0])
    return pred.numpy()


def locate_last_conv_layer(model):
    # Traverse nested Models (e.g., Keras application backbones).
    candidates = []
    def walk(layer, prefix=""):
        if isinstance(layer, keras.Model):
            for sub in layer.layers:
                walk(sub, prefix + layer.name + "/")
        else:
            try:
                shape = tuple(layer.output.shape)
                if len(shape) == 4 and isinstance(
                    layer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D,
                          keras.layers.SeparableConv2D)
                ):
                    candidates.append((prefix + layer.name, layer))
            except Exception:
                pass
    for layer in model.layers:
        walk(layer)
    return candidates[-1][1] if candidates else None


def gradcam_family(model, image, model_name, method="Grad-CAM"):
    """
    Grad-CAM and Grad-CAM++.
    Attempts a connected graph from selected conv layer to model prediction.
    If nested graph extraction fails, method is marked unsupported rather than
    generating a misleading map.
    """
    conv_layer = locate_last_conv_layer(model)
    if conv_layer is None:
        raise RuntimeError("No convolutional feature layer was located.")

    # The layer output must be connected to the model input graph.
    try:
        grad_model = keras.Model(
            inputs=model.inputs[0] if model_name == "DANN" else model.inputs,
            outputs=[conv_layer.output, model.outputs[0] if isinstance(model.outputs, list)
                     else model.output]
        )
    except Exception as exc:
        raise RuntimeError(
            "Selected convolutional layer is not connected to the top-level "
            "model graph (common with nested application backbones). "
            "No Grad-CAM map was generated."
        ) from exc

    x = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x, training=False)
        class_idx = tf.argmax(predictions[0])
        score = predictions[:, class_idx]
    grads = tape.gradient(score, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradient could not be connected to selected feature map.")

    if method == "Grad-CAM":
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.reduce_sum(weights * conv_outputs, axis=-1)
    else:
        # Grad-CAM++ positive gradients and second/third-order approximations.
        grads2 = tf.square(grads)
        grads3 = grads2 * grads
        sum_activations = tf.reduce_sum(conv_outputs, axis=(1, 2), keepdims=True)
        denom = 2.0 * grads2 + grads3 * sum_activations
        alpha = grads2 / (denom + 1e-8)
        alpha = tf.where(tf.math.is_finite(alpha), alpha, tf.zeros_like(alpha))
        positive_grads = tf.nn.relu(grads)
        weights = tf.reduce_sum(alpha * positive_grads, axis=(1, 2), keepdims=True)
        cam = tf.reduce_sum(weights * conv_outputs, axis=-1)

    cam = tf.nn.relu(cam)[0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], [IMG_SIZE, IMG_SIZE]).numpy().squeeze()
    return cam, int(class_idx.numpy()), predictions.numpy()[0]


def save_cam_panel(image, cam, model_name, method, predicted, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.clip(image / 255.0, 0, 1))
    axes[0].set_title("Original MRI")
    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(method)
    axes[2].imshow(np.clip(image / 255.0, 0, 1))
    axes[2].imshow(cam, cmap="jet", alpha=0.40, vmin=0, vmax=1)
    axes[2].set_title(f"Overlay | Predicted: {CLASS_NAMES[predicted]}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{model_name} — {method}", fontweight="bold")
    save_fig(out_path)


def run_lime(model, image, model_name, out_path, num_samples=1200):
    """
    LIME with SLIC superpixels constrained to the visible head foreground.
    Positive and negative evidence are saved separately.
    """
    try:
        from lime import lime_image
        from skimage.segmentation import slic, mark_boundaries
    except ImportError:
        print("LIME unavailable. Install: pip install lime scikit-image")
        return

    head_mask = get_head_foreground_mask(image)

    def segmentation_fn(rgb_image):
        seg = slic(
            rgb_image,
            n_segments=90,
            compactness=0.12,
            sigma=1.0,
            start_label=1,
            channel_axis=-1,
            mask=head_mask
        )
        # Keep the excluded background as a single region.
        seg = np.asarray(seg, dtype=np.int32)
        seg[~head_mask] = 0
        return seg

    def predict_fn(batch):
        batch = np.asarray(batch, dtype=np.float32)
        return safe_model_predict(model, batch, model_name)

    explainer = lime_image.LimeImageExplainer(random_state=SEED)
    explanation = explainer.explain_instance(
        image.astype(np.uint8),
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=segmentation_fn
    )

    label = int(explanation.top_labels[0])
    _, positive_mask = explanation.get_image_and_mask(
        label, positive_only=True, num_features=12, hide_rest=False
    )
    _, negative_mask = explanation.get_image_and_mask(
        label, positive_only=False, negative_only=True,
        num_features=12, hide_rest=False
    )

    positive_mask = (positive_mask > 0) & head_mask
    negative_mask = (negative_mask > 0) & head_mask
    base = np.clip(image / 255.0, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(base)
    axes[0].set_title("Original MRI")
    axes[1].imshow(base)
    pos_overlay = np.zeros((*head_mask.shape, 4), dtype=np.float32)
    pos_overlay[..., 1] = 1.0
    pos_overlay[..., 3] = positive_mask.astype(np.float32) * 0.48
    axes[1].imshow(pos_overlay)
    axes[1].contour(positive_mask, levels=[0.5], colors=["yellow"], linewidths=0.7)
    axes[1].set_title("Positive evidence (green)")
    axes[2].imshow(base)
    neg_overlay = np.zeros((*head_mask.shape, 4), dtype=np.float32)
    neg_overlay[..., 0] = 1.0
    neg_overlay[..., 3] = negative_mask.astype(np.float32) * 0.48
    axes[2].imshow(neg_overlay)
    axes[2].contour(negative_mask, levels=[0.5], colors=["yellow"], linewidths=0.7)
    axes[2].set_title("Negative evidence (red)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{model_name} — LIME | predicted: {CLASS_NAMES[label]}")
    save_fig(out_path)


def run_shap(model, image, model_name, out_dir, background_count=12):
    try:
        import shap
    except ImportError:
        print("SHAP unavailable. Install: pip install shap")
        return

    # KernelExplainer is model-agnostic but computationally expensive.
    # Use a small deterministic background and coarse superpixel features.
    # We explain a downsampled input to limit runtime; interpretation is
    # regional/coarse, not pixel-perfect.
    small = tf.image.resize(image[None, ...], [56, 56]).numpy()
    background = np.repeat(small, background_count, axis=0)
    x = small.copy()

    def predict_small(batch):
        batch = np.asarray(batch, dtype=np.float32)
        batch = tf.image.resize(batch, [IMG_SIZE, IMG_SIZE]).numpy()
        return safe_model_predict(model, batch, model_name)

    try:
        explainer = shap.KernelExplainer(predict_small, background)
        shap_values = explainer.shap_values(x, nsamples=100)
        # Multi-class return forms differ by SHAP version.
        if isinstance(shap_values, list):
            values = np.mean(np.abs(np.asarray(shap_values)), axis=0)
        else:
            values = np.mean(np.abs(np.asarray(shap_values)), axis=-1)
        if values.ndim == 4:
            values = values.mean(axis=-1)
        if values.ndim == 3:
            values = values[0]
        plt.figure(figsize=(6, 6))
        plt.imshow(x[0].mean(axis=-1), cmap="gray")
        plt.imshow(values.squeeze(), cmap="magma", alpha=0.55)
        plt.title(f"{model_name} — SHAP (coarse 56×56)")
        plt.axis("off")
        save_fig(out_dir / f"{model_name.lower()}_shap.png")
    except Exception as e:
        print(f"SHAP skipped for {model_name}: {e}")


def run_occlusion(model, image, model_name, out_path, patch=16, stride=8):
    """
    Fine-grid occlusion sensitivity. Only pixels inside the approximate
    head foreground are replaced. Positive values mean occlusion reduced
    the target-class probability.
    """
    base_pred = safe_model_predict(model, image[None, ...], model_name)[0]
    target = int(np.argmax(base_pred))
    h, w = image.shape[:2]
    head_mask = get_head_foreground_mask(image)
    sensitivity = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    batch, positions = [], []
    fill_value = np.mean(image[head_mask], axis=0) if np.any(head_mask) else np.mean(image, axis=(0, 1))

    def evaluate_batch(batch_items, coords):
        probs = safe_model_predict(model, np.asarray(batch_items), model_name)
        for (yy, xx, region), p in zip(coords, probs):
            drop = float(base_pred[target] - p[target])
            local = sensitivity[yy:yy+patch, xx:xx+patch]
            local_count = counts[yy:yy+patch, xx:xx+patch]
            local[region] += drop
            local_count[region] += 1

    for y in range(0, h - patch + 1, stride):
        for x in range(0, w - patch + 1, stride):
            local_mask = head_mask[y:y+patch, x:x+patch]
            if local_mask.mean() < 0.35:
                continue
            altered = image.copy()
            patch_view = altered[y:y+patch, x:x+patch, :]
            patch_view[local_mask] = fill_value
            batch.append(altered)
            positions.append((y, x, local_mask.copy()))
            if len(batch) >= 32:
                evaluate_batch(batch, positions)
                batch, positions = [], []
    if batch:
        evaluate_batch(batch, positions)

    sensitivity /= np.maximum(counts, 1)
    sensitivity[~head_mask] = 0.0
    denom = np.max(np.abs(sensitivity)) + 1e-8
    normalized = sensitivity / denom

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(np.clip(image / 255.0, 0, 1))
    axes[0].set_title("Original MRI")
    axes[1].imshow(normalized, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title("Occlusion sensitivity")
    axes[2].imshow(np.clip(image / 255.0, 0, 1))
    axes[2].imshow(normalized, cmap="coolwarm", alpha=0.48, vmin=-1, vmax=1)
    axes[2].set_title(f"Target: {CLASS_NAMES[target]}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"{model_name} — Occlusion | patch={patch}, stride={stride}",
        fontweight="bold"
    )
    save_fig(out_path)


def select_representative_images(test_df, per_class=2):
    chosen = []
    for cls in CLASS_NAMES:
        subset = test_df[test_df.label == cls].sort_values("path")
        if subset.empty:
            continue
        chosen.extend(subset.head(per_class).to_dict("records"))
    return chosen


def main():
    parser = argparse.ArgumentParser(
        description="Post-training XAI for saved brain tumor MRI models (no retraining)"
    )
    parser.add_argument("--data_dir", required=True,
                        help="Dataset root with explicit test/Test folder")
    parser.add_argument("--models_dir", required=True,
                        help="Folder containing *_final.keras model files")
    parser.add_argument("--output_dir", default=None,
                        help="XAI output folder; defaults to <models_dir parent>/xai_extended")
    parser.add_argument("--models", nargs="+", default=[
        "CustomCNN", "VGG16", "ResNet50", "EfficientNetB0",
        "DenseNet121", "MobileNetV2", "InceptionV3", "Xception"
    ])
    parser.add_argument("--per_class", type=int, default=2,
                        help="Number of test MRI cases per class")
    parser.add_argument("--methods", nargs="+",
                        default=["GradCAM", "GradCAM++", "LIME", "Occlusion"],
                        choices=["GradCAM", "GradCAM++", "LIME", "SHAP", "Occlusion"])
    parser.add_argument("--lime_samples", type=int, default=800)
    args = parser.parse_args()

    models_dir = Path(args.models_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        models_dir.parent / "xai_extended"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    test_df = discover_test_images(args.data_dir)
    samples = select_representative_images(test_df, args.per_class)
    pd.DataFrame(samples).to_csv(output_dir / "selected_test_cases.csv", index=False)

    print(f"Selected {len(samples)} held-out test images across classes.")
    summary_rows = []

    for model_name in args.models:
        model_path = models_dir / MODEL_FILES.get(model_name, "")
        if not model_path.exists():
            print(f"SKIP {model_name}: model file not found: {model_path}")
            continue

        print(f"\nLoading {model_name}: {model_path}")
        try:
            model = keras.models.load_model(model_path, compile=False, safe_mode=False)
        except Exception as e:
            print(f"Could not load {model_name}: {e}")
            continue

        model_out = output_dir / model_name.lower()
        model_out.mkdir(parents=True, exist_ok=True)

        for idx, record in enumerate(samples, start=1):
            image = load_image(record["path"])
            true_label = record["label"]
            prefix = f"{idx:02d}_{true_label}"

            try:
                probs = safe_model_predict(model, image[None, ...], model_name)[0]
                predicted = int(np.argmax(probs))
                confidence = float(probs[predicted])
                summary_rows.append({
                    "model": model_name,
                    "image": record["path"],
                    "true_class": true_label,
                    "predicted_class": CLASS_NAMES[predicted],
                    "confidence": confidence,
                    "correct": CLASS_NAMES[predicted] == true_label
                })
            except Exception as e:
                print(f"Prediction failed ({model_name}, {record['path']}): {e}")
                continue

            for method in args.methods:
                try:
                    if method in {"GradCAM", "GradCAM++"}:
                        cam, class_idx, _ = gradcam_family(
                            model, image, model_name,
                            "Grad-CAM" if method == "GradCAM" else "Grad-CAM++"
                        )
                        save_cam_panel(
                            image, cam, model_name,
                            "Grad-CAM" if method == "GradCAM" else "Grad-CAM++",
                            class_idx,
                            model_out / f"{prefix}_{method.lower().replace('+','plus')}.png"
                        )
                    elif method == "LIME":
                        run_lime(
                            model, image, model_name,
                            model_out / f"{prefix}_lime.png",
                            num_samples=args.lime_samples
                        )
                    elif method == "SHAP":
                        run_shap(model, image, model_name, model_out)
                        src = model_out / f"{model_name.lower()}_shap.png"
                        dst = model_out / f"{prefix}_shap.png"
                        if src.exists():
                            src.replace(dst)
                    elif method == "Occlusion":
                        run_occlusion(
                            model, image, model_name,
                            model_out / f"{prefix}_occlusion.png",
                            patch=16, stride=8
                        )
                except Exception as e:
                    print(f"  {method} failed for {model_name}/{prefix}: {e}")

        del model
        tf.keras.backend.clear_session()

    pd.DataFrame(summary_rows).to_csv(output_dir / "xai_prediction_summary.csv", index=False)
    print("\nXAI processing complete.")
    print("Output:", output_dir)
    print("Review xai_prediction_summary.csv and per-model folders.")
    print("Note: foreground masking suppresses obvious background only; it is NOT tumour segmentation.")
    print("Note: explanations are model-behavior visualizations, not clinical evidence.")


if __name__ == "__main__":
    main()
