import os
import argparse
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]


class CLUENet(tf.keras.Model):
    def __init__(self, num_classes=4, proj_dim=128, dropout_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        base = DenseNet121(include_top=False, weights=None, input_shape=(224, 224, 3))
        # Wrapped as a Functional model so HDF5 path matches layers\functional
        self.backbone = models.Model(inputs=base.input, outputs=base.output)
        self.gap = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(dropout_rate)
        self.classifier = layers.Dense(num_classes)

        # Nested Sequential projection head matching projection\layers\*
        self.projection = models.Sequential(
            [
                layers.Dense(512, use_bias=False),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(proj_dim),
            ],
            name="projection",
        )

    def call(self, x, training=False):
        feat_map = self.backbone(x, training=training)
        pooled = self.gap(feat_map)
        dropped = self.dropout(pooled, training=training)
        logits = self.classifier(dropped)
        proj = self.projection(pooled, training=training)
        proj_norm = tf.math.l2_normalize(proj, axis=-1)
        return logits, proj_norm


def compute_gradcam(model, img_tensor, class_idx):
    # Find the final spatial convolutional feature map inside self.backbone
    last_conv_layer = None
    for layer in reversed(model.backbone.layers):
        if len(layer.output_shape) == 4 and (
            isinstance(layer, layers.Conv2D) or "concat" in layer.name or "relu" in layer.name
        ):
            last_conv_layer = layer
            break

    grad_backbone = models.Model(
        inputs=model.backbone.inputs,
        outputs=[last_conv_layer.output, model.backbone.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, backbone_out = grad_backbone(img_tensor, training=False)
        tape.watch(conv_outputs)
        pooled = model.gap(backbone_out)
        dropped = model.dropout(pooled, training=False)
        logits = model.classifier(dropped)
        target_score = logits[:, class_idx]

    grads = tape.gradient(target_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def mc_dropout_predict(model, img_tensor, mc_samples=30):
    # Repeat image across batch so all T stochastic forward passes run in one vectorized call
    batch_imgs = tf.repeat(img_tensor, repeats=mc_samples, axis=0)
    logits, _ = model(batch_imgs, training=True)
    probs = tf.nn.softmax(logits, axis=-1).numpy()  # [T, C]

    mean_probs = np.mean(probs, axis=0)
    pred_idx = int(np.argmax(mean_probs))
    confidence = float(mean_probs[pred_idx])

    # Predictive entropy H(y|x) and epistemic mutual information MI(y, w|x)
    eps = 1e-10
    pred_entropy = float(-np.sum(mean_probs * np.log(mean_probs + eps)))
    expected_entropy = float(np.mean(-np.sum(probs * np.log(probs + eps), axis=-1)))
    epistemic_mi = max(0.0, pred_entropy - expected_entropy)

    # Pixel-wise spatial variance across T stochastic feature maps
    feat_maps = model.backbone(batch_imgs, training=True).numpy()  # [T, 7, 7, 1024]
    spatial_var = np.mean(np.var(feat_maps, axis=0), axis=-1)      # [7, 7]
    spatial_var = spatial_var / (np.max(spatial_var) + 1e-8)

    return pred_idx, confidence, pred_entropy, epistemic_mi, spatial_var


def overlay_heatmap(rgb_img, heatmap, alpha=0.45):
    h, w, _ = rgb_img.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended = np.uint8(rgb_img * (1 - alpha) + colored_rgb * alpha)
    return blended


def resolve_test_dir(data_dir):
    for candidate in ["Testing", "test", "Test"]:
        full = os.path.join(data_dir, candidate)
        if os.path.isdir(full):
            return full
    raise FileNotFoundError(f"Could not find Testing/test folder inside {data_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mc_samples", type=int, default=30)
    parser.add_argument("--num_per_class", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    test_dir = resolve_test_dir(args.data_dir)

    # Build CLUE-Net and trigger variable creation before loading HDF5 weights
    model = CLUENet(num_classes=len(CLASSES))
    _ = model(tf.zeros((1, 224, 224, 3)), training=False)
    model.load_weights(args.weights)
    print(f"Successfully loaded CLUE-Net weights from: {args.weights}")

    records = []
    for true_idx, cls_name in enumerate(CLASSES):
        cls_folder = os.path.join(test_dir, cls_name)
        if not os.path.isdir(cls_folder):
            # Fallback for 'no_tumor' folder naming
            alt_folder = os.path.join(test_dir, "no_tumor")
            cls_folder = alt_folder if os.path.isdir(alt_folder) else cls_folder

        files = sorted(
            [f for f in os.listdir(cls_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )[: args.num_per_class]

        for fname in files:
            fpath = os.path.join(cls_folder, fname)
            bgr = cv2.imread(fpath)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_224 = cv2.resize(rgb, (224, 224))

            inp = preprocess_input(rgb_224.astype(np.float32))[None, ...]
            inp_tensor = tf.convert_to_tensor(inp)

            pred_idx, conf, entropy, epi_mi, spatial_var = mc_dropout_predict(
                model, inp_tensor, mc_samples=args.mc_samples
            )
            cam = compute_gradcam(model, inp_tensor, pred_idx)

            cam_overlay = overlay_heatmap(rgb_224, cam, alpha=0.45)
            unc_overlay = overlay_heatmap(rgb_224, spatial_var, alpha=0.45)

            # Side-by-side 3-panel figure: Original | Grad-CAM | Epistemic Uncertainty
            panel = np.concatenate([rgb_224, cam_overlay, unc_overlay], axis=1)
            out_name = f"{cls_name}_{os.path.splitext(fname)[0]}_xai.png"
            cv2.imwrite(os.path.join(args.out_dir, out_name), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

            records.append(
                {
                    "file": fname,
                    "true_class": cls_name,
                    "pred_class": CLASSES[pred_idx],
                    "confidence": round(conf, 4),
                    "predictive_entropy": round(entropy, 5),
                    "epistemic_mi": round(epi_mi, 5),
                    "xai_panel": out_name,
                }
            )
            print(
                f"[{cls_name} -> {CLASSES[pred_idx]}] {fname} | "
                f"Conf={conf:.4f}, Entropy={entropy:.4f}, Epistemic={epi_mi:.5f}"
            )

    summary_df = pd.DataFrame(records)
    summary_csv = os.path.join(args.out_dir, "xai_uncertainty_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved {len(summary_df)} 3-panel XAI visualizations and summary CSV to: {args.out_dir}")


if __name__ == "__main__":
    main()