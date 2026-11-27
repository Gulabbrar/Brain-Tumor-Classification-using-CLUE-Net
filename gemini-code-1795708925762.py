import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input

CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]


def build_clue_net(num_classes=4):
    inp = layers.Input(shape=(224, 224, 3), name="mri_input")
    backbone = DenseNet121(include_top=False, weights="imagenet", input_tensor=inp)
    gap = layers.GlobalAveragePooling2D(name="gap")(backbone.output)

    # Contrastive Projection Head
    proj = layers.Dense(512, name="proj_dense1")(gap)
    proj = layers.BatchNormalization(name="proj_bn")(proj)
    proj = layers.ReLU(name="proj_relu")(proj)
    proj = layers.Dropout(0.3, name="proj_drop")(proj)
    proj = layers.Dense(128, name="proj_dense2")(proj)
    proj_norm = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), name="proj_l2")(proj)

    # Stochastic Classification Head
    mc_drop = layers.Dropout(0.3, name="mc_dropout")(gap)
    logits = layers.Dense(num_classes, name="cls_logits")(mc_drop)

    return Model(inputs=inp, outputs=[logits, proj_norm], name="CLUE_Net")


def mc_predict_and_gradcam(model, img_batch, last_conv_name="conv5_block16_concat", mc_samples=30):
    # 1. MC Dropout forward passes (T=30)
    probs_list = []
    for _ in range(mc_samples):
        logits, _ = model(img_batch, training=True)  # keeps Dropout(0.3) active
        probs_list.append(tf.nn.softmax(logits, axis=-1).numpy()[0])

    probs_arr = np.stack(probs_list, axis=0)  # [T, 4]
    mean_probs = np.mean(probs_arr, axis=0)
    pred_idx = int(np.argmax(mean_probs))

    eps = 1e-12
    pred_entropy = -np.sum(mean_probs * np.log(mean_probs + eps))
    expected_entropy = -np.mean(np.sum(probs_arr * np.log(probs_arr + eps), axis=-1))
    epistemic_unc = max(0.0, pred_entropy - expected_entropy)

    # 2. Grad-CAM from last DenseNet-121 feature map
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_name).output, model.outputs[0]]
    )
    with tf.GradientTape() as tape:
        conv_out, logits_det = grad_model(img_batch, training=False)
        class_channel = logits_det[:, pred_idx]

    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    return pred_idx, mean_probs, pred_entropy, epistemic_unc, heatmap


def save_overlay(orig_rgb, heatmap, out_path, title_text):
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.6 * orig_rgb + 0.4 * jet_rgb)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Input MRI")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Activation")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("CLUE-Net Localization")
    axes[2].axis("off")

    plt.suptitle(title_text, fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out_dir", default="Results/CLUE_Net/XAI")
    parser.add_argument("--mc_samples", type=int, default=30)
    parser.add_argument("--num_per_class", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = build_clue_net(num_classes=len(CLASSES))
    model.load_weights(args.weights)

    test_dir = os.path.join(args.data_dir, "Test")
    for cls in CLASSES:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = sorted(os.listdir(cls_dir))[: args.num_per_class]
        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            orig_rgb = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB)
            inp_tensor = preprocess_input(np.expand_dims(orig_rgb.astype(np.float32), axis=0))

            pred_idx, mean_probs, ent, epi, heatmap = mc_predict_and_gradcam(
                model, inp_tensor, mc_samples=args.mc_samples
            )
            pred_cls = CLASSES[pred_idx]
            conf = mean_probs[pred_idx]
            title = (
                f"True: {cls} | Pred: {pred_cls} (p={conf:.3f}) | "
                f"Entropy={ent:.4f} | Epistemic={epi:.5f}"
            )
            out_file = os.path.join(args.out_dir, f"{cls}_{os.path.splitext(fname)[0]}_xai.png")
            save_overlay(orig_rgb, heatmap, out_file, title)
            print("Saved:", out_file)


if __name__ == "__main__":
    main()