"""
CLUE-Net: Contrastive Learning with Uncertainty Estimation
TensorFlow/Keras starter implementation for four-class brain MRI classification.

Expected directory structure:
DATA_ROOT/
    Train/
        glioma/
        meningioma/
        pituitary/
        no_tumor/
    Val/  (or Validation/)
        glioma/
        meningioma/
        pituitary/
        no_tumor/
    Test/
        glioma/
        meningioma/
        pituitary/
        no_tumor/

Install (tested target: TensorFlow 2.15.x):
    pip install tensorflow==2.15.1 scikit-learn matplotlib pandas

Example:
    python train_clue_net.py --data_dir "E:/Project/Brain Tumor/cleaned_dataset" \
        --backbone DenseNet121 --img_size 224 --epochs 30 --batch_per_class 2

IMPORTANT:
- Set --data_dir to the parent directory containing Train, Validation, Test.
- The script does not automatically infer patient IDs. Ensure patient-level separation
  and duplicate removal have been performed before training.
- Use validation data for model selection/calibration. Test data is evaluated once at end.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, f1_score
)
from sklearn.preprocessing import label_binarize

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def list_class_files(split_dir, class_names):
    files, labels = [], []
    split_dir = Path(split_dir)
    for idx, cls in enumerate(class_names):
        folder = split_dir / cls
        if not folder.exists():
            raise FileNotFoundError(f"Missing class directory: {folder}")
        cls_files = sorted(
            str(p) for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        files.extend(cls_files)
        labels.extend([idx] * len(cls_files))
    if not files:
        raise RuntimeError(f"No images found under {split_dir}")
    return np.asarray(files), np.asarray(labels, dtype=np.int32)


def decode_raw_image(path, img_size):
    """Read and resize an image in the [0, 255] range (augmentation input)."""
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [img_size, img_size], antialias=True)
    return tf.cast(image, tf.float32)


def preprocess_image(image, backbone):
    """Apply the selected backbone's preprocessing after augmentation."""
    if backbone == "VGG16":
        return tf.keras.applications.vgg16.preprocess_input(image)
    if backbone == "Xception":
        return tf.keras.applications.xception.preprocess_input(image)
    if backbone == "InceptionV3":
        return tf.keras.applications.inception_v3.preprocess_input(image)
    if backbone == "ResNet50":
        return tf.keras.applications.resnet50.preprocess_input(image)
    if backbone == "DenseNet121":
        return tf.keras.applications.densenet.preprocess_input(image)
    if backbone == "MobileNetV2":
        return tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image / 127.5 - 1.0


def decode_image(path, img_size, backbone):
    return preprocess_image(decode_raw_image(path, img_size), backbone)


def make_augmenter(img_size):
    # Keep transformations modest to avoid altering clinically relevant morphology.
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.035, fill_mode="reflect"),
        layers.RandomZoom(0.08, fill_mode="reflect"),
        layers.RandomContrast(0.08),
    ], name="mri_augmentation")


def make_backbone(name, img_size):
    shape = (img_size, img_size, 3)
    if name == "VGG16":
        return keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=shape)
    if name == "Xception":
        return keras.applications.Xception(include_top=False, weights="imagenet", input_shape=shape)
    if name == "InceptionV3":
        return keras.applications.InceptionV3(include_top=False, weights="imagenet", input_shape=shape)
    if name == "ResNet50":
        return keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=shape)
    if name == "DenseNet121":
        return keras.applications.DenseNet121(include_top=False, weights="imagenet", input_shape=shape)
    if name == "MobileNetV2":
        return keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=shape)
    if name == "CustomCNN":
        inp = keras.Input(shape=shape)
        x = inp
        for filters in (32, 64, 128, 256):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D()(x)
        return keras.Model(inp, x, name="CustomCNN")
    raise ValueError(f"Unsupported backbone: {name}")


class CLUENet(keras.Model):
    def __init__(self, backbone, num_classes, embedding_dim=128, dropout_rate=0.3,
                 freeze_backbone=False, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.backbone.trainable = not freeze_backbone
        self.gap = layers.GlobalAveragePooling2D()
        self.projection = keras.Sequential([
            layers.Dense(512, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dropout(dropout_rate),
            layers.Dense(embedding_dim),
        ], name="projection_head")
        self.classifier_dropout = layers.Dropout(dropout_rate, name="classifier_mc_dropout")
        self.classifier = layers.Dense(num_classes, name="class_logits")

    def call(self, inputs, training=False, mc_dropout=False):
    # Keep pretrained backbone BN statistics stable during fine-tuning.
        if isinstance(inputs, (tuple, list)):
            inputs = tf.concat(inputs, axis=0)
        features = self.backbone(inputs, training=False)
        pooled = self.gap(features)
        embedding = self.projection(pooled, training=training)
        embedding = tf.math.l2_normalize(embedding, axis=1)
        cls_features = self.classifier_dropout(pooled, training=(training or mc_dropout))
        logits = self.classifier(cls_features)
        return {"logits": logits, "embedding": embedding}


def supervised_contrastive_loss(features, labels, temperature=0.07):
    """SupCon loss. Features are L2-normalized; labels contain repeated view labels."""
    features = tf.math.l2_normalize(features, axis=1)
    labels = tf.reshape(labels, [-1, 1])
    batch_size = tf.shape(features)[0]

    logits = tf.matmul(features, features, transpose_b=True) / temperature
    logits = logits - tf.reduce_max(logits, axis=1, keepdims=True)

    self_mask = tf.eye(batch_size, dtype=tf.float32)
    positive_mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)
    positive_mask = positive_mask * (1.0 - self_mask)
    exp_logits = tf.exp(logits) * (1.0 - self_mask)
    log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-12)

    positives_per_anchor = tf.reduce_sum(positive_mask, axis=1)
    mean_log_prob_pos = tf.reduce_sum(positive_mask * log_prob, axis=1) / tf.maximum(positives_per_anchor, 1.0)
    valid = tf.cast(positives_per_anchor > 0, tf.float32)
    return -tf.reduce_sum(mean_log_prob_pos * valid) / tf.maximum(tf.reduce_sum(valid), 1.0)


class ContrastiveTrainer(keras.Model):
    """Wraps CLUENet to train two augmented views jointly."""
    def __init__(self, network, supcon_weight=0.5, temperature=0.07, **kwargs):
        super().__init__(**kwargs)
        self.network = network
        self.supcon_weight = supcon_weight
        self.temperature = temperature
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.ce_tracker = keras.metrics.Mean(name="ce_loss")
        self.sc_tracker = keras.metrics.Mean(name="supcon_loss")
        self.acc_tracker = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [self.loss_tracker, self.ce_tracker, self.sc_tracker, self.acc_tracker]

    def call(self, inputs, training=False):
        return self.network(inputs, training=training)

    def train_step(self, data):
        (view1, view2), labels = data
        labels = tf.cast(labels, tf.int32)
        images = tf.concat([view1, view2], axis=0)
        labels_twice = tf.concat([labels, labels], axis=0)

        with tf.GradientTape() as tape:
            outputs = self.network(images, training=True)
            logits = outputs["logits"]
            embeddings = outputs["embedding"]
            ce_each = tf.keras.losses.sparse_categorical_crossentropy(
                labels_twice, logits, from_logits=True
            )
            ce_loss = tf.reduce_mean(ce_each)
            sc_loss = supervised_contrastive_loss(
                embeddings, labels_twice, self.temperature
            )
            reg_loss = tf.add_n(self.network.losses) if self.network.losses else 0.0
            total_loss = ce_loss + self.supcon_weight * sc_loss + reg_loss

        grads = tape.gradient(total_loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(
            (g, v) for g, v in zip(grads, self.network.trainable_variables) if g is not None
        )

        self.loss_tracker.update_state(total_loss)
        self.ce_tracker.update_state(ce_loss)
        self.sc_tracker.update_state(sc_loss)
        self.acc_tracker.update_state(labels_twice, logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Validation dataset contains one image and one label (not two augmented views).
        images, labels = data
        labels = tf.cast(labels, tf.int32)
        outputs = self.network(images, training=False)
        logits = outputs["logits"]
        ce_each = tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True
        )
        ce_loss = tf.reduce_mean(ce_each)
        reg_loss = tf.add_n(self.network.losses) if self.network.losses else 0.0
        total_loss = ce_loss + reg_loss

        self.loss_tracker.update_state(total_loss)
        self.ce_tracker.update_state(ce_loss)
        # No contrastive validation loss is computed for the single-view validation pipeline.
        self.sc_tracker.update_state(0.0)
        self.acc_tracker.update_state(labels, logits)
        return {m.name: m.result() for m in self.metrics}


class BalancedTwoViewSequence(keras.utils.Sequence):
    """Class-balanced batches: K classes x M images/class, each with two augmented views."""
    def __init__(self, files, labels, class_count, samples_per_class=2,
                 img_size=224, backbone="DenseNet121", batch_count=None, seed=42):
        self.files = np.asarray(files)
        self.labels = np.asarray(labels)
        self.class_count = class_count
        self.samples_per_class = samples_per_class
        self.img_size = img_size
        self.backbone_name = backbone
        self.augmenter = make_augmenter(img_size)
        self.rng = np.random.default_rng(seed)
        self.indices_by_class = {
            c: np.where(self.labels == c)[0] for c in range(class_count)
        }
        empty = [c for c, ids in self.indices_by_class.items() if len(ids) == 0]
        if empty:
            raise ValueError(f"No training samples for class indices: {empty}")
        self.batch_count = batch_count or max(1, math.ceil(len(files) / (class_count * samples_per_class)))
        self.on_epoch_end()

    def __len__(self):
        return self.batch_count

    def __getitem__(self, idx):
        selected, batch_labels = [], []
        for c in range(self.class_count):
            chosen = self.rng.choice(
                self.indices_by_class[c], size=self.samples_per_class,
                replace=len(self.indices_by_class[c]) < self.samples_per_class
            )
            selected.extend(chosen.tolist())
            batch_labels.extend([c] * self.samples_per_class)

        images = []
        for i in selected:
            img = decode_raw_image(tf.constant(self.files[i]), self.img_size)
            images.append(img.numpy())
        x = np.stack(images).astype(np.float32)
        y = np.asarray(batch_labels, dtype=np.int32)

        # Augment raw [0,255] pixels first, then apply backbone normalization.
        v1 = self.augmenter(x, training=True)
        v2 = self.augmenter(x, training=True)
        v1 = preprocess_image(v1, self.backbone_name).numpy()
        v2 = preprocess_image(v2, self.backbone_name).numpy()
        return (v1, v2), y

    def on_epoch_end(self):
        pass


def make_eval_dataset(files, labels, img_size, backbone, batch_size=16):
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    def _map(path, label):
        return decode_image(path, img_size, backbone), label
    return ds.map(_map, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)


def collect_predictions(network, dataset, mc_samples=1):
    all_y, all_probs, all_entropy, all_epi = [], [], [], []
    for images, labels in dataset:
        draws = []
        for _ in range(mc_samples):
            out = network(images, training=False, mc_dropout=(mc_samples > 1))
            probs = tf.nn.softmax(out["logits"], axis=1).numpy()
            draws.append(probs)
        draws = np.stack(draws, axis=0)
        mean_probs = draws.mean(axis=0)
        entropy = -np.sum(mean_probs * np.log(np.clip(mean_probs, 1e-8, 1.0)), axis=1)
        ent_each = -np.sum(draws * np.log(np.clip(draws, 1e-8, 1.0)), axis=2)
        epistemic = entropy - ent_each.mean(axis=0)
        all_y.extend(labels.numpy().tolist())
        all_probs.extend(mean_probs.tolist())
        all_entropy.extend(entropy.tolist())
        all_epi.extend(epistemic.tolist())
    return np.asarray(all_y), np.asarray(all_probs), np.asarray(all_entropy), np.asarray(all_epi)


def evaluate_and_save(network, dataset, class_names, out_dir, split_name, mc_samples=30):
    y_true, probs, entropy, epistemic = collect_predictions(network, dataset, mc_samples)
    y_pred = probs.argmax(axis=1)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names)), zero_division=0
    )
    metrics = {
        "split": split_name,
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[0]),
        "macro_recall": float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[1]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mean_predictive_entropy": float(np.mean(entropy)),
        "mean_epistemic_uncertainty": float(np.mean(epistemic)),
    }
    try:
        metrics["macro_roc_auc_ovr"] = float(
            roc_auc_score(y_true, probs, multi_class="ovr", average="macro",
                          labels=np.arange(len(class_names)))
        )
    except Exception:
        metrics["macro_roc_auc_ovr"] = None

    os.makedirs(out_dir, exist_ok=True)
    with open(Path(out_dir) / f"{split_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({
        "true_label": [class_names[i] for i in y_true],
        "predicted_label": [class_names[i] for i in y_pred],
        "predictive_entropy": entropy,
        "epistemic_uncertainty": epistemic,
        **{f"prob_{c}": probs[:, i] for i, c in enumerate(class_names)}
    }).to_csv(Path(out_dir) / f"{split_name}_predictions.csv", index=False)
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names))),
                 index=class_names, columns=class_names).to_csv(
                     Path(out_dir) / f"{split_name}_confusion_matrix.csv")
    with open(Path(out_dir) / f"{split_name}_classification_report.txt", "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print(f"\n{split_name.upper()} METRICS")
    print(json.dumps(metrics, indent=2))
    return metrics


class ValidationMacroF1(keras.callbacks.Callback):
    """Selects checkpoints using deterministic validation macro-F1 (single-pass predictions)."""
    def __init__(self, network, val_ds, class_names, checkpoint_path):
        super().__init__()
        self.network = network
        self.val_ds = val_ds
        self.class_names = class_names
        self.checkpoint_path = checkpoint_path
        self.best = -1.0

    def on_epoch_end(self, epoch, logs=None):
        y_true, probs, _, _ = collect_predictions(self.network, self.val_ds, mc_samples=1)
        score = f1_score(y_true, probs.argmax(axis=1), average="macro", zero_division=0)
        logs = logs or {}
        logs["val_macro_f1"] = score
        print(f" — val_macro_f1: {score:.4f}")
        if score > self.best:
            self.best = score
            self.network.save_weights(self.checkpoint_path)
            print(f"Saved best checkpoint: {self.checkpoint_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Parent directory containing Train, Val (or Validation), and Test")
    parser.add_argument("--backbone", default="DenseNet121",
                        choices=["CustomCNN", "VGG16", "Xception", "InceptionV3", "ResNet50", "DenseNet121", "MobileNetV2"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_per_class", type=int, default=2)
    parser.add_argument("--supcon_weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--mc_samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="CLUE_Net_Results")
    args = parser.parse_args()

    set_seed(args.seed)
    class_names = ["glioma", "meningioma", "pituitary", "notumor"]
    root = Path(args.data_dir)
    train_files, train_labels = list_class_files(root / "Train", class_names)
    val_dir = root / "Validation"
    if not val_dir.is_dir():
        val_dir = root / "Val"
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Could not find validation folder. Expected {root / 'Val'} or {root / 'Validation'}")
    val_files, val_labels = list_class_files(val_dir, class_names)
    test_files, test_labels = list_class_files(root / "Test", class_names)

    print("Dataset counts:", {
        "train": len(train_files), "validation": len(val_files), "test": len(test_files)
    })
    print("Class mapping:", {c: i for i, c in enumerate(class_names)})

    # Mixed precision is deliberately not enabled by default for stable contrastive loss.
    backbone = make_backbone(args.backbone, args.img_size)
    network = CLUENet(
        backbone, num_classes=len(class_names), embedding_dim=128,
        dropout_rate=0.3, freeze_backbone=args.freeze_backbone, name="CLUE_Net"
    )
    # Build model before optimizer/checkpoint operations.
    network(tf.zeros((1, args.img_size, args.img_size, 3)), training=False)

    trainer = ContrastiveTrainer(
        network, supcon_weight=args.supcon_weight, temperature=args.temperature
    )
    trainer.compile(optimizer=keras.optimizers.AdamW(
        learning_rate=args.learning_rate, weight_decay=1e-4
    ))

    train_seq = BalancedTwoViewSequence(
        train_files, train_labels, len(class_names),
        samples_per_class=args.batch_per_class, img_size=args.img_size,
        backbone=args.backbone, seed=args.seed
    )
    val_ds = make_eval_dataset(val_files, val_labels, args.img_size, args.backbone)
    test_ds = make_eval_dataset(test_files, test_labels, args.img_size, args.backbone)

    os.makedirs(args.output_dir, exist_ok=True)
    best_weights = str(Path(args.output_dir) / "best_clue_net.weights.h5")
    callbacks = [
        ValidationMacroF1(network, val_ds, class_names, best_weights),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, mode="min", restore_best_weights=False
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.CSVLogger(str(Path(args.output_dir) / "training_log.csv"))
    ]

    #trainer.fit(train_seq, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    # trainer.fit(train_seq, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    backup_weights = str(Path(args.output_dir) / "epoch9_backup.weights.h5")
    if os.path.exists(backup_weights):
        network.load_weights(backup_weights)
        print(f"Loaded saved weights: {backup_weights}")
    elif os.path.exists(best_weights):
        network.load_weights(best_weights)
        print(f"Loaded saved weights: {best_weights}")
    else:
        raise FileNotFoundError("No saved checkpoint weights found!")

    if os.path.exists(best_weights):
        network.load_weights(best_weights)
    else:
        print("WARNING: no best checkpoint was saved; evaluating current weights.")

    evaluate_and_save(network, val_ds, class_names, args.output_dir, "validation", args.mc_samples)
    evaluate_and_save(network, test_ds, class_names, args.output_dir, "test", args.mc_samples)

    config = vars(args)
    config["class_names"] = class_names
    with open(Path(args.output_dir) / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nCompleted. Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
