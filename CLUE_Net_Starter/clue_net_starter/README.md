# CLUE-Net starter

## 1. Environment
Recommended for your TensorFlow 2.15.1 environment:

```bash
pip install tensorflow==2.15.1 scikit-learn matplotlib pandas
```

## 2. Dataset structure
The `--data_dir` folder must contain `Train`, `Validation`, and `Test`, each with:
`glioma`, `meningioma`, `pituitary`, `no_tumor`.

## 3. Run
Windows example:

```bash
python train_clue_net.py --data_dir "E:/Project/Brain Tumor/YOUR_CLEAN_SPLIT_FOLDER" --backbone DenseNet121 --img_size 224 --epochs 30 --batch_per_class 2
```

WSL example:

```bash
python train_clue_net.py --data_dir "/mnt/e/Project/Brain Tumor/YOUR_CLEAN_SPLIT_FOLDER" --backbone DenseNet121 --img_size 224 --epochs 30 --batch_per_class 2
```

For a 4 GB GPU, start with `--batch_per_class 2`. Each training batch contains 4 classes × 2 source images × 2 views = 16 images processed together. If out-of-memory occurs, try `--batch_per_class 1` (but contrastive positives still exist because two views are generated per source image).

## 4. Backbone options
`CustomCNN`, `VGG16`, `Xception`, `InceptionV3`, `ResNet50`, `DenseNet121`, `MobileNetV2`.

Use the correct image size for the backbone. Xception and InceptionV3 are commonly used at 299×299; DenseNet121, VGG16, ResNet50 and MobileNetV2 can use 224×224.

## 5. Outputs
The output folder contains:
- best weights selected using validation macro-F1
- epoch training log
- validation/test metrics JSON
- prediction CSV with class probabilities and uncertainty estimates
- confusion matrices
- classification reports
- run configuration

## 6. Research and validation cautions
- Verify patient-level independence and remove cross-split duplicates before training.
- Verify class folder names and class-index mapping.
- Review horizontal flipping for suitability to your MRI acquisition protocol.
- This starter estimates epistemic uncertainty using MC Dropout. Predictive entropy is total predictive uncertainty; it should not be described as a pure aleatoric estimate.
- Temperature scaling and calibration metrics are planned as a next implementation stage; they are not yet implemented in this starter.
- The test set should not be used for model selection or hyperparameter tuning.
- This is a research prototype, not a clinically validated diagnostic system.
