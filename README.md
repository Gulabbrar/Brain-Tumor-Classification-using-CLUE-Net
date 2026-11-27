# Brain-Tumor-Classification-using-CLUE-Net
Contrastive Learning and Uncertainty Estimation Network for Explainable Four-Class Brain Tumor MRI Classification using DenseNet-121, Supervised Contrastive Loss, Monte Carlo Dropout ($T=30$), and Grad-CAM.
# CLUE-Net: Contrastive Learning and Uncertainty Estimation Network for Brain Tumor MRI Classification

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow / Keras](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**CLUE-Net** is an explainable, uncertainty-aware deep learning framework built on a **DenseNet-121** backbone for four-class brain magnetic resonance imaging (MRI) classification (**Glioma**, **Meningioma**, **Pituitary Tumor**, and **Non-Tumor**). By unifying **two-view Supervised Contrastive Learning ($\mathcal{L}_{\text{SupCon}}$)** with **Monte Carlo Dropout Bayesian inference ($T = 30$)** and **Grad-CAM spatial localization**, CLUE-Net sharpens inter-class boundaries between radiographically similar tumors while flagging ambiguous predictions for radiologist review.

---

## Key Highlights

- **Supervised Contrastive Feature Clustering:** Pairs class-balanced two-view MRI augmentation with a 128-D $\ell_2$-normalized projection head ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{SupCon}}$), reducing off-diagonal glioma–meningioma confusion and boosting standalone DenseNet-121 accuracy by **+4.89%**.
- **Calibrated Bayesian Uncertainty Quantification:** Executes $T = 30$ stochastic forward passes at test time to decompose predictive entropy into **aleatoric** (data noise) and **epistemic** (model ignorance) uncertainty.
- **Automated Error Separation:** Misclassified held-out test scans ($n = 85$) exhibit a **13.9$\times$ higher mean predictive entropy** ($0.3180$ vs. $0.0228$) and **12.9$\times$ higher epistemic mutual information** ($0.01772$ vs. $0.00137$) than correctly classified scans ($n = 2{,}273$).
- **Dual Visual Explainability:** Generates three-panel diagnostic visualizations (**Original MRI | Grad-CAM Saliency Map | MC-Dropout Spatial Uncertainty Map**) for clinically transparent triage.

---

## Architecture Overview

```text
Input MRI (224x224x3)
       │
       ├──► Stochastic View 1 (T1) ──┐
       └──► Stochastic View 2 (T2) ──┴──► DenseNet-121 Backbone ──► Global Average Pooling (1024-D)
                                                                            │
                                            ┌───────────────────────────────┴───────────────────────────────┐
                                            ▼                                                               ▼
                             Classification Head (Dropout p=0.3)                            Projection Head (MLP -> 128-D L2)
                                            │                                                               │
                                   Cross-Entropy (L_CE)                                        Supervised Contrastive (L_SupCon)
                                            │
                          Inference: MC Dropout (T = 30 passes)
                         ├──► Calibrated Probability & Class Prediction
                         ├──► Predictive Entropy & Epistemic Uncertainty
                         └──► Grad-CAM + Spatial Variance Heatmap
