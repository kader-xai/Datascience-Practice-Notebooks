# Datascience Practice Notebooks

Hands-on notebooks that work through core machine-learning concepts on real, public datasets.

## Notebooks

### 🐱🐶 [Cat vs Dog — One Task Through the Whole Pipeline](Cat_vs_Dog_Pipeline.ipynb)

A single concrete task — classifying a photo as **cat** or **dog** — traced through the *entire* ML pipeline, with runnable code and real numbers at every step.

**Problem type:** binary image classification (not regression).
**Dataset:** the real, public **Kaggle "Dogs vs. Cats"** competition images (25,000 photos), loaded in one line via TensorFlow Datasets (`cats_vs_dogs`) — no Kaggle account needed. A CIFAR-10 fallback is built in for an instant, download-free run.

| Step | Topic |
|------|-------|
| 1. Raw image → feature vector | Linear algebra (vectors & matrices) |
| 2. Reduce dimensions ("eigen-cats") | Eigen analysis / PCA |
| 3. Inspect & scale the data | EDA (mean, std, balance, normalisation) |
| 4. Pick a paradigm & fit a model | Supervised learning + probability |
| 5. Classify a new image | Bayes' rule + Gaussian density |
| 6. Hit overfitting, then upgrade | SVM kernels → CNN → transfer learning |
| 7. Ship it | Deployment / productionization |

**Verified results** (on real photos): Naive Bayes ≈ 58%, kernel SVM ≈ 62%, transfer-learning CNN (MobileNetV2) ≈ 97–99% — the honest gap that shows why deep learning wins on natural images.

#### How to run
1. Open in [Google Colab](https://colab.research.google.com) (File → Upload notebook).
2. `Runtime → Change runtime type → GPU` (for Step 6).
3. `Runtime → Run all`.

For an instant trial without the ~800 MB download, set `USE_CIFAR_FALLBACK = True` in the first code cell.
