# 🧠 Datascience Practice Notebooks

A growing, hands-on library of the **major AI / machine-learning algorithms** — each in its own self-contained notebook, on a **real public dataset**, with full EDA → model → evaluation → findings. Every notebook runs end-to-end in **Google Colab** (just click the badge) and every result below was produced by actually running the code.

> **One-click run:** click any **Open in Colab** badge → `Runtime → Run all`. No setup, no downloads to manage — datasets load straight from `scikit-learn` / `keras`.

---

## 📓 The notebooks

### Supervised learning — Regression
| # | Notebook | Algorithm | Dataset | Verified result |
|---|----------|-----------|---------|-----------------|
| 01 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kader-xai/Datascience-Practice-Notebooks/blob/main/01_Linear_Regression_California_Housing.ipynb) [Linear Regression](01_Linear_Regression_California_Housing.ipynb) | Linear / Ridge vs ensembles | California Housing (20,640 districts) | Linear R²=0.58 → Random Forest **R²=0.81** |

### Supervised learning — Classification
| # | Notebook | Algorithm | Dataset | Verified result |
|---|----------|-----------|---------|-----------------|
| 02 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kader-xai/Datascience-Practice-Notebooks/blob/main/02_Logistic_Regression_Breast_Cancer.ipynb) [Logistic Regression](02_Logistic_Regression_Breast_Cancer.ipynb) | Logistic Regression | Breast Cancer Wisconsin (569) | **98.2%** acc, AUC **0.995** |
| 03 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kader-xai/Datascience-Practice-Notebooks/blob/main/03_Decision_Tree_and_Random_Forest_Wine.ipynb) [Decision Tree & Random Forest](03_Decision_Tree_and_Random_Forest_Wine.ipynb) | Trees, bagging | Wine (178, 3 cultivars) | Tree 95.6% → Forest **100%** |
| 04 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kader-xai/Datascience-Practice-Notebooks/blob/main/04_SVM_Handwritten_Digits.ipynb) [Support Vector Machine](04_SVM_Handwritten_Digits.ipynb) | SVM + kernel trick | Digits (1,797 8×8 images) | Linear 98.0% → RBF **98.2%** |
| 05 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kader-xai/Datascience-Practice-Notebooks/blob/main/05_KNN_Iris.ipynb) [K-Nearest Neighbors](05_KNN_Iris.ipynb) | KNN (instance-based) | Iris (150, 3 species) | k=14 → **95.6%** acc |

### Unsupervised learning
| # | Notebook | Algorithm | Dataset | Verified result |
|---|----------|-----------|---------|-----------------|
| 06 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kader-xai/Datascience-Practice-Notebooks/blob/main/06_KMeans_Clustering_Wine.ipynb) [K-Means Clustering](06_KMeans_Clustering_Wine.ipynb) | K-Means | Wine (178) | k=3, recovers cultivars **ARI 0.90** |
| 07 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kader-xai/Datascience-Practice-Notebooks/blob/main/07_PCA_Dimensionality_Reduction_Digits.ipynb) [PCA](07_PCA_Dimensionality_Reduction_Digits.ipynb) | Dimensionality reduction | Digits (1,797) | 40 comps = 95% variance, ~same accuracy |

### Deep learning
| # | Notebook | Algorithm | Dataset | Verified result |
|---|----------|-----------|---------|-----------------|
| 08 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kader-xai/Datascience-Practice-Notebooks/blob/main/08_Neural_Network_MNIST.ipynb) [Neural Network (MLP)](08_Neural_Network_MNIST.ipynb) | Feed-forward net | MNIST (70,000) | ~**98%** test acc |
| — | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kader-xai/Datascience-Practice-Notebooks/blob/main/Cat_vs_Dog_Pipeline.ipynb) [CNN — Cat vs Dog (full pipeline)](Cat_vs_Dog_Pipeline.ipynb) | CNN + transfer learning | Kaggle Dogs vs. Cats (25,000) | classical ~60% → transfer learning **~97–99%** |

---

## 🗺️ How to read the series

Each notebook follows the same shape so concepts transfer:
**problem framing → load real data → EDA → preprocess → baseline + focal model → evaluation with real metrics → plain-English findings → "try it yourself"**.

The **Cat vs Dog** notebook is the capstone: it threads one task (image → vector → PCA → EDA → Bayes → SVM → CNN → deployment) through every topic at once.

## 🧩 Algorithm map (regression vs classification vs unsupervised)
- **Regression** predicts a *continuous number* → notebook 01.
- **Classification** predicts a *category* → notebooks 02–05, 08, Cat vs Dog. (Note: *logistic* regression is a classification method despite its name.)
- **Unsupervised** finds structure with *no labels* → notebooks 06 (clustering), 07 (dimensionality reduction).

## 🔜 Coming next (Batch 2)
Naive Bayes (text classification) · Gradient Boosting (XGBoost-style) · RNN/LSTM (time series) · Transformer (NLP sentiment) · Q-Learning (reinforcement learning) · Anomaly detection.

---

*Every metric in this README was measured by executing the notebook's own cells — not estimated.*
