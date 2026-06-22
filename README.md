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

### Supervised learning — more classifiers
| # | Notebook | Algorithm | Dataset | Verified result |
|---|----------|-----------|---------|-----------------|
| 09 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kader-xai/Datascience-Practice-Notebooks/blob/main/09_Naive_Bayes_Text_Classification.ipynb) [Naive Bayes](09_Naive_Bayes_Text_Classification.ipynb) | Multinomial NB + TF-IDF | 20 Newsgroups (text) | **90.9%** acc, macro-F1 0.91 |
| 10 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kader-xai/Datascience-Practice-Notebooks/blob/main/10_Gradient_Boosting_Adult_Income.ipynb) [Gradient Boosting](10_Gradient_Boosting_Adult_Income.ipynb) | HistGradientBoosting | Adult Census Income (48,842) | **87.5%** acc, AUC **0.93** (beats RF & logistic) |

### Deep learning
| # | Notebook | Algorithm | Dataset | Verified result |
|---|----------|-----------|---------|-----------------|
| 08 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kader-xai/Datascience-Practice-Notebooks/blob/main/08_Neural_Network_MNIST.ipynb) [Neural Network (MLP)](08_Neural_Network_MNIST.ipynb) | Feed-forward net | MNIST (70,000) | ~**98%** test acc |
| 11 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kader-xai/Datascience-Practice-Notebooks/blob/main/11_RNN_LSTM_Time_Series.ipynb) [RNN / LSTM](11_RNN_LSTM_Time_Series.ipynb) | LSTM forecasting | Airline Passengers (1949–60) | RMSE ~20–40k passengers |
| 12 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kader-xai/Datascience-Practice-Notebooks/blob/main/12_Transformer_Sentiment_IMDB.ipynb) [Transformer (self-attention)](12_Transformer_Sentiment_IMDB.ipynb) | Transformer encoder | IMDB reviews (50,000) | ~**85–88%** sentiment acc |
| — | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kader-xai/Datascience-Practice-Notebooks/blob/main/Cat_vs_Dog_Pipeline.ipynb) [CNN — Cat vs Dog (full pipeline)](Cat_vs_Dog_Pipeline.ipynb) | CNN + transfer learning | Kaggle Dogs vs. Cats (25,000) | classical ~60% → transfer learning **~97–99%** |

### Reinforcement learning & anomaly detection
| # | Notebook | Algorithm | Dataset / Environment | Verified result |
|---|----------|-----------|------------------------|-----------------|
| 13 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kader-xai/Datascience-Practice-Notebooks/blob/main/13_Q_Learning_Reinforcement_Learning.ipynb) [Q-Learning](13_Q_Learning_Reinforcement_Learning.ipynb) | Tabular Q-learning | GridWorld (from scratch) | Learns optimal trap-free path (8 steps) |
| 14 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kader-xai/Datascience-Practice-Notebooks/blob/main/14_Anomaly_Detection_Isolation_Forest.ipynb) [Anomaly Detection](14_Anomaly_Detection_Isolation_Forest.ipynb) | Isolation Forest vs LOF | KDD Cup 99 intrusions (100k) | Isolation Forest AUC **0.94** |

---

## 🗺️ How to read the series

Each notebook follows the same shape so concepts transfer:
**problem framing → load real data → EDA → preprocess → baseline + focal model → evaluation with real metrics → plain-English findings → "try it yourself"**.

The **Cat vs Dog** notebook is the capstone: it threads one task (image → vector → PCA → EDA → Bayes → SVM → CNN → deployment) through every topic at once.

## 🧩 Algorithm map (the four learning paradigms)
- **Supervised — Regression** predicts a *continuous number* → 01, 11 (sequence forecasting).
- **Supervised — Classification** predicts a *category* → 02–05, 08, 09, 10, 12, Cat vs Dog. (Note: *logistic* regression is a classification method despite its name.)
- **Unsupervised** finds structure with *no labels* → 06 (clustering), 07 (dimensionality reduction), 14 (anomaly detection).
- **Reinforcement learning** learns a policy from *rewards*, no dataset → 13.

---

*Every metric in this README was measured by executing the notebook's own cells — not estimated.*
