# BankClassificationModels.

# Lab: Introduction to Classification Models
**Pontificial Xaverian University — Big Data Processing**

**Author:** Carlos Andrés Jr. Méndez Pachón  
**Date:** May 7th, 2026  
**Course:** Data Processing / Big Data Processing

---

## Overview

This lab explores the full pipeline for building binary classification models using **Apache Spark (PySpark)** on a bank marketing dataset (`bank-full.csv`). The goal is to predict whether a client will subscribe to a term deposit (`y` = yes/no), covering everything from data ingestion on an HDFS cluster to model evaluation and comparison.

---

## Objectives

- Initialize and configure a Spark session connected to a remote cluster
- Load data from HDFS into a Spark DataFrame
- Perform exploratory data analysis (EDA) and data cleaning
- Handle class imbalance via oversampling
- Encode categorical variables using One-Hot Encoding
- Train and evaluate five classification models
- Compare model performance across multiple metrics

---

## Dataset

- **Source:** `bank-full.csv` (UCI Bank Marketing Dataset), loaded via HDFS
- **Size:** ~45,000 records
- **Target variable:** `y` — whether the client subscribed a term deposit (`yes` / `no`)
- **Class imbalance:** ~11.6% `yes`, ~88.3% `no`

### Features

| Type | Columns |
|---|---|
| Numeric | `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous` |
| Categorical | `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome` |

---

## Pipeline

### 1. Environment Setup
- Libraries: `PySpark`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`
- Spark configured with FAIR scheduler and remote master at `spark://10.43.97.187:7077`

### 2. Data Loading
- Data read from HDFS (`hdfs://10.195.34.34:9000/csv/bank-full.csv`)
- All columns initially loaded as `String`; numeric columns cast to `Integer`

### 3. Exploratory Data Analysis (EDA)
- Class distribution analysis to identify imbalance
- Descriptive statistics for all variables
- Histograms and boxplots for numeric variables
- Bar charts and pie charts for categorical variables
- Correlation matrix using Pearson correlation (via PySpark ML)
- Cross-tabulation of categorical variables vs. target `y`

**Key findings:**
- Strong correlation between `duration` and `y` — excluded from realistic models (benchmark only)
- `pdays` and `previous` show multicollinearity
- `default` is highly skewed (98.2% "no")

### 4. Data Cleaning
- Removed records where `previous > 30` (outlier elimination)
- Dropped `pdays` column (multicollinearity + ~81% missing as `-1`)
- Applied **oversampling** on the minority class (`yes`) to balance the dataset

### 5. Feature Engineering
- **StringIndexer** + **OneHotEncoder** for all categorical variables
- **StringIndexer** for target label `y` → `label`
- **VectorAssembler** to combine all features into a single `features` vector
- Full pipeline saved as `pipeModel`; transformed data exported to `output.parquet`

### 6. Model Training & Evaluation
Train/test split: **80% / 20%** (seed = 4321), stratified balance verified.

---

## Models

| # | Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|---|---|---|---|---|---|---|
| 1 | Logistic Regression | ~0.83 | ~0.83 | ~0.83 | ~0.83 | ~0.90 |
| 2 | Decision Tree | ~0.82 | ~0.82 | ~0.82 | ~0.82 | ~0.82 |
| 3 | **Gradient Boosted Tree** | **~0.85** | **~0.85** | **~0.85** | **~0.85** | **~0.93** |
| 4 | Random Forest | ~0.82 | ~0.82 | ~0.82 | ~0.82 | ~0.88 |
| 5 | Support Vector Machine (LinearSVC) | ~0.83 | ~0.83 | ~0.83 | ~0.83 | ~0.89 |

> **Best model: Gradient Boosted Tree** — highest scores across all four metrics (>0.85) and AUC-ROC of ~0.93.

Each model is evaluated with:
- Confusion Matrix (heatmap)
- Precision, Recall, F1 Score, Accuracy (via `MulticlassClassificationEvaluator`)
- ROC Curve and AUC (via `BinaryClassificationEvaluator`)

---

## Dependencies

```
pyspark
findspark
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## How to Run

1. Ensure access to the Spark cluster and HDFS server configured in the notebook
2. Install dependencies: `pip install pyspark findspark pandas numpy matplotlib seaborn scikit-learn`
3. Open the notebook in Jupyter and run cells sequentially
4. The pipeline model will be saved to `pipeModel/` and features to `output.parquet`

> **Note:** The Spark master IP (`10.43.97.187`) and HDFS address (`10.195.34.34`) are cluster-specific. Update these for your environment.

---

## Conclusions

The lab demonstrates that a proper EDA and data preparation pipeline significantly impacts model performance. The **Gradient Boosted Tree** achieved the best results, with all metrics above 0.85 and AUC of ~0.93.

**Recommendations:**
- Define business priorities (is it more costly to miss a `yes` or flag a `no`?) to choose the right optimization metric
- Explore feature engineering and variable interactions to further reduce bias
- Consider adjusting `maxDepth` in Decision Tree to improve its AUC
- Exclude `duration` from production models, as it is only known after the call ends

---

## References
- https://archive.ics.uci.edu/dataset/222/bank+marketing
- One-Hot Encoding and Two-Hot Encoding: An Introduction. (2024). ResearchGate.
- [PySpark ML Classification docs](https://spark.apache.org/docs/latest/ml-classification-regression.html)
- [VectorAssembler API](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html)
- [OneHotEncoder API](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.OneHotEncoder.html)
