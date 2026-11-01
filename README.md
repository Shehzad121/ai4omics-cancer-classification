# AI for Omics: Cancer Transcriptomic Classification & Molecular Subtyping

Practical machine learning workflows for classifying cancer types and molecular subtypes from RNA-seq gene expression profiles ($\log_2(1+\text{FPKM})$ units). This project addresses the **AI4OneHealth Data Challenge** (Course: *Introduction to AI for Health*).

Dataset repository and reference: [epimed/ai4omics-data-challenge-cancers](https://github.com/epimed/ai4omics-data-challenge-cancers).

---

## Project Highlights

- **Task 1 (Lung Cancer Subtyping):** Classified primary lung tumors into Lung Adenocarcinoma (**ADK**) and Squamous Cell Carcinoma (**SQC**) using regularized Logistic Regression ($L_2$, $C=0.1$). Achieved **95.19% cross-validation accuracy** and **0.984 ROC-AUC** across 5 folds.
- **Task 2 (Tissue of Origin Identification):** Visualized healthy Brain, Breast, and Liver tissues using Principal Component Analysis (**PCA**). Trained a non-linear **Random Forest** classifier (300 estimators, 99.65% CV accuracy on normal tissues) to predict the primary origin of 1,628 unlabeled solid tumors.
- **Task 3 (Breast Cancer Molecular Subtyping):** Extracted tumors identified as breast cancer and mapped them against the **PAM50 gene signature** (48 matching genes). Applied unsupervised clustering and resolved molecular subtypes into **Luminal-A** and **Basal-like** guided by biological marker expression (*ESR1* and *PGR*).

---

## Repository Structure

```text
Data_Challenge_Submission/
├── ai4omics-data-challenge-cancers-main/     # Raw transcriptomic datasets
│   ├── T1_X_train.csv                       # Lung cancer training features (831 x 5135)
│   ├── T1_y_train.csv                       # Lung cancer training labels (ADK vs SQC)
│   ├── T1_X_test.csv                        # Lung cancer test features (200 x 5135)
│   ├── T1_y_test.csv                        # Prediction template
│   ├── T2_X_normal.csv                      # Normal healthy tissues (580 x 5135)
│   ├── T2_y_normal.csv                      # Normal tissue labels (Brain, Breast, Liver)
│   ├── T2_X_cancer.csv                      # Unlabeled cancer samples (1628 x 5135)
│   ├── T2_y_cancer.csv                      # Prediction template for Task 2 & 3
│   └── gene_list_pam50_breast_cancer.csv    # 50-gene PAM50 breast cancer panel
├── task1.ipynb                              # Pipeline for Task 1
├── task2.ipynb                              # Pipeline for Task 2 & Task 3
├── T1_y_test.csv                            # Final predictions for Task 1
├── T2_y_cancer.csv                          # Final predictions for Task 2 & Task 3
└── README.md                                # Project documentation
```

---

## Dataset Overview

The dataset consists of high-dimensional transcriptomic profiles measured across **5,135 genes** per sample, normalized in $\log_2(1 + \text{FPKM})$:

| File | Context | Samples | Features | Target / Annotations |
| :--- | :--- | :---: | :---: | :--- |
| `T1_X_train.csv` | Lung cancer training | 831 | 5,135 | `target` (459 ADK, 372 SQC) |
| `T1_X_test.csv` | Lung cancer evaluation | 200 | 5,135 | Missing labels to predict |
| `T2_X_normal.csv` | Healthy baseline tissues | 580 | 5,135 | `tissue` (323 Brain, 176 Breast, 81 Liver) |
| `T2_X_cancer.csv` | Unlabeled malignant tumors | 1,628 | 5,135 | Missing `tissue` and `subtype` |
| `gene_list_pam50` | Breast cancer signature | 50 genes | - | 48 genes overlapping in dataset |

All CSV files use semicolon (`;`) delimiters with sample IDs in the first column (`index_col=0`).

---

## Methodology & Results

### Task 1: Histological Subtypes of Lung Cancer

Lung adenocarcinoma (ADK) and squamous cell carcinoma (SQC) exhibit distinct gene expression programs, but their high feature dimensionality ($p = 5,135$) relative to sample size ($N = 831$) requires effective regularization to avoid overfitting.

1. **Preprocessing:** Verified complete data integrity (0 missing values). Standardized features using `StandardScaler` fitted exclusively on training folds to prevent data leakage.
2. **Modeling:** Trained an $L_2$-regularized `LogisticRegression` model ($C=0.1, \text{max\_iter}=1000$).
3. **Cross-Validation Performance (5-Fold Stratified):**
   - **Mean Accuracy:** $95.19\% \pm 2.01\%$
   - **Precision (SQC):** $95.96\% \pm 1.79\%$
   - **Recall (SQC):** $94.02\% \pm 3.89\%$
   - **F1-Score (SQC):** $94.93\% \pm 2.17\%$
   - **ROC-AUC:** $0.9847 \pm 0.0063$
4. **Test Set Prediction:** Transformed `T1_X_test.csv` (200 samples) and exported predictions to `T1_y_test.csv` (105 ADK, 95 SQC).

### Task 2: Cancer Tissue of Origin Identification

Malignant transformation often alters transcriptional lineage markers. To identify primary tumor origins, we leveraged 580 normal tissue samples across Brain, Breast, and Liver.

1. **PCA Visualization:** Applied `StandardScaler` followed by 2-component PCA on normal tissue profiles. The top two principal components cleanly separated the three tissue types into distinct, non-overlapping clusters.
2. **Model Formulation:** Trained a non-linear `RandomForestClassifier` (300 estimators, `random_state=42`). Non-linear tree ensembles naturally capture complex pathway rewiring in neoplastic cells without assuming linear separability.
   - **Normal 5-Fold CV Accuracy:** **$99.65\%$**
3. **Tumor Classification:** Evaluated the 1,628 unknown cancer samples (`T2_X_cancer.csv`), identifying:
   - **Breast Cancer:** 773 samples
   - **Brain Cancer:** 507 samples
   - **Liver Cancer:** 348 samples

### Task 3: Molecular Subtyping of Breast Cancer

For patients diagnosed with breast tumors in Task 2, molecular subtyping dictates clinical prognosis and hormone receptor targeting.

1. **PAM50 Signature Extraction:** Filtered the 773 predicted breast tumor samples and retained 48 available PAM50 genes.
2. **Biological Marker Mapping:**
   - **Luminal-A:** Strong expression of estrogen receptor 1 (*ESR1*) and progesterone receptor (*PGR*).
   - **Basal-like:** Low or absent expression of both hormone receptors (triple-negative phenotype characteristics).
3. **Clustering & Annotation:** Standardized the 48-gene subset and ran K-Means ($k=2$). Cluster centroids were evaluated for mean *ESR1* and *PGR* signal:
   - Cluster with elevated *ESR1* ($\mu \approx 4.0$) was assigned **Luminal-A** (573 samples).
   - Cluster with suppressed *ESR1* ($\mu \approx 0.3$) was assigned **Basal-like** (200 samples).
4. **Output Compliance:** Stored results in `T2_y_cancer.csv`. The `subtype` field was populated exclusively for breast samples, leaving non-breast tumors empty (`NaN`) as specified in the evaluation protocol.

---

## How to Run

### Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Execution

1. Clone or download this repository.
2. Open and run the notebooks sequentially:
   - Run `task1.ipynb` to inspect lung cancer stats, reproduce cross-validation metrics, plot ROC curves, and generate `T1_y_test.csv`.
   - Run `task2.ipynb` to generate normal tissue PCA visualizations, predict tumor origins, perform PAM50 clustering, and save `T2_y_cancer.csv`.

Both notebooks use local relative paths (`data_dir = "ai4omics-data-challenge-cancers-main"`), so no manual directory edits are required.

---

## References

- Epimed Group. *ai4omics-data-challenge-cancers*. [GitHub Repository](https://github.com/epimed/ai4omics-data-challenge-cancers).
- Parker JS, et al. *Supervised Risk Predictor of Breast Cancer Based on Intrinsic Subtypes*. J Clin Oncol. 2009. (PAM50 classification rationale).

