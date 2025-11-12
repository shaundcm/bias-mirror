# BiasMirror: Fairness Analysis in Machine Learning Models  

BiasMirror is a Python-based framework for **auditing algorithmic bias** in machine learning models.  
It evaluates fairness across demographic groups using multiple classifiers and advanced statistical metrics —  
helping researchers and developers understand **how fair their models really are.**

---

## Overview  

This project investigates algorithmic bias in ML models trained on the **UCI Adult Income dataset**.  
It compares how different models perform with respect to both **accuracy** and **fairness**,  
providing a complete evaluation pipeline — from preprocessing to statistical validation and fairness reporting.

---

## Features  

- Automated preprocessing and normalization  
- Multi-model training and evaluation (6 classifiers)  
- Fairness metrics: *SPD*, *DI*, *TPR*, *FPR*  
- Bootstrapped confidence intervals + permutation significance tests  
- Meta-probe analysis for demographic signal leakage  
- Automatic CSV and PDF report generation  

---

## Models Evaluated  

- Logistic Regression  
- Decision Tree  
- Random Forest  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Multi-Layer Perceptron (MLP)

---

## Dataset  

- **Source:** [Adult Income Dataset](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset) 
- **Size:** 48,842 records, 16 features  
- **Target Variable:** `income` (≤50K or >50K)  
- **Sensitive Attribute:** `race` (normalized to White, Black, Asian, Native, Other)  

---

## Methodology  

1. **Preprocessing**  
   - Handle missing values and normalize numeric features  
   - Encode categorical features  
   - Split dataset into train/test (70–30)

2. **Model Training**  
   - Train all six classifiers using identical settings  
   - Evaluate accuracy, F1-score, and AUC  

3. **Fairness Computation**  
   - Compute *Statistical Parity Difference (SPD)*, *Disparate Impact (DI)*, *TPR/FPR* per race group  

4. **Fairness Scoring**  
   - Average |SPD| across groups = fairness score  

5. **Meta-Probe Leakage Test**  
   - Train a logistic regression model on each classifier’s probability outputs  
   - Check if race can be predicted from model outputs (if yes → demographic leakage detected)

6. **Reporting**  
   - Consolidate all metrics into master CSVs and detailed PDFs  

---

## Results Summary  

| Model | Accuracy | Macro F1 | Fairness Score |
|:------|:---------:|:--------:|:--------------:|
| Decision Tree | 0.8509 | 0.6165 | **0.0646** |
| KNN | 0.8361 | 0.6378 | 0.1013 |
| Random Forest | 0.8534 | 0.6685 | 0.1018 |
| MLP | 0.8493 | 0.6656 | 0.1044 |
| Logistic Regression | 0.8542 | 0.6611 | 0.1098 |
| Naive Bayes | 0.6017 | 0.5271 | 0.3153 |

> **Decision Tree** performed as the fairest model overall, balancing predictive accuracy and fairness.  
> **Naive Bayes** showed the highest demographic disparity.

---

## Interpretation  

- Fairness does **not** directly correlate with accuracy — some of the most accurate models (e.g., Random Forest) still showed measurable bias.  
- The meta-probe test revealed weak demographic signal leakage (`balanced accuracy = 0.20`, `AUC = 0.707`),  
  indicating limited encoding of racial information in model outputs.  
- Models differ in *who* they perform best for, even if the overall accuracy seems fair.  

---

## 💾 Outputs  

All generated files are saved in:
