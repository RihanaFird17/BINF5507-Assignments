# Survival Analysis Algorithm Comparison: Kaplan-Meier survival curves, Cox Proportional Hazards regression, and Random Survival Forests (RSF)
Assignment 4: Survival Analysis of Head and Neck Cancer Patients

# Description
This project applies and compares three survival analysis methods: Kaplan-Meier survival curves, Cox Proportional Hazards regression, and Random Survival Forests (RSF) using the head and neck cancer patients' clinical dataset, sourced from the Cancer Imaging Archive. The focus is understanding, comparing, and reflecting on the methods' utility in clinical decision-making.

# Dataset:
Source: The Cancer Imaging Archive (TCIA)
File: RADCURE_Clinical_v04_20241219.xlsx (converted to RADCURE_Clinical_v04_20241219.csv)
Description: Clinical data for head and neck cancer patients, including demographics, treatment modalities, and survival outcomes.

# Data Preprocessing:
- Explored the dataset to identify missing values, data types, and distributions of key variables.
- Handled missing values by imputing the median for 'Smoking PY' and the mode for 'ECOG PS'.
## Feature Engineering:
- Created event indicator ('Event') based on the 'Status' column (Dead = 1, Alive = 0).
- Defined survival time ('Survival_Time') using the 'Length FU' (Length of Follow-Up) column.
- Cleaned and standardized the 'Tx Modality' column.
- Encoded categorical variables like 'Sex' using label encoding.

# Survival Analysis Model Building: 
## 1. Kaplan-Meier Analysis:
- Generated Kaplan-Meier survival curves to visualize survival probabilities over time for different subgroups:
- Treatment Modality ('Tx Modality')
- Smoking Status ('Smoking Status')
- Performed log-rank tests to assess the statistical significance of differences between survival curves.
- Log-rank Test Results:

### Treatment Modality:
- RT_alone vs. ChemoRT: p-value = 0.00000
- RT_alone vs. RT + EGFRI: p-value = 0.03326
- RT_alone vs. Postop RT alone: p-value = 0.19501
- ChemoRT vs. RT + EGFRI: p-value = 0.00000
- ChemoRT vs. Postop RT alone: p-value = 0.02041
- RT + EGFRI vs. Postop RT alone: p-value = 0.46327

### Smoking Status:
- Ex-smoker vs. Non-smoker: p-value = 0.00000
- Ex-smoker vs. Current: p-value = 0.00000
- Non-smoker vs. Current: p-value = 0.00000

## 2. Cox Proportional Hazards Regression:
- Built a Cox Proportional Hazards model to assess the impact of multiple covariates on survival.
- Covariates included: 'Age' (binned and mapped), 'Sex', and 'Smoking PY'.
- Assessed model assumptions.
- Cox Model summary:
  - **Concordance Index:** **0.65**
  - **Partial AIC:** 15,594.26
  - **Log-likelihood ratio test:** 356.11 (p < 0.005)
  - **Significant Variables:**
    - Age (HR: 1.04, p < 0.005)
    - Smoking PY (HR: 1.01, p < 0.005)

## 3. Random Survival Forests (RSF) and Comparative Performance
- Trained an RSF model to predict survival probabilities and compared predictive performance with Cox regression.
- C-index Comparison:
  - Cox Model: **0.654**
  - RSF Model: **0.685** (Higher predictive power)

# Key Findings:
- **Treatment Modality:** ChemoRT is associated with superior survival outcomes compared to RT_alone and RT + EGFRI, as indicated by Kaplan-Meier curves and log-rank tests.
- **Smoking Status:** Current smokers have significantly lower survival probabilities compared to non-smokers and ex-smokers.
- **Age:** Older age is associated with increased risk.
- **ECOG Performance Status:** Higher ECOG scores (worse performance status) are strongly associated with reduced survival probabilities.

- **Cox Model:** The Cox model revealed the individual impact of age, sex and smoking status on survival.
- The Random Survival Forest method had slightly higher predictive power than Cox.

# Clinical Significance:
The project identifies critical factors influencing survival in head and neck cancer patients:
- **Age** and **ECOG PS** help assess patient suitability for aggressive treatments.
- **Smoking status** informs prevention strategies and risk stratification.
- **Treatment modality** and **Stage** are central to tailoring therapy plans to maximize survival benefits.

# Conclusion:
This survival analysis project provides valuable insights into the factors influencing survival outcomes in head and neck cancer patients. By comparing Kaplan-Meier, Cox regression, and Random Survival Forests, the project highlights the strengths and limitations of each method in a clinical context.
