# Heart Disease Analysis and Prediction
Assignment 2: Regression and Classification Models

## Overview
This project aims to to predict the likelihood of heart disease using the Heart Disease dataset from the UCI Machine Learning Repository by implementing regression and classification models.

## Data loading and Pre-processing
Dataset: The dataset is loaded from the heart_disease_uci.csv file, which contains both numerical and categorical features.

Data Cleaning and pre-processing: 
- Duplicate Rows were removed to ensure data quality.
- Redundant Features with more than 50% missing values (ca and thal) were dropped from the dataset.
- Missing Data Handling: Numerical features with missing values were imputed using the median strategy, while categorical features were handled through one-hot encoding.
- Target Feature Selection: The target variable 'chol' (cholesterol level) was used for regression tasks, and 'num' (Presence of heart disease) was used for classification tasks.

Train-Test Split: The dataset was split into training and test sets using an 80-20 split ratio. The training data was pre-processed using a ColumnTransformer to scale numerical features and encode categorical ones.


## Building Models
### Regression Models
Linear Regression: A linear regression model was built to predict the cholesterol levels ('chol' feature) . The model was trained using the training data and evaluated on the test set using metrics like R² and RMSE.
    
ElasticNet Regression: ElasticNet, a regularized linear regression model, was applied next. Hyperparameters such as alpha (regularization strength) and l1_ratio (mix of Lasso and Ridge penalties) were tuned using GridSearchCV for optimal performance. The evaluation was conducted using R² and RMSE.
    
ElasticNet Hyperparameter Tuning: A heatmap was plotted to visualize the performance of different combinations of alpha and l1_ratio, showing the RMSE scores.

### Classification Models
#### Logistic Regression: 
A logistic regression model was used to predict whether a person has heart disease ('num' feature). The model was evaluated using metrics such as accuracy, F1 score, AUROC, and AUPRC. The ROC and Precision-Recall curves were plotted to compare the performance.

#### k-Nearest Neighbors (k-NN): 
A k-NN classifier with Manhattan distance and 15 neighbors was also used for classification as this was determined to be the best configuration. Similar to logistic regression, metrics such as accuracy, F1 score, AUROC, and AUPRC were evaluated. ROC and Precision-Recall curves were plotted to visualize performance.

## Results
The evaluation metrics for both regression and classification models are summarized below:

Linear Regression:
    R²: 0.5639
    RMSE: 72.5552

ElasticNet Regression:
    R²: 0.2954
    RMSE: 92.2247
    Best Hyperparameters: alpha = 0.0464, l1_ratio = 1.0
    Best RMSE: 69.8036

Logistic Regression:
    Accuracy: 0.8152
    F1 Score: 0.8455
    AUROC: 0.8758
    AUPRC: 0.8945

k-NN Classifier:
    Accuracy: 0.6957
    F1 Score: 0.7358
    AUROC: 0.7663
    AUPRC: 0.8215

This project demonstrates how regression and classification models can be used to predict heart disease outcomes based on various health-related features. 

While logistic regression performed best in terms of accuracy, F1 score, AUROC, and AUPRC; in the classification task, linear regression showed reasonable performance for predicting cholesterol levels. ElasticNet regression showed improvement after hyperparameter tuning. 

The models thus provide valuable insights, and the evaluation metrics help in assessing their effectiveness in making predictions.
