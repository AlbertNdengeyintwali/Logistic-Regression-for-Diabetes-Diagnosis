# Logistic Regression for Diabetes Diagnosis: Insights from Patient Data
## Introduction

Logistic regression is a powerful and widely used statistical analysis method that is employed to model the relationship between a binary dependent variable and one or more independent variables. It is particularly useful in situations where the outcome is a binary variable—such as predicting whether an individual has a disease (like diabetes) or not, based on several factors. 

Unlike linear regression, which models the relationship between independent variables and a continuous dependent variable, logistic regression is used when the dependent variable is categorical, typically binary. For example, in medical research, logistic regression can be applied to predict the probability of a patient having a disease based on a set of predictor variables like age, weight, blood pressure, etc.

### Logistic Regression Model

In the context of logistic regression, we model the probability of an event occurring (such as the presence of diabetes) as a function of independent variables. The formula for logistic regression is derived from the logistic function, which ensures that the predicted outcome is always between 0 and 1. 

We define:

- $Y$: The dependent variable (the binary outcome). For example, 1 might indicate that the individual has diabetes, and 0 means they do not.
- $X_1, X_2, \dots, X_n$: The independent variables (predictors or features) which could be variables such as age, BMI, blood pressure, and glucose levels.
- $\beta_0, \beta_1, \dots, \beta_n$: The coefficients (or parameters) that we aim to estimate during the model training process.

The logistic regression model can be written as:

$$
\begin{align*}
P(Y = 1) &= \frac{e^{\beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \beta_nX_n}}{1 + e^{\beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \beta_nX_n}}\\
P(Y = 0)&= 1- P(Y = 1) = \frac{1}{1 + e^{\beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \beta_nX_n}}
\end{align*}
$$

Where:
- $P(Y = 1)$ is the probability of the positive outcome (e.g., the person has diabetes).
- $P(Y = 0)$ is the probability of the negative outcome (e.g., the person has no diabetes).
- The expression in the denominator $1 + e^{\beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \beta_nX_n}$ ensures that the probability is between 0 and 1.

In simple terms, the logistic regression model calculates the likelihood that a given set of inputs (predictors) will result in a certain outcome, which is modeled as a probability.

### Odds and Log-Odds (Logit)

Logistic regression operates on the concept of odds and log-odds, which are essential in understanding the relationship between the independent variables and the dependent variable:

- **Odds**: The odds of an event occurring is the ratio of the probability of the event happening to the probability of it not happening. For example, the odds of a person developing diabetes are given by:

$$
\text{Odds of outcome} = \frac{P(Y = 1)}{P(Y = 0)}
$$

- **Log-Odds (Logit)**: The log-odds (also known as the logit) is the natural logarithm of the odds. This transformation helps in linearizing the relationship between the predictors and the outcome, allowing the logistic regression model to be estimated using linear regression techniques.

$$
\text{Log-Odds (Logit)} = \ln\left(\frac{P(Y = 1)}{P(Y = 0)}\right)
$$

## Problem Statement

In this project, we analyzed a **Diabetes dataset** (CSV format) to study the factors influencing the presence of diabetes, as indicated by the **'Outcome'** variable. The dataset contains medical and demographic information about individuals, helping us explore potential relationships between key health metrics and the likelihood of developing diabetes.

### Dataset Overview

The dataset includes the following independent variables:

- **Pregnancies**: Number of times the patient has been pregnant (Count).
- **Glucose**: Plasma glucose concentration during an oral glucose tolerance test ( $mg/dL$).
- **BloodPressure**: Diastolic blood pressure ($mm \quad Hg$).
- **SkinThickness**: Thickness of skin fold measurements (mm).
- **Insulin**: Serum insulin levels ($mu \quad U/ml$).
- **BMI (Body Mass Index)**: A measure of body weight relative to height($kg/m^2$).
- **DiabetesPedigreeFunction**: A score that represents genetic predisposition to diabetes (Dimensionless).
- **Age**: The age of the individual (Years).

## Methodology

The general methodology we followed in this analysis consists of the following key steps:

### 1. Data Preparation
   - Loaded the dataset and performed an initial inspection to understand its structure.
   - Checked for missing values and handled them appropriately through imputation or removal.
   - Split the dataset into **training** and **testing** subsets to evaluate model performance.

### 2. Exploratory Data Analysis (EDA)
   - Visualized features using **histograms, box plots** to see distribution and checking outliers.
   - Identified potential **outliers** and examined multicollinearity among predictors to ensure model stability.
   - Assessed variable distributions and transformations needed for better model performance.

### 3. Logistic Regression Model Building
   - Selected **Logistic Regression (LR)** as our classification model since it is well-suited for binary outcomes.
   - Trained the model using the **training dataset**, optimizing for accuracy and interpretability.
   - Evaluated model performance using metrics such as **accuracy, precision, recall, and F1-score**.
   - Used **ROC curves and AUC (Area Under the Curve)** to assess the model's ability to classify correctly.

### 4. Interpretation of Coefficients and P-Values
   - Examined the **regression coefficients** to understand how each independent variable influences diabetes risk.
   - Analyzed **p-values** to determine which predictors significantly contribute to the model.
   - Computed the **odds ratios** to quantify the impact of each variable on diabetes probability.
   - Used the logistic function to calculate the probability of diabetes presence for given input values.

## Results

### Table showing detailed information

### Histogram to see distribution of all explanatory variables

![Rplot](https://github.com/user-attachments/assets/1ffbe9cf-e7b7-44ff-933c-425cca584219)

### box plots to check outliers of all explanatory variables

![Rplot2](https://github.com/user-attachments/assets/ecf43254-c568-42b7-bc58-359d8cdfca12)

### Model Overview

This section provides an overview of the logistic regression model used for predicting the **Outcome** variable based on various predictors. Due to the presence of outliers in the dataset, a **robust logistic regression model** was used to minimize the influence of these outliers on the model's performance.

```Call:  glmrob(formula = Outcome ~ ., family = binomial, data = train_data,      method = "BY") 


Call:  glmrob(formula = Outcome ~ ., family = binomial, data = train_data,      method = "BY") 


Coefficients:
                           Estimate Std. Error z value Pr(>|z|)    
(Intercept)              -8.1177733  0.8625303  -9.412  < 2e-16 ***
Pregnancies               0.0893022  0.0513738   1.738  0.08216 .  
Glucose                   0.0345361  0.0043704   7.902 2.74e-15 ***
BloodPressure            -0.0179712  0.0065450  -2.746  0.00604 ** 
SkinThickness            -0.0008684  0.0090950  -0.095  0.92393    
Insulin                  -0.0014535  0.0012657  -1.148  0.25082    
BMI                       0.0891310  0.0198886   4.482 7.41e-06 ***
DiabetesPedigreeFunction  0.7250662  0.3979683   1.822  0.06847 .  
Age                       0.0292910  0.0155226   1.887  0.05916 .  ```








