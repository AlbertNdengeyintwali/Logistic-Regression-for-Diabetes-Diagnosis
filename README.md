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

```
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
Age                       0.0292910  0.0155226   1.887  0.05916 .
 ```
#### Interpretation of Coefficients

1. **Intercept (-8.11777)**:  
   The intercept represents the log-odds of the outcome (having diabetes or not) when all predictors are zero. The negative intercept indicates that when all predictors are at their baseline, the likelihood of having diabetes is low. The p-value for the intercept is very small (**< 2e-16**), indicating it is highly statistically significant.

2. **Pregnancies (0.08930)**:  
   A positive coefficient for **Pregnancies** suggests that as the number of pregnancies increases, the log-odds of having diabetes increase. Each additional pregnancy increases the log-odds of having diabetes by 0.0893. The p-value (**0.08216**) is above the usual 0.05 threshold, indicating a **marginal** effect.

3. **Glucose (0.03454)**:  
   Higher glucose levels are associated with an increased likelihood of having diabetes. Each unit increase in glucose results in a 0.0345 increase in the log-odds of having diabetes. The very small p-value (**2.74e-15**) indicates that glucose is a **highly significant predictor** of diabetes.

4. **BloodPressure (-0.01797)**:  
   A negative coefficient for **BloodPressure** suggests that higher blood pressure is associated with a lower likelihood of having diabetes. For each unit increase in blood pressure, the log-odds of having diabetes decrease by 0.018. The p-value (**0.00604**) is statistically significant, indicating that blood pressure is an important predictor of diabetes.

5. **SkinThickness (-0.00087)**:  
   The coefficient for **SkinThickness** is close to zero, meaning it has a very weak effect on the likelihood of having diabetes. The p-value (**0.92393**) is much higher than 0.05, indicating that skin thickness is **not a statistically significant predictor**.

6. **Insulin (-0.00145)**:  
   The coefficient for **Insulin** is negative, suggesting that higher insulin levels might decrease the likelihood of having diabetes. However, the p-value (**0.25082**) indicates that insulin is **not statistically significant** in predicting diabetes.

7. **BMI (0.08913)**:  
   The positive coefficient for **BMI** suggests that higher BMI is associated with a higher likelihood of having diabetes. Each unit increase in BMI results in a 0.0891 increase in the log-odds of having diabetes. The very small p-value (**7.41e-06**) indicates that BMI is a **highly significant predictor** of diabetes.

8. **DiabetesPedigreeFunction (0.72507)**:  
   The coefficient for **DiabetesPedigreeFunction** is positive, suggesting that a higher value of this function increases the likelihood of having diabetes. Each unit increase in this function increases the log-odds of having diabetes by 0.7251. The p-value (**0.06847**) indicates a **marginally significant** relationship.

9. **Age (0.02929)**:  
   The positive coefficient for **Age** indicates that as age increases, the likelihood of having diabetes also increases. Each year increase in age results in a 0.0293 increase in the log-odds of having diabetes. The p-value (**0.05916**) is marginally significant, suggesting that age has a **moderate influence** on the likelihood of having diabetes.

#### Summary of Significant Predictors

- **Highly Significant Predictors (p-value < 0.05)**:
  - Glucose
  - BloodPressure
  - BMI

- **Marginally Significant Predictors (0.05 <= p-value < 0.1)**:
  - Pregnancies
  - DiabetesPedigreeFunction
  - Age

- **Non-Significant Predictors (p-value > 0.1)**:
  - SkinThickness
  - Insulin

This model suggests that glucose levels, blood pressure, and BMI are the most important predictors of diabetes. Pregnancies, diabetes pedigree function, and age have some effect but are marginally significant. Skin thickness and insulin levels do not appear to significantly predict the outcome.

### Confusion Matrix and Model Performance

The confusion matrix provides a detailed view of the model's performance, showing the number of true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN). Below is a summary of the confusion matrix and important performance metrics:

```
Confusion Matrix and Statistics

          Reference
Prediction   0   1
         0 139  32
         1  18  41
                                          
               Accuracy : 0.7826          
                 95% CI : (0.7236, 0.8341)
    No Information Rate : 0.6826          
    P-Value [Acc > NIR] : 0.0005085       
                                          
                  Kappa : 0.4712          
                                          
 Mcnemar's Test P-Value : 0.0659921       
                                          
            Sensitivity : 0.8854          
            Specificity : 0.5616          
         Pos Pred Value : 0.8129          
         Neg Pred Value : 0.6949          
             Prevalence : 0.6826          
         Detection Rate : 0.6043          
   Detection Prevalence : 0.7435          
      Balanced Accuracy : 0.7235
```
#### Confusion Matrix

| Prediction | 0   | 1   |
|------------|-----|-----|
| **0**      | 139 | 32  |
| **1**      | 18  | 41  |

Where:
- **True Negatives (TN)**: 139
- **False Positives (FP)**: 32
- **False Negatives (FN)**: 18
- **True Positives (TP)**: 41

#### Performance Metrics

- **Accuracy**: 0.7826  
  This indicates that 78.26% of the model’s predictions were correct.
- **95% CI (Confidence Interval)**: (0.7236, 0.8341)  
  The 95% confidence interval for the accuracy is between 72.36% and 83.41%, indicating the range within which the true accuracy is likely to fall.
- **No Information Rate (NIR)**: 0.6826  
  This is the accuracy that would be achieved if the model simply predicted the most frequent class (in this case, class 0). The model’s accuracy is higher than the NIR, indicating it is providing useful predictions.
- **P-Value [Acc > NIR]**: 0.0005085  
  This value indicates that the model’s accuracy is statistically significantly better than the No Information Rate. A low p-value (less than 0.05) suggests that the model performs significantly better than random guessing.
- **Kappa Statistic**: 0.4712  
  The kappa statistic measures agreement between the predicted and actual values, accounting for chance. A kappa value of 0.4712 suggests moderate agreement.
- **McNemar's Test P-Value**: 0.0659921  
  This test compares the proportion of incorrect predictions made by the model. A p-value greater than 0.05 suggests that there is no significant difference between the two types of errors (false positives and false negatives).

#### Specific Performance Metrics

- **Sensitivity**: 0.8854  
  Sensitivity (or True Positive Rate) indicates that 88.54% of the actual positive cases were correctly identified by the model.
- **Specificity**: 0.5616  
  Specificity (or True Negative Rate) indicates that 56.16% of the actual negative cases were correctly identified.
- **Positive Predictive Value**: 0.8129  
  Precision indicates that 81.29% of the predicted positive cases were actually positive.
- **Negative Predictive Value**: 0.6949  
  Negative Predictive Value indicates that 69.49% of the predicted negative cases were actually negative.
- **Prevalence**: 0.6826  
  Prevalence refers to the proportion of the dataset that belongs to the positive class (class 1). In this case, approximately 68.26% of the instances in the dataset are in the positive class.
- **Detection Rate**: 0.6043  
  Detection Rate is the proportion of actual positives that are correctly detected by the model. In this case, 60.43% of the actual positives were detected by the model.
- **Detection Prevalence**: 0.7435  
  Detection Prevalence is the proportion of instances that the model predicted as positive. In this case, 74.35% of the predictions made by the model were for the positive class.
- **Balanced Accuracy**: 0.7235  
  Balanced accuracy is the average of sensitivity and specificity, which provides a more balanced evaluation when dealing with imbalanced classes. IIn this case, a balanced accuracy of 72.35% indicates that the model performs reasonably well in distinguishing between both the positive and negative classes, even when the class distribution is not equal. It reflects a good balance between correctly identifying both the true positives and true negatives.

### :bulb: Key Performance Message
- **Accuracy**: The model achieved an accuracy of **78.26%**, which is higher than the baseline accuracy of **68.26%**, indicating that the model is performing better and providing more valuable predictions.

### Predicted Probability of Diabetes

Using the logistic regression formula, the probability of having diabetes based on the individual with the first observations :

- Pregnancies: 6
- Glucose: 148
- BloodPressure: 72
- SkinThickness: 35
- Insulin: 0
- BMI: 33.6
- DiabetesPedigreeFunction: 0.627
- Age: 50

We can calculate the logit as:

$$
\text{logit}(P) = -8.1177733 + (0.0893022 \times 6) + (0.0345361 \times 148) + (-0.0179712 \times 72) + (-0.0008684 \times 35) + (-0.0014535 \times 0) + (0.0891310 \times 33.6) + (0.7250662 \times 0.627) + (0.0292910 \times 50) \approx 0.8088096
$$

Finally, using the logistic function:

$$
P(Y=1) = \dfrac{e^{1.11903}}{1 + e^{1.11903}} \approx 0.7538
$$

Therefore, the predicted probability of having diabetes for this individual is approximately **75.38%**.

This high probability suggests a significant likelihood of diabetes presence for the given input values.







