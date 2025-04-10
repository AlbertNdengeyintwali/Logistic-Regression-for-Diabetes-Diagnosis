# Logistic Regression for Diabetes Diagnosis: Insights from Patient Data
## Introduction

Logistic regression is a powerful and widely used statistical analysis method that is employed to model the relationship between a binary dependent variable and one or more independent variables. It is particularly useful in situations where the outcome is a binary variable such as predicting whether an individual has a disease (like diabetes) or not, based on several factors. 

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

### Table showing detailed information

| ni   | Pregnancies ($X_1$) | Glucose ($X_2$) | BloodPressure ($X_3$) | SkinThickness ($X_4$) | Insulin ($X_5$) | BMI ($X_6$) | DiabetesPedigreeFunction ($X_7$) | Age ($X_8$) | Outcome ($Y$) |
|------|---------------------|-----------------|-----------------------|-----------------------|-----------------|-------------|----------------------------------|-------------|---------------|
| 1    | 6                   | 148             | 72                    | 35                    | 0               | 33.6        | 0.627                            | 50          | 1             |
| 2    | 1                   | 85              | 66                    | 29                    | 0               | 26.6        | 0.351                            | 31          | 0             |
| 3    | 8                   | 183             | 64                    | 0                     | 0               | 23.3        | 0.672                            | 32          | 1             |
| 4    | 1                   | 89              | 66                    | 23                    | 94              | 28.1        | 0.167                            | 21          | 0             |
| 5    | 0                   | 137             | 40                    | 35                    | 168             | 43.1        | 2.288                            | 33          | 1             |
| 6    | 5                   | 116             | 74                    | 0                     | 0               | 25.6        | 0.201                            | 30          | 0             |
| 7    | 3                   | 78              | 50                    | 32                    | 88              | 31          | 0.248                            | 26          | 1             |
| 8    | 10                  | 115             | 0                     | 0                     | 0               | 35.3        | 0.134                            | 29          | 0             |
| 9    | 2                   | 197             | 70                    | 45                    | 543             | 30.5        | 0.158                            | 53          | 1             |
| 10   | 8                   | 125             | 96                    | 0                     | 0               | 0           | 0.232                            | 54          | 1             |
| ...  | ...                 | ...             | ...                   | ...                   | ...             | ...         | ...                              | ...         | ...           |
| 762  | 9                   | 170             | 74                    | 31                    | 0               | 44          | 0.403                            | 43          | 1             |
| 763  | 9                   | 89              | 62                    | 0                     | 0               | 22.5        | 0.142                            | 33          | 0             |
| 764  | 10                  | 101             | 76                    | 48                    | 180             | 32.9        | 0.171                            | 63          | 0             |
| 765  | 2                   | 122             | 70                    | 27                    | 0               | 36.8        | 0.34                             | 27          | 0             |
| 766  | 5                   | 121             | 72                    | 23                    | 112             | 26.2        | 0.245                            | 30          | 0             |
| 767  | 1                   | 126             | 60                    | 0                     | 0               | 30.1        | 0.349                            | 47          | 1             |
| 768  | 1                   | 93              | 70                    | 31                    | 0               | 30.4        | 0.315                            | 23          | 0             |

You can click [here](https://www.kaggle.com/datasets/saurabh00007/diabetescsv/) to see the whole dataset.


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



### Histogram to see distribution of all explanatory variables

![Rplot](https://github.com/user-attachments/assets/1ffbe9cf-e7b7-44ff-933c-425cca584219)

### Box plots to check outliers of all explanatory variables

![Rplot2](https://github.com/user-attachments/assets/ecf43254-c568-42b7-bc58-359d8cdfca12)

Since we have observed the presence of outliers in the predictor variables through the boxplots, it is crucial to address the impact of these outliers on the model's performance. Outliers can disproportionately influence the estimates and predictions of a standard logistic regression model, leading to biased or unstable results. To mitigate this issue, I have decided to use a **robust logistic regression model**. This model is designed to be less sensitive to extreme values and can provide more reliable parameter estimates, ensuring that the model's predictions are not unduly influenced by outliers in the data. This approach will help improve the robustness and accuracy of the analysis.

### Model Overview

This section provides an overview of the logistic regression model used for predicting the **Outcome** variable based on various predictors. Due to the presence of outliers in the dataset, a **robust logistic regression model** was used to minimize the influence of these outliers on the model's performance.

```
Call:  glmrob(formula = Outcome ~ ., family = binomial, data = train_data,      method = "BY") 


Coefficients:
                          Estimate Std. Error z value Pr(>|z|)    
(Intercept)              -8.515855   0.911580  -9.342  < 2e-16 ***
Pregnancies               0.104525   0.048442   2.158   0.0309 *  
Glucose                   0.036182   0.004898   7.388 1.50e-13 ***
BloodPressure            -0.012803   0.006281  -2.038   0.0415 *  
SkinThickness             0.003630   0.008747   0.415   0.6781    
Insulin                  -0.001745   0.001242  -1.405   0.1599    
BMI                       0.089775   0.020615   4.355 1.33e-05 ***
DiabetesPedigreeFunction  0.707081   0.432746   1.634   0.1023    
Age                       0.017299   0.012336   1.402   0.1608 
 ```
#### Interpretation of Coefficients

1. **Intercept (-8.515855)**:  
   The intercept represents the log-odds of the outcome (having diabetes or not) when all predictors are zero. The negative intercept indicates that when all predictors are at their baseline, the likelihood of having diabetes is low. The p-value for the intercept is very small (**< 2e-16**), indicating it is highly statistically significant.

2. **Pregnancies (0.104525)**:  
   A positive coefficient for **Pregnancies** suggests that as the number of pregnancies increases, the log-odds of having diabetes increase. Each additional pregnancy increases the log-odds of having diabetes by 0.104525. The p-value (**0.0309**) is below the usual 0.05 threshold, indicating a statistically significant.

3. **Glucose (0.036182)**:  
   Higher glucose levels are associated with an increased likelihood of having diabetes. Each unit increase in glucose results in a 0.036182 increase in the log-odds of having diabetes. The very small p-value (**1.50e-13**) indicates that glucose is a **highly significant predictor** of diabetes.

4. **BloodPressure (-0.012803)**:  
   A negative coefficient for **BloodPressure** suggests that higher blood pressure is associated with a lower likelihood of having diabetes. For each unit increase in blood pressure, the log-odds of having diabetes decrease by 0.012803. The p-value (**0.0415**) is statistically significant, indicating that blood pressure is an important predictor of diabetes.

5. **SkinThickness (-0.003630)**:  
   The coefficient for **SkinThickness** is close to zero, meaning it has a very weak effect on the likelihood of having diabetes. The p-value (**0.6781**) is much higher than 0.05, indicating that skin thickness is **not a statistically significant predictor**.

6. **Insulin (-0.00145)**:  
   The coefficient for **Insulin** is negative, suggesting that higher insulin levels might decrease the likelihood of having diabetes. However, the p-value (**0.25082**) indicates that insulin is **not statistically significant** in predicting diabetes.

7. **BMI (0.089775)**:  
   The positive coefficient for **BMI** suggests that higher BMI is associated with a higher likelihood of having diabetes. Each unit increase in BMI results in a 0.089775 increase in the log-odds of having diabetes. The very small p-value (**1.33e-05**) indicates that BMI is a **highly significant predictor** of diabetes.

8. **DiabetesPedigreeFunction (0.707081)**:  
   The coefficient for **DiabetesPedigreeFunction** is positive, suggesting that a higher value of this function increases the likelihood of having diabetes. Each unit increase in this function increases the log-odds of having diabetes by 0.707081. The p-value (**0.1023**) indicates a **no statistically significant** relationship.

9. **Age (0.017299)**:  
   The positive coefficient for **Age** indicates that as age increases, the likelihood of having diabetes also increases. Each year increase in age results in a 0.017299 increase in the log-odds of having diabetes. The p-value (**0.1608**) is not significant, suggesting that age has **no influence** on the likelihood of having diabetes.


This model suggests that Pregnancies, glucose levels, blood pressure, and BMI are the most important predictors of diabetes. diabetes pedigree function, Skin thickness and insulin and age do not appear to significantly predict the outcome.

### Confusion Matrix and Model Performance

The confusion matrix provides a detailed view of the model's performance, showing the number of true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN). Below is a summary of the confusion matrix and important performance metrics:

```
Confusion Matrix and Statistics

          Reference
Prediction   0   1
         0 137  36
         1  12  45
                                         
               Accuracy : 0.7913         
                 95% CI : (0.733, 0.8419)
    No Information Rate : 0.6478         
    P-Value [Acc > NIR] : 1.521e-06      
                                         
                  Kappa : 0.5095         
                                         
 Mcnemar's Test P-Value : 0.0009009      
                                         
            Sensitivity : 0.9195         
            Specificity : 0.5556         
         Pos Pred Value : 0.7919         
         Neg Pred Value : 0.7895         
             Prevalence : 0.6478         
         Detection Rate : 0.5957         
   Detection Prevalence : 0.7522         
      Balanced Accuracy : 0.7375         
                                         
       'Positive' Class : 0 
```
#### Confusion Matrix

| Prediction | 0   | 1   |
|------------|-----|-----|
| **0**      | 137 | 36  |
| **1**      | 12  | 45  |

Where:
- **True Negatives (TN)**: 137
- **False Positives (FP)**: 36
- **False Negatives (FN)**: 12
- **True Positives (TP)**: 45

#### Performance Metrics

- **Accuracy**: 0.7913  
  This indicates that 79.13% of the model’s predictions were correct.
- **95% CI (Confidence Interval)**: (0.733, 0.8419)  
  The 95% confidence interval for the accuracy is between 73.3% and 84.19%, indicating the range within which the true accuracy is likely to fall.
- **No Information Rate (NIR)**: 0.6478   
  This is the accuracy that would be achieved if the model simply predicted the most frequent class (in this case, class 0). The model’s accuracy is higher than the NIR, indicating it is providing useful predictions.
- **P-Value [Acc > NIR]**: 1.521e-06  
  This value indicates that the model’s accuracy is statistically significantly better than the No Information Rate. A low p-value (less than 0.05) suggests that the model performs significantly better than random guessing.
- **Kappa Statistic**:0.5095  
  The kappa statistic measures agreement between the predicted and actual values, accounting for chance. A kappa value of 0.5095 suggests moderate agreement.
- **McNemar's Test P-Value**: 0.0009009   
  This test compares the proportion of incorrect predictions made by the model. A p-value less than 0.05 suggests that there is significant difference between the two types of errors (false positives and false negatives).

#### Specific Performance Metrics

- **Sensitivity**: 0.9195 
  Sensitivity (or True Positive Rate) indicates that 91.95% of the actual positive cases were correctly identified by the model.
- **Specificity**: 0.5556  
  Specificity (or True Negative Rate) indicates that 55.56% of the actual negative cases were correctly identified.
- **Positive Predictive Value**: 0.7919  
  Precision indicates that 79.19% of the predicted positive cases were actually positive.
- **Negative Predictive Value**: 0.7895  
  Negative Predictive Value indicates that 78.95% of the predicted negative cases were actually negative.
- **Prevalence**: 0.6478  
  Prevalence refers to the proportion of the dataset that belongs to the positive class (class 1). In this case, approximately 64.78% of the instances in the dataset are in the positive class.
- **Detection Rate**: 0.5957  
  Detection Rate is the proportion of actual positives that are correctly detected by the model. In this case, 59.57% of the actual positives were detected by the model.
- **Detection Prevalence**: 0.7522  
  Detection Prevalence is the proportion of instances that the model predicted as positive. In this case, 75.22% of the predictions made by the model were for the positive class.
- **Balanced Accuracy**: 0.7375  
  Balanced accuracy is the average of sensitivity and specificity, which provides a more balanced evaluation when dealing with imbalanced classes. IIn this case, a balanced accuracy of 73.75% indicates that the model performs reasonably well in distinguishing between both the positive and negative classes, even when the class distribution is not equal. It reflects a good balance between correctly identifying both the true positives and true negatives.

When calculating The F1-score we get `F1-Score: 0.8509317` which is the harmonic mean of precision and recall, balancing both metrics that suggests a good balance between precision and recall.

### ROC Curve Interpretation

![ROC_Curves_Comparison](https://github.com/user-attachments/assets/627f9458-9483-4502-993f-52b75f0d79d1)

The ROC curve I generated visually represents my model’s ability to distinguish between diabetic and non-diabetic cases across different classification thresholds.

- **Area Under the Curve (AUC) Score: 0.8436**
   - My model achieved an AUC (Area Under the Curve) of 0.8436, meaning it has an 84.36% probability of correctly ranking a randomly chosen diabetic case higher than a non-diabetic case.
   - Since an AUC of 0.80 - 0.89 is considered good, my model has a strong ability to differentiate between the two classes.

### :bulb: Key Performance Message
- **Accuracy**: The model achieved an accuracy of **78.26%**, which is higher than the baseline accuracy of **68.26%**, indicating that the model is performing better and providing more valuable predictions. Again An F1-score of **85%** confirms a strong overall performance.

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
\begin{align*}
\text{logit}(P) &= -8.515855 + (0.104525 \times 6) + (0.036182 \times 148) + (-0.012803 \times 72) + (0.003630 \times 35)\\
& + (-0.001745 \times 0) + (0.089775 \times 33.6) + (0.707081 \times 0.627) + (0.017299 \times 50) \approx 0.9961948
\end{align*}
$$


Finally, using the logistic function:

$$
P(Y=1) = \dfrac{e^{9961948}}{1 + e^{9961948}} \approx 0.73
$$

Therefore, the predicted probability of having diabetes for this individual is approximately **73%**.

This high probability suggests a significant likelihood of diabetes presence for the given input values.


 ## Ownership
This project is owned and created by **Albert Ndengeyintwali**.

© 2025 Albert Ndengeyintwali. All rights reserved.






