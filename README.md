# Credit Card Default Prediction

## a. Problem statement
Predict whether a credit card client will default on their payment next month,
using demographic data, credit limit, and 6 months of payment history. This is a
binary classification problem. Five classifiers are trained and compared, then
deployed in a Streamlit app.

## b. Dataset description
- Source: UCI Machine Learning Repository, "Default of Credit Card Clients" (Taiwan, 2005)
- Instances: 30,000 | Features: 23 (after dropping ID) | Target: `default payment next month` (0 = no, 1 = yes)
- Features: credit limit (LIMIT_BAL), SEX, EDUCATION, MARRIAGE, AGE, repayment status (PAY_0, PAY_2 to PAY_6), bill amounts (BILL_AMT1 to 6), previous payments (PAY_AMT1 to 6)
- Class balance: 23,364 non-default (77.9%), 6,636 default (22.1%)
- No missing values. Features scaled with StandardScaler. 80/20 stratified train/test split (random_state=7).
- SVM was trained on an 8,000-row sample of the training set for speed.

## c. GitHub repository
https://github.com/rnivash/credit-default-ml-classifier

## d. Models used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8122 | 0.7317 | 0.7358 | 0.2351 | 0.3564 | 0.3420 |
| Decision Tree | 0.8253 | 0.7646 | 0.6929 | 0.3775 | 0.4888 | 0.4208 |
| KNN | 0.8018 | 0.7239 | 0.5925 | 0.3331 | 0.4264 | 0.3371 |
| Naive Bayes | 0.6147 | 0.7417 | 0.3345 | 0.7498 | 0.4626 | 0.2707 |
| Support Vector Machine | 0.8213 | 0.7229 | 0.7289 | 0.3060 | 0.4310 | 0.3914 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest precision (0.74) but lowest recall (0.24), so it is cautious and misses most defaulters. Its linear boundary is a poor fit for the imbalanced data. |
| Decision Tree | Best accuracy, AUC, F1 and MCC. A depth limit of 6 kept it from overfitting, and it captured non-linear patterns such as repayment-delay thresholds. |
| KNN | Middling results with the lowest precision among the high-accuracy models. Sensitive to scaling and to the 23-dimensional feature space. |
| Naive Bayes | Lowest accuracy (0.61) but highest recall (0.75). It flags many defaulters but also many false alarms, because its independence assumption fails for the correlated bill and payment features. |
| Support Vector Machine | Accuracy and precision close to Logistic Regression, with slightly better recall. Trained on a sample, which likely limited its performance. |
| **Overall winner** | **Decision Tree**: best on Accuracy, AUC, F1 and MCC. If catching defaulters matters most, Naive Bayes is the alternative thanks to its recall. |
