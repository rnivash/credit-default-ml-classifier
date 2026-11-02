import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

df = pd.read_excel('./dataset/default of credit card clients.xls', header=1)
df = df.drop(columns=['ID']).rename(columns={'default payment next month': 'target'})
X, y = df.drop(columns=['target']), df['target']

# MUST match your preprocess file (same test_size and random_state)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7, stratify=y)

scaler = joblib.load('model/scaler.joblib')
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# SVM is slow on 24k rows, so train it on a sample
svm_idx = X_train.sample(8000, random_state=7).index
svm_pos = X_train.index.get_indexer(svm_idx)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=7),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=7),
}

results = []
for name, model in models.items():
    if name == 'Support Vector Machine':
        model.fit(X_train_s[svm_pos], y_train.iloc[svm_pos])
    else:
        model.fit(X_train_s, y_train)

    pred = model.predict(X_test_s)
    prob = model.predict_proba(X_test_s)[:, 1]

    results.append({
        'ML Model Name': name,
        'Accuracy': accuracy_score(y_test, pred),
        'AUC': roc_auc_score(y_test, prob),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'F1': f1_score(y_test, pred),
        'MCC': matthews_corrcoef(y_test, pred),
    })
    joblib.dump(model, f"model/{name.lower().replace(' ', '_')}.joblib")

res = pd.DataFrame(results).round(4)
print(res.to_string(index=False))
res.to_csv('results.csv', index=False)