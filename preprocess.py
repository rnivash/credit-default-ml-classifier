import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib, os

df = pd.read_excel('./dataset/default of credit card clients.xls', header=1)
df = df.drop(columns=['ID'])
df = df.rename(columns={'default payment next month': 'target'})

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/scaler.joblib')

# save raw (unscaled) test data for the Streamlit app
test_df = X_test.copy()
test_df['target'] = y_test
test_df.to_csv('test_data.csv', index=False)