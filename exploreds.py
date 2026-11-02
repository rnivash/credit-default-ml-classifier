import pandas as pd

df = pd.read_excel('./dataset/default of credit card clients.xls', header=1)

print(df.shape)
print(df.columns.tolist())
print(df.isnull().sum().sum())
print(df.iloc[:, -1].value_counts())

df.to_csv("credit_default.csv", index=False)