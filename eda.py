import pandas as pd

df = pd.read_csv('creditcard.csv')

print(df['Class'].value_counts(normalize=True)*100)

print(df.isnull().sum())

print("===Not fraud===")
print(df[df['Class']==0]['Amount'].sort_values(ascending=False).head(10))
print(df[df['Class'] == 0]['Amount'].describe())
print("===Fraud===")
print(df[df['Class'] == 1]['Amount'].sort_values(ascending=False).head(10))
print(df[df['Class'] == 1]['Amount'].describe())