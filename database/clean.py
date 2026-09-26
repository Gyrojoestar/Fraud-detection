from operator import index

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("creditcard.csv")

df.drop(columns=["Time"],inplace=True)

df.columns = df.columns.str.lower()

keep, supa = train_test_split(df, train_size=0.2, stratify=df["class"])

print(keep.count(), supa.count())

keep.to_csv("card_cleaned_keep.csv", index=False)
supa.to_csv("card_cleaned_supa.csv", index=False)