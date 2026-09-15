import pandas as pd
from sklearn.model_selection import train_test_split
import os

random_state=42

df = pd.read_csv('creditcard.csv')

print(df['Class'].value_counts(normalize=True)*100)

print(df.isnull().sum())

print("===Not fraud===")
print(df[df['Class']==0]['Amount'].sort_values(ascending=False).head(10))
print(df[df['Class'] == 0]['Amount'].describe())
print("===Fraud===")
print(df[df['Class'] == 1]['Amount'].sort_values(ascending=False).head(10))
print(df[df['Class'] == 1]['Amount'].describe())

df = df.sort_values(by="Time").reset_index(drop=True)

split_idx = int(len(df) * 0.90)

keep = df.iloc[:split_idx]
add = df.iloc[split_idx:]

keep['Time'] = keep['Time'].astype(int)
add['Time'] = add['Time'].astype(int)

keep.columns = keep.columns.str.lower()
add.columns = add.columns.str.lower()

print(f"Length of dataset to keep: {len(keep)}, Length of dataset to add to supabase: {len(add)}")

df_list = [keep, add]
output_path = ["creditCardSupabase.csv", "creditCardRealTime.csv"]

# loop through the 2 list and assign the df to a csv file
# create file only if doesnt exist, else skip
# always print message (fail / success)
for i, df_list_elem in enumerate(df_list):
    if not os.path.exists(output_path[i]):
        df_list_elem.to_csv(output_path[i], index=False)
        print(f"file {output_path[i]} was created successfully.")
    else:
        print(f"file {output_path[i]} already exists. No files were created or changed.")