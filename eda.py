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

X = df.drop(columns=["Class"])
y = df["Class"]

# use train test split to evenly split the data, keep = use to simulate real time transaction
# add = add to supabase
keep_1, add_1, keep_2, add_2 = train_test_split(X, y, test_size=0.5, random_state=random_state, stratify=y)

keep_1 = keep_1.reset_index(drop=True)
keep_2 = keep_2.reset_index(drop=True)
add_1 = add_1.reset_index(drop=True)
add_2 = add_2.reset_index(drop=True)
# combine to make the full csv with class column, drop index column
keep = pd.concat([keep_1, keep_2], axis=1).reset_index(drop=True)
add = pd.concat([add_1, add_2], axis=1).reset_index(drop=True)

print(f"Length of dataset to keep: {len(keep)}, Length of dataset to add to supabase: {len(add)}")

# change from float to int (1.0 to 1) for supabase compatibility, follow original dataset format
keep["Time"] = keep_1["Time"].astype(int)
add["Time"] = add_1["Time"].astype(int)

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