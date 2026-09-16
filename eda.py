import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
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

behavior_features = ['Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                     'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
X = df[behavior_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=5, random_state=42)
X_pca = pca.fit_transform(X_scaled)

kmeans = MiniBatchKMeans(n_clusters=1000, batch_size=2048, random_state=42)
df['user_id'] = [f"{c:04d}" for c in kmeans.fit_predict(X_pca)]

# Verify result
print(f"Total Unique User Profiles: {df['user_id'].nunique()}")
print(f"Noise Count: {(df['user_id'] == '-1').sum()}")

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