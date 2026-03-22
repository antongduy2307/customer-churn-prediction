import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

df = pd.read_csv('/home/antongduy/customer-churn-prediction/data-pipeline/data/raw/train_period_1.csv')

df_processed = df.copy()

df.columns

# Drop rows with missing values
df_processed = df_processed.dropna()

# Count how many rows are duplicates based on CustomerID
duplicate_count = df_processed.duplicated(subset=['CustomerID']).sum()
print(f"Number of duplicate CustomerIDs: {duplicate_count}")

# View the actual duplicate rows
duplicates = df_processed[df_processed.duplicated(subset=['CustomerID'], keep=False)]
print(duplicates.sort_values(by='CustomerID').head())

# Keep the first occurrence and remove subsequent ones
df_processed = df_processed.drop_duplicates(subset=['CustomerID'], keep='first')

df_processed['CustomerID'] = df_processed['CustomerID'].astype("int64")

df_processed.info()

int_columns = ['Age', 'Tenure', 'Support Calls', 'Last Interaction']
for col in int_columns:
    df_processed[col] = df_processed[col].astype(int)

categorical_features  = ['Gender', 'Subscription Type', 'Contract Length']
for col in categorical_features :
    print(f"{col}: {df_processed[col].unique()}")

df_encoded = pd.get_dummies(df_processed, columns=categorical_features , drop_first=True)

numerical_features  = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 
                  'Payment Delay', 'Total Spend']

for col in numerical_features :
    Q1 = df_processed[col].quantile(0.25)
    Q3 = df_processed[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df_processed)*100:.2f}%)")

df_processed['Tenure_Age_Ratio'] = df_processed['Tenure'] / (df_processed['Age'] + 1)
df_processed['Spend_per_Usage'] = df_processed['Total Spend'] / (df_processed['Usage Frequency'] + 1)
df_processed['Support_Calls_per_Tenure'] = df_processed['Support Calls'] / (df_processed['Tenure'] + 1)

# Create customer segments based on spending
df_processed['Spending_Group'] = pd.qcut(df_processed['Total Spend'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

# Create tenure groups
df_processed['Tenure_Group'] = pd.cut(df_processed['Tenure'], 
                                      bins=[0, 12, 24, 36, 100], 
                                      labels=['<1yr', '1-2yr', '2-3yr', '3+yr'])

# Add categorical features to the list
categorical_features.extend(['Spending_Group', 'Tenure_Group'])
df_processed.head()


out_dir = '/home/antongduy/customer-churn-prediction/data-pipeline/data/processed'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'df_processed.csv')

df_processed.to_csv(out_path, index=False)
print(f"Exported df_processed to {out_path}")