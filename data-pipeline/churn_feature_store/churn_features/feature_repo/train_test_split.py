from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_parquet("/data/processed_churn_data.parquet")

train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

train_df.to_parquet(
    "/data/train.parquet",
    index=False
)

test_df.to_parquet(
    "/data/test.parquet",
    index=False
)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")