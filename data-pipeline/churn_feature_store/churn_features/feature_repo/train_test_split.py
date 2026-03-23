from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_parquet("/home/antongduy/customer-churn-prediction/data-pipeline/churn_feature_store/churn_features/feature_repo/data/processed_churn_data.parquet")

train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

train_df.to_parquet(
    "/home/antongduy/customer-churn-prediction/data-pipeline/churn_feature_store/churn_features/feature_repo/data/train.parquet",
    index=False
)

test_df.to_parquet(
    "/home/antongduy/customer-churn-prediction/data-pipeline/churn_feature_store/churn_features/feature_repo/data/test.parquet",
    index=False
)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")