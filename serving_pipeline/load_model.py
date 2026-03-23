import mlflow
import os


def load_model(model_uri: str = "runs:/36a7418142cb4ae39d9ac44d06436731/customer_churn_model_v2"):
    """
    Load model from MLflow
    
    Args:
        model_uri: MLflow model URI (default: latest model)
    
    Returns:
        Loaded MLflow model
    """
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

    # 1. Configure MLflow Tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    print(MLFLOW_TRACKING_URI)
    
    
    # 3. Load model
    model = mlflow.pyfunc.load_model(model_uri)
    
    print("Model loaded successfully!")
    
    return model


if __name__ == "__main__":
    # Test loading model
    model = load_model("runs:/36a7418142cb4ae39d9ac44d06436731/customer_churn_model_v2")
    
    # Get feature names
    custom_model_instance = model.unwrap_python_model()
    print("Feature names:", custom_model_instance.feature_names)
