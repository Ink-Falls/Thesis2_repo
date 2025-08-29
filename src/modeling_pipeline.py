# 1. IMPORTS
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm  # For a clean progress bar

# Scikit-learn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import warnings

warnings.filterwarnings("ignore")


# 2. CONFIGURATION
class Config:
    """
    A configuration class to hold all experimental parameters.
    This is the "control panel" for our entire pipeline.
    """

    # --- Self-Aware Pathing ---
    # Get the absolute path to the directory where this script is located (src/)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Get the project root by going one level up from the script's directory
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    # Data parameters
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    TARGET_VARIABLE = "Result"

    # Experimental parameters
    DATASETS = {
        "baseline": os.path.join(DATA_DIR, "baseline.csv"),
        "ratio": os.path.join(DATA_DIR, "ratio.csv"),
        "threshold": os.path.join(DATA_DIR, "threshold.csv"),
    }
    MODELS = {
        "LogisticRegression": LogisticRegression,
        "KNN": KNeighborsClassifier,
        "RandomForest": RandomForestClassifier,
    }
    NUM_RUNS = 10
    TEST_SIZE = 0.15  # 85/15 split

    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)


# 3. CORE FUNCTIONS


def prepare_data_for_run(dataset_path, target_variable, test_size, random_state):
    """
    Loads data, encodes the target, splits into train/test, and applies column-specific
    preprocessing (scaling for numerical, one-hot encoding for categorical).
    This is the complete data preparation pipeline for a single run.
    """
    # 1. Load Data
    df = pd.read_csv(dataset_path)

    # 2. Encode Target Variable
    df[target_variable] = df[target_variable].map({"positive": 1, "negative": 0})

    # 3. Separate Features (X) and Target (y)
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    # 4. Split Data into Training and Testing Split (85/15)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 5. Dynamic Feature Identification
    # Identify numerical and categorical features present in THIS specific dataset
    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()

    # Infer categorical features by finding non-numeric columns
    categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

    print(f"\n--- Dynamic Feature Detection for {os.path.basename(dataset_path)} ---")
    print(f"Numerical: {numerical_features}")
    print(f"Categorical: {categorical_features}")

    # 6. Define Preprocessing Steps for Different Column Types
    # Identify numerical features dynamically
    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()

    # Create preprocessing pipelines for both numerical and categorical data
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # Use ColumnTransformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",  # Keep other columns if any
    )

    # 7. Fit the preprocessor on the training data and transform both sets
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test


def train_and_evaluate(X_train, y_train, X_test, y_test, model_class, random_state):
    """Trains a model and returns its performance metrics."""
    # Special handling for models that don't accept random_state at init
    if model_class.__name__ in ["LogisticRegression", "RandomForestClassifier"]:
        model = model_class(random_state=random_state)
    else:
        model = model_class()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }
    return metrics


# 4. MAIN EXECUTION BLOCK


def run_pipeline():
    """
    The main function to orchestrate the entire experimental pipeline.
    """
    print("--- Starting Modeling Pipeline ---")
    start_time = time.time()

    all_results = []

    # Calculate total iterations for the progress bar
    total_iterations = len(Config.DATASETS) * len(Config.MODELS) * Config.NUM_RUNS
    progress_bar = tqdm(total=total_iterations, desc="Executing Runs")

    # The main experimental loop
    for dataset_name, dataset_path in Config.DATASETS.items():
        for model_name, model_class in Config.MODELS.items():
            for i in range(Config.NUM_RUNS):
                run_id = i + 1
                # The random_state is unique for each run to ensure variability
                random_state = run_id

                try:
                    # Execute a single run
                    X_train, X_test, y_train, y_test = prepare_data_for_run(
                        dataset_path,
                        Config.TARGET_VARIABLE,
                        Config.TEST_SIZE,
                        random_state,
                    )

                    metrics = train_and_evaluate(
                        X_train, y_train, X_test, y_test, model_class, random_state
                    )

                    # Log the results
                    run_result = {
                        "dataset": dataset_name,
                        "model": model_name,
                        "run_id": run_id,
                        **metrics,  # Merge metrics dictionary
                    }
                    all_results.append(run_result)

                except Exception as e:
                    print(
                        f"\nERROR during run: {dataset_name}/{model_name}/run_{run_id}"
                    )
                    print(f"Error message: {e}")

                progress_bar.update(1)

    progress_bar.close()

    # Save the final results to a CSV file
    results_df = pd.DataFrame(all_results)
    output_path = os.path.join(Config.RESULTS_DIR, "model_performance.csv")
    results_df.to_csv(output_path, index=False)

    end_time = time.time()
    print(f"\n--- Pipeline Finished ---")
    print(f"Total runs executed: {len(all_results)}")
    print(f"Results saved to: {output_path}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    run_pipeline()
