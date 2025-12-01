# 1. IMPORTS
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm  # For a clean progress bar

# Scikit-learn Imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
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
    """

    # --- Self-Aware Pathing ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

    # Data parameters
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    TARGET_VARIABLE = "Result"

    # Experimental parameters
    # Note: 'statistical' uses the SAME raw file as 'baseline',
    # but we will treat it differently in the preprocessing step.
    DATASETS = {
        "baseline": os.path.join(DATA_DIR, "baseline.csv"),
        "ratio": os.path.join(DATA_DIR, "ratio.csv"),
        "threshold": os.path.join(DATA_DIR, "threshold.csv"),
        "statistical": os.path.join(DATA_DIR, "baseline.csv"),  # Points to raw data
    }

    MODELS = {
        "LogisticRegression": LogisticRegression,
        "KNN": KNeighborsClassifier,
        "RandomForest": RandomForestClassifier,
    }

    # Hyperparameter Grids for Tuning
    PARAM_GRIDS = {
        "LogisticRegression": {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["liblinear", "lbfgs"],
            "class_weight": ["balanced", None],
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        "RandomForest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "class_weight": ["balanced", "balanced_subsample", None],
        },
    }

    NUM_RUNS = 10
    TEST_SIZE = 0.15  # 85/15 split

    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)


# 3. CORE FUNCTIONS


def prepare_data_for_run(
    dataset_path, dataset_name, target_variable, test_size, random_state
):
    """
    Loads data and applies specific preprocessing based on the STRATEGY name.
    """
    # 1. Load Data
    df = pd.read_csv(dataset_path)

    # 2. Encode Target Variable
    df[target_variable] = df[target_variable].map({"positive": 1, "negative": 0})

    # 3. Separate Features (X) and Target (y)
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    # 4. Split Data (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 5. Dynamic Feature Identification
    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

    # 6. Define Preprocessing Strategy
    # LOGIC SWITCH: If strategy is "statistical", use PowerTransformer (Yeo-Johnson)
    # Otherwise, use standard scaling.
    if dataset_name == "statistical":
        print(
            f"   -> Applying PowerTransformer (Yeo-Johnson) for Statistical FE Strategy..."
        )
        numerical_transformer = PowerTransformer(method="yeo-johnson")
    else:
        numerical_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",
    )

    # 7. Fit & Transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test


def train_and_evaluate(
    X_train, y_train, X_test, y_test, model_name, model_class, random_state
):
    """Trains a model using GridSearchCV and returns metrics."""

    # Initialize base model
    if model_class.__name__ in ["LogisticRegression", "RandomForestClassifier"]:
        model = model_class(random_state=random_state)
    else:
        model = model_class()

    # Get param grid
    param_grid = Config.PARAM_GRIDS[model_name]

    # Initialize GridSearch
    # Using 'f1' as scoring since we want to optimize the balance
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=5,  # 3-fold internal CV for speed
        n_jobs=-1,  # Use all CPU cores
        verbose=0,
    )

    # Fit (Tuning happens here)
    grid_search.fit(X_train, y_train)

    # Use best estimator
    best_model = grid_search.best_estimator_

    # Predict
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "best_params": str(grid_search.best_params_),  # Save params for analysis
    }
    return metrics


# 4. MAIN EXECUTION BLOCK


def run_pipeline():
    """
    The main function to orchestrate the entire experimental pipeline.
    """
    print("--- Starting Modeling Pipeline (Enhanced with Statistical FE & Tuning) ---")
    start_time = time.time()

    all_results = []

    # Calculate total iterations
    total_iterations = len(Config.DATASETS) * len(Config.MODELS) * Config.NUM_RUNS
    progress_bar = tqdm(total=total_iterations, desc="Executing Runs")

    # The main experimental loop
    for dataset_name, dataset_path in Config.DATASETS.items():
        for model_name, model_class in Config.MODELS.items():
            for i in range(Config.NUM_RUNS):
                run_id = i + 1
                random_state = run_id

                try:
                    # Execute a single run
                    # Pass dataset_name to trigger the "statistical" logic check
                    X_train, X_test, y_train, y_test = prepare_data_for_run(
                        dataset_path,
                        dataset_name,
                        Config.TARGET_VARIABLE,
                        Config.TEST_SIZE,
                        random_state,
                    )

                    metrics = train_and_evaluate(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        model_name,
                        model_class,
                        random_state,
                    )

                    # Log the results
                    run_result = {
                        "dataset": dataset_name,
                        "model": model_name,
                        "run_id": run_id,
                        **metrics,
                    }
                    all_results.append(run_result)

                except Exception as e:
                    print(
                        f"\nERROR during run: {dataset_name}/{model_name}/run_{run_id}"
                    )
                    print(f"Error message: {e}")

                progress_bar.update(1)

    progress_bar.close()

    # Save the final results
    results_df = pd.DataFrame(all_results)
    output_path = os.path.join(Config.RESULTS_DIR, "model_performance_tuned.csv")
    results_df.to_csv(output_path, index=False)

    end_time = time.time()
    print(f"\n--- Pipeline Finished ---")
    print(f"Total runs executed: {len(all_results)}")
    print(f"Results saved to: {output_path}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    run_pipeline()
