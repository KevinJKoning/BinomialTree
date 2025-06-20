# NewBinomialTree/tests/test_harness.py
import math
import time
import xgboost as xgb
import pandas as pd
import numpy as np
from binomial_tree.tree import BinomialDecisionTree
from binomial_tree.utils import get_total_log_likelihood, calculate_p_hat, calculate_binomial_log_likelihood


def _format_metric(value):
    """Formats a float for printing, using scientific notation if it is very small."""
    if isinstance(value, (float, np.floating)):
        if 0 < abs(value) < 0.0001:
            return f"{value:.4e}"
        return f"{value:.6f}"
    return value


def calculate_mae(true_values, predicted_values):
    """Calculates Mean Absolute Error."""
    if len(true_values) != len(predicted_values):
        raise ValueError("Length of true_values and predicted_values must be the same.")
    if not true_values:
        return 0.0

    error_sum = 0.0
    for true, pred in zip(true_values, predicted_values):
        error_sum += abs(true - pred)
    return error_sum / len(true_values)

def calculate_mse(true_values, predicted_values):
    """Calculates Mean Squared Error."""
    if len(true_values) != len(predicted_values):
        raise ValueError("Length of true_values and predicted_values must be the same.")
    if not true_values:
        return 0.0

    error_sum_sq = 0.0
    for true, pred in zip(true_values, predicted_values):
        error_sum_sq += (true - pred)**2
    return error_sum_sq / len(true_values)

def evaluate_predictions(
    tree,
    test_data,
    target_column,
    exposure_column,
    known_p_column=None # If true generating p is known for each test sample
):
    """
    Evaluates the tree's predictions on test data.

    Args:
        tree (BinomialDecisionTree): The trained tree.
        test_data (list of dict): The test dataset.
        target_column (str): Name of the target 'k' column.
        exposure_column (str): Name of the exposure 'n' column.
        known_p_column (str, optional): Name of column containing true 'p' values if available.

    Returns:
        dict: A dictionary of evaluation metrics.
    """
    if not test_data:
        return {
            "mae_p_vs_observed": None,
            "mse_p_vs_observed": None,
            "mae_p_vs_known": None,
            "mse_p_vs_known": None,
            "avg_predicted_p": None,
            "avg_observed_p": None,
            "avg_known_p": None,
            "total_log_likelihood_on_test": None,
            "total_poisson_deviance": None,
            "num_test_samples": 0,
            "num_leaf_nodes": len([nid for nid, n in tree.nodes.items() if n.is_leaf]) if tree.nodes else 0,
            "max_depth_reached": max(n.depth for n in tree.nodes.values()) if tree.nodes else 0
        }

    predicted_p_values = tree.predict_p(test_data)

    observed_p_values = []
    known_p_values_list = [] # Only if known_p_column is provided

    total_k_test = 0
    total_n_test = 0

    test_set_log_likelihood = 0.0
    total_poisson_deviance = 0.0

    for i, row in enumerate(test_data):
        k_i = row[target_column]
        n_i = row[exposure_column]

        if n_i > 0:
            observed_p_values.append(k_i / n_i)
        else:
            observed_p_values.append(0.0)

        if known_p_column and known_p_column in row:
            known_p_values_list.append(row[known_p_column])

        total_k_test += k_i
        total_n_test += n_i

        p_pred_for_row = predicted_p_values[i]
        epsilon = 1e-9 # for log-likelihood calculation stability
        p_pred_for_row = max(epsilon, min(1.0 - epsilon, p_pred_for_row))

        if n_i > 0:
            try:
                test_set_log_likelihood += calculate_binomial_log_likelihood(k_i, n_i, p_pred_for_row)
            except (ValueError, OverflowError) as e:
                print(f"Warning: LL calculation error for k={k_i},n={n_i},p={p_pred_for_row}: {e}")
                test_set_log_likelihood += -float('inf') # Penalize heavily

            # Calculate Poisson Deviance for this sample
            mu_i = n_i * p_pred_for_row # Predicted mean count

            try:
                if k_i == 0:
                    # If k_i is 0, k_i * log(k_i / mu_i) term is 0. Deviance is 2 * mu_i.
                    sample_poisson_deviance = 2 * mu_i
                elif mu_i <= 0: # Should ideally not happen if p_pred_for_row is clamped > 0
                    print(f"Warning: mu_i <= 0 ({mu_i}) for k={k_i},n={n_i},p_pred={p_pred_for_row}. Setting sample_poisson_deviance to large value.")
                    sample_poisson_deviance = 1e12 # Penalize with a large finite number
                else:
                    # k_i > 0 and mu_i > 0
                    log_term_val = k_i * math.log(k_i / mu_i)
                    sample_poisson_deviance = 2 * (log_term_val - (k_i - mu_i))

                if not math.isfinite(sample_poisson_deviance):
                     print(f"Warning: Non-finite Poisson deviance for k={k_i},n={n_i},p_pred={p_pred_for_row},mu_i={mu_i}. Value: {sample_poisson_deviance}. Setting to large value.")
                     total_poisson_deviance += 1e12 # Penalize with a large finite number
                else:
                    total_poisson_deviance += sample_poisson_deviance
            except (ValueError, OverflowError, ZeroDivisionError) as e: # Added ZeroDivisionError
                print(f"Warning: Poisson Deviance calculation error for k={k_i},n={n_i},p_pred={p_pred_for_row},mu_i={mu_i}: {e}")
                total_poisson_deviance += 1e12 # Penalize heavily with a large finite number


    metrics = {}
    metrics["num_test_samples"] = len(test_data)
    metrics["avg_predicted_p"] = sum(predicted_p_values) / len(predicted_p_values) if predicted_p_values.size > 0 else 0

    if observed_p_values:
        metrics["mae_p_vs_observed"] = calculate_mae(observed_p_values, predicted_p_values)
        metrics["mse_p_vs_observed"] = calculate_mse(observed_p_values, predicted_p_values)
        metrics["avg_observed_p"] = sum(observed_p_values) / len(observed_p_values) if observed_p_values else 0
    else:
        metrics["mae_p_vs_observed"] = None
        metrics["mse_p_vs_observed"] = None
        metrics["avg_observed_p"] = None

    if known_p_values_list:
        metrics["mae_p_vs_known"] = calculate_mae(known_p_values_list, predicted_p_values)
        metrics["mse_p_vs_known"] = calculate_mse(known_p_values_list, predicted_p_values)
        metrics["avg_known_p"] = sum(known_p_values_list) / len(known_p_values_list) if known_p_values_list else 0
    else:
        metrics["mae_p_vs_known"] = None
        metrics["mse_p_vs_known"] = None
        metrics["avg_known_p"] = None

    metrics["total_log_likelihood_on_test"] = test_set_log_likelihood
    metrics["total_poisson_deviance"] = total_poisson_deviance
    metrics["num_leaf_nodes"] = len([nid for nid, n in tree.nodes.items() if n.is_leaf]) if tree.nodes else 0
    metrics["max_depth_reached"] = max(n.depth for n in tree.nodes.values()) if tree.nodes else 0
    metrics["total_k_test"] = total_k_test
    metrics["total_n_test"] = total_n_test
    metrics["overall_p_test"] = calculate_p_hat(total_k_test, total_n_test)

    return metrics

def run_test_scenario(
    dataset_name, # Changed from scenario_name to avoid confusion with harness scenario_name
    train_data,
    test_data,
    target_column,
    exposure_column,
    feature_columns,
    tree_params,
    feature_types=None,
    known_p_column=None,
    verbose=True
):
    """
    Runs a full test scenario: train a tree and evaluate it.

    Args:
        dataset_name (str): Name of the dataset/scenario.
        train_data (list of dict): Training data.
        test_data (list of dict): Test data.
        target_column (str): Name of the target 'k' column.
        exposure_column (str): Name of the exposure 'n' column.
        feature_columns (list of str): List of feature column names.
        tree_params (dict): Parameters for BinomialDecisionTree.
        feature_types (dict, optional): Mapping of feature names to 'numerical' or 'categorical'.
        known_p_column (str, optional): Name of column in test_data holding true 'p' values.
        verbose (bool): If True, prints information during the run.

    Returns:
        dict: A dictionary containing tree parameters, training time, and evaluation metrics.
    """
    if verbose:
        print(f"--- Running Test Scenario: {dataset_name} ---")
        print(f"Tree Params: {tree_params}")
        print(f"Training data size: {len(train_data)}, Test data size: {len(test_data)}")

    # Ensure 'epsilon_stopping' is passed if it's in tree_params, otherwise tree uses default
    # This is just to be explicit if test_accuracy.py is defining it.
    # tree_params_for_init = tree_params.copy() # No longer needed if tree_params is already fine

    # Pass the verbose flag from run_test_scenario to the tree constructor
    # tree_params may or may not contain 'verbose', so we pass it explicitly.
    # If 'verbose' is in tree_params, this explicit one will likely take precedence or cause an error
    # if BinomialDecisionTree doesn't expect it twice.
    # The BinomialDecisionTree now has verbose in its __init__, so this is fine.
    current_tree_params_for_init = tree_params.copy()
    # The verbose flag for the tree's internal logging should come from the test_harness's verbose flag
    tree = BinomialDecisionTree(**current_tree_params_for_init, verbose=verbose)

    start_time = time.time()
    try:
        tree.fit(
            data=train_data,
            target_column=target_column,
            exposure_column=exposure_column,
            feature_columns=feature_columns,
            feature_types=feature_types
        )
    except Exception as e:
        print(f"!!!!!! ERROR during tree.fit for scenario: {dataset_name} !!!!!!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "dataset_name": dataset_name, "error": str(e), "details": "Fitting failed",
            "tree_params": tree_params, "training_time_seconds": time.time() - start_time,
            "evaluation": {}, "num_nodes_total": 0
        }

    end_time = time.time()
    training_time = end_time - start_time

    if verbose:
        print(f"Training completed in {training_time:.2f} seconds.")
        # tree.print_tree() # Optionally print the tree structure

    evaluation_results = evaluate_predictions(
        tree,
        test_data,
        target_column,
        exposure_column,
        known_p_column
    )

    if verbose and evaluation_results:
        print("Evaluation Results:")
        for key, value in evaluation_results.items():
            print(f"  {key}: {_format_metric(value)}")
        print("--- Scenario End ---")

    results = {
        "dataset_name": dataset_name,
        "tree_params": tree_params,
        "training_time_seconds": training_time,
        "evaluation": evaluation_results,
        "num_nodes_total": len(tree.nodes) if tree.nodes else 0,
    }
    return results


def prepare_data_for_xgboost(
    train_data,
    test_data,
    feature_columns,
    target_column,
    exposure_column,
    feature_types
):
    """
    Prepares data for an XGBoost model.
    - Converts lists of dicts to Pandas DataFrames.
    - One-hot encodes categorical features.
    - Uses exposure ('n') as a feature.
    - Sets the target as rate ('k'/'n').
    """
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Use a copy to avoid modifying the original list
    feature_columns_xgb = feature_columns.copy()
    if exposure_column not in feature_columns_xgb:
        feature_columns_xgb.append(exposure_column)

    X_train_raw = train_df[feature_columns_xgb].copy()
    X_test_raw = test_df[feature_columns_xgb].copy()

    # Impute numerical NaNs first
    numerical_features = [f for f, t in feature_types.items() if t == 'numerical']
    for col in numerical_features:
        if X_train_raw[col].isnull().any():
            median_val = X_train_raw[col].median()
            X_train_raw.loc[:, col] = X_train_raw[col].fillna(median_val)
            X_test_raw.loc[:, col] = X_test_raw[col].fillna(median_val)

    # One-hot encode
    categorical_features = [f for f, t in feature_types.items() if t == 'categorical']
    X_train_encoded = pd.get_dummies(X_train_raw, columns=categorical_features, dummy_na=True, dtype=float)
    X_test_encoded = pd.get_dummies(X_test_raw, columns=categorical_features, dummy_na=True, dtype=float)

    # Align columns - crucial for when test set has categories not in train set or vice-versa
    X_train_final, X_test_final = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

    # Create target variable (rate)
    epsilon = 1e-9
    y_train = (train_df[target_column] / (train_df[exposure_column].replace(0, 1) + epsilon)).fillna(0)

    return X_train_final, y_train, X_test_final


def evaluate_xgboost_predictions(
    predicted_p_values,
    test_data,
    target_column,
    exposure_column,
    known_p_column=None,
    model_params=None
):
    """
    Evaluates an XGBoost model's predictions. This is similar to evaluate_predictions
    but adapted for a generic model that outputs p-hat values directly.
    """
    if not test_data:
        return {
            "mae_p_vs_observed": None, "mse_p_vs_observed": None,
            "mae_p_vs_known": None, "mse_p_vs_known": None,
            "avg_predicted_p": None, "avg_observed_p": None, "avg_known_p": None,
            "total_log_likelihood_on_test": None, "total_poisson_deviance": None,
            "num_test_samples": 0,
        }

    observed_p_values = []
    known_p_values_list = []

    total_k_test = 0
    total_n_test = 0
    test_set_log_likelihood = 0.0
    total_poisson_deviance = 0.0

    for i, row in enumerate(test_data):
        k_i = row[target_column]
        n_i = row[exposure_column]

        if n_i > 0:
            observed_p_values.append(k_i / n_i)
        else:
            observed_p_values.append(0.0)

        if known_p_column and known_p_column in row:
            known_p_values_list.append(row[known_p_column])

        total_k_test += k_i
        total_n_test += n_i

        p_pred_for_row = predicted_p_values[i]
        epsilon = 1e-9
        p_pred_for_row = max(epsilon, min(1.0 - epsilon, p_pred_for_row))

        if n_i > 0:
            try:
                test_set_log_likelihood += calculate_binomial_log_likelihood(k_i, n_i, p_pred_for_row)
            except (ValueError, OverflowError) as e:
                print(f"Warning: XGBoost LL calculation error for k={k_i},n={n_i},p={p_pred_for_row}: {e}")
                test_set_log_likelihood += -float('inf')

            mu_i = n_i * p_pred_for_row
            try:
                if k_i == 0:
                    sample_poisson_deviance = 2 * mu_i
                elif mu_i <= 0:
                    sample_poisson_deviance = 1e12
                else:
                    log_term_val = k_i * math.log(k_i / mu_i)
                    sample_poisson_deviance = 2 * (log_term_val - (k_i - mu_i))

                if not math.isfinite(sample_poisson_deviance):
                     total_poisson_deviance += 1e12
                else:
                    total_poisson_deviance += sample_poisson_deviance
            except (ValueError, OverflowError, ZeroDivisionError) as e:
                print(f"Warning: XGBoost Poisson Deviance calc error for k={k_i},n={n_i},p_pred={p_pred_for_row},mu_i={mu_i}: {e}")
                total_poisson_deviance += 1e12

    metrics = {}
    metrics["num_test_samples"] = len(test_data)
    metrics["avg_predicted_p"] = np.mean(predicted_p_values) if predicted_p_values.size > 0 else 0

    if observed_p_values:
        metrics["mae_p_vs_observed"] = calculate_mae(observed_p_values, predicted_p_values)
        metrics["mse_p_vs_observed"] = calculate_mse(observed_p_values, predicted_p_values)
        metrics["avg_observed_p"] = sum(observed_p_values) / len(observed_p_values) if observed_p_values else 0
    else:
        metrics["mae_p_vs_observed"] = None
        metrics["mse_p_vs_observed"] = None
        metrics["avg_observed_p"] = None

    if known_p_values_list:
        metrics["mae_p_vs_known"] = calculate_mae(known_p_values_list, predicted_p_values)
        metrics["mse_p_vs_known"] = calculate_mse(known_p_values_list, predicted_p_values)
        metrics["avg_known_p"] = sum(known_p_values_list) / len(known_p_values_list) if known_p_values_list else 0
    else:
        metrics["mae_p_vs_known"] = None
        metrics["mse_p_vs_known"] = None
        metrics["avg_known_p"] = None

    metrics["total_log_likelihood_on_test"] = test_set_log_likelihood
    metrics["total_poisson_deviance"] = total_poisson_deviance
    metrics["total_k_test"] = total_k_test
    metrics["total_n_test"] = total_n_test
    metrics["overall_p_test"] = calculate_p_hat(total_k_test, total_n_test)

    if model_params:
        metrics["n_estimators"] = model_params.get("n_estimators")
        metrics["max_depth_reached"] = model_params.get("max_depth")

    return metrics


def run_xgboost_peer_test(
    dataset_name,
    train_data,
    test_data,
    target_column,
    exposure_column,
    feature_columns,
    feature_types,
    known_p_column=None,
    xgboost_params=None,
    verbose=True
):
    """
    Runs a peer test scenario using XGBoost.
    """
    if verbose:
        print(f"--- Running XGBoost Peer Test Scenario: {dataset_name} ---")

    if xgboost_params is None:
        xgboost_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }
    if verbose:
        print(f"XGBoost Params: {xgboost_params}")

    start_time_prep = time.time()
    X_train, y_train, X_test = prepare_data_for_xgboost(
        train_data, test_data, feature_columns, target_column, exposure_column, feature_types
    )
    prep_time = time.time() - start_time_prep

    model = xgb.XGBRegressor(**xgboost_params)

    start_time_fit = time.time()
    try:
        model.fit(X_train, y_train, verbose=False)
    except Exception as e:
        print(f"!!!!!! ERROR during XGBoost model.fit for scenario: {dataset_name} !!!!!!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "dataset_name": f"{dataset_name}_XGBoost_Error",
            "error": str(e),
            "details": "XGBoost Fitting failed",
            "tree_params": xgboost_params,
            "training_time_seconds": time.time() - start_time_fit,
            "evaluation": {},
        }

    training_time = time.time() - start_time_fit

    if verbose:
        print(f"Data prep in {prep_time:.2f}s. Training in {training_time:.2f}s.")

    predicted_p_values = model.predict(X_test)
    predicted_p_values = np.clip(predicted_p_values, 0.0, 1.0)

    evaluation_results = evaluate_xgboost_predictions(
        predicted_p_values,
        test_data,
        target_column,
        exposure_column,
        known_p_column,
        model_params=xgboost_params
    )

    if verbose and evaluation_results:
        print("XGBoost Evaluation Results:")
        for key, value in evaluation_results.items():
            print(f"  {key}: {_format_metric(value)}")
        print("--- XGBoost Scenario End ---")

    results = {
        "dataset_name": f"{dataset_name}_XGBoost",
        "tree_params": xgboost_params,
        "training_time_seconds": training_time,
        "evaluation": evaluation_results,
    }
    return results


if __name__ == '__main__':
    # This is a placeholder for a simple test.
    # Actual tests would be run from test_accuracy.py using generated datasets.
    print("Test Harness Self-Test (Illustrative)")

    # Create a tiny mock dataset
    mock_train_data = [
        {'f1': 1, 'k': 1, 'n': 10, 'true_p': 0.1},
        {'f1': 2, 'k': 2, 'n': 10, 'true_p': 0.2},
        {'f1': 3, 'k': 1, 'n': 10, 'true_p': 0.1},
        {'f1': 10, 'k': 8, 'n': 10, 'true_p': 0.8},
        {'f1': 11, 'k': 9, 'n': 10, 'true_p': 0.9},
        {'f1': 12, 'k': 7, 'n': 10, 'true_p': 0.7},
    ]
    mock_test_data = [
        {'f1': 1.5, 'k': 1, 'n': 20, 'true_p': 0.1},
        {'f1': 2.5, 'k': 2, 'n': 20, 'true_p': 0.2},
        {'f1': 10.5, 'k': 18, 'n': 20, 'true_p': 0.8},
        {'f1': 11.5, 'k': 15, 'n': 20, 'true_p': 0.9},
    ]

    params = {
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_depth": 2,
        "alpha": 0.05,
        "confidence_level": 0.95
    }

    results = run_test_scenario(
        dataset_name="Mock_Simple_Numerical",
        train_data=mock_train_data,
        test_data=mock_test_data,
        target_column='k',
        exposure_column='n',
        feature_columns=['f1'],
        feature_types={'f1': 'numerical'},
        tree_params=params,
        known_p_column='true_p',
        verbose=True
    )

    assert results is not None
    if "evaluation" in results and results["evaluation"]: # Check if evaluation was successful
      assert results["evaluation"]["num_test_samples"] == len(mock_test_data)
      if results["evaluation"].get("mae_p_vs_known") is not None: # Check if metric exists
           print(f"Binomial Tree MAE vs Known P: {results['evaluation']['mae_p_vs_known']:.4f}")
    elif "error" in results:
        print(f"Test scenario failed with error: {results['error']}")

    print("\n--- Running XGBoost Peer Test ---")
    xgb_results = run_xgboost_peer_test(
        dataset_name="Mock_Simple_Numerical_XGBoost",
        train_data=mock_train_data,
        test_data=mock_test_data,
        target_column='k',
        exposure_column='n',
        feature_columns=['f1'],
        feature_types={'f1': 'numerical'},
        known_p_column='true_p',
        verbose=True
    )
    assert xgb_results is not None
    if "evaluation" in xgb_results and xgb_results["evaluation"]:
        if xgb_results["evaluation"].get("mae_p_vs_known") is not None:
            print(f"XGBoost MAE vs Known P: {xgb_results['evaluation']['mae_p_vs_known']:.4f}")
    elif "error" in xgb_results:
        print(f"XGBoost peer test failed with error: {xgb_results['error']}")

    print("\nTest Harness Self-Test Completed.")
