# NewBinomialTree/tests/test_harness.py
import math
import time
from binomial_tree.tree import BinomialDecisionTree
from binomial_tree.utils import get_total_log_likelihood, calculate_p_hat, calculate_binomial_log_likelihood

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
            data_list_of_dicts=train_data, # Corrected argument name
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
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print("--- Scenario End ---")

    results = {
        "dataset_name": dataset_name,
        "tree_params": tree_params,
        "training_time_seconds": training_time,
        "evaluation": evaluation_results,
        "num_nodes_total": len(tree.nodes) if tree.nodes else 0,
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
        "confidence_level": 0.95,
        "min_n_sum_for_statistical_stop": 5, # Low for small data
        "relative_width_factor": 1.0,
        "min_likelihood_gain": 0.001,
        "epsilon_stopping": 1e-6 # Added to match BinomialDecisionTree __init__
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
           print(f"MAE vs Known P: {results['evaluation']['mae_p_vs_known']:.4f}")
    elif "error" in results:
        print(f"Test scenario failed with error: {results['error']}")
    
    print("\nTest Harness Self-Test Completed.")