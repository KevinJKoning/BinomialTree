# NewBinomialTree/binomial_tree/stopping.py
import math

def check_pre_split_stopping_conditions(
    node_k_sum,
    node_n_sum,
    node_num_samples,
    current_depth,
    min_samples_split,
    max_depth
    ):
    """
    Checks for basic stopping conditions before attempting to find a split.
    This avoids the cost of split-finding for nodes that are already terminal.

    Args:
        node_k_sum (float): Sum of target 'k' values in the current node.
        node_n_sum (float): Sum of exposure 'n' values in the current node.
        node_num_samples (int): Number of data points (rows) in the current node.
        current_depth (int): Current depth of the node in the tree.
        min_samples_split (int): Minimum number of samples required in a node to consider splitting.
        max_depth (int): Maximum allowed depth for the tree.

    Returns:
        str or None: A string describing the reason for stopping, or None if no stopping condition is met.
    """

    if node_num_samples < min_samples_split:
        return f"min_samples_split ({node_num_samples} < {min_samples_split})"

    if current_depth >= max_depth:
        return f"max_depth ({current_depth} >= {max_depth})"

    if node_n_sum == 0:
        return "zero_exposure_sum"

    # Purity checks: if node is pure (all k=0 or all k=n), stop.
    if node_k_sum == 0:
        return "pure_node (k_sum is zero)"
    if node_k_sum == node_n_sum:
        return "pure_node (k_sum equals n_sum)"

    return None


def check_post_split_stopping_condition(
    all_feature_p_values,
    alpha,
    verbose=False,
    node_id_for_logs=None,
    node_depth_for_logs=0
):
    """
    Determines if splitting should stop based on statistical significance (ctree-like).
    This is called *after* finding the best split p-values for all features for a node.
    It applies a Bonferroni correction to the minimum p-value found.

    Args:
        all_feature_p_values (dict): A dictionary mapping feature names to their best p-value for the node.
        alpha (float): The pre-specified significance level for the global null hypothesis test.
        verbose (bool): Flag for detailed logging.
        node_id_for_logs: Identifier for the current node, for logging purposes.
        node_depth_for_logs (int): Depth of the node, for log indentation.

    Returns:
        str or None: A string describing the reason for stopping, or None if splitting should proceed.
    """
    indent = "  " * (node_depth_for_logs + 1)

    if not all_feature_p_values:
        if verbose:
            print(f"{indent}  Stat Stop Check (Node {node_id_for_logs}): No p-values available. Stopping.")
        return "no_significant_split_found"

    num_features = len(all_feature_p_values)
    min_p_value = min(all_feature_p_values.values())

    # Standard Bonferroni correction for the minimum p-value
    adjusted_min_p_value = min(1.0, min_p_value * num_features)

    if verbose:
        best_feature = min(all_feature_p_values, key=all_feature_p_values.get)
        print(f"{indent}  Stat Stop Check (Node {node_id_for_logs}):")
        print(f"{indent}    - Alpha: {alpha}")
        print(f"{indent}    - Num features for correction (m): {num_features}")
        print(f"{indent}    - Best feature: '{best_feature}' (raw p-value: {min_p_value:.5f})")
        print(f"{indent}    - Bonferroni-adjusted min p-value: {adjusted_min_p_value:.5f}")

    # The global null hypothesis is rejected if the adjusted minimum p-value is less than alpha
    if adjusted_min_p_value < alpha:
        if verbose:
            print(f"{indent}    - Decision: Reject H0. Splitting is significant (adj_p < alpha).")
        return None  # Do not stop, proceed with split
    else:
        if verbose:
            print(f"{indent}    - Decision: Fail to reject H0. Stop splitting.")
        return f"stat_stop:min_adj_p_value ({adjusted_min_p_value:.4f} >= {alpha})"


if __name__ == '__main__':
    # Example Usage for pre-split checks
    print("--- Testing pre-split stopping conditions ---")

    # Scenario 1: Min samples
    stop_reason = check_pre_split_stopping_conditions(
        node_k_sum=10, node_n_sum=100, node_num_samples=5, current_depth=3,
        min_samples_split=10, max_depth=5
    )
    print(f"Scenario 1 (Min samples): {stop_reason}") # Expected: min_samples_split

    # Scenario 2: Max depth
    stop_reason = check_pre_split_stopping_conditions(
        node_k_sum=10, node_n_sum=100, node_num_samples=15, current_depth=5,
        min_samples_split=10, max_depth=5
    )
    print(f"Scenario 2 (Max depth): {stop_reason}") # Expected: max_depth

    # Scenario 3: Pure node (k=0)
    stop_reason = check_pre_split_stopping_conditions(
        node_k_sum=0, node_n_sum=100, node_num_samples=20, current_depth=3,
        min_samples_split=10, max_depth=10
    )
    print(f"Scenario 3 (Pure k=0): {stop_reason}")

    # Scenario 4: No reason to stop
    stop_reason = check_pre_split_stopping_conditions(
        node_k_sum=20, node_n_sum=100, node_num_samples=50, current_depth=2,
        min_samples_split=10, max_depth=5
    )
    print(f"Scenario 4 (No stop condition met): {stop_reason}") # Expected: None


    # Example Usage for post-split checks
    print("\n--- Testing post-split (statistical) stopping condition ---")

    # Scenario 5: Significant p-value, should not stop
    p_values = {'feat1': 0.001, 'feat2': 0.1, 'feat3': 0.5}
    alpha = 0.05
    stop_reason_stat = check_post_split_stopping_condition(p_values, alpha, verbose=True, node_depth_for_logs=0)
    print(f"Scenario 5 (Significant p-value): {stop_reason_stat}") # Expected: None. adj_p = 0.001 * 3 = 0.003

    # Scenario 6: Non-significant p-value, should stop
    p_values = {'feat1': 0.02, 'feat2': 0.03, 'feat3': 0.04}
    alpha = 0.05
    stop_reason_stat = check_post_split_stopping_condition(p_values, alpha, verbose=True, node_depth_for_logs=0)
    print(f"Scenario 6 (Non-significant p-value): {stop_reason_stat}") # Expected: stat_stop... adj_p = 0.02 * 3 = 0.06

    # Scenario 7: Borderline p-value, should stop
    p_values = {'feat1': 0.04, 'feat2': 0.8} # min p is 0.04. adj_p = 0.04 * 2 = 0.08
    alpha = 0.05
    stop_reason_stat = check_post_split_stopping_condition(p_values, alpha, verbose=True, node_depth_for_logs=0)
    print(f"Scenario 7 (Borderline p-value): {stop_reason_stat}") # Expected: stat_stop...
