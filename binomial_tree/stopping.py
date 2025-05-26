# NewBinomialTree/binomial_tree/stopping.py
import math
from .utils import calculate_wilson_score_interval, calculate_p_hat

def check_node_stopping_conditions(
    node_k_sum,
    node_n_sum,
    node_num_samples,
    current_depth,
    min_samples_split,
    max_depth,
    confidence_level,
    min_n_sum_for_statistical_stop,
    relative_width_factor=0.5,  # if (CI_high - CI_low)/p_hat > this factor
    epsilon=1e-6           # threshold for "extremely close" to 0 or 1
    ):
    """
    Determines if a node should stop splitting and become a leaf based on various criteria.
    This is typically called *before* attempting to find the best split for a node.

    Args:
        node_k_sum (int): Sum of target 'k' values in the current node.
        node_n_sum (int): Sum of exposure 'n' values in the current node.
        node_num_samples (int): Number of data points (rows) in the current node.
        current_depth (int): Current depth of the node in the tree.
        min_samples_split (int): Minimum number of samples required in a node to consider splitting.
        max_depth (int): Maximum allowed depth for the tree.
        confidence_level (float): Confidence level for Wilson score interval (e.g., 0.95).
        min_n_sum_for_statistical_stop (int): Minimum total exposure 'n_sum' to apply statistical stopping.
                                               Below this, other criteria like min_samples or depth dominate.

        relative_width_factor (float): If p_hat is in a moderate range and (CI_high−CI_low)/p_hat > this factor,
                                       the interval is considered too wide relative to p, stopping the node.

    Returns:
        str or None: A string describing the reason for stopping, or None if no stopping condition is met.
    """

    if node_num_samples < min_samples_split:
        return f"min_samples_split ({node_num_samples} < {min_samples_split})"

    if current_depth >= max_depth:
        return f"max_depth ({current_depth} >= {max_depth})"

    if node_n_sum == 0: # No exposure, cannot estimate p
        return "zero_exposure_sum"

    # Purity checks: if node is pure (all k=0 or all k=n), stop.
    # This is checked after basic sample/depth/exposure checks.
    # The case node_n_sum == 0 is handled above.
    if node_n_sum > 0:
        if node_k_sum == 0:
            # All outcomes are 0 for the given exposure.
            return "pure_node (k_sum == 0)"
        if node_k_sum == node_n_sum:
            # All outcomes are 1 for the given exposure.
            return "pure_node (k_sum == n_sum)"


    # Statistical stopping criterion based on Wilson score interval
    if node_n_sum >= min_n_sum_for_statistical_stop:
        p_hat = calculate_p_hat(node_k_sum, node_n_sum)
        low, high = calculate_wilson_score_interval(node_k_sum, node_n_sum, confidence_level)
        interval_width = high - low

        # (Absolute‐precision stop removed; only relative interval width is enforced.)


        # Stop if the interval is too wide, indicating high uncertainty in p
        # This is the core "confidence interval too large" criterion.

        # Only apply when p_hat not extremely close to 0 or 1
        if epsilon < p_hat < (1.0 - epsilon):
            rel_width = interval_width / p_hat
            if rel_width > relative_width_factor:
                return f"stat_stop:relative_interval_width (width={interval_width:.3f}, p_hat={p_hat:.3f}, rel_width={rel_width:.2f} > {relative_width_factor})"

    # If no other condition met, node purity (based on likelihood gain from a split)
    # will be the deciding factor. If no split improves likelihood, it stops.
    # This is handled in the main tree fitting loop.

    return None


if __name__ == '__main__':
    # Example Usage
    print("Testing stopping conditions:")

    # Scenario 1: Min samples
    stop_reason = check_node_stopping_conditions(
        node_k_sum=10, node_n_sum=100, node_num_samples=5, current_depth=3,
        min_samples_split=10, max_depth=5, confidence_level=0.95,
        min_n_sum_for_statistical_stop=50, relative_width_factor=0.5
    )
    print(f"Scenario 1 (Min samples): {stop_reason}") # Expected: min_samples_split

    # Scenario 2: Max depth
    stop_reason = check_node_stopping_conditions(
        node_k_sum=10, node_n_sum=100, node_num_samples=15, current_depth=5,
        min_samples_split=10, max_depth=5, confidence_level=0.95,
        min_n_sum_for_statistical_stop=50, relative_width_factor=0.5
    )
    print(f"Scenario 2 (Max depth): {stop_reason}") # Expected: max_depth

    # Scenario 3: Statistically wide interval (relative)
    # k=5, n=50 -> p_hat = 0.1. Wilson interval for (5, 50) at 95% CI is approx (0.037, 0.218). Width = 0.181
    # Relative width = 0.181 / 0.1 = 1.81. If relative_width_factor = 0.5, this should stop.
    stop_reason = check_node_stopping_conditions(
        node_k_sum=5, node_n_sum=50, node_num_samples=20, current_depth=3,
        min_samples_split=10, max_depth=10, confidence_level=0.95,
        min_n_sum_for_statistical_stop=30, relative_width_factor=0.5
    )
    print(f"Scenario 3 (Statistically wide - relative): {stop_reason}")

    # Scenario 4: Statistically precise enough (p_precision_threshold met)
    # k=50, n=1000 -> p_hat = 0.05. Wilson for (50, 1000) is (0.038, 0.065). Width = 0.027
    # Relative width = 0.027 / 0.05 = 0.54. Let relative_width_factor = 1.0 (so no stop by rel width)
    # Let p_precision_threshold = 0.03. Since 0.027 < 0.03, this should *not* stop by this rule.
    # The current implementation has removed precision threshold as a direct stopper.
    stop_reason = check_node_stopping_conditions(
        node_k_sum=50, node_n_sum=1000, node_num_samples=100, current_depth=3,
        min_samples_split=10, max_depth=10, confidence_level=0.95,
        min_n_sum_for_statistical_stop=50, relative_width_factor=1.0
    )
    print(f"Scenario 4 (Statistically precise - should be None): {stop_reason}")

    # Scenario 5: p_hat near 0, high uncertainty
    # k=1, n=100 -> p_hat = 0.01. Wilson for (1,100) is (0.00025, 0.054). High = 0.054.
    # If we set the arbitrary "high > 0.05" for p near 0, it should stop.
    # Let's adjust the arbitrary threshold in the function to 0.05 for testing.
    # (Original function has 0.1, so this would pass with original code if not modified for test)
    # For this test, let's assume the internal threshold is `high > 0.05`.
    # Test with `p_hat = 0.01`, `high = 0.054`.
    # The code has `high > 0.1` (by default parameter now), so for k=1, n=100 -> high = 0.054, this will NOT stop.
    # Let's use k=1, n=30. p_hat = 0.033. Wilson for (1,30) approx (0.0008, 0.16). High = 0.16. This should stop.
    stop_reason = check_node_stopping_conditions(
        node_k_sum=1, node_n_sum=30, node_num_samples=20, current_depth=3,
        min_samples_split=10, max_depth=10, confidence_level=0.95,
        min_n_sum_for_statistical_stop=20, relative_width_factor=0.5
    )
    print(f"Scenario 5 (p near 0, high uncertainty): {stop_reason}")

    # Scenario 6: n_sum too low for statistical stop
    stop_reason = check_node_stopping_conditions(
        node_k_sum=1, node_n_sum=10, node_num_samples=20, current_depth=3,
        min_samples_split=10, max_depth=10, confidence_level=0.95,
        min_n_sum_for_statistical_stop=50, # n_sum=10 is less than this
        relative_width_factor=0.5
    )
    print(f"Scenario 6 (n_sum too low for stat stop - should be None): {stop_reason}")

    # Scenario 7: No reason to stop
    stop_reason = check_node_stopping_conditions(
        node_k_sum=20, node_n_sum=100, node_num_samples=50, current_depth=2,
        min_samples_split=10, max_depth=5, confidence_level=0.95,
        min_n_sum_for_statistical_stop=50,
        relative_width_factor=1.0 # Rel width = 0.165/0.2 = 0.825. This is not > 1.0
    )
    print(f"Scenario 7 (No stop condition met - should be None): {stop_reason}")
