# NewBinomialTree/binomial_tree/splitting.py
import math
import time # For performance logging
import numpy as np
# from scipy.special import loggamma # For log binomial coefficient - REMOVED
from .utils import calculate_p_hat, get_total_log_likelihood, calculate_log_binom_coeff # get_total_log_likelihood might not be needed by this func anymore

# _calculate_log_binomial_coefficient MOVED to utils.py as calculate_log_binom_coeff

def _calculate_child_ll_from_sums(k_sum, n_sum, lbc_sum):
    """
    Calculates the log-likelihood for a child node given K_sum, N_sum, and LBC_sum.
    LL = LBC_sum + K_sum * log(p_hat) + (N_sum - K_sum) * log(1 - p_hat)
    Handles p_hat = 0 or p_hat = 1 cases carefully for logs.
    """
    if n_sum == 0: 
        return 0.0 

    p_hat = calculate_p_hat(k_sum, n_sum)

    log_p_hat = 0.0
    log_one_minus_p_hat = 0.0

    if p_hat > 0:
        try:
            log_p_hat = math.log(p_hat)
        except ValueError: 
            log_p_hat = -float('inf') 
    else: 
        log_p_hat = -float('inf')

    if p_hat < 1:
        try:
            log_one_minus_p_hat = math.log(1 - p_hat)
        except ValueError: 
             log_one_minus_p_hat = -float('inf')
    else: 
        log_one_minus_p_hat = -float('inf')
    
    term_k_log_p = 0.0
    if k_sum > 0: 
        if p_hat == 0: 
            return -float('inf') 
        term_k_log_p = k_sum * log_p_hat
    
    term_nk_log_1p = 0.0
    if (n_sum - k_sum) > 0: 
        if p_hat == 1: 
            return -float('inf')
        term_nk_log_1p = (n_sum - k_sum) * log_one_minus_p_hat
        
    log_likelihood = lbc_sum + term_k_log_p + term_nk_log_1p
    
    return log_likelihood

def calculate_split_children_log_likelihood(left_observations, right_observations):
    if not left_observations or not right_observations:
        return -float('inf')

    left_k_sum = sum(obs[0] for obs in left_observations)
    left_n_sum = sum(obs[1] for obs in left_observations)
    right_k_sum = sum(obs[0] for obs in right_observations)
    right_n_sum = sum(obs[1] for obs in right_observations)

    if left_n_sum == 0 or right_n_sum == 0:
        return -float('inf')

    p_hat_left = calculate_p_hat(left_k_sum, left_n_sum)
    p_hat_right = calculate_p_hat(right_k_sum, right_n_sum)

    ll_left = get_total_log_likelihood(left_observations, p_hat_left)
    ll_right = get_total_log_likelihood(right_observations, p_hat_right)
    
    return ll_left + ll_right

def find_best_numerical_split(
    feature_column_data_full: np.ndarray,
    k_array_full: np.ndarray,
    n_array_full: np.ndarray,
    indices_for_node: np.ndarray,
    min_samples_leaf: int,
    feature_name: str,
    max_numerical_split_points: int, # Added new parameter
    verbose: bool = False,
    node_id_for_logs = None,
    node_depth_for_logs: int = 0
):
    best_split = {'log_likelihood': -float('inf')}
    indent = "  " * (node_depth_for_logs + 2)

    if indices_for_node.size < 2 * min_samples_leaf:
        if verbose:
            print(f"{indent}  NumericalSplit '{feature_name}' (Node {node_id_for_logs}): Not enough samples ({indices_for_node.size}) for split (min_samples_leaf: {min_samples_leaf}).")
        return best_split

    # Extract data for the current node
    feature_values_at_node = feature_column_data_full[indices_for_node]
    k_values_at_node = k_array_full[indices_for_node]
    n_values_at_node = n_array_full[indices_for_node]

    # Handle NaNs: these always go to the right child by convention
    nan_mask_at_node = np.isnan(feature_values_at_node)
    original_indices_nan = indices_for_node[nan_mask_at_node]
    k_nans = k_values_at_node[nan_mask_at_node]
    n_nans = n_values_at_node[nan_mask_at_node]
    
    sum_k_nans = np.sum(k_nans)
    sum_n_nans = np.sum(n_nans)
    sum_lbc_nans = 0.0
    for k_val, n_val in zip(k_nans, n_nans):
        sum_lbc_nans += calculate_log_binom_coeff(k_val, n_val)
    
    num_nans_in_node = len(original_indices_nan)

    # Process non-NaN values
    feature_vals_nonan = feature_values_at_node[~nan_mask_at_node]
    k_nonan = k_values_at_node[~nan_mask_at_node]
    n_nonan = n_values_at_node[~nan_mask_at_node]
    original_indices_nonan_at_node = indices_for_node[~nan_mask_at_node]

    if feature_vals_nonan.size == 0: # All NaNs or empty node after NaNs considered
        if verbose:
             print(f"{indent}  NumericalSplit '{feature_name}' (Node {node_id_for_logs}): No non-NaN values to split on.")
        return best_split # No split possible if only NaNs (or < min_samples_leaf for non-NaNs)

    # Create a list of tuples: (feature_value, k, n, lbc, original_index) for non-NaN data
    nonan_data_tuples = []
    total_k_nonan = 0.0
    total_n_nonan = 0.0
    total_lbc_nonan = 0.0
    for i in range(len(feature_vals_nonan)):
        k_val, n_val = k_nonan[i], n_nonan[i]
        lbc_val = calculate_log_binom_coeff(k_val, n_val)
        nonan_data_tuples.append((feature_vals_nonan[i], k_val, n_val, lbc_val, original_indices_nonan_at_node[i]))
        total_k_nonan += k_val
        total_n_nonan += n_val
        total_lbc_nonan += lbc_val
        
    # Sort by feature value for efficient iteration
    sorted_nonan_data = sorted(nonan_data_tuples, key=lambda x: x[0])

    # Determine unique feature values from sorted non-NaN data for generating split points
    # This avoids re-calculating np.unique on feature_vals_nonan if sorted_nonan_data is already available
    if not sorted_nonan_data: # Should be caught by feature_vals_nonan.size == 0
         unique_sorted_values_for_splitting = np.array([])
    else:
        # Extract unique feature values from the already sorted data
        unique_sorted_values_for_splitting = np.unique(np.array([item[0] for item in sorted_nonan_data]))


    if unique_sorted_values_for_splitting.size < 2:
        if verbose:
            print(f"{indent}  NumericalSplit '{feature_name}' (Node {node_id_for_logs}): Less than 2 unique non-NaN values ({unique_sorted_values_for_splitting.size}), no split possible.")
        return best_split
    
    if verbose:
        print(f"{indent}  NumericalSplit '{feature_name}' (Node {node_id_for_logs}): Original unique non-NaN values for splitting: {unique_sorted_values_for_splitting.size}.")

    # Sub-sample split points if necessary
    if unique_sorted_values_for_splitting.size > max_numerical_split_points:
        sampled_indices = np.linspace(0, unique_sorted_values_for_splitting.size - 1, max_numerical_split_points, dtype=int)
        values_for_threshold_generation = np.unique(unique_sorted_values_for_splitting[sampled_indices])
        if verbose:
            print(f"{indent}  NumericalSplit '{feature_name}' (Node {node_id_for_logs}): Sub-sampled to {values_for_threshold_generation.size} unique values for threshold generation.")
    else:
        values_for_threshold_generation = unique_sorted_values_for_splitting

    if values_for_threshold_generation.size < 2:
        if verbose:
            print(f"{indent}  NumericalSplit '{feature_name}' (Node {node_id_for_logs}): Less than 2 values after sub-sampling for threshold generation ({values_for_threshold_generation.size}), no split possible.")
        return best_split
        
    # Generate actual split values (midpoints)
    split_values_to_test = []
    for i in range(values_for_threshold_generation.size - 1):
        # Check if consecutive values are indeed different
        if values_for_threshold_generation[i] < values_for_threshold_generation[i+1]:
             split_values_to_test.append((values_for_threshold_generation[i] + values_for_threshold_generation[i+1]) / 2.0)
    
    if not split_values_to_test:
        if verbose:
            print(f"{indent}  NumericalSplit '{feature_name}' (Node {node_id_for_logs}): No valid mid-point split values generated after sub-sampling.")
        return best_split

    # Iteratively calculate split likelihoods
    current_k_left = 0.0
    current_n_left = 0.0
    current_lbc_sum_left = 0.0
    current_original_indices_left = [] # Store actual original indices
    
    sorted_data_ptr = 0 # Pointer for sorted_nonan_data

    for split_value in split_values_to_test:
        # Move data points from "right" to "left" based on the current split_value
        # All points with feature_value <= split_value go to the left.
        # The sorted_data_ptr ensures we only iterate through sorted_nonan_data once overall.
        while sorted_data_ptr < len(sorted_nonan_data) and sorted_nonan_data[sorted_data_ptr][0] <= split_value:
            _feat_val, k_val, n_val, lbc_val, orig_idx = sorted_nonan_data[sorted_data_ptr]
            current_k_left += k_val
            current_n_left += n_val
            current_lbc_sum_left += lbc_val
            current_original_indices_left.append(orig_idx)
            sorted_data_ptr += 1

        # Check min_samples_leaf condition
        num_samples_left = len(current_original_indices_left)
        # num_samples_right_nonan is count of items remaining in sorted_nonan_data
        num_samples_right_nonan = len(sorted_nonan_data) - num_samples_left 
        num_samples_right_total = num_samples_right_nonan + num_nans_in_node

        if num_samples_left < min_samples_leaf or num_samples_right_total < min_samples_leaf:
            continue
            
        # Calculate sums for the right child (non-NaN part)
        k_right_nonan = total_k_nonan - current_k_left
        n_right_nonan = total_n_nonan - current_n_left
        lbc_sum_right_nonan = total_lbc_nonan - current_lbc_sum_left

        # Final sums for right child (including NaNs)
        final_k_right = k_right_nonan + sum_k_nans
        final_n_right = n_right_nonan + sum_n_nans
        final_lbc_sum_right = lbc_sum_right_nonan + sum_lbc_nans

        # Ensure children are not empty in terms of N sum (important for p_hat calc)
        if current_n_left == 0 or final_n_right == 0:
            continue

        # Calculate log-likelihood for left and right children
        ll_left = _calculate_child_ll_from_sums(current_k_left, current_n_left, current_lbc_sum_left)
        ll_right = _calculate_child_ll_from_sums(final_k_right, final_n_right, final_lbc_sum_right)
        current_split_log_likelihood = ll_left + ll_right

        if current_split_log_likelihood > best_split['log_likelihood']:
            best_split['feature'] = feature_name
            best_split['value'] = split_value # The midpoint is the split value
            best_split['log_likelihood'] = current_split_log_likelihood
            best_split['type'] = 'numerical'
            
            best_split['left_indices'] = np.array(current_original_indices_left, dtype=indices_for_node.dtype)
            
            # Indices for right child: remaining non-NaNs + all NaNs
            # Remaining non-NaNs are from sorted_data_ptr to the end of sorted_nonan_data
            original_indices_right_nonan = [item[4] for item in sorted_nonan_data[sorted_data_ptr:]]
            
            if num_nans_in_node > 0:
                best_split['right_indices'] = np.concatenate(
                    (np.array(original_indices_right_nonan, dtype=indices_for_node.dtype), 
                     original_indices_nan) # original_indices_nan is already a NumPy array
                )
            else:
                best_split['right_indices'] = np.array(original_indices_right_nonan, dtype=indices_for_node.dtype)
                            
    return best_split

def find_best_categorical_split(
    feature_codes_full: np.ndarray, 
    k_array_full: np.ndarray,
    n_array_full: np.ndarray,
    indices_for_node: np.ndarray,
    min_samples_leaf: int,
    feature_name: str,
    verbose: bool = False, # Added verbose
    node_id_for_logs = None, # Added for logging context
    node_depth_for_logs: int = 0 # Added for logging indent
):
    best_split = {'log_likelihood': -float('inf')}
    # Temporary storage for the components of the best split's indices, to defer concatenation
    _best_split_left_idx_components_list = None
    _best_split_right_categories_start_idx = -1 
    
    indent = "  " * (node_depth_for_logs + 2) # Indent for verbose logging

    if indices_for_node.size < 2 * min_samples_leaf:
        if verbose:
            print(f"{indent}  CategoricalSplit '{feature_name}' (Node {node_id_for_logs}): Not enough samples ({indices_for_node.size}) for split (min_samples_leaf: {min_samples_leaf}).")
        return best_split

    codes_node = feature_codes_full[indices_for_node]
    k_values_node = k_array_full[indices_for_node]
    n_values_node = n_array_full[indices_for_node]
    
    # Calculate LBC for each observation in the node
    lbc_values_node = np.array([calculate_log_binom_coeff(k, n) for k, n in zip(k_values_node, n_values_node)])

    # Aggregate stats per unique category
    unique_category_codes, inverse_indices = np.unique(codes_node, return_inverse=True)

    if verbose:
        print(f"{indent}  CategoricalSplit '{feature_name}' (Node {node_id_for_logs}): Evaluating {unique_category_codes.size} unique categories.")

    if unique_category_codes.size < 2:
        if verbose:
            print(f"{indent}  CategoricalSplit '{feature_name}' (Node {node_id_for_logs}): Less than 2 unique categories, no split possible.")
        return best_split

    # Initialize storage for aggregated data per unique category
    agg_k_per_cat = np.zeros(unique_category_codes.size)
    agg_n_per_cat = np.zeros(unique_category_codes.size)
    agg_lbc_per_cat = np.zeros(unique_category_codes.size)
    agg_count_per_cat = np.zeros(unique_category_codes.size, dtype=int)
    agg_indices_per_cat = [[] for _ in range(unique_category_codes.size)]

    for i in range(len(codes_node)): # Iterate over each observation in the node
        cat_idx = inverse_indices[i] # Get the index of the category for this observation
        agg_k_per_cat[cat_idx] += k_values_node[i]
        agg_n_per_cat[cat_idx] += n_values_node[i]
        agg_lbc_per_cat[cat_idx] += lbc_values_node[i]
        agg_count_per_cat[cat_idx] += 1
        agg_indices_per_cat[cat_idx].append(indices_for_node[i])

    # Prepare list of category data for sorting
    # Each item: {'p_hat': ..., 'k': ..., 'n': ..., 'lbc': ..., 'count': ..., 'indices': ..., 'code': ...}
    categories_data_to_sort = []
    for i, cat_code_val in enumerate(unique_category_codes):
        if agg_n_per_cat[i] > 0: # Only consider categories with exposure (n_sum > 0)
            p_hat_cat = agg_k_per_cat[i] / agg_n_per_cat[i] if agg_n_per_cat[i] > 0 else 0.0
            categories_data_to_sort.append({
                'p_hat': p_hat_cat,
                'k': agg_k_per_cat[i],
                'n': agg_n_per_cat[i],
                'lbc': agg_lbc_per_cat[i],
                'count': agg_count_per_cat[i],
                'indices': np.array(agg_indices_per_cat[i], dtype=indices_for_node.dtype),
                'code': cat_code_val
            })

    # Sort categories by their p_hat values
    categories_data_to_sort.sort(key=lambda x: x['p_hat'])
    
    num_valid_categories_for_splitting = len(categories_data_to_sort)
    if num_valid_categories_for_splitting < 2:
        if verbose:
            print(f"{indent}  CategoricalSplit '{feature_name}' (Node {node_id_for_logs}): Less than 2 valid categories after filtering/aggregation, no split possible.")
        return best_split

    # Calculate total sums for all categories being considered for splitting
    total_k_all_cats = sum(item['k'] for item in categories_data_to_sort)
    total_n_all_cats = sum(item['n'] for item in categories_data_to_sort)
    total_lbc_all_cats = sum(item['lbc'] for item in categories_data_to_sort)
    total_count_all_cats = sum(item['count'] for item in categories_data_to_sort)

    # Iteratively build left partition and calculate split likelihoods
    current_k_left = 0.0
    current_n_left = 0.0
    current_lbc_left = 0.0
    current_count_left = 0
    current_left_indices_list = [] 
    current_left_codes_set = set()

    # Iterate N-1 times to create N-1 possible splits from N sorted categories
    for i in range(num_valid_categories_for_splitting - 1):
        cat_data = categories_data_to_sort[i]
        
        current_k_left += cat_data['k']
        current_n_left += cat_data['n']
        current_lbc_left += cat_data['lbc']
        current_count_left += cat_data['count']
        current_left_indices_list.append(cat_data['indices'])
        current_left_codes_set.add(cat_data['code'])

        # Calculate right partition stats
        k_right = total_k_all_cats - current_k_left
        n_right = total_n_all_cats - current_n_left
        lbc_right = total_lbc_all_cats - current_lbc_left
        count_right = total_count_all_cats - current_count_left
        
        # Check min_samples_leaf for both potential children based on observation counts
        if current_count_left < min_samples_leaf or count_right < min_samples_leaf:
            continue
        
        # Ensure children have non-zero N_sum for p_hat calculation (usually covered by count check if min_samples_leaf >=1)
        if current_n_left == 0 or n_right == 0:
            continue
            
        ll_left = _calculate_child_ll_from_sums(current_k_left, current_n_left, current_lbc_left)
        ll_right = _calculate_child_ll_from_sums(k_right, n_right, lbc_right)
        current_split_log_likelihood = ll_left + ll_right

        if current_split_log_likelihood > best_split['log_likelihood']:
            best_split['feature'] = feature_name
            best_split['value'] = {'codes_for_left_group': list(current_left_codes_set)} # Keep codes as integers for type consistency
            best_split['log_likelihood'] = current_split_log_likelihood
            best_split['type'] = 'categorical'
            
            # Store components to build final indices later, avoid repeated concatenation
            _best_split_left_idx_components_list = list(current_left_indices_list) # Shallow copy
            _best_split_right_categories_start_idx = i + 1
            
    # After iterating through all potential splits, if a best split was found, construct its indices
    if _best_split_left_idx_components_list is not None:
        # Ensure left_indices is populated if components exist
        if _best_split_left_idx_components_list:
            best_split['left_indices'] = np.concatenate(_best_split_left_idx_components_list)
        else:
            best_split['left_indices'] = np.array([], dtype=indices_for_node.dtype)

        # Construct right_indices from the stored start index for the right categories
        if _best_split_right_categories_start_idx >= 0 and _best_split_right_categories_start_idx < len(categories_data_to_sort):
            right_indices_list_final = [item['indices'] for item in categories_data_to_sort[_best_split_right_categories_start_idx:]]
            if right_indices_list_final:
                best_split['right_indices'] = np.concatenate(right_indices_list_final)
            else:
                best_split['right_indices'] = np.array([], dtype=indices_for_node.dtype)
        else: # Handles cases like all categories moving to left, or invalid index
            best_split['right_indices'] = np.array([], dtype=indices_for_node.dtype)
            
    return best_split

def find_best_split_for_node(
    tree,
    indices_for_node: np.ndarray,
    current_node_log_likelihood: float,
    verbose: bool = False, 
    node_id_for_logs = None, 
    node_depth_for_logs: int = 0,
    max_numerical_split_points: int = 255 # Added new parameter from tree
):
    overall_best_split = {'log_likelihood': -float('inf')}
    indent = "  " * (node_depth_for_logs + 1) # For logging within this function

    if indices_for_node.size < 2 * tree.min_samples_leaf: # This is already a pre-check in tree.py; redundant here but harmless
        if verbose:
             print(f"{indent}  find_best_split_for_node (Node {node_id_for_logs}): Not enough samples ({indices_for_node.size}) for any split (min_samples_leaf: {tree.min_samples_leaf}).")
        return {}

    for feature_idx, feature_name in enumerate(tree.feature_columns):
        ftype = tree.feature_types.get(feature_name)
        
        if verbose:
            t_feat_split_start = time.time()
            print(f"{indent}  Evaluating feature '{feature_name}' ({ftype}) for Node {node_id_for_logs}...")

        current_feature_best_split = None
        feature_column_data = tree.feature_matrix[:, feature_idx]

        if ftype == 'numerical':
            current_feature_best_split = find_best_numerical_split(
                feature_column_data_full=feature_column_data,
                k_array_full=tree.k_array,
                n_array_full=tree.n_array,
                indices_for_node=indices_for_node,
                min_samples_leaf=tree.min_samples_leaf,
                feature_name=feature_name,
                max_numerical_split_points=max_numerical_split_points, # Pass new param
                verbose=verbose, 
                node_id_for_logs=node_id_for_logs,
                node_depth_for_logs=node_depth_for_logs
            )
        elif ftype == 'categorical':
            current_feature_best_split = find_best_categorical_split(
                feature_codes_full=feature_column_data,
                k_array_full=tree.k_array,
                n_array_full=tree.n_array,
                indices_for_node=indices_for_node,
                min_samples_leaf=tree.min_samples_leaf,
                feature_name=feature_name,
                verbose=verbose, # Pass verbose
                node_id_for_logs=node_id_for_logs,
                node_depth_for_logs=node_depth_for_logs
            )
        else:
            if verbose:
                print(f"{indent}    Skipping feature '{feature_name}' due to unknown type '{ftype}'.")
            continue
        
        if verbose:
            t_feat_split_end = time.time()
            if current_feature_best_split and current_feature_best_split.get('log_likelihood', -float('inf')) > -float('inf'):
                raw_value_feat = current_feature_best_split['value']
                if isinstance(raw_value_feat, dict):
                    split_val_repr = f"'codes_for_left_group={raw_value_feat.get('codes_for_left_group')}'"
                elif isinstance(raw_value_feat, float):
                    split_val_repr = f"'{raw_value_feat:.3f}'"
                else:
                    split_val_repr = f"'{raw_value_feat}'"
                print(f"{indent}    Feature '{feature_name}' best split LL: {current_feature_best_split['log_likelihood']:.4f} (Val: {split_val_repr}). Took {t_feat_split_end - t_feat_split_start:.4f}s")
            else:
                print(f"{indent}    Feature '{feature_name}' did not yield a valid split. Took {t_feat_split_end - t_feat_split_start:.4f}s")


        if current_feature_best_split and \
           current_feature_best_split.get('log_likelihood', -float('inf')) > overall_best_split['log_likelihood']:
            current_feature_best_split['feature'] = feature_name
            overall_best_split = current_feature_best_split
            
    if overall_best_split['log_likelihood'] > -float('inf') and \
       overall_best_split['log_likelihood'] > current_node_log_likelihood:
        overall_best_split['log_likelihood_gain'] = overall_best_split['log_likelihood'] - current_node_log_likelihood
        if verbose:
            raw_value_overall = overall_best_split['value']
            if isinstance(raw_value_overall, dict):
                split_val_repr = f"'codes_for_left_group={raw_value_overall.get('codes_for_left_group')}'"
            elif isinstance(raw_value_overall, float):
                split_val_repr = f"'{raw_value_overall:.3f}'"
            else:
                split_val_repr = f"'{raw_value_overall}'"
            print(f"{indent}  Overall best split for Node {node_id_for_logs}: Feature '{overall_best_split['feature']}' ({overall_best_split['type']}), Val: {split_val_repr}, LL Gain: {overall_best_split['log_likelihood_gain']:.4f}")
        return overall_best_split
    else:
        if verbose:
            print(f"{indent}  No beneficial split found for Node {node_id_for_logs} (Best LL found: {overall_best_split['log_likelihood']:.4f}, Node LL: {current_node_log_likelihood:.4f}).")
        return {}

if __name__ == '__main__':
    
    class MockTree:
        def __init__(self, data_list_of_dicts, target_column, exposure_column, feature_columns, feature_types, min_samples_leaf):
            self.target_column = target_column
            self.exposure_column = exposure_column
            self.feature_columns = feature_columns
            self.feature_types = feature_types 
            self.min_samples_leaf = min_samples_leaf
            
            self.k_array = np.array([row.get(target_column, 0) for row in data_list_of_dicts], dtype=float)
            self.n_array = np.array([row.get(exposure_column, 0) for row in data_list_of_dicts], dtype=float)
            
            self.feature_matrix = np.empty((len(data_list_of_dicts), len(feature_columns)), dtype=float)
            self.categorical_maps = {} 

            for j, feat_name in enumerate(feature_columns):
                col_data_original = [row.get(feat_name) for row in data_list_of_dicts]
                if feature_types.get(feat_name) == 'categorical':
                    nan_placeholder = "__NaN__"
                    str_col_data = [str(x) if x is not None else nan_placeholder for x in col_data_original]
                    
                    unique_categories = []
                    for x_val in str_col_data:
                        if x_val not in unique_categories: 
                             unique_categories.append(x_val)
                    
                    value_to_code = {val: i for i, val in enumerate(unique_categories)}
                    
                    coded_column = np.array([value_to_code[val] for val in str_col_data], dtype=float)
                    self.feature_matrix[:, j] = coded_column
                elif feature_types.get(feat_name) == 'numerical':
                    self.feature_matrix[:, j] = [x if isinstance(x, (int,float)) else np.nan for x in col_data_original]

    print("--- Mock Data Setup (No Pandas) ---")
    mock_data_list_of_dicts = [
        {'id': 0, 'feature_num': 10.0, 'feature_cat': 'A', 'successes': 2, 'trials': 20},
        {'id': 1, 'feature_num': 12.0, 'feature_cat': 'B', 'successes': 8, 'trials': 25},
        {'id': 2, 'feature_num': 15.0, 'feature_cat': 'A', 'successes': 3, 'trials': 18},
        {'id': 3, 'feature_num': 11.0, 'feature_cat': 'C', 'successes': 15, 'trials': 30},
        {'id': 4, 'feature_num': 20.0, 'feature_cat': 'B', 'successes': 6, 'trials': 22},
        {'id': 5, 'feature_num': 22.0, 'feature_cat': 'A', 'successes': 1, 'trials': 15},
        {'id': 6, 'feature_num': 18.0, 'feature_cat': 'C', 'successes': 18, 'trials': 35},
        {'id': 7, 'feature_num': 25.0, 'feature_cat': 'B', 'successes': 5, 'trials': 20},
        {'id': 8, 'feature_num': np.nan, 'feature_cat': 'A', 'successes': 1, 'trials': 5}, 
        {'id': 9, 'feature_num': 10.0, 'feature_cat': None, 'successes': 2, 'trials': 10}, 
    ]
    
    _feature_cols = ['feature_num', 'feature_cat']
    _feature_types_map = {'feature_num': 'numerical', 'feature_cat': 'categorical'}
    _min_leaf = 1
    
    mock_tree_instance = MockTree(
        data_list_of_dicts=mock_data_list_of_dicts,
        target_column='successes',
        exposure_column='trials',
        feature_columns=_feature_cols,
        feature_types=_feature_types_map,
        min_samples_leaf=_min_leaf
    )

    all_indices = np.arange(len(mock_data_list_of_dicts))

    root_k_sum = mock_tree_instance.k_array.sum()
    root_n_sum = mock_tree_instance.n_array.sum()
    root_p_hat = calculate_p_hat(root_k_sum, root_n_sum)
    root_observations = list(zip(mock_tree_instance.k_array.tolist(), mock_tree_instance.n_array.tolist()))
    root_ll_as_leaf = get_total_log_likelihood(root_observations, root_p_hat)
    print(f"Root LL (if leaf): {root_ll_as_leaf:.4f}")

    print("\n--- Testing find_best_numerical_split (Numpy version) ---")
    num_feat_idx = mock_tree_instance.feature_columns.index('feature_num')
    best_num_split = find_best_numerical_split(
        feature_column_data_full=mock_tree_instance.feature_matrix[:, num_feat_idx],
        k_array_full=mock_tree_instance.k_array,
        n_array_full=mock_tree_instance.n_array,
        indices_for_node=all_indices,
        min_samples_leaf=mock_tree_instance.min_samples_leaf,
        feature_name='feature_num',
        max_numerical_split_points=100
    )
    if 'feature' in best_num_split and best_num_split['log_likelihood'] > -float('inf'):
        print(f"Best numerical split on '{best_num_split['feature']}' at {best_num_split['value']:.2f} "
              f"with children LL: {best_num_split['log_likelihood']:.4f}")
    else:
        print("No numerical split found.")

    print("\n--- Testing find_best_categorical_split (Numpy version) ---")
    cat_feat_idx = mock_tree_instance.feature_columns.index('feature_cat')
    best_cat_split = find_best_categorical_split(
        feature_codes_full=mock_tree_instance.feature_matrix[:, cat_feat_idx], 
        k_array_full=mock_tree_instance.k_array,
        n_array_full=mock_tree_instance.n_array,
        indices_for_node=all_indices,
        min_samples_leaf=mock_tree_instance.min_samples_leaf,
        feature_name='feature_cat'
    )
    if 'feature' in best_cat_split and best_cat_split['log_likelihood'] > -float('inf'):
        # The value stored is a dict with 'codes_for_left_group', 'split_definition_sorted_codes', etc.
        value_desc = best_cat_split['value'].get('codes_for_left_group', 'N/A')
        print(f"Best categorical split on '{best_cat_split['feature']}' for group {value_desc} "
              f"with children LL: {best_cat_split['log_likelihood']:.4f}")
    else:
        print("No categorical split found.")

    print("\n--- Testing find_best_split_for_node (using MockTree) ---")
    overall_best = find_best_split_for_node(
        tree=mock_tree_instance,
        indices_for_node=all_indices,
        current_node_log_likelihood=root_ll_as_leaf
    )
    if overall_best and 'feature' in overall_best:
        print(f"Overall best split: Feature '{overall_best['feature']}', Value/Group '{overall_best['value']}'")
        print(f"  Type: {overall_best['type']}")
        print(f"  Children Combined LL: {overall_best['log_likelihood']:.4f}")
        print(f"  Log Likelihood Gain: {overall_best.get('log_likelihood_gain', 'N/A'):.4f}")
    else:
        print("No beneficial split found for the node.")