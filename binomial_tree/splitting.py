# NewBinomialTree/binomial_tree/splitting.py
import math
import time # For performance logging
import numpy as np
from .utils import calculate_p_hat, get_total_log_likelihood

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

    if indices_for_node.size < 2 * min_samples_leaf: # Or simply min_samples_split from tree
        if verbose:
            print(f"{indent}  NumericalSplit '{feature_name}' (Node {node_id_for_logs}): Not enough samples ({indices_for_node.size}) for split (min_samples_leaf: {min_samples_leaf}).")
        return best_split

    feature_vals_node = feature_column_data_full[indices_for_node]
    k_vals_node = k_array_full[indices_for_node]
    n_vals_node = n_array_full[indices_for_node]

    # Handle NaNs explicitly: NaNs go to the right child.
    nan_mask_node = np.isnan(feature_vals_node)
    num_nans_in_node = np.sum(nan_mask_node)

    # Process non-NaN values for finding split points
    feature_vals_nonan = feature_vals_node[~nan_mask_node]
    
    if feature_vals_nonan.size < 2: # Not enough non-NaN values to form a split
        if verbose:
            print(f"{indent}  NumericalSplit '{feature_name}' (Node {node_id_for_logs}): Less than 2 non-NaN values, no split possible.")
        return best_split

    unique_sorted_values = np.unique(feature_vals_nonan)

    if unique_sorted_values.size < 2:
        if verbose:
            print(f"{indent}  NumericalSplit '{feature_name}' (Node {node_id_for_logs}): Less than 2 unique non-NaN values, no split possible.")
        return best_split

    if verbose:
        print(f"{indent}  NumericalSplit '{feature_name}' (Node {node_id_for_logs}): Original unique non-NaN values: {unique_sorted_values.size}.")

    # Sub-sample split points if necessary
    if unique_sorted_values.size > max_numerical_split_points:
        sampled_indices = np.linspace(0, unique_sorted_values.size - 1, max_numerical_split_points, dtype=int)
        # Ensure we pick unique values, linspace might pick same index if max_numerical_split_points is high relative to unique_sorted_values.size
        # However, unique_sorted_values[sampled_indices] will already be unique if unique_sorted_values itself is.
        values_for_threshold_generation = np.unique(unique_sorted_values[sampled_indices]) # np.unique also sorts
        if verbose:
            print(f"{indent}  NumericalSplit '{feature_name}' (Node {node_id_for_logs}): Sub-sampled to {values_for_threshold_generation.size} unique values for threshold generation.")
    else:
        values_for_threshold_generation = unique_sorted_values

    if values_for_threshold_generation.size < 2:
        if verbose:
            print(f"{indent}  NumericalSplit '{feature_name}' (Node {node_id_for_logs}): Less than 2 values after sub-sampling for threshold generation, no split possible.")
        return best_split
        
    # Generate actual split values (midpoints)
    split_values_to_test = []
    for i in range(values_for_threshold_generation.size - 1):
        # Check if consecutive values are indeed different to avoid issues with identical sampled values
        if values_for_threshold_generation[i] < values_for_threshold_generation[i+1]:
             split_values_to_test.append((values_for_threshold_generation[i] + values_for_threshold_generation[i+1]) / 2.0)
    
    if not split_values_to_test:
        if verbose:
            print(f"{indent}  NumericalSplit '{feature_name}' (Node {node_id_for_logs}): No valid mid-point split values generated after sub-sampling.")
        return best_split

    # Data for NaN observations (these always go to the right child)
    k_nans = k_vals_node[nan_mask_node]
    n_nans = n_vals_node[nan_mask_node]
    
    # Data for non-NaN observations
    k_nonan = k_vals_node[~nan_mask_node]
    n_nonan = n_vals_node[~nan_mask_node]
    original_indices_nonan = indices_for_node[~nan_mask_node]
    original_indices_nan = indices_for_node[nan_mask_node]

    for split_value in split_values_to_test:
        # Partition non-NaN data based on the current split_value
        left_mask_nonan = feature_vals_nonan <= split_value
        right_mask_nonan = ~left_mask_nonan # feature_vals_nonan > split_value

        num_left = np.sum(left_mask_nonan)
        num_right_nonan = np.sum(right_mask_nonan)
        num_right_total = num_right_nonan + num_nans_in_node # Effective number of samples in right child

        if num_left < min_samples_leaf or num_right_total < min_samples_leaf:
            continue
        
        # Get k and n for left child (only non-NaN data)
        k_left_child = np.sum(k_nonan[left_mask_nonan])
        n_left_child = np.sum(n_nonan[left_mask_nonan])

        # Get k and n for right child (non-NaN data from right partition + all NaN data)
        k_right_child = np.sum(k_nonan[right_mask_nonan]) + np.sum(k_nans)
        n_right_child = np.sum(n_nonan[right_mask_nonan]) + np.sum(n_nans)

        if n_left_child == 0 or n_right_child == 0:
            continue

        # Prepare observation lists for log-likelihood calculation
        # This part is still potentially slow if lists are very large.
        # The core idea of the optimization was to avoid re-summing, which we did above.
        # calculate_split_children_log_likelihood needs lists of (k,n) tuples.
        
        # For LL calculation, we need the distribution, not just sums.
        # Create obs lists for non-NaN data going left/right
        left_obs_list = list(zip(k_nonan[left_mask_nonan].tolist(), n_nonan[left_mask_nonan].tolist()))
        
        # For right child, combine non-NaNs going right and all NaNs
        right_obs_k = np.concatenate((k_nonan[right_mask_nonan], k_nans)) if num_nans_in_node > 0 else k_nonan[right_mask_nonan]
        right_obs_n = np.concatenate((n_nonan[right_mask_nonan], n_nans)) if num_nans_in_node > 0 else n_nonan[right_mask_nonan]
        right_obs_list = list(zip(right_obs_k.tolist(), right_obs_n.tolist()))

        current_split_log_likelihood = calculate_split_children_log_likelihood(left_obs_list, right_obs_list)

        if current_split_log_likelihood > best_split['log_likelihood']:
            best_split['feature'] = feature_name
            best_split['value'] = split_value
            best_split['log_likelihood'] = current_split_log_likelihood
            
            # Correctly assign original indices
            best_split['left_indices'] = original_indices_nonan[left_mask_nonan]
            
            right_indices_from_nonan = original_indices_nonan[right_mask_nonan]
            if num_nans_in_node > 0:
                best_split['right_indices'] = np.concatenate((right_indices_from_nonan, original_indices_nan))
            else:
                best_split['right_indices'] = right_indices_from_nonan
            
            best_split['type'] = 'numerical'
            
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
    
    indent = "  " * (node_depth_for_logs + 2) # Indent for verbose logging

    if indices_for_node.size < 2 * min_samples_leaf:
        if verbose:
            print(f"{indent}  CategoricalSplit '{feature_name}' (Node {node_id_for_logs}): Not enough samples ({indices_for_node.size}) for split (min_samples_leaf: {min_samples_leaf}).")
        return best_split

    codes_node = feature_codes_full[indices_for_node]
    k_node = k_array_full[indices_for_node]
    n_node = n_array_full[indices_for_node]
    
    unique_codes_in_node, inverse_indices = np.unique(codes_node, return_inverse=True)
    
    if verbose:
        print(f"{indent}  CategoricalSplit '{feature_name}' (Node {node_id_for_logs}): Evaluating {unique_codes_in_node.size} unique categories.")

    if unique_codes_in_node.size < 2:
        if verbose:
            print(f"{indent}  CategoricalSplit '{feature_name}' (Node {node_id_for_logs}): Less than 2 unique categories, no split possible.")
        return best_split

    k_sum_per_code = np.bincount(inverse_indices, weights=k_node, minlength=unique_codes_in_node.size)
    n_sum_per_code = np.bincount(inverse_indices, weights=n_node, minlength=unique_codes_in_node.size)
    
    valid_code_mask = n_sum_per_code > 0
    if np.sum(valid_code_mask) < 2: 
        return best_split

    # Calculate p_hat only for codes with exposure
    k_sum_valid = k_sum_per_code[valid_code_mask]
    n_sum_valid = n_sum_per_code[valid_code_mask]
    actual_codes_valid = unique_codes_in_node[valid_code_mask]
    
    p_hat_for_valid_codes = k_sum_valid / n_sum_valid
    
    sorted_indices_of_valid_codes = np.argsort(p_hat_for_valid_codes)
    sorted_actual_codes = actual_codes_valid[sorted_indices_of_valid_codes]
    
    num_sorted_valid_codes = sorted_actual_codes.size
    if num_sorted_valid_codes < 2:
        return best_split

    for i in range(num_sorted_valid_codes - 1):
        codes_for_left_split = set(sorted_actual_codes[:i+1])
        
        left_mask_node = np.array([c in codes_for_left_split for c in codes_node], dtype=bool)
        right_mask_node = ~left_mask_node 
        
        num_left = left_mask_node.sum()
        num_right = right_mask_node.sum()

        if num_left < min_samples_leaf or num_right < min_samples_leaf:
            continue

        n_sum_left = n_node[left_mask_node].sum()
        n_sum_right = n_node[right_mask_node].sum()

        if n_sum_left == 0 or n_sum_right == 0:
            continue
            
        left_indices_original = indices_for_node[left_mask_node]
        right_indices_original = indices_for_node[right_mask_node]

        left_obs_list = list(zip(k_node[left_mask_node].tolist(), n_node[left_mask_node].tolist()))
        right_obs_list = list(zip(k_node[right_mask_node].tolist(), n_node[right_mask_node].tolist()))
        
        current_split_log_likelihood = calculate_split_children_log_likelihood(left_obs_list, right_obs_list)

        if current_split_log_likelihood > best_split['log_likelihood']:
            best_split['feature'] = feature_name
            best_split['value'] = {
                'codes_for_left_group': list(map(float, codes_for_left_split)), 
                'split_definition_sorted_codes': sorted_actual_codes.tolist(), 
                'split_definition_index': i 
            }
            best_split['log_likelihood'] = current_split_log_likelihood
            best_split['left_indices'] = left_indices_original
            best_split['right_indices'] = right_indices_original
            best_split['type'] = 'categorical'
            
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
        feature_name='feature_num'
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