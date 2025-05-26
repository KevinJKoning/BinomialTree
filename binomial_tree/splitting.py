# NewBinomialTree/binomial_tree/splitting.py
import math
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
    feature_name: str
):
    best_split = {'log_likelihood': -float('inf')}
    
    if indices_for_node.size < 2 * min_samples_leaf: 
        return best_split

    feature_vals_node = feature_column_data_full[indices_for_node]
    k_vals_node = k_array_full[indices_for_node]
    n_vals_node = n_array_full[indices_for_node]

    valid_feature_indices_in_node = ~np.isnan(feature_vals_node)
    unique_sorted_values = np.unique(feature_vals_node[valid_feature_indices_in_node])
    
    if unique_sorted_values.size < 2: 
        return best_split

    for i in range(unique_sorted_values.size - 1):
        split_value = (unique_sorted_values[i] + unique_sorted_values[i+1]) / 2.0
        
        left_mask_node = feature_vals_node <= split_value 
        left_mask_node[np.isnan(feature_vals_node)] = False 

        right_mask_node = ~left_mask_node
        right_mask_node[np.isnan(feature_vals_node)] = True


        num_left = left_mask_node.sum()
        num_right = right_mask_node.sum()

        if num_left < min_samples_leaf or num_right < min_samples_leaf:
            continue
        
        n_sum_left = n_vals_node[left_mask_node].sum()
        n_sum_right = n_vals_node[right_mask_node].sum()

        if n_sum_left == 0 or n_sum_right == 0: 
            continue

        left_indices_original = indices_for_node[left_mask_node]
        right_indices_original = indices_for_node[right_mask_node]
        
        left_obs_list = list(zip(k_vals_node[left_mask_node].tolist(), n_vals_node[left_mask_node].tolist()))
        right_obs_list = list(zip(k_vals_node[right_mask_node].tolist(), n_vals_node[right_mask_node].tolist()))

        current_split_log_likelihood = calculate_split_children_log_likelihood(left_obs_list, right_obs_list)

        if current_split_log_likelihood > best_split['log_likelihood']:
            best_split['feature'] = feature_name
            best_split['value'] = split_value 
            best_split['log_likelihood'] = current_split_log_likelihood
            best_split['left_indices'] = left_indices_original
            best_split['right_indices'] = right_indices_original
            best_split['type'] = 'numerical'
            
    return best_split

def find_best_categorical_split(
    feature_codes_full: np.ndarray, 
    k_array_full: np.ndarray,
    n_array_full: np.ndarray,
    indices_for_node: np.ndarray,
    min_samples_leaf: int,
    feature_name: str
):
    best_split = {'log_likelihood': -float('inf')}

    if indices_for_node.size < 2 * min_samples_leaf: 
        return best_split

    codes_node = feature_codes_full[indices_for_node] 
    k_node = k_array_full[indices_for_node]
    n_node = n_array_full[indices_for_node]
    
    unique_codes_in_node, inverse_indices = np.unique(codes_node, return_inverse=True)
    
    if unique_codes_in_node.size < 2: 
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
    current_node_log_likelihood: float
):
    overall_best_split = {'log_likelihood': -float('inf')} 

    if indices_for_node.size < 2 * tree.min_samples_leaf:
        return {} 

    for feature_idx, feature_name in enumerate(tree.feature_columns):
        ftype = tree.feature_types.get(feature_name)
        
        current_feature_best_split = None
        feature_column_data = tree.feature_matrix[:, feature_idx] 

        if ftype == 'numerical':
            current_feature_best_split = find_best_numerical_split(
                feature_column_data_full=feature_column_data,
                k_array_full=tree.k_array,
                n_array_full=tree.n_array,
                indices_for_node=indices_for_node,
                min_samples_leaf=tree.min_samples_leaf,
                feature_name=feature_name
            )
        elif ftype == 'categorical':
            current_feature_best_split = find_best_categorical_split(
                feature_codes_full=feature_column_data, 
                k_array_full=tree.k_array,
                n_array_full=tree.n_array,
                indices_for_node=indices_for_node,
                min_samples_leaf=tree.min_samples_leaf,
                feature_name=feature_name
            )
        else:
            continue

        if current_feature_best_split and \
           current_feature_best_split.get('log_likelihood', -float('inf')) > overall_best_split['log_likelihood']:
            current_feature_best_split['feature'] = feature_name 
            overall_best_split = current_feature_best_split
            
    if overall_best_split['log_likelihood'] > -float('inf') and \
       overall_best_split['log_likelihood'] > current_node_log_likelihood:
        overall_best_split['log_likelihood_gain'] = overall_best_split['log_likelihood'] - current_node_log_likelihood
        return overall_best_split
    else:
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