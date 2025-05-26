# NewBinomialTree/binomial_tree/tree.py
import math
import uuid
import numpy as np
import warnings

from .utils import (
    calculate_p_hat,
    get_total_log_likelihood,
    calculate_wilson_score_interval
)
from .stopping import check_node_stopping_conditions
from .splitting import find_best_split_for_node 

class Node:
    def __init__(self, depth, indices, parent_id=None):
        self.id = uuid.uuid4()
        self.depth = depth
        self.indices = np.array(indices, dtype=int) 
        self.parent_id = parent_id
        self.children_ids = [] 
        self.is_leaf = False
        self.leaf_reason = None
        self.split_rule = None 

        self.num_samples = len(indices)
        self.k_sum = 0
        self.n_sum = 0
        self.p_hat = 0.0
        self.log_likelihood_self = -float('inf') 
        self.confidence_interval = (0.0, 0.0)

    def calculate_stats(self, tree_instance, k_array_full, n_array_full):
        if self.num_samples == 0:
            self.k_sum = 0
            self.n_sum = 0
            self.p_hat = 0.0
            self.log_likelihood_self = 0 
            self.confidence_interval = (0.0, 0.0)
            return

        node_k_values = k_array_full[self.indices]
        node_n_values = n_array_full[self.indices]

        self.k_sum = np.sum(node_k_values)
        self.n_sum = np.sum(node_n_values)
        self.p_hat = calculate_p_hat(self.k_sum, self.n_sum)
        
        node_observations = list(zip(node_k_values.tolist(), node_n_values.tolist()))
        self.log_likelihood_self = get_total_log_likelihood(node_observations, self.p_hat)
        
        if self.n_sum > 0:
            self.confidence_interval = calculate_wilson_score_interval(
                self.k_sum, self.n_sum, tree_instance.confidence_level
            )
        else:
            self.confidence_interval = (0.0, 0.0)

    def set_as_leaf(self, reason):
        self.is_leaf = True
        self.leaf_reason = reason

    def set_split_rule(self, feature, value, split_type, log_likelihood_gain):
        # Children IDs are set separately after they are created
        self.split_rule = {
            'feature': feature,
            'value': value, 
            'type': split_type,
            'log_likelihood_gain': log_likelihood_gain
        }
        self.is_leaf = False

    def __repr__(self):
        if self.is_leaf:
            return (f"Node(id={self.id}, Leaf, depth={self.depth}, samples={self.num_samples}, "
                    f"k={self.k_sum}, n={self.n_sum}, p_hat={self.p_hat:.4f}, reason='{self.leaf_reason}')")
        else:
            return (f"Node(id={self.id}, Split, depth={self.depth}, rule='{self.split_rule['feature']}', "
                    f"val={self.split_rule['value']}')")

import time # For performance logging

class BinomialDecisionTree:
    def __init__(self, min_samples_split=2, max_depth=5, min_samples_leaf=1, 
                 confidence_level=0.95, min_likelihood_gain=-1e-6, 
                 min_n_sum_for_statistical_stop=30,
                 relative_width_factor=0.5, epsilon_stopping=1e-6,
                 max_numerical_split_points=255, # Max number of split points for numerical features
                 verbose=False): # Added verbose
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.confidence_level = confidence_level
        self.min_likelihood_gain = min_likelihood_gain 
        
        self.min_n_sum_for_statistical_stop = min_n_sum_for_statistical_stop
        self.relative_width_factor = relative_width_factor
        self.epsilon_stopping = epsilon_stopping

        self.root_id = None
        self.nodes = {} 

        self.feature_matrix = None
        self.k_array = None
        self.n_array = None
        self.feature_columns = []
        self.target_column = ''
        self.exposure_column = ''
        self.feature_types = {} 
        self.categorical_maps = {} 
        self.numeric_medians = {}
        self.verbose = verbose # Store verbose flag
        self.max_numerical_split_points = max_numerical_split_points

    def _infer_feature_types(self, data_list_of_dicts, feature_columns):
        inferred_types = {}
        if not data_list_of_dicts:
            return {col: 'numerical' for col in feature_columns} 
        
        sample_row = data_list_of_dicts[0]
        for col in feature_columns:
            val = sample_row.get(col)
            if isinstance(val, (int, float)):
                inferred_types[col] = 'numerical'
            elif isinstance(val, str):
                inferred_types[col] = 'categorical'
            elif val is None: 
                found_type = False
                for r_idx in range(min(len(data_list_of_dicts), 100)): # Check up to 100 rows for None
                    if data_list_of_dicts[r_idx].get(col) is not None:
                        if isinstance(data_list_of_dicts[r_idx].get(col), (int, float)):
                            inferred_types[col] = 'numerical'
                        else:
                            inferred_types[col] = 'categorical' 
                        found_type = True
                        break
                if not found_type:
                     inferred_types[col] = 'categorical' # Default to categorical if all checked are None
            else: # bool, complex, etc.
                inferred_types[col] = 'categorical' 
        return inferred_types

    def _manual_factorize(self, column_data_list, feature_name):
        """
        Manually creates integer codes for a list of categorical values.
        Handles None by converting to a string placeholder '__NaN__'.
        Returns codes (np.array), value_to_code_map (dict), code_to_value_map (dict).
        """
        # Convert Nones and all values to string to ensure consistent typing for unique finding
        str_column_data = [str(x) if x is not None else '__NaN__' for x in column_data_list]
        
        unique_values = []
        for x in str_column_data:
            if x not in unique_values:
                unique_values.append(x)
        
        value_to_code_map = {val: i for i, val in enumerate(unique_values)}
        code_to_value_map = {i: val for i, val in enumerate(unique_values)}
        
        codes = np.array([value_to_code_map[val] for val in str_column_data], dtype=int)
        
        self.categorical_maps[feature_name] = {
            'value_to_code': value_to_code_map,
            'code_to_value': code_to_value_map
        }
        return codes

    def fit(self, data_list_of_dicts, target_column, exposure_column, feature_columns, feature_types=None):
        if self.verbose:
            fit_start_time = time.time()
            print(f"BinomialDecisionTree.fit started. Training data size: {len(data_list_of_dicts)} samples.")

        if not data_list_of_dicts:
            raise ValueError("Training data cannot be empty.")

        self.target_column = target_column
        self.exposure_column = exposure_column
        self.feature_columns = list(feature_columns) 

        if feature_types:
            self.feature_types = feature_types
        else:
            self.feature_types = self._infer_feature_types(data_list_of_dicts, self.feature_columns)

        n_samples = len(data_list_of_dicts)
        n_feats = len(self.feature_columns)
        
        self.k_array = np.array([row.get(target_column, 0) for row in data_list_of_dicts], dtype=float)
        self.n_array = np.array([row.get(exposure_column, 0) for row in data_list_of_dicts], dtype=float)
        self.feature_matrix = np.empty((n_samples, n_feats), dtype=float) 

        self.numeric_medians = {}
        self.categorical_maps = {}

        for j, feat_name in enumerate(self.feature_columns):
            col_data_list = [row.get(feat_name) for row in data_list_of_dicts]

            if self.feature_types.get(feat_name) == 'numerical':
                numeric_col_data = np.array([x if isinstance(x, (int,float)) else np.nan for x in col_data_list], dtype=float)
                nan_mask = np.isnan(numeric_col_data)
                if np.any(nan_mask):
                    median_val = np.nanmedian(numeric_col_data[~nan_mask]) if np.sum(~nan_mask) > 0 else 0.0
                    if np.isnan(median_val): 
                        median_val = 0.0 
                    self.numeric_medians[feat_name] = median_val
                    numeric_col_data[nan_mask] = median_val
                self.feature_matrix[:, j] = numeric_col_data
            
            elif self.feature_types.get(feat_name) == 'categorical':
                codes = self._manual_factorize(col_data_list, feat_name)
                self.feature_matrix[:, j] = codes 
            else:
                print(f"Warning: Feature '{feat_name}' has an unspecified or mixed type. Attempting float conversion.")
                try:
                    self.feature_matrix[:, j] = np.array(col_data_list, dtype=float)
                except ValueError:
                    print(f"Error: Could not convert feature '{feat_name}' to float. Re-treating as categorical.")
                    codes = self._manual_factorize(col_data_list, feat_name)
                    self.feature_matrix[:,j] = codes
                    self.feature_types[feat_name] = 'categorical' # Correct type
        
        if self.verbose:
            print(f"BinomialDecisionTree.fit: Preprocessing completed. {len(self.feature_columns)} features processed.")
            print(f"Feature types: {self.feature_types}")
            if self.numeric_medians: print(f"Numeric medians for imputation: {self.numeric_medians}")
            if self.categorical_maps: print(f"Categorical maps created for: {list(self.categorical_maps.keys())}")

        initial_indices = np.arange(n_samples)
        root_node = Node(depth=0, indices=initial_indices)
        root_node.calculate_stats(self, self.k_array, self.n_array)
        
        self.nodes[root_node.id] = root_node
        self.root_id = root_node.id

        if root_node.num_samples == 0: 
            root_node.set_as_leaf("empty_dataset")
            if self.verbose:
                print(f"  Root node {root_node.id} is LEAF. Reason: empty_dataset")
                fit_end_time = time.time()
                print(f"BinomialDecisionTree.fit completed in {fit_end_time - fit_start_time:.4f}s. Total nodes: {len(self.nodes)}")
            return

        queue = [root_node.id] 

        if self.verbose:
            print(f"  Root node {root_node.id} (Depth {root_node.depth}): {root_node.num_samples} samples. LL_self={root_node.log_likelihood_self:.2f}. Added to queue.")

        while queue:
            current_node_id = queue.pop(0)
            current_node = self.nodes[current_node_id]
            
            indent = "  " * (current_node.depth + 1) # For indented logging
            if self.verbose:
                print(f"{indent}Processing Node {current_node.id} (Depth {current_node.depth}): {current_node.num_samples} samples. LL_self={current_node.log_likelihood_self:.2f}")


            stop_reason = check_node_stopping_conditions(
                node_k_sum=current_node.k_sum,
                node_n_sum=current_node.n_sum,
                node_num_samples=current_node.num_samples,
                current_depth=current_node.depth,
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                confidence_level=self.confidence_level,
                min_n_sum_for_statistical_stop=self.min_n_sum_for_statistical_stop,
                relative_width_factor=self.relative_width_factor,
                epsilon=self.epsilon_stopping 
            )

            if stop_reason:
                current_node.set_as_leaf(stop_reason)
                if self.verbose:
                    print(f"{indent}  Node {current_node.id} becomes LEAF. Reason: {stop_reason}")
                continue
            
            if self.verbose:
                t_split_start = time.time()
                print(f"{indent}  Attempting to find best split for Node {current_node.id}...")

            best_split_found = find_best_split_for_node(
                tree=self, # Pass tree instance 
                indices_for_node=current_node.indices, 
                current_node_log_likelihood=current_node.log_likelihood_self,
                verbose=self.verbose, # Pass verbose flag
                node_id_for_logs=current_node.id, # Pass node_id for logging context
                node_depth_for_logs=current_node.depth, # Pass node_depth for logging context
                max_numerical_split_points=self.max_numerical_split_points # Pass new param
            )

            if self.verbose:
                t_split_end = time.time()
                print(f"{indent}  find_best_split_for_node for Node {current_node.id} took {t_split_end - t_split_start:.4f}s")

            if best_split_found and best_split_found.get('log_likelihood_gain', -float('inf')) > self.min_likelihood_gain:
                if self.verbose:
                    # Default representation, might be overridden for specific types
                    raw_value = best_split_found['value']
                    if isinstance(raw_value, dict): # Categorical split value is a dict
                        split_val_repr = f"'codes_for_left_group={raw_value.get('codes_for_left_group')}'"
                    elif isinstance(raw_value, float):
                         split_val_repr = f"'{raw_value:.3f}'"
                    else: # General case, simple string representation
                        split_val_repr = f"'{raw_value}'"
                    print(f"{indent}  Node {current_node.id} SPLIT on {best_split_found['feature']} ({best_split_found['type']}) Val: {split_val_repr}. Gain: {best_split_found['log_likelihood_gain']:.4f}")
                
                current_node.set_split_rule(
                    feature=best_split_found['feature'],
                    value=best_split_found['value'],
                    split_type=best_split_found['type'],
                    log_likelihood_gain=best_split_found['log_likelihood_gain']
                )

                left_child = Node(depth=current_node.depth + 1,
                                  indices=best_split_found['left_indices'], 
                                  parent_id=current_node.id)
                left_child.calculate_stats(self, self.k_array, self.n_array)
                self.nodes[left_child.id] = left_child

                right_child = Node(depth=current_node.depth + 1,
                                   indices=best_split_found['right_indices'], 
                                   parent_id=current_node.id)
                right_child.calculate_stats(self, self.k_array, self.n_array)
                self.nodes[right_child.id] = right_child
                
                current_node.children_ids = [left_child.id, right_child.id]
                
                if self.verbose:
                    print(f"{indent}    Added Left Child Node {left_child.id} (Depth {left_child.depth}, {left_child.num_samples} samples) to queue.")
                    print(f"{indent}    Added Right Child Node {right_child.id} (Depth {right_child.depth}, {right_child.num_samples} samples) to queue.")

                queue.append(left_child.id)
                queue.append(right_child.id)
            else:
                reason = "no_beneficial_split"
                if best_split_found and best_split_found.get('log_likelihood_gain', -float('inf')) <= self.min_likelihood_gain:
                    reason = "min_likelihood_gain_not_met"
                elif not best_split_found: # No split was found at all by splitting logic
                    reason = "no_split_found"
                current_node.set_as_leaf(reason)
                if self.verbose:
                    print(f"{indent}  Node {current_node.id} becomes LEAF. Reason: {reason}")

    def _traverse_tree(self, node_id, row_features_coded, feature_name_to_idx_map):
        node = self.nodes[node_id]
        if node.is_leaf:
            return node.p_hat, node.n_sum, node.id 

        rule = node.split_rule
        feature_to_check = rule['feature']
        split_value_rule = rule['value'] # For numerical: float; for categorical: dict {'codes_for_left_group': [codes]}
        feature_idx = feature_name_to_idx_map[feature_to_check]
        
        # row_features_coded is already an np.array of coded values (float for numerical, int codes for categorical)
        val_in_row_coded = row_features_coded[feature_idx]


        if rule['type'] == 'numerical':
            if np.isnan(val_in_row_coded): # How training handles NaNs (sent to right by splitting.py)
                return self._traverse_tree(node.children_ids[1], row_features_coded, feature_name_to_idx_map)
            if val_in_row_coded <= split_value_rule: # split_value_rule is the numeric threshold
                return self._traverse_tree(node.children_ids[0], row_features_coded, feature_name_to_idx_map)
            else:
                return self._traverse_tree(node.children_ids[1], row_features_coded, feature_name_to_idx_map)
        elif rule['type'] == 'categorical':
            # val_in_row_coded is already an integer code.
            # split_value_rule is {'codes_for_left_group': [code1, code2]}
            if val_in_row_coded in split_value_rule['codes_for_left_group']:
                return self._traverse_tree(node.children_ids[0], row_features_coded, feature_name_to_idx_map)
            else:
                return self._traverse_tree(node.children_ids[1], row_features_coded, feature_name_to_idx_map)
        else:
            raise ValueError(f"Unknown split type: {rule['type']}")

    def predict_p(self, data_list_of_dicts):
        if self.root_id is None:
            raise ValueError("Tree has not been fitted yet.")
        
        predictions = []
        feature_name_to_idx_map = {name: i for i, name in enumerate(self.feature_columns)}

        for row_dict in data_list_of_dicts:
            coded_row_features = np.empty(len(self.feature_columns), dtype=float)
            for j, feat_name in enumerate(self.feature_columns):
                val = row_dict.get(feat_name)
                if self.feature_types[feat_name] == 'numerical':
                    if val is None or np.isnan(val): 
                        coded_row_features[j] = self.numeric_medians.get(feat_name, 0.0) 
                    else:
                        try:
                            coded_row_features[j] = float(val)
                        except ValueError:
                            coded_row_features[j] = self.numeric_medians.get(feat_name, 0.0) # Fallback
                elif self.feature_types[feat_name] == 'categorical':
                    str_val = str(val) if val is not None else '__NaN__'
                    cat_map = self.categorical_maps.get(feat_name)
                    if cat_map:
                        code = cat_map['value_to_code'].get(str_val)
                        if code is None: # Unseen category
                            # Handle unseen: map to the code for '__NaN__' if it was part of training,
                            # otherwise use a default code (-1). This ensures new categories follow the path
                            # designated for missing/NaN values from training, or a generic 'unknown' path.
                            unknown_code = cat_map['value_to_code'].get('__NaN__', -1) # Use string literal and int fallback
                            coded_row_features[j] = unknown_code # unknown_code is int, will be stored as float in coded_row_features
                            warnings.warn(
                                f"Unseen category '{str_val}' for feature '{feat_name}'. "
                                f"Mapping to code {unknown_code} (path for '__NaN__' or default for unknowns).", # Use string literal in warning
                                UserWarning
                            )
                        else:
                            coded_row_features[j] = code # code from map is int, assigned to float array (becomes float)
                    else: # Should not happen if tree is fit
                        coded_row_features[j] = -1.0 # Default code for error state
            
            p_hat, _, _ = self._traverse_tree(self.root_id, coded_row_features, feature_name_to_idx_map)
            predictions.append(p_hat)
        return np.array(predictions)

    def get_params(self, deep=True):
        return {
            'min_samples_split': self.min_samples_split,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'confidence_level': self.confidence_level,
            'min_likelihood_gain': self.min_likelihood_gain,
            'min_n_sum_for_statistical_stop': self.min_n_sum_for_statistical_stop,
            'relative_width_factor': self.relative_width_factor,
            'epsilon_stopping': self.epsilon_stopping,
            'max_numerical_split_points': self.max_numerical_split_points
        }

    def print_tree(self, node_id=None, indent=""):
        if node_id is None:
            node_id = self.root_id
        
        node = self.nodes.get(node_id)
        if not node:
            print(f"{indent}No node found for ID: {node_id}")
            return

        node_stats = f"k={node.k_sum:.0f}, n={node.n_sum:.0f} (p̂={node.p_hat:.3f}) CI=[{node.confidence_interval[0]:.3f}, {node.confidence_interval[1]:.3f}] LL_self={node.log_likelihood_self:.2f} N={node.num_samples}"

        if node.is_leaf:
            print(f"{indent}Leaf: {node_stats} (Reason: {node.leaf_reason})")
        else:
            rule = node.split_rule
            feature = rule['feature']
            value_rule = rule['value']
            split_type = rule['type']
            gain = rule.get('log_likelihood_gain', 0.0)
            
            condition = ""
            if split_type == 'numerical':
                condition = f"{feature} <= {value_rule:.3f}"
            elif split_type == 'categorical':
                codes_left = value_rule.get('codes_for_left_group', [])
                if self.categorical_maps.get(feature):
                    cat_map_specific = self.categorical_maps[feature]['code_to_value']
                    original_values_left = [cat_map_specific.get(int(c), str(c) + "(code?)") for c in codes_left]
                    condition = f"{feature} in {set(original_values_left)}"
                else:
                    condition = f"{feature} codes_left: {codes_left}" 
            
            print(f"{indent}Split: {condition} (Gain={gain:.4f}) | {node_stats}")
            if node.children_ids:
                self.print_tree(node.children_ids[0], indent + "  |--L: ")
                self.print_tree(node.children_ids[1], indent + "  +--R: ")
            else: # Should not happen for a split node
                print(f"{indent}  Error: Split node missing children information.")