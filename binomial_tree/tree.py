# NewBinomialTree/binomial_tree/tree.py
import math
import uuid
import numpy as np
import warnings
import time

from .utils import (
    calculate_p_hat,
    get_total_log_likelihood,
    calculate_wilson_score_interval,
    Z_SCORES,
    is_pandas_dataframe,
    convert_pandas_to_list_of_dicts
)
from .stopping import check_pre_split_stopping_conditions, check_post_split_stopping_condition
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
        self.relative_ci_width = 0.0 # For information, not stopping

    def calculate_stats(self, tree_instance, k_array_full, n_array_full):
        if self.num_samples == 0:
            # This case should ideally not happen if min_samples_leaf >= 1
            self.k_sum, self.n_sum, self.p_hat = 0, 0, 0.0
            self.log_likelihood_self, self.confidence_interval = 0, (0.0, 0.0)
            self.relative_ci_width = 0.0
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
            ci_width = self.confidence_interval[1] - self.confidence_interval[0]
            if self.p_hat > 1e-9: # Avoid division by zero
                self.relative_ci_width = ci_width / self.p_hat
            else:
                self.relative_ci_width = float('inf')
        else:
            self.confidence_interval = (0.0, 0.0)
            self.relative_ci_width = float('inf')


    def set_as_leaf(self, reason):
        self.is_leaf = True
        self.leaf_reason = reason

    def set_split_rule(self, feature, value, split_type, p_value, log_likelihood_gain):
        self.split_rule = {
            'feature': feature,
            'value': value,
            'type': split_type,
            'p_value': p_value,
            'log_likelihood_gain': log_likelihood_gain
        }
        self.is_leaf = False

    def __repr__(self):
        if self.is_leaf:
            return (f"Node(id={self.id}, Leaf, depth={self.depth}, samples={self.num_samples}, "
                    f"k={self.k_sum}, n={self.n_sum}, p_hat={self.p_hat:.4f}, reason='{self.leaf_reason}')")
        else:
            rule = self.split_rule
            val_repr = f"{rule['value']:.3f}" if isinstance(rule['value'], float) else f"{rule['value']}"
            return (f"Node(id={self.id}, Split, depth={self.depth}, rule='{rule['feature']}', "
                    f"val={val_repr}')")

class BinomialDecisionTree:
    def __init__(
        self,
        min_samples_split=20,
        max_depth=5,
        min_samples_leaf=10,
        alpha=0.05,
        confidence_level=0.95,
        max_numerical_split_points=255,
        verbose=False
    ):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.confidence_level = confidence_level
        self.max_numerical_split_points = max_numerical_split_points
        self.verbose = verbose

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


    def _infer_feature_types(self, data_list_of_dicts, feature_columns):
        inferred_types = {}
        if not data_list_of_dicts: return {col: 'numerical' for col in feature_columns}

        sample_row = data_list_of_dicts[0]
        for col in feature_columns:
            val = sample_row.get(col)
            if isinstance(val, (int, float)):
                inferred_types[col] = 'numerical'
            else: # str, bool, None, etc. are treated as categorical
                inferred_types[col] = 'categorical'
        return inferred_types

    def _manual_factorize(self, column_data_list, feature_name):
        str_column_data = [str(x) if x is not None else '__NaN__' for x in column_data_list]
        unique_values = sorted(list(set(str_column_data)))
        value_to_code_map = {val: i for i, val in enumerate(unique_values)}
        code_to_value_map = {i: val for i, val in enumerate(unique_values)}
        codes = np.array([value_to_code_map[val] for val in str_column_data], dtype=int)

        self.categorical_maps[feature_name] = {
            'value_to_code': value_to_code_map,
            'code_to_value': code_to_value_map
        }
        return codes

    def fit(self, data, target_column, exposure_column, feature_columns, feature_types=None):
        if self.verbose:
            fit_start_time = time.time()
            print(f"BinomialDecisionTree.fit started. Data has {len(data)} rows.")

        if is_pandas_dataframe(data):
            data_list_of_dicts = convert_pandas_to_list_of_dicts(data)
        elif isinstance(data, list):
            data_list_of_dicts = data
        else:
            raise TypeError("Input data must be a Pandas DataFrame or a list of dictionaries.")

        if not data_list_of_dicts: raise ValueError("Training data cannot be empty.")

        self.target_column, self.exposure_column, self.feature_columns = target_column, exposure_column, list(feature_columns)
        self.feature_types = feature_types or self._infer_feature_types(data_list_of_dicts, self.feature_columns)

        n_samples, n_feats = len(data_list_of_dicts), len(self.feature_columns)
        self.k_array = np.array([row.get(target_column, 0) for row in data_list_of_dicts], dtype=float)
        self.n_array = np.array([row.get(exposure_column, 0) for row in data_list_of_dicts], dtype=float)
        self.feature_matrix = np.empty((n_samples, n_feats), dtype=float)
        self.numeric_medians, self.categorical_maps = {}, {}

        for j, feat_name in enumerate(self.feature_columns):
            col_data_list = [row.get(feat_name) for row in data_list_of_dicts]
            if self.feature_types.get(feat_name) == 'numerical':
                numeric_col = np.array([x if isinstance(x, (int, float)) else np.nan for x in col_data_list], dtype=float)
                nan_mask = np.isnan(numeric_col)
                if np.any(nan_mask):
                    median_val = np.nanmedian(numeric_col[~nan_mask]) if np.sum(~nan_mask) > 0 else 0.0
                    self.numeric_medians[feat_name] = median_val
                    numeric_col[nan_mask] = median_val
                self.feature_matrix[:, j] = numeric_col
            else: # Categorical
                self.feature_matrix[:, j] = self._manual_factorize(col_data_list, feat_name)

        initial_indices = np.arange(n_samples)
        root_node = Node(depth=0, indices=initial_indices)
        root_node.calculate_stats(self, self.k_array, self.n_array)
        self.nodes[root_node.id] = root_node
        self.root_id = root_node.id

        if root_node.num_samples == 0:
            root_node.set_as_leaf("empty_dataset")
            return

        queue = [root_node.id]
        if self.verbose: print(f"  Root node {root_node.id} added to queue (samples={root_node.num_samples}).")

        while queue:
            current_node_id = queue.pop(0)
            current_node = self.nodes[current_node_id]
            indent = "  " * (current_node.depth + 1)
            if self.verbose:
                print(f"{indent}Processing Node {current_node.id} (Depth {current_node.depth}): {current_node.num_samples} samples.")

            # 1. Check pre-split stopping conditions
            stop_reason = check_pre_split_stopping_conditions(
                node_k_sum=current_node.k_sum, node_n_sum=current_node.n_sum,
                node_num_samples=current_node.num_samples, current_depth=current_node.depth,
                min_samples_split=self.min_samples_split, max_depth=self.max_depth
            )
            if stop_reason:
                current_node.set_as_leaf(stop_reason)
                if self.verbose: print(f"{indent}  Node {current_node.id} becomes LEAF. Reason: {stop_reason}")
                continue

            # 2. Find the best possible split across all features
            best_split_found, all_p_values = find_best_split_for_node(
                tree=self, indices_for_node=current_node.indices,
                current_node_log_likelihood=current_node.log_likelihood_self,
                verbose=self.verbose, node_id_for_logs=current_node.id,
                node_depth_for_logs=current_node.depth,
                max_numerical_split_points=self.max_numerical_split_points
            )

            if not best_split_found:
                current_node.set_as_leaf("no_beneficial_split")
                if self.verbose: print(f"{indent}  Node {current_node.id} becomes LEAF. Reason: No split improved log-likelihood.")
                continue

            # 3. Check post-split (statistical) stopping condition
            stat_stop_reason = check_post_split_stopping_condition(
                all_feature_p_values=all_p_values, alpha=self.alpha,
                verbose=self.verbose, node_id_for_logs=current_node.id, node_depth_for_logs=current_node.depth
            )

            if stat_stop_reason:
                current_node.set_as_leaf(stat_stop_reason)
                if self.verbose: print(f"{indent}  Node {current_node.id} becomes LEAF. Reason: {stat_stop_reason}")
                continue

            # 4. If all checks pass, perform the split
            if self.verbose: print(f"{indent}  Node {current_node.id} SPLIT on {best_split_found['feature']}.")
            current_node.set_split_rule(
                feature=best_split_found['feature'], value=best_split_found['value'],
                split_type=best_split_found['type'], p_value=best_split_found['p_value'],
                log_likelihood_gain=best_split_found['log_likelihood_gain']
            )

            left_child = Node(depth=current_node.depth + 1, indices=best_split_found['left_indices'], parent_id=current_node.id)
            left_child.calculate_stats(self, self.k_array, self.n_array)
            self.nodes[left_child.id] = left_child

            right_child = Node(depth=current_node.depth + 1, indices=best_split_found['right_indices'], parent_id=current_node.id)
            right_child.calculate_stats(self, self.k_array, self.n_array)
            self.nodes[right_child.id] = right_child

            current_node.children_ids = [left_child.id, right_child.id]
            queue.extend([left_child.id, right_child.id])

        if self.verbose:
            fit_end_time = time.time()
            print(f"BinomialDecisionTree.fit completed in {fit_end_time - fit_start_time:.4f}s. Total nodes: {len(self.nodes)}")

    def _traverse_tree(self, node_id, row_features_coded, feature_name_to_idx_map):
        node = self.nodes[node_id]
        if node.is_leaf:
            return node.p_hat, node.n_sum, node.id

        rule = node.split_rule
        feature_idx = feature_name_to_idx_map[rule['feature']]
        val_in_row_coded = row_features_coded[feature_idx]
        split_value_rule = rule['value']

        child_node_id = None
        if rule['type'] == 'numerical':
            if np.isnan(val_in_row_coded): # NaNs during prediction follow training imputation
                 val_in_row_coded = self.numeric_medians.get(rule['feature'], 0.0)

            if val_in_row_coded <= split_value_rule:
                child_node_id = node.children_ids[0]
            else:
                child_node_id = node.children_ids[1]
        elif rule['type'] == 'categorical':
            if val_in_row_coded in split_value_rule['codes_for_left_group']:
                child_node_id = node.children_ids[0]
            else:
                child_node_id = node.children_ids[1]

        return self._traverse_tree(child_node_id, row_features_coded, feature_name_to_idx_map)

    def predict_p(self, data):
        if self.root_id is None: raise ValueError("Tree has not been fitted yet.")

        if is_pandas_dataframe(data):
            data_list_of_dicts = convert_pandas_to_list_of_dicts(data)
        elif isinstance(data, list):
            data_list_of_dicts = data
        else:
            raise TypeError("Input data must be a Pandas DataFrame or a list of dictionaries.")

        predictions = []
        feature_name_to_idx_map = {name: i for i, name in enumerate(self.feature_columns)}

        for row_dict in data_list_of_dicts:
            coded_row = np.empty(len(self.feature_columns), dtype=float)
            for j, feat_name in enumerate(self.feature_columns):
                val = row_dict.get(feat_name)
                if self.feature_types[feat_name] == 'numerical':
                    coded_row[j] = float(val) if isinstance(val, (int, float)) else self.numeric_medians.get(feat_name, 0.0)
                else: # Categorical
                    str_val = str(val) if val is not None else '__NaN__'
                    cat_map = self.categorical_maps.get(feat_name, {})
                    code = cat_map.get('value_to_code', {}).get(str_val)
                    if code is None:
                        unknown_code = cat_map.get('value_to_code', {}).get('__NaN__', -1)
                        coded_row[j] = unknown_code
                        warnings.warn(f"Unseen category '{str_val}' for feature '{feat_name}'. Mapping to NaN/unknown path.", UserWarning)
                    else:
                        coded_row[j] = code

            p_hat, _, _ = self._traverse_tree(self.root_id, coded_row, feature_name_to_idx_map)
            predictions.append(p_hat)
        return np.array(predictions)

    def get_params(self, deep=True):
        return {
            'min_samples_split': self.min_samples_split,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'alpha': self.alpha,
            'confidence_level': self.confidence_level,
            'max_numerical_split_points': self.max_numerical_split_points,
            'verbose': self.verbose
        }

    def print_tree(self, node_id=None, indent=""):
        if node_id is None: node_id = self.root_id
        node = self.nodes.get(node_id)
        if not node: return

        node_stats = (f"k={node.k_sum:.0f}, n={node.n_sum:.0f} (p̂={node.p_hat:.3f}) | "
                      f"CI_rel_width={node.relative_ci_width:.2f} | LL={node.log_likelihood_self:.2f} | N={node.num_samples}")

        if node.is_leaf:
            print(f"{indent}Leaf: {node_stats} (Reason: {node.leaf_reason})")
        else:
            rule, feature = node.split_rule, node.split_rule['feature']
            p_val, gain = rule.get('p_value', 0.0), rule.get('log_likelihood_gain', 0.0)

            if rule['type'] == 'numerical':
                condition = f"{feature} <= {rule['value']:.3f}"
            else: # Categorical
                codes_left = rule['value'].get('codes_for_left_group', [])
                cat_map = self.categorical_maps.get(feature, {}).get('code_to_value', {})
                vals_left = {cat_map.get(int(c), f"code({c})") for c in codes_left}
                condition = f"{feature} in {vals_left}"

            print(f"{indent}Split: {condition} (p-val={p_val:.4f}, gain={gain:.2f}) | {node_stats}")
            if node.children_ids:
                self.print_tree(node.children_ids[0], indent + "  |--L: ")
                self.print_tree(node.children_ids[1], indent + "  +--R: ")
