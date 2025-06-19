# NewBinomialTree/tests/test_accuracy.py
import sys
import os
import json
import numpy as np
import time

# Adjust path to import from parent directory
current_dir_for_path = os.path.dirname(os.path.abspath(__file__))
project_root_for_path = os.path.dirname(current_dir_for_path)
if project_root_for_path not in sys.path:
    sys.path.insert(0, project_root_for_path)

from tests.test_harness import run_test_scenario, run_xgboost_peer_test
from tests.generated_datasets import (
    dataset_generator_numerical,
    dataset_generator_categorical,
    dataset_generator_mixed
)

# --- Test Definitions ---

# Standard column names used across generated datasets
TARGET_COLUMN = 'k_target'
EXPOSURE_COLUMN = 'n_exposure'
TRUE_P_COLUMN = 'true_p'

# Configurations define the sets of hyperparameters to test.
# Each configuration will be run against all test scenarios.
CONFIGURATIONS_TO_TEST = [
    {
        "config_name": "Config_Baseline",
        "alpha": 0.05,
        "max_depth": 5,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "confidence_level": 0.95,
        "max_numerical_split_points": 100,
        "scenario_overrides": {
            "Numerical_Linear_Large_100K": {"max_depth": 7},
            "Numerical_Linear_Huge_1M": {"max_depth": 8, "max_numerical_split_points": 255},
            "Numerical_Step_Rare_Events": {"min_samples_leaf": 50, "min_samples_split": 100},
            "Categorical_High_Cardinality": {"min_samples_leaf": 30, "min_samples_split": 60, "max_depth": 6}
        }
    },
    {
        "config_name": "Config_Strict_Alpha",
        "alpha": 0.01, # Stricter statistical stopping
        "max_depth": 7, # Give more room for alpha to be the main constraint
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "confidence_level": 0.95,
        "max_numerical_split_points": 100,
        "scenario_overrides": {
             "Numerical_Linear_Huge_1M": {"max_numerical_split_points": 255},
        }
    },
    {
        "config_name": "Config_Loose_Alpha",
        "alpha": 0.10, # Looser statistical stopping
        "max_depth": 7,
        "min_samples_split": 10,
        "min_samples_leaf": 5,
        "confidence_level": 0.95,
        "max_numerical_split_points": 100,
        "scenario_overrides": {
             "Numerical_Linear_Huge_1M": {"max_numerical_split_points": 255},
        }
    },
    {
        "config_name": "Config_Shallow_Tree",
        "alpha": 0.05,
        "max_depth": 3, # Force shallow trees
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "confidence_level": 0.95,
        "max_numerical_split_points": 100
    },
    {
        "config_name": "Config_High_Min_Samples",
        "alpha": 0.05,
        "max_depth": 8, # Allow deeper trees, but sample limits should constrain
        "min_samples_split": 200,
        "min_samples_leaf": 100,
        "confidence_level": 0.95,
        "max_numerical_split_points": 100
    },
]

# Scenarios define the datasets to be tested.
# Each configuration will be run against each of these scenarios.
TEST_SCENARIOS_DEFINITIONS = [
    {
        "name": "Numerical_Step_Function",
        "generator_module": dataset_generator_numerical,
        "generator_function_name": "generate_numerical_step_p_data",
        "specific_generator_params": {
            "feature_name": "num_feat_step", "min_val": 0, "max_val": 100,
            "thresholds": [40, 70], "p_values": [0.1, 0.3, 0.05],
            "n_min": 50, "n_max": 200, "noise_on_p_stddev": 0.01
        },
        "feature_columns": ["num_feat_step"], "feature_types": {"num_feat_step": "numerical"}
    },
    {
        "name": "Numerical_Linear_Function",
        "generator_module": dataset_generator_numerical,
        "generator_function_name": "generate_numerical_linear_p_data",
        "specific_generator_params": {
            "feature_name": "num_feat_linear", "min_val": 0, "max_val": 1,
            "p_intercept": 0.05, "p_slope": 0.3,
            "n_min": 50, "n_max": 200, "noise_on_p_stddev": 0.01
        },
        "feature_columns": ["num_feat_linear"], "feature_types": {"num_feat_linear": "numerical"}
    },
    {
        "name": "Categorical_Simple_Levels",
        "generator_module": dataset_generator_categorical,
        "generator_function_name": "generate_categorical_p_data",
        "specific_generator_params": {
            "feature_name": "cat_feat_simple",
            "categories_p_map": {"GroupA": 0.1, "GroupB": 0.25, "GroupC": 0.08, "GroupD": 0.02},
            "n_min": 50, "n_max": 200, "noise_on_p_stddev": 0.005
        },
        "feature_columns": ["cat_feat_simple"], "feature_types": {"cat_feat_simple": "categorical"}
    },
    {
        "name": "Mixed_Features_Interaction",
        "generator_module": dataset_generator_mixed,
        "generator_function_name": "generate_mixed_p_data",
        "specific_generator_params": {
            "numerical_feature_name":"num_mix1", "categorical_feature_name":"cat_mix1",
            "num_min_val":0, "num_max_val":10, "categories_list": ["X","Y","Z"],
            "base_p":0.1, "num_coeff":0.02, "num_offset":5,
            "category_effects": {"X":-0.05, "Y":0.0, "Z":0.05},
            "n_min":30, "n_max":150, "noise_on_p_stddev": 0.01
        },
        "feature_columns": ["num_mix1", "cat_mix1"], "feature_types": {"num_mix1":"numerical", "cat_mix1":"categorical"}
    },
    {
        "name": "Numerical_Step_Rare_Events",
        "generator_module": dataset_generator_numerical,
        "generator_function_name": "generate_numerical_step_p_data",
        "specific_generator_params": {
            "feature_name": "num_feat_rare", "min_val": 0, "max_val": 100,
            "thresholds": [50], "p_values": [0.005, 0.015],
            "n_min": 1000, "n_max": 5000, "noise_on_p_stddev": 0.0001
        },
        "n_samples_train_override": 10000, "n_samples_test_override": 5000,
        "feature_columns": ["num_feat_rare"], "feature_types": {"num_feat_rare": "numerical"}
    },
    {
        "name": "Categorical_High_Cardinality",
        "generator_module": dataset_generator_categorical,
        "generator_function_name": "generate_categorical_p_data",
        "specific_generator_params": {
            "feature_name": "cat_feat_high_card",
            "categories_p_map": {f"HC_Cat_{i}": (0.01 + i*0.005) for i in range(30)},
            "n_min": 40, "n_max": 160, "noise_on_p_stddev": 0.001
        },
        "n_samples_train_override": 6000, "n_samples_test_override": 2000,
        "feature_columns": ["cat_feat_high_card"], "feature_types": {"cat_feat_high_card": "categorical"}
    },
    {
        "name": "Numerical_Linear_Large_100K",
        "generator_module": dataset_generator_numerical,
        "generator_function_name": "generate_numerical_linear_p_data",
        "specific_generator_params": {
            "feature_name": "num_feat_linear_large", "min_val": 0, "max_val": 1,
            "p_intercept": 0.05, "p_slope": 0.3, "n_min": 50, "n_max": 200, "noise_on_p_stddev": 0.01
        },
        "n_samples_train_override": 100000, "n_samples_test_override": 20000,
        "feature_columns": ["num_feat_linear_large"], "feature_types": {"num_feat_linear_large": "numerical"}
    },
]


def run_all_tests_for_config(config_params_base, verbose=False):
    """Iterates through all test scenarios for a given base configuration."""
    all_scenario_results = {}
    config_name = config_params_base["config_name"]
    print(f"\n\n========================================\nRUNNING TEST SUITE FOR CONFIGURATION: {config_name}\n========================================")

    DEFAULT_N_SAMPLES_TRAIN = 2000
    DEFAULT_N_SAMPLES_TEST = 1000

    for i, scenario_def in enumerate(TEST_SCENARIOS_DEFINITIONS):
        scenario_name = scenario_def["name"]
        print(f"\n  -> Running Scenario {i+1}/{len(TEST_SCENARIOS_DEFINITIONS)}: {scenario_name}...")

        # Start with base config and apply scenario-specific overrides
        current_tree_params = {k: v for k, v in config_params_base.items() if k not in ["config_name", "scenario_overrides"]}
        scenario_cfg_overrides = config_params_base.get("scenario_overrides", {}).get(scenario_name, {})
        current_tree_params.update(scenario_cfg_overrides)

        # Generate train and test data for the scenario
        generator_module = scenario_def["generator_module"]
        generator_func = getattr(generator_module, scenario_def["generator_function_name"])

        gen_params_train = scenario_def["specific_generator_params"].copy()
        gen_params_train["num_samples"] = scenario_def.get("n_samples_train_override", DEFAULT_N_SAMPLES_TRAIN)
        train_data = generator_func(**gen_params_train)

        gen_params_test = scenario_def["specific_generator_params"].copy()
        gen_params_test["num_samples"] = scenario_def.get("n_samples_test_override", DEFAULT_N_SAMPLES_TEST)
        test_data = generator_func(**gen_params_test)

        # Run the actual test
        results_scenario = run_test_scenario(
            dataset_name=scenario_name,
            train_data=train_data, test_data=test_data,
            target_column=TARGET_COLUMN, exposure_column=EXPOSURE_COLUMN,
            feature_columns=scenario_def["feature_columns"],
            tree_params=current_tree_params,
            feature_types=scenario_def["feature_types"],
            known_p_column=TRUE_P_COLUMN,
            verbose=False # Keep harness output clean unless debugging
        )
        all_scenario_results[scenario_name] = results_scenario
        print(f"  -> Scenario {scenario_name} (BinomialTree) completed.")

        # Run the XGBoost peer test for the same scenario
        print(f"    -> Running XGBoost peer test for {scenario_name}...")
        # Note: XGBoost params can be passed here if needed, otherwise harness uses defaults
        xgb_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'max_depth': current_tree_params.get('max_depth', 5), # Use same max_depth
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }
        results_xgb = run_xgboost_peer_test(
            dataset_name=scenario_name,
            train_data=train_data, test_data=test_data,
            target_column=TARGET_COLUMN, exposure_column=EXPOSURE_COLUMN,
            feature_columns=scenario_def["feature_columns"],
            feature_types=scenario_def["feature_types"],
            known_p_column=TRUE_P_COLUMN,
            xgboost_params=xgb_params,
            verbose=False # Keep harness output clean
        )
        all_scenario_results[f"{scenario_name}_XGBoost"] = results_xgb
        print(f"    -> XGBoost peer test for {scenario_name} completed.")


        if verbose:
            if results_scenario:
                print(f"--- Results for {scenario_name} (BinomialTree) ---")
                eval_metrics = results_scenario.get("evaluation", {})
                for key, value in eval_metrics.items():
                    if isinstance(value, float): print(f"  {key}: {value:.4f}")
                    else: print(f"  {key}: {value}")
                print("--- End of BinomialTree Results ---")
            if results_xgb:
                print(f"--- Results for {scenario_name} (XGBoost) ---")
                eval_metrics_xgb = results_xgb.get("evaluation", {})
                for key, value in eval_metrics_xgb.items():
                    if isinstance(value, float): print(f"  {key}: {value:.4f}")
                    else: print(f"  {key}: {value}")
                print("--- End of XGBoost Results ---")

    return all_scenario_results

def save_results_to_json(results_dict, filename_prefix="BinomialTree_AccuracyResults"):
    """Saves the final results dictionary to a JSON file."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_filename = f"{filename_prefix}_{timestamp}.json"
    results_dir = os.path.join(project_root_for_path, "tests", "results")
    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, base_filename)

    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert_numpy_types(i) for i in obj]
        return obj

    serializable_results = convert_numpy_types(results_dict)

    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    overall_summary = {}
    verbose_output = True

    # Allows running a single configuration from the command line, e.g. `python test_accuracy.py Config_Baseline`
    config_to_run_name = sys.argv[1] if len(sys.argv) > 1 else None

    if config_to_run_name:
        configs_to_execute = [cfg for cfg in CONFIGURATIONS_TO_TEST if cfg["config_name"] == config_to_run_name]
        if not configs_to_execute:
            print(f"Error: Specified configuration '{config_to_run_name}' not found.")
            sys.exit(1)
    else:
        configs_to_execute = CONFIGURATIONS_TO_TEST

    print(f"Found {len(configs_to_execute)} configuration(s) to run.")

    for idx, config_params in enumerate(configs_to_execute):
        config_name = config_params["config_name"]
        print(f"\n>>> Starting Configuration {idx+1}/{len(configs_to_execute)}: {config_name} <<<")
        try:
            results_for_this_config = run_all_tests_for_config(config_params, verbose=verbose_output)
            overall_summary[config_name] = results_for_this_config
        except Exception as e:
            import traceback
            print(f"!!!!!! FATAL ERROR running test suite for config: {config_name} !!!!!!\n{traceback.format_exc()}")
            overall_summary[config_name] = {"ERROR_MAIN_LOOP": str(e), "traceback": traceback.format_exc()}

    print("\n\n========================================\n========= OVERALL TEST SUMMARY =========\n========================================")
    for config_name_summary, results_data in overall_summary.items():
        print(f"\n--- Summary for Config: {config_name_summary} ---")
        if "ERROR_MAIN_LOOP" in results_data or not isinstance(results_data, dict):
            print(f"  Status: FAILED - {results_data.get('ERROR_MAIN_LOOP', 'Unknown error structure')}")
            continue

        # Group results by base scenario name for comparison
        scenarios = sorted(list(set([k.replace("_XGBoost", "") for k in results_data.keys()])))

        for scenario_base_name in scenarios:
            print(f"\n  Scenario: {scenario_base_name}")

            # Binomial Tree Results
            scenario_results_bt = results_data.get(scenario_base_name)
            print("    - BinomialTree:", end=" ")
            if isinstance(scenario_results_bt, dict) and scenario_results_bt.get("error"):
                print(f"FAILED ({scenario_results_bt['error']})")
            elif isinstance(scenario_results_bt, dict) and "evaluation" in scenario_results_bt and scenario_results_bt["evaluation"]:
                eval_res = scenario_results_bt["evaluation"]
                rmse = eval_res.get('mse_p_vs_known', np.nan)**0.5
                mae = eval_res.get('mae_p_vs_known', np.nan)
                deviance = eval_res.get('total_poisson_deviance', np.nan)
                leaves = eval_res.get('num_leaf_nodes', 'N/A')
                depth = eval_res.get('max_depth_reached', 'N/A')
                print(f"RMSE={rmse:.4f} | MAE={mae:.4f} | Deviance={deviance:.2f} | Leafs={leaves}, Depth={depth}")
            else:
                print(f"UNKNOWN OR INCOMPLETE RESULTS")

            # XGBoost Results
            scenario_results_xgb = results_data.get(f"{scenario_base_name}_XGBoost")
            if scenario_results_xgb:
                print("    - XGBoost:     ", end=" ")
                if isinstance(scenario_results_xgb, dict) and scenario_results_xgb.get("error"):
                    print(f"FAILED ({scenario_results_xgb['error']})")
                elif isinstance(scenario_results_xgb, dict) and "evaluation" in scenario_results_xgb and scenario_results_xgb["evaluation"]:
                    eval_res = scenario_results_xgb["evaluation"]
                    rmse = eval_res.get('mse_p_vs_known', np.nan)**0.5
                    mae = eval_res.get('mae_p_vs_known', np.nan)
                    deviance = eval_res.get('total_poisson_deviance', np.nan)
                    depth = eval_res.get('max_depth_reached', 'N/A')
                    estimators = eval_res.get('n_estimators', 'N/A')
                    print(f"RMSE={rmse:.4f} | MAE={mae:.4f} | Deviance={deviance:.2f} | Estimators={estimators}, Depth={depth}")
                else:
                    print(f"UNKNOWN OR INCOMPLETE RESULTS")

    save_results_to_json(overall_summary)
    print("\nTest suite finished.")
