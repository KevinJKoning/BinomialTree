# NewBinomialTree/tests/test_accuracy.py
import sys
import os
import json
import numpy as np
import time
import random # For test data generation if needed

# Adjust path to import from parent directory
current_dir_for_path = os.path.dirname(os.path.abspath(__file__))
project_root_for_path = os.path.dirname(current_dir_for_path)
if project_root_for_path not in sys.path:
    sys.path.insert(0, project_root_for_path)

from tests.test_harness import run_test_scenario
from tests.generated_datasets import (
    dataset_generator_numerical,
    dataset_generator_categorical,
    dataset_generator_mixed
)

# Standard column names
TARGET_COLUMN = 'k_target'
EXPOSURE_COLUMN = 'n_exposure'
TRUE_P_COLUMN = 'true_p'

# Define configurations to test
CONFIGURATIONS_TO_TEST = [
    {
        "config_name": "InitialConfig_Std",
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "confidence_level": 0.95,
        "min_n_sum_for_statistical_stop": 50,
        "relative_width_factor": 0.75,
        "min_likelihood_gain": 0.1,
        "epsilon_stopping": 1e-6,
        "general_max_depth": 5,
        "scenario_overrides": {
            "Numerical_Linear_Function": {"max_depth": 7},
            "Mixed_Features_Interaction": {"max_depth": 6},
            "Numerical_Step_Rare_Events": {
                "min_samples_leaf": 20, "relative_width_factor": 2.5, "min_n_sum_for_statistical_stop": 1000,
            }
        }
    },
    {
        "config_name": "Config_Global_LeafPlus20",
        "min_samples_split": 40, "min_samples_leaf": 20, "confidence_level": 0.95,
        "min_n_sum_for_statistical_stop": 50, "relative_width_factor": 0.75,
        "min_likelihood_gain": 0.1, "epsilon_stopping": 1e-6, "general_max_depth": 5,
        "scenario_overrides": {
             "Numerical_Step_Rare_Events": {"min_samples_leaf": 30} # Keep other rare event params
        }
    },
    {
        "config_name": "Config_MoreSplits", # Lower min_samples_leaf, lower gain threshold
        "min_samples_split": 10, "min_samples_leaf": 5, "confidence_level": 0.95,
        "min_n_sum_for_statistical_stop": 30, "relative_width_factor": 0.5,
        "min_likelihood_gain": 0.01, "epsilon_stopping": 1e-6, "general_max_depth": 7,
         "scenario_overrides": {
             "Numerical_Step_Rare_Events": {"min_samples_leaf": 10, "relative_width_factor": 1.0, "min_n_sum_for_statistical_stop": 500}
        }
    }
]

# Define test scenarios
# Each scenario: name, generator_module, generator_function_name (in module), specific_generator_params, feature_columns, feature_types
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
        "name": "Mixed_Features_Interaction", # Using the mixed generator directly
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
            "thresholds": [50], "p_values": [0.001, 0.005], # Very low p
            "n_min": 1000, "n_max": 5000, # High exposure
            "noise_on_p_stddev": 0.0001
        },
        "n_samples_train_override": 10000, # Larger sample size for rare events
        "n_samples_test_override": 5000,
        "feature_columns": ["num_feat_rare"], "feature_types": {"num_feat_rare": "numerical"}
    },
    {
        "name": "Categorical_High_Cardinality",
        "generator_module": dataset_generator_categorical,
        "generator_function_name": "generate_categorical_p_data",
        "specific_generator_params": {
            "feature_name": "cat_feat_high_card",
            "categories_p_map": {f"HC_Cat_{i}": (0.01 + i*0.005) for i in range(30)}, # 30 categories
            "n_min": 40, "n_max": 160, "noise_on_p_stddev": 0.001
        },
        "n_samples_train_override": 6000, 
        "n_samples_test_override": 2000,
        "feature_columns": ["cat_feat_high_card"], "feature_types": {"cat_feat_high_card": "categorical"}
    }
]


def run_all_tests_for_config(config_params_base, verbose=False):
    all_scenario_results = {}
    config_name = config_params_base["config_name"]
    print(f"\n\n========================================\nRUNNING TEST SUITE FOR CONFIGURATION: {config_name}\n========================================")

    DEFAULT_N_SAMPLES_TRAIN = 2000
    DEFAULT_N_SAMPLES_TEST = 1000

    for i, scenario_def in enumerate(TEST_SCENARIOS_DEFINITIONS):
        scenario_name = scenario_def["name"]
        print(f"\n\n==================== Test {i+1}/{len(TEST_SCENARIOS_DEFINITIONS)}: {scenario_name} ====================")
        
        # Determine tree parameters for this scenario
        current_tree_params = {k: v for k, v in config_params_base.items() 
                               if k not in ["config_name", "scenario_overrides", "general_max_depth"]}
        current_tree_params["max_depth"] = config_params_base.get("general_max_depth", 5)
        scenario_cfg_overrides = config_params_base.get("scenario_overrides", {}).get(scenario_name, {})
        current_tree_params.update(scenario_cfg_overrides)

        # Generate data
        generator_module = scenario_def["generator_module"]
        generator_func = getattr(generator_module, scenario_def["generator_function_name"])
        
        gen_params_train = scenario_def["specific_generator_params"].copy()
        gen_params_train["num_samples"] = scenario_def.get("n_samples_train_override", DEFAULT_N_SAMPLES_TRAIN)
        
        gen_params_test = scenario_def["specific_generator_params"].copy()
        gen_params_test["num_samples"] = scenario_def.get("n_samples_test_override", DEFAULT_N_SAMPLES_TEST)

        try:
            train_data = generator_func(**gen_params_train)
            test_data = generator_func(**gen_params_test)
        except Exception as e:
            print(f"!!!!!! ERROR generating data for scenario: {scenario_name} !!!!!!")
            print(f"Generator function: {scenario_def['generator_function_name']}")
            print(f"Train Params: {gen_params_train}")
            print(f"Test Params: {gen_params_test}")
            print(f"Error: {e}")
            all_scenario_results[scenario_name] = {"error": str(e), "details": "Data generation failed"}
            continue

        results_scenario = run_test_scenario(
            dataset_name=scenario_name,
            train_data=train_data,
            test_data=test_data,
            target_column=TARGET_COLUMN,
            exposure_column=EXPOSURE_COLUMN,
            feature_columns=scenario_def["feature_columns"],
            tree_params=current_tree_params,
            feature_types=scenario_def["feature_types"],
            known_p_column=TRUE_P_COLUMN,
            verbose=verbose
        )
        all_scenario_results[scenario_name] = results_scenario
        
        if verbose and results_scenario:
            print(f"--- Results for {scenario_name} ---")
            eval_metrics = results_scenario.get("evaluation", {})
            for key, value in eval_metrics.items(): # Print evaluation metrics
                if isinstance(value, float): print(f"  {key}: {value:.4f}")
                else: print(f"  {key}: {value}")
            print(f"  Training Time (s): {results_scenario.get('training_time_seconds', 'N/A'):.2f}")
            print(f"  Total Nodes: {results_scenario.get('num_nodes_total', 'N/A')}") # This is total nodes, not leaf nodes from evaluation.
            print("--- End of Scenario ---")
            
    return all_scenario_results

def save_results_to_json(results_dict, filename_prefix="test_results"):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_filename = f"{filename_prefix}_{timestamp}.json"
    
    # Define the results directory
    results_dir = os.path.join(project_root_for_path, "tests", "results") # project_root_for_path is defined globally
    
    # Create the results directory if it doesn't exist
    try:
        os.makedirs(results_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {results_dir}: {e}")
        # Fallback to current directory or handle error as appropriate
        # For now, we'll just print the error and continue saving to current dir for filename
        results_dir = "." # Fallback, though ideally this should be handled more robustly

    filename = os.path.join(results_dir, base_filename)

    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, (np.bool_, bool)): return bool(obj) 
        if isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
        if isinstance(obj, list): return [convert_numpy_types(i) for i in obj]
        return obj

    serializable_results = convert_numpy_types(results_dict)
    
    try:
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"\nResults saved to {filename}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

if __name__ == "__main__":
    overall_summary = {}
    verbose_output = True 

    config_to_run_name = None
    if len(sys.argv) > 1:
        config_to_run_name = sys.argv[1]
        print(f"Attempting to run only specified configuration: {config_to_run_name}")

    configs_to_execute = CONFIGURATIONS_TO_TEST
    if config_to_run_name:
        configs_to_execute = [cfg for cfg in CONFIGURATIONS_TO_TEST if cfg["config_name"] == config_to_run_name]
        if not configs_to_execute:
            print(f"Error: Specified configuration '{config_to_run_name}' not found.")
            sys.exit(1)

    for config_params in configs_to_execute:
        config_name = config_params["config_name"]
        try:
            results_for_this_config = run_all_tests_for_config(config_params, verbose=verbose_output)
            overall_summary[config_name] = results_for_this_config
        except Exception as e:
            print(f"!!!!!! ERROR running test suite for config: {config_name} !!!!!!")
            import traceback
            print(f"Error: {e}\n{traceback.format_exc()}")
            overall_summary[config_name] = {"ERROR_MAIN_LOOP": str(e), "traceback": traceback.format_exc()}
    
    print("\n\n========================================\n========= OVERALL TEST SUMMARY =========\n========================================")
    for config_name_summary, results_data in overall_summary.items():
        print(f"\n--- Summary for Config: {config_name_summary} ---")
        if "ERROR_MAIN_LOOP" in results_data or not isinstance(results_data, dict):
            print(f"  Status: FAILED (Outer Loop)")
            print(f"  Error: {results_data.get('ERROR_MAIN_LOOP', 'Unknown error structure')}")
            continue
        
        has_scenario_errors = False
        scenario_count = 0
        
        # Explicitly define metrics_agg with all expected keys
        metrics_agg = {
            "mse_p_vs_known": [],
            # "rmse_vs_known": [], # This was not being populated from eval_res, RMSE is calculated from MSE later
            "mae_p_vs_known": [],
            "total_log_likelihood_on_test": [],
            "total_poisson_deviance": [],
            "num_leaf_nodes": [],
            "max_depth_reached": []
        }

        for scenario_name, scenario_results in results_data.items():
            scenario_count +=1
            print(f"  Scenario: {scenario_name}")
            if isinstance(scenario_results, dict) and scenario_results.get("error"):
                print(f"    Status: FAILED ({scenario_results['error']})")
                has_scenario_errors = True
            elif isinstance(scenario_results, dict) and "evaluation" in scenario_results:
                eval_res = scenario_results["evaluation"]
                print(f"    RMSE (vs Known P): {eval_res.get('mse_p_vs_known', np.nan)**0.5:.4f}") 
                print(f"    MAE (vs Known P): {eval_res.get('mae_p_vs_known', np.nan):.4f}")
                print(f"    LogLikelihood: {eval_res.get('total_log_likelihood_on_test', np.nan):.2f}")
                print(f"    Poisson Deviance: {eval_res.get('total_poisson_deviance', np.nan):.2f}")
                print(f"    Leafs: {eval_res.get('num_leaf_nodes', np.nan)}, Depth: {eval_res.get('max_depth_reached', np.nan)}")
                
                # Populate metrics_agg
                for key_metric_agg in metrics_agg.keys(): # Iterate over keys defined in metrics_agg
                    value = eval_res.get(key_metric_agg) # Get value from evaluation results
                    if value is not None:
                        metrics_agg[key_metric_agg].append(value)
            else:
                 print(f"    Status: UNKNOWN or Incomplete Results")
                 has_scenario_errors = True


        if not has_scenario_errors and scenario_count > 0:
            print(f"  Average Metrics for {config_name_summary} ({scenario_count} scenarios):")
            if metrics_agg["mse_p_vs_known"]: 
                 avg_rmse = np.mean([m**0.5 for m in metrics_agg["mse_p_vs_known"]])
                 print(f"    Avg RMSE (vs Known P): {avg_rmse:.4f}")
            else:
                 print(f"    Avg RMSE (vs Known P): nan") # Handle case where mse_p_vs_known list is empty

            if metrics_agg["mae_p_vs_known"]: 
                print(f"    Avg MAE (vs Known P): {np.mean(metrics_agg['mae_p_vs_known']):.4f}")
            else:
                print(f"    Avg MAE (vs Known P): nan")

            if metrics_agg["total_log_likelihood_on_test"]: 
                print(f"    Avg LogLikelihood: {np.mean(metrics_agg['total_log_likelihood_on_test']):.2f}")
            else:
                print(f"    Avg LogLikelihood: nan")

            if metrics_agg["total_poisson_deviance"]: 
                print(f"    Avg Poisson Deviance: {np.mean(metrics_agg['total_poisson_deviance']):.2f}")
            else:
                print(f"    Avg Poisson Deviance: nan")

            if metrics_agg["num_leaf_nodes"]: 
                print(f"    Avg Leaf Nodes: {np.mean(metrics_agg['num_leaf_nodes']):.1f}")
            else:
                print(f"    Avg Leaf Nodes: nan")

            if metrics_agg["max_depth_reached"]: 
                print(f"    Avg Max Depth: {np.mean(metrics_agg['max_depth_reached']):.1f}")
            else:
                print(f"    Avg Max Depth: nan")

        elif scenario_count == 0:
            print(f"  Status: NO SCENARIOS RUN")
        else:
            print(f"  Status: PARTIAL FAILURE (some scenarios failed, averages not computed)")

    save_results_to_json(overall_summary, filename_prefix="BinomialTree_AccuracyResults")
    print("\nTest suite finished.")