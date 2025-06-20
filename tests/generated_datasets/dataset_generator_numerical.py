# NewBinomialTree/tests/generated_datasets/dataset_generator_numerical.py
import random
import math
from binomial_tree.utils import binomial_sampler # Adjusted relative import

# Standard column names for generated datasets
TARGET_COLUMN = 'k_target'
EXPOSURE_COLUMN = 'n_exposure'
TRUE_P_COLUMN = 'true_p'


def generate_numerical_linear_p_data(
    num_samples=200,
    feature_name='feature_num',
    min_val=0.0,
    max_val=1.0,
    p_intercept=0.1,
    p_slope=0.5, # p = p_intercept + p_slope * feature_val
    n_min=10,
    n_max=100,
    noise_on_p_stddev=0.01, # Optional noise on the true_p before sampling k
    p_clip=0.01
):
    """
    Generates data where true 'p' is a linear function of a numerical feature.
    p = intercept + slope * feature_value. p is clipped to [0.01, 0.99].
    """
    data = []
    for i in range(num_samples):
        feature_val = random.uniform(min_val, max_val)

        true_p = p_intercept + p_slope * feature_val

        # Add optional noise to true_p
        if noise_on_p_stddev > 0:
            true_p += random.gauss(0, noise_on_p_stddev)

        # Clip p to a valid range
        true_p = max(p_clip, min(1.0 - p_clip, true_p))

        n_exposure = random.randint(n_min, n_max)
        k_target = binomial_sampler(n_exposure, true_p)

        row = {
            feature_name: feature_val,
            TARGET_COLUMN: k_target,
            EXPOSURE_COLUMN: n_exposure,
            TRUE_P_COLUMN: true_p,
            'id': f'num_lin_{i}'
        }
        data.append(row)
    return data

def generate_numerical_step_p_data(
    num_samples=200,
    feature_name='feature_num_step',
    min_val=0.0,
    max_val=100.0,
    thresholds=None, # e.g., [25, 50, 75]
    p_values=None,   # e.g., [0.1, 0.3, 0.5, 0.2] - len must be len(thresholds)+1
    n_min=10,
    n_max=100,
    noise_on_p_stddev=0.01,
    p_clip=0.01
):
    """
    Generates data where true 'p' is a step function of a numerical feature.
    """
    if thresholds is None:
        thresholds = [30, 70]
    if p_values is None:
        p_values = [0.1, 0.5, 0.2] # p for val <= t1; t1 < val <= t2; val > t2

    if len(p_values) != len(thresholds) + 1:
        raise ValueError("Length of p_values must be len(thresholds) + 1")

    data = []
    sorted_thresholds = sorted(thresholds)

    for i in range(num_samples):
        feature_val = random.uniform(min_val, max_val)

        current_p = p_values[-1] # Default p for values greater than all thresholds
        for j, threshold in enumerate(sorted_thresholds):
            if feature_val <= threshold:
                current_p = p_values[j]
                break

        true_p = current_p
        if noise_on_p_stddev > 0:
            true_p += random.gauss(0, noise_on_p_stddev)
        true_p = max(p_clip, min(1.0 - p_clip, true_p))

        n_exposure = random.randint(n_min, n_max)
        k_target = binomial_sampler(n_exposure, true_p)

        row = {
            feature_name: feature_val,
            TARGET_COLUMN: k_target,
            EXPOSURE_COLUMN: n_exposure,
            TRUE_P_COLUMN: true_p,
            'id': f'num_step_{i}'
        }
        data.append(row)
    return data


def get_dataset(config=None):
    """
    Generates a train and test dataset based on the config.
    Config example:
    {
        "type": "linear" or "step",
        "num_samples_train": 200,
        "num_samples_test": 100,
        "feature_name": "x1",
        "linear_params": {"min_val": 0, "max_val": 1, "p_intercept": 0.05, "p_slope": 0.4},
        "step_params": {"min_val": 0, "max_val": 100, "thresholds": [40], "p_values": [0.1, 0.6]},
        "n_exposure_min": 50,
        "n_exposure_max": 200,
        "noise_on_p_stddev": 0.02,
        "p_clip": 0.01
    }
    """
    if config is None:
        config = { # Default config
            "type": "step",
            "num_samples_train": 1000,
            "num_samples_test": 500,
            "feature_name": "numeric_feature",
            "linear_params": {"min_val": 0, "max_val": 1, "p_intercept": 0.05, "p_slope": 0.4},
            "step_params": {"min_val": 0, "max_val": 100, "thresholds": [33, 66], "p_values": [0.1, 0.3, 0.05]},
            "n_exposure_min": 20,
            "n_exposure_max": 150,
            "noise_on_p_stddev": 0.01,
            "p_clip": 0.01
        }

    data_train = []
    data_test = []

    common_params = {
        "feature_name": config["feature_name"],
        "n_min": config["n_exposure_min"],
        "n_max": config["n_exposure_max"],
        "noise_on_p_stddev": config["noise_on_p_stddev"],
        "p_clip": config.get("p_clip", 0.01)
    }

    if config["type"] == "linear":
        lin_p = config["linear_params"]
        data_train = generate_numerical_linear_p_data(
            num_samples=config["num_samples_train"],
            min_val=lin_p["min_val"], max_val=lin_p["max_val"],
            p_intercept=lin_p["p_intercept"], p_slope=lin_p["p_slope"],
            **common_params
        )
        data_test = generate_numerical_linear_p_data(
            num_samples=config["num_samples_test"],
            min_val=lin_p["min_val"], max_val=lin_p["max_val"],
            p_intercept=lin_p["p_intercept"], p_slope=lin_p["p_slope"],
            **common_params
        )
    elif config["type"] == "step":
        step_p = config["step_params"]
        data_train = generate_numerical_step_p_data(
            num_samples=config["num_samples_train"],
            min_val=step_p["min_val"], max_val=step_p["max_val"],
            thresholds=step_p["thresholds"], p_values=step_p["p_values"],
            **common_params
        )
        data_test = generate_numerical_step_p_data(
            num_samples=config["num_samples_test"],
            min_val=step_p["min_val"], max_val=step_p["max_val"],
            thresholds=step_p["thresholds"], p_values=step_p["p_values"],
            **common_params
        )
    else:
        raise ValueError(f"Unknown dataset type in config: {config['type']}")

    return data_train, data_test, config["feature_name"]


if __name__ == '__main__':
    print("Generating sample numerical dataset (linear)...")
    lin_config = {
        "type": "linear", "num_samples_train": 20, "num_samples_test": 10,
        "feature_name": "x_lin",
        "linear_params": {"min_val": 0, "max_val": 10, "p_intercept": 0.05, "p_slope": 0.02},
        "n_exposure_min": 50, "n_exposure_max": 100, "noise_on_p_stddev": 0.01, "p_clip": 0.01
    }
    train_data_lin, test_data_lin, feat_name_lin = get_dataset(lin_config)
    print(f"Generated {len(train_data_lin)} training samples and {len(test_data_lin)} test samples.")
    print(f"Feature name: {feat_name_lin}")
    print("Sample training row:", random.choice(train_data_lin) if train_data_lin else "N/A")
    print("Sample testing row:", random.choice(test_data_lin) if test_data_lin else "N/A")

    print("\nGenerating sample numerical dataset (step)...")
    step_config = {
        "type": "step", "num_samples_train": 20, "num_samples_test": 10,
        "feature_name": "x_step",
        "step_params": {"min_val": 0, "max_val": 100, "thresholds": [50], "p_values": [0.1, 0.4]},
        "n_exposure_min": 30, "n_exposure_max": 80, "noise_on_p_stddev": 0.01, "p_clip": 0.01
    }
    train_data_step, test_data_step, feat_name_step = get_dataset(step_config)
    print(f"Generated {len(train_data_step)} training samples and {len(test_data_step)} test samples.")
    print(f"Feature name: {feat_name_step}")
    print("Sample training row:", random.choice(train_data_step) if train_data_step else "N/A")
    print("Sample testing row:", random.choice(test_data_step) if test_data_step else "N/A")
