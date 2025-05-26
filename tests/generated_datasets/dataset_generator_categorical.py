# NewBinomialTree/tests/generated_datasets/dataset_generator_categorical.py
import random
import math
from binomial_tree.utils import binomial_sampler # Adjusted relative import

# Standard column names for generated datasets
TARGET_COLUMN = 'k_target'
EXPOSURE_COLUMN = 'n_exposure'
TRUE_P_COLUMN = 'true_p'


def generate_categorical_p_data(
    num_samples=200,
    feature_name='feature_cat',
    categories_p_map=None,  # e.g., {'A': 0.1, 'B': 0.3, 'C': 0.05}
    n_min=10,
    n_max=100,
    noise_on_p_stddev=0.01 # Optional noise on the true_p before sampling k
):
    """
    Generates data where true 'p' is determined by a categorical feature.
    p is clipped to [0.01, 0.99].
    """
    if categories_p_map is None:
        categories_p_map = {'CAT_X': 0.1, 'CAT_Y': 0.4, 'CAT_Z': 0.2}

    if not categories_p_map:
        raise ValueError("categories_p_map cannot be empty.")

    category_names = list(categories_p_map.keys())
    data = []

    for i in range(num_samples):
        chosen_category = random.choice(category_names)
        true_p = categories_p_map[chosen_category]

        # Add optional noise to true_p
        if noise_on_p_stddev > 0:
            true_p += random.gauss(0, noise_on_p_stddev)

        # Clip p to a valid range
        true_p = max(0.01, min(0.99, true_p))

        n_exposure = random.randint(n_min, n_max)
        k_target = binomial_sampler(n_exposure, true_p)

        row = {
            feature_name: chosen_category,
            TARGET_COLUMN: k_target,
            EXPOSURE_COLUMN: n_exposure,
            TRUE_P_COLUMN: true_p,
            'id': f'cat_{i}'
        }
        data.append(row)
    return data


def get_dataset(config=None):
    """
    Generates a train and test dataset based on the config.
    Config example:
    {
        "num_samples_train": 200,
        "num_samples_test": 100,
        "feature_name": "category_feature",
        "categories_p_map": {"Alpha": 0.05, "Beta": 0.2, "Gamma": 0.15, "Delta": 0.3},
        "n_exposure_min": 50,
        "n_exposure_max": 200,
        "noise_on_p_stddev": 0.02
    }
    """
    if config is None:
        config = { # Default config
            "num_samples_train": 1000,
            "num_samples_test": 500,
            "feature_name": "categorical_feature",
            "categories_p_map": {"GroupA": 0.1, "GroupB": 0.25, "GroupC": 0.05, "GroupD": 0.4},
            "n_exposure_min": 20,
            "n_exposure_max": 150,
            "noise_on_p_stddev": 0.01
        }

    cat_p_map = config["categories_p_map"]

    data_train = generate_categorical_p_data(
        num_samples=config["num_samples_train"],
        feature_name=config["feature_name"],
        categories_p_map=cat_p_map,
        n_min=config["n_exposure_min"], n_max=config["n_exposure_max"],
        noise_on_p_stddev=config["noise_on_p_stddev"]
    )
    data_test = generate_categorical_p_data(
        num_samples=config["num_samples_test"],
        feature_name=config["feature_name"],
        categories_p_map=cat_p_map,
        n_min=config["n_exposure_min"], n_max=config["n_exposure_max"],
        noise_on_p_stddev=config["noise_on_p_stddev"]
    )

    return data_train, data_test, config["feature_name"]


if __name__ == '__main__':
    print("Generating sample categorical dataset...")
    cat_config = {
        "num_samples_train": 20,
        "num_samples_test": 10,
        "feature_name": "region",
        "categories_p_map": {"North": 0.08, "South": 0.15, "East": 0.05, "West": 0.22},
        "n_exposure_min": 40,
        "n_exposure_max": 120,
        "noise_on_p_stddev": 0.005
    }
    train_data_cat, test_data_cat, feat_name_cat = get_dataset(cat_config)

    print(f"Generated {len(train_data_cat)} training samples and {len(test_data_cat)} test samples.")
    print(f"Feature name: {feat_name_cat}")
    if train_data_cat:
        print("Sample training row:", random.choice(train_data_cat))
    if test_data_cat:
        print("Sample testing row:", random.choice(test_data_cat))

    # Test with few categories
    simple_cat_config = {
        "num_samples_train": 15,
        "num_samples_test": 8,
        "feature_name": "type",
        "categories_p_map": {"Type1": 0.1, "Type2": 0.5},
        "n_exposure_min": 20,
        "n_exposure_max": 50,
        "noise_on_p_stddev": 0.0
    }
    train_data_simple, test_data_simple, feat_name_simple = get_dataset(simple_cat_config)
    print(f"\nGenerated simple {len(train_data_simple)} training samples and {len(test_data_simple)} test samples.")
    print(f"Feature name: {feat_name_simple}")
    if train_data_simple:
        print("Sample training row (simple):", random.choice(train_data_simple))