# NewBinomialTree/tests/generated_datasets/dataset_generator_mixed.py
import random
import math
from binomial_tree.utils import binomial_sampler

# Standard column names for generated datasets
TARGET_COLUMN = 'k_target'
EXPOSURE_COLUMN = 'n_exposure'
TRUE_P_COLUMN = 'true_p'


def generate_mixed_p_data(
    num_samples=300,
    numerical_feature_name='num_feat1',
    categorical_feature_name='cat_feat1',
    num_min_val=0.0,
    num_max_val=1.0,
    categories_list=None, # e.g., ['TypeA', 'TypeB']
    # Parameters for p formula: p = base_p + num_coeff * (num_val - num_offset) + cat_effects[cat_val]
    base_p=0.1,
    num_coeff=0.2,
    num_offset=0.5, # Centering the numerical feature effect
    category_effects=None, # e.g., {'TypeA': 0.05, 'TypeB': -0.05}
    n_min=20,
    n_max=150,
    noise_on_p_stddev=0.01,
    p_clip=0.01
):
    """
    Generates data where true 'p' is a function of one numerical and one categorical feature.
    p = base_p + num_coeff * (numerical_value - num_offset) + category_effects[category_value]
    p is clipped to [p_clip, 1.0 - p_clip].
    """
    if categories_list is None:
        categories_list = ['M_Alpha', 'M_Beta', 'M_Gamma']

    if category_effects is None:
        category_effects = {cat: (i - len(categories_list)//2) * 0.05 for i, cat in enumerate(categories_list)}
        # e.g. {'M_Alpha': -0.05, 'M_Beta': 0.0, 'M_Gamma': 0.05} for 3 cats

    if not categories_list or not category_effects:
        raise ValueError("categories_list and category_effects must be provided and non-empty.")
    if not all(cat in category_effects for cat in categories_list):
        raise ValueError("All categories in categories_list must have an entry in category_effects.")

    data = []
    for i in range(num_samples):
        num_val = random.uniform(num_min_val, num_max_val)
        cat_val = random.choice(categories_list)

        true_p = base_p + num_coeff * (num_val - num_offset) + category_effects[cat_val]

        if noise_on_p_stddev > 0:
            true_p += random.gauss(0, noise_on_p_stddev)

        true_p = max(p_clip, min(1.0 - p_clip, true_p)) # Clip p

        n_exposure = random.randint(n_min, n_max)
        k_target = binomial_sampler(n_exposure, true_p)

        row = {
            numerical_feature_name: num_val,
            categorical_feature_name: cat_val,
            TARGET_COLUMN: k_target,
            EXPOSURE_COLUMN: n_exposure,
            TRUE_P_COLUMN: true_p,
            'id': f'mix_{i}'
        }
        data.append(row)
    return data

def get_dataset(config=None):
    """
    Generates a train and test dataset with mixed features.
    Config example:
    {
        "num_samples_train": 300,
        "num_samples_test": 150,
        "numerical_feature_name": "temperature",
        "categorical_feature_name": "product_type",
        "num_feature_params": {"min_val": 10, "max_val": 30},
        "cat_feature_params": {"categories_list": ["Basic", "Premium"],
                               "category_effects": {"Basic": -0.05, "Premium": 0.05}},
        "p_formula_params": {"base_p": 0.15, "num_coeff": 0.01, "num_offset": 20},
        "n_exposure_min": 30,
        "n_exposure_max": 180,
        "noise_on_p_stddev": 0.015,
        "p_clip": 0.01
    }
    Returns: (train_data, test_data, feature_names_list, feature_types_dict)
    """
    if config is None:
        # Default config
        default_num_feat = "num_mix_feat"
        default_cat_feat = "cat_mix_feat"
        default_cats = ["MixCat1", "MixCat2"]
        config = {
            "num_samples_train": 1200,
            "num_samples_test": 600,
            "numerical_feature_name": default_num_feat,
            "categorical_feature_name": default_cat_feat,
            "num_feature_params": {"min_val": 0, "max_val": 1},
            "cat_feature_params": {"categories_list": default_cats,
                                   "category_effects": {default_cats[0]: -0.05, default_cats[1]: 0.05}},
            "p_formula_params": {"base_p": 0.2, "num_coeff": 0.1, "num_offset": 0.5},
            "n_exposure_min": 25,
            "n_exposure_max": 125,
            "noise_on_p_stddev": 0.01,
            "p_clip": 0.01
        }

    num_params = config["num_feature_params"]
    cat_params = config["cat_feature_params"]
    p_params = config["p_formula_params"]

    common_args = {
        "numerical_feature_name": config["numerical_feature_name"],
        "categorical_feature_name": config["categorical_feature_name"],
        "num_min_val": num_params["min_val"],
        "num_max_val": num_params["max_val"],
        "categories_list": cat_params["categories_list"],
        "base_p": p_params["base_p"],
        "num_coeff": p_params["num_coeff"],
        "num_offset": p_params["num_offset"],
        "category_effects": cat_params["category_effects"],
        "n_min": config["n_exposure_min"],
        "n_max": config["n_exposure_max"],
        "noise_on_p_stddev": config["noise_on_p_stddev"],
        "p_clip": config.get("p_clip", 0.01)
    }

    data_train = generate_mixed_p_data(
        num_samples=config["num_samples_train"],
        **common_args
    )
    data_test = generate_mixed_p_data(
        num_samples=config["num_samples_test"],
        **common_args
    )

    feature_names = [config["numerical_feature_name"], config["categorical_feature_name"]]
    feature_types = {
        config["numerical_feature_name"]: "numerical",
        config["categorical_feature_name"]: "categorical"
    }

    return data_train, data_test, feature_names, feature_types


if __name__ == '__main__':
    print("Generating sample mixed dataset...")
    mixed_config = {
        "num_samples_train": 20,
        "num_samples_test": 10,
        "numerical_feature_name": "humidity",
        "categorical_feature_name": "season",
        "num_feature_params": {"min_val": 0.2, "max_val": 0.8}, # e.g. 20% to 80% humidity
        "cat_feature_params": {"categories_list": ["Spring", "Summer", "Autumn", "Winter"],
                               "category_effects": {"Spring": 0.0, "Summer": 0.05, "Autumn": -0.02, "Winter": -0.05}},
        "p_formula_params": {"base_p": 0.1, "num_coeff": 0.15, "num_offset": 0.5}, # p around 0.1, higher humidity means higher p
        "n_exposure_min": 50,
        "n_exposure_max": 100,
        "noise_on_p_stddev": 0.005,
        "p_clip": 0.01
    }

    train_data, test_data, f_names, f_types = get_dataset(mixed_config)

    print(f"Generated {len(train_data)} training samples and {len(test_data)} test samples.")
    print(f"Feature names: {f_names}")
    print(f"Feature types: {f_types}")
    if train_data:
        print("Sample training row:", random.choice(train_data))
    if test_data:
        print("Sample testing row:", random.choice(test_data))
