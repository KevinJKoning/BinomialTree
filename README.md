# NewBinomialTree

A custom-built regression decision tree in Python that models a binomial distribution at each leaf.

## Features

*   **Binomial Likelihood Maximization**: The tree is optimized to find splits that maximize the binomial log-likelihood, making it suitable for modeling rates, proportions, or conversion probabilities.
*   **Statistical Stopping Criterion**: To prevent overfitting, the tree employs a robust stopping mechanism inspired by Conditional Inference Trees (`ctree`). It uses a formal hypothesis testing framework with Bonferroni correction to decide whether a split is statistically significant, pruning branches that do not show a strong feature-response relationship.
*   **Efficient Splitting Strategies**:
    *   **Numerical Features**: Evaluates split points between sorted unique values, with an option to subsample for performance.
    *   **Categorical Features**: Sorts categories by their mean target rate (`p_hat`) and efficiently finds the optimal grouping.
*   **Python with Numpy**: Implemented using standard Python 3 libraries, with `numpy` for performance enhancements in numerical operations.
*   **Pandas Integration**: Seamlessly accepts Pandas DataFrames for fitting and prediction.

## Project Structure

*   `NewBinomialTree/`: Root directory.
    *   `binomial_tree/`: Contains the core logic for the decision tree.
        *   `__init__.py`
        *   `tree.py`: `Node` and `BinomialDecisionTree` classes.
        *   `splitting.py`: Functions for finding optimal splits.
        *   `stopping.py`: Logic for branch stopping criteria.
        *   `utils.py`: Helper functions (likelihood, confidence intervals, etc.).
    *   `tests/`: Contains the testing framework.
        *   `test_accuracy.py`: Main script to run accuracy tests.
        *   `generated_datasets/`: Scripts to generate various test datasets.
    *   `README.md`: This file.

## Installation

The library can be used by cloning the repository and ensuring its root directory (`NewBinomialTree`) is in your Python path.

**Dependencies:**
*   Python 3.x
*   Numpy
*   Pandas (optional, for DataFrame input)

## Usage

The `BinomialDecisionTree` can accept input data as a list of dictionaries or a Pandas DataFrame.

### 1. Prepare Data

Your data needs to have:
*   Feature columns (numerical or categorical).
*   A target column representing the number of successes (`k`).
*   An exposure column representing the number of trials (`n`).

**Option A: List of Dictionaries**
```python
train_data_list = [
    {'feature_num': 10.0, 'feature_cat': 'A', 'successes': 2, 'trials': 20},
    {'feature_num': 12.0, 'feature_cat': 'B', 'successes': 8, 'trials': 25},
    # ... more data
]

new_data_list = [
    {'feature_num': 13.0, 'feature_cat': 'A'},
    {'feature_num': 23.0, 'feature_cat': 'C'},
]
```

**Option B: Pandas DataFrame**
```python
import pandas as pd
import numpy as np

train_data_df = pd.DataFrame({
    'feature_num': [10.0, 12.0, 15.0, 11.0, 20.0, 22.0, 18.0, 25.0, np.nan, 10.0],
    'feature_cat': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', None],
    'successes': [2, 8, 3, 15, 6, 1, 18, 5, 1, 2],
    'trials': [20, 25, 18, 30, 22, 15, 35, 20, 5, 10]
})

new_data_df = pd.DataFrame({
    'feature_num': [13.0, 23.0, np.nan],
    'feature_cat': ['A', 'C', 'B']
})
```

### 2. Instantiate the Tree
```python
from binomial_tree.tree import BinomialDecisionTree

tree = BinomialDecisionTree(
    min_samples_split=20,     # Min samples in a node to consider splitting
    min_samples_leaf=10,      # Min samples in a leaf
    max_depth=5,              # Max depth of the tree
    alpha=0.05,               # Significance level for splitting
    verbose=True              # Set to True for detailed logs
)
```

### 3. Fit the Tree

The tree will automatically detect if the input is a Pandas DataFrame or a list of dictionaries.

```python
# Using the Pandas DataFrame from Option B
tree.fit(
    data=train_data_df,
    target_column='successes',
    exposure_column='trials',
    feature_columns=['feature_num', 'feature_cat']
)
```

### 4. Make Predictions
```python
# Using the Pandas DataFrame for new data
predicted_probabilities_df = tree.predict_p(new_data_df)
print("\nPredicted probabilities (from DataFrame input):", predicted_probabilities_df)
```
`predict_p` returns a NumPy array of predicted success probabilities (`p_hat`) for each input row.

### 5. Inspect the Tree
```python
print("\nTree Structure:")
tree.print_tree()
```
The `print_tree()` method outputs a text-based representation of the tree. For each node, it shows:
*   **Split nodes**: The splitting rule (e.g., `feature_num <= 15.500`), the p-value of the split, the log-likelihood gain, and node statistics.
*   **Leaf nodes**: Node statistics and the reason for stopping (e.g., `max_depth`, `stat_stop`, `pure_node`).
*   **Node Statistics**: `k` (successes), `n` (trials), `p̂` (estimated probability), `CI_rel_width` (relative width of the Wilson confidence interval for `p̂`), `LL` (log-likelihood), and `N` (number of samples).

### NaN Handling
*   **During `fit()`**:
    *   Numerical features: Missing values (`None` or `np.nan`) are imputed using the median of non-missing values for that feature.
    *   Categorical features: Missing values are treated as a distinct category (`'__NaN__'`).
*   **During `predict_p()`**:
    *   Numerical features: Missing values are imputed using the median learned during `fit()`.
    *   Categorical features: Unseen categories or NaNs are mapped to the path for the `'__NaN__'` category learned during training. A warning is issued for unseen categories.

## Hyperparameters

The `BinomialDecisionTree` can be customized with the following parameters during instantiation:

*   `max_depth` (int, default: `5`): The absolute maximum depth of the tree. Acts as a hard stop.
*   `min_samples_split` (int, default: `20`): The minimum number of samples required in a node to consider it for splitting. This is a pre-condition checked before the more expensive split-finding logic is run.
*   `min_samples_leaf` (int, default: `10`): The minimum number of samples required to be in each child node after a split. Any split that would create a leaf with fewer samples is discarded.
*   `alpha` (float, default: `0.05`): The significance level for the statistical stopping criterion. For each node, the algorithm calculates the p-value for the best split for every feature. It then applies a Bonferroni correction to find the best-adjusted p-value. If this value is not less than `alpha`, the node stops splitting. A lower `alpha` (e.g., `0.01`) makes the stopping condition stricter, leading to smaller trees.
*   `confidence_level` (float, default: `0.95`): The confidence level for calculating the Wilson score interval. This is **for informational purposes only** (displayed in `print_tree`) and is **not used as a stopping criterion**.
*   `max_numerical_split_points` (int, default: `255`): For numerical features, this limits the number of unique mid-points evaluated as potential split thresholds. If a feature has more unique values than this, a random subset is sampled to generate split points, balancing performance with split granularity.
*   `verbose` (bool, default: `False`): If `True`, the tree fitting process will print detailed logs about node processing, split evaluations, and stopping decisions.