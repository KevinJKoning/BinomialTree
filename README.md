# NewBinomialTree

A custom-built regression decision tree in Python that models a binomial distribution at each leaf.

## Features

*   **Binomial Likelihood Maximization**: The tree aims to maximize the binomial likelihood for each branch. It predicts the probability `p` for a given exposure `n`, where the target is the count of successes `k`.
*   **Statistical Overfitting Prevention**: Uses statistical techniques (e.g., based on confidence intervals like the Wilson score interval) to determine when to stop splitting branches, avoiding traditional train-test splits for this purpose during tree construction.
*   **Efficient Splitting Strategies**:
    *   **Numerical Features**: Evaluates split points between sorted unique values.
    *   **Categorical Features**: Sorts categories by their mean target rate (`p`) and considers splits along this sorted order.
*   **Python with Numpy**: Implemented using standard Python 3 libraries, with `numpy` for performance enhancements in numerical operations.
*   **Testing Framework**: Includes a testing suite with generated datasets to evaluate the tree's ability to recover known parameters.

## Project Structure

*   `NewBinomialTree/`: Root directory.
    *   `binomial_tree/`: Contains the core logic for the decision tree.
        *   `__init__.py`
        *   `tree.py`: `Node` and `BinomialDecisionTree` classes.
        *   `splitting.py`: Functions for finding optimal splits.
        *   `stopping.py`: Logic for branch stopping criteria.
        *   `utils.py`: Helper functions (likelihood, confidence intervals, etc.).
    *   `tests/`: Contains the testing framework.
        *   `__init__.py`
        *   `test_harness.py`: Core functions for running tests and evaluating results.
        *   `test_accuracy.py`: Main script to run all accuracy tests.
        *   `generated_datasets/`: Python scripts to generate various test datasets.
            *   `__init__.py`
            *   `dataset_generator_numerical.py`
            *   `dataset_generator_categorical.py`
            *   `dataset_generator_mixed.py`
    *   `README.md`: This file.

## Usage (Conceptual)

(Details to be added as implementation progresses)

```python
# Example (illustrative)
# from binomial_tree.tree import BinomialDecisionTree
# import pandas as pd

# Load or generate data
# data = pd.DataFrame(...) # Expects columns for features, 'k_target', 'n_exposure'

# tree = BinomialDecisionTree(min_samples_leaf=5, max_depth=10, confidence_level=0.95)
# tree.fit(data, target_column='k_target', exposure_column='n_exposure', feature_columns=['feature1', 'feature2'])

# predictions = tree.predict_p(new_data)
```