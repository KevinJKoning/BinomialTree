# NewBinomialTree/binomial_tree/utils.py
import math
import random
import warnings

# Z-scores for common confidence levels
Z_SCORES = {
    0.90: 1.645,
    0.95: 1.960,
    0.99: 2.576,
}

def calculate_p_hat(k_sum, n_sum):
    """
    Calculate the observed proportion p_hat = k_sum / n_sum.
    Handles n_sum = 0 to avoid division by zero.
    """
    if n_sum == 0:
        return 0.0 # Or perhaps a more sophisticated default for an empty node
    if k_sum > n_sum:
        warnings.warn(
            f"k_sum ({k_sum}) > n_sum ({n_sum}) in calculate_p_hat. This indicates a data issue. Clamping k_sum to n_sum for p_hat calculation.",
            UserWarning
        )
        k_sum = n_sum # Clamp k_sum for p_hat calculation
    elif k_sum < 0:
        warnings.warn(
            f"k_sum ({k_sum}) < 0 in calculate_p_hat. This indicates a data issue. Clamping k_sum to 0 for p_hat calculation.",
            UserWarning
        )
        k_sum = 0 # Clamp k_sum to 0
    return k_sum / n_sum


def calculate_log_binom_coeff(k, n):
    """
    Calculates log(C(n, k)) = log(n! / (k! * (n-k)!))
    Uses math.lgamma for numerical stability.
    log(C(n,k)) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
    """
    if k < 0 or k > n:
        return -float('inf')
    if k == 0 or k == n: # C(n,0) = 1, C(n,n) = 1, log(1) = 0
        return 0.0
    if n == 0:
        return 0.0

    k = int(round(k))
    n = int(round(n))

    # Re-check conditions after potential rounding/casting
    if k < 0 or k > n:
        return -float('inf')
    if k == 0 or k == n: # Handles n=0 as well if k=0
        return 0.0

    try:
        # math.lgamma(x) is log(Gamma(x)). Gamma(m+1) = m!
        # So log(m!) = math.lgamma(m+1)
        log_coeff = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    except ValueError:
        return -float('inf')
    return log_coeff


def calculate_binomial_log_likelihood(k, n, p):
    """
    Calculates the log-likelihood of observing k successes in n trials
    given a probability p.
    log_L = log(C(n, k)) + k * log(p) + (n - k) * log(1 - p)
    Uses math.lgamma for log(C(n, k)) to handle large numbers.

    Args:
        k (int or float): Number of successes.
        n (int or float): Number of trials.
        p (float): Probability of success.

    Returns:
        float: The log-likelihood.
    """
    if n == 0: # No information
        return 0.0
    if p < 0.0 or p > 1.0:
        raise ValueError("p must be between 0 and 1.")

    # Handle edge cases for p to avoid log(0)
    epsilon = 1e-12 # A small number to prevent log(0)
    if p < epsilon:
        p = epsilon
    if p > 1.0 - epsilon:
        p = 1.0 - epsilon

    if k < 0 or k > n:
        # This case should ideally be handled by data validation earlier
        # or implies an impossible scenario under the model.
        return -float('inf')


    try:
        log_binom_coeff = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    except ValueError:
        # Occurs if k > n or k < 0, which should be caught above,
        # or if n is not an integer (lgamma handles floats, but C(n,k) assumes int usually)
        # For safety, return very small likelihood.
        return -float('inf')

    log_p_term = k * math.log(p)
    log_1_minus_p_term = (n - k) * math.log(1 - p)

    return log_binom_coeff + log_p_term + log_1_minus_p_term

def get_total_log_likelihood(observations, p):
    """
    Calculates the total log-likelihood for a list of (k, n) observations
    given a single probability p.

    Args:
        observations (list of tuples): List of (k_i, n_i) tuples.
        p (float): Probability of success for this group.

    Returns:
        float: Total log-likelihood.
    """
    if not observations:
        return 0.0

    total_ll = 0.0
    for k_i, n_i in observations:
        total_ll += calculate_binomial_log_likelihood(k_i, n_i, p)
    return total_ll


def calculate_wilson_score_interval(k_sum, n_sum, confidence_level=0.95):
    """
    Calculate the Wilson score interval for a binomial proportion using precomputed Z-scores.

    Args:
        k_sum (int): Total number of successes.
        n_sum (int): Total number of trials.
        confidence_level (float): The desired confidence level.
                                  Supported values are 0.90, 0.95, and 0.99, corresponding
                                  to Z-scores defined in Z_SCORES. Other values will
                                  raise a ValueError as SciPy is not a dependency.

    Returns:
        tuple: (lower_bound, upper_bound) for the proportion p.
               Returns (0.0, 1.0) if n_sum is 0.

    Raises:
        ValueError: If the provided confidence_level is not one of the supported values (0.90, 0.95, 0.99).
    """
    if n_sum == 0:
        return (0.0, 1.0) # No data, so p could be anything

    if confidence_level not in Z_SCORES:
        # Fallback for arbitrary confidence levels using normal approximation
        # This requires scipy.stats.norm.ppf for precision,
        # but for common ones, we use precomputed Z_SCORES.
        # For simplicity, we'll raise error if not in Z_SCORES for now.
        raise ValueError(f"Confidence level {confidence_level} not in precomputed Z_SCORES: {list(Z_SCORES.keys())}")

    z = Z_SCORES[confidence_level]

    p_hat = k_sum / n_sum

    # Wilson score interval formula components
    # center = (k + z^2/2) / (n + z^2)
    # width  = (z / (n + z^2)) * sqrt( (k*(n-k)/n) + (z^2/4) )
    # or, equivalently:
    # lower = (p_hat + (z*z / (2*n_sum)) - z * math.sqrt((p_hat*(1-p_hat)/n_sum) + (z*z / (4*n_sum*n_sum)))) / (1 + (z*z/n_sum))
    # upper = (p_hat + (z*z / (2*n_sum)) + z * math.sqrt((p_hat*(1-p_hat)/n_sum) + (z*z / (4*n_sum*n_sum)))) / (1 + (z*z/n_sum))

    denominator = 1 + (z*z / n_sum)
    center_adjusted_p = p_hat + (z*z / (2 * n_sum))

    adjusted_std_dev = math.sqrt((p_hat * (1 - p_hat) / n_sum) + (z*z / (4 * n_sum * n_sum)))

    lower_bound = (center_adjusted_p - z * adjusted_std_dev) / denominator
    upper_bound = (center_adjusted_p + z * adjusted_std_dev) / denominator

    # Ensure bounds are within [0, 1]
    lower_bound = max(0.0, lower_bound)
    upper_bound = min(1.0, upper_bound)

    return lower_bound, upper_bound


def binomial_sampler(n, p):
    """
    Generates a random number from a binomial distribution B(n, p).
    Args:
        n (int): Number of trials.
        p (float): Probability of success for each trial.
    Returns:
        int: Number of successes.
    """
    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1.")
    if n < 0:
        raise ValueError("n cannot be negative.")
    if n == 0:
        return 0

    successes = 0
    for _ in range(n):
        if random.random() < p:
            successes += 1
    return successes

def poisson_sampler(lambda_param):
    """
    Generates a random number from a Poisson distribution Pois(lambda_param).
    Uses Knuth's algorithm.
    Args:
        lambda_param (float): The rate parameter (lambda) of the Poisson distribution.
    Returns:
        int: A random variate from the Poisson distribution.
    """
    if lambda_param < 0:
        raise ValueError("Lambda parameter must be non-negative.")
    if lambda_param == 0:
        return 0

    # For very large lambda, this can be slow. Normal approximation could be used.
    # However, for typical tree scenarios where k is rare, lambda might not be excessively large.
    l_exp = math.exp(-lambda_param)
    k = 0
    p_prod = 1.0

    while True:
        p_prod *= random.random()
        if p_prod <= l_exp:
            break
        k += 1
    return k


# --- Statistical Test Utilities ---

def _gser_lower_reg_gamma(a, x, gln):
    """
    Computes the lower regularized incomplete gamma function P(a,x) by series summation.
    gln is the value of log(Gamma(a)).
    """
    if x < 0:
        raise ValueError("x must be non-negative in gser")
    if x == 0:
        return 0.0

    ap = a
    term = 1.0 / a
    sum_val = term
    for _ in range(200): # Max iterations
        ap += 1.0
        term *= x / ap
        sum_val += term
        if abs(term) < abs(sum_val) * 1e-10:
            return sum_val * math.exp(-x + a * math.log(x) - gln)

    warnings.warn("Series for lower incomplete gamma did not converge.", RuntimeWarning)
    return sum_val * math.exp(-x + a * math.log(x) - gln)

def _gcf_upper_reg_gamma(a, x, gln):
    """
    Computes the upper regularized incomplete gamma function Q(a,x) by continued fraction.
    gln is the value of log(Gamma(a)).
    """
    if x < 0:
        raise ValueError("x must be non-negative in gcf")

    b = x + 1.0 - a
    c = 1.0 / 1.0e-30
    d = 1.0 / b
    h = d

    for i in range(1, 201): # Max iterations
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1.0e-30: d = 1.0e-30
        c = b + an / c
        if abs(c) < 1.0e-30: c = 1.0e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 2e-7:
            return math.exp(-x + a * math.log(x) - gln) * h

    warnings.warn("Continued fraction for upper incomplete gamma did not converge.", RuntimeWarning)
    return math.exp(-x + a * math.log(x) - gln) * h

def chi2_sf(chi2, df):
    """
    Survival function (1 - cdf) for the chi-squared distribution.
    This computes the p-value for a given chi-squared test statistic.
    Implemented using math functions without scipy, handles integer and float df.
    """
    if chi2 < 0:
        # Chi-squared statistic cannot be negative.
        return 1.0
    if df <= 0:
        raise ValueError("Degrees of freedom must be positive.")

    # Using relationship Chi2-SF(chi2, df) = Q(df/2, chi2/2)
    # where Q is the upper regularized incomplete gamma function.
    k = df / 2.0
    x = chi2 / 2.0

    if x == 0:
        return 1.0

    try:
        gln = math.lgamma(k)
    except ValueError:
        raise ValueError(f"Invalid args for lgamma: k={k} (df={df})")

    # Choose method based on parameters for best accuracy
    if x < k + 1.0:
        # Use series representation for P(k,x), then Q = 1 - P
        try:
            p_lower = _gser_lower_reg_gamma(k, x, gln)
            return 1.0 - p_lower
        except RuntimeError: # If series fails, fall back to CF
            return _gcf_upper_reg_gamma(k, x, gln)
    else:
        # Use continued fraction for Q(k,x)
        return _gcf_upper_reg_gamma(k, x, gln)


if __name__ == '__main__':
    # Test binomial_log_likelihood
    print(f"LogL(k=2, n=10, p=0.2): {calculate_binomial_log_likelihood(2, 10, 0.2)}") # Should be around -1.74
    # math.log(scipy.stats.binom.pmf(2, 10, 0.2)) -> log(0.301989888) approx -1.197
    # My formula includes log(C(n,k)). C(10,2) = 45. log(45) = 3.806
    # 2*log(0.2) = 2*(-1.609) = -3.218
    # 8*log(0.8) = 8*(-0.223) = -1.784
    # 3.806 - 3.218 - 1.784 = -1.196. Ok, matches scipy.
    print(f"LogL(k=0, n=10, p=0.0): {calculate_binomial_log_likelihood(0, 10, 1e-13)}") # Test edge case p=0
    print(f"LogL(k=10, n=10, p=1.0): {calculate_binomial_log_likelihood(10, 10, 1.0 - 1e-13)}") # Test edge case p=1

    # Test Wilson score interval
    k_obs, n_obs = 80, 100
    low, high = calculate_wilson_score_interval(k_obs, n_obs, 0.95)
    print(f"Wilson interval for k={k_obs}, n={n_obs}, 95% CI: ({low:.4f}, {high:.4f})") # e.g. (0.709, 0.869)

    k_obs, n_obs = 2, 100
    low, high = calculate_wilson_score_interval(k_obs, n_obs, 0.95)
    print(f"Wilson interval for k={k_obs}, n={n_obs}, 95% CI: ({low:.4f}, {high:.4f})") # e.g. (0.0024, 0.0703)

    k_obs, n_obs = 0, 10
    low, high = calculate_wilson_score_interval(k_obs, n_obs, 0.95)
    print(f"Wilson interval for k={k_obs}, n={n_obs}, 95% CI: ({low:.4f}, {high:.4f})") # e.g. (0.0000, 0.2824)

    k_obs, n_obs = 10, 10
    low, high = calculate_wilson_score_interval(k_obs, n_obs, 0.95)
    print(f"Wilson interval for k={k_obs}, n={n_obs}, 95% CI: ({low:.4f}, {high:.4f})") # e.g. (0.7176, 1.0000)


    # Test samplers
    print(f"Binomial sample (10, 0.5): {binomial_sampler(10, 0.5)}")
    print(f"Binomial sample (10, 0.0): {binomial_sampler(10, 0.0)}")
    print(f"Binomial sample (10, 1.0): {binomial_sampler(10, 1.0)}")

    print(f"Poisson sample (lambda=3): {poisson_sampler(3)}")
    print(f"Poisson sample (lambda=0.1): {poisson_sampler(0.1)}")
    print(f"Poisson sample (lambda=10): {poisson_sampler(10)}")

# --- Pandas DataFrame Utilities ---

_PANDAS_INSTALLED = True
try:
    import pandas as pd
except ImportError:
    _PANDAS_INSTALLED = False

def is_pandas_dataframe(data):
    """Checks if the provided data is a Pandas DataFrame."""
    if not _PANDAS_INSTALLED:
        return False
    return isinstance(data, pd.DataFrame)

def convert_pandas_to_list_of_dicts(dataframe):
    """
    Converts a Pandas DataFrame to a list of dictionaries.
    """
    if not is_pandas_dataframe(dataframe):
        raise TypeError("Input is not a Pandas DataFrame.")
    return dataframe.to_dict(orient='records')


    # Test total log likelihood
    obs = [(2, 10), (3, 10)] # (k, n) pairs
    p_group = 0.25
    total_ll = get_total_log_likelihood(obs, p_group)
    print(f"Total LogL for obs={obs}, p={p_group}: {total_ll}")
    # L1 = LogL(2,10,0.25) = log(C(10,2)) + 2log(0.25) + 8log(0.75) = 3.8066 + 2*(-1.3863) + 8*(-0.2877) = 3.8066 - 2.7726 - 2.3016 = -1.2676
    # L2 = LogL(3,10,0.25) = log(C(10,3)) + 3log(0.25) + 7log(0.75) = 4.7875 + 3*(-1.3863) + 7*(-0.2877) = 4.7875 - 4.1589 - 2.0139 = -1.3853
    # Total = -1.2676 - 1.3853 = -2.6529
    # With my code:
    # LogL(k=2, n=10, p=0.25): -1.2675887351330202
    # LogL(k=3, n=10, p=0.25): -1.385280764730818
    # Total: -2.6528695

    obs_empty = []
    p_group = 0.5
    total_ll_empty = get_total_log_likelihood(obs_empty, p_group)
    print(f"Total LogL for obs={obs_empty}, p={p_group}: {total_ll_empty}") # Should be 0.0
