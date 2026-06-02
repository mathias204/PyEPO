"""
This script includes code adapted from the PredOpt benchmarks repository:
https://github.com/ML-KULeuven/Solver-Free-DFL/
"""

import numpy as np
from sklearn.datasets import fetch_california_housing

def generate_california_house_prices_mapping(num_data, num_houses_per_instance=10, seed=42):
    rng = np.random.default_rng(seed)
    # Load processed_data
    california_housing = fetch_california_housing(as_frame=True)
    df = california_housing.frame

    # Split into features and targets
    features = df.drop('MedHouseVal', axis=1).values
    targets = df['MedHouseVal'].values

    # Calculate how many houses we need
    total_houses_needed = num_data * num_houses_per_instance
    
    # If we don't have enough houses, we'll need to sample with replacement
    if total_houses_needed > len(features):
        # Sample with replacement to get enough houses
        indices = rng.choice(len(features), size=total_houses_needed, replace=True)
        features_sampled = features[indices]
        targets_sampled = targets[indices]
    else:
        # Shuffle and take the first total_houses_needed houses
        indices = rng.permutation(len(features))[:total_houses_needed]
        features_sampled = features[indices]
        targets_sampled = targets[indices]

    # Reshape into instances
    features_reshaped = features_sampled.reshape(num_data, num_houses_per_instance, features.shape[1])
    targets_reshaped = targets_sampled.reshape(num_data, num_houses_per_instance)

    return features_reshaped, targets_reshaped