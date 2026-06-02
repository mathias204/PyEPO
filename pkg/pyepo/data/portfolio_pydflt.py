"""
This script includes code adapted from the PredOpt benchmarks repository:
https://github.com/PyDFLT/PyDFLT
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

PORTFOLIO_DATA_PATH_TEMPLATE = "data/portfolio/portfolio_10_{seed}.npz"
NUM_RELEVANT_SECURITY = 7


def load_data_from_npz(path: str | None = None):
    """
    Loads a compressed .npz data dictionary from a specified file path.

    Args:
        path: The full path to the .npz file containing the data dictionary.

    Returns:
        The loaded data dictionary, mapping string keys to numpy arrays.
    """
    assert path is not None, "Specify the path to data_dict!"

    with np.load(path, allow_pickle=True) as data:
        data_dict = {key: data[key] for key in data.files}
    print(f"Loaded data from {path}")
    for key, value in data_dict.items():
        print(key, value.shape)

    return data_dict


def portfolio_pre_run_hook(seed: int, train_ratio: float) -> dict:
    """
    For the portfolio problem, we set a bank return rate based on the median return of the relevant securities
    """
    data_path = PORTFOLIO_DATA_PATH_TEMPLATE.format(seed=seed)
    data = load_data_from_npz(data_path)
    
    
    
    
    y_train_data = data["return"][: int(data["features"].shape[0] * train_ratio)]



    x_train_data = data["features"][: int(data["features"].shape[0] * train_ratio)]
    y_tmp_data = data["return"][int(data["features"].shape[0] * train_ratio): ]
    x_tmp_data = data["features"][int(data["features"].shape[0] * train_ratio): ]
    y_val_data = y_tmp_data[: int(y_tmp_data.shape[0] * 0.5)]
    x_val_data = x_tmp_data[: int(x_tmp_data.shape[0] * 0.5)]
    y_test_data = y_tmp_data[int(y_tmp_data.shape[0] * 0.5):]
    x_test_data = x_tmp_data[int(x_tmp_data.shape[0] * 0.5): ]

    scaler = StandardScaler()
    x_train_data = scaler.fit_transform(x_train_data)
    x_val_data = scaler.transform(x_val_data)
    x_test_data = scaler.transform(x_test_data)


    bank_return = float(np.median(np.partition(y_train_data, NUM_RELEVANT_SECURITY - 1, axis=1)[:, NUM_RELEVANT_SECURITY - 1]))
    return x_train_data, y_train_data, x_val_data, y_val_data, x_test_data, y_test_data, bank_return