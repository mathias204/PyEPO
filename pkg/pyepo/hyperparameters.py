from pyepo.predictive import KernelPrescription

k_param_grid = {
    "k": [1, 3, 5, 10],
}

kernel_param_grid = {
    **k_param_grid,
    "kernel" : [
        KernelPrescription._naive_kernel,
        KernelPrescription._epanechnikov_kernel,
        KernelPrescription._tricubic_kernel,
    ]
}

rf_param_grid = {
    "n_est": [50, 100, 200],
    "depth": [5, 10, 20, None],
}

weight_model_param_grid = {
    "hidden_dim": [32, 64, 128],
    "dropout": [0.1],
    "num_hidden_layers": [0,1],
}

train_param_grid = {
    "epochs": [1000],
    "batch_size": [128],
    "lr": [1e-3, 5e-4],
}

dfl_model_param_grid = {
    "hidden_dim": [32, 64, 128],
    "dropout": [0.1],
    "num_hidden_layers": [0,1],
}
