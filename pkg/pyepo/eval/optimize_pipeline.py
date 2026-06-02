from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pyepo.predictive import NearestPrediction, RandomForestPrescription, LOESS, KernelPrescription, RecursiveKernelPrescription, CartPrescription, SAA
from pyepo.predictive.utils import test_model, WeightingTypeFunction, finetune_predictive_prescription, finetune_neural_prescription
from pyepo.dfl.finetuner import dfl_finetune
import matplotlib.ticker as mtick
import torch
import pickle
import hashlib
import json
import os
import time
import pandas as pd
from pathos.multiprocessing import ProcessingPool
import multiprocessing as mp

MODEL_COLOR_MAP = {
    '$\\hat{z}^{kNN}_N$': '#4C72B0',
    '$\\hat{z}^{kNN}_N(x)$': '#4C72B0',
    '$\\hat{z}^{LOESS}_N$': '#55A868',
    '$\\hat{z}^{LOESS}_N(x)$': '#55A868',
    '$\\hat{z}^{KR}_N$': '#C44E52',
    '$\\hat{z}^{KR}_N(x)$': '#C44E52',
    '$\\hat{z}^{Rec.KR}_N$': '#8172B2',
    '$\\hat{z}^{Rec.KR}_N(x)$': '#8172B2',
    '$\\hat{z}^{RF}_N$': '#CCB974',
    '$\\hat{z}^{RF}_N(x)$': '#CCB974',
    '$\\hat{z}^{CART}_N$': '#64B5CD',
    '$\\hat{z}^{CART}_N(x)$': '#64B5CD',
    '$\\hat{z}^{DER}_N$': '#DD8452',
    '$\\hat{z}^{DER}_N(x)$': '#DD8452',
    '$\\hat{z}^{SPO+}_N$': '#937860',
    '$\\hat{z}^{SPO+}_N(x)$': '#937860',
    '$z^{SPO+}$': '#DA8BC3',
    '$z^{SFGE}$': '#8C8C8C',
    '$z^{MSE}$': '#4C72B0',
}

translation = {
    r'$\hat{z}^{kNN}_N(x)$': r'$\hat{z}^{kNN}_N$',
    r'$\hat{z}^{LOESS}_N(x)$': r'$\hat{z}^{LOESS}_N$',
    r'$\hat{z}^{KR}_N(x)$': r'$\hat{z}^{KR}_N$',
    r'$\hat{z}^{Rec.-KR}_N(x)$': r'$\hat{z}^{Rec.KR}_N$',
    r'$\hat{z}^{RF}_N(x)$': r'$\hat{z}^{RF}_N$',
    r'$\hat{z}^{CART}_N(x)$': r'$\hat{z}^{CART}_N$',
    r'$\hat{z}^{SAA}_N(x)$': r'$\hat{z}^{SAA}_N$',
    r'$\hat{z}^{DER}_N(x)$': r'$\hat{z}^{DER}_N$',
    r'$\hat{z}^{SPO+}_N(x)$': r'$\hat{z}^{SPO+}_N$',
    r'$\hat{z}^{SPO+}_N(x)_add$' : r'$\hat{z}^{SPO+}_N$',
    r'$z^{SPO+}(x)$': r'$z^{SPO+}$',
    r'$z^{SFGE}(x)$': r'$z^{SFGE}$',
    r'$z^{MSE}(x)$': r'$z^{MSE}$',
}


class PredictOptimizePipeline:
    """Core experimental workflow manager."""
    def __init__(self, data_sizes, data_generator, num_runs = 5):
        """
        data_sizes: Array of integers representing dataset sizes.
        data_generator: Callable that takes an integer (dataset size) 
                        and returns (x, c, optmodel).
        """
        self.data_sizes:np.ndarray = data_sizes
        self.data_generator = data_generator
        self.num_runs = num_runs
        self.models = {}
        self.results = {}

        processes = 0
        processes = mp.cpu_count() if processes == 0 else processes
        pool = ProcessingPool(processes)

        self.mutli_processing = [processes, pool]

    def add_model(self, name, model_type: WeightingTypeFunction, **kwargs):
        """Registers a predictive model."""
        self.models[name] = {'type': model_type, 'params': kwargs}
        self.results[name] = np.empty((len(self.data_sizes), self.num_runs), dtype=object)

    def _generate_cache_filepath(self, save_dir, model_name, config, num_data, run):
        """Generates a unique file path based on model parameters, data size, and run."""
        os.makedirs(save_dir, exist_ok=True)
        config_str = json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()
        filename = f"{model_name}_size_{num_data}_run_{run}_{config_hash}.pkl"
        return os.path.join(save_dir, filename)
    
    def _save_cached_model(self, filepath, predictor, result):
        """Safely serializes the model by isolating C-level bindings and PyTorch autograd functions."""
        detached_pointers = []
        
        # Attributes to detach: optimization models and unpicklable PyEPO training components
        unpicklable_attrs = ['optmodel', 'model', 'spo_plus', 'optimizer', 'early_stopper']
        
        # 1. Detach top-level references
        for attr in unpicklable_attrs:
            if hasattr(predictor, attr) and getattr(predictor, attr) is not None:
                detached_pointers.append((predictor, attr, getattr(predictor, attr)))
                setattr(predictor, attr, None)
                
        # 2. Detach 1st-level nested references inside object dictionaries
        if hasattr(predictor, '__dict__'):
            for attr, val in vars(predictor).items():
                if val is not None and hasattr(val, '__dict__'):
                    for sub_attr in unpicklable_attrs:
                        if hasattr(val, sub_attr) and getattr(val, sub_attr) is not None:
                            detached_pointers.append((val, sub_attr, getattr(val, sub_attr)))
                            setattr(val, sub_attr, None)

        try:
            with open(filepath, 'wb') as f:
                pickle.dump((predictor, result), f)
        except TypeError as e:
            # Fallback for pure PyTorch neural models 
            if hasattr(predictor, 'state_dict'):
                import torch
                torch.save({'state_dict': predictor.state_dict(), 'result': result}, filepath)
            else:
                raise RuntimeError(f"Failed to serialize {type(predictor)}: {e}")
        finally:
            # Reinject bindings to keep the in-memory object functional for the current run
            for obj, attr, val in detached_pointers:
                setattr(obj, attr, val)

    def _load_cached_model(self, filepath, optmodel):
        """Loads the serialized model and reinjects the required inference attributes."""
        try:
            with open(filepath, 'rb') as f:
                predictor, result = pickle.load(f)
        except (pickle.UnpicklingError, TypeError):
            # Fallback for pure PyTorch-based neural predictors
            import torch
            checkpoint = torch.load(filepath)
            result = checkpoint['result']
            predictor = None 
            
        if predictor is not None:
            # Re-inject top-level optimization models required for inference
            if hasattr(predictor, 'optmodel') or not hasattr(predictor, 'model'):
                predictor.optmodel = optmodel
            if hasattr(predictor, 'model') or not hasattr(predictor, 'optmodel'):
                predictor.model = optmodel
                
            # Re-inject 1st-level nested optimization models
            if hasattr(predictor, '__dict__'):
                for attr, val in vars(predictor).items():
                    if val is not None and hasattr(val, '__dict__'):
                        if 'optmodel' in val.__dict__:
                            val.optmodel = optmodel
                        if 'model' in val.__dict__:
                            val.model = optmodel
                            
        return predictor, result

    def save_results_to_csv(self, filepath="experimental_results.csv"):
        """
        Flattens the recorded experimental results and exports them to a CSV file.
        """
        data = []

        for model_name in self.models.keys():
            for idx, num_data in enumerate(self.data_sizes):
                for run in range(self.num_runs):
                    try:
                        # Retrieve the info dictionary for the specific iteration
                        info = self.results[model_name][idx, run]
                        
                        # Create the base identifier columns
                        row_data = {
                            'model_name': model_name,
                            'num_data': num_data,
                            'run': run
                        }
                        
                        # Merge the numerical metrics into the row
                        if isinstance(info, dict):
                            row_data.update(info)
                        else:
                            row_data['result'] = info
                            
                        data.append(row_data)
                        
                    except (KeyError, IndexError):
                        # Skips iterations that might have failed or are missing
                        continue

        df = pd.DataFrame(data)

        # Create the target directory if it does not exist
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        df.to_csv(filepath, index=False)

    def execute(self, save_dir, force_run=False):
        """Iterates through data sizes, trains models, records regret, and caches results."""
        for idx, num_data in enumerate(self.data_sizes):
            for run in range(self.num_runs):
                x_train, c_train, x_val, c_val, x_test, c_test, optmodel, aux = self.data_generator(num_data, seed=run)

                model_names_exlude = [r'$\hat{z}^{KR}_N(x)$', r'$\hat{z}^{LOESS}_N(x)$', r'$\hat{z}^{Rec.-KR}_N(x)$']

                for model_name, config in self.models.items():
                    cache_filepath = self._generate_cache_filepath(save_dir, model_name, config, num_data, run)

                    if os.path.exists(cache_filepath) and not force_run and model_name not in model_names_exlude:
                        print(f"Loading cached {model_name} | Size: {num_data} | Run: {run+1}/{self.num_runs}")
                        predictor, result = self._load_cached_model(cache_filepath, optmodel)
                        
                        if predictor is None:
                            # Reconstruct PyTorch architecture if state_dict was loaded
                            predictor = self._initialize_and_train(config, x_train, c_train, x_val, c_val, optmodel, m_train=aux.get('train'), m_val=aux.get('val'), seed=run)
                            checkpoint = torch.load(cache_filepath)
                            predictor.load_state_dict(checkpoint['state_dict'])
                            
                        self.results[model_name][idx, run] = result
                        
                    else:
                        print(f"Training {model_name} | Size: {num_data} | Run: {run+1}/{self.num_runs}")
                        predictor, info = self._initialize_and_train(config, x_train, c_train, x_val, c_val, optmodel, m_train=aux.get('train'), m_val=aux.get('val'), seed=run)

                        start_time = time.perf_counter()
                        result = test_model(predictor, optmodel, x_test, c_test, m_test=aux.get('test'), processes=self.mutli_processing[0], pool=self.mutli_processing[1])
                        end_time = time.perf_counter()
                        info['testing_time'] = end_time - start_time
                        info['result'] = result

                        self.results[model_name][idx, run] = info
                        self._save_cached_model(cache_filepath, predictor, info)

    def _initialize_and_train(self, config, x_train, c_train, x_val, c_val, optmodel, m_train = None, m_val = None, seed = None):
        """Handles specific model instantiation and training logic."""

        params = config.get('params').copy()
        match config["type"]:
            case WeightingTypeFunction.NEAREST_NEIGHBOUR:
                param_grid = params.get('param_grid')
                return finetune_predictive_prescription(NearestPrediction, x_train, c_train, x_val, c_val, optmodel, param_grid, m_val=m_val, seed=seed, mutli_processing=self.mutli_processing)
            
            case WeightingTypeFunction.LOESS:
                param_grid = params.get('param_grid')
                return finetune_predictive_prescription(LOESS, x_train, c_train, x_val, c_val, optmodel, param_grid, m_val=m_val, seed=seed, mutli_processing=self.mutli_processing)
            
            case WeightingTypeFunction.KERNEL:
                param_grid = params.get('param_grid')
                return finetune_predictive_prescription(KernelPrescription, x_train, c_train, x_val, c_val, optmodel, param_grid, m_val=m_val, seed=seed, mutli_processing=self.mutli_processing)
            
            case WeightingTypeFunction.RKERNEL:
                param_grid = params.get('param_grid')
                return finetune_predictive_prescription(RecursiveKernelPrescription, x_train, c_train, x_val, c_val, optmodel, param_grid, m_val=m_val, seed=seed, mutli_processing=self.mutli_processing)
            
            case WeightingTypeFunction.CART:
                feats = np.concatenate((x_train, x_val), axis=0)
                costs = np.concatenate((c_train, c_val), axis=0)
                start_time = time.perf_counter()
                model = CartPrescription(feats, costs, optmodel, seed=seed)
                end_time = time.perf_counter()
                training_info = {"training_time": end_time - start_time}
                return model, training_info
            
            case WeightingTypeFunction.SAA:
                feats = np.concatenate((x_train, x_val), axis=0)
                costs = np.concatenate((c_train, c_val), axis=0)
                start_time = time.perf_counter()
                model = SAA(feats, costs, optmodel, seed=seed)
                end_time = time.perf_counter()
                training_info = {"training_time": end_time - start_time}
                return model, training_info
        
            case WeightingTypeFunction.RANDOM_FOREST:
                param_grid = params.get('param_grid')
                return finetune_predictive_prescription(RandomForestPrescription, x_train, c_train, x_val, c_val, optmodel, param_grid, m_val=m_val, seed=seed, mutli_processing=self.mutli_processing)
        
            case WeightingTypeFunction.NEURAL:
                feats = np.concatenate((x_train, x_val), axis=0)
                costs = np.concatenate((c_train, c_val), axis=0)
                loss_type = params.pop('loss')

                weight_model_param_grid = params.get("weight_model_param_grid")

                train_param_grid = params.get("train_param_grid")

                return finetune_neural_prescription(
                    feats,
                    costs,
                    optmodel,
                    weight_model_param_grid,
                    train_param_grid,
                    loss_type,
                    seed=seed,
                )
            
            case WeightingTypeFunction.NEURAL_GROUPED:
                feats = np.concatenate((x_train, x_val), axis=0)
                costs = np.concatenate((c_train, c_val), axis=0)
                loss_type = params.pop('loss')

                weight_model_param_grid = params.get("weight_model_param_grid")

                train_param_grid = params.get("train_param_grid")

                return finetune_neural_prescription(
                    feats,
                    costs,
                    optmodel,
                    weight_model_param_grid,
                    train_param_grid,
                    loss_type,
                    grouped=True,
                    m_train=m_train,
                    m_val=m_val,
                    seed=seed
                )
            
            case WeightingTypeFunction.NEURAL_DFL:
                loss_type = params.pop('loss')

                weight_model_param_grid = params.get("dfl_predictor_param_grid")
                train_param_grid = params.get("train_param_grid")

                return dfl_finetune(
                    x_train,
                    c_train,
                    x_val,
                    c_val,
                    optmodel,
                    weight_model_param_grid,
                    train_param_grid,
                    loss_type,
                    seed=seed
                )
            
            case _:
                raise ValueError(f"Unknown model type {config['type']}")
            
    def _check_path(self, path):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

    def plot_results(self, save_path, title='Regret vs Number of Data Points'):
        """Plots mean regret with error bars representing standard deviation."""
        custom_settings = {
            'font.size': 20,
            'axes.titlesize': 21,
            'axes.labelsize': 20
        }

        with plt.rc_context(rc=custom_settings):
            plt.figure(figsize=(12, 5))
            
            for model_name, run_data in self.results.items():
                run_data = np.array([[run_dict["result"] for run_dict in run_list] for run_list in run_data])
                means = np.mean(run_data, axis=1)
                stds = np.std(run_data, axis=1)
                
                plt.errorbar(
                    self.data_sizes, 
                    means, 
                    yerr=stds, 
                    label=model_name,
                    fmt='-o',
                    capsize=5
                )
            
            plt.xlabel('Number of Data Points')
            plt.ylabel('Relative Regret')
            plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.ylim(bottom=0)
            plt.xlim(left=0)
            plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            self._check_path(save_path)
            plt.savefig(save_path)
            plt.close()


    def plot_boxplot(self, target_data_size, save_path, title=None):
        """Plots a boxplot for relative regret across models for a single dataset size."""
        custom_settings = {
            'font.size': 20,
            'axes.titlesize': 21,
            'axes.labelsize': 20
        }

        with plt.rc_context(rc=custom_settings):
            if target_data_size not in self.data_sizes:
                raise ValueError(f"Data size {target_data_size} not found in evaluated sizes.")

            # Identify the index for the requested data size
            size_idx = np.where(self.data_sizes == target_data_size)[0][0]
            model_names = list(self.results.keys())
            
            # data_to_plot = [self.results[name][size_idx, :]["result"] * 100 for name in model_names]
            data_to_plot = [
                [run_dict["result"] * 100 for run_dict in self.results[name][size_idx, :]] 
                for name in model_names
            ]

            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax = plt.subplots(figsize=(12, 5))

            box = ax.boxplot(data_to_plot, patch_artist=True, widths=0.4,
                        medianprops=dict(color='#4d4d4d', linewidth=1.5),
                        flierprops=dict(marker='d', markersize=4, markerfacecolor='#4d4d4d', alpha=0.8))


            colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']
            for patch, color in zip(box['boxes'], colors * (len(model_names) // len(colors) + 1)):
                patch.set_facecolor(color)
                patch.set_edgecolor('#4d4d4d')
                patch.set_alpha(0.9)

            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))

            ax.grid(True, axis='y', color='white', linestyle='-', linewidth=1)
            ax.grid(False, axis='x') # Usually no vertical lines in these plots
            ax.set_facecolor('#EAEAF2') # Standard seaborn gray
            
            ax.set_ylabel('relative regret')
            ax.set_xticklabels(model_names)
            
            ax.set_xlabel(f'Models (Data Size: {target_data_size})', labelpad=10)

            for spine in ax.spines.values():
                spine.set_visible(False)

            if title:
                plt.title(title, pad=20)

            plt.tight_layout()
            self._check_path(save_path)
            plt.savefig(save_path, dpi=300)
            plt.close()

    def plot_weight_distribution(self, save_dir, data_size, save_path, title=None):
        
        custom_settings = {
            'font.size': 18,
            'axes.titlesize': 18,
            'axes.labelsize': 18
        }


        with plt.rc_context(rc=custom_settings):
            num_models = len(self.models)
            # Calculate rows needed for 2 columns using ceiling division
            num_rows = (num_models + 1) // 2 
            
            fig, axes = plt.subplots(num_rows, 2, figsize=(15, 3 * num_rows), sharex=True)
            run = 0

            x_train, c_train, x_val, c_val, x_test, _, optmodel, _ = self.data_generator(data_size, seed=run)
            x_sample = x_test[0]

            # Flatten to 1D array to keep loop indexing clean regardless of grid shape
            axes = axes.flatten() 

            weights_len = 0

            for i, (model_name, config) in enumerate(self.models.items()):
                color = MODEL_COLOR_MAP.get(model_name)
                cache_filepath = self._generate_cache_filepath(save_dir, model_name, config, data_size, run)

                if os.path.exists(cache_filepath):
                    print(f"Loading cached {model_name} | Size: {data_size} | Run: {run+1}/{self.num_runs}")
                    predictor, result = self._load_cached_model(cache_filepath, optmodel)
                    print(predictor)
                else:
                    raise ValueError("predictor not found")

                if config['type'] == WeightingTypeFunction.NEURAL_DFL:
                    continue

                weights = predictor._get_weights(x_sample)
                
                if isinstance(weights, torch.Tensor):
                    weights = weights.detach().cpu().numpy().flatten()

                weights_len = len(weights)
                indices = np.arange(weights_len)
                
                # Plot directly to the corresponding axis
                ax = axes[i]
                ax.bar(indices, weights, alpha=0.8, color=color)
                
                print(translation.get(model_name))
                ax.set_title(translation.get(model_name))
                ax.set_ylabel('Weight Value')
                
                ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True, nbins=20))
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                ax.set_xlim(left=0, right=weights_len)

                if i in [5, 6]:
                    ax.set_xticks(range(0, 46, 5))
                    # Force Matplotlib to display the labels despite sharex=True
                    ax.tick_params(labelbottom=True)
                else:
                    # Ensure no labels are shown for others
                    plt.setp(ax.get_xticklabels(), visible=False)

            # Hide any unused subplots if the total number of models is odd
            for j in range(num_models, len(axes)):
                fig.delaxes(axes[j])

            # Assuming 'ax' is the axes object for the plot
            ax.set_xticks(range(0, 46, 5))
            # Use supxlabel for a centered x-axis label across multiple columns
            fig.supxlabel(r'Data Point Index $i$')
            plt.tight_layout()
            self._check_path(save_path)
            plt.savefig(save_path, dpi=300)
            plt.close()