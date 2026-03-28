import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyepo.predictive import NearestPrediction, RandomForestPrescription, LOESS, KernelPrescription, RecursiveKernelPrescription, CartPrescription, SAA
from pyepo.predictive.utils import test_model, WeightingTypeFunction, finetune_predictive_prescription, finetune_neural_prescription

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

    def add_model(self, name, model_type: WeightingTypeFunction, **kwargs):
        """Registers a predictive model."""
        self.models[name] = {'type': model_type, 'params': kwargs}
        self.results[name] = np.zeros((len(self.data_sizes), self.num_runs))

    def execute(self):
        """Iterates through data sizes, trains models, and records regret."""
        for idx, num_data in enumerate(self.data_sizes):
            for run in range(self.num_runs):
                x_train, c_train, x_val, c_val, x_test, c_test, optmodel = self.data_generator(num_data)

                for model_name, config in self.models.items():
                    print(f"Training {model_name} | Size: {num_data} | Run: {run+1}/{self.num_runs}")
                    predictor = self._initialize_and_train(config, x_train, c_train, x_val, c_val, optmodel)
                    self.results[model_name][idx, run] = test_model(predictor, optmodel, x_test, c_test)

    def _initialize_and_train(self, config, x_train, c_train, x_val, c_val, optmodel):
        """Handles specific model instantiation and training logic."""

        params = config.get('params').copy()
        match config["type"]:
            case WeightingTypeFunction.NEAREST_NEIGHBOUR:
                param_grid = params.get('param_grid')
                return finetune_predictive_prescription(NearestPrediction, x_train, c_train, x_val, c_val, optmodel, param_grid)
            
            case WeightingTypeFunction.LOESS:
                param_grid = params.get('param_grid')
                return finetune_predictive_prescription(LOESS, x_train, c_train, x_val, c_val, optmodel, param_grid)
            
            case WeightingTypeFunction.KERNEL:
                param_grid = params.get('param_grid')
                return finetune_predictive_prescription(KernelPrescription, x_train, c_train, x_val, c_val, optmodel, param_grid)
            
            case WeightingTypeFunction.RKERNEL:
                param_grid = params.get('param_grid')
                return finetune_predictive_prescription(RecursiveKernelPrescription, x_train, c_train, x_val, c_val, optmodel, param_grid)
            
            case WeightingTypeFunction.CART:
                feats = np.concatenate((x_train, x_val), axis=0)
                costs = np.concatenate((c_train, c_val), axis=0)
                return CartPrescription(feats, costs, optmodel)
            
            case WeightingTypeFunction.SAA:
                feats = np.concatenate((x_train, x_val), axis=0)
                costs = np.concatenate((c_train, c_val), axis=0)
                return SAA(feats, costs, optmodel)
        
            case WeightingTypeFunction.RANDOM_FOREST:
                param_grid = params.get('param_grid')
                return finetune_predictive_prescription(RandomForestPrescription, x_train, c_train, x_val, c_val, optmodel, param_grid)
        
            case WeightingTypeFunction.NEURAL:
                feats = np.concatenate((x_train, x_val), axis=0)
                costs = np.concatenate((c_train, c_val), axis=0)
                weight_model_class = params.pop('weight_model')
                loss_type = params.pop('loss')

                weight_model_param_grid = params.get("weight_model_param_grid")

                train_param_grid = params.get("train_param_grid")

                return finetune_neural_prescription(
                    feats,
                    costs,
                    optmodel,
                    weight_model_class,
                    weight_model_param_grid,
                    train_param_grid,
                    loss_type
                )
            
            case _:
                ValueError(f"Unknown model type {config['type']}")

    def plot_results(self, save_path, title='Regret vs Number of Data Points'):
        """Plots mean regret with shaded area representing standard deviation."""
        plt.figure(figsize=(10, 6))
        
        for model_name, run_data in self.results.items():
            means = np.mean(run_data, axis=1)
            stds = np.std(run_data, axis=1)
            
            line = plt.plot(self.data_sizes, means, label=model_name)
            color = line[0].get_color()
            
            # Shaded region for variance
            plt.fill_between(
                self.data_sizes, 
                means - stds, 
                means + stds, 
                color=color, 
                alpha=0.2
            )
        
        plt.xlabel('Number of Data Points')
        plt.ylabel('Regret')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def plot_normalized_bar_chart(self, target_data_size, reference_model_name, save_path, title=None):
        """Plots a normalized bar chart for a single dataset size with reference model baseline."""
        if target_data_size not in self.data_sizes:
            raise ValueError(f"Data size {target_data_size} not found in evaluated sizes.")
        if reference_model_name not in self.results:
            raise ValueError(f"Reference model {reference_model_name} not found in results.")

        # Calculate normalized performance
        size_idx = np.argwhere(self.data_sizes == target_data_size)[0]
        model_names = list(self.results.keys())
        
        ref_data = self.results[reference_model_name][size_idx, :]

        means = []
        stds = []
        for name in model_names:
            run_data = self.results[name][size_idx, :]
            
            normalized_data = run_data / ref_data
            
            means.append(np.mean(normalized_data))
            stds.append(np.std(normalized_data))


        # Plot figure    
        fig, ax = plt.subplots(figsize=(6, 5))

        colors = ['#353155', '#556488', '#B7D8D6', '#8CAEC1', '#6987A7', '#D88A30']
        colors = [colors[i % len(colors)] for i in range(len(model_names))]

        width = 1.0
        x_pos = np.arange(len(model_names)) * width

        ax.bar(x_pos, means, width, color=colors, 
               yerr=stds, error_kw=dict(ecolor='black', lw=3, capsize=0))

        ax.set_ylabel('normalized absolute regret', fontsize=14)
        
        ax.set_xticks([x_pos.mean()])
        ax.set_xticklabels([f'Size: {target_data_size}'], fontsize=14)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.tick_params(axis='y', labelsize=12)

        legend_patches = [mpatches.Patch(color=colors[i], label=name) 
                          for i, name in enumerate(model_names)]
        
        lgd = ax.legend(handles=legend_patches, loc='upper center', 
                  bbox_to_anchor=(0.5, 1.15), ncol=min(3, len(model_names)), 
                  frameon=False, fontsize=12, handlelength=2.5)

        if title:
            plt.title(title, pad=60, fontsize=14)

        plt.savefig(save_path, dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()