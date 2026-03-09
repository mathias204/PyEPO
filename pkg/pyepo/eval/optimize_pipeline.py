import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pyepo.predictive import NeuralPrediction, NearestPrediction, RandomForestPrescription
from pyepo.predictive.utils import test_model, WeightingTypeFunction

class PredictOptimizePipeline:
    """Core experimental workflow manager."""
    def __init__(self, data_sizes, data_generator, num_runs = 5):
        """
        data_sizes: Array of integers representing dataset sizes.
        data_generator: Callable that takes an integer (dataset size) 
                        and returns (x, c, optmodel).
        """
        self.data_sizes = data_sizes
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
                x, c, optmodel = self.data_generator(num_data)
                
                x_train, x_test, c_train, c_test = train_test_split(
                    x, c, test_size=0.2, random_state=run 
                )

                for model_name, config in self.models.items():
                    print(f"Training {model_name} | Size: {num_data} | Run: {run+1}/{self.num_runs}")
                    predictor = self._initialize_and_train(config, x_train, c_train, optmodel)
                    self.results[model_name][idx, run] = test_model(predictor, optmodel, x_test, c_test)

    def _initialize_and_train(self, config, x_train, c_train, optmodel):
        """Handles specific model instantiation and training logic."""

        match config["type"]:
            case WeightingTypeFunction.NEAREST_NEIGBHOUR:
                k = config['params'].get('k', 5)
                return NearestPrediction(x_train, c_train, k, optmodel)
        
            case WeightingTypeFunction.RANDOM_FOREST:
                return RandomForestPrescription(x_train, c_train, optmodel)
        
            case WeightingTypeFunction.NEURAL:
                weight_model = config['params']['weight_model'](x_train.shape[1])
                predictor = NeuralPrediction(x_train, c_train, weight_model, optmodel)
                predictor.train_model(epochs=config['params']['epochs'], loss_type=config['params']['loss'])
                return predictor
            
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