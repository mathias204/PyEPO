from pyepo.predictive.pred import Predictor
import torch

class DFLPredictor(Predictor):
    def __init__(self, optmodel, neural_model):
        self.model = optmodel
        self.neural_model = neural_model

    
    def optimize(self, x, m = None): 
        # Predict
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

        device = next(self.neural_model.parameters()).device
        x = x.to(device)

        with torch.no_grad():
            pred_costs = self.neural_model(x)

        
        self.model.setObj(pred_costs)
        sol, obj = self.model.solve()

        return sol, obj