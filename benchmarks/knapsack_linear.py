import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pyepo
from pyepo.model.grb import optGrbModel
import torch
from pyepo.predictive.neural import NeuralPrediction
from torch import nn
from pyepo.predictive.nn import NearestPrediction
from pyepo.predictive.forest import RandomForestPrescription
from sklearn.model_selection import train_test_split
from pyepo.predictive.utils import test_model
import matplotlib.pyplot as plt

# Weight model
class weigth_prediction(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, features): 
        """
        x: [B, D] query features
        features: [N, D] reference features
        returns: [B, N] normalized weights
        """
        B, N, D = features.shape

        # expand to compare every query with all reference features
        x_exp = x.unsqueeze(1).expand(-1, N, -1)        # [B, N, D]

        # concatenate query with corresponding reference features
        inp = torch.cat([x_exp, features], dim=-1)      # [B, N, 2D]

        weights = self.net(inp).squeeze(-1)
        weights = torch.softmax(weights, dim=1)
        return weights
    

# optimization model
class knapSackModel(optGrbModel):
    def __init__(self, weights):
        self.weights = np.array(weights)
        self.num_item = len(weights[0])
        super().__init__()

    def _getModel(self):
        # ceate a model
        m = gp.Model()
        # varibles
        x = m.addVars(self.num_item, name="x", vtype=GRB.BINARY)
        # model sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        m.addConstr(gp.quicksum([self.weights[0,i] * x[i] for i in range(self.num_item)]) <= 7)
        m.addConstr(gp.quicksum([self.weights[1,i] * x[i] for i in range(self.num_item)]) <= 8)
        m.addConstr(gp.quicksum([self.weights[2,i] * x[i] for i in range(self.num_item)]) <= 9)
        return m, x
    
    def cal_obj(self, c, x):
        # check if c is a PyTorch tensor
        if isinstance(c, torch.Tensor):
            c = c.detach().cpu().numpy()
        else:
            c = np.asarray(c, dtype=np.float32)

        # check if x is a PyTorch tensor
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x, dtype=np.float32)
        if c.shape[-1] != x.shape[-1]:
            raise ValueError(f"Mismatch: c has {c.shape[-1]} features, expected {x.shape[-1]}.")
            
        # Case 1: x shape (C)
        if x.ndim == 1:
            # c shape (C,)
            return np.dot(c, x)

        # Case 2: x shape (B, C)
        elif x.ndim == 2:
            # c can be (C,) or (B, C)
            if c.ndim == 1:
                # simple broadcasting
                return np.sum(c * x, axis=-1)
            elif c.ndim == 2:
                return np.sum(c * x, axis=-1)
            else:
                raise ValueError("c must be shape (C,) or (B, C)")

        # Case 3: x shape (B, S, C)
        elif x.ndim == 3:
            # c can be (C,) or (B, C)
            if c.ndim == 1:
                c_exp = c[None, None, :]       # broadcast to (1, 1, C)
            elif c.ndim == 2:
                c_exp = c[:, None, :]          # broadcast to (B, 1, C)
            else:
                raise ValueError("c must be shape (C,) or (B, C)")

            return np.sum(c_exp * x, axis=-1)   # output (B, S)

        else:
            raise ValueError(f"Unsupported x shape {x.shape}")
        
numbers_data = np.linspace(10,750,20).astype(int)
# numbers_data = np.linspace(100,300,2).astype(int)

nearest_neigbor_regrets = np.zeros(len(numbers_data))
neural_regrets_SFGE = np.zeros(len(numbers_data))
neural_regrets_SPO = np.zeros(len(numbers_data))
neural_regrets_NOVEL = np.zeros(len(numbers_data))
rf_regrets = np.zeros(len(numbers_data))

num_feat = 5 # size of feature
num_item = 10 # number of items

for idx, num_data in enumerate(numbers_data):
    # generate data
    weights, x, c = pyepo.data.knapsack.genData(num_data, num_feat, num_item,dim=3, deg=4, noise_width=0.5, seed=135)

    x_train, x_test, c_train, c_test = train_test_split(x, c, test_size=0.2, random_state=42) #This split is not ideal in this scenario
    optmodel = knapSackModel(weights) 

    nearest_neigbor_predictor = NearestPrediction(x_train, c_train, 5, optmodel)

    weight_model = weigth_prediction(x_train.shape[1])
    neural_predictor_SFGE = NeuralPrediction(x_train, c_train, weight_model, optmodel)
    neural_predictor_SFGE.train_model(epochs=1000, loss_type=pyepo.predictive.neural.LossType.SFGE)

    weight_model = weigth_prediction(x_train.shape[1])
    neural_predictor_SPO = NeuralPrediction(x_train, c_train, weight_model, optmodel)
    neural_predictor_SPO.train_model(epochs=500, loss_type=pyepo.predictive.neural.LossType.SPO)

    weight_model = weigth_prediction(x_train.shape[1])
    neural_predictor_NOVEL = NeuralPrediction(x_train, c_train, weight_model, optmodel)
    neural_predictor_NOVEL.train_model(epochs=500, loss_type=pyepo.predictive.neural.LossType.NOVEL)

    rf_predictor = RandomForestPrescription(x_train, c_train, optmodel)

    #Run tests
    nearest_neigbor_regrets[idx] = test_model(nearest_neigbor_predictor, optmodel, x_test, c_test)
    neural_regrets_SFGE[idx] = test_model(neural_predictor_SFGE, optmodel, x_test, c_test)
    neural_regrets_SPO[idx] = test_model(neural_predictor_SPO, optmodel, x_test, c_test)
    neural_regrets_NOVEL[idx] = test_model(neural_predictor_NOVEL, optmodel, x_test, c_test)
    rf_regrets[idx] = test_model(rf_predictor, optmodel, x_test, c_test)


plt.figure(figsize=(10, 6))
plt.plot(numbers_data, nearest_neigbor_regrets, label='Nearest Neighbor')
plt.plot(numbers_data, neural_regrets_SFGE, label='Neural Network - SFGE')
plt.plot(numbers_data, neural_regrets_SPO, label='Neural Network - SPO')
plt.plot(numbers_data, neural_regrets_NOVEL, label='Neural Network - NOVEL')
plt.plot(numbers_data, rf_regrets, label='Random Forest')
plt.xlabel('Number of Data Points')
plt.ylabel('Regret')
plt.title('Regret vs Number of Data Points')
plt.legend()
plt.grid(True)
# plt.show()

# save figure in /results
plt.savefig('results/knapsack_linear_regret.png')