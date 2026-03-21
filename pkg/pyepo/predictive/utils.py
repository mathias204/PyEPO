import copy
from pyepo.predictive.pred import PredictivePrescription
from pyepo import EPO
from pyepo.model.opt import optModel
from enum import Enum

class WeightingTypeFunction(Enum):
    NEURAL = "neural"
    NEAREST_NEIGHBOUR = "nearest_neighbour"
    RANDOM_FOREST = "random_forest"

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_state_dict = None

    def step(self, validation_loss, model):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
            return False 
        else:
            self.counter += 1
            if self.counter >= self.patience:
                # restore best weights and stop
                if self.best_state_dict is not None:
                    model.load_state_dict(self.best_state_dict)
                return True  # stop training
            return False    

def test_model(prediction_model: PredictivePrescription, opt_model: optModel, x_test, c_test):
    # TODO: can be made a little more efficient by batching, only setting objective and solving can't be batched
    loss = 0
    optsum = 0

    for x, true_cost in zip(x_test, c_test):

        pred_sol, _ = prediction_model.optimize(x)

        opt_model.setObj(true_cost)
        _, true_obj = opt_model.solve()

        pred_obj = opt_model.cal_obj(true_cost, pred_sol)

        if opt_model.modelSense == EPO.MINIMIZE:
            loss += pred_obj - true_obj
        if opt_model.modelSense == EPO.MAXIMIZE:
            loss += true_obj - pred_obj

        optsum += abs(true_obj)

    return loss/(optsum + 1e-7)
