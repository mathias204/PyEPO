"""
This script includes code adapted from the PredOpt benchmarks repository:
https://github.com/PredOpt/predopt-benchmarks
"""
import pickle

def get_cora():
    """
    Get X,y
    """
    # 
    with open('data/cora_data.pickle', 'rb') as f:
        gt, ft, M = pickle.load(f)
    return ft, gt, M