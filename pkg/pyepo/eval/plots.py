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
import re
import matplotlib.patches as mpatches

custom_settings = {
    'font.size': 18,
    'axes.titlesize': 11,
    'axes.labelsize': 18
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
    r'$z^{SPO+}(x)$': r'$z^{SPO+}$',
    r'$z^{SFGE}(x)$': r'$z^{SFGE}$',
    r'$z^{MSE}(x)$': r'$z^{MSE}$',
    
}

MODEL_COLOR_MAP = {
    '$\\hat{z}^{kNN}_N$': '#4C72B0',
    '$\\hat{z}^{LOESS}_N$': '#55A868',
    '$\\hat{z}^{KR}_N$': '#C44E52',
    '$\\hat{z}^{Rec.KR}_N$': '#8172B2',
    '$\\hat{z}^{RF}_N$': '#CCB974',
    '$\\hat{z}^{CART}_N$': '#64B5CD',
    '$\\hat{z}^{DER}_N$': '#DD8452',
    '$\\hat{z}^{SPO+}_N$': '#937860',
    r'$\hat{z}^{SPO+}_{dot}$' : '#937860',
    r'$\hat{z}^{DER}_{dot}$' : '#DD8452',
    r'$\hat{z}^{DER}_{add}$': '#A5592E',
    r'$\hat{z}^{SPO+}_{add}$': '#634E3C',
    '$z^{SPO+}$': '#DA8BC3',
    '$z^{SFGE}$': '#8C8C8C',
    '$z^{MSE}$': '#4C72B0',
}

def parse_tensor_string(val):
    if isinstance(val, str) and "tensor" in val:
        # Extract the number inside the parentheses
        match = re.search(r"tensor\((.*?)\)", val)
        if match:
            return float(match.group(1))
    return float(val)


def _check_path(path):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

def plot_boxplot(csv_path, save_path, title=None):
    """Plots a boxplot for relative regret across models for a single dataset size."""
    with plt.rc_context(rc=custom_settings):

        df = pd.read_csv(csv_path)

        model_names = list(df['model_name'].unique())
        if r'$\hat{z}^{SAA}_N(x)$' in model_names:
            model_names.remove(r'$\hat{z}^{SAA}_N(x)$')
        if r'$\hat{z}^{CART}_N(x)$' in model_names:
            model_names.remove(r'$\hat{z}^{CART}_N(x)$')
            
        data_to_plot = [df[df['model_name'] == name]['result'].apply(parse_tensor_string) * 100 for name in model_names]
        data_size = df['num_data'].iloc[0]
        num_models = len(model_names)

        for i, name in enumerate(model_names):
            if name in translation:
                model_names[i] = translation[name]

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 3))

        # Add background shading for the last two models to distinguish them
        if num_models >= 2:
            # Boxplots are positioned at x=1, 2, ..., num_models
            # Shading starts halfway before the second-to-last model
            ax.axvspan(num_models - 2.5, num_models + 0.5, color='#FADBD8', alpha=1, zorder=0)

        # ax.axvspan(0.5, num_models - 2.5, color="#FADBD8", alpha=1, zorder=0)

        # Added zorder=2 to ensure the boxes are drawn on top of the shading
        box = ax.boxplot(data_to_plot, patch_artist=True, widths=0.8,
                    medianprops=dict(color='#4d4d4d', linewidth=1.5),
                    flierprops=dict(marker='d', markersize=4, markerfacecolor='#4d4d4d', alpha=0.8),
                    zorder=2)


        for patch, name in zip(box['boxes'], model_names):
            color = MODEL_COLOR_MAP.get(name, '#000000') 
            patch.set_facecolor(color)
            patch.set_edgecolor('#4d4d4d')
            patch.set_alpha(0.9)

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))

        # Added zorder=1 so the white grid lines stay visible over the blue shading
        ax.grid(True, axis='y', color='white', linestyle='-', linewidth=1, zorder=1)
        ax.grid(False, axis='x') 
        ax.set_facecolor('#EAEAF2') 
        
        ax.set_ylabel('Relative Regret')
        ax.set_xticklabels(model_names)

        for spine in ax.spines.values():
            spine.set_visible(False)

        if title:
            plt.title(title, pad=20)

        # Create proxy artists for the legend
        wpp_patch = mpatches.Patch(color='#EAEAF2', label='WPP')
        other_patch = mpatches.Patch(color='#FADBD8', label='Other')
        
        # Add the legend to the plot
        # loc='upper left' usually works well, but you can change it to 'best' or place it outside the bounding box
        ax.legend(handles=[wpp_patch, other_patch], loc='upper right', frameon=True, facecolor='white')

        plt.tight_layout()
        _check_path(save_path)
        plt.savefig(save_path, dpi=300)
        plt.close()

def plot_training_time(csv_path, save_path, title=None, exclude_models=None):
    """Plots a bar chart of mean training time with standard deviation across models."""
    with plt.rc_context(rc=custom_settings):

        df = pd.read_csv(csv_path)

        model_names = list(df['model_name'].unique())
        if r'$\hat{z}^{SAA}_N(x)$' in model_names:
            model_names.remove(r'$\hat{z}^{SAA}_N(x)$')

        if exclude_models:
            model_names = [name for name in model_names if name not in exclude_models]

        data_to_plot = [df[df['model_name'] == name]['training_time'] for name in model_names]
        
        means = [np.mean(data) for data in data_to_plot]
        stds = [np.std(data) for data in data_to_plot]
        
        num_models = len(model_names)

        for i, name in enumerate(model_names):
            if name in translation:
                model_names[i] = translation[name]

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(8, 4))

        # Bar plots are 0-indexed (0, 1, ..., num_models - 1)
        # Shading the last 3 models covers from N-3.5 to N-0.5
        # if num_models >= 3:
        #     ax.axvspan(num_models - 3.5, num_models - 0.5, color='#FADBD8', alpha=1, zorder=0)

        # # colors = [
        #     '#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', 
        #     '#64B5CD', '#DD8452', '#937860', '#DA8BC3', '#8C8C8C'
        # ]
        x_pos = np.arange(num_models)
        
        bars = ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.9, 
                      ecolor='#4d4d4d', capsize=5, zorder=2)

        for bar, name in zip(bars, model_names):
            color = MODEL_COLOR_MAP.get(name, '#000000')
            bar.set_facecolor(color)
            bar.set_edgecolor('#4d4d4d')
            bar.set_linewidth(1.5)

        ax.grid(True, axis='y', color='white', linestyle='-', linewidth=1, zorder=1)
        ax.grid(False, axis='x') 
        ax.set_facecolor('#EAEAF2') 
        
        ax.set_ylabel('Training Time (sec)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names)

        # Removes the default matplotlib white space padding on the left and right
        ax.set_xlim(-0.5, num_models - 0.5)

        for spine in ax.spines.values():
            spine.set_visible(False)

        if title:
            plt.title(title, pad=20)

        # Create proxy artists for the legend
        wpp_patch = mpatches.Patch(color='#EAEAF2', label='WPP')
        other_patch = mpatches.Patch(color='#FADBD8', label='Other')
        
        # Add the legend to the plot
        # loc='upper left' usually works well, but you can change it to 'best' or place it outside the bounding box
        # ax.legend(handles=[wpp_patch, other_patch], loc='upper right', frameon=True, facecolor='white')

        plt.tight_layout()
        print(save_path)
        _check_path(save_path)
        plt.savefig(save_path, dpi=300)
        plt.close()

def plot_inference_time(csv_path, save_path, title=None):
    """Plots a bar chart of mean inference time with standard deviation across models."""
    with plt.rc_context(rc=custom_settings):

        df = pd.read_csv(csv_path)

        model_names = list(df['model_name'].unique())
        if r'$\hat{z}^{SAA}_N(x)$' in model_names:
            model_names.remove(r'$\hat{z}^{SAA}_N(x)$')
        if r'$\hat{z}^{CART}_N(x)$' in model_names:
            model_names.remove(r'$\hat{z}^{CART}_N(x)$')
            
        data_to_plot = [df[df['model_name'] == name]['testing_time'] for name in model_names]
        
        means = [np.mean(data) for data in data_to_plot]
        stds = [np.std(data) for data in data_to_plot]
        
        num_models = len(model_names)

        for i, name in enumerate(model_names):
            if name in translation:
                model_names[i] = translation[name]

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 3))

        # Bar plots are 0-indexed (0, 1, ..., num_models - 1)
        # Shading the last 3 models covers from N-3.5 to N-0.5
        if num_models >= 3:
            ax.axvspan(num_models - 3.5, num_models - 0.5, color='#FADBD8', alpha=1, zorder=0)

        # colors = [
        #     '#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', 
        #     '#64B5CD', '#DD8452', '#937860', '#DA8BC3', '#8C8C8C'
        # ]

        x_pos = np.arange(num_models)
        
        bars = ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.9, 
                      ecolor='#4d4d4d', capsize=5, zorder=2)

        for bar, name in zip(bars, model_names):
            color = MODEL_COLOR_MAP.get(name, '#000000')
            bar.set_facecolor(color)
            bar.set_edgecolor('#4d4d4d')
            bar.set_linewidth(1.5)

        ax.grid(True, axis='y', color='white', linestyle='-', linewidth=1, zorder=1)
        ax.grid(False, axis='x') 
        ax.set_facecolor('#EAEAF2') 
        
        ax.set_ylabel('Inference Time (sec)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names)

        # Removes the default matplotlib white space padding on the left and right
        ax.set_xlim(-0.5, num_models - 0.5)

        for spine in ax.spines.values():
            spine.set_visible(False)

        if title:
            plt.title(title, pad=20)

                # Create proxy artists for the legend
        wpp_patch = mpatches.Patch(color='#EAEAF2', label='WPP')
        other_patch = mpatches.Patch(color='#FADBD8', label='Other')
        
        # Add the legend to the plot
        # loc='upper left' usually works well, but you can change it to 'best' or place it outside the bounding box
        ax.legend(handles=[wpp_patch, other_patch], loc='upper right', frameon=True, facecolor='white')

        plt.tight_layout()
        _check_path(save_path)
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_epoch_time(csv_path, save_path, title=None):
    """Plots a bar chart of mean epoch time with standard deviation across models."""
    with plt.rc_context(rc=custom_settings):

        df = pd.read_csv(csv_path)

        model_names = list(df['model_name'].unique())
        # remove SAA estimator if present
        if r'$\hat{z}^{SAA}_N(x)$' in model_names:
            model_names.remove(r'$\hat{z}^{SAA}_N(x)$')

        # only keep model names that have at least one non-NaN mean_epoch_time entry
        if 'mean_epoch_time' in df.columns:
            filtered_names = []
            for name in model_names:
                series = df[df['model_name'] == name]['mean_epoch_time'].dropna()
                if series.size > 0:
                    filtered_names.append(name)
            model_names = filtered_names

        data_to_plot = [df[df['model_name'] == name]['mean_epoch_time'] for name in model_names]
        
        means = [np.mean(data) for data in data_to_plot]
        stds = [np.std(data) for data in data_to_plot]
        
        num_models = len(model_names)

        for i, name in enumerate(model_names):
            if name in translation:
                model_names[i] = translation[name]

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(8, 3))

        # Bar plots are 0-indexed (0, 1, ..., num_models - 1)
        # Shading the last 3 models covers from N-3.5 to N-0.5
        if num_models >= 3:
            ax.axvspan(num_models - 3.5, num_models - 0.5, color='#FADBD8', alpha=1, zorder=0)

        # colors = [
        #     '#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', 
        #     '#64B5CD', '#DD8452', '#937860', '#DA8BC3', '#8C8C8C'
        # ]

        x_pos = np.arange(num_models)
        
        bars = ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.9, 
                      ecolor='#4d4d4d', capsize=5, zorder=2)

        for bar, name in zip(bars, model_names):
            color = MODEL_COLOR_MAP.get(name, '#000000')
            bar.set_facecolor(color)
            bar.set_edgecolor('#4d4d4d')
            bar.set_linewidth(1.5)

        ax.grid(True, axis='y', color='white', linestyle='-', linewidth=1, zorder=1)
        ax.grid(False, axis='x') 
        ax.set_facecolor('#EAEAF2') 
        
        ax.set_ylabel('Epoch Time (sec)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names)

        # Removes the default matplotlib white space padding on the left and right
        ax.set_xlim(-0.5, num_models - 0.5)

        for spine in ax.spines.values():
            spine.set_visible(False)

        if title:
            plt.title(title, pad=20)

        # Create proxy artists for the legend
        wpp_patch = mpatches.Patch(color='#EAEAF2', label='WPP')
        other_patch = mpatches.Patch(color='#FADBD8', label='Other')
        
        # Add the legend to the plot
        # loc='upper left' usually works well, but you can change it to 'best' or place it outside the bounding box
        ax.legend(handles=[wpp_patch, other_patch], loc='upper right', frameon=True, facecolor='white')

        plt.tight_layout()
        _check_path(save_path)
        plt.savefig(save_path, dpi=300)
        plt.close()

def plot_num_epochs(csv_path, save_path, title=None, epoch_col='final_epoch'):
    """Plots a bar chart of the mean number of epochs taken to train across models with standard deviation."""
    with plt.rc_context(rc=custom_settings):

        df = pd.read_csv(csv_path)

        model_names = list(df['model_name'].unique())
        # remove SAA estimator if present
        if r'$\hat{z}^{SAA}_N(x)$' in model_names:
            model_names.remove(r'$\hat{z}^{SAA}_N(x)$')

        # only keep model names that have at least one non-NaN entry for the epoch column
        if epoch_col in df.columns:
            filtered_names = []
            for name in model_names:
                series = df[df['model_name'] == name][epoch_col].dropna()
                if series.size > 0:
                    filtered_names.append(name)
            model_names = filtered_names

        data_to_plot = [df[df['model_name'] == name][epoch_col] for name in model_names]
        
        means = [np.mean(data) for data in data_to_plot]
        stds = [np.std(data) for data in data_to_plot]
        
        num_models = len(model_names)

        for i, name in enumerate(model_names):
            if name in translation:
                model_names[i] = translation[name]

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(8, 3))

        # Bar plots are 0-indexed (0, 1, ..., num_models - 1)
        # Shading the last 3 models covers from N-3.5 to N-0.5
        if num_models >= 3:
            ax.axvspan(num_models - 3.5, num_models - 0.5, color='#FADBD8', alpha=1, zorder=0)

        x_pos = np.arange(num_models)
        
        bars = ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.9, 
                      ecolor='#4d4d4d', capsize=5, zorder=2)

        for bar, name in zip(bars, model_names):
            color = MODEL_COLOR_MAP.get(name, '#000000')
            bar.set_facecolor(color)
            bar.set_edgecolor('#4d4d4d')
            bar.set_linewidth(1.5)

        ax.grid(True, axis='y', color='white', linestyle='-', linewidth=1, zorder=1)
        ax.grid(False, axis='x') 
        ax.set_facecolor('#EAEAF2') 
        
        ax.set_ylabel('Number of Epochs')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names)

        # Removes the default matplotlib white space padding on the left and right
        ax.set_xlim(-0.5, num_models - 0.5)

        for spine in ax.spines.values():
            spine.set_visible(False)

        if title:
            plt.title(title, pad=20)

        # Create proxy artists for the legend
        wpp_patch = mpatches.Patch(color='#EAEAF2', label='WPP')
        other_patch = mpatches.Patch(color='#FADBD8', label='Other')
        
        # Add the legend to the plot
        ax.legend(handles=[wpp_patch, other_patch], loc='upper right', frameon=True, facecolor='white')

        plt.tight_layout()
        _check_path(save_path)
        plt.savefig(save_path, dpi=300)
        plt.close()



if __name__ == "__main__":
    degree = 8
