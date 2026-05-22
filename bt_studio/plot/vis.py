import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D


def plot_tune_landscape_3d(results_df, param_x, param_y, target="metrics_score"):
    """
        3D Meshgrid
    """
    # 过滤无效值 (-inf 或 nan)
    valid_df = results_df[np.isfinite(results_df[target])].copy()
    if len(valid_df) < 5:
        return

    x = valid_df[param_x].values
    y = valid_df[param_y].values
    z = valid_df[target].values

    # 2D Meshgrid
    grid_x, grid_y = np.mgrid[
        min(x):max(x):100j, # 100j insert 100 points
        min(y):max(y):100j
    ]

    # method='cubic' / 'linear'
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', 
                           edgecolor='none', alpha=0.8)

    # scatter 
    ax.scatter(x, y, z, color='black', s=15, zorder=3, label='Sampled Trials')

    ax.set_xlabel(param_x.replace('config/', ''))
    ax.set_ylabel(param_y.replace('config/', ''))
    ax.set_zlabel(target)
    ax.set_title(f"Hyperparameter Landscape: {target}")
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.legend()
    plt.show()

