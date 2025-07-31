#!/usr/bin/env python3
"""
Simple FIS Model Results Plotter
Creates gradient plots of all 6 FIS model outputs.
"""

import rasterio
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os

def plot_all_models():
    """Plot all 6 FIS models with gradient colors."""
    
    # Files to plot
    files = [
        'app/files/output/output_300m_config_max.tif',
        'app/files/output/output_300m_config_median.tif',
        'app/files/output/output_300m_config_minimum.tif',
        'app/files/output/output_300m_config_mode.tif',
        'app/files/output/output_300m_config_round_down.tif',
        'app/files/output/output_300m_config_round_up.tif'
    ]
    
    names = ['Maximum', 'Median', 'Minimum', 'Mode', 'Round Down', 'Round Up']
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    # Get global min/max for consistent scaling
    all_data = []
    for f in files:
        with rasterio.open(f) as src:
            all_data.append(src.read(1))
    
    vmin = min(np.nanmin(d) for d in all_data)
    vmax = max(np.nanmax(d) for d in all_data)
    
    print(f"Plotting with range: {vmin:.2f} to {vmax:.2f}")
    
    # Plot each model
    for i, (file_path, name) in enumerate(zip(files, names)):
        print(f"Plotting {name}...")
        
        with rasterio.open(file_path) as src:
            data = src.read(1)
            
            ax = axes[i]
            im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
            
            ax.set_title(name, fontsize=14, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add stats
            mean_val = np.nanmean(data)
            ax.text(0.02, 0.98, f'Mean: {mean_val:.2f}', 
                   transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes, shrink=0.8, aspect=30)
    cbar.set_label('Priority Score (0-10)', fontsize=12)
    
    # Main title
    fig.suptitle('FIS Model Results - 300m Resolution', fontsize=16, fontweight='bold')
    
    # Save
    os.makedirs('plots', exist_ok=True)
    output_path = 'plots/fis_results_gradient.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved: {output_path}")
    
    plt.close()
    return output_path

def plot_statistics():
    """Create statistics comparison."""
    
    files = [
        'app/files/output/output_300m_config_max.tif',
        'app/files/output/output_300m_config_median.tif',
        'app/files/output/output_300m_config_minimum.tif',
        'app/files/output/output_300m_config_mode.tif',
        'app/files/output/output_300m_config_round_down.tif',
        'app/files/output/output_300m_config_round_up.tif'
    ]
    
    names = ['Maximum', 'Median', 'Minimum', 'Mode', 'Round Down', 'Round Up']
    
    # Collect stats
    stats = []
    for file_path, name in zip(files, names):
        with rasterio.open(file_path) as src:
            data = src.read(1)
            stats.append({
                'name': name,
                'mean': np.nanmean(data),
                'std': np.nanstd(data),
                'min': np.nanmin(data),
                'max': np.nanmax(data)
            })
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Table
    ax1.axis('off')
    table_data = [[s['name'], f"{s['mean']:.2f}", f"{s['std']:.2f}"] for s in stats]
    table = ax1.table(cellText=table_data, 
                     colLabels=['Model', 'Mean', 'Std Dev'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    ax1.set_title('Model Statistics', fontsize=14, fontweight='bold', pad=20)
    
    # Bar chart
    means = [s['mean'] for s in stats]
    stds = [s['std'] for s in stats]
    
    bars = ax2.bar(range(len(names)), means, yerr=stds, capsize=5)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45)
    ax2.set_ylabel('Mean Priority Score')
    ax2.set_title('Mean Scores by Model')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{mean:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save
    output_path = 'plots/fis_statistics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Statistics saved: {output_path}")
    
    plt.close()
    return output_path

if __name__ == "__main__":
    print("Creating FIS model plots...")
    
    # Create main plot
    main_plot = plot_all_models()
    
    # Create statistics
    stats_plot = plot_statistics()
    
    print(f"\n✅ All plots created successfully!")
    print(f"Main plot: {main_plot}")
    print(f"Statistics: {stats_plot}") 