#!/usr/bin/env python3
import rasterio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def plot_with_consistent_colors():
    """Plot all models with consistent color scaling."""
    files = [
        'app/files/output/output_300m_config_max.tif',
        'app/files/output/output_300m_config_median.tif',
        'app/files/output/output_300m_config_minimum.tif',
        'app/files/output/output_300m_config_mode.tif',
        'app/files/output/output_300m_config_round_down.tif',
        'app/files/output/output_300m_config_round_up.tif'
    ]
    
    names = ['Maximum', 'Median', 'Minimum', 'Mode', 'Round Down', 'Round Up']
    
    # First, get global min/max for consistent scaling
    print("Reading all files to get global range...")
    all_data = []
    for file_path in files:
        with rasterio.open(file_path) as src:
            all_data.append(src.read(1))
    
    global_min = min(np.nanmin(data) for data in all_data)
    global_max = max(np.nanmax(data) for data in all_data)
    print(f"Global range: {global_min:.2f} to {global_max:.2f}")
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot each file with consistent scaling
    for i, (file_path, name) in enumerate(zip(files, names)):
        print(f"Plotting {name}...")
        
        with rasterio.open(file_path) as src:
            data = src.read(1)
            
            ax = axes[i]
            im = ax.imshow(data, cmap='viridis', vmin=global_min, vmax=global_max)
            ax.set_title(f'{name} Model', fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add statistics
            mean_val = np.nanmean(data)
            min_val = np.nanmin(data)
            max_val = np.nanmax(data)
            ax.text(0.02, 0.98, f'Mean: {mean_val:.2f}\nRange: {min_val:.1f}-{max_val:.1f}', 
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes, shrink=0.8, aspect=30)
    cbar.set_label('Priority Score (0-10)', fontsize=12, fontweight='bold')
    cbar.set_ticks(np.arange(0, 11, 1))
    
    # Main title
    fig.suptitle('FIS Model Results Comparison - 300m Resolution', fontsize=16, fontweight='bold', y=0.95)
    fig.text(0.5, 0.92, 'Environmental Priority Assessment using Different Aggregation Methods', 
             fontsize=12, ha='center', style='italic')
    
    # Save
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/fis_results_consistent_colors.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("✅ Main plot saved: plots/fis_results_consistent_colors.png")
    plt.close()

def plot_statistics():
    """Create statistics comparison plot."""
    files = [
        'app/files/output/output_300m_config_max.tif',
        'app/files/output/output_300m_config_median.tif',
        'app/files/output/output_300m_config_minimum.tif',
        'app/files/output/output_300m_config_mode.tif',
        'app/files/output/output_300m_config_round_down.tif',
        'app/files/output/output_300m_config_round_up.tif'
    ]
    
    names = ['Maximum', 'Median', 'Minimum', 'Mode', 'Round Down', 'Round Up']
    
    # Collect statistics
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
    
    # Statistics table
    ax1.axis('off')
    table_data = []
    for stat in stats:
        table_data.append([
            stat['name'],
            f"{stat['min']:.2f}",
            f"{stat['max']:.2f}",
            f"{stat['mean']:.2f}",
            f"{stat['std']:.2f}"
        ])
    
    table = ax1.table(cellText=table_data, 
                     colLabels=['Model', 'Min', 'Max', 'Mean', 'Std Dev'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # Style table
    for i in range(5):  # 5 columns
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax1.set_title('FIS Model Statistics', fontsize=14, fontweight='bold', pad=20)
    
    # Bar chart with gradient colors
    means = [s['mean'] for s in stats]
    stds = [s['std'] for s in stats]
    
    # Create gradient colors based on mean values
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(means), max(means))
    colors = [cmap(norm(mean)) for mean in means]
    
    bars = ax2.bar(range(len(names)), means, yerr=stds, capsize=5, alpha=0.8, color=colors)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Mean Priority Score', fontweight='bold')
    ax2.set_title('Mean Priority Scores by Model', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    plt.savefig('plots/fis_statistics_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("✅ Statistics plot saved: plots/fis_statistics_comparison.png")
    plt.close()

if __name__ == "__main__":
    print("Creating FIS model plots with consistent colors...")
    
    # Create main plot with consistent colors
    plot_with_consistent_colors()
    
    # Create statistics plot
    plot_statistics()
    
    print("\n✅ All plots created successfully!")
    print("Files created:")
    print("  - plots/fis_results_consistent_colors.png")
    print("  - plots/fis_statistics_comparison.png") 