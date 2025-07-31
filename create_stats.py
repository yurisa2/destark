#!/usr/bin/env python3
import rasterio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def create_statistics_plot():
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
        print(f"Processing {name}...")
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
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax1.set_title('FIS Model Statistics', fontsize=14, fontweight='bold', pad=20)
    
    # Bar chart
    means = [s['mean'] for s in stats]
    stds = [s['std'] for s in stats]
    
    bars = ax2.bar(range(len(names)), means, yerr=stds, capsize=5, alpha=0.8)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('Mean Priority Score', fontweight='bold')
    ax2.set_title('Mean Priority Scores by Model', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = 'plots/fis_statistics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Statistics plot saved: {output_path}")
    return output_path

if __name__ == "__main__":
    print("Creating statistics comparison...")
    create_statistics_plot()
    print("Done!") 