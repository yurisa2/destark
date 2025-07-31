#!/usr/bin/env python3
import rasterio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Simple plotting function
def plot_simple():
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
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot each file
    for i, (file_path, name) in enumerate(zip(files, names)):
        print(f"Processing {name}...")
        
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1)
                
                ax = axes[i]
                im = ax.imshow(data, cmap='viridis')
                ax.set_title(name)
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Add mean value
                mean_val = np.nanmean(data)
                ax.text(0.02, 0.98, f'Mean: {mean_val:.2f}', 
                       transform=ax.transAxes, fontsize=8,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
        except Exception as e:
            print(f"Error processing {name}: {e}")
            axes[i].text(0.5, 0.5, f'Error: {name}', ha='center', va='center')
    
    # Add colorbar
    plt.colorbar(im, ax=axes, shrink=0.8)
    
    # Save
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/simple_fis_results.png', dpi=100, bbox_inches='tight')
    print("✅ Plot saved: plots/simple_fis_results.png")
    plt.close()

if __name__ == "__main__":
    print("Creating simple plot...")
    plot_simple()
    print("Done!") 