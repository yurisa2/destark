#!/usr/bin/env python3
import rasterio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from PIL import Image

def plot_individual_model(file_path, name, output_dir):
    """Plot a single model and save it."""
    print(f"Plotting {name}...")
    
    with rasterio.open(file_path) as src:
        data = src.read(1)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Plot with viridis colormap
        im = ax.imshow(data, cmap='viridis')
        ax.set_title(f'{name} Model', fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add statistics
        mean_val = np.nanmean(data)
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        ax.text(0.02, 0.98, f'Mean: {mean_val:.2f}\nRange: {min_val:.1f}-{max_val:.1f}', 
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Priority Score (0-10)', fontsize=10)
        
        # Save individual plot
        output_path = os.path.join(output_dir, f'{name.lower().replace(" ", "_")}.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  Saved: {output_path}")
        return output_path

def create_merged_plot(individual_files, names):
    """Create a merged plot from individual files."""
    print("Creating merged plot...")
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Load and plot each individual image
    for i, (file_path, name) in enumerate(zip(individual_files, names)):
        # Load the saved image
        img = Image.open(file_path)
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(f'{name} Model', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Main title
    fig.suptitle('FIS Model Results Comparison - 300m Resolution', fontsize=16, fontweight='bold', y=0.95)
    fig.text(0.5, 0.92, 'Environmental Priority Assessment using Different Aggregation Methods', 
             fontsize=12, ha='center', style='italic')
    
    # Save merged plot
    output_path = 'plots/fis_results_merged.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Merged plot saved: {output_path}")
    return output_path

def create_low_res_comparison():
    """Create a low-resolution comparison by downsampling."""
    print("Creating low-resolution comparison...")
    
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
    
    # Get global min/max for consistent scaling
    print("Reading files to get global range...")
    all_data = []
    for file_path in files:
        with rasterio.open(file_path) as src:
            data = src.read(1)
            # Downsample to reduce memory usage
            data_downsampled = data[::10, ::10]  # Take every 10th pixel
            all_data.append(data_downsampled)
    
    global_min = min(np.nanmin(data) for data in all_data)
    global_max = max(np.nanmax(data) for data in all_data)
    print(f"Global range: {global_min:.2f} to {global_max:.2f}")
    
    # Plot each downsampled model
    for i, (file_path, name) in enumerate(zip(files, names)):
        print(f"Plotting {name} (downsampled)...")
        
        with rasterio.open(file_path) as src:
            data = src.read(1)
            # Downsample
            data_downsampled = data[::10, ::10]
            
            ax = axes[i]
            im = ax.imshow(data_downsampled, cmap='viridis', vmin=global_min, vmax=global_max)
            ax.set_title(f'{name} Model (Low Res)', fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add statistics
            mean_val = np.nanmean(data_downsampled)
            ax.text(0.02, 0.98, f'Mean: {mean_val:.2f}', 
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes, shrink=0.8, aspect=30)
    cbar.set_label('Priority Score (0-10)', fontsize=12, fontweight='bold')
    
    # Main title
    fig.suptitle('FIS Model Results - Low Resolution Comparison', fontsize=16, fontweight='bold', y=0.95)
    
    # Save
    output_path = 'plots/fis_results_low_res.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Low-res plot saved: {output_path}")
    return output_path

def main():
    """Main function to create all plots."""
    print("=" * 60)
    print("CREATING FIS MODEL PLOTS - INDIVIDUAL APPROACH")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('plots', exist_ok=True)
    os.makedirs('plots/individual', exist_ok=True)
    
    # Files and names
    files = [
        'app/files/output/output_300m_config_max.tif',
        'app/files/output/output_300m_config_median.tif',
        'app/files/output/output_300m_config_minimum.tif',
        'app/files/output/output_300m_config_mode.tif',
        'app/files/output/output_300m_config_round_down.tif',
        'app/files/output/output_300m_config_round_up.tif'
    ]
    
    names = ['Maximum', 'Median', 'Minimum', 'Mode', 'Round Down', 'Round Up']
    
    # Plot each model individually
    individual_files = []
    for file_path, name in zip(files, names):
        output_path = plot_individual_model(file_path, name, 'plots/individual')
        individual_files.append(output_path)
    
    # Create merged plot
    merged_plot = create_merged_plot(individual_files, names)
    
    # Create low-resolution comparison
    low_res_plot = create_low_res_comparison()
    
    print(f"\n{'='*60}")
    print("ALL PLOTS CREATED SUCCESSFULLY!")
    print(f"{'='*60}")
    print("Files created:")
    print("  Individual plots: plots/individual/")
    print("  Merged plot: plots/fis_results_merged.png")
    print("  Low-res comparison: plots/fis_results_low_res.png")

if __name__ == "__main__":
    main() 