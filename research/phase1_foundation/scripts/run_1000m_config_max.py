#!/usr/bin/env python3
"""
Run FIS Model with config_max on 1000m Data
This script runs the config_max configuration on 1000m resolution data
for comparison with existing results.
"""

import os
import sys
import multiprocessing as mp
import time
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from raster_fuzzy_lib import UnifiedRasterFuzzyInferenceSystem


def get_optimal_core_count():
    """Get optimal number of cores (all available minus one)."""
    total_cores = mp.cpu_count()
    optimal_cores = max(1, total_cores - 1)
    return optimal_cores, total_cores


def run_1000m_config_max():
    """Run the config_max FIS model on 1000m data."""
    print(f"\n{'='*80}")
    print(f"PROCESSING 1000m DATA WITH CONFIG_MAX")
    print(f"{'='*80}")
    
    # Get optimal core count
    optimal_cores, total_cores = get_optimal_core_count()
    
    print(f"Total CPU cores available: {total_cores}")
    print(f"Using cores: {optimal_cores} (all minus one)")
    print(f"Configuration: config_max.json")
    print(f"Resolution: 1000m")
    
    # Define file paths
    base_dir = Path(__file__).parent
    config_path = base_dir / "app" / "config" / "config_max.json"
    input_dir = base_dir / "app" / "files" / "input" / "base"
    output_dir = base_dir / "app" / "files" / "output"
    
    # Input files (1000m resolution)
    social_tiff = input_dir / "socioeconomico_1000m.tif"
    environmental_tiff = input_dir / "ambiental_1000m.tif"
    strategic_tiff = input_dir / "estratégico_1000m.tif"
    
    # Output file
    output_tiff = output_dir / "output_1000m_config_max.tif"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify input files exist
    input_files = [social_tiff, environmental_tiff, strategic_tiff]
    for file_path in input_files:
        if not file_path.exists():
            print(f"❌ Error: Input file not found: {file_path}")
            return False
    
    # Verify config file exists
    if not config_path.exists():
        print(f"❌ Error: Configuration file not found: {config_path}")
        return False
    
    print(f"📁 Output: {output_tiff}")
    
    # Check if output already exists
    if output_tiff.exists():
        print(f"⚠️  Output file already exists: {output_tiff}")
        response = input("Do you want to overwrite it? (y/N): ")
        if response.lower() != 'y':
            print("Processing cancelled.")
            return False
    
    try:
        # Initialize the FIS system
        print("🔄 Initializing FIS system...")
        start_time = time.time()
        
        fis = UnifiedRasterFuzzyInferenceSystem(str(config_path))
        
        init_time = time.time() - start_time
        print(f"✅ FIS system initialized in {init_time:.2f} seconds")
        
        # Process the rasters
        print("🔄 Processing rasters...")
        processing_start = time.time()
        
        fis.process_rasters(
            social_tiff=str(social_tiff),
            environmental_tiff=str(environmental_tiff),
            strategic_tiff=str(strategic_tiff),
            output_tiff=str(output_tiff),
            nodata_value=5.0,
            parallel=True,
            num_cores=optimal_cores,
            chunk_size=100
        )
        
        processing_time = time.time() - processing_start
        total_time = time.time() - start_time
        
        # Check if output file was created
        if output_tiff.exists():
            output_size_mb = output_tiff.stat().st_size / (1024 * 1024)
            print(f"\n✅ 1000m CONFIG_MAX COMPLETED SUCCESSFULLY!")
            print(f"   Processing time: {processing_time:.2f} seconds")
            print(f"   Total time: {total_time:.2f} seconds")
            print(f"   Output file size: {output_size_mb:.1f} MB")
            print(f"   Output location: {output_tiff}")
            
            # Print comparison info
            print(f"\n📊 COMPARISON READY:")
            print(f"   New file: {output_tiff}")
            print(f"   Existing 1000m: app/files/output/1000m_output_tifffile.tif")
            print(f"   Existing 300m: app/files/output/output_300m_config_minimum.tif")
            print(f"\n   Run comparison with: python utils/compare_1000m_300m_results.py")
            
            return True
        else:
            print(f"❌ Error: Output file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run 1000m config_max processing."""
    print("=" * 80)
    print("RUNNING FIS MODEL: CONFIG_MAX ON 1000m DATA")
    print("=" * 80)
    
    # Check if config_max exists
    config_path = Path(__file__).parent / "app" / "config" / "config_max.json"
    if not config_path.exists():
        print(f"❌ Error: config_max.json not found at {config_path}")
        print("Available configs:")
        config_dir = Path(__file__).parent / "app" / "config"
        for config_file in config_dir.glob("config_*.json"):
            print(f"   - {config_file.name}")
        return
    
    # Run the processing
    if run_1000m_config_max():
        print(f"\n🎉 SUCCESS! 1000m config_max processing completed.")
        print(f"   You can now compare the results using the comparison script.")
    else:
        print(f"\n❌ FAILED! 1000m config_max processing failed.")


if __name__ == "__main__":
    main() 