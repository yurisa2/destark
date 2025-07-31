#!/usr/bin/env python3
"""
AWS Glue Job for FIS Processing with Tifffile
This job processes raster data using the Fuzzy Inference System with tifffile library.
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
import boto3
import numpy as np
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(__file__))

# Import the tifffile FIS implementation
from raster_fuzzy_lib_tifffile import UnifiedRasterFuzzyInferenceSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GlueFISProcessor:
    """AWS Glue FIS Processor using tifffile implementation."""
    
    def __init__(self, s3_bucket, s3_prefix, config_name="config_max"):
        """
        Initialize the Glue FIS Processor.
        
        Args:
            s3_bucket (str): S3 bucket name
            s3_prefix (str): S3 prefix for input/output files
            config_name (str): Configuration file name (without .json extension)
        """
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.config_name = config_name
        self.s3_client = boto3.client('s3')
        self.local_temp_dir = "/tmp/glue_fis"
        
        # Create local temp directory
        os.makedirs(self.local_temp_dir, exist_ok=True)
        
        logger.info(f"Initialized Glue FIS Processor")
        logger.info(f"S3 Bucket: {s3_bucket}")
        logger.info(f"S3 Prefix: {s3_prefix}")
        logger.info(f"Config: {config_name}")
    
    def download_from_s3(self, s3_key, local_path):
        """
        Download file from S3 to local path.
        
        Args:
            s3_key (str): S3 object key
            local_path (str): Local file path
        """
        try:
            logger.info(f"Downloading s3://{self.s3_bucket}/{s3_key} to {local_path}")
            self.s3_client.download_file(self.s3_bucket, s3_key, local_path)
            
            if os.path.exists(local_path):
                file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
                logger.info(f"✅ Downloaded successfully ({file_size_mb:.1f} MB)")
                return True
            else:
                logger.error(f"❌ File was not created: {local_path}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to download s3://{self.s3_bucket}/{s3_key}: {e}")
            return False
    
    def upload_to_s3(self, local_path, s3_key):
        """
        Upload file from local path to S3.
        
        Args:
            local_path (str): Local file path
            s3_key (str): S3 object key
        """
        try:
            if not os.path.exists(local_path):
                logger.error(f"❌ Local file does not exist: {local_path}")
                return False
            
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            logger.info(f"Uploading {local_path} ({file_size_mb:.1f} MB) to s3://{self.s3_bucket}/{s3_key}")
            
            self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
            logger.info(f"✅ Uploaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload {local_path}: {e}")
            return False
    
    def get_optimal_core_count(self):
        """Get optimal number of cores for parallel processing."""
        import multiprocessing as mp
        total_cores = mp.cpu_count()
        # Use 75% of available cores for Glue
        optimal_cores = max(1, int(total_cores * 0.75))
        logger.info(f"CPU cores: {total_cores} total, {optimal_cores} optimal")
        return optimal_cores
    
    def process_fis_model(self, resolution="1000m"):
        """
        Process FIS model for a specific resolution.
        
        Args:
            resolution (str): Resolution (e.g., "1000m", "300m")
        """
        logger.info(f"Processing FIS model for {resolution} resolution")
        
        # Define file paths
        config_s3_key = f"{self.s3_prefix}/config/{self.config_name}.json"
        social_s3_key = f"{self.s3_prefix}/input/{resolution}/socioeconomico_{resolution}.tif"
        environmental_s3_key = f"{self.s3_prefix}/input/{resolution}/ambiental_{resolution}.tif"
        strategic_s3_key = f"{self.s3_prefix}/input/{resolution}/estrategico_{resolution}.tif"
        output_s3_key = f"{self.s3_prefix}/output/output_{resolution}_{self.config_name}_glue.tif"
        
        # Local file paths
        config_local = os.path.join(self.local_temp_dir, f"{self.config_name}.json")
        social_local = os.path.join(self.local_temp_dir, f"social_{resolution}.tif")
        environmental_local = os.path.join(self.local_temp_dir, f"environmental_{resolution}.tif")
        strategic_local = os.path.join(self.local_temp_dir, f"strategic_{resolution}.tif")
        output_local = os.path.join(self.local_temp_dir, f"output_{resolution}_{self.config_name}.tif")
        
        # Download input files
        logger.info("Downloading input files from S3...")
        downloads_ok = True
        downloads_ok &= self.download_from_s3(config_s3_key, config_local)
        downloads_ok &= self.download_from_s3(social_s3_key, social_local)
        downloads_ok &= self.download_from_s3(environmental_s3_key, environmental_local)
        downloads_ok &= self.download_from_s3(strategic_s3_key, strategic_local)
        
        if not downloads_ok:
            logger.error("❌ Some input files failed to download")
            return False
        
        # Verify all files exist
        input_files = [config_local, social_local, environmental_local, strategic_local]
        for file_path in input_files:
            if not os.path.exists(file_path):
                logger.error(f"❌ Input file missing: {file_path}")
                return False
        
        # Get optimal core count
        optimal_cores = self.get_optimal_core_count()
        
        try:
            # Initialize FIS system
            logger.info("Initializing FIS system...")
            start_time = time.time()
            
            fis = UnifiedRasterFuzzyInferenceSystem(config_local)
            
            init_time = time.time() - start_time
            logger.info(f"✅ FIS system initialized in {init_time:.2f} seconds")
            
            # Process rasters
            logger.info("Processing rasters with FIS...")
            processing_start = time.time()
            
            fis.process_rasters(
                social_tiff=social_local,
                environmental_tiff=environmental_local,
                strategic_tiff=strategic_local,
                output_tiff=output_local,
                nodata_value=5.0,
                parallel=True,
                num_cores=optimal_cores,
                chunk_size=100
            )
            
            processing_time = time.time() - processing_start
            total_time = time.time() - start_time
            
            # Check if output was created
            if os.path.exists(output_local):
                output_size_mb = os.path.getsize(output_local) / (1024 * 1024)
                logger.info(f"✅ FIS processing completed successfully!")
                logger.info(f"   Processing time: {processing_time:.2f} seconds")
                logger.info(f"   Total time: {total_time:.2f} seconds")
                logger.info(f"   Output file size: {output_size_mb:.1f} MB")
                
                # Upload result to S3
                logger.info("Uploading result to S3...")
                if self.upload_to_s3(output_local, output_s3_key):
                    logger.info(f"✅ Result uploaded to s3://{self.s3_bucket}/{output_s3_key}")
                    return True
                else:
                    logger.error("❌ Failed to upload result to S3")
                    return False
            else:
                logger.error("❌ FIS processing failed - output file not created")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error during FIS processing: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            import shutil
            if os.path.exists(self.local_temp_dir):
                shutil.rmtree(self.local_temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.local_temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary files: {e}")


def main():
    """Main function for the Glue job."""
    
    # Get job parameters from Glue context
    try:
        from awsglue.context import GlueContext
        from pyspark.context import SparkContext
        
        # Initialize Spark and Glue contexts
        sc = SparkContext()
        glueContext = GlueContext(sc)
        spark = glueContext.spark_session
        
        logger.info("✅ Spark and Glue contexts initialized")
        
    except ImportError:
        logger.warning("Glue context not available - running in standalone mode")
        spark = None
    
    # Get job parameters
    # These can be set via job parameters in the Glue console
    s3_bucket = os.environ.get('S3_BUCKET', '<AWS-BUCKET>-unifile')
    s3_prefix = os.environ.get('S3_PREFIX', 'unifile_test')
    config_name = os.environ.get('CONFIG_NAME', 'config_max')
    resolution = os.environ.get('RESOLUTION', '1000m')
    
    logger.info("Job Parameters:")
    logger.info(f"  S3_BUCKET: {s3_bucket}")
    logger.info(f"  S3_PREFIX: {s3_prefix}")
    logger.info(f"  CONFIG_NAME: {config_name}")
    logger.info(f"  RESOLUTION: {resolution}")
    
    # Initialize processor
    processor = GlueFISProcessor(s3_bucket, s3_prefix, config_name)
    
    try:
        # Process FIS model
        success = processor.process_fis_model(resolution)
        
        if success:
            logger.info("🎉 Glue FIS job completed successfully!")
            
            # Log output location
            output_s3_key = f"{s3_prefix}/output/output_{resolution}_{config_name}_glue.tif"
            logger.info(f"📁 Output available at: s3://{s3_bucket}/{output_s3_key}")
            
        else:
            logger.error("❌ Glue FIS job failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Unexpected error in Glue job: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Cleanup
        processor.cleanup()
        
        # Stop Spark context if available
        if spark:
            spark.stop()


if __name__ == "__main__":
    main() 