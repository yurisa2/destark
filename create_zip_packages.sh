#!/bin/bash

# Create a directory for the zip files
mkdir -p zip_packages

# List of packages to zip (excluding .dist-info directories)
packages=("affine" "certifi" "click" "click_plugins" "cligj" "pyparsing" "rasterio" "rasterio.libs" "zipp")

# Create individual zip files for each package
for package in "${packages[@]}"; do
    if [ -d "python_from_container/$package" ]; then
        echo "Creating zip for $package..."
        cd python_from_container
        zip -r "../zip_packages/${package}.zip" "$package"
        cd ..
    fi
done

# Also create zip files for the .dist-info directories
dist_info_dirs=("affine-2.4.0.dist-info" "certifi-2025.7.14.dist-info" "click-8.1.8.dist-info" "click_plugins-1.1.1.2.dist-info" "cligj-0.7.2.dist-info" "pyparsing-3.2.3.dist-info" "rasterio-1.4.3.dist-info" "zipp-3.23.0.dist-info")

for dist_info in "${dist_info_dirs[@]}"; do
    if [ -d "python_from_container/$dist_info" ]; then
        echo "Creating zip for $dist_info..."
        cd python_from_container
        zip -r "../zip_packages/${dist_info}.zip" "$dist_info"
        cd ..
    fi
done

echo "All zip files created in zip_packages/ directory"
ls -la zip_packages/ 