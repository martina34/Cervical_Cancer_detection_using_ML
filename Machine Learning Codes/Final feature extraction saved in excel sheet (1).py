#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import csv
import numpy as np
from skimage import io, color, morphology, measure

def extract_features(image_path):
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist.")
        return None
    
    # Load the image
    image = io.imread(image_path)
    # Convert to grayscale
    image_gray = color.rgb2gray(image)
    # Apply morphological opening
    image_open = morphology.opening(image_gray, morphology.disk(5))
    # Threshold the image
    image_threshold = image_open < 0.5
    # Label the image regions
    labeled_image = measure.label(image_threshold)
    # Extract the region properties
    region_properties = measure.regionprops(labeled_image, intensity_image=image_gray)
    # Return the region properties
    return region_properties

# Define the directory containing the image files
image_dir = 'D:/Martina/Faculty/Graduation Project/Datasets/Cervical cancer pap smears/smear2005/New database pictures'

# Iterate over all image files in the directory
with open('features.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write the feature names to the CSV file
    writer.writerow(['ID', 'Nucleus Area', 'Cytoplasm Area', 'N/C Ratio', 'Nucleus Brightness', 'Cytoplasm Brightness', 'Nucleus Longest Parameter', 'Cytoplasm Longest Diameter', 'Nucleus Shortest Parameter', 'Cytoplasm Shortest Diameter', 'Nucleus Roundness', 'Cytoplasm Roundness', 'Nucleus Elongation', 'Cytoplasm Elongation', 'Nucleus Perimeter', 'Cytoplasm Perimeter', 'Nucleus Position', 'Nucleus Minima', 'Nucleus Maxima', 'Cytoplasm Minima', 'Cytoplasm Maxima'])
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            # Check if the file is a BMP image
            if file.endswith('.BMP'):
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Call the extract_features function on the image file
                region_properties = extract_features(file_path)
                # Print features for each region
                for region in region_properties:
                    nucleus_area = region.area   
                    cytoplasm_area = region.convex_area - nucleus_area
                    nc_ratio = nucleus_area / nucleus_area + cytoplasm_area
                    nucleus_brightness = region.mean_intensity
                    cytoplasm_brightness = (region.intensity_image.sum() - region.mean_intensity * nucleus_area) / cytoplasm_area
                    nucleus_longest_parameter = max(region.major_axis_length, region.minor_axis_length)
                    cytoplasm_longest_diameter = 2 * region.equivalent_diameter - nucleus_longest_parameter
                    nucleus_shortest_parameter = min(region.major_axis_length, region.minor_axis_length)
                    cytoplasm_shortest_diameter = 2 * region.equivalent_diameter - nucleus_shortest_parameter
                    nucleus_roundness = region.perimeter / (2 * (region.major_axis_length + region.minor_axis_length))
                    cytoplasm_roundness = region.perimeter / (2 * region.equivalent_diameter)
                    nucleus_elongation = max(region.major_axis_length, region.minor_axis_length) / (min(region.major_axis_length, region.minor_axis_length)+1)
                    cytoplasm_elongation = region.equivalent_diameter / min(region.major_axis_length, region.minor_axis_length)
                    nucleus_perimeter = region.perimeter
                    cytoplasm_perimeter = region.perimeter + nucleus_perimeter
                    nucleus_position = region.centroid
                    nucleus_minima = region.min_intensity
                    nucleus_maxima = region.max_intensity
                    cytoplasm_minima = region.min_intensity
                    cytoplasm_maxima = region.max_intensity
                    
                    # Write the feature values to the CSV file
                writer.writerow([file_path, nucleus_area, cytoplasm_area, nc_ratio, nucleus_brightness, cytoplasm_brightness, nucleus_longest_parameter, cytoplasm_longest_diameter, nucleus_shortest_parameter, cytoplasm_shortest_diameter, nucleus_roundness, cytoplasm_roundness, nucleus_elongation, cytoplasm_elongation, nucleus_perimeter, cytoplasm_perimeter, nucleus_position, nucleus_minima, nucleus_maxima,cytoplasm_minima, cytoplasm_maxima ])



# In[ ]:




