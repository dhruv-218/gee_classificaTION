import ee
import geemap
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import display

ee.Authenticate()  # Only needed once
ee.Initialize(project='classification-459108')

# Define Area of Interest (AOI)
aoi = ee.Geometry.Rectangle([83.26, 17.69, 83.29, 17.71])

# Create interactive map and center on AOI
Map = geemap.Map()
Map.centerObject(aoi, 12)

# Sentinel-2 SR collection with cloud filtering
collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
    .filterDate('2020-01-01', '2020-01-30') \
    .filterBounds(aoi) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

# Cloud masking
def maskS2clouds(image):
    qa = image.select('QA60')
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
    return image.updateMask(mask).divide(10000)

composite = collection.map(maskS2clouds).median()

# Clip the image to our AOI
composite = composite.clip(aoi)

# Calculate NDVI and add it as a band
ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI')
composite = composite.addBands(ndvi)

# Select relevant bands including NDVI
bands = ['B2', 'B3', 'B4', 'B8', 'NDVI']

# Sample training data using polygons instead of points
water = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Polygon([
        [[83.2735, 17.6975], [83.2735, 17.6985], [83.2745, 17.6985], [83.2745, 17.6975], [83.2735, 17.6975]]
    ]), {'landcover': 0}),
])

vegetation = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Polygon([
        [[83.275, 17.702], [83.275, 17.703], [83.276, 17.703], [83.276, 17.702], [83.275, 17.702]]
    ]), {'landcover': 1}),
])

urban = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Polygon([
        [[83.278, 17.700], [83.278, 17.701], [83.279, 17.701], [83.279, 17.700], [83.278, 17.700]]
    ]), {'landcover': 2}),
])

# Merge all classes
training_data = water.merge(vegetation).merge(urban)

# Sample the composite image using training polygons
training = composite.select(bands).sampleRegions(
    collection=training_data,
    properties=['landcover'],
    scale=10
)

# Add a random column for splitting data
training = training.randomColumn('random')

# Split into training (70%) and validation (30%) sets
train_set = training.filter(ee.Filter.lt('random', 0.7))
val_set = training.filter(ee.Filter.gte('random', 0.7))

# Train the classifier
classifier = ee.Classifier.smileRandomForest(100).train(
    features=train_set,
    classProperty='landcover',
    inputProperties=bands
)

# Evaluate on training data (for comparison)
classified_training = train_set.classify(classifier)
train_conf_matrix = classified_training.errorMatrix('landcover', 'classification')
print('Confusion Matrix (on training data):')
print(train_conf_matrix.getInfo())
print('Training Accuracy:', train_conf_matrix.accuracy().getInfo())

# Evaluate on validation data (more reliable)
val_classified = val_set.classify(classifier)
val_conf_matrix = val_classified.errorMatrix('landcover', 'classification')
print('Validation Confusion Matrix:')
print(val_conf_matrix.getInfo())
print('Validation Accuracy:', val_conf_matrix.accuracy().getInfo())

# Classify the image
classified = composite.select(bands).classify(classifier)

# Visualization parameters for classified map
class_vis = {
    'min': 0,
    'max': 2,
    'palette': ['0000FF', '00FF00', 'FF0000']  # water, vegetation, urban
}

# Add layers to map
Map.addLayer(composite, {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3}, 'RGB Composite')
Map.addLayer(classified, class_vis, 'Land Cover Classification')
Map.addLayer(aoi, {}, 'Area of Interest', False)
Map.addLayerControl()

# Add visualization for export - this makes exporting the classification easier
# Fix: classified is a single-band image, so we need to specify bands correctly
visualization_params = {
    'min': 0,
    'max': 2,
    'palette': ['0000FF', '00FF00', 'FF0000']  # water, vegetation, urban
}
Map.addLayer(classified.visualize(**visualization_params), {}, 'Classification for Export')

# Function to overlay classification results on base map
def overlay_classification(base_map_path, classification_path, output_path=None, alpha=0.5):
    """
    Overlay classification results on a base map with transparency
    
    Parameters:
    ----------
    base_map_path : str
        Path to the base map image (e.g., satellite imagery, OpenStreetMap screenshot)
    classification_path : str
        Path to the classification result image
    output_path : str, optional
        Path to save the blended result. If None, will only display
    alpha : float, optional
        Transparency level for the classification overlay (0-1)
    """
    # Load the base map and classified overlay
    base_map = cv2.imread(base_map_path)
    overlay = cv2.imread(classification_path)
    
    if base_map is None or overlay is None:
        raise ValueError("Could not load one or both of the input images")
    
    # Resize if necessary to match dimensions
    overlay = cv2.resize(overlay, (base_map.shape[1], base_map.shape[0]))
    
    # Convert BGR to RGB for plotting
    base_map_rgb = cv2.cvtColor(base_map, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    # Blend overlay with base map
    blended = cv2.addWeighted(overlay_rgb, alpha, base_map_rgb, 1 - alpha, 0)
    
    # Create figure for visualization
    plt.figure(figsize=(12, 10))
    plt.imshow(blended)
    plt.axis('off')
    
    # Add legend
    legend_patches = [
        mpatches.Patch(color='blue', label='Water'),
        mpatches.Patch(color='green', label='Vegetation'),
        mpatches.Patch(color='red', label='Urban')
    ]
    plt.legend(handles=legend_patches, loc='lower right', fontsize='large')
    plt.title("Land Cover Classification Overlay")
    
    # Save result if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Blended visualization saved to: {output_path}")
    
    plt.show()

# Export instructions
print("\nTo create a blended visualization:")
print("1. Click on the 'Layers' button in the map")
print("2. Toggle visibility to show only 'RGB Composite' and 'Classification for Export'")
print("3. Use the 'Camera' icon to take screenshots of each layer separately")
print("4. Run the following code to blend the layers (modify paths as needed):")
print("\nbase_map_path = '/path/to/rgb_composite_screenshot.png'")
print("classification_path = '/path/to/classification_screenshot.png'")
print("output_path = '/path/to/save/blended_result.png'  # Optional")
print("overlay_classification(base_map_path, classification_path, output_path, alpha=0.5)")

# Add visualization of training areas to help identify potential issues
print("\nNOTE: Your validation accuracy is 95.6%, but there's confusion between urban and water classes.")
print("To improve classification, visualize and check your training areas:")

# Style the training areas for visualization
water_style = {'color': '0000FF', 'fillColor': '0000FF80'}  # Blue with transparency
vegetation_style = {'color': '00FF00', 'fillColor': '00FF0080'}  # Green with transparency
urban_style = {'color': 'FF0000', 'fillColor': 'FF000080'}  # Red with transparency

# Add training areas to map for verification
Map.addLayer(water.style(**water_style), {}, 'Water Training Areas')
Map.addLayer(vegetation.style(**vegetation_style), {}, 'Vegetation Training Areas')
Map.addLayer(urban.style(**urban_style), {}, 'Urban Training Areas')

print("\nClassification Quality Tips:")
print("1. Check if training areas accurately represent each class (now visible on map)")
print("2. Consider adding more training samples in areas of confusion")
print("3. Industrial areas near water might be misclassified - add specific training samples there")
print("4. Try adjusting the class colors in visualization if they don't match expectations:")
print("   class_vis = {'min': 0, 'max': 2, 'palette': ['YOUR_WATER_COLOR', 'YOUR_VEG_COLOR', 'YOUR_URBAN_COLOR']}")

# Display the map
Map
