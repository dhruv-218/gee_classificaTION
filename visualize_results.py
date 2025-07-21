import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches

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

if __name__ == "__main__":
    # Example usage
    
    # Check if files exist
    base_map = input("Enter path to base map image: ")
    classification = input("Enter path to classification image: ")
    output = input("Enter output file path (or press Enter to skip saving): ")
    
    alpha_value = 0.5
    try:
        alpha_input = input(f"Enter transparency value (0-1, default is {alpha_value}): ")
        if alpha_input:
            alpha_value = float(alpha_input)
            if not 0 <= alpha_value <= 1:
                print("Alpha must be between 0 and 1. Using default value 0.5.")
                alpha_value = 0.5
    except ValueError:
        print("Invalid input for alpha. Using default value 0.5.")
        alpha_value = 0.5
    
    if not output:
        output = None
    
    overlay_classification(base_map, classification, output, alpha_value)
    
    print("\nTip: To export your classification image from GEE:")
    print("1. In your GEE script, add: 'Map.addLayer(classified.visualize(class_vis), {}, 'Classification for Export')'")
    print("2. Use the 'Export' button in the map to download the visible layer as PNG")
```
