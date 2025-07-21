import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np

def evaluate_confusion_matrix(confusion_matrix):
    """Analyze a confusion matrix and provide detailed metrics"""
    # Calculate per-class metrics
    n_classes = confusion_matrix.size().getInfo()[0]
    total_accuracy = confusion_matrix.accuracy().getInfo()
    
    # Get the raw matrix values
    matrix_array = confusion_matrix.getInfo()
    
    # Calculate precision, recall, and F1 score for each class
    class_names = ['Water', 'Vegetation', 'Urban']
    metrics = []
    
    total_samples = sum(sum(row) for row in matrix_array)
    print(f"Total samples: {total_samples}, Overall Accuracy: {total_accuracy:.4f}")
    print("\nPer-class metrics:")
    print("-" * 60)
    print(f"{'Class':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'Support':<10}")
    print("-" * 60)
    
    for i in range(n_classes):
        # True positives: diagonal element
        tp = matrix_array[i][i]
        
        # Sum of row (actual class)
        row_sum = sum(matrix_array[i])
        
        # Sum of column (predicted class)
        col_sum = sum(row[i] for row in matrix_array)
        
        # Calculate metrics
        precision = tp / col_sum if col_sum > 0 else 0
        recall = tp / row_sum if row_sum > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{class_names[i]:<10} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f} | {row_sum:<10}")
        metrics.append({'precision': precision, 'recall': recall, 'f1': f1, 'support': row_sum})
    
    print("-" * 60)
    return metrics

def plot_training_areas(Map, water, vegetation, urban, aoi):
    """
    Add training areas to the map with clearer visualization
    and return a figure showing their distribution
    """
    # Create a feature collection of all training areas with class labels
    all_training = water.map(lambda f: f.set('class_name', 'Water')) \
        .merge(vegetation.map(lambda f: f.set('class_name', 'Vegetation'))) \
        .merge(urban.map(lambda f: f.set('class_name', 'Urban')))
    
    # Style each class
    water_vis = {'color': '0000FF', 'fillColor': '0000FF80', 'width': 3}
    veg_vis = {'color': '00FF00', 'fillColor': '00FF0080', 'width': 3}
    urban_vis = {'color': 'FF0000', 'fillColor': 'FF000080', 'width': 3}
    
    # Add to map
    Map.addLayer(water.style(**water_vis), {}, 'Water Training', False)
    Map.addLayer(vegetation.style(**veg_vis), {}, 'Vegetation Training', False) 
    Map.addLayer(urban.style(**urban_vis), {}, 'Urban Training', False)
    
    # Create a training areas folder
    Map.add_layer_manager()
    
    return Map

def suggest_improvements(train_metrics, val_metrics):
    """Analyze metrics and suggest improvements"""
    suggestions = []
    
    # Find classes with low precision or recall in validation
    for i, (train_m, val_m) in enumerate(zip(train_metrics, val_metrics)):
        class_names = ['Water', 'Vegetation', 'Urban']
        class_name = class_names[i]
        
        # Check for overfitting
        if train_m['f1'] - val_m['f1'] > 0.15:
            suggestions.append(f"Class '{class_name}' shows signs of overfitting. Add more diverse training samples.")
        
        # Check for low precision
        if val_m['precision'] < 0.85:
            suggestions.append(f"Low precision for '{class_name}' (other classes being misclassified as {class_name}). "
                              f"Check training areas that might be misrepresenting this class.")
        
        # Check for low recall
        if val_m['recall'] < 0.85:
            suggestions.append(f"Low recall for '{class_name}' (some {class_name} areas being classified as something else). "
                              f"Add more training samples for this class in varied locations.")
    
    # Overall suggestions
    suggestions.append("Consider adding more spectral indices as features (NDWI for water, NDBI for built-up areas)")
    suggestions.append("Try different classifier parameters or algorithms (SVM, CART)")
    
    return suggestions

print("Load this module after running your main classification script to analyze results:")
print("from improve_classification import evaluate_confusion_matrix, plot_training_areas, suggest_improvements")
print("\nExample usage:")
print("train_metrics = evaluate_confusion_matrix(train_conf_matrix)")
print("val_metrics = evaluate_confusion_matrix(val_conf_matrix)")
print("suggestions = suggest_improvements(train_metrics, val_metrics)")
print("for suggestion in suggestions:")
print("    print(f'- {suggestion}')")
print("Map = plot_training_areas(Map, water, vegetation, urban, aoi)")
