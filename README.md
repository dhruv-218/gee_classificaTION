# gee_classificaTION

A project focused on **Land Cover Classification using Google Earth Engine (GEE) with Random Forest Machine Learning**.

## Overview

**gee_classificaTION** is designed to perform supervised land cover classification by leveraging satellite data, machine learning, and the Google Earth Engine cloud platform. The project utilizes a Random Forest classifier to generate a land cover map for a selected region based on training points and input satellite imagery. Outputs include classified maps, model accuracy assessments, and supporting visualizations, suitable for research, monitoring, and spatial analysis.

## Features

- **Supervised Land Cover Classification** using Google Earth Engine (GEE)
- **Random Forest Model** implementation for robust classification
- Training Data Preparation with custom sample points and polygons
- Interactive Results Visualization: classified maps in HTML and PNG formats
- Accuracy Evaluation: confusion matrix and metrics
- Reusable Model: JSON file containing the trained random forest model for further application and validation
- Documentation and supporting files for replicability

## Repository Structure

| File/Filename                        | Description                                                                          |
|------------------------------------|--------------------------------------------------------------------------------------|
| `final_gee_classification.ipynb`   | Main Jupyter Notebook: details code and explanations for the complete workflow        |
| `gee_random_forest_model_1000.json`| Exported random forest model for classification in GEE                               |
| `datapoints.html`                  | Visualization of training data points used for the model                             |
| `land_cover_classification_map.html`| Interactive map of the classification output                                         |
| `matrix_randomforest.png`          | Confusion matrix plot showing model accuracy and performance                         |
| `polygon_region.html`              | Map detailing the region of interest (ROI) polygons                                  |
| `result_3.png`                    | Example output/classification result visualization                                   |
| `.gitignore`, `.DS_Store`         | Standard project and OS files                                                        |
| `isro_certificate.pdf`             | (If present) Certificate or report related to ISRO/validation                        |

## Workflow

### 1. Data Preparation
- Define a region of interest (ROI), e.g., by drawing polygons.
- Collect **training data**: Use polygons/points labeled with target classes (e.g., water, urban, vegetation).
- Visualize these with `datapoints.html`.

### 2. Satellite Data Selection
- Download or access cloud-based satellite imagery (e.g., Landsat, Sentinel-2) via GEE.

### 3. Model Training and Classification
- Use the `final_gee_classification.ipynb` notebook.
- The Random Forest model is trained on the labeled samples and applied to the input image.
- Results are exported as:
  - Interactive map (`land_cover_classification_map.html`)
  - Static result image (`result_3.png`)
  - Confusion matrix plot (`matrix_randomforest.png`) for accuracy assessment

### 4. Model Export/Reuse
- The trained model (`gee_random_forest_model_1000.json`) can be reloaded and applied to new imagery or used for validation.

## Usage

### Requirements

- Google Account with access to Google Earth Engine
- Python 3 (recommended environment: Jupyter Notebook)
- Libraries: geemap, earthengine-api, numpy, pandas, matplotlib

### Installation and Setup

Clone the repository:

