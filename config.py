"""
Configuration file for PatchCore anomaly detection
"""

# Model Configuration
BACKBONE = "wide_resnet50_2"  # Backbone network
LAYERS = ["layer2", "layer3"]  # Feature extraction layers
DEVICE = "cuda"  # Device: 'cuda' or 'cpu'

# Image Configuration
IMAGE_SIZE = 224  # Input image size
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# PatchCore Configuration
CORESET_SAMPLING_RATIO = 0.1  # Keep 10% of patches
NEIGHBORS = 9  # Number of nearest neighbors for scoring

# Threshold Configuration
THRESHOLD_METHOD = "mean_std"  # 'mean_std' or 'percentile'
THRESHOLD_STD_MULTIPLIER = 3.0  # For mean_std method
THRESHOLD_PERCENTILE = 99.0  # For percentile method

# Paths
DATA_ROOT = "mvtec_anomaly_detection"
MODELS_ROOT = "models/patchcore"
RESULTS_ROOT = "results"

# Categories (all MVTec categories)
CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

# Evaluation Configuration
SAVE_VISUALIZATIONS = True
SAVE_HEATMAPS = True
