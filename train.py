"""
Training script for PatchCore anomaly detection
"""

import os
import argparse
from tqdm import tqdm

from patchcore import PatchCore
from patchcore.utils import load_image, load_dataset_images
import config


def train_category(category: str, device: str = None):
    """
    Train PatchCore model for a specific category
    
    Args:
        category: Category name (e.g., 'bottle')
        device: Device to use ('cuda' or 'cpu')
    """
    print(f"\n{'='*60}")
    print(f"Training PatchCore for category: {category}")
    print(f"{'='*60}\n")
    
    # Use config device if not specified
    if device is None:
        device = config.DEVICE
    
    # Check if category exists
    category_path = os.path.join(config.DATA_ROOT, category)
    if not os.path.exists(category_path):
        raise ValueError(f"Category not found: {category}")
    
    # Load training images
    print("Loading training images...")
    train_image_paths = load_dataset_images(config.DATA_ROOT, category, 'train')
    print(f"Found {len(train_image_paths)} training images")
    
    # Load and preprocess images
    print("\nPreprocessing images...")
    train_images = []
    for img_path in tqdm(train_image_paths, desc="Loading images"):
        img_tensor, _ = load_image(img_path, size=config.IMAGE_SIZE)
        train_images.append(img_tensor)
    
    # Initialize PatchCore model
    print("\nInitializing PatchCore model...")
    model = PatchCore(
        backbone=config.BACKBONE,
        layers=config.LAYERS,
        device=device,
        sampling_ratio=config.CORESET_SAMPLING_RATIO,
        n_neighbors=config.NEIGHBORS
    )
    
    # Train model (build memory bank)
    model.fit(train_images)
    
    # Compute threshold from training data
    print("\nComputing anomaly threshold...")
    # Use subset of training images for threshold computation
    threshold_images = train_images[:min(20, len(train_images))]
    threshold = model.compute_threshold(
        threshold_images,
        method=config.THRESHOLD_METHOD,
        std_multiplier=config.THRESHOLD_STD_MULTIPLIER,
        percentile=config.THRESHOLD_PERCENTILE
    )
    
    # Save model
    save_path = os.path.join(config.MODELS_ROOT, category)
    model.save(save_path)
    
    print(f"\n{'='*60}")
    print(f"Training complete for category: {category}")
    print(f"Model saved to: {save_path}")
    print(f"Threshold: {threshold:.4f}")
    print(f"{'='*60}\n")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train PatchCore for anomaly detection")
    parser.add_argument(
        '--category',
        type=str,
        default='bottle',
        help='Category to train (default: bottle)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use (default: from config)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Train on all categories'
    )
    
    args = parser.parse_args()
    
    if args.all:
        # Train on all categories
        print("\nTraining on all categories...")
        for category in config.CATEGORIES:
            try:
                train_category(category, args.device)
            except Exception as e:
                print(f"Error training {category}: {e}")
                continue
    else:
        # Train on single category
        train_category(args.category, args.device)


if __name__ == '__main__':
    main()
