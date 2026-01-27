"""
Inference script for PatchCore anomaly detection
"""

import os
import argparse
from tqdm import tqdm
import cv2

from patchcore import PatchCore
from patchcore.utils import load_image, load_dataset_images, visualize_result, load_ground_truth_mask
import config


def inference_single_image(model: PatchCore, image_path: str, output_dir: str = None):
    """
    Run inference on a single image
    
    Args:
        model: Trained PatchCore model
        image_path: Path to input image
        output_dir: Directory to save results
    
    Returns:
        Dictionary with results
    """
    # Load image
    image_tensor, original_image = load_image(image_path, size=config.IMAGE_SIZE)
    
    # Run prediction
    anomaly_score, anomaly_map, patch_scores = model.predict(image_tensor, return_heatmap=True)
    
    # Check if anomalous
    is_anomaly = model.is_anomaly(anomaly_score)
    
    # Print results
    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Anomaly Score: {anomaly_score:.4f}")
    print(f"Threshold: {model.threshold:.4f}")
    print(f"Status: {'ANOMALOUS' if is_anomaly else 'NORMAL'}")
    
    # Save visualization if output directory provided
    if output_dir and anomaly_map is not None:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save visualization
        vis_path = os.path.join(output_dir, f"{base_name}_result.png")
        visualize_result(
            original_image,
            anomaly_map,
            anomaly_score,
            threshold=model.threshold,
            save_path=vis_path
        )
        
        # Save heatmap separately
        heatmap_path = os.path.join(output_dir, f"{base_name}_heatmap.png")
        heatmap_colored = cv2.applyColorMap((anomaly_map * 255).astype('uint8'), cv2.COLORMAP_JET)
        cv2.imwrite(heatmap_path, heatmap_colored)
    
    return {
        'image_path': image_path,
        'anomaly_score': anomaly_score,
        'is_anomaly': is_anomaly,
        'anomaly_map': anomaly_map,
        'threshold': model.threshold
    }


def inference_category(category: str, save_results: bool = True):
    """
    Run inference on all test images for a category
    
    Args:
        category: Category name
        save_results: Whether to save visualizations
    
    Returns:
        List of results
    """
    print(f"\n{'='*60}")
    print(f"Running inference for category: {category}")
    print(f"{'='*60}\n")
    
    # Load model
    model_path = os.path.join(config.MODELS_ROOT, category)
    if not os.path.exists(model_path):
        raise ValueError(f"Model not found: {model_path}. Please train the model first.")
    
    print(f"Loading model from {model_path}...")
    model = PatchCore(
        backbone=config.BACKBONE,
        layers=config.LAYERS,
        device=config.DEVICE,
        sampling_ratio=config.CORESET_SAMPLING_RATIO,
        n_neighbors=config.NEIGHBORS
    )
    model.load(model_path)
    
    # Load test images
    print("\nLoading test images...")
    test_data = load_dataset_images(config.DATA_ROOT, category, 'test')
    print(f"Found {len(test_data)} test images")
    
    # Prepare output directory
    output_dir = None
    if save_results:
        output_dir = os.path.join(config.RESULTS_ROOT, category)
        os.makedirs(output_dir, exist_ok=True)
    
    # Run inference
    results = []
    print("\nRunning inference...")
    
    for image_path, defect_type in tqdm(test_data, desc="Processing images"):
        # Load image
        image_tensor, original_image = load_image(image_path, size=config.IMAGE_SIZE)
        
        # Run prediction
        anomaly_score, anomaly_map, _ = model.predict(image_tensor, return_heatmap=True)
        
        # Check if anomalous
        is_anomaly = model.is_anomaly(anomaly_score)
        
        # Get ground truth
        image_name = os.path.basename(image_path)
        gt_mask = load_ground_truth_mask(config.DATA_ROOT, category, defect_type, image_name)
        
        # Save visualization if requested
        if save_results and anomaly_map is not None:
            defect_output_dir = os.path.join(output_dir, defect_type)
            os.makedirs(defect_output_dir, exist_ok=True)
            
            base_name = os.path.splitext(image_name)[0]
            vis_path = os.path.join(defect_output_dir, f"{base_name}_result.png")
            
            visualize_result(
                original_image,
                anomaly_map,
                anomaly_score,
                threshold=model.threshold,
                save_path=vis_path,
                ground_truth=gt_mask
            )
        
        # Store results
        results.append({
            'image_path': image_path,
            'defect_type': defect_type,
            'anomaly_score': anomaly_score,
            'is_anomaly': is_anomaly,
            'ground_truth_label': defect_type != 'good',
            'anomaly_map': anomaly_map,
            'gt_mask': gt_mask
        })
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Inference complete for category: {category}")
    print(f"Processed {len(results)} images")
    if save_results:
        print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference with PatchCore")
    parser.add_argument(
        '--category',
        type=str,
        default='bottle',
        help='Category to run inference on (default: bottle)'
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path to single image (optional)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save visualizations'
    )
    
    args = parser.parse_args()
    
    # Load model
    model_path = os.path.join(config.MODELS_ROOT, args.category)
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print(f"Please train the model first using: python train.py --category {args.category}")
        return
    
    print(f"Loading model from {model_path}...")
    model = PatchCore(
        backbone=config.BACKBONE,
        layers=config.LAYERS,
        device=config.DEVICE,
        sampling_ratio=config.CORESET_SAMPLING_RATIO,
        n_neighbors=config.NEIGHBORS
    )
    model.load(model_path)
    
    if args.image:
        # Single image inference
        output_dir = args.output or os.path.join(config.RESULTS_ROOT, args.category, 'single')
        inference_single_image(model, args.image, output_dir if not args.no_save else None)
    else:
        # Full category inference
        inference_category(args.category, save_results=not args.no_save)


if __name__ == '__main__':
    main()
