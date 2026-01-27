"""
Evaluation script for PatchCore anomaly detection
Computes image-level and pixel-level AUROC metrics
"""

import os
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

from patchcore import PatchCore
from patchcore.utils import load_image, load_dataset_images, load_ground_truth_mask
import config


def compute_image_level_metrics(results: list, threshold: float = None):
    """
    Compute image-level anomaly detection metrics
    
    Args:
        results: List of result dictionaries
        threshold: Optional custom threshold
    
    Returns:
        Dictionary with metrics
    """
    # Extract scores and labels
    scores = np.array([r['anomaly_score'] for r in results])
    labels = np.array([r['ground_truth_label'] for r in results])
    
    # Compute AUROC
    auroc = roc_auc_score(labels, scores)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # If threshold provided, compute accuracy at that threshold
    if threshold is not None:
        predictions = scores > threshold
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'auroc': auroc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'threshold': threshold,
            'fpr': fpr,
            'tpr': tpr,
            'roc_thresholds': thresholds
        }
    else:
        metrics = {
            'auroc': auroc,
            'fpr': fpr,
            'tpr': tpr,
            'roc_thresholds': thresholds
        }
    
    return metrics


def compute_pixel_level_metrics(results: list):
    """
    Compute pixel-level anomaly localization metrics
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Pixel-level AUROC
    """
    all_masks = []
    all_heatmaps = []
    
    for r in results:
        if r['gt_mask'] is not None and r['anomaly_map'] is not None:
            # Resize heatmap to match ground truth mask
            heatmap = r['anomaly_map']
            gt_mask = r['gt_mask']
            
            if heatmap.shape != gt_mask.shape:
                heatmap = cv2.resize(heatmap, (gt_mask.shape[1], gt_mask.shape[0]))
            
            all_masks.append(gt_mask.flatten())
            all_heatmaps.append(heatmap.flatten())
    
    if len(all_masks) == 0:
        print("Warning: No ground truth masks found for pixel-level evaluation")
        return None
    
    # Concatenate all pixels
    all_masks = np.concatenate(all_masks)
    all_heatmaps = np.concatenate(all_heatmaps)
    
    # Compute pixel-level AUROC
    pixel_auroc = roc_auc_score(all_masks, all_heatmaps)
    
    return pixel_auroc


def plot_roc_curve(metrics: dict, save_path: str = None):
    """
    Plot ROC curve
    
    Args:
        metrics: Metrics dictionary with fpr, tpr, auroc
        save_path: Path to save plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(metrics['fpr'], metrics['tpr'], linewidth=2, label=f"AUROC = {metrics['auroc']:.4f}")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Image-Level Detection', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.close()


def plot_score_distribution(results: list, threshold: float, save_path: str = None):
    """
    Plot distribution of anomaly scores
    
    Args:
        results: List of result dictionaries
        threshold: Threshold value
        save_path: Path to save plot
    """
    normal_scores = [r['anomaly_score'] for r in results if not r['ground_truth_label']]
    anomaly_scores = [r['anomaly_score'] for r in results if r['ground_truth_label']]
    
    plt.figure(figsize=(10, 6))
    
    bins = np.linspace(0, max(max(normal_scores), max(anomaly_scores)), 50)
    
    plt.hist(normal_scores, bins=bins, alpha=0.6, label='Normal', color='green', edgecolor='black')
    plt.hist(anomaly_scores, bins=bins, alpha=0.6, label='Anomalous', color='red', edgecolor='black')
    plt.axvline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.4f}')
    
    plt.xlabel('Anomaly Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Anomaly Scores', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Score distribution saved to {save_path}")
    
    plt.close()


def evaluate_category(category: str, save_plots: bool = True):
    """
    Evaluate PatchCore on a category
    
    Args:
        category: Category name
        save_plots: Whether to save evaluation plots
    
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating PatchCore for category: {category}")
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
    
    # Run inference
    results = []
    print("\nRunning inference...")
    
    for image_path, defect_type in tqdm(test_data, desc="Processing images"):
        # Load image
        image_tensor, _ = load_image(image_path, size=config.IMAGE_SIZE)
        
        # Run prediction
        anomaly_score, anomaly_map, _ = model.predict(image_tensor, return_heatmap=True)
        
        # Get ground truth
        image_name = os.path.basename(image_path)
        gt_mask = load_ground_truth_mask(config.DATA_ROOT, category, defect_type, image_name)
        
        # Store results
        results.append({
            'image_path': image_path,
            'defect_type': defect_type,
            'anomaly_score': anomaly_score,
            'ground_truth_label': defect_type != 'good',
            'anomaly_map': anomaly_map,
            'gt_mask': gt_mask
        })
    
    # Compute metrics
    print("\n" + "="*60)
    print("Computing Metrics...")
    print("="*60)
    
    # Image-level metrics
    image_metrics = compute_image_level_metrics(results, threshold=model.threshold)
    
    print("\n📊 IMAGE-LEVEL METRICS:")
    print(f"  AUROC: {image_metrics['auroc']:.4f}")
    if 'accuracy' in image_metrics:
        print(f"  Accuracy: {image_metrics['accuracy']:.4f}")
        print(f"  Precision: {image_metrics['precision']:.4f}")
        print(f"  Recall: {image_metrics['recall']:.4f}")
        print(f"  F1-Score: {image_metrics['f1_score']:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    True Positives:  {image_metrics['true_positives']}")
        print(f"    True Negatives:  {image_metrics['true_negatives']}")
        print(f"    False Positives: {image_metrics['false_positives']}")
        print(f"    False Negatives: {image_metrics['false_negatives']}")
    
    # Pixel-level metrics
    print("\n🎯 PIXEL-LEVEL METRICS:")
    pixel_auroc = compute_pixel_level_metrics(results)
    if pixel_auroc is not None:
        print(f"  Pixel AUROC: {pixel_auroc:.4f}")
    else:
        print("  Pixel AUROC: N/A (no ground truth masks)")
    
    # Per-defect-type analysis
    print("\n📋 PER-DEFECT-TYPE ANALYSIS:")
    defect_types = {}
    for r in results:
        dt = r['defect_type']
        if dt not in defect_types:
            defect_types[dt] = []
        defect_types[dt].append(r['anomaly_score'])
    
    for dt in sorted(defect_types.keys()):
        scores = defect_types[dt]
        print(f"  {dt:20s}: {len(scores):3d} images, "
              f"Mean score: {np.mean(scores):.4f}, "
              f"Std: {np.std(scores):.4f}")
    
    # Save plots
    if save_plots:
        plot_dir = os.path.join(config.RESULTS_ROOT, category, 'evaluation')
        os.makedirs(plot_dir, exist_ok=True)
        
        # ROC curve
        roc_path = os.path.join(plot_dir, 'roc_curve.png')
        plot_roc_curve(image_metrics, roc_path)
        
        # Score distribution
        dist_path = os.path.join(plot_dir, 'score_distribution.png')
        plot_score_distribution(results, model.threshold, dist_path)
        
        print(f"\n📁 Evaluation plots saved to: {plot_dir}")
    
    print("\n" + "="*60)
    print(f"Evaluation complete for category: {category}")
    print("="*60 + "\n")
    
    return {
        'category': category,
        'image_metrics': image_metrics,
        'pixel_auroc': pixel_auroc,
        'results': results,
        'defect_types': defect_types
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PatchCore model")
    parser.add_argument(
        '--category',
        type=str,
        default='bottle',
        help='Category to evaluate (default: bottle)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Do not save evaluation plots'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Evaluate all categories'
    )
    
    args = parser.parse_args()
    
    if args.all:
        # Evaluate all categories
        all_results = {}
        for category in config.CATEGORIES:
            try:
                result = evaluate_category(category, save_plots=not args.no_plots)
                all_results[category] = result
            except Exception as e:
                print(f"Error evaluating {category}: {e}")
                continue
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY - ALL CATEGORIES")
        print("="*60)
        print(f"\n{'Category':<15} {'Image AUROC':<12} {'Pixel AUROC':<12}")
        print("-"*40)
        for category, result in all_results.items():
            img_auroc = result['image_metrics']['auroc']
            pix_auroc = result['pixel_auroc'] if result['pixel_auroc'] is not None else 0
            print(f"{category:<15} {img_auroc:<12.4f} {pix_auroc:<12.4f}")
        print("="*60 + "\n")
    else:
        # Evaluate single category
        evaluate_category(args.category, save_plots=not args.no_plots)


if __name__ == '__main__':
    main()
