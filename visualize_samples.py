"""
Visualize anomaly detection results on sample images (one per defect type)
"""

import os
import matplotlib.pyplot as plt
from patchcore import PatchCore
from patchcore.utils import load_image, load_dataset_images, load_ground_truth_mask
import config
import numpy as np
import cv2


def visualize_samples(category='bottle', output_path='sample_visualizations.png'):
    """
    Create a grid visualization showing sample results from each defect type
    """
    print(f"\n{'='*70}")
    print(f"Visualizing Sample Results for Category: {category}")
    print(f"{'='*70}\n")
    
    # Load model
    model_path = os.path.join(config.MODELS_ROOT, category)
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print(f"Please train the model first: python train.py --category {category}")
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
    
    # Load test images
    print("Loading test images...")
    test_data = load_dataset_images(config.DATA_ROOT, category, 'test')
    
    # Group by defect type and pick one sample from each
    defect_samples = {}
    for img_path, defect_type in test_data:
        if defect_type not in defect_samples:
            defect_samples[defect_type] = img_path
    
    print(f"\nFound {len(defect_samples)} defect types:")
    for dt in sorted(defect_samples.keys()):
        print(f"  - {dt}")
    
    # Process samples
    print("\nProcessing samples...")
    results = []
    
    for defect_type in sorted(defect_samples.keys()):
        img_path = defect_samples[defect_type]
        img_name = os.path.basename(img_path)
        
        # Load image
        img_tensor, img_original = load_image(img_path, size=config.IMAGE_SIZE)
        
        # Run inference
        score, heatmap, _ = model.predict(img_tensor, return_heatmap=True)
        is_anomaly = model.is_anomaly(score)
        
        # Load ground truth mask if available
        gt_mask = load_ground_truth_mask(config.DATA_ROOT, category, defect_type, img_name)
        
        # Resize images for consistent display
        display_size = (224, 224)
        img_resized = cv2.resize(img_original, display_size)
        heatmap_resized = cv2.resize(heatmap, display_size)
        
        # Create colored heatmap
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Create overlay
        overlay = cv2.addWeighted(img_resized, 0.5, heatmap_colored, 0.5, 0)
        
        # Resize ground truth if available
        if gt_mask is not None:
            gt_mask_resized = cv2.resize(gt_mask, display_size)
        else:
            gt_mask_resized = None
        
        results.append({
            'defect_type': defect_type,
            'image': img_resized,
            'heatmap': heatmap_resized,
            'overlay': overlay,
            'gt_mask': gt_mask_resized,
            'score': score,
            'is_anomaly': is_anomaly,
            'threshold': model.threshold
        })
        
        print(f"  ✓ {defect_type:20s} - Score: {score:.4f} - {'ANOMALOUS' if is_anomaly else 'NORMAL'}")
    
    # Create visualization
    print(f"\nCreating visualization grid...")
    n_samples = len(results)
    
    # Create figure with 4 columns: Original, Heatmap, Overlay, Ground Truth
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
    
    # Handle single sample case
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        # Original image
        axes[i, 0].imshow(result['image'])
        axes[i, 0].set_title(f"{result['defect_type']}\nScore: {result['score']:.4f}", fontsize=10)
        axes[i, 0].axis('off')
        
        # Heatmap
        axes[i, 1].imshow(result['heatmap'], cmap='jet')
        axes[i, 1].set_title('Anomaly Heatmap', fontsize=10)
        axes[i, 1].axis('off')
        
        # Overlay
        axes[i, 2].imshow(result['overlay'])
        status = "ANOMALY" if result['is_anomaly'] else "NORMAL"
        color = 'red' if result['is_anomaly'] else 'green'
        axes[i, 2].set_title(f'Overlay\n{status}', fontsize=10, color=color, fontweight='bold')
        axes[i, 2].axis('off')
        
        # Ground truth
        if result['gt_mask'] is not None:
            axes[i, 3].imshow(result['gt_mask'], cmap='gray')
            axes[i, 3].set_title('Ground Truth', fontsize=10)
        else:
            axes[i, 3].text(0.5, 0.5, 'No GT Available', 
                          ha='center', va='center', fontsize=10)
            axes[i, 3].set_xlim(0, 1)
            axes[i, 3].set_ylim(0, 1)
        axes[i, 3].axis('off')
    
    # Add overall title
    fig.suptitle(f'InspectAI - Anomaly Detection Sample Results\nCategory: {category} | Threshold: {model.threshold:.4f}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save
    os.makedirs('visualizations', exist_ok=True)
    save_path = os.path.join('visualizations', output_path)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {save_path}")
    
    # Also save individual results
    print("\nSaving individual result images...")
    for result in results:
        defect_dir = os.path.join('visualizations', category, result['defect_type'])
        os.makedirs(defect_dir, exist_ok=True)
        
        # Save overlay
        overlay_bgr = cv2.cvtColor(result['overlay'], cv2.COLOR_RGB2BGR)
        overlay_path = os.path.join(defect_dir, 'sample_overlay.png')
        cv2.imwrite(overlay_path, overlay_bgr)
    
    print(f"✅ Individual results saved to: visualizations/{category}/")
    
    print(f"\n{'='*70}")
    print("Visualization Complete!")
    print(f"{'='*70}\n")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize sample anomaly detection results")
    parser.add_argument('--category', type=str, default='bottle', help='Category to visualize')
    parser.add_argument('--output', type=str, default='sample_visualizations.png', help='Output filename')
    
    args = parser.parse_args()
    
    visualize_samples(args.category, args.output)
