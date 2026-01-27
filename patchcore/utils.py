"""
Utility functions for PatchCore
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List
import torchvision.transforms as transforms


def load_image(image_path: str, size: int = 224) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load and preprocess image
    
    Args:
        image_path: Path to image
        size: Target size
    
    Returns:
        tensor: Preprocessed tensor [1, 3, H, W]
        original: Original image as numpy array
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    original = np.array(image)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    tensor = transform(image).unsqueeze(0)
    
    return tensor, original


def load_dataset_images(data_root: str, category: str, split: str) -> List[str]:
    """
    Load image paths from dataset
    
    Args:
        data_root: Root directory of dataset
        category: Category name (e.g., 'bottle')
        split: 'train' or 'test'
    
    Returns:
        List of image paths
    """
    if split == 'train':
        image_dir = os.path.join(data_root, category, split, 'good')
        if not os.path.exists(image_dir):
            raise ValueError(f"Directory not found: {image_dir}")
        
        image_paths = []
        for fname in sorted(os.listdir(image_dir)):
            if fname.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(image_dir, fname))
        
        return image_paths
    
    elif split == 'test':
        test_dir = os.path.join(data_root, category, split)
        if not os.path.exists(test_dir):
            raise ValueError(f"Directory not found: {test_dir}")
        
        image_paths = []
        defect_types = []
        
        # Iterate through all defect type folders
        for defect_type in sorted(os.listdir(test_dir)):
            defect_dir = os.path.join(test_dir, defect_type)
            if not os.path.isdir(defect_dir):
                continue
            
            for fname in sorted(os.listdir(defect_dir)):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(defect_dir, fname))
                    defect_types.append(defect_type)
        
        return list(zip(image_paths, defect_types))
    
    else:
        raise ValueError(f"Unknown split: {split}")


def load_ground_truth_mask(data_root: str, category: str, defect_type: str, image_name: str) -> np.ndarray:
    """
    Load ground truth mask for a test image
    
    Args:
        data_root: Root directory of dataset
        category: Category name
        defect_type: Defect type (e.g., 'broken_large')
        image_name: Image filename (e.g., '000.png')
    
    Returns:
        Ground truth mask as binary numpy array [H, W] or None if not available
    """
    if defect_type == 'good':
        return None
    
    # Construct mask path
    mask_name = image_name.replace('.png', '_mask.png')
    mask_path = os.path.join(data_root, category, 'ground_truth', defect_type, mask_name)
    
    if not os.path.exists(mask_path):
        return None
    
    # Load and convert to binary
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.uint8)
    
    return mask


def visualize_result(
    image: np.ndarray,
    heatmap: np.ndarray,
    score: float,
    threshold: float = None,
    save_path: str = None,
    ground_truth: np.ndarray = None
) -> np.ndarray:
    """
    Visualize anomaly detection result
    
    Args:
        image: Original image [H, W, 3]
        heatmap: Anomaly heatmap [H, W]
        score: Anomaly score
        threshold: Threshold value
        save_path: Path to save visualization
        ground_truth: Ground truth mask (optional)
    
    Returns:
        Overlay image
    """
    # Resize image if needed
    if image.shape[:2] != heatmap.shape:
        image = cv2.resize(image, (heatmap.shape[1], heatmap.shape[0]))
    
    # Create heatmap overlay
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original image
    overlay = cv2.addWeighted(image, 0.5, heatmap_colored, 0.5, 0)
    
    # Create visualization
    if ground_truth is not None:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title(f'Anomaly Heatmap\nScore: {score:.4f}')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        axes[3].imshow(ground_truth, cmap='gray')
        axes[3].set_title('Ground Truth')
        axes[3].axis('off')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title(f'Anomaly Heatmap\nScore: {score:.4f}')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
    
    # Add threshold line if provided
    if threshold is not None:
        is_anomaly = score > threshold
        status = "ANOMALOUS" if is_anomaly else "NORMAL"
        color = "red" if is_anomaly else "green"
        fig.suptitle(f'Status: {status} (Threshold: {threshold:.4f})', 
                     fontsize=14, fontweight='bold', color=color)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.close()
    
    return overlay


def save_heatmap(heatmap: np.ndarray, save_path: str):
    """
    Save heatmap as image
    
    Args:
        heatmap: Anomaly heatmap [H, W]
        save_path: Path to save
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert to 8-bit
    heatmap_8bit = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
    
    cv2.imwrite(save_path, heatmap_colored)


def create_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create simple overlay of heatmap on image
    
    Args:
        image: Original image [H, W, 3]
        heatmap: Anomaly heatmap [H, W]
        alpha: Blending factor
    
    Returns:
        Overlay image
    """
    # Resize image if needed
    if image.shape[:2] != heatmap.shape:
        image = cv2.resize(image, (heatmap.shape[1], heatmap.shape[0]))
    
    # Create colored heatmap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay
