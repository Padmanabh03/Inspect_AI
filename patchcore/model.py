"""
PatchCore Main Model
"""

import torch
import numpy as np
from typing import Tuple, Optional
import cv2
from scipy.ndimage import gaussian_filter

from .feature_extractor import FeatureExtractor
from .memory_bank import MemoryBank


class PatchCore:
    """
    PatchCore anomaly detection model
    """
    
    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: list = None,
        device: str = "cuda",
        sampling_ratio: float = 0.1,
        n_neighbors: int = 9
    ):
        """
        Initialize PatchCore model
        
        Args:
            backbone: Backbone network name
            layers: List of layers to extract features from
            device: Device to use ('cuda' or 'cpu')
            sampling_ratio: Coreset sampling ratio
            n_neighbors: Number of nearest neighbors for scoring
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.backbone = backbone
        self.layers = layers or ["layer2", "layer3"]
        self.sampling_ratio = sampling_ratio
        self.n_neighbors = n_neighbors
        
        # Initialize components
        self.feature_extractor = FeatureExtractor(
            backbone=backbone,
            layers=self.layers,
            device=self.device
        )
        
        self.memory_bank = MemoryBank(
            sampling_ratio=sampling_ratio,
            device=self.device
        )
        
        self.threshold = None
        self.feature_shape = None
    
    def fit(self, images: list):
        """
        Train PatchCore on normal images
        
        Args:
            images: List of normal image tensors or numpy arrays
        """
        print(f"\nTraining PatchCore on {len(images)} images...")
        
        all_features = []
        
        for i, image in enumerate(images):
            if i % 10 == 0:
                print(f"Processing image {i+1}/{len(images)}")
            
            # Convert to tensor if needed
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()
            
            # Ensure correct shape [1, C, H, W]
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            image = image.to(self.device)
            
            # Extract patch features
            features = self.feature_extractor.extract_patch_features(image)
            
            # Store shape for later
            if self.feature_shape is None:
                B, N, D = features.shape
                # Calculate spatial dimensions from N_patches
                self.feature_shape = int(np.sqrt(N))
            
            # Move to CPU and store
            all_features.append(features.cpu().numpy())
        
        # Concatenate all features [Total_patches, Feature_dim]
        all_features = np.concatenate(all_features, axis=0)
        all_features = all_features.reshape(-1, all_features.shape[-1])
        
        print(f"\nTotal patches collected: {all_features.shape[0]}")
        
        # Build memory bank
        self.memory_bank.fit(all_features)
        
        print("Training complete!")
    
    def predict(
        self,
        image: torch.Tensor,
        return_heatmap: bool = True
    ) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Predict anomaly score for a single image
        
        Args:
            image: Input image tensor [1, C, H, W] or [C, H, W]
            return_heatmap: Whether to return anomaly heatmap
        
        Returns:
            anomaly_score: Image-level anomaly score
            anomaly_map: Pixel-level anomaly map (if return_heatmap=True)
            patch_scores: Raw patch scores
        """
        # Ensure correct shape
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        original_size = image.shape[2:]
        image = image.to(self.device)
        
        # Extract patch features
        features = self.feature_extractor.extract_patch_features(image)
        B, N, D = features.shape
        
        # Reshape for memory bank prediction
        features_flat = features.reshape(-1, D).cpu().numpy()
        
        # Get nearest neighbor distances
        distances, _ = self.memory_bank.predict(features_flat, self.n_neighbors)
        
        # Reshape to spatial dimensions
        spatial_size = int(np.sqrt(N))
        patch_scores = distances.reshape(spatial_size, spatial_size)
        
        # Image-level score (max or high percentile)
        anomaly_score = float(np.max(patch_scores))
        
        # Generate anomaly heatmap
        anomaly_map = None
        if return_heatmap:
            anomaly_map = self._generate_heatmap(patch_scores, original_size)
        
        return anomaly_score, anomaly_map, patch_scores
    
    def _generate_heatmap(self, patch_scores: np.ndarray, target_size: tuple) -> np.ndarray:
        """
        Generate smooth anomaly heatmap from patch scores
        
        Args:
            patch_scores: Patch-level anomaly scores [H, W]
            target_size: Target image size (H, W)
        
        Returns:
            Smooth heatmap [H, W] normalized to [0, 1]
        """
        # Apply Gaussian smoothing
        heatmap = gaussian_filter(patch_scores, sigma=4)
        
        # Resize to original image size
        heatmap = cv2.resize(
            heatmap,
            (target_size[1], target_size[0]),
            interpolation=cv2.INTER_CUBIC
        )
        
        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def set_threshold(self, threshold: float):
        """
        Set anomaly threshold
        
        Args:
            threshold: Threshold value
        """
        self.threshold = threshold
        print(f"Threshold set to: {threshold:.4f}")
    
    def compute_threshold(
        self,
        normal_images: list,
        method: str = "mean_std",
        std_multiplier: float = 3.0,
        percentile: float = 99.0
    ) -> float:
        """
        Compute anomaly threshold from normal images
        
        Args:
            normal_images: List of normal image tensors
            method: 'mean_std' or 'percentile'
            std_multiplier: Multiplier for standard deviation (if method='mean_std')
            percentile: Percentile value (if method='percentile')
        
        Returns:
            Threshold value
        """
        print(f"\nComputing threshold from {len(normal_images)} normal images...")
        
        scores = []
        for image in normal_images:
            score, _, _ = self.predict(image, return_heatmap=False)
            scores.append(score)
        
        scores = np.array(scores)
        
        if method == "mean_std":
            threshold = np.mean(scores) + std_multiplier * np.std(scores)
        elif method == "percentile":
            threshold = np.percentile(scores, percentile)
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        
        self.threshold = threshold
        print(f"Threshold computed: {threshold:.4f} (method: {method})")
        print(f"Normal scores - Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
        
        return threshold
    
    def is_anomaly(self, anomaly_score: float) -> bool:
        """
        Check if score indicates anomaly
        
        Args:
            anomaly_score: Anomaly score
        
        Returns:
            True if anomalous, False otherwise
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() or compute_threshold() first.")
        
        return anomaly_score > self.threshold
    
    def save(self, path: str):
        """
        Save model to disk
        
        Args:
            path: Directory path to save model
        """
        import os
        import json
        
        os.makedirs(path, exist_ok=True)
        
        # Save memory bank
        memory_bank_path = os.path.join(path, "memory_bank.npy")
        self.memory_bank.save(memory_bank_path)
        
        # Save configuration
        config = {
            'backbone': self.backbone,
            'layers': self.layers,
            'sampling_ratio': self.sampling_ratio,
            'n_neighbors': self.n_neighbors,
            'threshold': self.threshold,
            'feature_shape': self.feature_shape
        }
        
        config_path = os.path.join(path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nModel saved to {path}")
    
    def load(self, path: str):
        """
        Load model from disk
        
        Args:
            path: Directory path to load model from
        """
        import os
        import json
        
        # Load configuration
        config_path = os.path.join(path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update model parameters
        self.threshold = config.get('threshold')
        self.feature_shape = config.get('feature_shape')
        
        # Load memory bank
        memory_bank_path = os.path.join(path, "memory_bank.npy")
        self.memory_bank.load(memory_bank_path)
        
        print(f"\nModel loaded from {path}")
