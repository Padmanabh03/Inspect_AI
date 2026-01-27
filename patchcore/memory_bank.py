"""
Memory Bank for PatchCore with Coreset Sampling
"""

import torch
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from typing import Optional
import faiss


class MemoryBank:
    """
    Memory Bank for storing and querying patch features with coreset sampling
    """
    
    def __init__(self, sampling_ratio: float = 0.1, device: str = "cuda"):
        """
        Initialize Memory Bank
        
        Args:
            sampling_ratio: Ratio of patches to keep (0 < ratio <= 1)
            device: Device to use
        """
        self.sampling_ratio = sampling_ratio
        self.device = device
        self.memory_bank = None
        self.index = None
        self.feature_dim = None
    
    def fit(self, features: np.ndarray):
        """
        Build memory bank from training features using coreset sampling
        
        Args:
            features: Training features [N_samples, Feature_dim]
        """
        print(f"Building memory bank from {features.shape[0]} patches...")
        
        self.feature_dim = features.shape[1]
        
        # Apply coreset sampling to reduce memory
        n_samples = int(features.shape[0] * self.sampling_ratio)
        n_samples = max(1, n_samples)  # Ensure at least 1 sample
        
        print(f"Applying coreset sampling: {features.shape[0]} -> {n_samples} patches")
        sampled_features = self._greedy_coreset_sampling(features, n_samples)
        
        # Store in memory bank
        self.memory_bank = sampled_features
        
        # Build FAISS index for efficient nearest neighbor search
        self._build_index()
        
        print(f"Memory bank built: {self.memory_bank.shape}")
    
    def _greedy_coreset_sampling(self, features: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Greedy k-center coreset sampling
        
        Args:
            features: Input features [N, D]
            n_samples: Number of samples to select
        
        Returns:
            Sampled features [n_samples, D]
        """
        N = features.shape[0]
        
        if n_samples >= N:
            return features
        
        # Random projection for faster distance computation (optional, for large datasets)
        if features.shape[1] > 512:
            reducer = SparseRandomProjection(n_components=512, random_state=42)
            features_reduced = reducer.fit_transform(features)
        else:
            features_reduced = features
        
        # Initialize with a random point
        selected_indices = [np.random.randint(N)]
        selected = [features[selected_indices[0]]]
        
        # Compute initial distances
        distances = np.linalg.norm(
            features_reduced - features_reduced[selected_indices[0]],
            axis=1
        )
        
        # Greedily select farthest points
        for _ in range(n_samples - 1):
            # Select point with maximum distance to any selected point
            farthest_idx = np.argmax(distances)
            selected_indices.append(farthest_idx)
            selected.append(features[farthest_idx])
            
            # Update distances
            new_distances = np.linalg.norm(
                features_reduced - features_reduced[farthest_idx],
                axis=1
            )
            distances = np.minimum(distances, new_distances)
        
        return np.array(selected)
    
    def _build_index(self):
        """Build FAISS index for efficient nearest neighbor search"""
        # Normalize features for better distance computation
        self.memory_bank = self.memory_bank.astype(np.float32)
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.feature_dim)
        self.index.add(self.memory_bank)
    
    def predict(self, features: np.ndarray, n_neighbors: int = 1) -> tuple:
        """
        Compute anomaly scores for query features
        
        Args:
            features: Query features [N_patches, Feature_dim]
            n_neighbors: Number of nearest neighbors to use
        
        Returns:
            distances: Nearest neighbor distances [N_patches]
            indices: Indices of nearest neighbors [N_patches, n_neighbors]
        """
        if self.index is None:
            raise ValueError("Memory bank not fitted. Call fit() first.")
        
        features = features.astype(np.float32)
        
        # Search for nearest neighbors
        distances, indices = self.index.search(features, n_neighbors)
        
        # Use mean distance if using multiple neighbors
        if n_neighbors > 1:
            distances = distances.mean(axis=1)
        else:
            distances = distances.squeeze()
        
        return distances, indices
    
    def save(self, path: str):
        """
        Save memory bank to disk
        
        Args:
            path: Path to save file
        """
        if self.memory_bank is None:
            raise ValueError("Memory bank not fitted. Call fit() first.")
        
        np.save(path, {
            'memory_bank': self.memory_bank,
            'sampling_ratio': self.sampling_ratio,
            'feature_dim': self.feature_dim
        })
        print(f"Memory bank saved to {path}")
    
    def load(self, path: str):
        """
        Load memory bank from disk
        
        Args:
            path: Path to load file
        """
        data = np.load(path, allow_pickle=True).item()
        
        self.memory_bank = data['memory_bank']
        self.sampling_ratio = data['sampling_ratio']
        self.feature_dim = data['feature_dim']
        
        # Rebuild index
        self._build_index()
        
        print(f"Memory bank loaded from {path}: {self.memory_bank.shape}")
