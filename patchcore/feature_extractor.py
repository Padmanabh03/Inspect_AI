"""
Feature Extractor for PatchCore using WideResNet50
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict


class FeatureExtractor(nn.Module):
    """
    Extract multi-scale features from intermediate layers of WideResNet50
    """
    
    def __init__(self, backbone: str = "wide_resnet50_2", layers: List[str] = None, device: str = "cuda"):
        """
        Initialize feature extractor
        
        Args:
            backbone: Backbone network name
            layers: List of layer names to extract features from
            device: Device to run on ('cuda' or 'cpu')
        """
        super().__init__()
        
        self.backbone = backbone
        self.layers = layers or ["layer2", "layer3"]
        self.device = device
        
        # Load pretrained backbone
        if backbone == "wide_resnet50_2":
            self.model = models.wide_resnet50_2(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Register hooks to extract intermediate features
        self.features = {}
        self.hooks = []
        self._register_hooks()
        
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _register_hooks(self):
        """Register forward hooks to extract features from specified layers"""
        
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook
        
        for layer_name in self.layers:
            layer = dict([*self.model.named_modules()])[layer_name]
            handle = layer.register_forward_hook(get_activation(layer_name))
            self.hooks.append(handle)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from input image
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Dictionary of features {layer_name: tensor}
        """
        self.features.clear()
        
        with torch.no_grad():
            _ = self.model(x)
        
        return self.features
    
    def extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract and aggregate patch features from multiple layers
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Patch features [B, N_patches, Feature_dim]
        """
        features = self.forward(x)
        
        # Get feature maps from each layer
        feature_maps = []
        for layer_name in self.layers:
            feat = features[layer_name]  # [B, C, H, W]
            
            # Adaptive pooling to ensure same spatial resolution
            if len(feature_maps) > 0:
                target_size = feature_maps[0].shape[2:]
                feat = nn.functional.adaptive_avg_pool2d(feat, target_size)
            
            feature_maps.append(feat)
        
        # Concatenate along channel dimension
        combined_features = torch.cat(feature_maps, dim=1)  # [B, C_total, H, W]
        
        # Reshape to [B, C_total, H*W] then transpose to [B, H*W, C_total]
        B, C, H, W = combined_features.shape
        patch_features = combined_features.view(B, C, -1).permute(0, 2, 1)
        
        return patch_features
    
    def __del__(self):
        """Remove hooks when object is destroyed"""
        for hook in self.hooks:
            hook.remove()
