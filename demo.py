"""
Quick demo script to test PatchCore implementation
Tests on a small subset of data before full training
"""

import os
import torch
from patchcore import PatchCore
from patchcore.utils import load_image, load_dataset_images
import config

def quick_demo():
    """
    Run a quick demo to verify everything works
    """
    print("\n" + "="*70)
    print("  PatchCore Anomaly Detection - Quick Demo")
    print("="*70 + "\n")
    
    # Check dataset
    category = 'bottle'
    print(f"📁 Checking dataset for category: {category}")
    
    category_path = os.path.join(config.DATA_ROOT, category)
    if not os.path.exists(category_path):
        print(f"❌ Error: Category not found at {category_path}")
        return
    
    # Load training images
    print("\n📷 Loading training images...")
    train_paths = load_dataset_images(config.DATA_ROOT, category, 'train')
    print(f"   Found {len(train_paths)} training images")
    
    # Load test images
    print("\n🔍 Loading test images...")
    test_data = load_dataset_images(config.DATA_ROOT, category, 'test')
    print(f"   Found {len(test_data)} test images")
    
    # Count defect types
    defect_types = {}
    for _, defect_type in test_data:
        defect_types[defect_type] = defect_types.get(defect_type, 0) + 1
    
    print("\n   Defect types:")
    for dt, count in sorted(defect_types.items()):
        print(f"     - {dt}: {count} images")
    
    # Check device
    print("\n💻 Checking compute device...")
    if torch.cuda.is_available():
        print(f"   ✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   ℹ️  GPU not available, will use CPU (slower)")
    
    # Test image loading
    print("\n🖼️  Testing image loading...")
    test_img_path = train_paths[0]
    img_tensor, img_original = load_image(test_img_path, size=config.IMAGE_SIZE)
    print(f"   Image loaded: {img_tensor.shape}")
    print(f"   Original size: {img_original.shape}")
    
    # Mini training test (use only 5 images)
    print("\n🚀 Running mini training test (5 images)...")
    print("   This will take 1-2 minutes...")
    
    train_images = []
    for img_path in train_paths[:5]:
        img_tensor, _ = load_image(img_path, size=config.IMAGE_SIZE)
        train_images.append(img_tensor)
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PatchCore(
        backbone=config.BACKBONE,
        layers=config.LAYERS,
        device=device,
        sampling_ratio=config.CORESET_SAMPLING_RATIO,
        n_neighbors=config.NEIGHBORS
    )
    
    # Train on mini dataset
    try:
        model.fit(train_images)
        print("   ✅ Mini training successful!")
    except Exception as e:
        print(f"   ❌ Error during training: {e}")
        return
    
    # Test inference
    print("\n🔮 Testing inference...")
    test_img_path, defect_type = test_data[0]
    img_tensor, img_original = load_image(test_img_path, size=config.IMAGE_SIZE)
    
    try:
        # Compute threshold first
        model.compute_threshold(train_images[:3], method="percentile", percentile=95)
        
        # Run inference
        score, heatmap, _ = model.predict(img_tensor, return_heatmap=True)
        is_anomaly = model.is_anomaly(score)
        
        print(f"   Image: {os.path.basename(test_img_path)}")
        print(f"   Defect type: {defect_type}")
        print(f"   Anomaly score: {score:.4f}")
        print(f"   Threshold: {model.threshold:.4f}")
        print(f"   Predicted: {'ANOMALOUS' if is_anomaly else 'NORMAL'}")
        print(f"   Ground truth: {'ANOMALOUS' if defect_type != 'good' else 'NORMAL'}")
        print(f"   Heatmap shape: {heatmap.shape}")
        print("   ✅ Inference successful!")
        
    except Exception as e:
        print(f"   ❌ Error during inference: {e}")
        return
    
    # Summary
    print("\n" + "="*70)
    print("  ✅ Demo Complete - All Systems Working!")
    print("="*70)
    print("\n📋 Next Steps:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Train on full dataset: python train.py --category bottle")
    print("   3. Run inference: python inference.py --category bottle")
    print("   4. Evaluate: python evaluate.py --category bottle")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    try:
        quick_demo()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print("  1. Dependencies are installed: pip install -r requirements.txt")
        print("  2. MVTec dataset is in the correct location")
        print("  3. Python version is 3.8 or higher")
