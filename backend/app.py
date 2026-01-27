"""
InspectAI Backend - FastAPI REST API for Industrial Visual Inspection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import base64
import io
import numpy as np
from PIL import Image
import torch
import cv2
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patchcore import PatchCore
import config as app_config

# Initialize FastAPI app
app = FastAPI(
    title="InspectAI API",
    description="Industrial Visual Inspection System - Anomaly Detection API",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
model_cache = {}

# Mount static files (frontend)
frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


# Pydantic models for API
class InspectionResult(BaseModel):
    """Inspection result response"""
    category: str
    anomaly_score: float
    threshold: float
    is_anomaly: bool
    decision: str  # "PASS" or "FAIL"
    heatmap_base64: Optional[str] = None
    overlay_base64: Optional[str] = None
    timestamp: str
    

class CategoryInfo(BaseModel):
    """Category information"""
    name: str
    model_loaded: bool
    threshold: Optional[float] = None


class InspectionHistory(BaseModel):
    """Historical inspection record"""
    id: int
    category: str
    timestamp: str
    anomaly_score: float
    decision: str


# Helper functions
def load_model(category: str) -> PatchCore:
    """
    Load PatchCore model for a category (with caching)
    """
    if category in model_cache:
        return model_cache[category]
    
    model_path = os.path.join(app_config.MODELS_ROOT, category)
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Model not found for category: {category}. Please train the model first."
        )
    
    # Initialize and load model
    model = PatchCore(
        backbone=app_config.BACKBONE,
        layers=app_config.LAYERS,
        device=app_config.DEVICE,
        sampling_ratio=app_config.CORESET_SAMPLING_RATIO,
        n_neighbors=app_config.NEIGHBORS
    )
    model.load(model_path)
    
    # Cache the model
    model_cache[category] = model
    
    return model


def preprocess_image(image_file: bytes) -> tuple:
    """
    Preprocess uploaded image for inference
    
    Returns:
        (tensor, original_image_array)
    """
    # Load image from bytes
    image = Image.open(io.BytesIO(image_file)).convert('RGB')
    original = np.array(image)
    
    # Resize and normalize
    image_resized = image.resize((app_config.IMAGE_SIZE, app_config.IMAGE_SIZE))
    img_array = np.array(image_resized).astype(np.float32) / 255.0
    
    # Normalize with ImageNet stats
    mean = np.array(app_config.IMAGENET_MEAN)
    std = np.array(app_config.IMAGENET_STD)
    img_normalized = (img_array - mean) / std
    
    # Convert to tensor [1, C, H, W]
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
    
    return img_tensor, original


def encode_image_base64(image_array: np.ndarray) -> str:
    """
    Encode numpy image array to base64 string
    """
    # Convert to PIL Image
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    
    pil_image = Image.fromarray(image_array)
    
    # Encode to base64
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_base64


def create_overlay(original: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """
    Create overlay visualization
    """
    # Resize heatmap to match original
    if original.shape[:2] != heatmap.shape:
        heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    
    # Resize original to reasonable size for display
    display_size = (512, 512)
    original_resized = cv2.resize(original, display_size)
    heatmap_resized = cv2.resize(heatmap, display_size)
    
    # Create colored heatmap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlay = cv2.addWeighted(original_resized, 0.6, heatmap_colored, 0.4, 0)
    
    return overlay


# API Endpoints

@app.get("/api")
async def root():
    """API root endpoint"""
    return {
        "service": "InspectAI",
        "version": "1.0.0",
        "description": "Industrial Visual Inspection System",
        "status": "operational"
    }


@app.get("/")
async def serve_frontend():
    """Serve frontend HTML"""
    frontend_index = os.path.join(frontend_path, "index.html")
    if os.path.exists(frontend_index):
        return FileResponse(frontend_index)
    return {"message": "Frontend not found. Please ensure frontend files are in the 'frontend' directory."}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": app_config.DEVICE,
        "cuda_available": torch.cuda.is_available()
    }


@app.get("/categories", response_model=List[CategoryInfo])
async def get_categories():
    """
    Get list of available categories
    """
    categories = []
    
    for category in app_config.CATEGORIES:
        model_path = os.path.join(app_config.MODELS_ROOT, category)
        model_loaded = os.path.exists(model_path)
        
        threshold = None
        if category in model_cache:
            threshold = model_cache[category].threshold
        
        categories.append(CategoryInfo(
            name=category,
            model_loaded=model_loaded,
            threshold=threshold
        ))
    
    return categories


@app.post("/inspect", response_model=InspectionResult)
async def inspect_image(
    category: str = Form(...),
    image: UploadFile = File(...),
    return_visualizations: bool = Form(True)
):
    """
    Inspect an uploaded image for anomalies
    
    Parameters:
    - category: Product category (e.g., 'bottle', 'cable')
    - image: Uploaded image file
    - return_visualizations: Whether to return heatmap and overlay images
    
    Returns:
    - InspectionResult with anomaly score, decision, and visualizations
    """
    try:
        # Load model
        model = load_model(category)
        
        # Read and preprocess image
        image_bytes = await image.read()
        img_tensor, original_image = preprocess_image(image_bytes)
        
        # Run inference
        anomaly_score, heatmap, _ = model.predict(img_tensor, return_heatmap=True)
        
        # Make decision
        is_anomaly = model.is_anomaly(anomaly_score)
        decision = "FAIL" if is_anomaly else "PASS"
        
        # Create visualizations if requested
        heatmap_base64 = None
        overlay_base64 = None
        
        if return_visualizations and heatmap is not None:
            # Create overlay
            overlay = create_overlay(original_image, heatmap)
            
            # Encode to base64
            heatmap_display = cv2.resize(heatmap, (512, 512))
            heatmap_colored = cv2.applyColorMap(
                (heatmap_display * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            heatmap_base64 = encode_image_base64(heatmap_colored)
            overlay_base64 = encode_image_base64(overlay)
        
        # Create response
        result = InspectionResult(
            category=category,
            anomaly_score=float(anomaly_score),
            threshold=float(model.threshold),
            is_anomaly=bool(is_anomaly),
            decision=decision,
            heatmap_base64=heatmap_base64,
            overlay_base64=overlay_base64,
            timestamp=datetime.now().isoformat()
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info/{category}")
async def get_model_info(category: str):
    """
    Get information about a specific model
    """
    try:
        model = load_model(category)
        
        return {
            "category": category,
            "threshold": float(model.threshold),
            "backbone": model.backbone,
            "device": model.device,
            "loaded": True
        }
    except HTTPException as e:
        return {
            "category": category,
            "loaded": False,
            "error": e.detail
        }


@app.post("/model/reload/{category}")
async def reload_model(category: str):
    """
    Reload a model (clear cache and reload from disk)
    """
    if category in model_cache:
        del model_cache[category]
    
    try:
        model = load_model(category)
        return {
            "status": "success",
            "message": f"Model for {category} reloaded successfully",
            "threshold": float(model.threshold)
        }
    except HTTPException as e:
        raise e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
