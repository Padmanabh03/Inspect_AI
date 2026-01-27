# InspectAI - Deployment Guide

Complete guide for deploying the InspectAI industrial visual inspection system.

## 🎯 Overview

InspectAI is a production-ready anomaly detection system designed for industrial quality assurance. The system uses unsupervised learning (PatchCore) to detect defects without requiring labeled defect samples.

## 📋 Prerequisites

- Python 3.8+
- 4GB+ RAM (8GB+ recommended)
- GPU with CUDA support (optional, for faster inference)
- Docker & Docker Compose (for containerized deployment)

## 🚀 Quick Start (Local Development)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Model (if not already done)

```bash
# Train on bottle category
python train.py --category bottle

# Or train on all categories
python train.py --all
```

### 3. Start the Application

```bash
python start_app.py
```

This will:
- Start the FastAPI backend on port 8000
- Start the frontend on port 3000
- Open the inspection console in your browser

### 4. Access the System

- **Inspection Console**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **API Endpoint**: http://localhost:8000

## 🐳 Docker Deployment

For production deployment with Docker:

### Build and Run

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Access Points

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000

## 📡 API Usage

### Inspect Image

**Endpoint**: `POST /inspect`

**Parameters**:
- `category`: Product category (e.g., "bottle")
- `image`: Image file (JPG/PNG)
- `return_visualizations`: Boolean (default: true)

**Response**:
```json
{
  "category": "bottle",
  "anomaly_score": 45.57,
  "threshold": 25.43,
  "is_anomaly": true,
  "decision": "FAIL",
  "heatmap_base64": "...",
  "overlay_base64": "...",
  "timestamp": "2026-01-27T10:30:00"
}
```

### Example with cURL

```bash
curl -X POST "http://localhost:8000/inspect" \
  -F "category=bottle" \
  -F "image=@test_image.png"
```

### Example with Python

```python
import requests

url = "http://localhost:8000/inspect"
files = {"image": open("test_image.png", "rb")}
data = {"category": "bottle"}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Decision: {result['decision']}")
print(f"Anomaly Score: {result['anomaly_score']}")
```

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Model Configuration
BACKBONE = "wide_resnet50_2"
DEVICE = "cuda"  # or "cpu"
IMAGE_SIZE = 224

# PatchCore Settings
CORESET_SAMPLING_RATIO = 0.1
NEIGHBORS = 9

# Threshold Method
THRESHOLD_METHOD = "mean_std"
THRESHOLD_STD_MULTIPLIER = 3.0
```

## 📊 Performance Tuning

### For Better Accuracy
- Increase `CORESET_SAMPLING_RATIO` (more memory usage)
- Adjust `THRESHOLD_STD_MULTIPLIER` based on your tolerance

### For Faster Inference
- Reduce `IMAGE_SIZE` (e.g., 128 or 196)
- Decrease `CORESET_SAMPLING_RATIO`
- Use GPU with CUDA

### Memory Optimization
- Lower `CORESET_SAMPLING_RATIO`
- Reduce `IMAGE_SIZE`
- Use CPU instead of GPU if RAM limited

## 🔐 Production Considerations

### Security
1. **Authentication**: Add API key authentication to `/inspect` endpoint
2. **Rate Limiting**: Implement rate limiting to prevent abuse
3. **CORS**: Configure CORS properly for production domains
4. **HTTPS**: Use reverse proxy (Nginx) with SSL certificates

### Scalability
1. **Load Balancing**: Deploy multiple backend instances behind a load balancer
2. **Model Caching**: Models are cached in memory after first load
3. **Async Processing**: Consider queue-based processing for high volume

### Monitoring
1. **Health Checks**: Use `/health` endpoint for monitoring
2. **Logging**: Implement structured logging for all inspections
3. **Metrics**: Track inspection rates, scores, and decisions

## 📝 Maintenance

### Update Models

To retrain a model with new data:

```bash
# Retrain specific category
python train.py --category bottle

# Reload model in running application
curl -X POST "http://localhost:8000/model/reload/bottle"
```

### Backup

Important files to backup:
- `models/patchcore/` - Trained model weights
- Configuration files
- Inspection history (if using database)

## 🐛 Troubleshooting

### Backend won't start
- Check if port 8000 is available
- Verify all dependencies are installed
- Check `models/` directory exists with trained models

### CUDA out of memory
- Set `DEVICE="cpu"` in config.py
- Reduce `IMAGE_SIZE`
- Close other GPU applications

### Poor detection accuracy
- Verify training data contains only defect-free images
- Check threshold is appropriate for your use case
- Consider retraining with more data

### Slow inference
- Use GPU instead of CPU
- Reduce image size in config
- Reduce coreset sampling ratio

## 📞 Support

For issues and questions:
1. Check this documentation
2. Review API docs at `/docs`
3. Check training/evaluation metrics
4. Verify sample visualizations work correctly

## 🎓 Training New Categories

To add support for a new product category:

1. **Prepare Data Structure**:
   ```
   mvtec_anomaly_detection/
   └── new_category/
       ├── train/
       │   └── good/        # Normal images only
       └── test/
           ├── good/        # Normal test images
           └── defect_type/ # Defect images
   ```

2. **Train Model**:
   ```bash
   python train.py --category new_category
   ```

3. **Evaluate**:
   ```bash
   python evaluate.py --category new_category
   ```

4. **Add to Config**:
   Add category name to `CATEGORIES` list in `config.py`

5. **Test**:
   Use the web interface to upload and inspect images

---

**InspectAI v1.0** | Industrial Visual Inspection System
