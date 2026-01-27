# InspectAI - Render.com Deployment Guide

Complete step-by-step guide to deploy InspectAI on Render.com for FREE!

## 🎯 What You'll Get

- **Free hosting** (750 hours/month)
- **Automatic HTTPS/SSL**
- **Public URL**: `https://inspectai-xxxx.onrender.com`
- **Auto-deploy** from GitHub commits
- **No credit card required**

## ⚠️ Important Notes

1. **Cold Starts**: Free tier apps sleep after 15 min of inactivity. First request takes 30-60 seconds to wake up.
2. **CPU Only**: No GPU on free tier (inference will be slower but functional)
3. **Build Time**: First deployment takes 10-15 minutes

---

## 📋 Prerequisites

✅ GitHub account
✅ Trained model files in `models/patchcore/`
✅ All code ready (already prepared!)

---

## 🚀 Step-by-Step Deployment

### **Step 1: Push Code to GitHub**

Open terminal in your project directory and run:

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Deploy InspectAI to Render"

# Set main branch
git branch -M main

# Add your remote
git remote add origin https://github.com/Padmanabh03/Inspect_AI.git

# Push to GitHub
git push -u origin main
```

**Note**: The `.gitignore` file will automatically exclude the large dataset files, so only code and models will be uploaded.

---

### **Step 2: Sign Up on Render**

1. Go to [render.com](https://render.com)
2. Click "Get Started"
3. Sign up with GitHub (recommended)
4. Authorize Render to access your repositories

---

### **Step 3: Create New Web Service**

1. Click "New +" button (top right)
2. Select "Web Service"
3. Click "Connect" next to your `Inspect_AI` repository
4. If you don't see it, click "Configure account" to grant access

---

### **Step 4: Configure Service**

Fill in the following settings:

**Name:** `inspectai` (or your preferred name)

**Region:** Choose closest to you (e.g., Oregon, Frankfurt)

**Branch:** `main`

**Root Directory:** (leave blank)

**Runtime:** `Docker`

**Instance Type:** `Free`

**Environment Variables:** (Click "Add Environment Variable")
```
DEVICE=cpu
```

**Auto-Deploy:** ✓ Yes (checked)

---

### **Step 5: Deploy**

1. Click "Create Web Service" at the bottom
2. Render will start building your Docker image
3. **Wait 10-15 minutes** for first build (grab a coffee! ☕)
4. Watch the build logs for progress

---

### **Step 6: Get Your URL**

Once deployed, you'll see:
- **Status**: "Live" (green)
- **URL**: `https://inspectai-xxxx.onrender.com`

Click the URL to access your app!

---

## 🎨 Using Your Deployed App

### Access the Frontend
```
https://inspectai-xxxx.onrender.com/
```

### Access API Documentation
```
https://inspectai-xxxx.onrender.com/docs
```

### API Endpoints
```
POST https://inspectai-xxxx.onrender.com/inspect
GET  https://inspectai-xxxx.onrender.com/categories
GET  https://inspectai-xxxx.onrender.com/health
```

---

## 🔧 Testing the Deployment

### Test 1: Health Check
```bash
curl https://inspectai-xxxx.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "device": "cpu",
  "cuda_available": false
}
```

### Test 2: Get Categories
```bash
curl https://inspectai-xxxx.onrender.com/categories
```

### Test 3: Inspect Image
Upload an image through the web interface at your Render URL!

---

## 🐛 Troubleshooting

### Problem: Build Failed

**Check the logs** in Render dashboard:
- Look for error messages
- Common issue: Missing model files
- Solution: Ensure `models/` folder was pushed to GitHub

### Problem: App Shows "Application Failed to Respond"

**Cause**: App is still waking up from sleep
**Solution**: Wait 30-60 seconds and refresh

### Problem: "Model not found" Error

**Cause**: Model files not in repository
**Solution**: 
1. Check if `models/patchcore/bottle/` exists locally
2. Ensure it's not in `.gitignore`
3. Push again:
```bash
git add models/
git commit -m "Add model files"
git push
```

### Problem: Slow Inference

**Cause**: CPU-only inference on free tier
**Expected**: 2-5 seconds per image
**Solution**: Upgrade to paid tier with better CPU

---

## 📊 Performance Expectations (Free Tier)

| Metric | Value |
|--------|-------|
| Cold Start | 30-60 seconds |
| Warm Inference | 2-5 seconds/image |
| Concurrent Requests | 1-2 (limited) |
| Monthly Hours | 750 hours free |

---

## 🔄 Updating Your Deployment

To update your app after making changes:

```bash
# Make your changes locally
# ...

# Commit and push
git add .
git commit -m "Update: description of changes"
git push

# Render will auto-deploy in 5-10 minutes!
```

---

## 💰 Upgrading (Optional)

If you need better performance:

**Starter Plan ($7/month):**
- No cold starts
- Better CPU
- More memory
- Faster inference

**To upgrade:**
1. Go to Render dashboard
2. Select your service
3. Click "Upgrade"
4. Choose "Starter"

---

## 🔐 Security Best Practices

### For Production Use:

1. **Add Authentication**
   - Implement API key verification
   - Add user login system

2. **Configure CORS**
   - Update `backend/app.py` to allow only your domain
   - Change `allow_origins=["*"]` to specific domains

3. **Environment Variables**
   - Store sensitive config in Render environment variables
   - Don't commit secrets to GitHub

4. **Rate Limiting**
   - Add rate limiting to prevent abuse
   - Use middleware or API gateway

---

## 📝 Custom Domain (Optional)

To use your own domain:

1. Buy domain (e.g., `inspectai.com`)
2. In Render dashboard:
   - Go to your service
   - Click "Settings" → "Custom Domain"
   - Add your domain
3. Update DNS records as shown by Render
4. SSL certificate is automatic!

---

## 🎉 Success Checklist

- [x] Code pushed to GitHub
- [x] Render service created
- [x] Build completed successfully
- [x] App is "Live"
- [x] Frontend accessible via URL
- [x] Can upload and inspect images
- [x] Models load correctly
- [x] Visualizations display properly

---

## 📞 Need Help?

**Render Documentation**: https://render.com/docs
**Render Community**: https://community.render.com

**Common Issues**:
- Check Render logs (in dashboard)
- Verify all files pushed to GitHub
- Ensure model files exist
- Wait for cold start to complete

---

## 🌟 What's Next?

Now that your app is deployed:

1. **Share your URL** with others!
2. **Test with different products**
3. **Monitor usage** in Render dashboard
4. **Consider upgrading** if you need better performance
5. **Add custom domain** for professional look

---

**Congratulations! 🎉**
Your InspectAI application is now live on the internet!

---

*InspectAI v1.0 - Developed by Padmanabh Butala*
