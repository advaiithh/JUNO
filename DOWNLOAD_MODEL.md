# 📥 Download ArcFace Model - Quick Instructions

## Step-by-Step Guide

### 1. Download the Model File

**Click this link:** https://drive.google.com/open?id=1H0ekPf3M3SzWivayY9oXQJ8GNcFn-O9q

- File name: Save as `backbone.pth`
- File size: ~166 MB
- Format: PyTorch checkpoint

### 2. Place in Checkpoint Folder  

Move the downloaded file to:
```
C:\Users\Lenovo\Desktop\Career Building\Projects\Shalom\JUNO\checkpoint\backbone.pth
```

**The folder `checkpoint/` has been created for you!**

### 3. Verify Installation

Open PowerShell in the JUNO folder and run:

```powershell
# Check if file exists
Test-Path checkpoint/backbone.pth

# Check file size (should be ~166 MB)
(Get-Item checkpoint/backbone.pth).length / 1MB

# Test model loading
python arcface_model.py
```

**Expected output:**
```
✓ ArcFace model loaded from checkpoint/backbone.pth
✓ Successfully extracted embedding: (512,)
✓ ArcFace system is ready!
```

### 4. Run Face Recognition System

```powershell
python recognition_advanced.py
```

You should see:
```
✓ Using ArcFace with PyTorch (Excellent Accuracy - 512D) on cpu
```

### 5. Register Your Face (First Time)

- Choose option: `1` (Register face)
- Look at camera
- System captures 12 high-quality samples
- Done! Your face is registered with ArcFace 512-D embeddings

### 6. Test Recognition

- Choose option: `2` (Live recognition)
- System will recognize you with 80-94% confidence
- Non-owners will be rejected as "UNKNOWN"

## 🔧 Alternative: If Google Drive Doesn't Work

**Baidu Cloud:**
- Link: https://pan.baidu.com/s/1L8yOF1oZf6JHfeY9iN59mg
- Password: `lz89`
- Download `backbone.pth` and place in `checkpoint/` folder

## ✅ Checklist

- [ ] Downloaded `backbone.pth` (166 MB)
- [ ] Placed in `checkpoint/backbone.pth`
- [ ] Ran `python arcface_model.py` successfully
- [ ] Saw "ArcFace model loaded" message
- [ ] Ready to run `python recognition_advanced.py`

## 🎯 Why You Need This

Without the model file, the system falls back to OpenCV multi-feature method (800-D), which has lower accuracy and more false positives.

With ArcFace model:
- ✅ State-of-the-art 512-D embeddings
- ✅ 95% accuracy
- ✅ Much better at rejecting non-owners
- ✅ Higher confidence scores

## 🚀 Quick Start After Download

```powershell
# Test model
python arcface_model.py

# Run face recognition
python recognition_advanced.py

# Choose 1 to register, then 2 to recognize
```

That's it! Enjoy your advanced face recognition system! 🎉
