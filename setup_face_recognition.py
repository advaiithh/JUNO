"""
Quick setup script for face recognition
Automatically detects if re-registration is needed
"""
import os
import pickle

print("="*70)
print("FACE RECOGNITION SETUP CHECK")
print("="*70)

# Check if face_recognition is installed
try:
    import face_recognition
    print("\n✓ face_recognition library: INSTALLED")
    print("  Using deep learning model (dlib ResNet)")
    has_face_rec = True
except ImportError:
    print("\n❌ face_recognition library: NOT INSTALLED")
    print("   Run: pip install face-recognition")
    has_face_rec = False

# Check registration file
if os.path.exists("registered_faces_advanced.pkl"):
    with open("registered_faces_advanced.pkl", 'rb') as f:
        data = pickle.load(f)
    
    was_created_with_face_rec = data.get("use_face_recognition", False)
    
    print(f"\n✓ Found registration file")
    print(f"  - Number of samples: {len(data['encodings'])}")
    print(f"  - Created with deep learning: {was_created_with_face_rec}")
    
    if has_face_rec and not was_created_with_face_rec:
        print("\n" + "!"*70)
        print("⚠️  ACTION REQUIRED: RE-REGISTRATION NEEDED")
        print("!"*70)
        print("\nYour face was registered using the FALLBACK method (weak features).")
        print("Now that face_recognition is installed, you need to RE-REGISTER")
        print("to use the DEEP LEARNING model for accurate recognition.")
        print("\nSteps:")
        print("1. Delete old file: del registered_faces_advanced.pkl")
        print("2. Run: python recognition_advanced.py")
        print("3. Choose option 1 to re-register")
        print("4. Capture 10 samples looking directly at camera")
        print("5. Test with option 2")
        
        delete = input("\nDelete old registration and start fresh? (y/n): ").strip().lower()
        if delete == 'y':
            os.remove("registered_faces_advanced.pkl")
            print("\n✓ Deleted old registration file")
            print("\nNow run: python recognition_advanced.py")
            print("And register your face (option 1)")
        else:
            print("\nOld file kept. Re-register manually when ready.")
    
    elif has_face_rec and was_created_with_face_rec:
        print("\n✓ EVERYTHING IS GOOD!")
        print("  Your registration uses the deep learning model.")
        print("  Just run: python recognition_advanced.py")
    
    elif not has_face_rec:
        print("\n⚠️  Install face_recognition to use deep learning:")
        print("   pip install face-recognition")
else:
    print("\n❌ No registration file found")
    if has_face_rec:
        print("\n✓ Ready to register!")
        print("  Run: python recognition_advanced.py")
        print("  Choose option 1 to register your face")
    else:
        print("\n⚠️  First install: pip install face-recognition")
        print("  Then run: python recognition_advanced.py")

print("\n" + "="*70)
