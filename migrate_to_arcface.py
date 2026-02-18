"""
ArcFace Migration Script
========================

This script helps you migrate from the old OpenCV-based recognition
to the new ArcFace-based recognition with 512-D embeddings.

What it does:
1. Backs up your old registration (if it exists)
2. Deletes the old registration file
3. Prompts you to re-register with ArcFace

Usage:
    python migrate_to_arcface.py
"""

import os
import shutil
from datetime import datetime

def migrate_to_arcface():
    print("=" * 60)
    print("ArcFace Migration Tool")
    print("=" * 60)
    print()
    
    OLD_FILE = "registered_faces_advanced.pkl"
    BACKUP_DIR = "backups"
    
    # Check if old registration exists
    if not os.path.exists(OLD_FILE):
        print("✓ No old registration found. You're ready to register with ArcFace!")
        print()
        print("Next step: Run 'python recognition_advanced.py' and choose option 1")
        return
    
    # Get info about old registration
    file_size = os.path.getsize(OLD_FILE)
    modified_time = datetime.fromtimestamp(os.path.getmtime(OLD_FILE))
    
    print(f"Found old registration file:")
    print(f"  File: {OLD_FILE}")
    print(f"  Size: {file_size:,} bytes")
    print(f"  Created: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Ask for confirmation
    print("⚠️  WARNING:")
    print("  This file was created with the OLD recognition method (800-D features)")
    print("  To use ArcFace (512-D), you need to RE-REGISTER your face")
    print()
    print("What happens:")
    print("  1. Old registration will be backed up to 'backups/' folder")
    print("  2. Old registration will be deleted")
    print("  3. You'll need to run recognition_advanced.py and choose option 1")
    print()
    
    response = input("Do you want to migrate to ArcFace? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print()
        print("Migration cancelled. No changes made.")
        print()
        print("Note: You can still use the old registration, but you won't")
        print("      benefit from ArcFace's superior accuracy.")
        return
    
    # Create backup directory
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        print(f"✓ Created backup directory: {BACKUP_DIR}/")
    
    # Create backup with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f"{BACKUP_DIR}/registered_faces_advanced_backup_{timestamp}.pkl"
    
    print()
    print("Creating backup...")
    shutil.copy2(OLD_FILE, backup_name)
    print(f"✓ Backup created: {backup_name}")
    
    # Delete old registration
    print("Deleting old registration...")
    os.remove(OLD_FILE)
    print(f"✓ Deleted: {OLD_FILE}")
    
    print()
    print("=" * 60)
    print("✓ Migration complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run: python recognition_advanced.py")
    print("  2. Choose option 1 (Register face)")
    print("  3. System will use ArcFace to capture 512-D embeddings")
    print("  4. Enjoy superior accuracy and security!")
    print()
    print("Benefits of ArcFace:")
    print("  ✓ 512-dimensional embeddings (vs 800-D)")
    print("  ✓ State-of-the-art accuracy (95%)")
    print("  ✓ Better rejection of non-owners")
    print("  ✓ Higher confidence scores")
    print()

if __name__ == "__main__":
    try:
        migrate_to_arcface()
    except KeyboardInterrupt:
        print()
        print()
        print("Migration cancelled by user.")
    except Exception as e:
        print()
        print(f"✗ Error during migration: {e}")
        print()
        print("Your original registration file has NOT been modified.")
