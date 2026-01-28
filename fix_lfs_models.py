#!/usr/bin/env python3
"""
Fix Git LFS model files by replacing pointers with actual binary data.
This script checks if model files are LFS pointers and converts them to actual files.
"""

import os
import subprocess
import sys

def is_lfs_pointer(filepath):
    """Check if a file is a Git LFS pointer"""
    try:
        with open(filepath, 'rb') as f:
            first_line = f.readline()
            # LFS pointers start with "version https://git-lfs.github.com/spec/v1"
            return first_line.startswith(b'version https://git-lfs.github.com')
    except:
        return False

def fix_lfs_files():
    """Find and fix LFS pointer files"""
    models_dir = 'models/patchcore'
    
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return
    
    print("Checking for LFS pointer files...")
    
    fixed_count = 0
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.endswith('.npy'):
                filepath = os.path.join(root, file)
                if is_lfs_pointer(filepath):
                    print(f"Found LFS pointer: {filepath}")
                    print("Attempting to pull actual file...")
                    
                    try:
                        # Try to pull the specific file
                        result = subprocess.run(
                            ['git', 'lfs', 'pull', '--include', filepath],
                            capture_output=True,
                            text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__))
                        )
                        
                        if result.returncode == 0:
                            print(f"✓ Successfully pulled: {filepath}")
                            fixed_count += 1
                        else:
                            print(f"✗ Failed to pull: {filepath}")
                            print(f"Error: {result.stderr}")
                    except Exception as e:
                        print(f"✗ Error pulling {filepath}: {e}")
                else:
                    print(f"✓ Already binary: {filepath}")
    
    if fixed_count > 0:
        print(f"\nFixed {fixed_count} LFS pointer file(s)")
    else:
        print("\nNo LFS pointers found or unable to fix them")
    
    return fixed_count > 0

if __name__ == '__main__':
    success = fix_lfs_files()
    sys.exit(0 if success else 1)
