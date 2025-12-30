#!/usr/bin/env python3
"""
Download checkpoints from Google Drive
This script downloads the model weights needed for the SoccernetGSR pipeline
from a specified Google Drive folder.
"""

import os
import sys
import yaml
import shutil

try:
    import gdown
except ImportError:
    print("gdown is required for downloading from Google Drive.")
    print("Please install it with: pip install gdown")
    sys.exit(1)

# Google Drive Folder URL
# https://drive.google.com/drive/u/0/folders/1kgZGxUYGkYhM9AwHzjuSVo7r7FFZ16mh
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1kgZGxUYGkYhM9AwHzjuSVo7r7FFZ16mh"

def load_config(config_path):
    """Load the YAML configuration file."""
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return None
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            print(f"❌ Error parsing config file: {e}")
            return None

def get_required_weights(config):
    """Extract required weight files from the config."""
    weights = set()
    
    # Sections to check for MODEL_PATH
    sections = ['TRACKER', 'REID', 'SFR', 'GTA', 'LLAMA', 'CLIP']
    
    for section in sections:
        if section in config and 'MODEL_PATH' in config[section]:
            path = config[section]['MODEL_PATH']
            # We only care about the filename, assuming they are all in checkpoints/
            # But the config includes 'checkpoints/' prefix usually.
            weights.add(path)
            
    return list(weights)

def download_weights(destination_dir):
    """Download weights from Google Drive folder."""
    print(f"Downloading weights from Google Drive to '{destination_dir}'...")
    print(f"Source: {DRIVE_FOLDER_URL}")
    
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print(f"Created directory: {destination_dir}")
    
    try:
        # gdown.download_folder downloads the contents of the folder
        files = gdown.download_folder(url=DRIVE_FOLDER_URL, output=destination_dir, quiet=False, use_cookies=False)
        
        if not files:
            print("⚠️ No files were downloaded. The folder might be empty or not accessible.")
            print("Please ensure the Google Drive folder is shared with 'Anyone with the link'.")
            return False
            
        print(f"✅ Downloaded {len(files)} files/folders.")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def verify_weights(required_weights, base_dir="."):
    """Verify that required weights exist."""
    print("\nVerifying downloaded weights...")
    missing = []
    
    for weight_path in required_weights:
        full_path = os.path.join(base_dir, weight_path)
        if os.path.exists(full_path):
            print(f"  ✅ Found: {weight_path}")
        else:
            print(f"  ❌ Missing: {weight_path}")
            missing.append(weight_path)
            
    if missing:
        print(f"\n⚠️ {len(missing)} required weight file(s) are missing.")
        return False
    
    print("\n✅ All required weights are present!")
    return True

def main():
    config_path = "configs/config.yaml"
    
    print("--- SoccernetGSR Weight Downloader ---")
    
    # 1. Load Config
    config = load_config(config_path)
    if not config:
        return 1
        
    # 2. Identify required weights
    required_weights = get_required_weights(config)
    print(f"Identified {len(required_weights)} required weight files from config.")
    
    # 3. Download weights
    # The config paths are like 'checkpoints/model.pth'. 
    # We want to download into 'checkpoints' directory.
    # gdown.download_folder(output='checkpoints') will put files inside 'checkpoints'.
    
    # Check if we should download
    print("\nInitiating download...")
    success = download_weights("checkpoints")
    
    if not success:
        print("\n❌ Download process encountered issues.")
        # We continue to verification anyway, in case files were already there
    
    # 4. Verify
    if verify_weights(required_weights):
        print("\n🎉 Setup complete! You are ready to run the pipeline.")
        return 0
    else:
        print("\nPlease manually check the Google Drive folder and ensure all files are present.")
        print(f"Drive Link: {DRIVE_FOLDER_URL}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
