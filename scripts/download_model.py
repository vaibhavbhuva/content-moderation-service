# download_model.py
"""
Script to download and save the language detection model locally.
Run this once to download the model.
"""

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def download_model(model_name: str = "ZheYu03/xlm-r-langdetect-model", 
                   save_directory: str = "./models/xlm-roberta-langdetect"):
    """
    Download and save the model locally.
    
    Args:
        model_name: HuggingFace model name
        save_directory: Local directory to save the model
    """
    print(f"📥 Downloading model: {model_name}")
    print(f"📁 Save directory: {save_directory}")
    
    # Create directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)
    
    try:
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_directory)
        
        # Download model
        print("Downloading model weights...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.save_pretrained(save_directory)
        
        # Save model config info
        config_info = {
            "model_name": model_name,
            "num_labels": model.config.num_labels,
            "model_type": model.config.model_type,
        }
        
        import json
        with open(os.path.join(save_directory, "download_info.json"), "w") as f:
            json.dump(config_info, f, indent=2)
        
        print("✅ Model downloaded successfully!")
        print(f"📊 Model size: {get_dir_size(save_directory):.2f} MB")
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        raise

def get_dir_size(path):
    """Get directory size in MB."""
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_dir_size(entry.path)
    return total / (1024 * 1024)

if __name__ == "__main__":
    download_model()
