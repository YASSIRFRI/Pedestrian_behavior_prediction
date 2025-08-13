#!/usr/bin/env python3
"""
Download only the required models for behavior classifier.
Run this script on the login node with internet access.
"""

import os
import torch
from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    AutoProcessor, AutoModelForZeroShotObjectDetection,
    pipeline
)

def download_models():
    """Download only the required models to cache."""
    
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    print(f"📁 Models will be cached to: {cache_dir}")
    
    # Set offline mode to False for downloading
    os.environ.pop('HF_HUB_OFFLINE', None)
    os.environ.pop('TRANSFORMERS_OFFLINE', None)
    
    print("🚀 Starting model downloads...")
    
    # 1. BLIP-2 OPT 2.7B (main model you need)
    try:
        print(f"\n📥 Downloading BLIP-2 OPT 2.7B...")
        print(f"   Model ID: Salesforce/blip2-opt-2.7b")
        
        # Download processor
        print("   📦 Downloading BLIP-2 processor...")
        blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        
        # Download model
        print("   🤖 Downloading BLIP-2 model...")
        blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
        
        print(f"   ✅ BLIP-2 OPT 2.7B downloaded successfully!")
        
        # Clean up memory
        del blip2_processor, blip2_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"   ❌ Failed to download BLIP-2 OPT 2.7B: {e}")
        return False
    
    # 2. Grounding DINO (for object detection)
    try:
        print(f"\n📥 Downloading Grounding DINO...")
        print(f"   Model ID: IDEA-Research/grounding-dino-tiny")
        
        # Download processor
        print("   📦 Downloading Grounding DINO processor...")
        gd_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        
        # Download model
        print("   🤖 Downloading Grounding DINO model...")
        gd_model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
        
        print(f"   ✅ Grounding DINO downloaded successfully!")
        
        # Clean up memory
        del gd_processor, gd_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"   ❌ Failed to download Grounding DINO: {e}")
        return False
    
    # 3. BART (for zero-shot classification) - THIS WAS MISSING!
    try:
        print(f"\n📥 Downloading BART Large MNLI...")
        print(f"   Model ID: facebook/bart-large-mnli")
        
        print("   🔤 Downloading BART classifier...")
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        print(f"   ✅ BART Large MNLI downloaded successfully!")
        
        # Clean up memory
        del classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"   ❌ Failed to download BART: {e}")
        return False
    
    print(f"\n🎉 All required models downloaded successfully!")
    print(f"📁 Models cached in: {cache_dir}")
    print(f"💡 You can now run your code on GPU nodes without internet")
    return True

def check_specific_models():
    """Check if the specific models we need are cached."""
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    transformers_cache = os.path.join(cache_dir, "transformers")
    
    print(f"🔍 Checking for required models in: {cache_dir}")
    
    required_models = [
        "blip2-opt-2.7b",
        "grounding-dino-tiny", 
        "bart-large-mnli"
    ]
    found_models = []
    missing_models = []
    
    if os.path.exists(transformers_cache):
        cached_items = os.listdir(transformers_cache)
        
        for required in required_models:
            found = False
            for cached in cached_items:
                if required.replace("-", "_") in cached.lower() or required.replace("_", "-") in cached.lower():
                    found_models.append(required)
                    found = True
                    break
            if not found:
                missing_models.append(required)
    else:
        missing_models = required_models
    
    print(f"✅ Found models: {found_models}")
    print(f"❌ Missing models: {missing_models}")
    
    return len(missing_models) == 0

def test_offline_loading():
    """Test loading models in offline mode."""
    print(f"\n🧪 Testing offline model loading...")
    
    # Set offline mode
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    try:
        # Test BLIP-2
        print("   Testing BLIP-2...")
        blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", local_files_only=True)
        print("   ✅ BLIP-2 processor loads offline")
        del blip2_processor
        
        # Test Grounding DINO
        print("   Testing Grounding DINO...")
        gd_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny", local_files_only=True)
        print("   ✅ Grounding DINO processor loads offline")
        del gd_processor
        
        # Test BART
        print("   Testing BART...")
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", local_files_only=True)
        print("   ✅ BART classifier loads offline")
        del classifier
        
        print("   🎉 All models work in offline mode!")
        return True
        
    except Exception as e:
        print(f"   ❌ Offline test failed: {e}")
        return False
    finally:
        # Remove offline mode
        os.environ.pop('HF_HUB_OFFLINE', None)
        os.environ.pop('TRANSFORMERS_OFFLINE', None)

if __name__ == "__main__":
    print("🤖 BLIP-2 Behavior Classifier Model Downloader")
    print("=" * 50)
    print("📋 Required models:")
    print("   1. Salesforce/blip2-opt-2.7b (BLIP-2 main model)")
    print("   2. IDEA-Research/grounding-dino-tiny (object detection)")
    print("   3. facebook/bart-large-mnli (zero-shot classification)")
    print()
    
    # Check current cache
    all_cached = check_specific_models()
    
    if not all_cached:
        print(f"\n🌐 Starting downloads...")
        success = download_models()
        
        if success:
            print(f"\n📊 Final check:")
            check_specific_models()
            
            print(f"\n🧪 Testing offline mode:")
            test_offline_loading()
        else:
            print(f"\n❌ Download failed!")
            exit(1)
    else:
        print(f"\n✅ All required models already cached!")
        print(f"\n🧪 Testing offline mode:")
        test_offline_loading()
    
    print(f"\n💡 Usage on GPU node:")
    print(f"   export HF_HUB_OFFLINE=1")
    print(f"   export TRANSFORMERS_OFFLINE=1")
    print(f"   python main.py")