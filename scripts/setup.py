"""
rl_sage/scripts/setup.py

Environment bootstrapper:
  - Validates Python version and CUDA availability.
  - Creates required directory structure.
  - Pre-downloads GSM8K and ARC-Easy datasets.
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Configure basic logging for setup
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

required_dirs = [
    "data",
    "data/gsm8k",
    "data/arc",
    "data/humaneval",
    "checkpoints",
    "logs",
]

def check_environment():
    """Verify python version and torch/cuda."""
    logger.info("Checking environment...")
    
    # Python version
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        logger.error(f"Python 3.9+ required, found {v.major}.{v.minor}")
        sys.exit(1)
        
    # Torch / CUDA
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1e9
            logger.info(f"CUDA available: {props.name} ({vram_gb:.1f} GB VRAM)")
            if vram_gb < 5.5:
                logger.warning(
                    "⚠️ Your GPU has less than 6 GB VRAM! "
                    "Make sure to use TinyLlama instead of Phi-2, and ensure "
                    "no other applications (like browsers) are using the GPU."
                )
        else:
            logger.warning("No CUDA detected. Training will be extremely slow on CPU.")
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

def create_directories():
    """Create local data/log directories."""
    logger.info("Setting up directories...")
    root = Path(__file__).parent.parent
    for d in required_dirs:
        path = root / d
        if not path.exists():
            path.mkdir(parents=True)
            logger.info(f"  Created: {path}")

def download_datasets():
    """Download datasets via HuggingFace."""
    logger.info("Downloading datasets to local cache...")
    try:
        from datasets import load_dataset
        
        # 1. GSM8K
        logger.info("  Downloading GSM8K (main)...")
        load_dataset("openai/gsm8k", "main")
        
        # 2. ARC-Easy
        logger.info("  Downloading AI2 ARC (ARC-Easy)...")
        load_dataset("allenai/ai2_arc", "ARC-Easy")
        
        logger.info("✓ Datasets downloaded successfully.")
    except ImportError:
        logger.error("datasets library not installed. Run: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to download datasets: {e}")
        logger.info("You can still proceed; datasets will attempt to download during training.")

def main():
    print("=" * 50)
    print(" RL-SAGE Environment Setup ")
    print("=" * 50)
    
    check_environment()
    create_directories()
    download_datasets()
    
    print("=" * 50)
    print(" READY TO TRAIN ✓")
    print(" You can now run: python scripts/launch.py")
    print("=" * 50)

if __name__ == "__main__":
    main()
