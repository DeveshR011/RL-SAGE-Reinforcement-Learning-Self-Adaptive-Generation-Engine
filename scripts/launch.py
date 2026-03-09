"""
rl_sage/scripts/launch.py

One-click training launcher for RL-SAGE.

Auto-detects VRAM and selects the appropriate model and batch size constraints.
Wraps `scripts/train.py` and logs launch metadata.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

def check_vram() -> float:
    """Return available VRAM in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / 1e9
    except ImportError:
        pass
    return 0.0

def generate_launch_overrides(vram_gb: float) -> list:
    """Generate command-line arguments to override config based on hardware."""
    args = []
    
    if vram_gb == 0.0:
        print("⚠️ No CUDA VRAM detected. Forcing DEBUG mode with CPU models.")
        args.append("--debug")
        # In debug mode, we'll let train.py pick distilgpt2
        return args

    print(f"CUDA VRAM detected: {vram_gb:.1f} GB")
    
    # Very constrained (e.g., 6 GB or less)
    if vram_gb < 7.0:
        print("⚠️ Constrained VRAM (< 7 GB).")
        print("Launcher does not auto-edit config; if you hit OOM, switch base_model to TinyLlama in config/training_config.yaml.")
    else:
        print("✅ Sufficient VRAM for Phi-2 (2.7B).")

    return args

def log_launch(vram_gb: float, args: list):
    """Save metadata about this training run."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    launch_doc = {
        "timestamp": datetime.now().isoformat(),
        "vram_gb": vram_gb,
        "launch_args": args,
        "python_version": sys.version,
    }
    
    with open(log_dir / "launch_history.jsonl", "a") as f:
        f.write(json.dumps(launch_doc) + "\n")

def main():
    print("=" * 60)
    print(" RL-SAGE Auto-Launcher")
    print("=" * 60)
    
    # 1. Run Setup checks
    print("\n[1/3] Running environment setup...")
    setup_script = Path(__file__).parent / "setup.py"
    res = subprocess.run([sys.executable, str(setup_script)])
    if res.returncode != 0:
        print("❌ Setup failed. Aborting launch.")
        sys.exit(1)
        
    # 2. Hardware constraints
    print("\n[2/3] Analyzing hardware constraints...")
    vram_gb = check_vram()
    dynamic_args = generate_launch_overrides(vram_gb)
    
    # 3. Launch training
    print("\n[3/3] Launching training process...")
    train_script = Path(__file__).parent / "train.py"
    
    cmd = [sys.executable, str(train_script), "--config", "config/training_config.yaml"] + dynamic_args
    
    # Add any user-provided args (e.g., --debug passed to launch.py)
    user_args = []
    for i in range(1, len(sys.argv)):
        user_args.append(sys.argv[i])
        
    for arg in user_args:
        arg_str = str(arg)
        if arg_str not in cmd:
            cmd.append(arg_str)
            
    print(f"\n🚀 Executing: {' '.join(cmd)}\n")
    log_launch(vram_gb, cmd)
    
    try:
        # Use Popen to stream output
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        process.kill()
        sys.exit(0)

if __name__ == "__main__":
    main()
