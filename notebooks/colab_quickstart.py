"""
Colab Quickstart Setup Script
===============================

Run this at the start of your Colab session to configure the environment.

Usage in Colab:
    %run notebooks/colab_quickstart.py
"""

import os
import sys
from pathlib import Path


def setup_colab_environment():
    """Configure Sequence environment for Google Colab."""
    print("="*60)
    print("Sequence Trading System - Colab Setup")
    print("="*60)

    # Get repository root
    ROOT = Path.cwd()
    while not (ROOT / "CLAUDE.md").exists() and ROOT.parent != ROOT:
        ROOT = ROOT.parent

    if not (ROOT / "CLAUDE.md").exists():
        print("❌ ERROR: Could not find Sequence repository root!")
        print("   Make sure you're in the Sequence directory")
        return False

    print(f"\n✅ Found repository root: {ROOT}")

    # Fix NumPy compatibility (common Colab issue)
    print("\n" + "="*60)
    print("Checking NumPy compatibility...")
    print("="*60)

    try:
        import numpy as np
        # Try importing a compiled extension to test compatibility
        from numpy.random import RandomState  # noqa: F401
        print("✅ NumPy compatibility check passed")
    except ValueError as e:
        if "numpy.dtype size changed" in str(e):
            print("⚠️  NumPy binary incompatibility detected!")
            print("   This is a common Colab issue when packages were compiled")
            print("   against a different NumPy version.\n")
            print("🔧 SOLUTION:")
            print("   1. Run: !pip install --upgrade numpy pandas scikit-learn scipy --quiet")
            print("   2. Restart runtime: Runtime → Restart runtime")
            print("   3. Re-run this setup script\n")
            print("❌ Setup cannot continue until NumPy is fixed.")
            return False
        else:
            raise
    except ImportError:
        print("⚠️  NumPy not installed. Run: !pip install -r requirements.txt")

    # Setup Python path
    paths_to_add = [str(ROOT), str(ROOT / "run")]
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
            print(f"✅ Added to sys.path: {path}")

    # Change to repository root
    os.chdir(ROOT)
    print(f"✅ Changed directory to: {ROOT}")

    # Verify imports work
    print("\n" + "="*60)
    print("Verifying imports...")
    print("="*60)

    try:
        from config.config import DataConfig, ModelConfig, TrainingConfig  # noqa: F401
        print("✅ config.config imports successful")
    except ImportError as e:
        print(f"❌ config.config import failed: {e}")
        return False

    try:
        from utils.logger import get_logger  # noqa: F401
        print("✅ utils.logger imports successful")
    except ImportError as e:
        print(f"❌ utils.logger import failed: {e}")
        return False

    try:
        from train.features.agent_features import build_feature_frame  # noqa: F401
        print("✅ train.features imports successful")
    except ImportError as e:
        print(f"❌ train.features import failed: {e}")
        return False

    # Check for GPU
    print("\n" + "="*60)
    print("Hardware Configuration")
    print("="*60)

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU Available: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
        else:
            print("⚠️  No GPU detected. Consider changing runtime type:")
            print("   Runtime → Change runtime type → GPU")
    except ImportError:
        print("⚠️  PyTorch not installed yet. Run: !pip install -r requirements.txt")

    # Create data directories
    print("\n" + "="*60)
    print("Creating data directories...")
    print("="*60)

    data_dirs = [
        ROOT / "data" / "data",
        ROOT / "models",
        ROOT / "output_central" / "cache",
    ]

    for dir_path in data_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path.relative_to(ROOT)}")

    # Check for Google Drive
    drive_mount = Path("/content/drive")
    if drive_mount.exists():
        print(f"\n✅ Google Drive is mounted at {drive_mount}")
        print("   Consider using Drive for data persistence:")
        print("   os.environ['SEQUENCE_DATA_DIR'] = '/content/drive/MyDrive/Sequence/data'")
    else:
        print("\n💡 TIP: Mount Google Drive for data persistence:")
        print("   from google.colab import drive")
        print("   drive.mount('/content/drive')")

    # Print summary
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\n🚀 You can now run Sequence commands:")
    print("   !python data/prepare_dataset.py --pairs gbpusd --t-in 120 --t-out 10")
    print("   !python run/training_pipeline.py --pairs gbpusd --epochs 10")
    print("\n📖 See COLAB_SETUP.md for more examples and workflows")
    print("="*60 + "\n")

    return True


if __name__ == "__main__":
    success = setup_colab_environment()
    sys.exit(0 if success else 1)
