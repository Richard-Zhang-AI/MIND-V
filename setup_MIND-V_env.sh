#!/bin/bash

# MIND-V Environment Setup Script
# For RoboVideo Environment Setup

set -e

echo "🚀 Setting up MIND-V environment for RoboVideo..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python $python_version detected. Python 3.8+ is required."
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv robovideo_env
source robovideo_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CUDA version)
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install requirements_robovideo.txt

# Install additional dependencies for specific components
echo "🔧 Installing additional dependencies..."

# For SAM2
echo "📦 Installing SAM2..."
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# For xformers (memory efficient attention)
echo "⚡ Installing xformers..."
pip install xformers

# For flash attention (optional, requires CUDA)
echo "💡 Installing flash attention (optional)..."
pip install flash-attn --no-build-isolation || echo "⚠️  Flash attention installation failed (this is optional)"

# Install development tools
echo "🛠️  Installing development tools..."
pip install black flake8 isort pytest

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p ckpts
mkdir -p output
mkdir -p temp_files
mkdir -p logs

echo "✅ Environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Activate the environment: source robovideo_env/bin/activate"
echo "2. Download model checkpoints and place them in ckpts/"
echo "3. Update config.py with your model paths"
echo "4. Run: python long_horizon_video_pipeline.py --help"
echo ""
echo "📋 Important notes:"
echo "- Make sure you have CUDA 11.8+ installed"
echo "- Some models may require specific versions"
echo "- Check config.py for configuration options"
echo ""
echo "🐛 If you encounter issues:"
echo "- Try installing without flash attention: comment out the flash-attn line"
echo "- For CPU-only setup, install torch-cpu instead"
echo "- Check the requirements.txt for version compatibility"

# Optional: Test installation
echo ""
echo "🧪 Testing installation..."
python -c "
import torch
import diffusers
import transformers
import cv2
import numpy as np
print('✅ Core packages imported successfully')
print(f'🔥 PyTorch version: {torch.__version__}')
print(f'⚡ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'🎮 GPU: {torch.cuda.get_device_name()}')
print('🎉 Installation test passed!')
" || echo "⚠️  Some packages may not be installed correctly"