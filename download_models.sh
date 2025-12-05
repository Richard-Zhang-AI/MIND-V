#!/bin/bash

# MIND-V Model Download Script
# This script downloads all required models for MIND-V

set -e

echo "🚀 MIND-V Model Download Script"
echo "================================"

# Create directories
echo "📁 Creating directories..."
mkdir -p ckpts/CogVideoX-Fun-V1.5-5b-InP
mkdir -p ckpts/MIND_V
mkdir -p ckpts/sam2
mkdir -p ckpts/affordance-r1/huggingface
mkdir -p ckpts/vjepa2

# Function to download with progress bar
download_file() {
    local url="$1"
    local output="$2"
    local description="$3"

    echo "📥 Downloading: $description"

    if command -v wget >/dev/null 2>&1; then
        wget --progress=bar:force -O "$output" "$url"
    elif command -v curl >/dev/null 2>&1; then
        curl -L --progress-bar -o "$output" "$url"
    else
        echo "❌ Neither wget nor curl found. Please install one of them."
        exit 1
    fi

    if [ $? -eq 0 ]; then
        echo "✅ Successfully downloaded: $description"
    else
        echo "❌ Failed to download: $description"
        exit 1
    fi
}

# Download SAM2 (always available)
echo ""
echo "📦 Downloading SAM2..."
download_file "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt" \
    "ckpts/sam2/sam2.1_hiera_tiny.pt" \
    "SAM2 Hiera Tiny Model"

# Download V-JEPA2 models
echo ""
echo "🧠 Downloading V-JEPA2 World Models..."
echo "======================================"

# Download V-JEPA2 ViT-L/16 (300M parameters, 256px resolution)
download_file "https://dl.fbaipublicfiles.com/vjepa2/vitl.pt" \
    "ckpts/vjepa2/vitl16_256px.pt" \
    "V-JEPA2 ViT-L/16 (300M) - 256px"

# Download V-JEPA2 ViT-g/16 (1B parameters, best for PFC)
download_file "https://dl.fbaipublicfiles.com/vjepa2/vitg.pt" \
    "ckpts/vjepa2/vitg16_256px.pt" \
    "V-JEPA2 ViT-g/16 (1B) - 256px"

# Download V-JEPA2 ViT-g/16 384px (higher resolution for better accuracy)
download_file "https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt" \
    "ckpts/vjepa2/vitg16_384px.pt" \
    "V-JEPA2 ViT-g/16 (1B) - 384px"

# Download V-JEPA2 Action-Conditioned model (for robotics)
download_file "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt" \
    "ckpts/vjepa2/vjepa2_ac_vitg.pt" \
    "V-JEPA2 Action-Conditioned (Robotics)"

echo ""
echo "🎯 Downloading V-JEPA2 Evaluation Probes..."
# Download Something-Something v2 probe for motion understanding
download_file "https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitg-384-64x2x3.pt" \
    "ckpts/vjepa2/ssv2_probe_vitg384.pt" \
    "V-JEPA2 SSV2 Evaluation Probe"

# Download EK100 action anticipation probe
download_file "https://dl.fbaipublicfiles.com/vjepa2/evals/ek100-vitg-384.pt" \
    "ckpts/vjepa2/ek100_probe_vitg384.pt" \
    "V-JEPA2 EK100 Action Anticipation Probe"

# Download Affordance-R1 models
echo ""
echo "🤖 Downloading Affordance-R1 Models..."
echo "=================================="

# Check for huggingface CLI for Affordance-R1
if command -v huggingface-cli >/dev/null 2>&1; then
    echo ""
    echo "📥 Downloading Affordance-R1 using HuggingFace CLI..."
    huggingface-cli download hqking/affordance-r1 \
        --local-dir ckpts/affordance-r1/huggingface \
        --local-dir-use-symlinks False
    echo "✅ Affordance-R1 model downloaded successfully!"
else
    echo ""
    echo "⚠️  huggingface-cli not found for Affordance-R1 download."
    echo "Please install it: pip install huggingface_hub"
    echo "Or download manually from: https://huggingface.co/hqking/affordance-r1"

    # Manual download instructions for Affordance-R1
    echo ""
    echo "🔄 Alternative Affordance-R1 Download Methods:"
    echo "1. Using Python:"
    echo "   from huggingface_hub import snapshot_download"
    echo "   snapshot_download('hqking/affordance-r1', "
    echo "                   local_dir='ckpts/affordance-r1/huggingface')"
fi

# Download Qwen2.5-VL-7B base model for Affordance-R1 (if needed)
echo ""
echo "📝 Checking for Qwen2.5-VL-7B (required for Affordance-R1)..."
if [ ! -d "ckpts/affordance-r1/qwen2.5-vl-7b" ]; then
    echo "📥 Downloading Qwen2.5-VL-7B for Affordance-R1..."
    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
            --local-dir ckpts/affordance-r1/qwen2.5-vl-7b \
            --local-dir-use-symlinks False
        echo "✅ Qwen2.5-VL-7B downloaded successfully!"
    else
        echo "⚠️  Qwen2.5-VL-7B requires manual download:"
        echo "   https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    fi
else
    echo "✅ Qwen2.5-VL-7B already exists"
fi

# Check for huggingface CLI
if command -v huggingface-cli >/dev/null 2>&1; then
    echo ""
    echo "📥 Downloading CogVideoX model using HuggingFace CLI..."
    huggingface-cli download THUDM/CogVideoX-Fun-V1.5-5b-InP \
        --local-dir ckpts/CogVideoX-Fun-V1.5-5b-Inp \
        --local-dir-use-symlinks False

    echo "✅ CogVideoX model downloaded successfully!"
else
    echo ""
    echo "⚠️  huggingface-cli not found. Please install it:"
    echo "   pip install huggingface_hub"
    echo ""
    echo "Or download manually from:"
    echo "   https://huggingface.co/THUDM/CogVideoX-Fun-V1.5-5b-InP"
fi

# MIND-V model (placeholder - needs actual URL)
echo ""
echo "🤖 MIND-V Fine-tuned Model"
echo "================================"
echo "⚠️  MIND-V model download requires access to the trained weights."
echo ""
echo "Please:"
echo "1. Contact the authors for model access"
echo "2. Or train your own model using the provided training scripts"
echo "3. Place the model files in: ckpts/MIND_V/"
echo ""
echo "Expected files in ckpts/MIND_V/:"
echo "  - config.json"
echo "  - diffusion_pytorch_model.bin"
echo "  - [other model-specific files]"

# Check disk space
echo ""
echo "💾 Checking disk space..."
AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
echo "Available disk space: ${AVAILABLE_SPACE}GB"

if [ "$AVAILABLE_SPACE" -lt 50 ]; then
    echo "⚠️  Warning: Less than 50GB available space."
    echo "   Models require ~47GB for complete setup."
    echo "   Consider freeing up disk space."
fi

# File sizes and requirements
echo ""
echo "📋 Model Size Information:"
echo "=========================="
echo "SAM2 Tiny:                    ~100MB  ✅ Downloaded"
echo "CogVideoX-5B:                 ~10GB"
echo "MIND-V Fine-tuned:            ~15GB"
echo "V-JEPA2 Models:               ~8GB   ✅ Downloaded"
echo "  - ViT-L/16 (300M):          ~1.2GB"
echo "  - ViT-g/16 (1B):            ~3.8GB"
echo "  - ViT-g/16 384px:           ~3.9GB"
echo "  - Action-Conditioned:      ~3.9GB"
echo "  - Evaluation Probes:       ~500MB"
echo "Affordance-R1 Models:         ~14GB"
echo "  - Main Model:              ~7GB"
echo "  - Qwen2.5-VL-7B Base:       ~7GB"
echo "Total Required:                ~47GB"

# Verification script
echo ""
echo "🔍 Creating verification script..."
cat > verify_models.py << 'EOF'
#!/usr/bin/env python3
"""
Verify that all required models are properly downloaded
"""

import os
import hashlib
from pathlib import Path

def check_model_exists(path, description):
    """Check if a model file/directory exists"""
    if os.path.exists(path):
        if os.path.isfile(path):
            size = os.path.getsize(path) / (1024*1024)  # MB
            print(f"✅ {description}: {size:.1f} MB")
        elif os.path.isdir(path):
            files = list(Path(path).rglob('*'))
            print(f"✅ {description}: {len(files)} files")
        return True
    else:
        print(f"❌ {description}: {path}")
        return False

def main():
    print("🔍 Verifying Model Downloads")
    print("=" * 40)

    models = [
        ("ckpts/sam2/sam2.1_hiera_tiny.pt", "SAM2 Hiera Tiny"),
        ("ckpts/CogVideoX-Fun-V1.5-5b-Inp", "CogVideoX Base"),
        ("ckpts/MIND_V", "MIND-V Fine-tuned"),
        ("ckpts/vjepa2/vitl16_256px.pt", "V-JEPA2 ViT-L/16"),
        ("ckpts/vjepa2/vitg16_256px.pt", "V-JEPA2 ViT-g/16"),
        ("ckpts/vjepa2/vitg16_384px.pt", "V-JEPA2 ViT-g/16 384px"),
        ("ckpts/vjepa2/vjepa2_ac_vitg.pt", "V-JEPA2 Action-Conditioned"),
        ("ckpts/vjepa2/ssv2_probe_vitg384.pt", "V-JEPA2 SSV2 Probe"),
        ("ckpts/vjepa2/ek100_probe_vitg384.pt", "V-JEPA2 EK100 Probe"),
        ("ckpts/affordance-r1/huggingface", "Affordance-R1"),
        ("ckpts/affordance-r1/qwen2.5-vl-7b", "Qwen2.5-VL-7B")
    ]

    all_exist = True
    for path, desc in models:
        if not check_model_exists(path, desc):
            all_exist = False

    print("\n" + "=" * 40)
    if all_exist:
        print("🎉 All models verified successfully!")
        print("You can now run the quick_start.py script")
    else:
        print("⚠️  Some models are missing")
        print("Please run ./download_models.sh again")

if __name__ == "__main__":
    main()
EOF

chmod +x verify_models.py

echo ""
echo "✅ Download script completed!"
echo ""
echo "📋 Summary:"
echo "=========="
echo "- SAM2 model: ✅ Downloaded"
echo "- V-JEPA2 models: ✅ Downloaded"
echo "- Affordance-R1 models: ⏳ Check huggingface-cli installation"
echo "- CogVideoX model: ⏳ Check huggingface-cli installation"
echo "- MIND-V model: ❌ Requires manual download"
echo ""
echo "📝 Next steps:"
echo "1. Install huggingface-cli if not installed:"
echo "   pip install huggingface_hub"
echo ""
echo "2. Run this script again if needed:"
echo "   ./download_models.sh"
echo ""
echo "3. Download MIND-V model (see instructions above)"
echo ""
echo "4. Verify your setup:"
echo "   python verify_models.py"
echo ""
echo "5. Test your installation:"
echo "   python quick_start.py"

# Check if we need to show alternatives
echo ""
echo "🔄 Alternative Download Methods:"
echo "==============================="
echo ""
echo "If huggingface-cli doesn't work, try:"
echo ""
echo "1. Using wget/curl manually:"
echo "   wget https://huggingface.co/THUDM/CogVideoX-Fun-V1.5-5b-InP/resolve/main/config.json"
echo "   # Download all required files"
echo ""
echo "2. Using Python:"
echo "   from huggingface_hub import snapshot_download"
echo "   snapshot_download('THUDM/CogVideoX-Fun-V1.5-5b-Inp', "
echo "                   local_dir='ckpts/CogVideoX-Fun-V1.5-5b-Inp')"
echo ""
echo "3. Using ModelScope (for users in China):"
echo "   pip install modelscope"
echo "   from modelscope import snapshot_download"
echo "   snapshot_download('THUDM/CogVideoX-Fun-V1.5-5b-InP')"

echo ""
echo "📚 For detailed instructions, see MODEL_DOWNLOAD.md"