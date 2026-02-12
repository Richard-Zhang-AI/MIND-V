#!/bin/bash

# MIND-V Model Download Script
# This script downloads all required models for MIND-V

set -e

echo "🚀 MIND-V Model Download Script"
echo "================================"

# Create directories
echo "📁 Creating directories..."
mkdir -p ckpts/CogVideoX-Fun-V1.5-5b-InP
mkdir -p ckpts/MIND-V
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

download_file "https://dl.fbaipublicfiles.com/vjepa2/vitl.pt" \
    "ckpts/vjepa2/vitl16_256px.pt" \
    "V-JEPA2 ViT-L/16 (300M) - 256px"

download_file "https://dl.fbaipublicfiles.com/vjepa2/vitg.pt" \
    "ckpts/vjepa2/vitg16_256px.pt" \
    "V-JEPA2 ViT-g/16 (1B) - 256px"

download_file "https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt" \
    "ckpts/vjepa2/vitg16_384px.pt" \
    "V-JEPA2 ViT-g/16 (1B) - 384px"

download_file "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt" \
    "ckpts/vjepa2/vjepa2_ac_vitg.pt" \
    "V-JEPA2 Action-Conditioned (Robotics)"

echo ""
echo "🎯 Downloading V-JEPA2 Evaluation Probes..."
download_file "https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitg-384-64x2x3.pt" \
    "ckpts/vjepa2/ssv2_probe_vitg384.pt" \
    "V-JEPA2 SSV2 Evaluation Probe"

download_file "https://dl.fbaipublicfiles.com/vjepa2/evals/ek100-vitg-384.pt" \
    "ckpts/vjepa2/ek100_probe_vitg384.pt" \
    "V-JEPA2 EK100 Action Anticipation Probe"

# Download Affordance-R1 models
echo ""
echo "🤖 Downloading Affordance-R1 Models..."
echo "=================================="

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
    echo "Please install it: pip install \"huggingface_hub[cli]\""
    echo "Or download manually from: https://huggingface.co/hqking/affordance-r1"
fi

# Download Qwen2.5-VL-7B base model for Affordance-R1
echo ""
echo "📝 Checking for Qwen2.5-VL-7B (required for Affordance-R1)..."
if [ ! -d "ckpts/affordance-r1/qwen2.5-vl-7b" ]; then
    if command -v huggingface-cli >/dev/null 2>&1; then
        echo "📥 Downloading Qwen2.5-VL-7B for Affordance-R1..."
        huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
            --local-dir ckpts/affordance-r1/qwen2.5-vl-7b \
            --local-dir-use-symlinks False
        echo "✅ Qwen2.5-VL-7B downloaded successfully!"
    else
        echo "⚠️  huggingface-cli not found. Please install it."
        echo "Manual download: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    fi
else
    echo "✅ Qwen2.5-VL-7B already exists"
fi

# Download CogVideoX
echo ""
echo "🎥 Downloading CogVideoX-Fun-V1.5-5b-InP..."
if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download THUDM/CogVideoX-Fun-V1.5-5b-InP \
        --local-dir ckpts/CogVideoX-Fun-V1.5-5b-InP \
        --local-dir-use-symlinks False
    echo "✅ CogVideoX downloaded successfully!"
else
    echo "⚠️  huggingface-cli not found."
    echo "Please install: pip install \"huggingface_hub[cli]\""
    echo "Or download manually: https://huggingface.co/THUDM/CogVideoX-Fun-V1.5-5b-InP"
fi

# Download MIND-V finetuned checkpoints
echo ""
echo "🧠 Downloading MIND-V Fine-tuned Checkpoints"
echo "============================================="
echo "Repository: https://huggingface.co/Richard-ZZZZZ/MIND-V"

if command -v huggingface-cli >/dev/null 2>&1; then
    echo ""
    echo "📥 Downloading MIND-V using HuggingFace CLI..."
    huggingface-cli download Richard-ZZZZZ/MIND-V \
        --local-dir ckpts/MIND-V \
        --local-dir-use-symlinks False
    echo "✅ MIND-V finetuned model downloaded successfully!"
else
    echo ""
    echo "⚠️  huggingface-cli not found for MIND-V download."
    echo "Please install it with:"
    echo "   pip install \"huggingface_hub[cli]\""
    echo ""
    echo "Then run:"
    echo "   huggingface-cli download Richard-ZZZZZ/MIND-V --local-dir ./ckpts/MIND-V"
    echo ""
    echo "Alternative (Python way):"
    echo "   from huggingface_hub import snapshot_download"
    echo "   snapshot_download(repo_id='Richard-ZZZZZ/MIND-V', local_dir='ckpts/MIND-V')"
fi

# Check disk space
echo ""
echo "💾 Checking disk space..."
AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
echo "Available disk space: ${AVAILABLE_SPACE}GB"

if [ "$AVAILABLE_SPACE" -lt 50 ]; then
    echo "⚠️  Warning: Less than 50GB available space."
    echo "   Models require ~47GB for complete setup."
fi

# File sizes and requirements (updated MIND-V size estimate)
echo ""
echo "📋 Model Size Information:"
echo "=========================="
echo "SAM2 Tiny:                    ~100MB  ✅ Downloaded"
echo "CogVideoX-5B:                 ~10GB"
echo "MIND-V Fine-tuned:            ~15GB   (estimated)"
echo "V-JEPA2 Models:               ~8GB   ✅ Downloaded"
echo "Affordance-R1 Models:         ~14GB"
echo "Total Required:                ~47GB"

# Verification script (updated path for MIND-V)
echo ""
echo "🔍 Creating verification script..."
cat > verify_models.py << 'EOF'
#!/usr/bin/env python3
"""
Verify that all required models are properly downloaded
"""

import os
from pathlib import Path

def check_model_exists(path, description):
    if os.path.exists(path):
        if os.path.isfile(path):
            size = os.path.getsize(path) / (1024*1024)  # MB
            print(f"✅ {description}: {size:.1f} MB")
        elif os.path.isdir(path):
            files = list(Path(path).rglob('*'))
            print(f"✅ {description}: {len(files)} files in directory")
        return True
    else:
        print(f"❌ {description}: not found at {path}")
        return False

def main():
    print("🔍 Verifying Model Downloads")
    print("=" * 40)

    models = [
        ("ckpts/sam2/sam2.1_hiera_tiny.pt", "SAM2 Hiera Tiny"),
        ("ckpts/CogVideoX-Fun-V1.5-5b-InP", "CogVideoX Base"),
        ("ckpts/MIND-V", "MIND-V Fine-tuned"),
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
        print("You can now run the inference / quick_start script")
    else:
        print("⚠️  Some models are missing")
        print("Please re-run ./download_models.sh or check the error messages")

if __name__ == "__main__":
    main()
EOF

chmod +x verify_models.py

echo ""
echo "✅ Download script completed!"
echo ""
echo "📋 Summary:"
echo "=========="
echo "- SAM2 model:               ✅ Downloaded"
echo "- V-JEPA2 models:           ✅ Downloaded"
echo "- Affordance-R1 models:     ⏳ Check huggingface-cli"
echo "- CogVideoX model:          ⏳ Check huggingface-cli"
echo "- MIND-V fine-tuned:        ⏳ Now supports huggingface-cli download"
echo ""
echo "📝 Next steps:"
echo "1. Install huggingface-cli if needed:"
echo "   pip install \"huggingface_hub[cli]\""
echo "2. Re-run this script if any part failed"
echo "3. Verify models:"
echo "   python verify_models.py"
echo "4. Test:"
echo "   python quick_start.py   (or your inference script)"
