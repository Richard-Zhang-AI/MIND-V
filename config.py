"""
Project Configuration File
For paper submission - all paths are relative or use placeholders
"""

import os

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODEL_CONFIG = {
    "base_model_path": "ckpts/CogVideoX-Fun-V1.5-5b-InP",
    "trained_model_path": "ckpts/MIND_V",
    "lora_model_path": "flow_grpo/checkpoints/latest/lora/adapter_model.safetensors",
    "sam2_checkpoint": "ckpts/sam2/sam2.1_hiera_tiny.pt",
    "affordance_model": "ckpts/affordance-r1/huggingface"
}

# Data paths
DATA_CONFIG = {
    "input_dir": "demos",
    "output_dir": "output",
    "temp_dir": "temp_files",
    "log_dir": "logs"
}

# VLM API paths
VLM_CONFIG = {
    "gemini_credentials": "vlm_api/gemini_credentials.json",
    "sam2_path": "sam2",
    "yoloe_path": "yoloe"
}

# System configuration
SYSTEM_CONFIG = {
    "device": "cuda" if os.system("command -v nvidia-smi") == 0 else "cpu",
    "seed": 42,
    "num_workers": 4
}

# Inference parameters
INFERENCE_CONFIG = {
    "num_inference_steps": 20,
    "guidance_scale": 7.5,
    "height": 720,
    "width": 1280,
    "num_frames": 49
}