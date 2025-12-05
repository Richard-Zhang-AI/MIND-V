"""
LoRA工具函数
用于处理LoRA模型的创建、合并和取消合并
"""

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
from typing import Optional, Union


def create_network(
    model,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.0,
    target_modules: Optional[list] = None,
    **kwargs
):
    """
    为给定模型创建LoRA网络

    参数:
        model: 要应用LoRA的模型
        lora_rank: LoRA的秩
        lora_alpha: LoRA的alpha参数
        lora_dropout: LoRA的dropout率
        target_modules: 要应用LoRA的目标模块列表
        **kwargs: 其他参数

    返回:
        带有LoRA的模型
    """
    if target_modules is None:
        # 默认目标模块，适用于transformer模型
        target_modules = [
            "to_k", "to_q", "to_v", "to_out.0",
            "add_k_proj", "add_q_proj", "add_v_proj", "to_add_out",
        ]

    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )

    return get_peft_model(model, config)


def merge_lora(pipeline, lora_path: str, multiplier: float = 1.0):
    """
    将LoRA权重合并到pipeline中

    参数:
        pipeline: 要合并LoRA的pipeline
        lora_path: LoRA模型路径
        multiplier: LoRA权重的乘数

    返回:
        合并了LoRA的pipeline
    """
    if lora_path == "none" or lora_path is None:
        return pipeline

    try:
        # 如果pipeline是PEFT模型，直接加载和合并
        if hasattr(pipeline.transformer, 'load_adapter'):
            pipeline.transformer.load_adapter(lora_path, adapter_name="default")
            # 设置适配器缩放
            pipeline.transformer.set_adapter("default")
            if hasattr(pipeline.transformer, 'set_adapters_multiplier'):
                pipeline.transformer.set_adapters_multiplier(["default"], [multiplier])
            return pipeline

        # 否则，尝试作为PEFT模型加载
        if hasattr(pipeline, 'transformer'):
            pipeline.transformer = PeftModel.from_pretrained(
                pipeline.transformer,
                lora_path,
                adapter_name="default"
            )
            pipeline.transformer.set_adapter("default")
            if hasattr(pipeline.transformer, 'set_adapters_multiplier'):
                pipeline.transformer.set_adapters_multiplier(["default"], [multiplier])

        return pipeline

    except Exception as e:
        print(f"Warning: Failed to merge LoRA from {lora_path}: {e}")
        return pipeline


def unmerge_lora(pipeline, lora_path: str, multiplier: float = 1.0):
    """
    从pipeline中取消合并LoRA权重

    参数:
        pipeline: 要取消合并LoRA的pipeline
        lora_path: LoRA模型路径
        multiplier: LoRA权重的乘数

    返回:
        取消合并后的pipeline
    """
    if lora_path == "none" or lora_path is None:
        return pipeline

    try:
        # 如果有适配器，禁用它
        if hasattr(pipeline.transformer, 'disable_adapter'):
            pipeline.transformer.disable_adapter()
        elif hasattr(pipeline.transformer, 'set_adapter'):
            # 设置为无适配器状态
            pipeline.transformer.set_adapter([])

        return pipeline

    except Exception as e:
        print(f"Warning: Failed to unmerge LoRA from {lora_path}: {e}")
        return pipeline


def save_lora_weights(model, save_path: str):
    """
    保存LoRA权重

    参数:
        model: 带有LoRA的模型
        save_path: 保存路径
    """
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(save_path)
    else:
        # 如果是PEFT包装的模型
        unwrapped_model = model
        if hasattr(model, 'base_model'):
            unwrapped_model = model.base_model

        if hasattr(unwrapped_model, 'save_pretrained'):
            unwrapped_model.save_pretrained(save_path)


def load_lora_weights(model, lora_path: str):
    """
    加载LoRA权重

    参数:
        model: 要加载LoRA的模型
        lora_path: LoRA权重路径

    返回:
        加载了LoRA的模型
    """
    try:
        return PeftModel.from_pretrained(model, lora_path)
    except Exception as e:
        print(f"Warning: Failed to load LoRA weights from {lora_path}: {e}")
        return model