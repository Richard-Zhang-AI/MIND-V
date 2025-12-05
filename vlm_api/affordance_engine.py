#!/usr/bin/env python3
"""
AffordanceEngine: In-process Affordance-R1 reasoning + SAM2 segmentation

Loads the Qwen2.5-VL reasoning model once and reuses it across calls.
Given an image and a natural-language affordance query, it produces
pixel points and a binary mask by prompting SAM2 with bbox+point.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image as PILImage
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class AffordanceEngine:
    def __init__(self, reasoning_model_path: str, sam2_checkpoint: str = None, device: str = "auto") -> None:
        """
        初始化 AffordanceEngine。

        设计目标：
        - 优先在 GPU 上加载 Qwen2.5-VL 推理模型；
        - 避免在 CPU 上启用 FlashAttention2 导致的错误；
        - 在当前进程中复用同一个模型与 SAM2 预测器。
        """
        dtype = torch.bfloat16

        # 优先使用 GPU；若当前环境无法访问 CUDA，则直接抛出异常，让上层显式感知
        if device == "auto":
            if torch.cuda.is_available():
                device_map = "cuda"
            else:
                raise RuntimeError(
                    "AffordanceEngine requires a CUDA-capable device, "
                    "but torch.cuda.is_available() is False in the current environment."
                )
        else:
            device_map = device

        # 为了最大兼容性，这里不强制使用 FlashAttention2，
        # 由 Transformers 根据环境选择合适的注意力实现（如 sdpa/eager）。
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            reasoning_model_path,
            dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(reasoning_model_path, padding_side="left")

        # 内部初始化SAM2
        self.sam2_predictor = None
        if sam2_checkpoint:
            self._init_sam2_predictor(sam2_checkpoint)

    def _init_sam2_predictor(self, sam2_checkpoint: str):
        """内部初始化SAM2预测器"""
        try:
            # 添加SAM2路径
            sam2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sam2')
            if sam2_path not in sys.path:
                sys.path.insert(0, sam2_path)

            # 保存当前工作目录并切换到SAM2目录
            original_cwd = os.getcwd()
            try:
                os.chdir(sam2_path)

                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                # 根据checkpoint推断配置
                ckpt_name = os.path.basename(sam2_checkpoint)
                is_v21 = "2.1" in ckpt_name

                if is_v21:
                    base_cfg = "configs/sam2.1/"
                    if "tiny" in ckpt_name:
                        cfg = base_cfg + "sam2.1_hiera_t.yaml"
                    elif "small" in ckpt_name:
                        cfg = base_cfg + "sam2.1_hiera_s.yaml"
                    elif "base_plus" in ckpt_name or "b+" in ckpt_name:
                        cfg = base_cfg + "sam2.1_hiera_b+.yaml"
                    elif "large" in ckpt_name:
                        cfg = base_cfg + "sam2.1_hiera_l.yaml"
                    else:
                        cfg = base_cfg + "sam2.1_hiera_t.yaml"
                else:
                    base_cfg = "configs/sam2/"
                    if "tiny" in ckpt_name:
                        cfg = base_cfg + "sam_hiera_t.yaml"
                    elif "small" in ckpt_name:
                        cfg = base_cfg + "sam_hiera_s.yaml"
                    elif "base_plus" in ckpt_name or "b+" in ckpt_name:
                        cfg = base_cfg + "sam_hiera_b+.yaml"
                    elif "large" in ckpt_name:
                        cfg = base_cfg + "sam_hiera_l.yaml"
                    else:
                        cfg = base_cfg + "sam_hiera_t.yaml"

                sam_model = build_sam2(config_file=cfg, ckpt_path=sam2_checkpoint)
                self.sam2_predictor = SAM2ImagePredictor(sam_model, mask_threshold=-1.0)

            finally:
                os.chdir(original_cwd)

        except Exception as e:
            print(f"Warning: Failed to initialize SAM2 predictor: {e}")
            self.sam2_predictor = None

    @staticmethod
    def _question_template() -> str:
        return (
            "Please answer \"{Question}\" with bboxs and points."
            "Analyze the functional properties of specific parts of each object in the image and carefully find all the part(s) that matches the problem."
            "Output the thinking process in <think> </think>, rethinking process in <rethink> </rethink> and final answer in <answer> </answer> tags."
            "Output the bbox(es) and point(s) and affordance tpye(s) inside the interested object(s) in JSON format."
            "i.e., <think> thinking process here </think>,"
            "<rethink> rethinking process here </rethink>,"
            "<answer>{Answer}</answer>"
        )

    @staticmethod
    def _extract_bbox_points(text: str, x_factor: float, y_factor: float) -> Tuple[List[List[int]], List[List[int]]]:
        import re, json
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
        pred_bboxes, pred_points = [], []

        # Debug: 打印原始模型输出
        print(f"🔍 DEBUG - Raw model output:")
        print(f"Full text: {repr(text)}")

        if json_match:
            answer_content = json_match.group(1)
            print(f"🔍 DEBUG - Extracted answer content: {repr(answer_content)}")

            # 尝试解析JSON，如果有错误则尝试修复
            try:
                data = json.loads(answer_content)
                print(f"🔍 DEBUG - Parsed JSON successfully: {data}")
            except json.JSONDecodeError as e:
                print(f"🔍 DEBUG - JSON parsing failed: {e}")
                print(f"🔍 DEBUG - Content that failed to parse: {repr(answer_content)}")

                # 直接使用手动数字提取，这是最可靠的方法
                print(f"🔧 DEBUG - Attempting to extract numbers manually...")
                data = []

                # 查找所有数字（更宽松的模式，允许包含}的错误）
                # 先尝试正常格式
                bbox_pattern_normal = r'"bbox_2d":\s*\[(\d+),(\d+),(\d+),(\d+)\]'
                point_pattern_normal = r'"point_2d":\s*\[(\d+),(\d+)\]'
                # 再尝试带}的错误格式
                bbox_pattern_error = r'"bbox_2d":\s*\[(\d+),(\d+),(\d+),(\d+)\}]'  # 带}的错误格式
                point_pattern_error = r'"point_2d":\s*\[(\d+),(\d+)\}]'  # 带}的错误格式

                # 先尝试匹配正常格式
                bbox_match = re.search(bbox_pattern_normal, answer_content)
                point_match = re.search(point_pattern_normal, answer_content)

                # 如果正常格式没匹配到，尝试带}的错误格式
                if not bbox_match:
                    bbox_match = re.search(bbox_pattern_error, answer_content)
                if not point_match:
                    point_match = re.search(point_pattern_error, answer_content)

                if bbox_match and point_match:
                    bbox = [int(bbox_match.group(i)) for i in range(1, 5)]
                    point = [int(point_match.group(i)) for i in range(1, 3)]
                    data = [{"bbox_2d": bbox, "point_2d": point}]
                    print(f"🔧 DEBUG - Extracted data manually: {data}")
                else:
                    print(f"🔧 DEBUG - bbox_match: {bbox_match}")
                    print(f"🔧 DEBUG - point_match: {point_match}")
                    raise ValueError("Could not extract bbox and point data from malformed JSON")

            if data:
                for item in data:
                    if 'bbox_2d' in item:
                        bx = item['bbox_2d']
                        pred_bboxes.append([
                            int(bx[0] * x_factor + 0.5),
                            int(bx[1] * y_factor + 0.5),
                            int(bx[2] * x_factor + 0.5),
                            int(bx[3] * y_factor + 0.5),
                        ])
                    if 'point_2d' in item:
                        pt = item['point_2d']
                        pred_points.append([
                            int(pt[0] * x_factor + 0.5),
                            int(pt[1] * y_factor + 0.5),
                        ])
        else:
            print(f"🔍 DEBUG - No <answer> tags found in output")

        return pred_bboxes, pred_points

    def infer(self, image_path: str, text: str, sam2_predictor=None, multimask_output: bool = False) -> Dict:
        """
        Run affordance reasoning to produce bbox/point, then use SAM2 to segment.

        Args:
            image_path: path to RGB image
            text: natural language query (e.g., affordance prompt)
            sam2_predictor: an initialized SAM2ImagePredictor (optional, will use internal one if None)
            multimask_output: pass-through to SAM2

        Returns:
            {success, mask: np.ndarray[bool], points_px: List[List[int]], bboxes_px: List[List[int]]}
        """
        # 使用内部SAM2预测器，如果没有提供外部的话
        if sam2_predictor is None:
            sam2_predictor = self.sam2_predictor
            if sam2_predictor is None:
                raise RuntimeError("No SAM2 predictor available. Either provide one or initialize AffordanceEngine with sam2_checkpoint.")
        try:
            image = PILImage.open(image_path).convert("RGB")
            original_width, original_height = image.size
            resize_size = 840
            x_factor, y_factor = original_width / resize_size, original_height / resize_size

            # Build chat message
            question = text.lower().strip('.')
            tpl = self._question_template()
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image.resize((resize_size, resize_size), PILImage.BILINEAR)},
                    {"type": "text", "text": tpl.format(Question=question, Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}]")},
                ],
            }]

            # Processor
            text_inputs = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
            image_inputs, video_inputs = process_vision_info([messages])
            inputs = self.processor(text=text_inputs, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

            # Generate
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            # Parse to pixel coords
            bboxes_px, points_px = self._extract_bbox_points(output_text, x_factor, y_factor)

            # Validate that we got valid detection
            if not bboxes_px or not points_px:
                raise RuntimeError(f"No valid bbox or point detected in affordance inference. Text output: {output_text}")

            # Segment with SAM2
            mask_all = np.zeros((original_height, original_width), dtype=bool)
            sam2_predictor.set_image(image)
            for bbox, point in zip(bboxes_px, points_px):
                masks, scores, _ = sam2_predictor.predict(
                    point_coords=np.array([point], dtype=np.float32),
                    point_labels=np.array([1], dtype=np.int32),
                    box=np.array(bbox, dtype=np.float32),
                    multimask_output=multimask_output,
                    return_logits=False,
                    normalize_coords=True,
                )
                # pick best
                best = int(np.argmax(scores)) if scores.ndim else 0
                cur = masks[best].astype(bool)
                mask_all = np.logical_or(mask_all, cur)

            # Validate mask is not empty
            if not np.any(mask_all):
                raise RuntimeError("Generated mask is empty - no object segmented")

            return {"success": True, "mask": mask_all, "points_px": points_px, "bboxes_px": bboxes_px}

        except Exception as e:
            return {"success": False, "error": str(e)}
