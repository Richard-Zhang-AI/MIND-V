#!/usr/bin/env python3
"""
通用智能机器人控制系统 (Universal Intelligent Robot Controller)
适用于任意场景的物体识别、分割和轨迹规划

子任务结构约定（与任务分解保持一致）：
- 高层子任务聚焦“目标效果”，而非底层执行细节；例如：
  - 正确："把黄柄勺子放入金属锅中"、"把蓝色抹布放入金属锅中"、"把金属锅移出画面"
  - 避免："抓取X" / "移动到Y上方" / "放下X" 等微步骤拆分
- 每个子任务输入到控制器时，控制器内部再完成“识别+分割+轨迹规划”等细节，不需要上游拆分为抓取/移动/放置。
- 此约定确保：上游任务分解输出与下游控制器能力解耦，子任务数量更少、语义更清晰。

通用工作流程：
1. 解析用户指令，提取操作对象和移动指令（高层语义）
2. 执行操作对象识别（Affordance + SAM2/YOLOE 兜底）
3. 执行轨迹规划（自动完成抓取/搬运/放置等细节）
4. 输出完整的分割和轨迹数据
"""

import os
import sys
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
import math
import random
from typing import Tuple, Optional, List, Dict
from loguru import logger

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_api.affordance_engine import AffordanceEngine
from vlm_api.object_movement_planner import ObjectMovementPlanner

class UniversalRobotController:
    def __init__(self,
                 sam2_checkpoint="/data/rczhang/MIND-V/ckpts/sam2/sam2.1_hiera_tiny.pt",
                 sam2_config="configs/sam2.1/sam2.1_hiera_t.yaml",
                 sam2_mask_threshold: float = -1.0,
                 output_dir="./universal_robot_output",
                 aff_reasoning_model="/data/rczhang/MIND-V/ckpts/affordance-r1/huggingface"):
        """
        初始化简化版机器人控制器 - 仅使用Affordance视觉定位

        Args:
            sam2_checkpoint: SAM2模型权重路径
            sam2_config: SAM2配置文件路径
            sam2_mask_threshold: SAM2掩码阈值
            output_dir: 输出目录
            aff_reasoning_model: Affordance推理模型路径
        """
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.sam2_mask_threshold = sam2_mask_threshold
        self.output_dir = output_dir
        self.aff_reasoning_model = aff_reasoning_model

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        self.movement_planner = ObjectMovementPlanner(
            output_dir=os.path.join(output_dir, "movement")
        )

        # 设置日志
        logger.add(f"{output_dir}/universal_robot_control.log", rotation="10 MB")
        logger.info("Simplified robot controller initialized with Affordance-only positioning")

        # 缓存 SAM2 预测器用于细化（避免重复加载）
        self._sam2_predictor = None
        # 缓存 Affordance 引擎（避免重复加载 Qwen 模型）
        self._aff_engine = None

    def _get_sam2_predictor(self):
        """
        懒加载并缓存 SAM2 预测器，用于基于先验 mask 的细化。
        """
        if self._sam2_predictor is not None:
            return self._sam2_predictor

        try:
            # 添加SAM2路径到系统路径
            sam2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sam2')
            if sam2_path not in sys.path:
                sys.path.insert(0, sam2_path)

            # 检查checkpoint文件是否存在
            if not os.path.exists(self.sam2_checkpoint):
                raise FileNotFoundError(f"SAM2 checkpoint file not found: {self.sam2_checkpoint}")

            # 保存当前工作目录
            original_cwd = os.getcwd()

            try:
                # 切换到SAM2目录以便找到配置文件
                os.chdir(sam2_path)

                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                # 根据checkpoint名称推断配置文件
                ckpt_name = os.path.basename(self.sam2_checkpoint)
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
                        cfg = base_cfg + "sam2.1_hiera_t.yaml"  # 默认使用tiny配置
                        logger.warning(f"Cannot infer SAM2.1 config from checkpoint name: {ckpt_name}, using default")
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
                        cfg = base_cfg + "sam_hiera_t.yaml"  # 默认使用tiny配置
                        logger.warning(f"Cannot infer SAM2 config from checkpoint name: {ckpt_name}, using default")

                logger.info(f"Loading SAM2 model with config: {cfg}, checkpoint: {self.sam2_checkpoint}")
                sam_model = build_sam2(config_file=cfg, ckpt_path=self.sam2_checkpoint)
                self._sam2_predictor = SAM2ImagePredictor(sam_model, mask_threshold=self.sam2_mask_threshold)
                logger.info(f"SAM2 predictor initialized for mask refinement (mask_threshold={self.sam2_mask_threshold})")

            finally:
                # 恢复原始工作目录
                os.chdir(original_cwd)

        except Exception as e:
            logger.error(f"Failed to initialize SAM2 predictor: {e}")
            logger.error(f"SAM2 checkpoint: {self.sam2_checkpoint}")
            self._sam2_predictor = None

        return self._sam2_predictor

    def _get_aff_engine(self):
        if self._aff_engine is not None:
            return self._aff_engine
        try:
            # 传入SAM2 checkpoint给AffordanceEngine进行内部初始化
            self._aff_engine = AffordanceEngine(
                reasoning_model_path=self.aff_reasoning_model,
                sam2_checkpoint=self.sam2_checkpoint,
                device="auto"
            )
            logger.info("AffordanceEngine initialized (Qwen2.5-VL loaded once) with integrated SAM2")
        except Exception as e:
            logger.error(f"Failed to initialize AffordanceEngine: {e}")
            self._aff_engine = None
        return self._aff_engine

    def _refine_object_mask_with_sam2(self, image_path: str, base_mask: np.ndarray,
                                       positive_points: Optional[List[List[int]]] = None,
                                       negative_points: Optional[List[List[int]]] = None,
                                       expand_ratio: float = 0.12,
                                       use_box: bool = False) -> Optional[np.ndarray]:
        """
        使用 SAM2 的 mask_input（低分辨率 logits）+ 点/框 对初始掩码进行细化，期望补齐缺失部分。

        Args:
            image_path: 原始图像路径
            base_mask: 初始的布尔掩码（HxW）。若尺寸与图像不符，会自动缩放。
            positive_points: 额外的前景点列表[[x,y], ...]
            negative_points: 额外的背景点列表[[x,y], ...]
            expand_ratio: 生成外接框时的边界扩展比例（相对宽/高）

        Returns:
            细化后的布尔掩码（HxW）或 None（失败时）
        """
        try:
            predictor = self._get_sam2_predictor()
            if predictor is None:
                return None

            img = Image.open(image_path).convert("RGB")
            w, h = img.size

            # 确保掩码尺寸与图像一致
            if base_mask.shape != (h, w):
                import cv2 as _cv2
                base_mask_resized = _cv2.resize(base_mask.astype('uint8') * 255, (w, h), interpolation=_cv2.INTER_NEAREST) > 0
            else:
                base_mask_resized = base_mask

            # 统计掩码范围，用于可选外接框
            ys, xs = np.where(base_mask_resized)
            if len(xs) == 0:
                logger.warning("Refine skipped: base mask is empty")
                return base_mask_resized

            box_xyxy = None
            if use_box:
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                rx = max(1, int((x2 - x1 + 1) * expand_ratio))
                ry = max(1, int((y2 - y1 + 1) * expand_ratio))
                x1 = max(0, x1 - rx)
                y1 = max(0, y1 - ry)
                x2 = min(w - 1, x2 + rx)
                y2 = min(h - 1, y2 + ry)
                box_xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)

            # mask_input 低分辨率 logits（1x256x256）
            import cv2 as _cv2
            low = _cv2.resize(base_mask_resized.astype('uint8') * 255, (256, 256), interpolation=_cv2.INTER_NEAREST)
            logits = (low > 0).astype(np.float32)
            logits = (logits * 20.0) - 10.0  # True->+10, False->-10（范围[-10,10]，SAM2内部会再clamp）
            mask_input = logits[None, :, :]

            # 组织点提示（正/负）
            pts = []
            labels = []
            if positive_points:
                pts.extend(positive_points)
                labels.extend([1] * len(positive_points))
            if negative_points:
                pts.extend(negative_points)
                labels.extend([0] * len(negative_points))

            point_coords = np.array(pts, dtype=np.float32) if pts else None
            point_labels = np.array(labels, dtype=np.int32) if labels else None

            # 设图并细化预测
            predictor.set_image(img)
            masks_np, iou_np, lowres_np = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_xyxy,
                mask_input=mask_input,
                multimask_output=False,
                return_logits=False,
                normalize_coords=True,
            )

            # 取单张结果
            if masks_np.ndim == 3:
                refined = masks_np[0].astype(bool)
            else:
                refined = masks_np.astype(bool)

            # 若细化后面积极小/极大，做一个保守合并
            area_base = int(base_mask_resized.sum())
            area_ref = int(refined.sum())
            if area_ref < max(50, 0.3 * area_base):
                refined = np.logical_or(refined, base_mask_resized)

            return refined

        except Exception as e:
            logger.error(f"SAM2 refine failed: {e}")
            return None

    def _to_english_object(self, object_name: str) -> str:
        """
        将常见中文目标与方位词转换为简洁英文短语，便于 Affordance-R1 推理鲁棒性。
        仅做最小必要替换，不依赖外部服务。
        """
        if not object_name:
            return object_name

        s = object_name
        # 方位词
        s = s.replace("左边的", "left ")
        s = s.replace("右边的", "right ")
        s = s.replace("中间的", "middle ")

        # 常见物体
        mapping = {
            "牛油果": "avocado",
            "苹果": "apple",
            "杯子": "cup",
            "马克杯": "mug",
            "刀": "knife",
            "锤子": "hammer",
            "椅子": "chair",
            "球": "ball",
            "勺子": "spoon",
            "罐子": "jar",
        }
        for k, v in mapping.items():
            s = s.replace(k, v)

        # 去除多余空格
        s = ' '.join(s.split())
        return s

    def _parse_user_instruction_universal(self, instruction: str) -> Dict:
        """
        简化用户指令解析 - 使用规则提取

        Args:
            instruction: 用户输入指令

        Returns:
            解析结果字典
        """
        logger.info(f"Parsing user instruction: {instruction}")

        # 简化的规则解析模式
        patterns = [
            # 模式1: [把/将] [物体] [移动动作] [目标位置]
            r'(?:把|将)\s*(.+?)\s*(?:移动|放|拿|滚|推|拉|投|抛)\s*(.+)',
            # 模式2: [物体] [移动动作] [目标位置]
            r'(.+?)\s*(?:移动|放|拿|滚|推|拉|投|抛)\s*(.+)',
            # 模式3: [物体] [动作描述]
            r'(.+?)\s*(拿起|放下|抓取|释放)(?:到|在)?\s*(.*)',
            # 模式4: 简单识别关键词后提取物体名称
            r'(?:把|将)\s*([^\s]+)(.*)',
            # 默认模式：提取第一个关键词作为物体
            r'([^\s]+)(.*)'
        ]

        for pattern in patterns:
            match = re.match(pattern, instruction.strip())
            if match:
                groups = match.groups()
                if len(groups) >= 2 and groups[0].strip() and groups[1].strip():
                    object_name = groups[0].strip()
                    movement_instruction = (groups[1] if len(groups) > 1 else instruction).strip()

                    # 清理物体名称中的方位词
                    object_name = re.sub(r'^(左边的|右边的|中间的|上面的|下面的|前面的|后面的)', '', object_name).strip()

                    result = {
                        "success": True,
                        "object_name": object_name,
                        "movement_instruction": movement_instruction,
                        "original_instruction": instruction
                    }

                    logger.info(f"Parsed instruction: {result}")
                    return result

        # 如果所有模式都失败，使用默认解析
        logger.warning("Using fallback parsing strategy")
        words = instruction.split()
        if len(words) >= 2:
            object_name = words[1] if len(words) > 1 else words[0]
            movement_instruction = instruction
        else:
            object_name = instruction
            movement_instruction = "移动"

        result = {
            "success": True,
            "object_name": object_name,
            "movement_instruction": movement_instruction,
            "original_instruction": instruction
        }

        logger.info(f"Fallback parsed instruction: {result}")
        return result

    def _execute_object_segmentation_universal(self, image_path: str, object_name: str, user_instruction: str = None) -> Dict:
        """
        执行物体识别和分割 - 仅使用Affordance

        Args:
            image_path: 图像路径
            object_name: 操作对象名称
            user_instruction: 用户原始指令（未使用，保持接口兼容）

        Returns:
            分割结果字典

        Raises:
            RuntimeError: 当Affordance推理失败或mask为空时
        """
        logger.info(f"Executing affordance-only object segmentation for: {object_name}")

        try:
            result = self._get_affordance_and_object_masks(
                image_path=image_path,
                object_name=object_name,
                task_dir=self.output_dir,
                user_instruction=user_instruction
            )
            logger.info(f"Affordance segmentation successful: {object_name}")
            return result
        except Exception as e:
            error_msg = f"Affordance segmentation failed for {object_name}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def _run_affordance_inference(self, image_path: str, text: str, output_dir: str) -> Dict:
        """
        使用集成了SAM2的AffordanceEngine执行推理。

        Raises:
            RuntimeError: 当推理失败或mask为空时
        """
        os.makedirs(output_dir, exist_ok=True)
        engine = self._get_aff_engine()
        if engine is None:
            raise RuntimeError("AffordanceEngine not available - failed to initialize")

        logger.info(f"Affordance(in-process with integrated SAM2): {text}")
        res = engine.infer(image_path=image_path, text=text, sam2_predictor=None, multimask_output=False)

        if not res.get("success"):
            error_msg = res.get('error', 'Unknown AffordanceEngine error')
            raise RuntimeError(f"AffordanceEngine inference failed: {error_msg}")

        # 验证返回的mask是否有效
        mask = res.get("mask")
        if mask is None or not np.any(mask):
            raise RuntimeError("AffordanceEngine returned empty or invalid mask")

        mask_bool = mask.astype(bool)
        points_px = res.get("points_px", [])

        if not points_px:
            raise RuntimeError("AffordanceEngine returned no point coordinates")

        mask_path = os.path.join(output_dir, "mask.npy")
        np.save(mask_path, mask_bool.astype(np.uint8))

        logger.info(f"Affordance inference successful: mask shape={mask_bool.shape}, points={len(points_px)}")
        return {"success": True, "mask_path": mask_path, "points_px": points_px}

    def _get_affordance_and_object_masks(self, image_path: str, object_name: str, task_dir: str, user_instruction: Optional[str] = None) -> Dict:
        """
        使用 Affordance 获取：
        - 可交互区域 mask（交互问法）
        - 完整物体 mask（定位问法）
        并计算可交互区域中心点（质心）。

        Raises:
            RuntimeError: 当任何affordance推理失败时
        """
        seg_root = os.path.join(task_dir, "segmentation")
        aff_interactive_dir = os.path.join(seg_root, "affordance_interactive")
        aff_object_dir = os.path.join(seg_root, "affordance_object")
        os.makedirs(seg_root, exist_ok=True)
        os.makedirs(aff_interactive_dir, exist_ok=True)
        os.makedirs(aff_object_dir, exist_ok=True)

        # 交互区域问法：可交互区域 mask
        # 你确认的模板："To control the {object} safely, where should I"
        # 如需动词可接 hold/grasp，但按你的提醒这里省略动词也可得到交互区域
        # 将 object_name 转为英文以提升 infer.py 输出 JSON 的稳定性
        en_object = self._to_english_object(object_name)
        interactive_text = f"To control the {en_object} safely, where should I hold?"
        # 记录交互区域提示词
        try:
            with open(os.path.join(seg_root, "affordance_interactive_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(interactive_text)
        except Exception as _e:
            logger.warning(f"Failed to save affordance interactive prompt: {_e}")
        inter_res = self._run_affordance_inference(image_path, interactive_text, aff_interactive_dir)
        inter_mask_src = inter_res["mask_path"]

        # 完整物体问法：强调“完整/整个”物体的分割与覆盖
        # 目标：让推理模型返回覆盖该物体全部可见部分的掩码（不要只返回局部）
        locate_text = (
            f"Please segment the entire {en_object}. "
            f"Return a binary mask that covers the complete, whole object (all visible parts) without missing regions. "
            f"Include thin or elongated parts (e.g., handles, edges), and exclude background."
        )
        # 记录完整物体提示词
        try:
            with open(os.path.join(seg_root, "affordance_object_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(locate_text)
        except Exception as _e:
            logger.warning(f"Failed to save affordance object prompt: {_e}")
        obj_res = self._run_affordance_inference(image_path, locate_text, aff_object_dir)
        obj_mask_src = obj_res["mask_path"]

        # 复制/标准化命名
        object_mask_path = os.path.join(seg_root, f"{object_name}_mask.npy")
        affordance_mask_path = os.path.join(seg_root, f"{object_name}_affordance_mask.npy")

        # 统一保存为 bool npy
        obj_mask = np.load(obj_mask_src)
        inter_mask = np.load(inter_mask_src)
        obj_mask_bool = (obj_mask > 0)
        inter_mask_bool = (inter_mask > 0)

        # 先保存原始的 Affordance 结果
        np.save(object_mask_path, obj_mask_bool)
        np.save(affordance_mask_path, inter_mask_bool)

        # 生成可视化图（叠加到原图）：object 与 affordance 两张（refine 前）
        try:
            base_img = Image.open(image_path).convert("RGBA")
            w, h = base_img.size

            # object mask 可视化（蓝色透明叠加）
            obj_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            obj_pixels = obj_overlay.load()
            # 将布尔 mask 调整为图像尺寸
            if obj_mask_bool.shape != (h, w):
                # 常见是 (H, W) = (480, 640)，需要确保匹配尺寸
                import cv2 as _cv2
                obj_resized = _cv2.resize(obj_mask_bool.astype('uint8') * 255, (w, h), interpolation=_cv2.INTER_NEAREST) > 0
            else:
                obj_resized = obj_mask_bool
            for yy in range(h):
                for xx in range(w):
                    if obj_resized[yy, xx]:
                        obj_pixels[xx, yy] = (0, 122, 255, 90)  # 蓝色半透明
            obj_vis = Image.alpha_composite(base_img, obj_overlay)
            obj_vis = obj_vis.convert("RGB")
            obj_vis_path = os.path.join(seg_root, f"{object_name}_object_mask_overlay.png")
            obj_vis.save(obj_vis_path)

            # affordance mask 可视化（绿色透明叠加），并标注中心点
            aff_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            aff_pixels = aff_overlay.load()
            if inter_mask_bool.shape != (h, w):
                import cv2 as _cv2
                aff_resized = _cv2.resize(inter_mask_bool.astype('uint8') * 255, (w, h), interpolation=_cv2.INTER_NEAREST) > 0
            else:
                aff_resized = inter_mask_bool
            for yy in range(h):
                for xx in range(w):
                    if aff_resized[yy, xx]:
                        aff_pixels[xx, yy] = (0, 200, 0, 90)  # 绿色半透明
            aff_vis = Image.alpha_composite(base_img, aff_overlay)
            draw = ImageDraw.Draw(aff_vis)
            # 中心点稍后计算，先默认不画
            aff_vis = aff_vis.convert("RGB")
            aff_vis_path = os.path.join(seg_root, f"{object_name}_affordance_mask_overlay.png")
            aff_vis.save(aff_vis_path)
        except Exception as _e:
            logger.warning(f"Failed to create mask visualization images: {_e}")

        # 计算可交互区域中心点（质心），若为空则退化到首个点
        aff_mask_bool = inter_mask_bool
        ys, xs = np.where(aff_mask_bool)
        if len(xs) > 0:
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
        elif inter_res.get("points_px"):
            cx, cy = inter_res["points_px"][0]
        else:
            raise RuntimeError("Affordance interactive mask is empty and no points returned.")

        # 相对坐标
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        rel_x = cx / float(w)
        rel_y = cy / float(h)

        logger.info(f"Affordance center (px) = ({cx}, {cy}), rel=({rel_x:.4f}, {rel_y:.4f})")

        # 在 affordance 可视化图上标注中心点（若已生成）
        try:
            aff_vis_path = os.path.join(seg_root, f"{object_name}_affordance_mask_overlay.png")
            if os.path.exists(aff_vis_path):
                aff_img = Image.open(aff_vis_path).convert("RGB")
                d = ImageDraw.Draw(aff_img)
                r = max(2, int(0.01 * max(w, h)))
                d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 0, 0), width=2)
                d.line([(cx - 2 * r, cy), (cx + 2 * r, cy)], fill=(255, 0, 0), width=2)
                d.line([(cx, cy - 2 * r), (cx, cy + 2 * r)], fill=(255, 0, 0), width=2)
                aff_img.save(aff_vis_path)
        except Exception as _e:
            logger.warning(f"Failed to draw affordance center on visualization: {_e}")

        # === 使用 SAM2 基于初始“完整物体”掩码进行细化（可弥补不完整处） ===
        try:
            pos_pts = [[cx, cy]]  # 以交互中心作为一个正点
            refined_mask = self._refine_object_mask_with_sam2(
                image_path=image_path,
                base_mask=obj_mask_bool,
                positive_points=pos_pts,
                negative_points=None,
                expand_ratio=0.12,
            )
            if refined_mask is not None:
                obj_mask_bool = refined_mask
                np.save(object_mask_path, obj_mask_bool)

                # 细化后的可视化覆盖原图（蓝色半透明）
                try:
                    base_img2 = Image.open(image_path).convert("RGBA")
                    obj_overlay2 = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                    obj_pixels2 = obj_overlay2.load()
                    # 若维度不一致则调整
                    if obj_mask_bool.shape != (h, w):
                        import cv2 as _cv2
                        obj_resized2 = _cv2.resize(obj_mask_bool.astype('uint8') * 255, (w, h), interpolation=_cv2.INTER_NEAREST) > 0
                    else:
                        obj_resized2 = obj_mask_bool
                    for yy in range(h):
                        for xx in range(w):
                            if obj_resized2[yy, xx]:
                                obj_pixels2[xx, yy] = (0, 122, 255, 90)
                    obj_vis2 = Image.alpha_composite(base_img2, obj_overlay2).convert("RGB")
                    obj_vis_path2 = os.path.join(seg_root, f"{object_name}_object_mask_overlay.png")
                    obj_vis2.save(obj_vis_path2)
                    logger.info("Object mask refined by SAM2 and visualization updated.")
                except Exception as _e:
                    logger.warning(f"Failed to update refined object mask visualization: {_e}")
        except Exception as _e:
            logger.warning(f"SAM2 refine step skipped due to error: {_e}")

        # 生成最终综合可视化：参考 Affordance 的可视化风格
        # 效果：整体图像略微变暗，物体（最终掩码）区域以显眼的半透红色高亮，并标注交互中心
        try:
            base_img3 = Image.open(image_path).convert("RGBA")
            # 对象掩码（已细化）
            if obj_mask_bool.shape != (h, w):
                import cv2 as _cv2
                obj_resized3 = _cv2.resize(obj_mask_bool.astype('uint8') * 255, (w, h), interpolation=_cv2.INTER_NEAREST) > 0
            else:
                obj_resized3 = obj_mask_bool

            # 1) 旁边暗一点：非掩码区域叠加半透明黑色，掩码区域不变
            dim_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 140))
            dim_pixels = dim_overlay.load()
            for yy in range(h):
                for xx in range(w):
                    if obj_resized3[yy, xx]:
                        dim_pixels[xx, yy] = (0, 0, 0, 0)  # 掩码区域不变，外部变暗
            merged = Image.alpha_composite(base_img3, dim_overlay)

            # 2) 掩码区域用显眼且与其他视图区分的颜色（洋红）高亮
            red_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            red_pixels = red_overlay.load()
            for yy in range(h):
                for xx in range(w):
                    if obj_resized3[yy, xx]:
                        red_pixels[xx, yy] = (255, 0, 255, 140)  # 半透洋红（更醒目，且与其他视图区分）
            merged = Image.alpha_composite(merged, red_overlay).convert("RGB")

            # 最终完整图不画中心点标记

            final_vis_path = os.path.join(seg_root, f"{object_name}_final_overlay.png")
            merged.save(final_vis_path)
            logger.info(f"Final combined overlay saved: {final_vis_path}")
        except Exception as _e:
            logger.warning(f"Failed to create final combined overlay: {_e}")

        # 构造 segmentation_result 与现有结构对齐
        return {
            "success": True,
            "final_mask_path": object_mask_path,  # 供视频生成使用
            "affordance_mask_path": affordance_mask_path,
            "coordinates": (rel_x, rel_y),  # 起点（相对坐标）
            "position_iterations": 0,
            "segmentation_iterations": 0,
            "fallback_used": False
        }

    def _compute_destination_target_point(self, image_path: str, destination_name: str,
                                          user_instruction: Optional[str] = None,
                                          start_point: Optional[Tuple[float, float]] = None) -> Optional[Tuple[float, float]]:
        """
        使用 Affordance 对"目的地"对象进行定位，返回推荐的放置点（相对坐标）。

        说明：
        - 这里的 destination_name 通常是"金属锅"、"盘子"、"桌子右侧区域"等高层语义目标。
        - 目的地不需要可操作点，只需要分割出完整mask并取中心点作为轨迹终点。
        - 如果没有准确的目的地，使用模糊目的地定位，并确保距离起点至少100像素。
        """
        if not destination_name:
            return None

        logger.info(f"Computing destination target point via affordance for: {destination_name}")

        # 第一阶段：标准的目的地对象分割（只使用完整物体分割，不需要可操作点）
        dest_mask = None
        try:
            # 直接使用完整的对象分割，不需要交互式affordance
            dest_mask = self._get_destination_object_mask(
                image_path=image_path,
                destination_name=destination_name,
                task_dir=self.output_dir
            )
        except RuntimeError as e:
            logger.error(f"Destination segmentation failed for {destination_name}: {e}")
            dest_mask = None

        if dest_mask is not None:
            # 使用目的地 mask 的质心作为目标点
            mask_path = dest_mask.get("mask_path")
            if mask_path and os.path.exists(mask_path):
                try:
                    mask_arr = np.load(mask_path)
                    ys, xs = np.where(mask_arr > 0)
                    if len(xs) > 0:
                        h, w = mask_arr.shape
                        cx = float(xs.mean()) / float(w)
                        cy = float(ys.mean()) / float(h)
                        logger.info(f"Using destination object mask center as target point: ({cx:.4f}, {cy:.4f})")
                        return (cx, cy)
                except Exception as e:
                    logger.warning(f"Failed to compute destination center from mask: {e}")

        # 第二阶段：目的地分割失败或无可靠中心 -> 启用"模糊目的地" Affordance 规划
        logger.info(
            f"Standard destination segmentation did not yield a reliable point for '{destination_name}'. "
            f"Falling back to fuzzy destination affordance planning."
        )

        # 构造更有语义约束的提示词，请求模型直接给出"最终放置区域"
        fuzzy_text = (
            "You are assisting a robot arm to complete a tabletop manipulation task.\n"
            f"The full natural-language instruction for this step is: \"{user_instruction}\".\n"
            f"The intended destination is: \"{destination_name}\".\n"
            "Please highlight exactly one compact region in the image that is the SINGLE BEST final resting location "
            "where the manipulated object should end up after the operation, so that the object is safely placed "
            "inside / on the intended destination and does not float in the air or overlap with other objects.\n"
            "Do NOT describe the reasoning or multiple options; instead, output a JSON list of exactly one bbox_2d "
            "and one point_2d in <answer> tags as required."
        )

        fuzzy_dir = os.path.join(self.output_dir, "destination_affordance")
        try:
            res = self._run_affordance_inference(
                image_path=image_path,
                text=fuzzy_text,
                output_dir=fuzzy_dir
            )
        except RuntimeError as e:
            logger.warning(f"Fuzzy destination affordance planning also failed for {destination_name}: {e}")
            return None

        # 使用模糊目的地 mask 的质心作为终点
        mask_path = res.get("mask_path")
        if mask_path and os.path.exists(mask_path):
            try:
                mask_arr = np.load(mask_path)
                ys, xs = np.where(mask_arr > 0)
                if len(xs) > 0:
                    h, w = mask_arr.shape
                    cx = float(xs.mean()) / float(w)
                    cy = float(ys.mean()) / float(h)

                    # 应用距离约束：确保目的地点距离起点至少100像素
                    if start_point is not None:
                        enforced_cx, enforced_cy = self._enforce_minimum_distance(
                            cx, cy, start_point[0], start_point[1], w, h, min_distance_pixels=100
                        )
                        if (enforced_cx, enforced_cy) != (cx, cy):
                            logger.info(
                                f"Enforced minimum distance constraint: original ({cx:.4f}, {cy:.4f}) -> "
                                f"enforced ({enforced_cx:.4f}, {enforced_cy:.4f})"
                            )
                            cx, cy = enforced_cx, enforced_cy

                    logger.info(
                        f"Using fuzzy destination affordance center as target point for '{destination_name}': "
                        f"({cx:.4f}, {cy:.4f})"
                    )
                    return (cx, cy)
            except Exception as e:
                logger.warning(f"Failed to compute fuzzy destination center from mask: {e}")

        logger.warning(f"Could not compute reliable destination target point for: {destination_name}")
        return None

    def _get_destination_object_mask(self, image_path: str, destination_name: str,
                                     task_dir: str) -> Optional[Dict]:
        """
        专门用于目的地对象的完整分割，不需要可操作点

        Args:
            image_path: 图像路径
            destination_name: 目的地对象名称
            task_dir: 任务目录

        Returns:
            包含mask路径的字典
        """
        logger.info(f"Getting destination object mask for: {destination_name}")

        # 只使用完整的对象分割提示词
        locate_text = (
            f"Please segment the entire {destination_name}. "
            f"Return a binary mask that covers the complete, whole object (all visible parts) without missing regions. "
            f"Include thin or elongated parts (e.g., handles, edges), and exclude background."
        )

        # 创建输出目录
        dest_dir = os.path.join(task_dir, "destination_segmentation")
        os.makedirs(dest_dir, exist_ok=True)

        try:
            # 直接运行affordance推理，只使用完整对象分割
            result = self.affordance_engine.infer(
                image=image_path,
                text=locate_text,
                output_dir=dest_dir,
                confidence_threshold=0.3
            )

            if result and result.get("mask_path"):
                mask_path = result["mask_path"]
                logger.info(f"Successfully generated destination mask: {mask_path}")
                return {"mask_path": mask_path}
            else:
                logger.warning(f"Destination segmentation failed to generate mask for: {destination_name}")
                return None

        except Exception as e:
            logger.error(f"Error in destination object segmentation: {e}")
            return None

    def _enforce_minimum_distance(self, cx: float, cy: float, sx: float, sy: float,
                                  image_width: int, image_height: int,
                                  min_distance_pixels: int = 100) -> Tuple[float, float]:
        """
        确保目的地点距离起点至少指定距离，如果不够则延长（不超出图像尺寸）

        Args:
            cx, cy: 目的地点的相对坐标 (0-1)
            sx, sy: 起点的相对坐标 (0-1)
            image_width, image_height: 图像尺寸
            min_distance_pixels: 最小距离（像素）

        Returns:
            调整后的目的地点相对坐标
        """
        # 转换为像素坐标
        cx_px = cx * image_width
        cy_px = cy * image_height
        sx_px = sx * image_width
        sy_px = sy * image_height

        # 计算当前距离
        dx = cx_px - sx_px
        dy = cy_px - sy_px
        current_distance = math.sqrt(dx*dx + dy*dy)

        # 如果当前距离已经满足要求，直接返回
        if current_distance >= min_distance_pixels:
            return cx, cy

        # 需要延长距离
        if current_distance == 0:
            # 如果起点和终点重合，默认向右延长
            dx, dy = min_distance_pixels, 0
        else:
            # 计算延长因子
            scale_factor = min_distance_pixels / current_distance
            dx *= scale_factor
            dy *= scale_factor

        # 计算新的目标点
        new_cx_px = sx_px + dx
        new_cy_px = sy_px + dy

        # 确保不超出图像边界
        new_cx_px = max(0, min(image_width - 1, new_cx_px))
        new_cy_px = max(0, min(image_height - 1, new_cy_px))

        # 转换回相对坐标
        new_cx = new_cx_px / image_width
        new_cy = new_cy_px / image_height

        return new_cx, new_cy

    def _create_simplified_trajectory_visualization(self, image_path: str, object_name: str,
                                                   object_mask_path: Optional[str],
                                                   affordance_point: Tuple[float, float],
                                                   robot_approach_trajectory: List[List[int]],
                                                   object_trajectory_points: List[List[int]],
                                                   robot_exit_trajectory: List[List[int]]) -> str:
        """
        创建简化的轨迹可视化，只包含：
        1. 操作对象的mask
        2. 操作对象的可操作点
        3. 完整的三段式轨迹

        Args:
            image_path: 原始图像路径
            object_name: 对象名称
            object_mask_path: 对象mask路径
            affordance_point: 可操作点相对坐标
            robot_approach_trajectory: 机械臂接近轨迹
            object_trajectory_points: 对象移动轨迹
            robot_exit_trajectory: 机械臂退出轨迹

        Returns:
            保存的可视化图像路径
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        logger.info(f"Creating simplified trajectory visualization for: {object_name}")

        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return ""

        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # 创建图像
        plt.figure(figsize=(12, 8))
        plt.imshow(image_rgb)

        # 1. 显示操作对象mask（半透明绿色覆盖）
        if object_mask_path and os.path.exists(object_mask_path):
            try:
                if object_mask_path.endswith('.npy'):
                    mask = np.load(object_mask_path)
                else:
                    mask = cv2.imread(object_mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = mask > 127

                # 创建绿色半透明覆盖
                overlay = np.zeros_like(image_rgb)
                overlay[mask] = [0, 255, 0]  # 绿色
                plt.imshow(overlay, alpha=0.3)  # 半透明
                logger.info("Applied object mask overlay (green)")
            except Exception as e:
                logger.warning(f"Failed to apply object mask: {e}")

        # 2. 显示可操作点（红色圆圈）
        affordance_abs = (int(affordance_point[0] * w), int(affordance_point[1] * h))
        plt.scatter(affordance_abs[0], affordance_abs[1], c='red', s=300, marker='o',
                   edgecolors='white', linewidth=3, label='Affordance Point', zorder=10)

        # 添加可操作点标注
        plt.annotate(f'Affordance\n({affordance_point[0]:.3f}, {affordance_point[1]:.3f})',
                    xy=affordance_abs, xytext=(15, 15), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.8),
                    fontsize=11, color='white', fontweight='bold')

        # 3. 显示三段式轨迹
        # 3.1 机械臂接近轨迹（蓝色）
        if robot_approach_trajectory:
            approach_array = np.array(robot_approach_trajectory)
            plt.plot(approach_array[:, 0], approach_array[:, 1], 'b-', linewidth=4,
                    label='Robot Approach', alpha=0.8, marker='o', markersize=6)

        # 3.2 对象移动轨迹（橙色）
        if object_trajectory_points:
            object_array = np.array(object_trajectory_points)
            plt.plot(object_array[:, 0], object_array[:, 1], color='orange', linewidth=4,
                    label='Object Movement', alpha=0.8, marker='s', markersize=6)

        # 3.3 机械臂退出轨迹（紫色）
        if robot_exit_trajectory:
            exit_array = np.array(robot_exit_trajectory)
            plt.plot(exit_array[:, 0], exit_array[:, 1], color='purple', linewidth=4,
                    label='Robot Exit', alpha=0.8, marker='^', markersize=6)

        # 设置标题和标签
        plt.title(f'Simplified Trajectory Visualization: {object_name}',
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('X (pixels)', fontsize=12)
        plt.ylabel('Y (pixels)', fontsize=12)

        # 设置图例
        plt.legend(loc='upper right', fontsize=11, framealpha=0.9)

        # 去掉网格和边框，保持简洁
        plt.grid(False)
        plt.axis('on')

        # 保存图像
        viz_dir = os.path.join(self.output_dir, "visualization")
        os.makedirs(viz_dir, exist_ok=True)
        filename = f"{object_name}_trajectory.png"
        filepath = os.path.join(viz_dir, filename)

        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"Saved simplified trajectory visualization to: {filepath}")
        return filepath

    def _execute_trajectory_planning_universal(self, image_path: str, object_name: str,
                                              movement_instruction: str, start_point: Tuple[float, float],
                                              target_point: Optional[Tuple[float, float]] = None) -> Dict:
        """
        执行通用轨迹规划

        Args:
            image_path: 图像路径
            object_name: 操作对象名称
            movement_instruction: 通用移动指令
            start_point: 起始点坐标

        Returns:
            轨迹规划结果字典
        """
        logger.info(f"Executing universal trajectory planning for: {object_name}")

        result = self.movement_planner.plan_object_movement(
            image_path=image_path,
            object_name=object_name,
            movement_instruction=movement_instruction,
            start_point=start_point,
            explicit_target=target_point
        )

        if result["success"]:
            logger.info(f"Universal trajectory planning successful: {object_name}")
        else:
            logger.error(f"Universal trajectory planning failed: {result.get('error', 'Unknown error')}")

        return result

    def _generate_robot_trajectory(self, object_first_point: Tuple[int, int], image_width: int = 640, image_height: int = 480) -> List[List[int]]:
        """
        生成机械臂轨迹：从图像边界到操作对象

        Args:
            object_first_point: 操作对象的第一个轨迹点 (x, y)
            image_width: 图像宽度
            image_height: 图像高度

        Returns:
            机械臂轨迹点列表，共11个点
        """
        # 计算左右边界中心点
        left_center = (0, image_height // 2)
        right_center = (image_width - 1, image_height // 2)

        # 选择距离更近的边界点作为起点
        dist_to_left = math.sqrt((object_first_point[0] - left_center[0])**2 + (object_first_point[1] - left_center[1])**2)
        dist_to_right = math.sqrt((object_first_point[0] - right_center[0])**2 + (object_first_point[1] - right_center[1])**2)

        start_point = left_center if dist_to_left <= dist_to_right else right_center

        # 生成11个点的轨迹（包括起点和终点）
        trajectory_points = []
        for i in range(11):
            t = i / 10.0  # 插值参数从0到1
            x = int(start_point[0] + t * (object_first_point[0] - start_point[0]))
            y = int(start_point[1] + t * (object_first_point[1] - start_point[1]))
            trajectory_points.append([x, y])

        logger.info(f"Generated robot trajectory from {start_point} to {object_first_point} with {len(trajectory_points)} points")
        return trajectory_points

    def _generate_separated_trajectories(self, object_first_point: List[int],
                                       object_trajectory_normalized: List[List[float]],
                                       previous_robot_exit_position: Optional[List[int]] = None) -> tuple:
        """
        生成分离的轨迹数据：
        - 机械臂进入轨迹（9个点，frames 0-8）
        - 物体轨迹（19个点，frames 8-27）
        - 机械臂退出轨迹（9个点，frames 27-36）
        总共37个点对应37帧

        Args:
            object_first_point: 物体的第一个轨迹点（机械臂目标位置）
            object_trajectory_normalized: 标准化的物体轨迹 (0-1范围)
            previous_robot_exit_position: 上一个任务的机械臂退出位置（用于轨迹连续性）

        Returns:
            (robot_approach_trajectory, object_trajectory_points, robot_exit_trajectory): 分离的轨迹数据
        """
        # === 生成机械臂进入轨迹（9个点，frames 0-8） ===
        # 定义边界中心（用于进入和退出计算）
        left_center = [0, 240]
        right_center = [639, 240]

        if previous_robot_exit_position:
            # 使用上一个任务的退出位置作为起点，实现轨迹连续性
            approach_start_point = previous_robot_exit_position.copy()
            logger.info(f"Using previous robot exit position for continuity: {approach_start_point}")
        else:
            # 第一个任务：从图像边界中心开始
            dist_left = np.sqrt((object_first_point[0] - left_center[0])**2 +
                               (object_first_point[1] - left_center[1])**2)
            dist_right = np.sqrt((object_first_point[0] - right_center[0])**2 +
                                (object_first_point[1] - right_center[1])**2)

            approach_start_point = left_center if dist_left <= dist_right else right_center
            logger.info(f"First task: starting from {approach_start_point}")

        # 计算距离（用于退出轨迹选择）
        dist_left = np.sqrt((object_first_point[0] - left_center[0])**2 +
                           (object_first_point[1] - left_center[1])**2)
        dist_right = np.sqrt((object_first_point[0] - right_center[0])**2 +
                            (object_first_point[1] - right_center[1])**2)

        approach_end_point = object_first_point.copy()

        # 生成9个点的机械臂进入轨迹（包括起点和终点）
        robot_approach_trajectory = []
        for i in range(9):
            t = i / 8.0  # 插值参数从0到1
            x = int(approach_start_point[0] + t * (approach_end_point[0] - approach_start_point[0]))
            y = int(approach_start_point[1] + t * (approach_end_point[1] - approach_start_point[1]))
            robot_approach_trajectory.append([x, y])

        # === 生成物体轨迹（19个点，frames 8-27） ===
        # 将标准化物体轨迹转换为像素坐标
        object_trajectory_points = self._interpolate_object_trajectory(
            object_trajectory_normalized, target_points=19
        )

        # 确保物体轨迹的第一个点与机械臂进入轨迹的最后一个点一致
        if object_trajectory_points:
            object_trajectory_points[0] = approach_end_point.copy()

        # === 生成机械臂退出轨迹（9个点，frames 27-36） ===
        # 从物体轨迹的终点退出到图像边界中心
        if object_trajectory_points:
            exit_start_point = object_trajectory_points[-1].copy()
        else:
            exit_start_point = approach_end_point.copy()

        # 退出到相反的边界（如果从左边进入，从右边退出）
        exit_end_point = right_center if dist_left <= dist_right else left_center

        # 生成9个点的机械臂退出轨迹（包括起点和终点）
        robot_exit_trajectory = []
        for i in range(9):
            t = i / 8.0  # 插值参数从0到1
            x = int(exit_start_point[0] + t * (exit_end_point[0] - exit_start_point[0]))
            y = int(exit_start_point[1] + t * (exit_end_point[1] - exit_start_point[1]))
            robot_exit_trajectory.append([x, y])

        logger.info(f"Generated separated trajectories:")
        logger.info(f"  Robot approach trajectory: {len(robot_approach_trajectory)} points (frames 0-8)")
        logger.info(f"  Object trajectory: {len(object_trajectory_points)} points (frames 8-27)")
        logger.info(f"  Robot exit trajectory: {len(robot_exit_trajectory)} points (frames 27-36)")
        logger.info(f"  Robot approach: {approach_start_point} -> {approach_end_point}")
        logger.info(f"  Object: {object_trajectory_points[0] if object_trajectory_points else 'None'} -> {object_trajectory_points[-1] if object_trajectory_points else 'None'}")
        logger.info(f"  Robot exit: {exit_start_point} -> {exit_end_point}")

        return robot_approach_trajectory, object_trajectory_points, robot_exit_trajectory

    def _interpolate_object_trajectory(self, object_trajectory: List[List[float]], target_points: int = 26) -> List[List[int]]:
        """
        将物体轨迹插值到指定数量的点

        Args:
            object_trajectory: 原始物体轨迹
            target_points: 目标点数

        Returns:
            插值后的物体轨迹点列表
        """
        if len(object_trajectory) >= target_points:
            # 如果轨迹点数量足够，均匀采样
            indices = np.linspace(0, len(object_trajectory) - 1, target_points, dtype=int)
            return [[int(point[0] * 640), int(point[1] * 480)] for point in [object_trajectory[i] for i in indices]]
        else:
            # 如果轨迹点数量不够，进行插值
            interpolated_trajectory = []
            num_segments = len(object_trajectory) - 1
            points_per_segment = target_points // num_segments
            remainder = target_points % num_segments

            for i in range(num_segments):
                start_point = object_trajectory[i]
                end_point = object_trajectory[i + 1]
                segment_points = points_per_segment + (1 if i < remainder else 0)

                for j in range(segment_points):
                    t = j / segment_points if segment_points > 1 else 0
                    x = start_point[0] + t * (end_point[0] - start_point[0])
                    y = start_point[1] + t * (end_point[1] - start_point[1])
                    interpolated_trajectory.append([int(x * 640), int(y * 480)])

            # 添加最后一个点
            interpolated_trajectory.append([int(object_trajectory[-1][0] * 640), int(object_trajectory[-1][1] * 480)])

            return interpolated_trajectory[:target_points]

    def _save_video_generation_data(self, object_name: str, user_instruction: str,
                                   image_path: str, segmentation_result: Dict,
                                   trajectory_result: Dict, enhanced_prompts: Optional[Dict] = None,
                                   previous_robot_exit_position: Optional[List[int]] = None) -> str:
        """
        保存视频生成所需的完整数据

        Args:
            object_name: 操作对象名称
            user_instruction: 用户指令
            image_path: 图像路径
            segmentation_result: 分割结果
            trajectory_result: 轨迹规划结果
            enhanced_prompts: 增强提示词数据（可选）

        Returns:
            保存的JSON文件路径
        """
        # 生成随机种子
        random_seed = random.randint(1, 10000)

        # 获取物体轨迹的第一个点作为机械臂轨迹的终点
        object_first_point = [int(trajectory_result["start_point"][0] * 640),
                             int(trajectory_result["start_point"][1] * 480)]

        # 生成分离的轨迹数据：
        # - 机械臂进入轨迹（9个点，frames 0-8）
        # - 物体轨迹（19个点，frames 8-27）
        # - 机械臂退出轨迹（9个点，frames 27-36）
        robot_approach_trajectory, object_trajectory_points, robot_exit_trajectory = self._generate_separated_trajectories(
            object_first_point, trajectory_result["trajectory"], previous_robot_exit_position
        )

        # 创建37帧的完整轨迹用于兼容
        robot_trajectory_full = []
        object_trajectory_full = []

        # 机械臂轨迹：帧0-8使用进入轨迹，帧9-26使用占位符，帧27-36使用退出轨迹
        for i in range(37):
            if i < 9:
                # 帧0-8：机械臂进入轨迹
                robot_trajectory_full.append(robot_approach_trajectory[i])
            elif i < 27: 
                # 帧9-26：使用占位符（重复进入轨迹的最后一个点）
                robot_trajectory_full.append(robot_approach_trajectory[-1])
            else:
                # 帧27-36：机械臂退出轨迹
                idx = i - 27
                if idx < len(robot_exit_trajectory):
                    robot_trajectory_full.append(robot_exit_trajectory[idx])
                else:  
                    robot_trajectory_full.append(robot_exit_trajectory[-1])

        # 物体轨迹：帧0-7使用占位符，帧8-26使用实际轨迹点，帧27-36使用占位符
        for i in range(37):
            if i < 8:
                # 帧0-7：使用占位符（重复第一个有效点）
                object_trajectory_full.append(object_trajectory_points[0])
            elif i < 27:
                # 帧8-26：使用实际轨迹点（最多19个点）
                idx = i - 8
                if idx < len(object_trajectory_points):
                    object_trajectory_full.append(object_trajectory_points[idx])
                else:
                    # 如果轨迹不够，使用最后一个点
                    object_trajectory_full.append(object_trajectory_points[-1])
            else:
                # 帧27-36：使用占位符（重复最后一个有效点）
                object_trajectory_full.append(object_trajectory_points[-1])

        # 确保物体掩码文件存在并复制到正确位置
        mask_dest_path = os.path.join(self.output_dir, f"{object_name}_obj_mask.npy")

        # 检查是否使用了回退值
        if segmentation_result.get("fallback_used", False):
            logger.warning(f"Using fallback mask for {object_name}")
            # 创建一个默认的掩码（一个小圆圈）
            mask = np.zeros((480, 640), dtype=np.uint8)
            center_x = int(trajectory_result["start_point"][0] * 640)
            center_y = int(trajectory_result["start_point"][1] * 480)
            cv2.circle(mask, (center_x, center_y), 30, 255, -1)
            np.save(mask_dest_path, mask)
            logger.info(f"Created fallback mask: {mask_dest_path}")
        else:
            mask_source_path = None
            if "final_mask_path" in segmentation_result:
                mask_source_path = segmentation_result["final_mask_path"]
            else:
                # 尝试从分割结果中找到掩码文件
                possible_mask_paths = [
                    os.path.join(self.output_dir, "segmentation", f"{object_name}_mask_iter1.png"),
                    os.path.join(self.output_dir, "segmentation", f"{object_name}_final_mask.png"),
                    os.path.join(self.output_dir, f"{object_name}_final_mask.png")
                ]
                for path in possible_mask_paths:
                    if os.path.exists(path):
                        mask_source_path = path
                        break

            if mask_source_path and os.path.exists(mask_source_path):
                if mask_source_path.lower().endswith('.npy'):
                    # 直接使用 NPY 掩码，统一保存为 bool
                    mask_arr = np.load(mask_source_path)
                    np.save(mask_dest_path, mask_arr > 0)
                    logger.info(f"Copied object mask (npy) to: {mask_dest_path}")
                else:
                    # 将PNG掩码转换为NPY格式
                    mask_image = cv2.imread(mask_source_path, cv2.IMREAD_GRAYSCALE)
                    if mask_image is not None:
                        # 二值化掩码
                        _, binary_mask = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
                        np.save(mask_dest_path, binary_mask > 0)
                        logger.info(f"Saved object mask to: {mask_dest_path}")
                    else:
                        logger.error(f"Failed to load mask from: {mask_source_path}")
                        # 创建回退掩码
                        mask = np.zeros((480, 640), dtype=np.uint8)
                        center_x = int(trajectory_result["start_point"][0] * 640)
                        center_y = int(trajectory_result["start_point"][1] * 480)
                        cv2.circle(mask, (center_x, center_y), 30, 255, -1)
                        np.save(mask_dest_path, mask > 0)
                        logger.info(f"Created fallback mask: {mask_dest_path}")
            else:
                logger.warning(f"Mask file not found for: {object_name}, creating fallback mask")
                # 创建一个默认的掩码（一个小圆圈）
                mask = np.zeros((480, 640), dtype=np.uint8)
                center_x = int(trajectory_result["start_point"][0] * 640)
                center_y = int(trajectory_result["start_point"][1] * 480)
                cv2.circle(mask, (center_x, center_y), 30, 255, -1)
                np.save(mask_dest_path, mask)
                logger.info(f"Created fallback mask: {mask_dest_path}")

        # 构建完整的视频生成数据
        video_data = {
            "metadata": {
                "object_name": object_name,
                "user_instruction": user_instruction,
                "random_seed": random_seed,
                "transit_frames": [8, 27, 36],
                "total_frames": 37,
                "generated_by": "universal_robot_controller",
                "created_at": "2025-10-25T08:00:00Z",
                "trajectory_format": "three_phase"  # 标记为三段式格式
            },
            "data_files": {
                "prompt_txt": user_instruction,
                "enhanced_positive_prompt": enhanced_prompts.get("positive_prompt", user_instruction) if enhanced_prompts else user_instruction,
                "enhanced_negative_prompt": enhanced_prompts.get("negative_prompt", "") if enhanced_prompts else "",
                "seed_txt": random_seed,
                # 保存37帧完整轨迹（用于兼容）
                "robot_trajectory": robot_trajectory_full,
                "object_trajectory": object_trajectory_full,
                # 保存分离的轨迹点（实际有效数据）
                "robot_approach_trajectory": robot_approach_trajectory,
                "object_trajectory_points": object_trajectory_points,
                "robot_exit_trajectory": robot_exit_trajectory,
                "transit_frames": [8, 27, 36]
            },
            "file_info": {
                "image_path": image_path,
                "mask_path": mask_dest_path,
                # Affordance 相关（若可用）
                "affordance_mask_path": segmentation_result.get("affordance_mask_path", None),
                "affordance_center_rel": list(segmentation_result.get("coordinates", (trajectory_result["start_point"][0], trajectory_result["start_point"][1]))),
                "affordance_center_px": object_first_point,
                # 分离轨迹的实际有效长度
                "robot_approach_trajectory_length": len(robot_approach_trajectory),  # 9
                "object_trajectory_points_length": len(object_trajectory_points),  # 19
                "robot_exit_trajectory_length": len(robot_exit_trajectory),  # 9
                # 完整轨迹长度（用于兼容）
                "robot_trajectory_length": len(robot_trajectory_full),  # 37
                "object_trajectory_length": len(object_trajectory_full),  # 37
                # 实际的轨迹数据点
                "robot_approach_trajectory": robot_approach_trajectory,
                "object_trajectory_points": object_trajectory_points,
                "robot_exit_trajectory": robot_exit_trajectory,
                # 完整轨迹（包含占位符）
                "robot_trajectory_full": robot_trajectory_full,
                "object_trajectory_full": object_trajectory_full
            }
        }

        # 保存JSON文件
        json_filename = f"{object_name}_video_data.json"
        json_filepath = os.path.join(self.output_dir, json_filename)

        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(video_data, f, indent=2, ensure_ascii=False)

        # 保存单独的文件以兼容现有系统
        txt_filepath = os.path.join(self.output_dir, f"{object_name}.txt")
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(user_instruction)

        seed_filepath = os.path.join(self.output_dir, f"{object_name}_seed.txt")
        with open(seed_filepath, 'w') as f:
            f.write(str(random_seed))

        # 保存轨迹数据为npy格式（兼容性）
        robot_traj_path = os.path.join(self.output_dir, f"{object_name}_robot.npy")
        obj_traj_path = os.path.join(self.output_dir, f"{object_name}_obj.npy")
        transit_path = os.path.join(self.output_dir, f"{object_name}_transit.npy")

        np.save(robot_traj_path, np.array(robot_trajectory_full))
        np.save(obj_traj_path, np.array(object_trajectory_full))
        np.save(transit_path, np.array([8, 27, 36]))

        logger.info(f"Saved video generation data to: {json_filepath}")
        logger.info(f"Robot trajectory: {len(robot_trajectory_full)} points")
        logger.info(f"Object trajectory: {len(object_trajectory_full)} points")
        logger.info(f"Random seed: {random_seed}")
        if mask_dest_path:
            logger.info(f"Object mask saved to: {mask_dest_path}")

        # 计算并返回机械臂退出位置（用于任务连续性）
        robot_exit_position = robot_exit_trajectory[-1] if robot_exit_trajectory else None

        return json_filepath, robot_exit_position

    def _save_trajectory_json_universal(self, object_name: str, start_point: Tuple[float, float],
                                        trajectory: List[List[float]], end_point: Tuple[float, float]) -> str:
        """
        保存通用轨迹数据为JSON文件

        Args:
            object_name: 物体名称
            start_point: 起始点坐标
            trajectory: 轨迹点列表
            end_point: 终点坐标

        Returns:
            JSON文件路径
        """
        trajectory_data = {
            "object_name": object_name,
            "metadata": {
                "created_at": "2025-10-25T08:00:00Z",
                "trajectory_type": "arc",
                "total_points": len(trajectory),
                "system": "universal_robot_controller"
            },
            "trajectory": {
                "start_point": {
                    "relative": list(start_point),
                    "pixels": {
                        "x": int(start_point[0] * 640),
                        "y": int(start_point[1] * 480)
                    }
                },
                "end_point": {
                    "relative": list(end_point),
                    "pixels": {
                        "x": int(end_point[0] * 640),
                        "y": int(end_point[1] * 480)
                    }
                },
                "intermediate_points": []
            }
        }

        # 添加中间点
        for i, point in enumerate(trajectory):
            trajectory_data["trajectory"]["intermediate_points"].append({
                "index": i + 1,
                "relative": point,
                "pixels": {
                    "x": int(point[0] * 640),
                    "y": int(point[1] * 480)
                }
            })

        # 保存JSON文件
        json_filename = f"{object_name}_trajectory.json"
        json_filepath = os.path.join(self.output_dir, json_filename)

        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved universal trajectory JSON to: {json_filepath}")
        return json_filepath

    def _copy_final_files_universal(self, object_name: str) -> Dict[str, str]:
        """
        复制最终输出文件到主输出目录的visualization子目录

        Args:
            object_name: 物体名称

        Returns:
            文件路径字典
        """
        file_mapping = {}

        # 创建visualization目录（如果不存在）
        visualization_dir = os.path.join(self.output_dir, "visualization")
        os.makedirs(visualization_dir, exist_ok=True)

        segmentation_dir = os.path.join(self.output_dir, "segmentation")

        # 复制最终mask
        mask_source = os.path.join(segmentation_dir, f"{object_name}_mask_iter1.png")
        if os.path.exists(mask_source):
            mask_dest = os.path.join(visualization_dir, f"{object_name}_final_mask.png")
            os.system(f"cp {mask_source} {mask_dest}")
            file_mapping["final_mask"] = mask_dest

        # 复制分割可视化
        seg_source = os.path.join(segmentation_dir, f"{object_name}_segmentation_iter1.png")
        if os.path.exists(seg_source):
            seg_dest = os.path.join(visualization_dir, f"{object_name}_segmentation.png")
            os.system(f"cp {seg_source} {seg_dest}")
            file_mapping["segmentation"] = seg_dest

        # 如果使用YOLOE辅助，复制YOLOE相关文件
        if False:  # 简化版不再使用YOLOE辅助
            # 复制YOLOE检测结果
            yoloe_source = os.path.join(segmentation_dir, "yoloe_detection_result.png")
            if os.path.exists(yoloe_source):
                yoloe_dest = os.path.join(visualization_dir, f"{object_name}_yoloe_detection.png")
                os.system(f"cp {yoloe_source} {yoloe_dest}")
                file_mapping["yoloe_detection"] = yoloe_dest

            # 复制mask后的图像
            masked_source = os.path.join(segmentation_dir, "masked_image.png")
            if os.path.exists(masked_source):
                masked_dest = os.path.join(visualization_dir, f"{object_name}_masked_image.png")
                os.system(f"cp {masked_source} {masked_dest}")
                file_mapping["masked_image"] = masked_dest

        # 复制轨迹可视化
        traj_source = os.path.join(self.output_dir, "movement", f"{object_name}_movement_trajectory.png")
        if os.path.exists(traj_source):
            traj_dest = os.path.join(visualization_dir, f"{object_name}_trajectory.png")
            os.system(f"cp {traj_source} {traj_dest}")
            file_mapping["trajectory"] = traj_dest

        # 复制日志文件到logs目录
        log_source = os.path.join(self.output_dir, "universal_robot_control.log")
        logs_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        if os.path.exists(log_source):
            log_dest = os.path.join(logs_dir, f"{object_name}_robot_control.log")
            os.system(f"cp {log_source} {log_dest}")
            file_mapping["robot_control_log"] = log_dest

        logger.info(f"Copied universal final files to visualization directory: {file_mapping}")
        return file_mapping

    def execute_universal_robot_task_structured(self, image_path: str, structured_subtask: Dict, previous_robot_exit_position: Optional[List[int]] = None) -> Dict:
        """
        执行基于结构化子任务的机器人任务

        Args:
            image_path: 输入图像路径
            structured_subtask: 包含ActionType、Object、Destination的结构化子任务数据
            previous_robot_exit_position: 上一个任务的机械臂退出位置

        Returns:
            任务执行结果字典
        """
        action_type = structured_subtask.get("ActionType", "")
        object_name = structured_subtask.get("Object", "")
        destination = structured_subtask.get("Destination", "")
        original_text = structured_subtask.get("original_text", "")

        logger.info(f"Starting structured robot task: {action_type} | {object_name} | {destination}")
        logger.info(f"Original text: {original_text}")

        # 步骤1：执行操作对象识别和定位（Object的完整mask + 可操作点）
        logger.info("Step 1: Executing affordance-based object segmentation")
        try:
            segmentation_result = self._execute_object_segmentation_universal(image_path, object_name, original_text)
            trajectory_start = segmentation_result["coordinates"]
            logger.info(f"Using affordance segmentation coordinates as trajectory start: {trajectory_start}")
        except RuntimeError as e:
            logger.error(f"Critical error in object segmentation: {str(e)}")
            return {
                "success": False,
                "error": f"Object segmentation failed: {str(e)}",
                "error_type": "affordance_detection_failed"
            }

        # 步骤2：执行目标定位（Destination定位）
        target_point: Optional[Tuple[float, float]] = None
        destination_name = destination

        # 过滤非空间目的地
        explicit_spatial_destination = False
        if destination_name:
            ambiguous_keywords = ["画面外", "出画面", "原位", "原地"]
            if not any(k in destination_name for k in ambiguous_keywords):
                explicit_spatial_destination = True
                logger.info(f"Detected explicit destination: {destination_name}")
                target_point = self._compute_destination_target_point(
                    image_path=image_path,
                    destination_name=destination_name,
                    user_instruction=original_text,
                    start_point=trajectory_start
                )
            else:
                logger.info(f"Destination '{destination_name}' treated as non-spatial")

        # 如果有明确的空间目的地但未能得到可靠终点，则报错
        if explicit_spatial_destination and target_point is None:
            logger.error(f"Explicit destination '{destination_name}' provided, but destination planning failed")
            return {
                "success": False,
                "error": f"Destination affordance planning failed for '{destination_name}'.",
                "error_type": "destination_affordance_failed"
            }

        # 步骤3：执行轨迹规划
        logger.info("Step 3: Executing universal trajectory planning")
        trajectory_result = self._execute_trajectory_planning_universal(
            image_path, object_name, original_text, trajectory_start, target_point
        )

        if not trajectory_result["success"]:
            logger.error(f"Universal trajectory planning failed: {trajectory_result['error']}")
            return {"success": False, "error": "Universal trajectory planning failed"}

        # 步骤3.5：生成简化的轨迹可视化
        logger.info("Step 3.5: Creating simplified trajectory visualization")
        try:
            # 获取三段式轨迹
            robot_approach_trajectory = trajectory_result.get("robot_approach_trajectory", [])
            object_trajectory_points = trajectory_result.get("object_trajectory_points", [])
            robot_exit_trajectory = trajectory_result.get("robot_exit_trajectory", [])

            # 获取对象mask路径
            object_mask_path = segmentation_result.get("final_mask_path")

            # 创建简化可视化
            simplified_viz_path = self._create_simplified_trajectory_visualization(
                image_path=image_path,
                object_name=object_name,
                object_mask_path=object_mask_path,
                affordance_point=trajectory_start,
                robot_approach_trajectory=robot_approach_trajectory,
                object_trajectory_points=object_trajectory_points,
                robot_exit_trajectory=robot_exit_trajectory
            )
            logger.info(f"Simplified trajectory visualization created: {simplified_viz_path}")
        except Exception as e:
            logger.warning(f"Failed to create simplified trajectory visualization: {e}")

        # 步骤4：生成视频生成数据
        logger.info("Step 4: Generating video generation data")

        # 准备增强提示词数据
        task_enhanced_prompts = {
            "action_type": action_type,
            "object_name": object_name,
            "destination": destination,
            "original_subtask": original_text,
            "structured_subtask": structured_subtask
        }

        video_data_path, robot_exit_position = self._save_video_generation_data(
            object_name,
            original_text,
            image_path,
            segmentation_result,
            trajectory_result,
            task_enhanced_prompts,
            previous_robot_exit_position
        )

        # 步骤5：生成轨迹JSON文件
        logger.info("Step 5: Generating universal trajectory JSON")
        trajectory_json_path = self._save_trajectory_json_universal(
            object_name,
            trajectory_start,
            trajectory_result["trajectory"],
            trajectory_result["end_point"]
        )

        logger.info("✅ Structured robot task completed successfully")
        logger.info(f"📹 Video data saved to: {video_data_path}")
        logger.info(f"📍 Trajectory saved to: {trajectory_json_path}")

        return {
            "success": True,
            "video_data_path": video_data_path,
            "trajectory_json_path": trajectory_json_path,
            "object_name": object_name,
            "action_type": action_type,
            "destination": destination,
            "original_text": original_text,
            "trajectory_start": trajectory_start,
            "trajectory_end": trajectory_result.get("end_point"),
            "robot_exit_position": robot_exit_position,
            "segmentation_result": segmentation_result,
            "trajectory_result": trajectory_result
        }

    def execute_universal_robot_task(self, image_path: str, user_instruction: str, enhanced_prompt_data: Optional[Dict] = None, previous_robot_exit_position: Optional[List[int]] = None) -> Dict:
        """
        执行完整的通用机器人任务

        Args:
            image_path: 输入图像路径
            user_instruction: 用户指令
            enhanced_prompt_data: 增强提示词数据（可选）
            previous_robot_exit_position: 上一个任务的机械臂退出位置（用于轨迹连续性）

        Returns:
            任务执行结果字典
        """
        logger.info(f"Starting universal robot task execution: {user_instruction}")

        # 如果提供了增强提示词数据，记录当前子任务的结构化信息
        structured_action = None
        structured_object = None
        structured_destination = None
        structured_subtask_text = None

        if enhanced_prompt_data:
            # 使用新的ActionType-Object-Destination结构
            structured_action = enhanced_prompt_data.get("action_type")
            structured_object = enhanced_prompt_data.get("object_name")
            structured_destination = enhanced_prompt_data.get("destination")
            structured_subtask_text = enhanced_prompt_data.get("original_subtask", user_instruction)

            # 如果新结构中没有数据，尝试从structured_subtask中获取
            if not structured_action or not structured_object or not structured_destination:
                structured_subtask = enhanced_prompt_data.get("structured_subtask", {})
                if structured_subtask:
                    structured_action = structured_action or structured_subtask.get("ActionType")
                    structured_object = structured_object or structured_subtask.get("Object")
                    structured_destination = structured_destination or structured_subtask.get("Destination")

            logger.info("Using enhanced prompt data for current subtask")
            logger.info(f"  original_subtask = {structured_subtask_text}")
            logger.info(f"  action_type       = {structured_action}")
            logger.info(f"  object_name       = {structured_object}")
            logger.info(f"  destination       = {structured_destination}")
        else:
            logger.info("No enhanced prompt data provided, using standard parsing")

        # 步骤1：解析用户指令（用于回退和冗余校验）
        logger.info("Step 1: Parsing universal user instruction")
        parsed_instruction = self._parse_user_instruction_universal(user_instruction)

        if not parsed_instruction["success"]:
            logger.error(f"Failed to parse universal instruction: {parsed_instruction['error']}")
            return {"success": False, "error": "Failed to parse instruction"}

        object_name = parsed_instruction["object_name"]
        movement_instruction = parsed_instruction["movement_instruction"]

        # 如果结构化信息中给出了更精确的对象与子任务描述，则优先使用
        if structured_object:
            object_name = structured_object
        if structured_subtask_text:
            movement_instruction = structured_subtask_text

        logger.info(f"Resolved universal instruction after fusion:")
        logger.info(f"  Object: {object_name}")
        logger.info(f"  Movement instruction: {movement_instruction}")

        # 步骤2：执行操作对象识别
        logger.info("Step 2: Executing affordance-only object segmentation")
        try:
            segmentation_result = self._execute_object_segmentation_universal(image_path, object_name, user_instruction)
            # 使用分割返回的坐标作为轨迹起点
            trajectory_start = segmentation_result["coordinates"]
            logger.info(f"Using affordance segmentation coordinates as trajectory start: {trajectory_start}")
        except RuntimeError as e:
            logger.error(f"Critical error in object segmentation: {str(e)}")
            logger.error("Affordance-based object detection failed - unable to continue without valid object mask")
            return {
                "success": False,
                "error": f"Object segmentation failed: {str(e)}. The system requires successful affordance detection to proceed.",
                "error_type": "affordance_detection_failed"
            }

        # 基于结构化 destination 信息，决定是否执行“明确终点”规划
        target_point: Optional[Tuple[float, float]] = None
        destination_name = None
        if structured_destination is not None:
            # destination 为 None 表示“无明确终点”；字符串则表示一个具体目标或区域
            if isinstance(structured_destination, str):
                destination_name = structured_destination.strip()

        explicit_spatial_destination = False
        if destination_name:
            # 对于明确终点，过滤掉“画面外/原位”等非空间容器描述
            ambiguous_keywords = ["画面外", "出画面", "原位", "原地"]
            if not any(k in destination_name for k in ambiguous_keywords):
                explicit_spatial_destination = True
                logger.info(f"Detected explicit destination for trajectory planning: {destination_name}")
                target_point = self._compute_destination_target_point(
                    image_path=image_path,
                    destination_name=destination_name,
                    user_instruction=user_instruction,
                    start_point=trajectory_start
                )
            else:
                logger.info(f"Destination '{destination_name}' treated as non-spatial (e.g., out-of-frame); using fuzzy planning.")
        else:
            logger.info("No explicit spatial destination provided; using fuzzy trajectory planning based on instruction alone.")

        # 若存在明确的空间目的地但多轮 Affordance 规划仍未能给出可靠终点，则直接报错，
        # 避免退回到粗糙的网格规则规划，防止“看起来成功但落点错误”的情况。
        if explicit_spatial_destination and target_point is None:
            logger.error(
                f"Explicit destination '{destination_name}' provided, but affordance-based destination "
                f"planning failed to produce a reliable target point."
            )
            return {
                "success": False,
                "error": f"Destination affordance planning failed for '{destination_name}'.",
                "error_type": "destination_affordance_failed"
            }

        # 步骤3：执行轨迹规划
        logger.info("Step 3: Executing universal trajectory planning")
        trajectory_result = self._execute_trajectory_planning_universal(
            image_path, object_name, movement_instruction, trajectory_start, target_point
        )

        if not trajectory_result["success"]:
            logger.error(f"Universal trajectory planning failed: {trajectory_result['error']}")
            return {"success": False, "error": "Universal trajectory planning failed"}

        # 步骤3.5：生成简化的轨迹可视化
        logger.info("Step 3.5: Creating simplified trajectory visualization")
        try:
            # 获取三段式轨迹
            robot_approach_trajectory = trajectory_result.get("robot_approach_trajectory", [])
            object_trajectory_points = trajectory_result.get("object_trajectory_points", [])
            robot_exit_trajectory = trajectory_result.get("robot_exit_trajectory", [])

            # 获取对象mask路径
            object_mask_path = segmentation_result.get("final_mask_path")

            # 创建简化可视化
            simplified_viz_path = self._create_simplified_trajectory_visualization(
                image_path=image_path,
                object_name=object_name,
                object_mask_path=object_mask_path,
                affordance_point=trajectory_start,
                robot_approach_trajectory=robot_approach_trajectory,
                object_trajectory_points=object_trajectory_points,
                robot_exit_trajectory=robot_exit_trajectory
            )
            logger.info(f"Simplified trajectory visualization created: {simplified_viz_path}")
        except Exception as e:
            logger.warning(f"Failed to create simplified trajectory visualization: {e}")

        # 步骤4：生成视频生成数据
        logger.info("Step 4: Generating video generation data")

        # 从增强提示词数据中提取当前任务的提示词
        task_enhanced_prompts = None
        if enhanced_prompt_data and "subtasks" in enhanced_prompt_data:
            # 查找与当前object_name匹配的子任务
            for subtask in enhanced_prompt_data["subtasks"]:
                if object_name in subtask.get("original_subtask", ""):
                    task_enhanced_prompts = subtask
                    logger.info(f"Found enhanced prompts for {object_name}")
                    break

        video_data_path, robot_exit_position = self._save_video_generation_data(
            object_name,
            user_instruction,
            image_path,
            segmentation_result,
            trajectory_result,
            task_enhanced_prompts,
            previous_robot_exit_position
        )

        # 步骤5：生成轨迹JSON文件
        logger.info("Step 5: Generating universal trajectory JSON")
        trajectory_json_path = self._save_trajectory_json_universal(
            object_name,
            trajectory_start,
            trajectory_result["trajectory"],
            trajectory_result["end_point"]
        )

        # 步骤6：复制最终文件
        logger.info("Step 6: Copying universal final output files")
        final_files = self._copy_final_files_universal(object_name)

        # 构建最终结果
        result = {
            "success": True,
            "user_instruction": user_instruction,
            "parsed_instruction": parsed_instruction,
            "object_name": object_name,
            "movement_instruction": movement_instruction,
            "segmentation_result": segmentation_result,
            "trajectory_result": trajectory_result,
            "trajectory_start": trajectory_start,
            "video_data_path": video_data_path,
            "trajectory_json_path": trajectory_json_path,
            "final_files": final_files,
            "output_directory": self.output_dir,
            "robot_exit_position": robot_exit_position,  # 添加机械臂退出位置用于任务连续性
            "system": "universal_robot_controller"
        }

        logger.info(f"Universal robot task completed successfully: {object_name}")
        return result


def main():
    """主函数示例 - 通用机器人控制"""
    # 初始化通用机器人控制器
    controller = UniversalRobotController(
        output_dir="/data/rczhang/MIND-V/vlm_api/universal_robot_output"
    )

    # 示例1：物体移动场景
    # image_path = "/data/rczhang/MIND-V/demos/diverse_ood_objs/avocado.png"
    # user_instruction = "将右边的牛油果移动到桌子左侧"

    # 示例2：抓取场景
    # image_path = "/path/to/image.png"
    # user_instruction = "拿起左边的杯子"

    # 示例3：滚动场景
    # image_path = "/path/to/image.png"
    # user_instruction = "把中间的球向右滚动"

    # 使用当前可用的测试图像
    image_path = "/data/rczhang/MIND-V/demos/diverse_ood_objs/avocado.png"
    user_instruction = "将右边的牛油果移动到桌子左侧"

    # 执行通用机器人任务
    result = controller.execute_universal_robot_task(
        image_path=image_path,
        user_instruction=user_instruction
    )

    # 输出结果
    if result["success"]:
        print("✅ 通用机器人任务执行成功！")
        print(f"用户指令: {result['user_instruction']}")
        print(f"操作对象: {result['object_name']}")
        print(f"移动指令: {result['movement_instruction']}")
        print(f"轨迹起点: {result['trajectory_start']}")
        print(f"轨迹规划起点: {result['trajectory_result']['start_point']}")
        print(f"轨迹终点: {result['trajectory_result']['end_point']}")
        print(f"轨迹点数量: {result['trajectory_result']['num_trajectory_points']}")
        print(f"位置调整次数: {result['segmentation_result']['position_iterations']}")
        print(f"分割尝试次数: {result['segmentation_result']['segmentation_iterations']}")
        print(f"系统: {result['system']}")
        print(f"输出目录: {result['output_directory']}")
        print("\n生成的文件:")
        for file_type, file_path in result['final_files'].items():
            print(f"  {file_type}: {file_path}")
        print(f"  trajectory_json: {result['trajectory_json_path']}")
    else:
        print("❌ 通用机器人任务执行失败")
        print(f"错误信息: {result['error']}")


if __name__ == "__main__":
    main()
