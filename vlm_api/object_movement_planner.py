#!/usr/bin/env python3
"""
物体移动轨迹规划程序 (Object Movement Trajectory Planner)
结合Gemini API实现智能物体移动轨迹规划

工作流程：
1. 使用Gemini API规划物体移动的终点位置
2. 在终点位置标记黄色五角星并返回给API确认
3. 根据反馈调整终点位置，直到确认准确
4. 计算起点（mask中心）到终点的弧形轨迹
5. 对轨迹进行采样并可视化保存
"""

import os
import json
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from PIL import Image
import cv2
import math
from typing import Tuple, Optional, List, Dict
from loguru import logger

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ObjectMovementPlanner:
    def __init__(self,
                 output_dir="./movement_output"):
        """
        初始化简化版物体移动轨迹规划器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 设置日志
        logger.add(f"{output_dir}/movement_planning.log", rotation="10 MB")

    def _load_image(self, image_path: str) -> np.ndarray:
        """加载图像并转换为RGB格式"""
        try:
            image = Image.open(image_path)
            image = np.array(image.convert("RGB"))
            logger.info(f"Loaded image: {image.shape}")
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise

    def _save_image_with_star(self, image: np.ndarray, point: Tuple[float, float],
                              output_path: str, star_size: int = 20):
        """
        在图像上绘制黄色五角星并保存

        Args:
            image: 输入图像
            point: 五角星坐标 (x, y) 相对坐标
            output_path: 输出路径
            star_size: 五角星大小
        """
        # 创建图像副本
        img_with_star = image.copy()

        # 转换坐标
        h, w = image.shape[:2]
        x = int(point[0] * w)
        y = int(point[1] * h)

        # 计算五角星的顶点
        outer_radius = star_size
        inner_radius = star_size * 0.4

        points = []
        for i in range(10):
            angle = math.pi * i / 5 - math.pi / 2
            if i % 2 == 0:  # 外顶点
                px = x + outer_radius * math.cos(angle)
                py = y + outer_radius * math.sin(angle)
            else:  # 内顶点
                px = x + inner_radius * math.cos(angle)
                py = y + inner_radius * math.sin(angle)
            points.append([px, py])

        points = np.array(points, dtype=np.int32)

        # 绘制黄色五角星
        cv2.fillPoly(img_with_star, [points], (0, 255, 255))  # 黄色 (BGR格式)
        cv2.polylines(img_with_star, [points], True, (0, 200, 200), 2)  # 边框

        # 保存图像
        cv2.imwrite(output_path, cv2.cvtColor(img_with_star, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved image with star to: {output_path}")

    def _parse_coordinates(self, text: str) -> Optional[Tuple[float, float]]:
        """
        从文本中解析坐标

        Args:
            text: 包含坐标的文本

        Returns:
            坐标元组 (x, y) 或 None
        """
        # 查找坐标模式 - 支持负数坐标
        patterns = [
            r'坐标[：:\s]*\(?([+-]?[0-9.]+)\s*[,，]\s*([+-]?[0-9.]+)\)?',
            r'coordinate[：:\s]*\(?([+-]?[0-9.]+)\s*[,，]\s*([+-]?[0-9.]+)\)?',
            r'\(?([+-]?[0-9.]+)\s*[,，]\s*([+-]?[0-9.]+)\)?',
            r'终点[：:\s]*\(?([+-]?[0-9.]+)\s*[,，]\s*([+-]?[0-9.]+)\)?',
            r'目标[：:\s]*\(?([+-]?[0-9.]+)\s*[,，]\s*([+-]?[0-9.]+)\)?',
            r'x[：:\s]*([+-]?[0-9.]+)\s*[,\s]+y[：:\s]*([+-]?[0-9.]+)',
            r'x[：:\s]*([+-]?[0-9.]+)\s*[，\s]+y[：:\s]*([+-]?[0-9.]+)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    x, y = float(matches[0][0]), float(matches[0][1])

                    # 边界约束到[0,1]范围
                    original_x, original_y = x, y
                    needs_clamping = False

                    if x < 0:
                        x = 0.0
                        needs_clamping = True
                    elif x > 1:
                        x = 1.0
                        needs_clamping = True

                    if y < 0:
                        y = 0.0
                        needs_clamping = True
                    elif y > 1:
                        y = 1.0
                        needs_clamping = True

                    if needs_clamping:
                        logger.warning(f"Coordinates bounded: ({original_x}, {original_y}) -> ({x}, {y})")
                    else:
                        logger.info(f"Parsed coordinates: ({x}, {y})")

                    return (x, y)
                except ValueError:
                    continue

        logger.warning(f"Could not parse coordinates from: {text}")
        return None

    def _parse_adjustment_suggestion(self, response: str, current_coord: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        从API响应中解析调整建议并计算新坐标

        Args:
            response: API响应文本
            current_coord: 当前坐标 (x, y) 相对坐标

        Returns:
            新坐标或None
        """
        # 查找方向性调整建议 - 增加调整幅度以减少迭代次数
        adjustments = {
            "向上": (0, -0.08), "向下": (0, 0.08), "向左": (-0.08, 0), "向右": (0.08, 0),
            "往上": (0, -0.08), "往下": (0, 0.08), "往左": (-0.08, 0), "往右": (0.08, 0),
            "up": (0, -0.08), "down": (0, 0.08), "left": (-0.08, 0), "right": (0.08, 0)
        }

        # 查找数值调整建议
        numeric_patterns = [
            r'x[坐标]?[：:\s]*([+-]?[0-9.]+)',
            r'y[坐标]?[：:\s]*([+-]?[0-9.]+)',
            r'横坐标?[：:\s]*([+-]?[0-9.]+)',
            r'纵坐标?[：:\s]*([+-]?[0-9.]+)'
        ]

        response_lower = response.lower()

        # 检查方向性调整
        for direction, (dx, dy) in adjustments.items():
            if direction in response_lower:
                new_x = max(0, min(1, current_coord[0] + dx))
                new_y = max(0, min(1, current_coord[1] + dy))
                logger.info(f"Adjustment suggestion: {direction}, new coordinates: ({new_x}, {new_y})")
                return (new_x, new_y)

        # 检查数值调整
        new_x, new_y = current_coord
        for pattern in numeric_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                try:
                    value = float(matches[0])
                    if 'x' in pattern.lower() or '横' in pattern:
                        new_x = max(0, min(1, current_coord[0] + value))
                    elif 'y' in pattern.lower() or '纵' in pattern:
                        new_y = max(0, min(1, current_coord[1] + value))
                except ValueError:
                    continue

        if (new_x, new_y) != current_coord:
            logger.info(f"Numeric adjustment: new coordinates: ({new_x}, {new_y})")
            return (new_x, new_y)

        # 如果没有明确的调整建议，返回None
        logger.warning("No clear adjustment suggestion found")
        return None

    def _is_position_correct(self, response: str) -> Tuple[bool, bool]:
        """
        判断API返回的位置确认是否为正确，支持早停机制

        Args:
            response: API响应文本

        Returns:
            (is_correct, can_stop) 是否正确和是否可以早停
        """
        response_lower = response.lower().strip()

        # 完全正确 - 可以早停
        perfect_correct_keywords = [
            "正确", "对", "是的", "准确", "精准", "正好", "恰好在", "就是这里", "可以", "合适", "完全适合",
            "correct", "yes", "accurate", "precise", "exactly", "right", "good", "ok", "acceptable", "perfect"
        ]

        # 基本正确 - 可以早停
        good_enough_keywords = [
            "基本适合", "基本可以", "基本正确", "基本合适", "基本符合", "基本ok",
            "大部分满足", "主要满足", "基本满足要求", "基本可行"
        ]

        # 明确表示错误的关键词
        incorrect_keywords = [
            "错误", "不对", "不是", "偏差", "偏了", "移动", "调整", "需要修改", "太偏", "不够",
            "incorrect", "wrong", "no", "adjust", "move", "shift", "bias", "too far", "not enough", "需要调整"
        ]

        # 检查完全正确 - 优先匹配
        for keyword in perfect_correct_keywords:
            if keyword in response_lower:
                logger.info(f"Position marked as perfect correct (found: {keyword}) - EARLY STOP")
                return True, True

        # 检查基本正确 - 也可以早停
        for keyword in good_enough_keywords:
            if keyword in response_lower:
                logger.info(f"Position marked as good enough (found: {keyword}) - EARLY STOP")
                return True, True

        # 检查需要调整
        for keyword in incorrect_keywords:
            if keyword in response_lower:
                logger.info(f"Position marked as needing adjustment (found: {keyword})")
                return False, False

        # 如果没有明确关键词，默认为需要调整（保守策略）
        logger.warning("No clear confirmation in response, assuming position needs adjustment")
        return False, False

    def _calculate_mask_center(self, mask: np.ndarray) -> Tuple[float, float]:
        """
        计算掩码的中心点坐标

        Args:
            mask: 掩码数组

        Returns:
            中心点相对坐标 (x, y)
        """
        # 获取掩码的像素坐标
        y_indices, x_indices = np.where(mask)

        if len(y_indices) == 0:
            logger.warning("Empty mask, returning center of image")
            return (0.5, 0.5)

        # 计算质心
        center_y = np.mean(y_indices)
        center_x = np.mean(x_indices)

        # 转换为相对坐标
        h, w = mask.shape
        rel_x = center_x / w
        rel_y = center_y / h

        logger.info(f"Mask center: ({rel_x:.3f}, {rel_y:.3f}) in relative coordinates")
        return (rel_x, rel_y)

    def _calculate_arc_trajectory(self, start_point: Tuple[float, float],
                               end_point: Tuple[float, float],
                               num_points: int = 20) -> np.ndarray:
        """
        计算起点到终点的弧形轨迹

        Args:
            start_point: 起点相对坐标 (x, y)
            end_point: 终点相对坐标 (x, y)
            num_points: 轨迹采样点数量

        Returns:
            轨迹点数组，形状为 (num_points, 2)
        """
        # 计算直线距离
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        distance = math.sqrt(dx**2 + dy**2)

        # 计算控制点（使轨迹呈弧形）
        # 控制点在起点和终点连线的垂直方向上偏移
        mid_point = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)

        # 垂直向量（归一化）
        if distance > 0:
            perp_x = -dy / distance
            perp_y = dx / distance
        else:
            perp_x, perp_y = 0, 1

        # 弧形高度（与距离成正比）
        arc_height = distance * 0.3  # 弧形高度为直线距离的30%

        # 控制点
        control_point = (
            mid_point[0] + perp_x * arc_height,
            mid_point[1] + perp_y * arc_height
        )

        logger.info(f"Trajectory: start={start_point}, end={end_point}, control={control_point}")

        # 生成二次贝塞尔曲线上的点
        trajectory_points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            # 二次贝塞尔曲线公式
            x = (1-t)**2 * start_point[0] + 2*(1-t)*t * control_point[0] + t**2 * end_point[0]
            y = (1-t)**2 * start_point[1] + 2*(1-t)*t * control_point[1] + t**2 * end_point[1]
            trajectory_points.append([x, y])

        return np.array(trajectory_points)

    def _visualize_trajectory(self, image: np.ndarray, start_point: Tuple[float, float],
                             end_point: Tuple[float, float], trajectory: np.ndarray,
                             object_name: str) -> str:
        """
        可视化轨迹并保存图像

        Args:
            image: 原始图像
            start_point: 起点相对坐标
            end_point: 终点相对坐标
            trajectory: 轨迹点数组
            object_name: 物体名称

        Returns:
            保存的文件路径
        """
        # 创建可视化图像
        plt.figure(figsize=(12, 8))

        # 显示原始图像
        plt.imshow(image)

        # 转换坐标
        h, w = image.shape[:2]
        start_abs = (int(start_point[0] * w), int(start_point[1] * h))
        end_abs = (int(end_point[0] * w), int(end_point[1] * h))
        trajectory_abs = trajectory * np.array([w, h])

        # 绘制轨迹
        plt.plot(trajectory_abs[:, 0], trajectory_abs[:, 1], 'b-', linewidth=3,
                label='Movement Trajectory', alpha=0.7)

        # 绘制轨迹点
        plt.scatter(trajectory_abs[:, 0], trajectory_abs[:, 1], c='blue', s=30, alpha=0.6)

        # 绘制起点（绿色圆圈）
        plt.scatter(start_abs[0], start_abs[1], c='green', s=200, marker='o',
                   edgecolors='white', linewidth=2, label='Start Point', zorder=5)

        # 绘制终点（红色五角星）
        star_size = 20
        outer_radius = star_size
        inner_radius = star_size * 0.4

        star_points = []
        for i in range(10):
            angle = math.pi * i / 5 - math.pi / 2
            if i % 2 == 0:
                px = end_abs[0] + outer_radius * math.cos(angle)
                py = end_abs[1] + outer_radius * math.sin(angle)
            else:
                px = end_abs[0] + inner_radius * math.cos(angle)
                py = end_abs[1] + inner_radius * math.sin(angle)
            star_points.append([px, py])

        star_points = np.array(star_points)
        plt.scatter(end_abs[0], end_abs[1], c='red', s=500, marker='*',
                   edgecolors='white', linewidth=2, label='End Point', zorder=5)

        # 添加坐标标注
        plt.annotate(f'Start\n({start_point[0]:.2f}, {start_point[1]:.2f})',
                    xy=start_abs, xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                    fontsize=10, color='white')

        plt.annotate(f'End\n({end_point[0]:.2f}, {end_point[1]:.2f})',
                    xy=end_abs, xytext=(10, -30), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                    fontsize=10, color='white')

        plt.title(f'Object Movement Trajectory: {object_name}', fontsize=14, fontweight='bold')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.axis('on')

        # 保存图像
        filename = f"{object_name}_movement_trajectory.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

        logger.info(f"Saved trajectory visualization to: {filepath}")
        return filepath

    def _save_trajectory_points(self, trajectory: np.ndarray, object_name: str) -> str:
        """
        保存轨迹点数据到文件

        Args:
            trajectory: 轨迹点数组
            object_name: 物体名称

        Returns:
            保存的文件路径
        """
        filename = f"{object_name}_trajectory_points.txt"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Movement Trajectory Points for {object_name}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total points: {len(trajectory)}\n\n")
            f.write("Index\tX (relative)\tY (relative)\tX (pixels)\tY (pixels)\n")
            f.write("-" * 50 + "\n")

            for i, point in enumerate(trajectory):
                f.write(f"{i+1}\t{point[0]:.4f}\t\t{point[1]:.4f}\t\t")
                f.write(f"{int(point[0] * 640)}\t\t{int(point[1] * 480)}\n")

        logger.info(f"Saved trajectory points to: {filepath}")
        return filepath

    def _calculate_target_position_from_rules(self, start_point: Tuple[float, float], movement_instruction: str) -> Optional[Tuple[float, float]]:
        """
        基于规则从移动指令计算终点位置

        Args:
            start_point: 起点相对坐标 (x, y)
            movement_instruction: 移动指令

        Returns:
            终点相对坐标 (x, y) 或 None
        """
        instruction = movement_instruction.lower()

        # 定义移动向量
        move_distance = 0.3  # 移动距离

        # 方向关键词映射
        directions = {
            "左": (-move_distance, 0),
            "右": (move_distance, 0),
            "上": (0, -move_distance),  # 图像坐标系中y向上是减少
            "下": (0, move_distance),
            "左上": (-move_distance * 0.7, -move_distance * 0.7),
            "右上": (move_distance * 0.7, -move_distance * 0.7),
            "左下": (-move_distance * 0.7, move_distance * 0.7),
            "右下": (move_distance * 0.7, move_distance * 0.7)
        }

        # 位置关键词映射
        positions = {
            "中间": (0.5, 0.5),
            "中心": (0.5, 0.5),
            "左边": (0.2, 0.5),
            "右侧": (0.8, 0.5),
            "上面": (0.5, 0.2),
            "下方": (0.5, 0.8),
            "左上角": (0.2, 0.2),
            "右上角": (0.8, 0.2),
            "左下角": (0.2, 0.8),
            "右下角": (0.8, 0.8)
        }

        # 1. 检查绝对位置指令（如"移到左边"、"放到中间"）
        for pos_keyword, pos_coord in positions.items():
            if pos_keyword in instruction:
                logger.info(f"Found absolute position '{pos_keyword}', using coordinates: {pos_coord}")
                return pos_coord

        # 2. 检查相对移动指令（如"向左移动"、"向右推"）
        for dir_keyword, dir_vector in directions.items():
            if dir_keyword in instruction:
                target_x = start_point[0] + dir_vector[0]
                target_y = start_point[1] + dir_vector[1]
                logger.info(f"Found direction '{dir_keyword}', applying vector: {dir_vector}")
                return (target_x, target_y)

        # 3. 特殊模式匹配
        if "拿起" in instruction or "抓取" in instruction:
            # 抓取动作 - 向物体靠近
            logger.info("Detected grabbing action, keeping object position")
            return start_point

        if "滚动" in instruction:
            # 滚动动作 - 根据起始位置判断滚动方向
            if start_point[0] < 0.5:  # 在左边，向右滚动
                return (min(start_point[0] + move_distance, 0.9), start_point[1])
            else:  # 在右边，向左滚动
                return (max(start_point[0] - move_distance, 0.1), start_point[1])

        # 4. 默认行为：如果无法解析，向右移动一点
        logger.warning("Could not parse movement instruction, using default rightward movement")
        return (min(start_point[0] + 0.2, 0.9), start_point[1])

    def plan_object_movement(self, image_path: str, object_name: str,
                           movement_instruction: str, start_point: Tuple[float, float],
                           max_position_attempts: int = 5,
                           explicit_target: Optional[Tuple[float, float]] = None) -> Dict:
        """
        执行简化版物体移动轨迹规划 - 基于规则的终点计算

        Args:
            image_path: 输入图像路径
            object_name: 要移动的物体名称
            movement_instruction: 移动指令（如："将右边的牛油果移动到桌子的左侧"）
            start_point: 起点相对坐标 (x, y)
            max_position_attempts: 未使用，保持接口兼容

        Returns:
            包含结果的字典
        """
        logger.info(f"Starting rule-based movement planning for: {object_name}")
        logger.info(f"Movement instruction: {movement_instruction}")
        logger.info(f"Start point: {start_point}")

        # 加载图像尺寸（不需要加载实际图像数据）
        try:
            image = self._load_image(image_path)
            h, w = image.shape[:2]
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}, using default dimensions: {e}")
            w, h = 640, 480

        # 步骤1：确定终点位置
        if explicit_target is not None:
            logger.info(f"Step 1: Using explicit target coordinate from upstream planning: {explicit_target}")
            target_coord = explicit_target
        else:
            logger.info("Step 1: Calculating target position using rule-based planner")
            target_coord = self._calculate_target_position_from_rules(start_point, movement_instruction)

            if not target_coord:
                logger.error("Failed to calculate target position from movement instruction")
                return {"success": False, "error": "Failed to parse movement instruction"}

        # 确保坐标在有效范围内
        target_coord = (
            max(0.1, min(0.9, target_coord[0])),
            max(0.1, min(0.9, target_coord[1]))
        )

        logger.info(f"Final target coordinates (clamped): {target_coord}")

        # 步骤2：计算弧形轨迹
        logger.info("Step 2: Calculating arc trajectory")
        trajectory = self._calculate_arc_trajectory(start_point, target_coord, num_points=20)

        # 步骤3：可视化轨迹
        logger.info("Step 3: Visualizing trajectory")
        trajectory_image_path = self._visualize_trajectory(
            image, start_point, target_coord, trajectory, object_name
        )

        # 步骤4：保存轨迹点数据
        trajectory_data_path = self._save_trajectory_points(trajectory, object_name)

        # 返回成功结果
        result = {
            "success": True,
            "object_name": object_name,
            "movement_instruction": movement_instruction,
            "start_point": start_point,
            "end_point": target_coord,
            "trajectory": trajectory.tolist(),
            "trajectory_image_path": trajectory_image_path,
            "trajectory_data_path": trajectory_data_path,
            "position_iterations": 1,
            "num_trajectory_points": len(trajectory)
        }

        logger.info(f"Movement planning completed successfully: {result}")
        return result


def main():
    """主函数示例"""
    # 初始化移动规划器
    planner = ObjectMovementPlanner(
        output_dir="/data/rczhang/MIND-V/vlm_api/movement_output"
    )

    # 图像路径
    image_path = "/data/rczhang/MIND-V/demos/diverse_ood_objs/avocado.png"

    # 执行移动规划（将右边的牛油果移动到桌子的左侧）
    result = planner.plan_object_movement(
        image_path=image_path,
        object_name="右边的牛油果",
        movement_instruction="将右边的牛油果移动到桌子的左侧",
        start_point=(0.84, 0.75)  # 基于日志中的坐标 (537, 311) 转换为相对坐标
    )

    # 输出结果
    if result["success"]:
        print("✅ 移动轨迹规划成功！")
        print(f"物体名称: {result['object_name']}")
        print(f"移动指令: {result['movement_instruction']}")
        print(f"起始点: {result['start_point']}")
        print(f"结束点: {result['end_point']}")
        print(f"轨迹点数量: {result['num_trajectory_points']}")
        print(f"位置调整次数: {result['position_iterations']}")
        print(f"轨迹可视化图像: {result['trajectory_image_path']}")
        print(f"轨迹数据文件: {result['trajectory_data_path']}")
    else:
        print("❌ 移动轨迹规划失败")
        print(f"错误信息: {result['error']}")


if __name__ == "__main__":
    main()
