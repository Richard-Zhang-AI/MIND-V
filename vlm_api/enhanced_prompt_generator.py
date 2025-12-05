#!/usr/bin/env python3
"""
增强提示词生成器 - Enhanced Prompt Generator
为机械臂操作任务生成优化的正向和负向提示词

Enhanced Prompt Generator for Robotic Arm Manipulation Tasks
Generates optimized positive and negative prompts for robotic operations
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple
from loguru import logger

from vlm_api.gemini_api import call_gemini_api


class EnhancedPromptGenerator:
    """
    增强提示词生成器
    负责为长程机械臂任务生成优化的提示词
    """

    def __init__(self,
                 gemini_model="gemini-2.5-flash",
                 gemini_region="us-central1",
                 gemini_project="captioner-test-1017",
                 gemini_credential_file="/data/rczhang/MIND-V/vlm_api/captioner.json"):
        """
        初始化增强提示词生成器

        Args:
            gemini_model: Gemini模型名称
            gemini_region: Google Cloud区域
            gemini_project: 项目ID
            gemini_credential_file: 凭证文件路径
        """
        self.gemini_model = gemini_model
        self.gemini_region = gemini_region
        self.gemini_project = gemini_project
        self.gemini_credential_file = gemini_credential_file

        logger.info("Enhanced Prompt Generator initialized")

    def _call_gemini_with_context(self, prompt: str, image_path: Optional[str] = None) -> str:
        """
        调用Gemini API，带有系统上下文
        """
        # 构建完整的提示词，包含系统上下文
        full_prompt = f"""
你是一个专业的机械臂操作视频生成AI助手。你的任务是为长程机械臂操作生成优化的提示词。

## 你的核心职责：
1. 理解复杂的机械臂操作指令
2. 将其分解为有序的子任务
3. 为每个子任务生成优化的正向引导词和负向提示词
4. 确保视频生成质量，避免常见的生成问题

## 机械臂操作的基本原则：
- 操作应该自然流畅，符合物理规律
- 动作应该精确、稳定，避免抖动
- 轨迹应该合理，避免碰撞和异常路径
- 操作对象应该保持清晰的识别度

## 视频生成质量要求：
- 画面清晰度高，细节丰富
- 动作流畅，帧率稳定
- 光照自然，色彩真实
- 避免畸变、模糊、伪影等问题
- 背景应该合理，与场景协调

## 常见需要避免的问题（固定负向提示词）：
- Worst quality, object deformation, normal quality, low quality
- low resolution, blurry, distorted, wrong, sketch, duplicate
- ugly, monochrome, horror, geometric shapes, mutation, disgusting
- poor anatomy, disproportionate, inferior, malformed
- out of frame, out of focus, shriveled, disfigured
- extra mechanical arms, odd proportions, jpeg

{prompt}
"""

        try:
            response = call_gemini_api(
                prompt=full_prompt,
                image_path=image_path,
                model=self.gemini_model,
                region=self.gemini_region,
                project=self.gemini_project,
                credential_file=self.gemini_credential_file,
                temperature=0.1,
                seed=42
            )
            return response
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            return None

    def generate_enhanced_prompts_for_task(self, instruction: str, image_path: str) -> Dict[str, any]:
        """
        为整个长程任务生成增强的提示词

        Args:
            instruction: 用户的复杂指令
            image_path: 输入图像路径

        Returns:
            包含任务分解和提示词的字典
        """
        logger.info(f"Generating enhanced prompts for task: {instruction}")

        # 步骤1：分析任务并分解
        task_decomposition_prompt = f"""
请分析以下机械臂操作任务并分解为高层子任务：

原始指令：{instruction}
场景图像：已提供

分解原则（严格遵守）：
1. 控制在1-3个高层子任务，尽量少而完整；不要将一个自然动作拆成“抓取/移动/放置”等微步骤。
2. 每个子任务必须显式包含：动作原语(ActionType) + 对象(Object) + 目标位置/容器(Destination)。例如“把X放入Y”、“将X移动到Y”、“将X移出画面”。如目标缺省，请明确写为“画面外/原位”等，并在下面结构化字段中将 destination 设为 null。
3. 子任务按自然顺序排列，对应“再/然后/接着”等逻辑连接。
4. 每个子任务必须“独立完整”，不得在同一字符串中继续包含“再/然后/接着/并且/同时”等连接词；若存在此类连接，请拆分成两条。

Few-shot 示例：
- 输入: "把白柄蘑菇玩具放入金属锅中"
  输出: [
    {{
      "original_subtask": "把白柄蘑菇玩具放入金属锅中",
      "action": "抓取并放置",
      "object": "白柄蘑菇玩具",
      "destination": "金属锅",
      "positive_prompt": "...",
      "negative_prompt": "..."
    }}
  ]

- 输入: "把左边的勺子放进锅里，再把蓝色抹布放进锅里，再把锅拿出画面"
  输出: [
    {{
      "original_subtask": "把左边的勺子放进锅里",
      "action": "抓取并放置",
      "object": "左边的勺子",
      "destination": "锅",
      "positive_prompt": "...",
      "negative_prompt": "..."
    }},
    {{
      "original_subtask": "把蓝色抹布放进锅里",
      "action": "抓取并放置",
      "object": "蓝色抹布",
      "destination": "锅",
      "positive_prompt": "...",
      "negative_prompt": "..."
    }},
    {{
      "original_subtask": "把锅拿出画面",
      "action": "抓取并移动出画面",
      "object": "锅",
      "destination": "画面外",
      "positive_prompt": "...",
      "negative_prompt": "..."
    }}
  ]

要求：
- 输出为JSON数组，数组元素为对象，必须包含字段：
  - original_subtask: 子任务原文（动作+对象+目标 的中文短句）
  - action: 动作原语（如“抓取并放置”、“抓取并移动到目标区域”、“抓取并抬起”等，需简洁稳定）
  - object: 被操作的实体名称（如“毛巾”、“左边的勺子”）
  - destination: 目标位置/容器；若有明确目标物体或区域，写成简洁名词短语（如“金属锅”、“桌子右侧区域”）；若无明确目标，则严格写为 null
  - positive_prompt: 针对该子任务的视频正向提示词（中文字符串）
  - negative_prompt: 针对该子任务的视频负向提示词（中文字符串）
- original_subtask 必须独立完整，不含“再/然后/接着/并且/同时”等连接词。
- destination 字段在 JSON 中必须出现；没有明确终点时必须设置为 null（不要使用空字符串、\"无\"、\"None\"、\"N/A\" 等）。
- 请只返回符合上述结构的 JSON 数组，不要包含任何其他解释文字或注释。
"""

        # 发送请求
        response = self._call_gemini_with_context(task_decomposition_prompt, image_path)
        if not response:
            logger.error("Failed to get enhanced task decomposition from AI")
            return self._fallback_response(instruction)

        logger.info(f"Enhanced task decomposition received: {response[:100]}...")

        # 解析增强子任务
        try:
            enhanced_subtasks = self._parse_enhanced_subtasks_response(response)
            logger.info(f"Enhanced task decomposition completed: {len(enhanced_subtasks)} subtasks")
        except Exception as e:
            logger.error(f"Failed to parse enhanced subtasks: {str(e)}")
            return self._fallback_response(instruction)

        # 构建最终结果
        result = {
            "success": True,
            "original_instruction": instruction,
            "task_analysis": response,
            "subtasks": enhanced_subtasks,
            "total_subtasks": len(enhanced_subtasks),
            "enhanced_by_ai": True
        }

        logger.info(f"Enhanced prompt generation completed for {len(enhanced_subtasks)} subtasks")
        return result

    def _generate_single_task_prompts(self, subtask: str, task_index: int) -> Dict[str, str]:
        """
        为单个子任务生成优化的正向和负向提示词

        Args:
            subtask: 子任务描述
            task_index: 任务索引

        Returns:
            包含增强提示词的字典
        """
        logger.info(f"Generating enhanced prompts for subtask {task_index}: {subtask}")

        prompt_request = f"""
现在请为第{task_index}个子任务生成优化的视频生成提示词：

子任务：{subtask}

请生成以下内容：

1. **正向引导词（Positive Prompt）**：
   - 基于原子任务，适当扩展描述
   - 添加动作细节、环境描述、质量要求
   - 确保提示词具体、生动、专业
   - 长度适中（50-100字）

2. **负向提示词（Negative Prompt）**：
   - 针对这个具体任务，指出需要避免的问题
   - 包括质量、动作、物体、背景等方面的负面描述
   - 确保负向提示词精确相关
   - 长度适中（30-60字）

请按照以下JSON格式返回：
{{
    "positive_prompt": "具体的正向引导词",
    "negative_prompt": "具体的负向提示词"
}}

要求：
- 正向提示词要体现专业机械臂操作的特点
- 负向提示词要针对具体任务，避免泛泛而谈
- 两个提示词都要用中文
- 请只返回JSON，不要包含其他解释文字
"""

        response = self._call_gemini_with_context(prompt_request)
        if not response:
            logger.warning(f"Failed to generate enhanced prompts for subtask {task_index}")
            return self._fallback_single_task_prompts(subtask, task_index)

        # 解析提示词响应
        try:
            prompts = self._parse_prompts_response(response)
            logger.info(f"Generated enhanced prompts for subtask {task_index}")
            return prompts
        except Exception as e:
            logger.error(f"Failed to parse prompts for subtask {task_index}: {str(e)}")
            return self._fallback_single_task_prompts(subtask, task_index)

    def _parse_subtasks_response(self, response: str) -> List[str]:
        """解析子任务响应"""
        try:
            # 清理响应文本
            response = response.strip()

            # 如果包含```json标记，提取JSON部分
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                if end != -1:
                    response = response[start:end].strip()

            # 解析JSON
            parsed = json.loads(response)

            if isinstance(parsed, list):
                subtasks = []
                for item in parsed:
                    if isinstance(item, str) and item.strip():
                        subtasks.append(item.strip())
                return subtasks
            else:
                raise ValueError("Response is not a JSON array")

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Parse error: {str(e)}")
            raise

    def _parse_enhanced_subtasks_response(self, response: str) -> List[Dict[str, str]]:
        """解析增强子任务响应"""
        try:
            # 清理响应文本
            response = response.strip()

            # 如果包含```json标记，提取JSON部分
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                if end != -1:
                    response = response[start:end].strip()

            # 解析JSON
            parsed = json.loads(response)

            if isinstance(parsed, list):
                enhanced_subtasks = []
                for i, item in enumerate(parsed, 1):
                    if isinstance(item, dict):
                        enhanced_task = {
                            'task_id': i,
                            'original_subtask': item.get('original_subtask', f"Task {i}"),
                            'positive_prompt': item.get('positive_prompt', ''),
                            'negative_prompt': item.get('negative_prompt', ''),
                            # 结构化三元组：动作 / 对象 / 目的地
                            'action': item.get('action', '').strip() if isinstance(item.get('action', ''), str) else '',
                            'object': item.get('object', '').strip() if isinstance(item.get('object', ''), str) else '',
                            # destination 允许为 null；保持原始值（None 或 字符串）
                            'destination': item.get('destination', None)
                        }
                        enhanced_subtasks.append(enhanced_task)
                return enhanced_subtasks
            else:
                raise ValueError("Response is not a JSON array")

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Parse error: {str(e)}")
            raise

    def _parse_prompts_response(self, response: str) -> Dict[str, str]:
        """解析提示词响应"""
        try:
            # 清理响应文本
            response = response.strip()

            # 如果包含```json标记，提取JSON部分
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                if end != -1:
                    response = response[start:end].strip()

            # 解析JSON
            parsed = json.loads(response)

            if isinstance(parsed, dict) and 'positive_prompt' in parsed and 'negative_prompt' in parsed:
                return {
                    'positive_prompt': parsed['positive_prompt'].strip(),
                    'negative_prompt': parsed['negative_prompt'].strip(),
                    'original_subtask': '',  # 将在调用处填入
                    'task_id': 0  # 将在调用处填入
                }
            else:
                raise ValueError("Response does not contain required prompt fields")

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Parse error: {str(e)}")
            raise

    def _fallback_response(self, instruction: str) -> Dict[str, any]:
        """降级响应：当AI生成失败时使用简单方法"""
        logger.warning("Using fallback response method")

        # 简单的规则分解
        separators = ['，然后', '，接着', '，之后', '，再', '；']
        subtasks = [instruction]  # 默认作为单个任务

        for sep in separators:
            if sep in instruction:
                subtasks = [task.strip() for task in instruction.split(sep) if task.strip()]
                break

        # 为每个子任务生成基本提示词
        enhanced_subtasks = []
        for i, subtask in enumerate(subtasks, 1):
            enhanced_subtasks.append(self._fallback_single_task_prompts(subtask, i))

        return {
            "success": False,
            "original_instruction": instruction,
            "subtasks": enhanced_subtasks,
            "total_subtasks": len(enhanced_subtasks),
            "enhanced_by_ai": False,
            "fallback_used": True
        }

    def _fallback_single_task_prompts(self, subtask: str, task_index: int) -> Dict[str, str]:
        """降级单个任务提示词生成"""
        return {
            "task_id": task_index,
            "original_subtask": subtask,
            "positive_prompt": f"机械臂精准执行：{subtask}。动作流畅稳定，轨迹精确，操作专业。",
            "negative_prompt": "低分辨率，画面模糊，动作异常，轨迹畸变，物体变形，背景不协调。"
        }

    def get_session_info(self) -> Dict[str, any]:
        """获取会话信息"""
        return {
            "gemini_model": self.gemini_model,
            "gemini_region": self.gemini_region,
            "gemini_project": self.gemini_project,
            "session_active": True
        }


# def test_enhanced_prompt_generator():
#     """测试增强提示词生成器"""
#     print("🧪 Testing Enhanced Prompt Generator...")

#     generator = EnhancedPromptGenerator()

#     # 测试用例
#     test_instruction = "拿起桌子上的勺子，然后拿起桌子上的罐子"
#     test_image = "/data/rczhang/MIND-V/demos/long_video/bridge1_s3.png"

#     if os.path.exists(test_image):
#         result = generator.generate_enhanced_prompts_for_task(test_instruction, test_image)

#         print("✅ Enhanced prompt generation completed!")
#         print(f"📋 Original instruction: {result['original_instruction']}")
#         print(f"🔢 Total subtasks: {result['total_subtasks']}")
#         print(f"🤖 AI enhanced: {result['enhanced_by_ai']}")

#         for i, subtask in enumerate(result['subtasks'], 1):
#             print(f"\n📝 Subtask {i}:")
#             print(f"   Original: {subtask['original_subtask']}")
#             print(f"   Positive: {subtask['positive_prompt']}")
#             print(f"   Negative: {subtask['negative_prompt']}")
#     else:
#         print(f"⚠️  Test image not found: {test_image}")
#         print("Testing with text-only prompt...")
#         result = generator.generate_enhanced_prompts_for_task(test_instruction, None)
#         print("✅ Text-only test completed!")


# if __name__ == "__main__":
#     test_enhanced_prompt_generator()
