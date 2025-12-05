#!/usr/bin/env python3
"""
任务分解器 - Task Decomposer (Gemini-only)
仅使用 Gemini 生成子任务与提示词，不再启用规则或简单切分等回退方式。

Task Decomposer for Breaking Down Complex Instructions (Gemini-only)
Uses Gemini for task decomposition; rule-based/simple-split fallbacks are disabled.
"""

import os
import json
import time
import typing
from typing import List, Dict, Optional
from loguru import logger
from vlm_api.enhanced_prompt_generator import EnhancedPromptGenerator


class TaskDecomposer:
    """
    任务分解器类
    负责将复杂的自然语言指令分解为有序的子任务序列
    """

    def __init__(self, api_key: Optional[str] = None, use_enhanced_prompts: bool = True):
        """
        初始化任务分解器

        Args:
            api_key: Gemini API密钥，如果不提供则尝试从环境变量获取
            use_enhanced_prompts: 是否使用增强提示词生成器
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model_name = "gemini-1.5-flash"
        self.use_enhanced_prompts = use_enhanced_prompts

        # 初始化增强提示词生成器（Gemini-only 模式：若失败，不启用任何本地回退）
        if use_enhanced_prompts:
            try:
                self.enhanced_generator = EnhancedPromptGenerator()
                logger.info("Enhanced Prompt Generator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Enhanced Prompt Generator (Gemini-only): {str(e)}")
                self.use_enhanced_prompts = False
                self.enhanced_generator = None
        else:
            self.enhanced_generator = None

        # 设置日志
        logger.info("Task Decomposer initialized")

        # 如果没有API密钥，发出警告（Gemini-only 模式下将导致分解失败）
        if not self.api_key:
            logger.warning("No Gemini API key provided; Gemini-only mode may fail at decomposition")

    def decompose_task_with_fallback(self, instruction: str) -> List[str]:
        """
        Gemini-only 任务分解。
        仅调用 Gemini，不再启用任何规则/简单切分回退；若失败将抛出异常。

        Args:
            instruction: 复杂指令字符串

        Returns:
            子任务列表
        """
        logger.info(f"[Gemini-only] Starting task decomposition for instruction: {instruction}")

        result = self._decompose_with_gemini_api(instruction)
        if not self._validate_decomposition(result, instruction):
            raise ValueError("Gemini decomposition did not pass validation")
        logger.info(f"[Gemini-only] Generated {len(result)} subtasks: {result}")
        return result

    def decompose_task_with_structured_output(self, instruction: str, image_path: Optional[str] = None) -> Dict:
        """
        使用Gemini API直接进行结构化任务分解，输出ActionType-Object-Destination三元组

        Args:
            instruction: 复杂指令字符串
            image_path: 输入图像路径（可选，用于提供上下文）

        Returns:
            包含结构化子任务的字典
        """
        logger.info(f"Starting structured task decomposition using Gemini API: {instruction}")

        if not self.api_key:
            logger.error("No Gemini API key available for structured decomposition")
            return {
                "success": False,
                "original_instruction": instruction,
                "subtasks": [],
                "total_subtasks": 0,
                "structured_by_gemini": False,
                "error": "Gemini API key not available"
            }

        try:
            import google.generativeai as genai

            # 配置API
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)

            # 使用结构化提示词
            structured_prompt = self._build_gemini_structured_prompt(instruction)

            # 调用API
            logger.info("Calling Gemini API for structured task decomposition...")
            response = model.generate_content(structured_prompt)

            if response.text:
                # 解析结构化响应
                result = self._parse_gemini_structured_response(response.text)

                if result["success"]:
                    logger.info(f"✅ Structured decomposition completed successfully!")
                    logger.info(f"📋 Generated {result['total_subtasks']} structured subtasks:")

                    for subtask in result["subtasks"]:
                        logger.info(f"   Task {subtask['task_id']}: {subtask['ActionType']} | {subtask['Object']} | {subtask['Destination']}")
                        logger.info(f"        Original: {subtask['original_text']}")

                    # 添加额外信息
                    result.update({
                        "original_instruction": instruction,
                        "enhanced_by_ai": True,
                        "structured_by_gemini": True
                    })

                    return result
                else:
                    logger.error(f"❌ Structured decomposition failed: {result.get('error', 'Unknown error')}")
                    return result
            else:
                error_msg = "Empty response from Gemini API"
                logger.error(error_msg)
                return {
                    "success": False,
                    "original_instruction": instruction,
                    "subtasks": [],
                    "total_subtasks": 0,
                    "structured_by_gemini": False,
                    "error": error_msg
                }

        except ImportError:
            error_msg = "google.generativeai package not installed"
            logger.error(error_msg)
            return {
                "success": False,
                "original_instruction": instruction,
                "subtasks": [],
                "total_subtasks": 0,
                "structured_by_gemini": False,
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"Gemini API call failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "original_instruction": instruction,
                "subtasks": [],
                "total_subtasks": 0,
                "structured_by_gemini": False,
                "error": error_msg
            }

    def decompose_task_with_enhanced_prompts(self, instruction: str, image_path: Optional[str] = None) -> Dict:
        """
        使用增强提示词生成器分解任务并生成优化的提示词

        Args:
            instruction: 复杂指令字符串
            image_path: 输入图像路径（可选）

        Returns:
            包含子任务和增强提示词的字典
        """
        logger.info(f"Starting enhanced task decomposition for instruction: {instruction}")

        if not self.use_enhanced_prompts or not self.enhanced_generator:
            logger.error("Enhanced prompts not available in Gemini-only mode; decomposition will fail")
            return {
                "success": False,
                "original_instruction": instruction,
                "subtasks": [],
                "total_subtasks": 0,
                "enhanced_by_ai": False,
                "error": "EnhancedPromptGenerator unavailable"
            }

        try:
            # 使用增强提示词生成器
            result = self.enhanced_generator.generate_enhanced_prompts_for_task(instruction, image_path)

            # 确保每个子任务都有正确的task_id和original_subtask字段
            for i, subtask in enumerate(result["subtasks"], 1):
                subtask["task_id"] = i
                if "original_subtask" not in subtask:
                    subtask["original_subtask"] = subtask.get("subtask", f"Task {i}")

            # 强制子任务"独立/不含连接词"：若某条仍包含"再/然后/接着/并且/同时"等连接词，拆分为多条
            result["subtasks"] = self._enforce_independent_subtasks(result["subtasks"])

            # 解析每个子任务为 {ActionType, Object, Destination} 三元组
            for sub in result["subtasks"]:
                original_text = sub.get("original_subtask", sub.get("subtask", ""))
                if original_text:
                    structured = self.parse_subtask_structure(original_text)
                    sub["action_type"] = structured["ActionType"]
                    sub["object_name"] = structured["Object"]
                    sub["destination"] = structured["Destination"]
                    sub["structured_subtask"] = structured

            # 重新标号
            for idx, sub in enumerate(result["subtasks"], 1):
                sub["task_id"] = idx
            result["total_subtasks"] = len(result["subtasks"])

            logger.info(f"Enhanced decomposition completed: {result['total_subtasks']} subtasks with AI-generated prompts")
            return result

        except Exception as e:
            logger.error(f"Enhanced decomposition failed (Gemini-only): {str(e)}")
            # 不再进行任何规则/简单切分回退
            return {
                "success": False,
                "original_instruction": instruction,
                "subtasks": [],
                "total_subtasks": 0,
                "enhanced_by_ai": False,
                "error": str(e)
            }

    def _decompose_with_gemini_api(self, instruction: str) -> List[str]:
        """
        使用Gemini API进行任务分解
        """
        if not self.api_key:
            raise ValueError("Gemini API key not available")

        try:
            import google.generativeai as genai

            # 配置API
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)

            # 构建prompt
            prompt = self._build_gemini_prompt(instruction)

            # 调用API
            logger.info("Calling Gemini API for task decomposition...")
            response = model.generate_content(prompt)

            if response.text:
                # 解析响应
                subtasks = self._parse_gemini_response(response.text)
                return subtasks
            else:
                raise ValueError("Empty response from Gemini API")

        except ImportError:
            raise ImportError("google.generativeai package not installed")
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            raise

    def _decompose_with_rule_based(self, instruction: str) -> List[str]:
        """
        基于规则的任务分解方法
        """
        logger.info("Using rule-based decomposition (disabled in Gemini-only mode)")

        # 定义的分隔词模式
        separators = [
            r'然后.*?，',
            r'接着.*?，',
            r'之后.*?，',
            r'再.*?，',
            r'，然后',
            r'，接着',
            r'，之后',
            r'，再',
            r'；',
            r'并且',
            r'同时'
        ]

        import re

        # 尝试各种分隔模式
        for sep_pattern in separators:
            parts = re.split(sep_pattern, instruction)
            if len(parts) > 1:
                # 清理和验证分割结果
                cleaned_parts = []
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 3:  # 过滤太短的片段
                        cleaned_parts.append(part)

                if len(cleaned_parts) > 1:
                    logger.info(f"Rule-based decomposition found {len(cleaned_parts)} subtasks")
                    return cleaned_parts

        # Gemini-only: 不再启用规则回退
        raise ValueError("Rule-based decomposition is disabled in Gemini-only mode")

    def _decompose_with_simple_split(self, instruction: str) -> List[str]:
        """
        简单的基于逗号分割的fallback方法
        """
        logger.info("Using simple split-based decomposition (disabled in Gemini-only mode)")

        # 尝试简单的逗号分割
        if '，' in instruction:
            parts = [part.strip() for part in instruction.split('，') if part.strip()]
            if len(parts) > 1:
                return parts

        # 尝试中文句号分割
        if '。' in instruction:
            parts = [part.strip() for part in instruction.split('。') if part.strip()]
            if len(parts) > 1:
                return parts

        # 如果都失败了，尝试长度分割（不太准确但作为最后手段）
        if len(instruction) > 20:
            mid = len(instruction) // 2
            # 尝试在空格或接近空格的位置分割
            split_pos = mid
            for i in range(mid-5, mid+5):
                if i < len(instruction) and instruction[i] in ' ，。、；；':
                    split_pos = i + 1
                    break

            if split_pos < len(instruction):
                part1 = instruction[:split_pos].strip()
                part2 = instruction[split_pos:].strip()
                if part1 and part2:
                    return [part1, part2]

        # Gemini-only: 不再启用简单切分回退
        raise ValueError("Simple split decomposition is disabled in Gemini-only mode")

    def _build_gemini_prompt(self, instruction: str) -> str:
        """
        构建用于Gemini API的prompt
        """
        prompt = f"""
你是一个机械臂操作任务分解专家。请将以下复杂的机械臂操作指令分解为“高层次、可直接执行”的子任务。

原始指令: "{instruction}"

分解要求（务必遵守）:
1. 子任务要独立而完整；严格避免将一个自然动作拆成“抓取/移动/放置”等微步骤。
2. 每个子任务必须显式包含：动作原语(ActionType) + 对象(Object) + 目标位置/容器(Destination)。例如“把X放入Y”、“将X移动到Y”、“将X移出画面”。如目标缺省，请明确写为“画面外/原位”等。
3. 子任务按照自然顺序排列，保持语义连贯；如指令包含多个“再/然后/接着”，通常对应多个高层子任务。
4. 每个子任务必须“独立完整”，不得在同一字符串中继续包含“再/然后/接着/并且/同时”等连接词；若存在此类连接，请拆分成两条。
5. 子任务用简洁中文短句表达“动作+对象+目标”，避免加入执行细节（如对准、上方、轨迹、微调）。
6. 只输出JSON数组，每个元素为一个字符串，不要包含其他说明文字。

Few-shot 示例（必须模仿这种风格）：
- 输入: "把白柄蘑菇玩具放入金属锅中"
  输出: ["把白柄蘑菇玩具放入金属锅中"]

- 输入: "把左边的勺子放进锅里，再把蓝色抹布放进锅里，再把锅拿出画面"
  输出: ["把左边的勺子放进锅里", "把蓝色抹布放进锅里", "把锅拿出画面"]

- 输入: "桌子上有两个牛油果，先把左边的牛油果向上拿出画面，再把右边的牛油果拿出画面"
  输出: ["桌子上有两个牛油果，把左边的牛油果向上拿出画面", "桌子上有两个牛油果，把右边的牛油果拿出画面"]

现在请分解以下指令:
"{instruction}"

只返回JSON数组，不要包含其他说明文字。
"""
        return prompt

    def _build_gemini_structured_prompt(self, instruction: str) -> str:
        """
        构建用于Gemini API的结构化分解提示词
        直接输出ActionType-Object-Destination三元组结构
        """
        prompt = f"""
你是一个专业的机械臂操作任务分解专家。请将以下复杂指令分解为结构化的子任务，每个子任务必须包含ActionType、Object、Destination三个明确要素。

原始指令: "{instruction}"

分解要求（必须严格遵守）:

1. 输出格式要求:
   - 必须返回JSON格式的数组
   - 每个子任务必须包含完整的结构化信息
   - 不要包含任何解释性文字，只返回JSON

2. 子任务结构要求:
   每个子任务必须包含以下字段：
   - "task_id": 子任务序号（从1开始）
   - "original_text": 原始子任务文本
   - "ActionType": 动作类型（必须是以下之一：放置、取出、抓取、释放、移动）
   - "Object": 操作对象的具体名称（如"勺子"、"牛油果"、"杯子"等）
   - "Destination": 目标位置（如"锅里"、"画面外"、"桌子上面"等）

3. 分解原则:
   - 每个子任务应该是独立完整的动作
   - 避免将一个自然动作拆分为微步骤
   - 严格遵循"动作+对象+目标"的结构
   - 目标位置要具体明确

4. 标准输出格式:
   {{
     "subtasks": [
       {{
         "task_id": 1,
         "original_text": "把左边的勺子放进锅里",
         "ActionType": "放置",
         "Object": "勺子",
         "Destination": "锅里"
       }},
       {{
         "task_id": 2,
         "original_text": "把右边的牛油果拿出画面",
         "ActionType": "取出",
         "Object": "牛油果",
         "Destination": "画面外"
       }}
     ]
   }}

示例分析:

输入: "把左边的勺子放进锅里，再把右边的牛油果拿出画面"

输出格式分析:
- 第一个子任务: 动作是"放置"，对象是"勺子"，目标是"锅里"
- 第二个子任务: 动作是"取出"，对象是"牛油果"，目标是"画面外"

现在请分解以下指令，严格按照上述格式输出:
"{instruction}"

要求：
- 必须返回有效的JSON格式
- 包含"subtasks"数组和所有必需字段
- 不要添加任何解释或说明文字
"""
        return prompt

    def _parse_gemini_structured_response(self, response_text: str) -> Dict:
        """
        解析Gemini API返回的结构化响应
        返回包含ActionType-Object-Destination三元组的结构化数据
        """
        try:
            response_text = response_text.strip()

            # 如果响应包含```json标记，提取JSON部分
            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                if end != -1:
                    response_text = response_text[start:end].strip()

            # 解析JSON
            parsed = json.loads(response_text)

            if "subtasks" in parsed and isinstance(parsed["subtasks"], list):
                # 验证每个子任务是否包含必需字段
                subtasks = []
                for i, subtask in enumerate(parsed["subtasks"], 1):
                    if self._validate_structured_subtask(subtask):
                        # 确保task_id正确
                        subtask["task_id"] = i
                        subtasks.append(subtask)
                        logger.info(f"Validated structured subtask {i}: {subtask.get('ActionType')} | {subtask.get('Object')} | {subtask.get('Destination')}")
                    else:
                        logger.warning(f"Invalid structured subtask found: {subtask}")

                if subtasks:
                    logger.info(f"Successfully parsed {len(subtasks)} structured subtasks from Gemini response")
                    return {
                        "success": True,
                        "subtasks": subtasks,
                        "total_subtasks": len(subtasks),
                        "structured_by_gemini": True
                    }
                else:
                    raise ValueError("No valid structured subtasks found in response")
            else:
                raise ValueError("Response does not contain valid 'subtasks' array")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured JSON response: {e}")
            logger.error(f"Response text: {response_text}")
            return {"success": False, "error": f"JSON parsing failed: {str(e)}"}
        except Exception as e:
            logger.error(f"Error parsing structured Gemini response: {e}")
            return {"success": False, "error": str(e)}

    def _validate_structured_subtask(self, subtask: Dict) -> bool:
        """
        验证结构化子任务是否包含所有必需字段
        """
        required_fields = ["task_id", "original_text", "ActionType", "Object", "Destination"]
        valid_action_types = ["放置", "取出", "抓取", "释放", "移动"]

        # 检查必需字段
        for field in required_fields:
            if field not in subtask or not subtask[field]:
                logger.warning(f"Missing or empty field: {field}")
                return False

        # 检查ActionType是否有效
        if subtask["ActionType"] not in valid_action_types:
            logger.warning(f"Invalid ActionType: {subtask['ActionType']}")
            return False

        return True

    def _parse_gemini_response(self, response_text: str) -> List[str]:
        """
        解析Gemini API的响应
        """
        try:
            # 尝试直接解析JSON
            response_text = response_text.strip()

            # 如果响应包含```json标记，提取JSON部分
            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                if end != -1:
                    response_text = response_text[start:end].strip()

            # 尝试解析JSON
            parsed = json.loads(response_text)

            if isinstance(parsed, list):
                # 验证所有元素都是字符串
                subtasks = []
                for item in parsed:
                    if isinstance(item, str) and item.strip():
                        subtasks.append(item.strip())

                if subtasks:
                    return subtasks
            else:
                raise ValueError("Response is not a JSON array")

        except json.JSONDecodeError:
            # 如果JSON解析失败，尝试文本解析
            logger.warning("Failed to parse JSON response, trying text parsing")
            return self._parse_text_response(response_text)

        raise ValueError("Could not parse Gemini response")

    def _parse_text_response(self, response_text: str) -> List[str]:
        """
        解析纯文本响应（fallback方法）
        """
        # 尝试提取编号列表
        import re

        # 匹配编号模式: 1. 任务1 2. 任务2
        numbered_tasks = re.findall(r'\d+\.\s*([^0-9\n]+)', response_text)
        if len(numbered_tasks) > 1:
            return [task.strip() for task in numbered_tasks if task.strip()]

        # 匹配引号模式: "任务1", "任务2"
        quoted_tasks = re.findall(r'"([^"]+)"', response_text)
        if len(quoted_tasks) > 1:
            return [task.strip() for task in quoted_tasks if task.strip()]

        # 按行分割
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        if len(lines) > 1:
            return lines

        raise ValueError("Could not parse text response")

    def _enforce_independent_subtasks(self, enhanced_subtasks: List[Dict]) -> List[Dict]:
        """
        将仍然包含多个动作连接词的子任务拆分为多条独立子任务。
        规则：遇到 '，再'/'，然后'/'，接着'/'；'/'并且'/'同时' 等连接词进行切分；
        子任务开头若含有 '再/然后/接着' 则去掉这些连接副词。
        对拆分出的新子任务，若缺少提示词，使用简易正向/负向提示作为兜底。
        """
        import re

        def split_segments(text: str) -> List[str]:
            if not text:
                return []
            # 统一连接词以便切分
            t = text
            t = t.replace('并且', '，再')
            t = t.replace('同时', '，再')
            # 在 '，再' / '，然后' / '，接着' / '；' 之前插入分隔标记
            t = re.sub(r'，\s*(再|然后|接着)', r'|||\g<0>', t)
            t = re.sub(r'；+', '|||', t)
            parts = [p.strip() for p in t.split('|||') if p.strip()]
            # 清理各段落开头的连接词
            cleaned = []
            for p in parts:
                p = re.sub(r'^(再|然后|接着)\s*', '', p)
                # 去除多余标点
                p = p.lstrip('，。；、 ')
                if p:
                    cleaned.append(p)
            return cleaned if cleaned else [text]

        new_list: List[Dict] = []
        for item in enhanced_subtasks:
            text = item.get('original_subtask') or item.get('subtask') or ''
            segs = split_segments(text)
            if len(segs) <= 1:
                new_list.append(item)
                continue
            # 多段：拆分为多条；为每条准备简易提示词作为兜底（不再额外请求API）
            for seg in segs:
                new_item = {
                    'task_id': 0,
                    'original_subtask': seg,
                    'positive_prompt': seg,
                    'negative_prompt': ''
                }
                new_list.append(new_item)
        return new_list

    def parse_subtask_structure(self, subtask_text: str) -> Dict[str, str]:
        """
        将子任务文本解析为 {ActionType, Object, Destination} 三元组结构

        Args:
            subtask_text: 子任务文本，如 "把左边的勺子放进锅里"

        Returns:
            包含 ActionType, Object, Destination 的字典
        """
        import re

        # 常见动作类型
        action_patterns = [
            r'^(把|将|拿)(.*?)(放入|放进|放置到|移到|移动到|搬到)(.*?)$',
            r'^(把|将|拿)(.*?)(拿出|取出|移出|搬出)(.*?)$',
            r'^(把|将|拿)(.*?)(向上|向下|向左|向右)(.*?)(拿出|移出)(.*?)$',
            r'^(把|将|拿)(.*?)(放到|放置在)(.*?)(上面|下面|左边|右边|里面|外面)$',
            r'^(拿起|抓取|夹起)(.*?)$',
            r'^(放开|释放|放置)(.*?)$'
        ]

        # 初始化默认值
        action_type = "移动"
        object_name = ""
        destination = ""

        # 尝试匹配各种动作模式
        for pattern in action_patterns:
            match = re.match(pattern, subtask_text.strip())
            if match:
                groups = match.groups()

                if len(groups) >= 4 and groups[2] in ['放入', '放进', '放置到', '移到', '移动到', '搬到']:
                    # 放置类动作
                    action_type = "放置"
                    object_name = groups[1].strip()
                    destination = groups[3].strip()
                elif len(groups) >= 3 and groups[2] in ['拿出', '取出', '移出', '搬出']:
                    # 取出类动作
                    action_type = "取出"
                    object_name = groups[1].strip()
                    destination = "画面外"
                elif len(groups) >= 5 and groups[4] in ['拿出', '移出']:
                    # 方向性取出
                    direction = groups[3].strip()
                    action_type = f"向{direction}取出"
                    object_name = groups[1].strip()
                    destination = f"画面{direction}"
                elif len(groups) >= 5 and groups[4] in ['上面', '下面', '左边', '右边', '里面', '外面']:
                    # 方向性放置
                    direction = groups[4].strip()
                    action_type = f"放置到{direction}"
                    object_name = groups[1].strip()
                    destination = groups[3].strip()
                elif groups[0] in ['拿起', '抓取', '夹起']:
                    # 抓取类动作
                    action_type = "抓取"
                    object_name = groups[1].strip()
                    destination = "原位"
                elif groups[0] in ['放开', '释放', '放置']:
                    # 释放类动作
                    action_type = "释放"
                    object_name = groups[1].strip() if len(groups) > 1 else ""
                    destination = "当前位置"

                break

        # 如果没有匹配到任何模式，使用简单的关键词提取
        if not object_name:
            # 提取可能的物体名称（通常在"把/将"之后，动词之前）
            object_match = re.search(r'[把将](.*?)[的]' + r'(.*?)([放入放进放置到移到移动到搬到拿出取出移出搬出])', subtask_text)
            if object_match:
                object_name = object_match.group(2).strip()
            else:
                # 如果没有明确的物体，尝试提取名词
                nouns = re.findall(r'([勺子|锅|牛油果|蘑菇|抹布|玩具|碗|盘子|杯子|瓶子|书|手机|笔|桌子|椅子])', subtask_text)
                if nouns:
                    object_name = nouns[0]
                else:
                    object_name = "未知物体"

        # 提取目标位置
        if not destination:
            # 查找"到/在/向"等介词后的位置信息
            dest_match = re.search(r'(到|在|向)(.*?)([上面下面左右里面外面])', subtask_text)
            if dest_match:
                destination = dest_match.group(2).strip() + dest_match.group(3).strip()
            elif '画面' in subtask_text:
                destination = "画面外"
            elif '原位' in subtask_text:
                destination = "原位"
            else:
                destination = "未指定位置"

        return {
            "ActionType": action_type,
            "Object": object_name,
            "Destination": destination,
            "original_text": subtask_text.strip()
        }

    def _validate_decomposition(self, subtasks: List[str], original_instruction: str) -> bool:
        """
        验证分解结果的质量
        """
        if not subtasks or len(subtasks) == 0:
            return False

        if len(subtasks) == 1 and subtasks[0] == original_instruction:
            return False

        # 检查每个子任务
        for task in subtasks:
            if not task or len(task.strip()) < 2:
                return False

            # 检查是否包含基本的动作词汇
            action_words = ['拿', '放', '移动', '抓取', '放置', '推', '拉', '举', '提', '搬']
            has_action = any(word in task for word in action_words)
            if not has_action:
                logger.warning(f"Subtask might not be a valid action: {task}")

        # 检查子任务数量是否合理
        if len(subtasks) > 6:
            logger.warning(f"Too many subtasks ({len(subtasks)}), might be over-decomposed")
            return False

        return True

    def get_decomposition_stats(self) -> Dict:
        """
        获取分解统计信息（用于调试和监控）
        """
        return {
            "api_key_configured": bool(self.api_key),
            "model_name": self.model_name,
            "supported_methods": [
                "gemini_api",
                "gemini_enhanced_prompts"
            ]
        }


def test_decomposer():
    """
    测试任务分解器
    """
    decomposer = TaskDecomposer()

    test_cases = [
        "将桌子上的勺子拿走，然后将桌子上的锅拿走",
        "把右边桌子靠左的牛油果放在桌子左边，然后把右边桌子里靠右的牛油果也放在桌子左边",
        "先拿起苹果，然后移动到盘子旁边，最后放下苹果"
    ]

    for i, instruction in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {instruction}")
        try:
            subtasks = decomposer.decompose_task_with_fallback(instruction)
            print(f"分解结果 ({len(subtasks)} 个子任务):")
            for j, task in enumerate(subtasks, 1):
                print(f"  {j}. {task}")
        except Exception as e:
            print(f"分解失败: {str(e)}")


if __name__ == "__main__":
    test_decomposer()
