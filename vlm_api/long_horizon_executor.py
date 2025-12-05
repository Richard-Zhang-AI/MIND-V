#!/usr/bin/env python3
"""
Long Horizon Task Executor
Sequentially executes multiple subtasks, manages file organization and state transfer

Long Horizon Task Executor for Sequential Subtask Execution
Manages file organization and state transfer between subtasks
"""

import os
import sys
import json
import shutil
import numpy as np
import random
import torch
import time
import threading
from typing import Dict, Optional, List, Tuple
from loguru import logger

# Try importing cv2, use ffmpeg if failed
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV (cv2) not available, will use ffmpeg for frame extraction")

# Import existing module - delayed import to avoid dependency issues during testing
UniversalRobotController = None


class LongHorizonExecutor:
    """
    Long Horizon Task Executor
    Responsible for executing multiple subtasks in sequence, managing file organization and state transfer
    """

    def __init__(self, base_temp_dir: str, keep_temp: bool = False,
                 num_inference_steps: int = 20, num_candidates: int = 2,
                 enable_gpu_stats: bool = False):
        """
        Initialize the Long Horizon Task Executor

        Args:
            base_temp_dir: Base directory for temporary files
            keep_temp: Whether to keep temporary files (default: False)
            num_inference_steps: Number of denoising steps for video generation (default: 20)
            num_candidates: Number of candidate videos generated per subtask (default: 2)
            enable_gpu_stats: Whether to enable GPU statistics (default: False to reduce nvidia-smi overhead)
        """
        self.base_temp_dir = base_temp_dir
        self.keep_temp = keep_temp
        self.num_inference_steps = num_inference_steps
        self.num_candidates = num_candidates
        # Whether to enable GPU stats (default off to reduce nvidia-smi overhead)
        self.enable_gpu_stats = bool(enable_gpu_stats)
        os.makedirs(base_temp_dir, exist_ok=True)

        # Set up logging
        logger.add(f"{base_temp_dir}/long_horizon_execution.log", rotation="10 MB")

        # Initialize Gemini client (maintain conversation continuity)
        self._gemini_client = None

        # Store previous evaluation history for context
        self.evaluation_history = []

        logger.info(f"Initialized LongHorizonExecutor with num_inference_steps={num_inference_steps}, num_candidates={num_candidates}")

    def _get_gemini_client(self):
        """
        Get or create a persistent Gemini client instance.
        Maintains conversation continuity so subsequent calls are aware of previous evaluation results.
        """
        if self._gemini_client is None:
            try:
                from vlm_api.gemini_api import GeminiChat

                # Get absolute path of the credential file
                credential_file_path = os.path.join(os.path.dirname(__file__), "captioner-test-1017-b3fa56e15267.json")

                # Initialize Gemini client (create only once)
                self._gemini_client = GeminiChat(
                    model="gemini-2.0-flash",
                    region="us-central1",
                    project="captioner-test-1017",
                    credential_file=credential_file_path,
                    temperature=0.1
                )

                # Add initial system prompt to establish evaluation background
                initial_prompt = """
You are a professional robotic arm operation video evaluation expert. Throughout the long-horizon task execution process, you will evaluate multiple subtasks.

Your tasks are:
1. Evaluate candidate videos for each subtask.
2. Select the optimal video.
3. Remember previous evaluation results to maintain consistency and continuity in subsequent subtasks.
4. Provide suggestions for improvement.

I will show you candidate videos for each subtask; please evaluate them and remember key information.
"""
                self._gemini_client.add_message("user", initial_prompt)
                self._gemini_client.chat("I understand the evaluation task", retry_attempts=1)

                logger.info("Initialized persistent Gemini client, established conversation continuity")

            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                self._gemini_client = None

        return self._gemini_client

    def _build_contextual_evaluation_prompt(self, candidate_videos: List[Dict], subtask: str, task_id: int) -> str:
        """
        Build an evaluation prompt containing historical context.

        Args:
            candidate_videos: List of candidate videos for the current subtask
            subtask: Description of the current subtask
            task_id: Current task ID

        Returns:
            str: Evaluation prompt containing historical context
        """
        # Professional evaluation prompt
        base_prompt = f"""
You are an expert robotics vision evaluator specializing in assessing manipulation tasks for robotic arms. Please evaluate {len(candidate_videos)} candidate videos that were generated to execute the subtask: "{subtask}".

I have extracted 5 equally-spaced keyframes from each candidate video to represent the complete action sequence. Additionally, I will provide the original task prompt and trajectory planning visualization to give you complete context for evaluation.

## Three Evaluation Dimensions:

### 1. Task Completion Judge (1-10 points)
- **Success Criterion**: Does the robot successfully complete the specified manipulation task?
- **Key Aspects**: Object grasping accuracy, trajectory execution, goal achievement
- **Evaluation**: Check if the intended action is fully completed from start to finish

### 2. Physical Plausibility Judge (1-10 points)
- **Success Criterion**: Are the robot motions and object interactions physically realistic?
- **Key Aspects**: Kinematic feasibility, collision avoidance, gravity compliance, object dynamics
- **Evaluation**: Assess whether the robot movements obey physical constraints and natural motion patterns

### 3. Visual Quality Judge (1-10 points)
- **Success Criterion**: Are the generated visuals clear, coherent, and realistic?
- **Key Aspects**: Image clarity, temporal consistency, visual artifacts, object realism
- **Evaluation**: Evaluate the overall visual fidelity and generation quality

## Evaluation Process:

For each candidate video, please:
1. Analyze all 5 keyframes in sequence
2. Cross-reference with the trajectory planning visualization
3. Consider the original task requirements
4. Apply the three evaluation dimensions consistently

## Scoring Guidelines:
- 8.0-10.0: Excellent - Fully meets all criteria
- 7.0-7.9: Good - Meets criteria with minor issues
- 6.0-6.9: Fair - Meets criteria with noticeable issues
- 5.0-5.9: Poor - Partially meets criteria
- 1.0-4.9: Failed - Does not meet criteria

## Output Format:
Please return results in this JSON format:
{{
    "task_completion_judge": {{
        "scores": [8.5, 7.2, 9.1, 6.8, 8.0],
        "best_candidate_id": 2,
        "best_score": 9.1,
        "evaluation_details": "Detailed analysis of task completion for each candidate"
    }},
    "physical_plausibility_judge": {{
        "scores": [8.0, 7.5, 8.8, 7.0, 8.2],
        "best_candidate_id": 2,
        "best_score": 8.8,
        "evaluation_details": "Detailed analysis of physical realism for each candidate"
    }},
    "visual_quality_judge": {{
        "scores": [8.7, 7.8, 8.9, 7.3, 8.5],
        "best_candidate_id": 2,
        "best_score": 8.9,
        "evaluation_details": "Detailed analysis of visual quality for each candidate"
    }},
    "final_selection": {{
        "selected_candidate_id": 2,
        "overall_score": 8.93,
        "meets_threshold": true,
        "comprehensive_feedback": "Overall assessment summary with specific recommendations"
    }}
}}

Threshold for success: Overall score >= 7.5 AND all three dimensions >= 6.0
"""

        # If there is previous evaluation history, add context information
        if self.evaluation_history:
            context_info = "\n\n**Historical Evaluation Context:**\n"
            context_info += f"This is the {task_id + 1}th subtask. {len(self.evaluation_history)} subtasks have been evaluated previously.\n"

            # Add brief info for recent tasks
            recent_history = self.evaluation_history[-2:]  # Show only the last 2 tasks
            if recent_history:
                context_info += "Previous subtask evaluation results:\n"
                for i, history in enumerate(recent_history):
                    context_info += f"- Subtask {history['task_id'] + 1}: {history['subtask'][:50]}...\n"
                    context_info += f"  Selected Candidate: {history['best_candidate_id']}, Score: {history['best_score']}\n"
                    if history.get('meets_threshold', False):
                        context_info += "  Status: Success\n"
                    else:
                        context_info += f"  Status: Needs Improvement - {history.get('feedback', '')[:50]}...\n"

            context_info += "\nPlease consider the evaluation standards of previous tasks when evaluating the current subtask to maintain a consistent scoring scale."

            return base_prompt + context_info
        else:
            # Prompt for the first subtask
            intro_info = f"""
This is the {task_id + 1}th subtask of a long-horizon task (total {len(self.evaluation_history) + 1} planned). Please establish evaluation standards; subsequent subtasks will reference this scoring scale.
"""
            return intro_info + base_prompt

    def execute_structured_subtasks(self, original_image_path: str, enhanced_task_data: Dict, seed: int = 42) -> Dict:
        """
        Execute structured subtasks (using ActionType-Object-Destination triplets generated directly by Gemini API).

        Args:
            original_image_path: Path to original image
            enhanced_task_data: Dictionary containing structured subtask data
            seed: Random seed

        Returns:
            Dictionary of execution results
        """
        logger.info(f"Starting execution of structured long-horizon task with {enhanced_task_data['total_subtasks']} subtasks")
        logger.info(f"Original image: {original_image_path}")

        # Record structured task info
        if enhanced_task_data.get("structured_by_gemini"):
            logger.info("Using structured subtask data generated by Gemini AI")

        current_image_path = original_image_path
        task_results = []
        video_paths = []

        # Phase statistics aggregation
        planning_time_total = 0.0
        generation_time_total = 0.0
        planning_peak_bytes = 0
        generation_peak_bytes = 0

        # Variables for trajectory continuity between tasks
        trajectory_continuity = {
            "previous_robot_exit_position": None,
            "last_robot_trajectory": []
        }

        # Use structured subtask data directly
        structured_subtasks = enhanced_task_data.get("subtasks", [])

        for i, structured_subtask in enumerate(structured_subtasks, 1):
            # Set base seed for re-planning (base seed + task index)
            base_task_seed = seed + i
            task_id = structured_subtask.get("task_id", i)

            # Handle two different data formats
            if "structured_subtask" in structured_subtask:
                # Enhanced decomposition format: structured info is in 'structured_subtask' field
                sub_data = structured_subtask["structured_subtask"]
                original_text = structured_subtask.get("original_subtask", f"Task {i}")
                action_type = sub_data.get("ActionType", "Unknown")
                object_name = sub_data.get("Object", "Unknown")
                destination = sub_data.get("Destination", "Unknown")
            else:
                # Direct structured decomposition format: structured info is at root level
                original_text = structured_subtask.get("original_text", f"Task {i}")
                action_type = structured_subtask.get("ActionType", "Unknown")
                object_name = structured_subtask.get("Object", "Unknown")
                destination = structured_subtask.get("Destination", "Unknown")

            logger.info(f"\n{'='*80}")
            logger.info(f"Executing Structured Subtask {i}/{len(structured_subtasks)}: {original_text}")
            logger.info(f"   ActionType: {action_type}")
            logger.info(f"   Object: {object_name}")
            logger.info(f"   Destination: {destination}")
            logger.info(f"{'='*80}")

            # Create separate directory structure for each subtask
            task_name = f"task{i}"
            task_dir = os.path.join(self.base_temp_dir, task_name)
            task_robot_output_dir = os.path.join(task_dir, "robot_output")
            task_video_input_dir = os.path.join(task_dir, "video_input")
            task_video_output_dir = os.path.join(task_dir, "video_output")

            # Clean current subtask directory
            if os.path.exists(task_dir) and not self.keep_temp:
                logger.info(f"Cleaning current subtask directory: {task_dir}")
                try:
                    shutil.rmtree(task_dir)
                except Exception as e:
                    logger.warning(f"Failed to clean subtask directory: {task_dir}, Error: {str(e)}")

            os.makedirs(task_robot_output_dir, exist_ok=True)
            os.makedirs(task_video_input_dir, exist_ok=True)
            os.makedirs(task_video_output_dir, exist_ok=True)

            logger.info(f"Robot arm output dir: {task_robot_output_dir}")
            logger.info(f"Video input dir: {task_video_input_dir}")
            logger.info(f"Video output dir: {task_video_output_dir}")
            logger.info(f"Input image: {current_image_path}")

            # Step 1: Execute Robot Controller
            # Pass structured subtask data directly
            planning_t0 = time.perf_counter()

            # Standardize subtask data structure for UniversalRobotController
            standardized_subtask = {
                "ActionType": action_type,
                "Object": object_name,
                "Destination": destination,
                "original_text": original_text,
                "task_id": task_id
            }

            json_data_path = self._execute_robot_controller_structured(
                current_image_path, standardized_subtask, task_robot_output_dir, trajectory_continuity, task_id
            )

            if not json_data_path:
                logger.error(f"Subtask {i} robot control failed")
                return {
                    "success": False,
                    "error": f"Task {i} robot control failed",
                    "completed_tasks": task_results,
                    "current_task_id": i
                }

            # Step 2: Prepare video generation data
            prepared_files = self._prepare_video_generation_data(
                current_image_path, json_data_path, task_video_input_dir, task_name
            )

            if not prepared_files:
                logger.error(f"Subtask {i} video data preparation failed")
                return {
                    "success": False,
                    "error": f"Task {i} video data preparation failed",
                    "completed_tasks": task_results,
                    "current_task_id": i
                }

            planning_t1 = time.perf_counter()
            planning_time_total += (planning_t1 - planning_t0)

            # Step 3: Generate candidate videos
            generation_t0 = time.perf_counter()
            gen_result = self._generate_candidate_videos(
                task_video_input_dir, task_video_output_dir, task_name, seed
            )
            generation_t1 = time.perf_counter()
            generation_time_total += (generation_t1 - generation_t0)

            if not gen_result or not gen_result.get("success"):
                logger.error(f"Subtask {i} candidate video generation failed")
                return {
                    "success": False,
                    "error": f"Task {i} candidate video generation failed",
                    "completed_tasks": task_results,
                    "current_task_id": i
                }

            # Step 4: Evaluate candidate videos using Gemini API and select the best one
            logger.info(f"Starting evaluation of {len(gen_result['candidate_videos'])} candidate videos for subtask {i}")
            evaluation_result = self._evaluate_candidate_videos_with_gemini(
                gen_result["candidate_videos"], original_text, task_video_output_dir, i
            )

            if not evaluation_result or not evaluation_result.get("success"):
                logger.error(f"Subtask {i} video evaluation failed")
                return {
                    "success": False,
                    "error": f"Task {i} video evaluation failed",
                    "completed_tasks": task_results,
                    "current_task_id": i
                }

            selected_video_path = evaluation_result["selected_video_path"]
            video_path = selected_video_path  # Use variable name video_path for compatibility

            # Record generation stats
            gen_duration = float(gen_result.get("time_sec", 0.0))
            gen_peak = int(gen_result.get("gpu_peak_bytes", 0))
            generation_time_total += gen_duration
            generation_peak_bytes = max(generation_peak_bytes, gen_peak)

            logger.info(f"Subtask {i} selected best video: {os.path.basename(selected_video_path)}")
            logger.info(f"Evaluation Score: {evaluation_result.get('evaluation_score', 'N/A')}")

            # Step 5: If best video does not meet requirements, perform replanning
            if not evaluation_result.get("meets_threshold", False):
                logger.warning(f"Subtask {i} best video did not meet success threshold, replanning required")
                replan_result = self._replan_and_regenerate(
                    task_video_input_dir, task_video_output_dir, task_name, original_text,
                    base_task_seed, evaluation_result.get("feedback", ""), i
                )

                if replan_result and replan_result.get("success"):
                    selected_video_path = replan_result["selected_video_path"]
                    video_path = selected_video_path  # Update video_path
                    logger.info(f"Subtask {i} replanning successful, using new video: {os.path.basename(selected_video_path)}")
                else:
                    logger.error(f"Subtask {i} replanning failed, continuing with original video")

            # Store task results, ensuring compatibility with both data formats
            task_result = {
                "task_id": task_id,
                "task_name": task_name,
                "original_text": original_text,
                "structured_subtask": structured_subtask,
                # Fields for direct access compatibility
                "action_type": action_type,
                "object_name": object_name,
                "destination": destination,
                "video_path": video_path,
                "input_image": current_image_path,
                "output_directory": task_dir,
                "json_data_path": json_data_path,
                "prepared_files": prepared_files
            }
            task_results.append(task_result)
            video_paths.append(video_path)

            logger.info(f"✅ Subtask {i} completed, video generated: {video_path}")

            # Update trajectory continuity info
            try:
                with open(json_data_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                robot_exit_position = json_data.get("robot_exit_position")
                if robot_exit_position:
                    trajectory_continuity["previous_robot_exit_position"] = robot_exit_position
                    logger.info(f"Updated robot exit position for next task: {robot_exit_position}")

            except Exception as e:
                logger.warning(f"Failed to update trajectory continuity: {e}")

        logger.info(f"🎉 All structured subtasks completed!")
        logger.info(f"📹 Number of videos generated: {len(video_paths)}")

        return {
            "success": True,
            "task_results": task_results,
            "video_paths": video_paths,
            "total_tasks": len(task_results),
            "planning_time_sec": planning_time_total,
            "generation_time_sec": generation_time_total,
            "base_temp_dir": self.base_temp_dir
        }

    def execute_subtasks(self, original_image_path: str, subtasks: List[str], seed: int = 42, enhanced_prompt_data: Optional[Dict] = None) -> Dict:
        """
        Execute all subtasks

        Args:
            original_image_path: Path to original image
            subtasks: List of subtasks
            enhanced_prompt_data: Enhanced prompt data (optional)

        Returns:
            Dictionary of execution results
        """
        logger.info(f"Starting execution of long-horizon task with {len(subtasks)} subtasks")
        logger.info(f"Original image: {original_image_path}")

        # Record enhanced prompt info
        if enhanced_prompt_data:
            logger.info(f"Using enhanced prompt data, AI generated: {enhanced_prompt_data.get('enhanced_by_ai', False)}")
        else:
            logger.info("No enhanced prompt data provided")

        current_image_path = original_image_path
        task_results = []
        video_paths = []

        # Phase statistics aggregation (excluding task decomposition time; decomposition is stats in main flow)
        planning_time_total = 0.0
        generation_time_total = 0.0
        planning_peak_bytes = 0
        generation_peak_bytes = 0

        # Variables for trajectory continuity between tasks
        trajectory_continuity = {
            "previous_robot_exit_position": None,  # Robot exit position from previous task
            "last_robot_trajectory": []  # Complete robot trajectory from previous task
        }

        for i, subtask in enumerate(subtasks, 1):
            task_name = f"task{i}"

            # Create separate directory structure for each subtask (all under base_temp_dir)
            task_dir = os.path.join(self.base_temp_dir, task_name)
            task_robot_output_dir = os.path.join(task_dir, "robot_output")
            task_video_input_dir = os.path.join(task_dir, "video_input")
            task_video_output_dir = os.path.join(task_dir, "video_output")

            # Clean current subtask directory, ensure no old files remain (based on keep_temp setting)
            if os.path.exists(task_dir) and not self.keep_temp:
                logger.info(f"Cleaning current subtask directory: {task_dir}")
                try:
                    shutil.rmtree(task_dir)
                except Exception as e:
                    logger.warning(f"Failed to clean subtask directory: {task_dir}, Error: {str(e)}")
            elif os.path.exists(task_dir) and self.keep_temp:
                logger.info(f"Keeping subtask directory for debugging: {task_dir}")

            os.makedirs(task_robot_output_dir, exist_ok=True)
            os.makedirs(task_video_input_dir, exist_ok=True)
            os.makedirs(task_video_output_dir, exist_ok=True)

            logger.info(f"\n{'='*80}")
            logger.info(f"Executing Subtask {i}/{len(subtasks)}: {subtask}")
            logger.info(f"Robot arm output dir: {task_robot_output_dir}")
            logger.info(f"Video input dir: {task_video_input_dir}")
            logger.info(f"Video output dir: {task_video_output_dir}")
            logger.info(f"Input image: {current_image_path}")
            logger.info(f"{'='*80}")

            # Step 1: Execute Robot Controller (Output to robot_output directory)
            # Get enhanced prompts for current subtask (containing structured ActionType-Object-Destination info)
            current_task_enhanced_prompts = None
            if enhanced_prompt_data and "subtasks" in enhanced_prompt_data:
                if i-1 < len(enhanced_prompt_data["subtasks"]):
                    current_task_enhanced_prompts = enhanced_prompt_data["subtasks"][i-1]

                    # Add debug info to show structured subtask info
                    if "structured_subtask" in current_task_enhanced_prompts:
                        structured = current_task_enhanced_prompts["structured_subtask"]
                        logger.info(f"Structured subtask info:")
                        logger.info(f"  ActionType: {structured.get('ActionType', 'N/A')}")
                        logger.info(f"  Object: {structured.get('Object', 'N/A')}")
                        logger.info(f"  Destination: {structured.get('Destination', 'N/A')}")
                        logger.info(f"  Original Text: {structured.get('original_text', 'N/A')}")

            # Planning phase (this subtask): From starting controller execution to fully preparing generation input files
            # Only count peak memory for this process (PyTorch) to avoid interference from other processes
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
            planning_t0 = time.perf_counter()

            json_data_path = self._execute_robot_controller(
                current_image_path, subtask, task_robot_output_dir, current_task_enhanced_prompts,
                trajectory_continuity, task_id=i
            )

            if not json_data_path:
                logger.error(f"Subtask {i} robot control failed")
                return {
                    "success": False,
                    "error": f"Task {i} robot control failed",
                    "completed_tasks": task_results,
                    "current_task_id": i
                }

            # Step 2: Prepare video generation data (Copy files from robot_output to video_input directory)
            prepared_files = self._prepare_video_generation_data(
                current_image_path, json_data_path, task_video_input_dir, task_name
            )

            if not prepared_files:
                logger.error(f"Subtask {i} video data preparation failed")
                return {
                    "success": False,
                    "error": f"Task {i} video data preparation failed",
                    "completed_tasks": task_results,
                    "current_task_id": i
                }

            # End planning phase timing and VRAM peak for this subtask
            planning_t1 = time.perf_counter()
            planning_duration = planning_t1 - planning_t0
            # Read planning phase peak VRAM (this process only)
            planning_peak_torch = 0
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    planning_peak_torch = int(torch.cuda.max_memory_allocated())
                except Exception:
                    planning_peak_torch = 0
            planning_peak = int(planning_peak_torch)
            planning_time_total += float(planning_duration)
            planning_peak_bytes = max(planning_peak_bytes, planning_peak)

            # Step 3: Generate multiple candidate videos (Read from video_input, output to video_output)
            # Generate different seeds for each subtask to ensure diversity but maintain reproducibility
            base_task_seed = seed + i  # Base seed + task index

            # Generation phase: Generate multiple candidate videos
            gen_result = self._generate_candidate_videos(
                task_video_input_dir, task_video_output_dir, task_name, base_task_seed
            )

            if not gen_result or not gen_result.get("success"):
                logger.error(f"Subtask {i} candidate video generation failed")
                return {
                    "success": False,
                    "error": f"Task {i} candidate video generation failed",
                    "completed_tasks": task_results,
                    "current_task_id": i
                }

            # Step 4: Evaluate candidate videos using Gemini API and select the best one
            logger.info(f"Starting evaluation of {len(gen_result['candidate_videos'])} candidate videos for subtask {i}")
            evaluation_result = self._evaluate_candidate_videos_with_gemini(
                gen_result["candidate_videos"], subtask, task_video_output_dir, i
            )

            if not evaluation_result or not evaluation_result.get("success"):
                logger.error(f"Subtask {i} video evaluation failed")
                return {
                    "success": False,
                    "error": f"Task {i} video evaluation failed",
                    "completed_tasks": task_results,
                    "current_task_id": i
                }

            selected_video_path = evaluation_result["selected_video_path"]
            video_paths.append(selected_video_path)

            # Log info for all candidate videos for debugging
            gen_duration = float(gen_result.get("time_sec", 0.0))
            gen_peak = int(gen_result.get("gpu_peak_bytes", 0))
            generation_time_total += gen_duration
            generation_peak_bytes = max(generation_peak_bytes, gen_peak)

            logger.info(f"Subtask {i} selected best video: {os.path.basename(selected_video_path)}")
            logger.info(f"Evaluation Score: {evaluation_result.get('evaluation_score', 'N/A')}")

            # Step 5: If best video does not meet requirements, perform replanning
            if not evaluation_result.get("meets_threshold", False):
                logger.warning(f"Subtask {i} best video did not meet success threshold, replanning required")
                replan_result = self._replan_and_regenerate(
                    task_video_input_dir, task_video_output_dir, task_name, subtask,
                    base_task_seed, evaluation_result.get("feedback", ""), i
                )

                if replan_result and replan_result.get("success"):
                    selected_video_path = replan_result["selected_video_path"]
                    # Update video path in video_paths
                    video_paths[-1] = selected_video_path
                    logger.info(f"Subtask {i} replanning successful, using new video: {os.path.basename(selected_video_path)}")
                else:
                    logger.error(f"Subtask {i} replanning failed, continuing with original video")
                    # Continue using original video, but do not mark as failed

            # Step 6: Extract last frame as input for next task (if not the last task)
            if i < len(subtasks):
                next_image_path = self._extract_last_frame(
                    selected_video_path, task_video_output_dir, task_name
                )
                current_image_path = next_image_path
                logger.info(f"Extracted last frame as input for next task: {current_image_path}")
            else:
                logger.info("This is the last subtask, no need to extract last frame")

            # Step 7: Copy visualization files to main output directory
            self._copy_visualization_files(
                task_robot_output_dir, task_name, subtask, i
            )

            task_results.append({
                "task_id": i,
                "task_name": task_name,
                "subtask": subtask,
                "video_path": selected_video_path,
                "input_image": current_image_path if i == len(subtasks) else f"{task_name}_last_frame.png",
                "prepared_files": prepared_files,
                "json_data_path": json_data_path,
                "robot_output_dir": task_robot_output_dir,
                "video_input_dir": task_video_input_dir,
                "video_output_dir": task_video_output_dir
            })

        logger.info(f"All subtasks completed, generated {len(video_paths)} videos")
        return {
            "success": True,
            "task_results": task_results,
            "video_paths": video_paths,
            "total_tasks": len(subtasks),
            "metrics": {
                # Note: planning_time_sec/gpu_peak_bytes are stats for this executor phase, excluding overall task decomposition time
                "planning_time_sec": planning_time_total,
                "planning_gpu_peak_bytes": int(planning_peak_bytes),
                "generation_time_sec": generation_time_total,
                "generation_gpu_peak_bytes": int(generation_peak_bytes)
            }
        }

    def _execute_robot_controller_structured(self, image_path: str, structured_subtask: Dict, task_dir: str, trajectory_continuity: Optional[Dict] = None, task_id: int = 1) -> Optional[str]:
        """
        Execute robot controller for structured subtasks.

        Args:
            image_path: Input image path
            structured_subtask: Structured subtask data (containing ActionType, Object, Destination)
            task_dir: Task output directory
            trajectory_continuity: Trajectory continuity information dictionary
            task_id: Current task ID

        Returns:
            Path to JSON data file or None
        """
        logger.info("Executing structured robot controller...")

        try:
            # Delayed import of UniversalRobotController
            global UniversalRobotController
            if UniversalRobotController is None:
                from vlm_api.universal_robot_controller import UniversalRobotController

            # Get robot exit position from previous task
            previous_robot_exit_position = trajectory_continuity["previous_robot_exit_position"] if trajectory_continuity else None

            # Initialize robot controller
            controller = UniversalRobotController(output_dir=task_dir)

            # Execute robot task, passing structured subtask data directly
            result = controller.execute_universal_robot_task_structured(
                image_path=image_path,
                structured_subtask=structured_subtask,
                previous_robot_exit_position=previous_robot_exit_position
            )

            if result["success"]:
                json_data_path = result["video_data_path"]
                logger.info(f"Structured robot controller executed successfully: {json_data_path}")
                logger.info(f"Target Object: {structured_subtask.get('Object', 'Unknown')}")

                # Update robot exit position in trajectory_continuity
                if "robot_exit_position" in result and result["robot_exit_position"]:
                    trajectory_continuity["previous_robot_exit_position"] = result["robot_exit_position"]
                    logger.info(f"Updated robot exit position for next task: {trajectory_continuity['previous_robot_exit_position']}")

                return json_data_path
            else:
                logger.error(f"Structured robot controller execution failed: {result.get('error', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Structured robot controller execution exception: {str(e)}")
            return None

    def _execute_robot_controller(self, image_path: str, instruction: str, task_dir: str, enhanced_prompt_data: Optional[Dict] = None, trajectory_continuity: Optional[Dict] = None, task_id: int = 1) -> Optional[str]:
        """
        Execute robot controller - Reuses existing logic.

        Args:
            image_path: Input image path
            instruction: User instruction
            task_dir: Task output directory
            enhanced_prompt_data: Enhanced prompt data (optional)
            trajectory_continuity: Trajectory continuity info dictionary (for task continuity)
            task_id: Current task ID (for debugging)

        Returns:
            Path to JSON data file or None
        """
        logger.info("Executing robot controller...")

        try:
            # Delayed import of UniversalRobotController to avoid dependency issues during testing
            global UniversalRobotController
            if UniversalRobotController is None:
                from vlm_api.universal_robot_controller import UniversalRobotController

            # Get robot exit position from previous task
            previous_robot_exit_position = trajectory_continuity["previous_robot_exit_position"] if trajectory_continuity else None

            # Initialize robot controller
            controller = UniversalRobotController(output_dir=task_dir)

            # Execute robot task
            result = controller.execute_universal_robot_task(
                image_path=image_path,
                user_instruction=instruction,
                enhanced_prompt_data=enhanced_prompt_data,
                previous_robot_exit_position=previous_robot_exit_position
            )

            if result["success"]:
                json_data_path = result["video_data_path"]
                logger.info(f"Robot controller executed successfully: {json_data_path}")
                logger.info(f"Target Object: {result['object_name']}")

                # Update robot exit position in trajectory_continuity
                if "robot_exit_position" in result and result["robot_exit_position"]:
                    trajectory_continuity["previous_robot_exit_position"] = result["robot_exit_position"]
                    logger.info(f"Updated robot exit position for next task: {trajectory_continuity['previous_robot_exit_position']}")

                return json_data_path
            else:
                logger.error(f"Robot controller execution failed: {result.get('error', 'Unknown error')}")
                return None

        except Exception as e:
            logger.error(f"Robot controller execution exception: {str(e)}")
            return None

    def _prepare_video_generation_data(self, image_path: str, json_data_path: str,
                                     video_input_dir: str, task_name: str) -> Optional[Dict[str, str]]:
        """
        Prepare video generation data - Copy files from robot_output directory to video_input directory

        Args:
            image_path: Input image path
            json_data_path: JSON data path generated by robot controller
            video_input_dir: Video input directory
            task_name: Task name

        Returns:
            Dictionary of prepared file paths or None
        """
        logger.info("Preparing video generation data...")

        try:
            # Load JSON data
            with open(json_data_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # Copy image file to video_input directory, using task name
            prepared_image_path = os.path.join(video_input_dir, f"{task_name}.png")
            shutil.copy2(image_path, prepared_image_path)

            # Generate all required files into video_input directory
            files = {}

            # Text prompt
            txt_path = os.path.join(video_input_dir, f"{task_name}.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(json_data['metadata']['user_instruction'])
            files["txt"] = txt_path

            # Random seed
            seed_path = os.path.join(video_input_dir, f"{task_name}_seed.txt")
            with open(seed_path, 'w') as f:
                f.write(str(json_data['metadata']['random_seed']))
            files["seed"] = seed_path

            # Trajectory data - Use full trajectory data (37 frames) for compatibility
            robot_traj_path = os.path.join(video_input_dir, f"{task_name}_robot.npy")
            obj_traj_path = os.path.join(video_input_dir, f"{task_name}_obj.npy")
            transit_path = os.path.join(video_input_dir, f"{task_name}_transit.npy")

            robot_traj_data = json_data['file_info']['robot_trajectory_full']
            obj_traj_data = json_data['file_info']['object_trajectory_full']
            transit_data = json_data['metadata']['transit_frames']

            np.save(robot_traj_path, np.array(robot_traj_data))
            np.save(obj_traj_path, np.array(obj_traj_data))
            np.save(transit_path, np.array(transit_data))

            files["robot_traj"] = robot_traj_path
            files["obj_traj"] = obj_traj_path
            files["transit"] = transit_path

            # Object mask - Copy from robot_output directory to video_input directory
            mask_source = json_data['file_info']['mask_path']
            mask_dest = os.path.join(video_input_dir, f"{task_name}_obj_mask.npy")

            if mask_source and os.path.exists(mask_source):
                # Load mask and ensure it's bool format, consistent with official format
                mask_data = np.load(mask_source)

                # If mask is uint8 format (0, 255), convert to bool format
                if mask_data.dtype == np.uint8:
                    # Set values > 0 to True, rest to False
                    bool_mask = mask_data > 0
                    np.save(mask_dest, bool_mask)
                    logger.info(f"Converted mask to bool format: {np.sum(bool_mask)} True pixels")
                else:
                    # If already bool format, save directly
                    np.save(mask_dest, mask_data)
                    logger.info(f"Copied bool mask: {np.sum(mask_data)} True pixels")
            else:
                logger.warning(f"Mask file does not exist: {mask_source}")
                return None

            logger.info("Video generation data preparation complete:")
            logger.info(f"  Image: {prepared_image_path}")
            logger.info(f"  Text: {txt_path}")
            logger.info(f"  Seed: {seed_path}")
            logger.info(f"  Robot Traj: {robot_traj_path}")
            logger.info(f"  Object Traj: {obj_traj_path}")
            logger.info(f"  Mask: {mask_dest}")

            return files

        except Exception as e:
            logger.error(f"Failed to prepare video generation data: {str(e)}")
            return None

    def _generate_video(self, video_input_dir: str, video_output_dir: str, task_name: str, seed: int = 42) -> Dict:
        """
        Generate a single video - Calls inference.py

        Args:
            video_input_dir: Video input directory (contains all required files)
            video_output_dir: Video output directory
            task_name: Task name
            seed: Random seed

        Returns:
            dict: {"video_path": str or None, "time_sec": float, "gpu_peak_bytes": int}
        """
        logger.info("Starting video generation...")

        # Set random seed to ensure video generation reproducibility
        # Use passed seed to ensure controlled randomness for each subtask
        logger.info(f"Setting video generation random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Write seed to seed file for inference.py to use
        # Here we use the seed filename corresponding to task1.png
        input_files = os.listdir(video_input_dir)
        for input_file in input_files:
            if input_file.endswith('.png'):
                base_name = input_file.replace('.png', '')
                seed_file = os.path.join(video_input_dir, f"{base_name}_seed.txt")
                with open(seed_file, 'w') as f:
                    f.write(str(seed))
                logger.info(f"Seed written to file: {seed_file}")
                break

        try:
            import subprocess

            cmd = [
                "python", "inference.py",
                "--input_path", video_input_dir,
                "--output_path", video_output_dir,
                "--model_path", "/data/rczhang/MIND-V/ckpts/CogVideoX-Fun-V1.5-5b-InP",
                "--transformer_path", "/data/rczhang/MIND-V/ckpts/MIND-V",
                "--num_inference_steps", str(self.num_inference_steps)
            ]

            logger.info(f"Running video generation command: {' '.join(cmd)}")
            logger.info(f"Input dir: {video_input_dir}")
            logger.info(f"Output dir: {video_output_dir}")

            # Run video generation, displaying output in real-time
            logger.info("Starting video generation process...")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True,
                cwd=os.getcwd()
            )

            # Sample peak VRAM by PID; fallback to global GPU peak sampling if unavailable
            pid_sampler = None
            global_sampler = None
            if self.enable_gpu_stats:
                pid_sampler = self._start_nvidia_smi_pid_sampler(process.pid)
                if pid_sampler is None:
                    global_sampler = self._start_nvidia_smi_sampler()
            gen_t0 = time.perf_counter()

            # Display output in real-time
            for line in iter(process.stdout.readline, ''):
                if line.strip():
                    logger.info(f"VideoGen: {line.strip()}")

            # Wait for process to complete
            return_code = process.wait()
            gen_t1 = time.perf_counter()
            gen_peak_bytes = 0
            if self.enable_gpu_stats:
                gen_peak_bytes = self._stop_nvidia_smi_pid_sampler(pid_sampler)
                if not gen_peak_bytes and global_sampler is not None:
                    gen_peak_bytes = self._stop_nvidia_smi_sampler(global_sampler)

            if return_code == 0:
                logger.info("Video generation complete")

                # Find generated video file
                possible_video_names = [
                    f"{task_name}_generated.mp4",
                    f"{task_name}.mp4",
                    "generated_video.mp4",
                    "output.mp4"
                ]

                for video_name in possible_video_names:
                    video_path = os.path.join(video_output_dir, video_name)
                    if os.path.exists(video_path):
                        logger.info(f"Found generated video: {video_path}")
                        return {"video_path": video_path, "time_sec": float(gen_t1 - gen_t0), "gpu_peak_bytes": int(gen_peak_bytes)}

                # If no standard names found, look for any mp4 file in the directory
                for file in os.listdir(video_output_dir):
                    if file.endswith('.mp4'):
                        video_path = os.path.join(video_output_dir, file)
                        logger.info(f"Found video file: {video_path}")
                        return {"video_path": video_path, "time_sec": float(gen_t1 - gen_t0), "gpu_peak_bytes": int(gen_peak_bytes)}

                logger.error("Generated video file not found")
                return {"video_path": None, "time_sec": float(gen_t1 - gen_t0), "gpu_peak_bytes": int(gen_peak_bytes)}
            else:
                logger.error(f"Video generation failed, return code: {return_code}")
                return {"video_path": None, "time_sec": float(gen_t1 - gen_t0), "gpu_peak_bytes": int(gen_peak_bytes)}

        except Exception as e:
            logger.error(f"Video generation exception: {str(e)}")
            return {"video_path": None, "time_sec": 0.0, "gpu_peak_bytes": 0}

    def _extract_last_frame(self, video_path: str, task_dir: str, task_name: str) -> str:
        """
        Extract the last frame of the video as input for the next task.

        Args:
            video_path: Video file path
            task_dir: Task directory
            task_name: Task name

        Returns:
            Path to the extracted last frame image
        """
        logger.info(f"Extracting last frame of video: {video_path}")

        try:
            next_task_image_path = os.path.join(task_dir, f"{task_name}_last_frame.png")

            if CV2_AVAILABLE:
                # Use OpenCV to extract last frame
                return self._extract_last_frame_cv2(video_path, next_task_image_path)
            else:
                # Use ffmpeg to extract last frame
                return self._extract_last_frame_ffmpeg(video_path, next_task_image_path)

        except Exception as e:
            logger.error(f"Failed to extract last frame: {str(e)}")
            logger.info("Will use original input image as input for next task")

            # If extraction fails, return original image path
            original_image_path = os.path.join(task_dir, f"{task_name}.png")
            if os.path.exists(original_image_path):
                return original_image_path
            else:
                logger.error("Original input image also does not exist")
                raise Exception("Neither extracted frame nor original image available")

    def _extract_last_frame_cv2(self, video_path: str, output_path: str) -> str:
        """Use OpenCV to extract the last frame"""
        logger.info("Using OpenCV to extract the last frame")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            raise Exception(f"Cannot open video file: {video_path}")

        # Get total frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Total video frames: {frame_count}")

        if frame_count <= 0:
            logger.error("Video frame count is 0 or invalid")
            cap.release()
            raise Exception("Invalid video frame count")

        # Jump to last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, last_frame = cap.read()

        if ret:
            # Save last frame
            cv2.imwrite(output_path, last_frame)

            # Verify file was saved successfully
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"Successfully extracted last frame: {output_path} (size: {file_size} bytes)")

                # Get image dimensions
                height, width = last_frame.shape[:2]
                logger.info(f"Image dimensions: {width}x{height}")

                cap.release()
                return output_path
            else:
                logger.error("Failed to save last frame")
                cap.release()
                raise Exception("Failed to save last frame")
        else:
            logger.error("Cannot read last frame")
            cap.release()
            raise Exception("Cannot read last frame")

    def _extract_last_frame_ffmpeg(self, video_path: str, output_path: str) -> str:
        """Use ffmpeg to extract the last frame"""
        logger.info("Using ffmpeg to extract the last frame")

        try:
            import subprocess

            # Use ffmpeg to extract last frame
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-vf", "select=eq(n\\,framecount-1)",
                "-vsync", "vfr",
                "-frames:v", "1",
                "-y",  # Overwrite output file
                output_path
            ]

            logger.info(f"Running ffmpeg command: {' '.join(cmd)}")

            # Run ffmpeg command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 seconds timeout
            )

            if result.returncode == 0:
                # Verify file was saved successfully
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    logger.info(f"Successfully extracted last frame: {output_path} (size: {file_size} bytes)")
                    return output_path
                else:
                    logger.error("ffmpeg succeeded but output file not found")
                    raise Exception("ffmpeg succeeded but output file not found")
            else:
                logger.error(f"ffmpeg failed: {result.stderr}")
                raise Exception(f"ffmpeg failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.error("ffmpeg execution timeout")
            raise Exception("ffmpeg execution timeout")
        except FileNotFoundError:
            logger.error("ffmpeg not found, please ensure ffmpeg is installed")
            raise Exception("ffmpeg not found")
        except Exception as e:
            logger.error(f"ffmpeg extraction failed: {str(e)}")
            raise Exception(f"ffmpeg extraction failed: {str(e)}")

    def _extract_frames_for_evaluation(self, video_path: str, output_dir: str, candidate_id: int, num_frames: int = 3) -> List[str]:
        """
        Extract multiple keyframes from video for Gemini API evaluation

        Args:
            video_path: Video file path
            output_dir: Output directory
            candidate_id: Candidate video ID
            num_frames: Number of frames to extract (default: 3)

        Returns:
            List[str]: List of extracted frame image paths
        """
        logger.info(f"Extracting {num_frames} keyframes for candidate video {candidate_id}: {video_path}")

        try:
            os.makedirs(output_dir, exist_ok=True)
            frame_paths = []

            if CV2_AVAILABLE:
                # Use OpenCV to extract frames
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logger.error(f"Cannot open video file: {video_path}")
                    raise Exception(f"Cannot open video file: {video_path}")

                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                logger.info(f"Total video frames: {frame_count}")

                if frame_count <= 0:
                    logger.error("Video frame count is 0 or invalid")
                    cap.release()
                    raise Exception("Invalid video frame count")

                # Calculate indices of frames to extract (evenly distributed)
                frame_indices = []
                if frame_count <= num_frames:
                    # If frame count less than requested, extract all frames
                    frame_indices = list(range(frame_count))
                else:
                    # Evenly select frames
                    step = frame_count // num_frames
                    frame_indices = [i * step for i in range(num_frames)]
                    # Ensure last frame is included
                    if frame_indices[-1] != frame_count - 1:
                        frame_indices[-1] = frame_count - 1

                # Extract frames
                for i, frame_idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()

                    if ret:
                        frame_path = os.path.join(output_dir, f"candidate_{candidate_id}_frame_{i+1}.png")
                        cv2.imwrite(frame_path, frame)
                        frame_paths.append(frame_path)
                        logger.info(f"Extracted frame {i+1}/{len(frame_indices)}: {frame_path}")
                    else:
                        logger.warning(f"Could not read frame {frame_idx}")

                cap.release()

            else:
                # Use ffmpeg to extract frames
                import subprocess

                frame_count = self._get_video_frame_count(video_path)

                if frame_count <= num_frames:
                    # If frame count less than requested, extract all frames
                    frame_indices = list(range(frame_count))
                else:
                    # Evenly select frames
                    step = frame_count // num_frames
                    frame_indices = [i * step for i in range(num_frames)]
                    # Ensure last frame is included
                    if frame_indices[-1] != frame_count - 1:
                        frame_indices[-1] = frame_count - 1

                # Use ffmpeg to extract each frame
                for i, frame_idx in enumerate(frame_indices):
                    frame_path = os.path.join(output_dir, f"candidate_{candidate_id}_frame_{i+1}.png")

                    cmd = [
                        "ffmpeg",
                        "-i", video_path,
                        "-vf", f"select=eq(n\\,{frame_idx})",
                        "-vsync", "vfr",
                        "-frames:v", "1",
                        "-y",
                        frame_path
                    ]

                    logger.info(f"Extracting frame {i+1}/{len(frame_indices)}: Frame Index {frame_idx}")

                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )

                    if result.returncode == 0 and os.path.exists(frame_path):
                        frame_paths.append(frame_path)
                        logger.info(f"Successfully extracted frame: {frame_path}")
                    else:
                        logger.error(f"ffmpeg failed to extract frame: {result.stderr}")

            if not frame_paths:
                logger.error("No frames extracted")
                raise Exception("No frames extracted from video")

            logger.info(f"Successfully extracted {len(frame_paths)} keyframes for evaluation")
            return frame_paths

        except Exception as e:
            logger.error(f"Failed to extract frames for evaluation: {str(e)}")
            raise Exception(f"Failed to extract frames for evaluation: {str(e)}")

    def _get_video_frame_count(self, video_path: str) -> int:
        """
        Get the number of frames in a video

        Args:
            video_path: Video file path

        Returns:
            int: Number of frames
        """
        try:
            import subprocess

            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=nb_read_packets",
                "-of", "csv=p=0",
                video_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                frame_count = int(result.stdout.strip())
                logger.info(f"Video frame count: {frame_count}")
                return frame_count
            else:
                logger.error(f"Failed to get video frame count: {result.stderr}")
                # Return a reasonable default
                return 30  # Assume at least 30 frames

        except Exception as e:
            logger.error(f"Exception getting video frame count: {str(e)}")
            return 30  # Return a reasonable default

    def _start_nvidia_smi_sampler(self):
        """Starts a background thread to periodically sample nvidia-smi memory usage (MiB). Returns control handle. Returns None if unavailable."""
        try:
            import shutil as _shutil
            if _shutil.which("nvidia-smi") is None:
                return None
        except Exception:
            return None

        stop_event = threading.Event()
        peak_holder = {"mib": 0}

        def _sampler():
            import subprocess as _sp
            while not stop_event.is_set():
                try:
                    res = _sp.run(
                        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if res.returncode == 0 and res.stdout:
                        vals = []
                        for line in res.stdout.splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                vals.append(int(line))
                            except Exception:
                                pass
                        if vals:
                            current_max = max(vals)
                            if current_max > peak_holder["mib"]:
                                peak_holder["mib"] = current_max
                except Exception:
                    pass
                time.sleep(0.2)

        th = threading.Thread(target=_sampler, daemon=True)
        th.start()
        return (stop_event, th, peak_holder)

    def _stop_nvidia_smi_sampler(self, handle) -> int:
        """Stops memory sampling thread and returns peak memory (bytes). Returns 0 if handle is None."""
        if not handle:
            return 0
        stop_event, th, peak_holder = handle
        try:
            stop_event.set()
            th.join(timeout=2.0)
        except Exception:
            pass
        try:
            mib = int(peak_holder.get("mib", 0))
        except Exception:
            mib = 0
        return mib * 1024 * 1024

    def _start_nvidia_smi_pid_sampler(self, pid: int):
        """Sample peak memory (MiB) by PID to avoid interference from other processes. Returns handle. stop returns peak (bytes)."""
        try:
            import shutil as _shutil
            if _shutil.which("nvidia-smi") is None:
                return None
        except Exception:
            return None

        stop_event = threading.Event()
        peak_holder = {"mib": 0}

        def _sampler():
            import subprocess as _sp
            while not stop_event.is_set():
                try:
                    # Query all compute processes, filter by PID
                    res = _sp.run(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=pid,used_gpu_memory",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    if res.returncode == 0 and res.stdout:
                        peak_mib = 0
                        for line in res.stdout.splitlines():
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) < 2:
                                continue
                            try:
                                p = int(parts[0])
                                mem_mib = int(parts[1])
                            except Exception:
                                continue
                            if p == int(pid):
                                if mem_mib > peak_mib:
                                    peak_mib = mem_mib
                        if peak_mib > peak_holder["mib"]:
                            peak_holder["mib"] = peak_mib
                except Exception:
                    pass
                time.sleep(0.2)

        th = threading.Thread(target=_sampler, daemon=True)
        th.start()
        return (stop_event, th, peak_holder)

    def _stop_nvidia_smi_pid_sampler(self, handle) -> int:
        if not handle:
            return 0
        stop_event, th, peak_holder = handle
        try:
            stop_event.set()
            th.join(timeout=2.0)
        except Exception:
            pass
        try:
            mib = int(peak_holder.get("mib", 0))
        except Exception:
            mib = 0
        return mib * 1024 * 1024

    def _copy_visualization_files(self, task_robot_output_dir: str, task_name: str, subtask: str, task_id: int):
        """
        Copy visualization files to the 'visualization' subdirectory of the main output directory.

        Args:
            task_robot_output_dir: Robot output directory
            task_name: Task name
            subtask: Subtask description
            task_id: Task ID
        """
        logger.info(f"Copying visualization files: {task_name} - {subtask}")

        try:
            # Main output directory ('visualization' directory inside the parent of base_temp_dir)
            main_output_dir = os.path.dirname(os.path.dirname(self.base_temp_dir))
            visualization_dir = os.path.join(main_output_dir, "visualization")
            os.makedirs(visualization_dir, exist_ok=True)

            # Keep only necessary visualizations:
            # 1) Object mask
            # 2) Object affordance output
            # 3) Trajectory visualization (will be redrawn as three-stage trajectory in long-horizon pipeline)

            # Copy segmentation results (if they exist)
            segmentation_dir = os.path.join(task_robot_output_dir, "segmentation")
            if os.path.exists(segmentation_dir):
                for file_name in os.listdir(segmentation_dir):
                    if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    source_path = os.path.join(segmentation_dir, file_name)
                    fname_lower = file_name.lower()

                    # Object affordance actionable area
                    if "affordance" in fname_lower and "mask" in fname_lower:
                        dest_name = f"task{task_id}_affordance_mask.png"
                    # Object mask
                    elif ("object" in fname_lower or "obj" in fname_lower) and "mask" in fname_lower:
                        dest_name = f"task{task_id}_object_mask.png"
                    else:
                        # Other visualizations (segmentation, final_overlay, etc.) are not copied in long-horizon tasks to avoid redundancy
                        continue

                    dest_path = os.path.join(visualization_dir, dest_name)
                    shutil.copy2(source_path, dest_path)
                    logger.info(f"Copied segmentation file: {dest_name}")

            # Copy trajectory visualization (if it exists)
            movement_dir = os.path.join(task_robot_output_dir, "movement")
            if os.path.exists(movement_dir):
                for file_name in os.listdir(movement_dir):
                    if "trajectory" in file_name.lower() and file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        source_path = os.path.join(movement_dir, file_name)
                        dest_name = f"task{task_id}_trajectory.png"
                        dest_path = os.path.join(visualization_dir, dest_name)
                        shutil.copy2(source_path, dest_path)
                        logger.info(f"Copied trajectory file: {dest_name}")

            # Save subtask description to file (text info preserved for debugging)
            desc_file = os.path.join(visualization_dir, f"task{task_id}_description.txt")
            with open(desc_file, 'w', encoding='utf-8') as f:
                f.write(f"Task {task_id}: {subtask}\n")
                f.write(f"Task Name: {task_name}\n")
            logger.info(f"Saved task description: task{task_id}_description.txt")

            logger.info(f"Visualization files copied to: {visualization_dir}")

        except Exception as e:
            logger.error(f"Failed to copy visualization files: {str(e)}")

    def get_task_summary(self) -> Dict:
        """
        Get task execution summary

        Returns:
            Task summary information
        """
        try:
            # Count task directories
            task_dirs = []
            for item in os.listdir(self.base_temp_dir):
                item_path = os.path.join(self.base_temp_dir, item)
                if os.path.isdir(item_path) and item.startswith("task"):
                    task_dirs.append(item)

            task_dirs.sort()  # Sort by task ID

            summary = {
                "total_tasks": len(task_dirs),
                "task_directories": task_dirs,
                "base_temp_dir": self.base_temp_dir,
                "log_file": f"{self.base_temp_dir}/long_horizon_execution.log"
            }

            # Collect info for each task
            task_info = []
            for task_dir in task_dirs:
                task_path = os.path.join(self.base_temp_dir, task_dir)
                info = {
                    "task_name": task_dir,
                    "path": task_path,
                    "files": []
                }

                if os.path.exists(task_path):
                    for file in os.listdir(task_path):
                        file_path = os.path.join(task_path, file)
                        if os.path.isfile(file_path):
                            file_size = os.path.getsize(file_path)
                            info["files"].append({
                                "name": file,
                                "size": file_size,
                                "path": file_path
                            })

                task_info.append(info)

            summary["task_details"] = task_info
            return summary

        except Exception as e:
            logger.error(f"Failed to generate task summary: {str(e)}")
            return {"error": str(e)}

    def _get_optimal_parallel_workers(self) -> int:
        """
        Determine optimal parallel workers based on GPU memory and number of candidates

        Returns:
            int: Optimal number of parallel workers
        """
        try:
            import subprocess

            # Get GPU info
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, check=True, timeout=10
                )
                # Handle multi-line output, get memory of first GPU
                memory_lines = result.stdout.strip().split('\n')
                if memory_lines:
                    gpu_memory_mb = int(memory_lines[0].strip())
                    gpu_memory_gb = gpu_memory_mb // 1024
                    logger.info(f"Detected GPU memory: {gpu_memory_gb}GB")
                else:
                    raise ValueError("No GPU memory output found")
            except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
                logger.warning("Failed to get GPU memory info, using default parallelism")
                gpu_memory_gb = 80  # Conservative estimate

            # Determine optimal parallelism based on memory size
            if gpu_memory_gb >= 140:  # 140GB+ - Huge memory, fully utilize
                max_workers = self.num_candidates
            elif gpu_memory_gb >= 120:  # 120GB+
                max_workers = min(self.num_candidates, 6)
            elif gpu_memory_gb >= 80:   # 80GB+
                max_workers = min(self.num_candidates, 4)
            elif gpu_memory_gb >= 40:   # 40GB+
                max_workers = min(self.num_candidates, 3)
            else:                     # <40GB
                max_workers = min(self.num_candidates, 2)

            # Ensure not exceeding number of candidates
            max_workers = min(max_workers, self.num_candidates)

            logger.info(f"Based on {gpu_memory_gb}GB memory, setting parallelism to {max_workers}")
            return max_workers

        except Exception as e:
            logger.error(f"Error determining parallelism: {str(e)}")
            # Fallback to conservative default
            return min(self.num_candidates, 3)

    def _get_available_gpus(self) -> List[int]:
        """
        Get list of available GPU devices

        Returns:
            List[int]: List of available GPU device IDs
        """
        try:
            import subprocess

            # Use nvidia-smi to detect available GPUs
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True, timeout=10
            )

            if result.returncode == 0:
                gpu_lines = result.stdout.strip().split('\n')
                available_gpus = []
                for line in gpu_lines:
                    line = line.strip()
                    if line.isdigit():
                        gpu_id = int(line)
                        available_gpus.append(gpu_id)
                logger.info(f"Detected available GPUs: {available_gpus}")
                return available_gpus
            else:
                logger.warning("nvidia-smi command failed, using default GPU 0")
                return [0]

        except Exception as e:
            logger.error(f"Failed to detect GPU devices: {str(e)}")
            logger.warning("Using default GPU 0")
            return [0]

    def _generate_candidate_videos(self, video_input_dir: str, video_output_dir: str, task_name: str, base_seed: int) -> Dict:
        """
        Generate multiple candidate videos in parallel for evaluation

        Args:
            video_input_dir: Video input directory
            video_output_dir: Video output directory
            task_name: Task name
            base_seed: Base random seed

        Returns:
            dict: {"success": bool, "candidate_videos": List[Dict], "time_sec": float, "gpu_peak_bytes": int}
        """
        logger.info(f"Starting parallel generation of {self.num_candidates} candidate videos...")
        logger.info(f"Sufficient GPU memory, using parallelization to accelerate generation process...")

        import subprocess
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Create subdirectories for each candidate
        candidate_dirs = []
        for candidate_idx in range(self.num_candidates):
            candidate_seed = base_seed + candidate_idx * 1000  # Ensure different seed for each candidate
            candidate_output_dir = os.path.join(video_output_dir, f"candidate_{candidate_idx}")
            os.makedirs(candidate_output_dir, exist_ok=True)

            candidate_dirs.append({
                "candidate_id": candidate_idx,
                "seed": candidate_seed,
                "output_dir": candidate_output_dir,
                "video_input_dir": video_input_dir,
                "task_name": task_name
            })

        def generate_single_candidate(candidate_info):
            """Worker function for generating a single candidate video (supports multi-GPU)"""
            candidate_idx = candidate_info["candidate_id"]
            candidate_seed = candidate_info["seed"]
            candidate_output_dir = candidate_info["output_dir"]
            assigned_gpu = candidate_info.get("assigned_gpu", 0)  # Get assigned GPU ID

            logger.info(f"Starting generation process for candidate video {candidate_idx + 1}/{self.num_candidates} (seed: {candidate_seed}, GPU: {assigned_gpu})")

            # Build command, add GPU ID parameter logic
            cmd = [
                "python", "inference.py",
                "--input_path", candidate_info["video_input_dir"],
                "--output_path", candidate_output_dir,
                "--model_path", "/data/rczhang/MIND-V/ckpts/CogVideoX-Fun-V1.5-5b-InP",
                "--transformer_path", "/data/rczhang/MIND-V/ckpts/MIND-V",
                "--num_inference_steps", str(self.num_inference_steps)
            ]

            # Set GPU environment variable
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)

            try:
                # Record start time
                start_time = time.perf_counter()

                # Start subprocess using the specified GPU
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    universal_newlines=True,
                    env=env  # Use env with GPU setting
                )

                # Real-time monitoring of output (optional, for debugging)
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        logger.debug(f"Candidate {candidate_idx} (GPU {assigned_gpu}): {output.strip()}")

                # Wait for process completion
                return_code = process.wait()
                end_time = time.perf_counter()
                duration = end_time - start_time

                if return_code == 0:
                    # Find generated video files
                    video_files = [f for f in os.listdir(candidate_output_dir) if f.endswith('.mp4')]
                    if video_files:
                        video_path = os.path.join(candidate_output_dir, video_files[0])
                        logger.info(f"Candidate video {candidate_idx + 1} generated successfully: {os.path.basename(video_path)} (Time: {duration:.2f}s, GPU: {assigned_gpu})")
                        return {
                            "candidate_id": candidate_idx,
                            "seed": candidate_seed,
                            "video_path": video_path,
                            "time_sec": duration,
                            "gpu_peak_bytes": 0,  # GPU peak memory needs external monitoring
                            "success": True,
                            "assigned_gpu": assigned_gpu  # Record GPU used
                        }
                    else:
                        logger.error(f"Candidate video {candidate_idx + 1} generation completed but no video file found (GPU: {assigned_gpu})")
                        return {
                            "candidate_id": candidate_idx,
                            "seed": candidate_seed,
                            "success": False,
                            "error": "No video file found",
                            "assigned_gpu": assigned_gpu
                        }
                else:
                    stderr_output = process.stderr.read() if process.stderr else ""
                    logger.error(f"Candidate video {candidate_idx + 1} generation failed (Return code: {return_code}, GPU: {assigned_gpu})")
                    if stderr_output:
                        logger.error(f"Error output: {stderr_output}")
                    return {
                        "candidate_id": candidate_idx,
                        "seed": candidate_seed,
                        "success": False,
                        "error": f"Process failed with return code {return_code}",
                        "assigned_gpu": assigned_gpu
                    }

            except Exception as e:
                logger.error(f"Candidate video {candidate_idx + 1} generation exception: {str(e)} (GPU: {assigned_gpu})")
                return {
                    "candidate_id": candidate_idx,
                    "seed": candidate_seed,
                    "success": False,
                    "error": str(e),
                    "assigned_gpu": assigned_gpu
                }

        # Use ThreadPoolExecutor for parallel generation
        start_time = time.perf_counter()
        candidate_videos = []
        failed_count = 0

        # Determine optimal parallelism based on GPU memory and number of candidates
        # For 140G VRAM, higher parallelism is supported
        max_workers = self._get_optimal_parallel_workers()

        # Detect available GPUs, assign different GPUs to each candidate
        available_gpus = self._get_available_gpus()
        logger.info(f"GPU memory sufficient, using {max_workers} parallel workers to generate candidate videos")
        logger.info(f"Detected {len(available_gpus)} available GPUs: {available_gpus}")

        # If #GPUs >= #Candidates, assign unique GPU to each candidate
        use_multi_gpu = len(available_gpus) >= self.num_candidates
        if use_multi_gpu:
            logger.info(f"Using multi-GPU mode, assigning unique GPU to each candidate video")
            # Assign GPU to each candidate
            for i, candidate_dir in enumerate(candidate_dirs):
                candidate_dir["assigned_gpu"] = available_gpus[i % len(available_gpus)]
        else:
            logger.info(f"Using single-GPU mode, all candidates use GPU {available_gpus[0] if available_gpus else 0}")
            # All candidates use the same GPU
            for candidate_dir in candidate_dirs:
                candidate_dir["assigned_gpu"] = available_gpus[0] if available_gpus else 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_candidate = {
                executor.submit(generate_single_candidate, candidate_dir): candidate_dir
                for candidate_dir in candidate_dirs
            }

            # Collect results
            for future in as_completed(future_to_candidate):
                candidate_dir = future_to_candidate[future]
                try:
                    result = future.result()
                    if result and result.get("success"):
                        candidate_videos.append(result)
                    else:
                        failed_count += 1
                        logger.error(f"Candidate {candidate_dir['candidate_id']} generation failed: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Exception getting result for candidate {candidate_dir['candidate_id']}: {str(e)}")

        total_time = time.perf_counter() - start_time

        # Sort results by candidate_id
        candidate_videos.sort(key=lambda x: x["candidate_id"])

        success_count = len(candidate_videos)
        logger.info(f"Parallel candidate video generation complete: {success_count}/{self.num_candidates} successful (Total time: {total_time:.2f}s)")

        # Log GPU usage
        gpu_usage = {}
        for video in candidate_videos:
            gpu_id = video.get("assigned_gpu", 0)
            gpu_usage[gpu_id] = gpu_usage.get(gpu_id, 0) + 1

        if gpu_usage:
            usage_str = ", ".join([f"GPU {gpu}: {count}" for gpu, count in gpu_usage.items()])
            logger.info(f"GPU usage: {usage_str}")

        if failed_count > 0:
            logger.warning(f"{failed_count} candidate videos failed to generate")

        return {
            "success": success_count > 0,
            "candidate_videos": candidate_videos,
            "time_sec": total_time,
            "gpu_peak_bytes": 0,  # Monitor via nvidia-smi in actual run
            "parallel_generation": True,
            "max_workers_used": max_workers
        }

    def _collect_task_evaluation_context(self, output_dir: str) -> Dict:
        """
        Collect task evaluation context, including original task prompt and trajectory visualization.

        Args:
            output_dir: Output directory

        Returns:
            Dict: Dictionary containing task context information
        """
        context = {"task_prompt": None, "trajectory_visualizations": []}

        try:
            # 1. Find original task prompt
            prompt_files = [
                os.path.join(output_dir, "task_prompt.txt"),
                os.path.join(output_dir, "original_instruction.txt"),
                os.path.join(os.path.dirname(output_dir), "original_instruction.txt")
            ]

            for prompt_file in prompt_files:
                if os.path.exists(prompt_file):
                    with open(prompt_file, 'r', encoding='utf-8') as f:
                        context["task_prompt"] = f.read().strip()
                        logger.info(f"Found task prompt file: {prompt_file}")
                        break

            # 2. Find trajectory visualization images
            robot_output_dir = os.path.join(output_dir, "robot_output")
            if os.path.exists(robot_output_dir):
                # Find trajectory visualization files
                viz_patterns = [
                    "trajectory*.png",
                    "task*_trajectory.png",
                    "*_object_destination_positioning.png",
                    "*_visualization.png"
                ]

                import glob
                for pattern in viz_patterns:
                    viz_files = glob.glob(os.path.join(robot_output_dir, pattern))
                    for viz_file in viz_files:
                        if os.path.exists(viz_file):
                            context["trajectory_visualizations"].append(viz_file)
                            logger.info(f"Found trajectory visualization: {viz_file}")

        except Exception as e:
            logger.warning(f"Error collecting task evaluation context: {str(e)}")

        return context

    def _evaluate_candidate_videos_with_gemini(self, candidate_videos: List[Dict], subtask: str, output_dir: str, task_id: int) -> Dict:
        """
        Evaluate candidate videos using Gemini API and select the best one.

        Args:
            candidate_videos: List of candidate videos
            subtask: Current subtask description
            output_dir: Output directory
            task_id: Task ID

        Returns:
            dict: {"success": bool, "selected_video_path": str, "evaluation_score": float, "meets_threshold": bool, "feedback": str}
        """
        if not candidate_videos:
            return {"success": False, "error": "No candidate videos to evaluate"}

        logger.info("Starting evaluation of candidate videos using Gemini API...")

        try:
            # Get persistent Gemini client
            gemini_client = self._get_gemini_client()
            if gemini_client is None:
                return {"success": False, "error": "Failed to initialize Gemini client"}

            # Build evaluation prompt with historical context
            evaluation_prompt = self._build_contextual_evaluation_prompt(candidate_videos, subtask, task_id)

            # Extract keyframes for each candidate video and add to evaluation
            evaluation_frames_dir = os.path.join(output_dir, "evaluation_frames")
            os.makedirs(evaluation_frames_dir, exist_ok=True)

            for candidate in candidate_videos:
                video_path = candidate["video_path"]
                if os.path.exists(video_path):
                    logger.info(f"Preparing to evaluate video: {os.path.basename(video_path)}")

                    # Extract keyframes for evaluation
                    try:
                        frame_paths = self._extract_frames_for_evaluation(
                            video_path,
                            evaluation_frames_dir,
                            candidate["candidate_id"],
                            num_frames=5  # Extract 5 evenly spaced keyframes
                        )

                        # Add description for all 5 keyframes for each candidate
                        frame_description = f"Candidate Video {candidate['candidate_id']} - Contains {len(frame_paths)} evenly spaced keyframes"

                        # Add all 5 keyframes in chronological order
                        if frame_paths:
                            for i, frame_path in enumerate(frame_paths):
                                if i == 0:
                                    frame_type = "Start"
                                elif i < 4:
                                    frame_type = "Middle"
                                else:
                                    frame_type = "End"
                                frame_info = f"Frame {i+1} ({frame_type})"
                                gemini_client.add_message("user", f"{frame_description} - {frame_info}", frame_path)
                                logger.info(f"Candidate Video {candidate['candidate_id']} - Frame {i+1} used for evaluation")

                            logger.info(f"Candidate Video {candidate['candidate_id']} extracted {len(frame_paths)} keyframes for evaluation")

                    except Exception as e:
                        logger.error(f"Failed to extract frames from candidate {candidate['candidate_id']}: {str(e)}")
                        return {"success": False, "error": f"Failed to extract frames from candidate {candidate['candidate_id']}: {str(e)}"}

                else:
                    logger.error(f"Video file not found: {video_path}")
                    return {"success": False, "error": f"Video file not found: {video_path}"}

            # Add task prompt and trajectory visualization images (if available)
            task_context = self._collect_task_evaluation_context(output_dir)
            if task_context:
                # Add original task prompt
                if task_context.get("task_prompt"):
                    gemini_client.add_message("user", "Original Task Prompt:", task_context["task_prompt"])

                # Add trajectory planning visualization
                if task_context.get("trajectory_visualizations"):
                    gemini_client.add_message("user", "Trajectory Planning Visualization:", task_context["trajectory_visualizations"])

            # Add evaluation instruction (including historical context)
            gemini_client.add_message("user", evaluation_prompt)

            # Call Gemini API for professional evaluation
            evaluation_instruction = """
As an expert robotics vision evaluator, please now perform the comprehensive evaluation based on:
1. The 5 keyframes from each candidate video (in temporal order)
2. The original task requirements and context
3. The trajectory planning visualizations (if provided)

Please apply the three evaluation dimensions consistently and provide the detailed JSON analysis as specified in the prompt.
"""
            response = gemini_client.chat(
                evaluation_instruction,
                retry_attempts=3
            )

            if response:
                logger.info(f"Gemini API Response: {response}")

                # Parse JSON response
                import json
                try:
                    # Attempt to extract JSON part
                    import re
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        evaluation_result = json.loads(json_str)

                        # Validate evaluation result (new three-dimension evaluation format)
                        if "final_selection" in evaluation_result:
                            final_selection = evaluation_result["final_selection"]
                            best_id = final_selection["selected_candidate_id"]
                            overall_score = final_selection.get("overall_score", 0.0)
                            meets_threshold = final_selection.get("meets_threshold", False)
                            comprehensive_feedback = final_selection.get("comprehensive_feedback", "")

                            best_candidate = None
                            for candidate in candidate_videos:
                                if candidate["candidate_id"] == best_id:
                                    best_candidate = candidate
                                    break

                            if best_candidate:
                                # Save evaluation history record
                                history_record = {
                                    "task_id": task_id,
                                    "subtask": subtask,
                                    "best_candidate_id": best_id,
                                    "best_score": overall_score,
                                    "meets_threshold": meets_threshold,
                                    "feedback": comprehensive_feedback,
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    # Add detailed evaluation dimension info
                                    "task_completion_score": evaluation_result.get("task_completion_judge", {}).get("best_score", 0.0),
                                    "physical_plausibility_score": evaluation_result.get("physical_plausibility_judge", {}).get("best_score", 0.0),
                                    "visual_quality_score": evaluation_result.get("visual_quality_judge", {}).get("best_score", 0.0)
                                }
                                self.evaluation_history.append(history_record)

                                logger.info(f"Saved evaluation history: Task {task_id + 1}, Subtask: {subtask[:50]}..., Overall Score: {overall_score}")

                                # Calculate average score for each dimension for logging
                                tc_score = evaluation_result.get("task_completion_judge", {}).get("best_score", 0.0)
                                pp_score = evaluation_result.get("physical_plausibility_judge", {}).get("best_score", 0.0)
                                vq_score = evaluation_result.get("visual_quality_judge", {}).get("best_score", 0.0)

                                logger.info(f"Evaluation Details - Task Completion: {tc_score:.1f}, Physical Plausibility: {pp_score:.1f}, Visual Quality: {vq_score:.1f}")

                                return {
                                    "success": True,
                                    "selected_video_path": best_candidate["video_path"],
                                    "selected_candidate_id": best_candidate["candidate_id"],
                                    "evaluation_score": overall_score,
                                    "meets_threshold": meets_threshold,
                                    "feedback": comprehensive_feedback,
                                    # Detailed evaluation dimension info
                                    "task_completion_score": tc_score,
                                    "physical_plausibility_score": pp_score,
                                    "visual_quality_score": vq_score,
                                    "evaluation_history_count": len(self.evaluation_history),
                                    # Full evaluation result
                                    "full_evaluation_result": evaluation_result
                                }
                            else:
                                logger.error(f"Video for Candidate ID {best_id} not found")
                                return {"success": False, "error": f"Candidate ID {best_id} not found"}
                        else:
                            logger.error("Invalid evaluation result format")
                            return {"success": False, "error": "Invalid evaluation result format"}
                    else:
                        logger.error("Cannot extract JSON from response")
                        return {"success": False, "error": "Cannot extract JSON from response"}

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing failed: {e}")
                    logger.error(f"Original response: {response}")
                    return {"success": False, "error": f"JSON parsing failed: {e}"}
            else:
                logger.error("Empty response from Gemini API")
                return {"success": False, "error": "Empty response from Gemini API"}

        except Exception as e:
            logger.error(f"Gemini API evaluation failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _replan_and_regenerate(self, video_input_dir: str, video_output_dir: str, task_name: str,
                              subtask: str, base_seed: int, feedback: str, task_id: int) -> Dict:
        """
        Replan and generate new candidate videos based on feedback

        Args:
            video_input_dir: Video input directory
            video_output_dir: Video output directory
            task_name: Task name
            subtask: Subtask description
            base_seed: Base seed
            feedback: Feedback from Gemini API
            task_id: Task ID

        Returns:
            dict: {"success": bool, "selected_video_path": str, "evaluation_score": float}
        """
        logger.info("Starting replanning and generation of new candidate videos...")
        logger.info(f"Feedback info: {feedback}")

        try:
            # Generate new candidate video
            new_base_seed = base_seed + 10000  # Use different seed to avoid duplication
            replan_output_dir = os.path.join(video_output_dir, "replan")
            os.makedirs(replan_output_dir, exist_ok=True)

            # Generate single replanned video
            result = self._generate_video(video_input_dir, replan_output_dir, f"{task_name}_replan", new_base_seed)

            if result and result.get("video_path"):
                logger.info("Replanned video generation successful")
                return {
                    "success": True,
                    "selected_video_path": result["video_path"],
                    "evaluation_score": 7.5,  # Placeholder
                    "feedback": "Replanning successful"
                }
            else:
                logger.error("Replanned video generation failed")
                return {"success": False, "error": "Replanned video generation failed"}

        except Exception as e:
            logger.error(f"Replanning failed: {str(e)}")
            return {"success": False, "error": str(e)}


def test_long_horizon_executor():
    """
    Test basic functionality of Long Horizon Task Executor
    """
    print("🧪 Testing Long Horizon Task Executor...")

    # Create test directory
    test_dir = "./test_long_horizon_temp"
    os.makedirs(test_dir, exist_ok=True)

    try:
        executor = LongHorizonExecutor(test_dir)

        # Test task summary
        summary = executor.get_task_summary()
        print("✅ Task summary functionality test passed")
        print(f"   Test directory: {test_dir}")
        print(f"   Task count: {summary['total_tasks']}")

        print("🎉 Long Horizon Task Executor basic functionality test completed!")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

    finally:
        # Clean test directory
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print("🧹 Test directory cleaned")


if __name__ == "__main__":
    test_long_horizon_executor()