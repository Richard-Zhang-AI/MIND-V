#!/usr/bin/env python3
"""
Long-Horizon Robotic Arm Manipulation Video Generation Pipeline
Long-Horizon Robotic Arm Manipulation Video Generation Pipeline

Based on complete_video_pipeline.py, extended to support complex multi-step task video generation.
Decomposes complex instructions into multiple subtasks, executes them sequentially, and concatenates them to generate the final long-horizon task video.

Usage:
python long_horizon_video_pipeline.py --image "/data/rczhang/MIND-V/demos/long_video/bridge1_s1.png" --instruction "First put the towel into the metal pot, then put the spoon into the metal pot" --output "/data/rczhang/MIND-V/output_zrc/11-18"

Based on complete_video_pipeline.py, extended to support complex multi-step task video generation.
"""

import os
import sys
import json
import argparse
import shutil
import random
import time
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, Optional, List

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from vlm_api.task_decomposer import TaskDecomposer
from vlm_api.long_horizon_executor import LongHorizonExecutor
from vlm_api.video_composer import VideoComposer


class PipelineStats:
    """
    Simple statistics collector: Records time and peak VRAM for planning (decomposition) and generation stages, and saves results.
    """

    def __init__(self, enable_gpu_stats: bool = False):
        # Whether to enable GPU statistics (default disabled to improve speed)
        self.enable_gpu_stats = bool(enable_gpu_stats)
        self.gpu_available = False
        self.device_info = {}
        if self.enable_gpu_stats and torch.cuda.is_available():
            try:
                dev = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(dev)
                free_b, total_b = torch.cuda.mem_get_info()
                self.device_info = {
                    "device_index": int(dev),
                    "device_name": props.name,
                    "total_mem_bytes": int(total_b),
                    "free_mem_bytes_at_init": int(free_b)
                }
                self.gpu_available = True
            except Exception:
                self.gpu_available = False
                self.device_info = {}

        self.stages = {
            "planning": {},
            "generation": {},
            "composition": {}
        }
        self._timers = {}
        self.subtasks_count = 0

    def _reset_gpu_peak(self):
        if self.enable_gpu_stats and self.gpu_available:
            try:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    def _read_gpu_peak(self) -> int:
        if self.enable_gpu_stats and self.gpu_available:
            try:
                torch.cuda.synchronize()
                return int(torch.cuda.max_memory_allocated())
            except Exception:
                return 0
        return 0

    def start_stage(self, name: str):
        self._reset_gpu_peak()
        self._timers[name] = time.perf_counter()
        # Record starting VRAM only on the first time
        start_alloc = 0
        if self.enable_gpu_stats and self.gpu_available:
            try:
                start_alloc = int(torch.cuda.memory_allocated())
            except Exception:
                start_alloc = 0
        st = self.stages.setdefault(name, {})
        # Only record starting memory if not already present
        if "gpu_alloc_start_bytes" not in st:
            st["gpu_alloc_start_bytes"] = start_alloc

    def end_stage(self, name: str):
        start = self._timers.get(name)
        end = time.perf_counter()
        duration = float(end - start) if start is not None else 0.0
        peak = self._read_gpu_peak()
        end_alloc = 0
        if self.enable_gpu_stats and self.gpu_available:
            try:
                end_alloc = int(torch.cuda.memory_allocated())
            except Exception:
                end_alloc = 0
        st = self.stages.setdefault(name, {})
        prev_t = float(st.get("time_sec", 0.0))
        st["time_sec"] = prev_t + duration
        # Take maximum peak
        prev_peak = int(st.get("gpu_peak_alloc_bytes", 0))
        st["gpu_peak_alloc_bytes"] = max(prev_peak, int(peak))
        st["gpu_alloc_end_bytes"] = end_alloc

    def add_to_stage(self, name: str, add_time_sec: float = 0.0, new_peak_bytes: int = 0):
        st = self.stages.setdefault(name, {})
        st["time_sec"] = float(st.get("time_sec", 0.0)) + float(add_time_sec)
        st["gpu_peak_alloc_bytes"] = max(int(st.get("gpu_peak_alloc_bytes", 0)), int(new_peak_bytes))

    def finalize(self, subtasks_count: int = 0):
        self.subtasks_count = int(subtasks_count)
        plan_t = float(self.stages.get("planning", {}).get("time_sec", 0.0))
        gen_t = float(self.stages.get("generation", {}).get("time_sec", 0.0))
        comp_t = float(self.stages.get("composition", {}).get("time_sec", 0.0))
        total_t = plan_t + gen_t + comp_t

        plan_peak = int(self.stages.get("planning", {}).get("gpu_peak_alloc_bytes", 0))
        gen_peak = int(self.stages.get("generation", {}).get("gpu_peak_alloc_bytes", 0))
        comp_peak = int(self.stages.get("composition", {}).get("gpu_peak_alloc_bytes", 0))
        max_stage_peak = max(plan_peak, gen_peak, comp_peak)
        two_stage_peak_sum = plan_peak + gen_peak

        # Time ratio within planning/generation stages
        two_stage_time = plan_t + gen_t
        planning_time_ratio = (plan_t / two_stage_time) if two_stage_time > 0 else 0.0
        generation_time_ratio = (gen_t / two_stage_time) if two_stage_time > 0 else 0.0

        # VRAM ratio (relative to peak of both stages) and relative to total device memory
        total_mem_bytes = int(self.device_info.get("total_mem_bytes", 0))
        def ratio(a, b):
            return (float(a) / float(b)) if b else 0.0

        avg_gen_time_per_subtask = (gen_t / self.subtasks_count) if self.subtasks_count > 0 else 0.0

        stats = {
            "gpu_available": self.gpu_available,
            "device_info": self.device_info,
            "subtasks_count": self.subtasks_count,
            "stages": self.stages,
            "overall": {
                "total_time_sec": total_t,
                "max_stage_gpu_peak_alloc_bytes": max_stage_peak
            },
            "proportions": {
                "planning_time_ratio": planning_time_ratio,
                "generation_time_ratio": generation_time_ratio,
                # VRAM ratio uses sum of peaks from both stages as denominator, which aligns better with "proportion" intuition
                "planning_peak_ratio_among_stages": ratio(plan_peak, two_stage_peak_sum),
                "generation_peak_ratio_among_stages": ratio(gen_peak, two_stage_peak_sum),
                "planning_peak_ratio_to_device": ratio(plan_peak, total_mem_bytes),
                "generation_peak_ratio_to_device": ratio(gen_peak, total_mem_bytes)
            },
            "derived": {
                "avg_generation_time_per_subtask_sec": avg_gen_time_per_subtask
            }
        }
        return stats

    @staticmethod
    def save_stats(stats: Dict, logs_dir: str):
        try:
            os.makedirs(logs_dir, exist_ok=True)
            stats_path = os.path.join(logs_dir, "metrics_summary.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"✅ Metrics summary saved: {stats_path}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to save metrics summary: {str(e)}")


def set_random_seed(seed: int = 1008):
    """
    Set all relevant random seeds to ensure reproducibility

    Args:
        seed: Random seed value, default is 42
    """
    print(f"🎲 Setting random seed to {seed} for reproducibility")

    # Python built-in random module
    random.seed(seed)

    # NumPy random seed
    np.random.seed(seed)

    # PyTorch random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Set seed for all GPUs

    # PyTorch backend settings - Allow cuDNN benchmark to improve inference speed
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"✅ Random seed {seed} set successfully")


def parse_arguments():
    """Parse command line arguments - Consistent structure with complete_video_pipeline.py"""
    parser = argparse.ArgumentParser(description="Long-Horizon Robotic Arm Manipulation Video Generation Pipeline")

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image"
    )

    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="Complex instruction for robotic arm manipulation (e.g., 'Put the avocado on the left side of the right table to the left of the table, then put the avocado on the right side of the right table to the left of the table')"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./long_horizon_output",
        help="Output directory for generated videos and intermediate results"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="ckpts/CogVideoX-Fun-V1.5-5b-InP",
        help="Path to base CogVideoX model"
    )

    parser.add_argument(
        "--transformer_path",
        type=str,
        default="ckpts/MIND-V",
        help="Path to trained MIND-V model"
    )

    parser.add_argument(
        "--temp_dir",
        type=str,
        default="./temp_long_horizon",
        help="Temporary directory for intermediate files"
    )

    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep temporary files for debugging"
    )

    parser.add_argument(
        "--transition_frames",
        type=int,
        default=5,
        help="Number of transition frames between subtask videos (default: 5)"
    )

    parser.add_argument(
        "--no_transitions",
        action="store_true",
        help="Disable transition effects between subtask videos"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of denoising steps for video generation (default: 20)"
    )

    parser.add_argument(
        "--num_candidates",
        type=int,
        default=2,
        help="Number of candidate videos to generate per subtask for evaluation (default: 2)"
    )

    # Commented out GPU stats arguments (User requested removal of time and VRAM stats)
    # parser.add_argument(
    #     "--enable_gpu_stats",
    #     action="store_true",
    #     help="Enable detailed GPU memory statistics tracking (default: disabled)"
    # )

    return parser.parse_args()


def setup_directories(output_dir: str, temp_dir: str, image_path: str, seed: int = 42) -> Dict[str, str]:
    """Setup directory structure - Organization based on image name, temp files also stored under output directory"""
    # Extract image filename (without extension) as subdirectory name
    image_base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Create directory name containing seed
    seed_suffix = f"seed{seed}"

    # Store temp files under output directory for easier management and cleanup
    base_output_dir = os.path.join(output_dir, f"{image_base_name}_{seed_suffix}")
    temp_dir = os.path.join(base_output_dir, "temp")

    # All subdirectories are under base_output_dir
    directories = {
        "output": base_output_dir,
        "temp": temp_dir,
        "subtasks": os.path.join(temp_dir, "subtasks"),
        "final_videos": os.path.join(base_output_dir, "final_videos"),
        "logs": os.path.join(base_output_dir, "logs"),
        "visualization": os.path.join(base_output_dir, "visualization")  # Visualization results directory
    }

    # Clean up previous temporary files (if they exist)
    if os.path.exists(temp_dir):
        print(f"🧹 Cleaning up previous temporary files for {image_base_name}: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up temp directory {temp_dir}: {str(e)}")
            # If cleanup fails, try to continue
            pass

    # Create all necessary directories
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)

    print(f"📁 Base output directory: {base_output_dir}")
    print(f"📁 Temporary files directory: {temp_dir}")
    print(f"📁 Visualization directory: {directories['visualization']}")

    return directories


def decompose_complex_task(instruction: str, image_path: str) -> Dict:
    """Decompose complex task into subtasks and generate enhanced prompts"""
    print("🧠 Decomposing complex instruction into subtasks with enhanced prompts...")

    try:
        decomposer = TaskDecomposer(use_enhanced_prompts=True)
        result = decomposer.decompose_task_with_enhanced_prompts(instruction, image_path)

        if result["success"]:
            print(f"✅ Enhanced task decomposition completed!")
            print(f"📋 Generated {result['total_subtasks']} subtasks with AI-generated prompts:")
            print(f"🤖 Enhanced by AI: {result['enhanced_by_ai']}")

            for i, subtask in enumerate(result["subtasks"], 1):
                print(f"   {i}. {subtask['original_subtask']}")
                print(f"      Positive: {subtask['positive_prompt'][:80]}...")
                print(f"      Negative: {subtask['negative_prompt'][:60]}...")
        else:
            print(f"⚠️  Enhanced decomposition failed (Gemini-only mode, no fallback)")
            print(f"📋 Generated {result['total_subtasks']} subtasks")

        return result

    except Exception as e:
        print(f"❌ Task decomposition failed: {str(e)}")
        return {"success": False, "subtasks": [], "total_subtasks": 0}


def execute_long_horizon_task(image_path: str, enhanced_task_data: Dict, directories: Dict[str, str],
                              seed: int = 42, num_inference_steps: int = 20, num_candidates: int = 2) -> Optional[Dict]:
    """Execute long-horizon task"""
    print("🚀 Starting Long-Horizon Task Execution...")

    try:
        executor = LongHorizonExecutor(
            directories["subtasks"],
            num_inference_steps=num_inference_steps,
            num_candidates=num_candidates
        )

        # Extract subtask list
        subtasks = [subtask["original_subtask"] for subtask in enhanced_task_data["subtasks"]]

        result = executor.execute_subtasks(image_path, subtasks, seed, enhanced_task_data)

        if result["success"]:
            print("✅ All subtasks completed successfully!")
            print(f"📹 Generated {len(result['video_paths'])} videos")

            # Get task summary
            summary = executor.get_task_summary()
            print(f"📊 Task Summary:")
            print(f"   Total tasks: {summary['total_tasks']}")
            print(f"   Base temp dir: {summary['base_temp_dir']}")

            return result
        else:
            print(f"❌ Long-horizon execution failed: {result.get('error', 'Unknown error')}")
            return None

    except Exception as e:
        print(f"❌ Error in long-horizon execution: {str(e)}")
        return None


def visualize_trajectories_for_subtasks(execution_result: Dict, directories: Dict[str, str]) -> None:
    """
    Generate visualization images containing object trajectory and robot trajectory (approach + exit) for each subtask.
    Visualization results are saved to the visualization subdirectory of the main output directory, filenames consistent with existing program: task{task_id}_trajectory.png.
    """
    try:
        if not execution_result:
            return

        task_results = execution_result.get("task_results", [])
        if not task_results:
            return

        visualization_dir = directories.get("visualization")
        if not visualization_dir:
            return

        os.makedirs(visualization_dir, exist_ok=True)

        for task_result in task_results:
            json_data_path = task_result.get("json_data_path")
            task_id = task_result.get("task_id")

            if not json_data_path or not os.path.exists(json_data_path):
                continue

            try:
                with open(json_data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                continue

            file_info = data.get("file_info", {})
            metadata = data.get("metadata", {})

            image_path = file_info.get("image_path") or task_result.get("input_image")
            if not image_path or not os.path.exists(image_path):
                continue

            # Trajectory data (pixel coordinates)
            obj_points = file_info.get("object_trajectory_points")
            robot_approach = file_info.get("robot_approach_trajectory")
            robot_exit = file_info.get("robot_exit_trajectory")

            # Skip task if key trajectories do not exist
            if obj_points is None or robot_approach is None or robot_exit is None:
                # Attempt to recover from full trajectory and transit_frames
                robot_full = file_info.get("robot_trajectory_full")
                object_full = file_info.get("object_trajectory_full")
                transit_frames = metadata.get("transit_frames", [8, 27, 36])
                if robot_full is None or object_full is None or len(transit_frames) < 3:
                    continue

                try:
                    # Three-stage: Approach [0, transit_start], Object motion [transit_start, transit_end], Exit [transit_end, end]
                    transit_start, transit_end, _ = transit_frames
                    robot_approach = robot_full[0:transit_start + 1]
                    robot_exit = robot_full[transit_end:]
                    obj_points = object_full[transit_start:transit_end + 1]
                except Exception:
                    continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                continue

            obj_arr = np.asarray(obj_points, dtype=np.float32)
            robot_approach_arr = np.asarray(robot_approach, dtype=np.float32)
            robot_exit_arr = np.asarray(robot_exit, dtype=np.float32)

            if obj_arr.ndim != 2 or obj_arr.shape[1] != 2:
                continue

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(image)

            # Plot object trajectory
            ax.plot(
                obj_arr[:, 0],
                obj_arr[:, 1],
                color="orange",
                linewidth=2.5,
                label="Object trajectory"
            )

            # Plot robot approach and exit trajectories (two segments)
            if robot_approach_arr.size > 0:
                ax.plot(
                    robot_approach_arr[:, 0],
                    robot_approach_arr[:, 1],
                    color="cyan",
                    linewidth=2.0,
                    label="Robot trajectory (approach)"
                )

            if robot_exit_arr.size > 0:
                ax.plot(
                    robot_exit_arr[:, 0],
                    robot_exit_arr[:, 1],
                    color="magenta",
                    linewidth=2.0,
                    label="Robot trajectory (exit)"
                )

            # Mark object start and end points
            start_pt = obj_arr[0]
            end_pt = obj_arr[-1]
            ax.scatter(
                [start_pt[0]],
                [start_pt[1]],
                c="green",
                s=80,
                marker="o",
                edgecolors="white",
                linewidths=1.5,
                label="Start"
            )
            ax.scatter(
                [end_pt[0]],
                [end_pt[1]],
                c="red",
                s=120,
                marker="*",
                edgecolors="white",
                linewidths=1.5,
                label="End"
            )

            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            if task_id is not None:
                ax.set_title(f"Task {task_id} Trajectories (Object + Robot)")
            else:
                ax.set_title("Trajectories (Object + Robot)")

            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

            # Save to the same visualization directory as existing program, keeping filename task{task_id}_trajectory.png
            if task_id is None:
                out_name = "task_trajectory.png"
            else:
                out_name = f"task{task_id}_trajectory.png"
            out_path = os.path.join(visualization_dir, out_name)

            fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)

            print(f"🖼️  Trajectory visualization saved with robot paths: {out_path}")

    except Exception as e:
        print(f"⚠️  Warning: Failed to generate trajectory visualizations: {str(e)}")


def compose_final_video(video_paths: List[str], output_path: str, transition_frames: int = 5,
                       use_transitions: bool = True) -> Optional[str]:
    """Compose final video"""
    print("🎬 Composing final long-horizon video...")

    try:
        composer = VideoComposer()

        # Determine whether to use transition effects
        actual_transition_frames = transition_frames if use_transitions else 0
        if not use_transitions:
            print("ℹ️  Transitions disabled, using direct concatenation")

        final_video_path = composer.compose_videos(
            video_paths=video_paths,
            output_path=output_path,
            transition_frames=actual_transition_frames,
            use_ffmpeg=True
        )

        if final_video_path:
            print(f"✅ Final video composed successfully!")
            print(f"📹 Output: {final_video_path}")

            # Get video info
            video_info = composer.get_video_info(final_video_path)
            if video_info:
                print(f"📊 Final Video Info:")
                print(f"   Duration: {video_info.get('duration', 0):.2f} seconds")
                print(f"   Resolution: {video_info.get('width', 0)}x{video_info.get('height', 0)}")
                print(f"   FPS: {video_info.get('fps', 0):.2f}")
                print(f"   File size: {video_info.get('size', 0) / (1024*1024):.2f} MB")

            return final_video_path
        else:
            print("❌ Video composition failed")
            return None

    except Exception as e:
        print(f"❌ Error in video composition: {str(e)}")
        return None


def save_execution_summary(directories: Dict[str, str], execution_result: Dict,
                          final_video_path: str, args):
    """Save execution summary"""
    print("💾 Saving execution summary...")

    try:
        summary = {
            "execution_info": {
                "timestamp": str(os.times()),
                "input_image": args.image,
                "complex_instruction": args.instruction,
                "total_subtasks": len(execution_result["task_results"]),
                "final_video": final_video_path,
                "output_directory": directories["output"],
                "temp_directory": directories["temp"],
                "seed": args.seed
            },
            "task_details": [],
            "system_info": {
                "model_path": args.model_path,
                "transformer_path": args.transformer_path,
                "transition_frames": args.transition_frames,
                "transitions_enabled": not args.no_transitions,
                "random_seed": args.seed
            }
        }

        # Commented out metrics functionality (User requested removal of time and VRAM stats)
        # if stats is not None:
        #     summary["metrics"] = stats

        # Add detailed info for each task
        for task_result in execution_result["task_results"]:
            task_info = {
                "task_id": task_result["task_id"],
                "task_name": task_result["task_name"],
                "subtask": task_result["subtask"],
                "video_path": task_result["video_path"],
                "input_image": task_result["input_image"],
                "prepared_files": list(task_result["prepared_files"].keys())
            }
            summary["task_details"].append(task_info)

        # Save summary file
        summary_path = os.path.join(directories["logs"], "execution_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"✅ Execution summary saved: {summary_path}")

    except Exception as e:
        print(f"⚠️  Warning: Failed to save execution summary: {str(e)}")


def cleanup_temp_files(temp_dir: str, keep_temp: bool = False):
    """Clean up temporary files - Cleanup based on image name"""
    if not keep_temp and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"🧹 Temporary files cleaned up: {temp_dir}")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean up temp files: {str(e)}")
    elif keep_temp:
        print(f"💾 Temporary files kept for debugging: {temp_dir}")

        # If keeping temp files, show detailed directory structure
        if os.path.exists(temp_dir):
            print(f"📁 Temporary files structure:")
            try:
                for root, dirs, files in os.walk(temp_dir):
                    level = root.replace(temp_dir, '').count(os.sep)
                    indent = ' ' * 2 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path)
                        print(f"{subindent}📄 {file} ({file_size:,} bytes)")
            except Exception as e:
                print(f"   Could not list directory contents: {str(e)}")


def validate_requirements(args):
    """Validate runtime requirements"""
    print("🔍 Validating requirements...")

    # Check input image
    if not os.path.exists(args.image):
        print(f"❌ Error: Input image not found: {args.image}")
        return False

    # Check model path (basic validation)
    if not os.path.exists(args.model_path):
        print(f"⚠️  Warning: Model path not found: {args.model_path}")

    if not os.path.exists(args.transformer_path):
        print(f"⚠️  Warning: Transformer path not found: {args.transformer_path}")

    # Check required Python modules
    try:
        from vlm_api.task_decomposer import TaskDecomposer
        from vlm_api.long_horizon_executor import LongHorizonExecutor
        from vlm_api.video_composer import VideoComposer
        print("✅ All required modules available")
    except ImportError as e:
        print(f"❌ Error: Required module not available: {str(e)}")
        return False

    # Check if Affordance inference script exists (for localization and segmentation)
    aff_infer_py = "/data/rczhang/MIND-V/Affordance-R1/inference_scripts/infer.py"
    if not os.path.exists(aff_infer_py):
        print(f"⚠️  Warning: Affordance infer script not found: {aff_infer_py}")

    print("✅ Requirements validation passed")
    return True


def main():
    """Main function - Consistent structure with complete_video_pipeline.py"""
    print("🚀 Starting Long-Horizon Robotic Arm Manipulation Video Generation Pipeline")
    print("=" * 80)

    # Parse command line arguments
    args = parse_arguments()

    # Set random seed to ensure reproducibility
    set_random_seed(args.seed)

    # Validate runtime requirements
    if not validate_requirements(args):
        return 1

    # Setup directory structure
    directories = setup_directories(args.output, args.temp_dir, args.image, args.seed)
    print(f"📁 Output directory: {directories['output']}")
    print(f"📁 Temporary directory: {directories['temp']}")
    print(f"📁 Subtasks directory: {directories['subtasks']}")
    print(f"📁 Final videos directory: {directories['final_videos']}")

    # Step 1: Task Decomposition (Planning Stage)
    print("\n" + "=" * 80)
    print("STEP 1: ENHANCED TASK DECOMPOSITION")
    print("=" * 80)

    enhanced_task_data = decompose_complex_task(args.instruction, args.image)
    if not enhanced_task_data or enhanced_task_data["total_subtasks"] == 0:
        print("❌ Pipeline failed at task decomposition stage")
        cleanup_temp_files(directories["temp"], args.keep_temp)
        return 1

    # Step 2: Long-Horizon Task Execution (Note: By definition, everything until obtaining video generation input files belongs to planning)
    print("\n" + "=" * 80)
    print("STEP 2: LONG-HORIZON TASK EXECUTION")
    print("=" * 80)

    execution_result = execute_long_horizon_task(
        args.image,
        enhanced_task_data,
        directories,
        args.seed,
        num_inference_steps=args.num_inference_steps,
        num_candidates=args.num_candidates
    )
    # Commented out stats collection (User requested removal of time and VRAM stats)
    # exec_metrics = (execution_result or {}).get("metrics", {})
    # stats_collector.add_to_stage(
    #     "planning",
    #     add_time_sec=float(exec_metrics.get("planning_time_sec", 0.0)),
    #     new_peak_bytes=int(exec_metrics.get("planning_gpu_peak_bytes", 0)),
    # )
    # stats_collector.add_to_stage(
    #     "generation",
    #     add_time_sec=float(exec_metrics.get("generation_time_sec", 0.0)),
    #     new_peak_bytes=int(exec_metrics.get("generation_gpu_peak_bytes", 0)),
    # )

    if not execution_result:
        print("❌ Pipeline failed at task execution stage")
        cleanup_temp_files(directories["temp"], args.keep_temp)
        return 1

    # Generate visualization containing robot trajectory for each subtask (consistent with existing visualization paths)
    visualize_trajectories_for_subtasks(execution_result, directories)

    # Step 3: Video Composition (Tracked separately for complete recording)
    print("\n" + "=" * 80)
    print("STEP 3: FINAL VIDEO COMPOSITION")
    print("=" * 80)

    final_video_filename = f"long_horizon_final_seed{args.seed}.mp4"
    final_video_path = os.path.join(directories["final_videos"], final_video_filename)

    use_transitions = not args.no_transitions
    composed_video = compose_final_video(
        video_paths=execution_result["video_paths"],
        output_path=final_video_path,
        transition_frames=args.transition_frames,
        use_transitions=use_transitions
    )

    if not composed_video:
        print("❌ Pipeline failed at video composition stage")
        cleanup_temp_files(directories["temp"], args.keep_temp)
        return 1

    # Commented out stats functionality (User requested removal of time and VRAM stats)
    # stats = stats_collector.finalize(subtasks_count=enhanced_task_data.get("total_subtasks", 0))

    # Step 4: Save Execution Summary
    print("\n" + "=" * 80)
    print("STEP 4: SAVING EXECUTION SUMMARY")
    print("=" * 80)

    # Commented out stats functionality (User requested removal of time and VRAM stats)
    save_execution_summary(directories, execution_result, composed_video, args)
    # PipelineStats.save_stats(stats, directories["logs"])

    # # Print required stats to terminal (Units: s, Gb)
    # print("\n" + "=" * 80)
    # print("METRICS SUMMARY")
    # print("=" * 80)

    # def _to_gb(bytes_val: int) -> float:
    #     try:
    #         return float(bytes_val) / (1024 ** 3)
    #     except Exception:
    #         return 0.0

    # plan = stats.get("stages", {}).get("planning", {})
    # gen = stats.get("stages", {}).get("generation", {})

    # plan_t = float(plan.get("time_sec", 0.0))
    # gen_t = float(gen.get("time_sec", 0.0))
    # two_stage_time = plan_t + gen_t
    # plan_time_ratio = (plan_t / two_stage_time) if two_stage_time > 0 else 0.0
    # gen_time_ratio = (gen_t / two_stage_time) if two_stage_time > 0 else 0.0

    # plan_peak = int(plan.get("gpu_peak_alloc_bytes", 0))
    # gen_peak = int(gen.get("gpu_peak_alloc_bytes", 0))
    # mem_sum = plan_peak + gen_peak
    # plan_mem_ratio = (plan_peak / mem_sum) if mem_sum > 0 else 0.0
    # gen_mem_ratio = (gen_peak / mem_sum) if mem_sum > 0 else 0.0

    # subtasks = int(stats.get("subtasks_count", 0))
    # avg_time_per_subtask = (gen_t / subtasks) if subtasks > 0 else 0.0

    # print(f"1) 规划阶段: 时间占比 {plan_time_ratio*100:.1f}%, 显存占比 {plan_mem_ratio*100:.1f}% (峰值 {_to_gb(plan_peak):.2f} Gb)")
    # print(f"2) 生成阶段: 时间占比 {gen_time_ratio*100:.1f}%, 显存占比 {gen_mem_ratio*100:.1f}% (峰值 {_to_gb(gen_peak):.2f} Gb)")
    # print(f"子任务数量: {subtasks}")
    # print(f"生成总时间: {gen_t:.2f} s; 平均每子任务: {avg_time_per_subtask:.2f} s")
    # print(f"生成阶段显存峰值: {_to_gb(gen_peak):.2f} Gb")
    # print(f"关系: {subtasks} 子任务 -> {gen_t:.2f} s, {_to_gb(gen_peak):.2f} Gb")

    # Complete
    print("\n" + "=" * 80)
    print("🎉 LONG-HORIZON PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"📹 Final video: {composed_video}")
    print(f"🖼️  Input image: {args.image}")
    print(f"💬 Complex instruction: {args.instruction}")
    print(f"📋 Subtasks executed: {enhanced_task_data['total_subtasks']}")
    print(f"🤖 Enhanced by AI: {enhanced_task_data.get('enhanced_by_ai', False)}")
    print(f"🎬 Individual videos: {len(execution_result['video_paths'])}")
    print(f"📁 Output directory: {directories['output']}")

    # Display subtask details
    print(f"\n📝 Subtask Details:")
    for i, task_result in enumerate(execution_result["task_results"], 1):
        subtask_data = enhanced_task_data["subtasks"][i-1] if i-1 < len(enhanced_task_data["subtasks"]) else {}
        # Get subtask text, prioritizing structured format
        if "original_text" in task_result:
            subtask_text = task_result["original_text"]
        elif "structured_subtask" in subtask_data and "original_text" in subtask_data["structured_subtask"]:
            subtask_text = subtask_data["structured_subtask"]["original_text"]
        elif "original_subtask" in subtask_data:
            subtask_text = subtask_data["original_subtask"]
        else:
            subtask_text = f"Task {i}"

        print(f"   Task {i}: {subtask_text}")
        # Display enhanced prompt info (if any)
        if "positive_prompt" in subtask_data:
            print(f"           Enhanced: {subtask_data['positive_prompt'][:80]}...")
        elif "enhanced_prompts" in subtask_data and "positive_prompt" in subtask_data["enhanced_prompts"]:
            print(f"           Enhanced: {subtask_data['enhanced_prompts']['positive_prompt'][:80]}...")
        print(f"           Video: {task_result['video_path']}")

    # Display generated files
    print(f"\n📁 Generated Files:")
    print(f"   📹 Final Video: {composed_video}")

    for i, task_result in enumerate(execution_result["task_results"], 1):
        print(f"   📹 Task {i} Video: {task_result['video_path']}")

    print(f"   📊 Execution Summary: {os.path.join(directories['logs'], 'execution_summary.json')}")

    # Clean up temporary files
    cleanup_temp_files(directories["temp"], args.keep_temp)

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)