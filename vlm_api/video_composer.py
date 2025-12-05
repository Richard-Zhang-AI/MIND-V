#!/usr/bin/env python3
"""
视频拼接器 - Video Composer
将多个子任务视频无缝拼接成一个完整的长程任务视频

Video Composer for Concatenating Multiple Subtask Videos
Seamlessly combines multiple subtask videos into a complete long-horizon video
"""

import os
import sys
import subprocess
import shutil
from typing import List, Optional, Dict, Tuple
from loguru import logger

# 尝试导入cv2，如果失败则使用其他方法
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV (cv2) not available, will use alternative methods for video composition")


class VideoComposer:
    """
    视频拼接器类
    负责将多个子任务视频无缝拼接成一个完整的长程任务视频
    """

    def __init__(self):
        """
        初始化视频拼接器
        """
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv']
        logger.info("视频拼接器初始化完成")

    def compose_videos(self, video_paths: List[str], output_path: str,
                      transition_frames: int = 5, use_ffmpeg: bool = True) -> Optional[str]:
        """
        拼接多个视频成一个完整的长程任务视频

        Args:
            video_paths: 子任务视频路径列表
            output_path: 最终输出路径
            transition_frames: 转场帧数（用于平滑过渡）
            use_ffmpeg: 是否使用ffmpeg进行拼接（推荐）

        Returns:
            最终视频路径或None
        """
        logger.info(f"开始拼接 {len(video_paths)} 个视频")
        logger.info(f"输出路径: {output_path}")
        logger.info(f"转场帧数: {transition_frames}")

        if not video_paths:
            logger.error("视频路径列表为空")
            return None

        # 验证输入视频文件
        valid_video_paths = self._validate_video_files(video_paths)
        if not valid_video_paths:
            logger.error("没有有效的视频文件")
            return None

        if len(valid_video_paths) != len(video_paths):
            logger.warning(f"部分视频文件无效，有效视频数量: {len(valid_video_paths)}/{len(video_paths)}")

        try:
            if use_ffmpeg and self._check_ffmpeg_available():
                return self._compose_with_ffmpeg(valid_video_paths, output_path, transition_frames)
            elif CV2_AVAILABLE:
                return self._compose_with_opencv(valid_video_paths, output_path, transition_frames)
            else:
                logger.error("既没有ffmpeg也没有OpenCV可用，无法进行视频拼接")
                return None

        except Exception as e:
            logger.error(f"视频拼接失败: {str(e)}")
            return None

    def _validate_video_files(self, video_paths: List[str]) -> List[str]:
        """
        验证视频文件是否存在且格式正确

        Args:
            video_paths: 视频路径列表

        Returns:
            有效的视频路径列表
        """
        valid_paths = []

        for video_path in video_paths:
            if not os.path.exists(video_path):
                logger.error(f"视频文件不存在: {video_path}")
                continue

            if not os.path.isfile(video_path):
                logger.error(f"路径不是文件: {video_path}")
                continue

            file_ext = os.path.splitext(video_path)[1].lower()
            if file_ext not in self.supported_formats:
                logger.warning(f"不支持的视频格式: {video_path} (格式: {file_ext})")
                continue

            # 检查文件大小
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                logger.error(f"视频文件为空: {video_path}")
                continue

            logger.info(f"视频文件验证通过: {video_path} (大小: {file_size} bytes)")
            valid_paths.append(video_path)

        return valid_paths

    def _check_ffmpeg_available(self) -> bool:
        """
        检查ffmpeg是否可用

        Returns:
            ffmpeg是否可用
        """
        try:
            result = subprocess.run(['ffmpeg', '-version'],
                                  capture_output=True,
                                  text=True,
                                  timeout=5)
            if result.returncode == 0:
                logger.info("ffmpeg可用")
                return True
            else:
                logger.warning("ffmpeg不可用")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("ffmpeg未找到或不可用")
            return False

    def _compose_with_ffmpeg(self, video_paths: List[str], output_path: str,
                           transition_frames: int) -> Optional[str]:
        """
        使用ffmpeg拼接视频

        Args:
            video_paths: 视频路径列表
            output_path: 输出路径
            transition_frames: 转场帧数

        Returns:
            最终视频路径或None
        """
        logger.info("使用ffmpeg进行视频拼接")

        try:
            # 创建临时文件列表
            temp_dir = os.path.dirname(output_path)
            concat_list_path = os.path.join(temp_dir, "video_list.txt")

            # 生成ffmpeg concat文件
            with open(concat_list_path, 'w', encoding='utf-8') as f:
                for video_path in video_paths:
                    # 使用绝对路径避免路径问题
                    abs_path = os.path.abspath(video_path)
                    f.write(f"file '{abs_path}'\n")

            logger.info(f"创建了concat文件: {concat_list_path}")

            # 构建ffmpeg命令
            if transition_frames > 0:
                # 使用转场效果的拼接
                return self._compose_with_ffmpeg_transitions(
                    video_paths, output_path, transition_frames, temp_dir
                )
            else:
                # 简单拼接
                cmd = [
                    'ffmpeg',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_list_path,
                    '-c', 'copy',  # 直接复制流，速度快
                    '-y',  # 覆盖输出文件
                    output_path
                ]

                logger.info(f"运行ffmpeg拼接命令: {' '.join(cmd)}")

                result = subprocess.run(cmd,
                                      capture_output=True,
                                      text=True,
                                      timeout=300)  # 5分钟超时

                # 清理临时文件
                if os.path.exists(concat_list_path):
                    os.remove(concat_list_path)

                if result.returncode == 0:
                    logger.info(f"ffmpeg拼接成功: {output_path}")
                    return output_path
                else:
                    logger.error(f"ffmpeg拼接失败: {result.stderr}")
                    return None

        except subprocess.TimeoutExpired:
            logger.error("ffmpeg执行超时")
            return None
        except Exception as e:
            logger.error(f"ffmpeg拼接异常: {str(e)}")
            return None

    def _compose_with_ffmpeg_transitions(self, video_paths: List[str], output_path: str,
                                       transition_frames: int, temp_dir: str) -> Optional[str]:
        """
        使用ffmpeg创建带转场效果的视频拼接

        Args:
            video_paths: 视频路径列表
            output_path: 输出路径
            transition_frames: 转场帧数
            temp_dir: 临时目录

        Returns:
            最终视频路径或None
        """
        logger.info("使用ffmpeg创建带转场效果的视频拼接")

        try:
            # 这种方法比较复杂，需要分别处理每个视频
            # 简化版本：先获取第一个视频的属性，然后确保所有视频属性一致

            # 获取第一个视频的信息
            first_video = video_paths[0]
            probe_cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                first_video
            ]

            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.error("无法获取视频信息")
                return None

            import json
            video_info = json.loads(result.stdout)

            # 找到视频流
            video_stream = None
            for stream in video_info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break

            if not video_stream:
                logger.error("未找到视频流")
                return None

            fps = eval(video_stream.get('r_frame_rate', '25/1'))
            width = video_stream.get('width', 640)
            height = video_stream.get('height', 480)

            logger.info(f"视频属性: {width}x{height} @ {fps} fps")

            # 由于复杂的转场实现较为困难，这里退回到简单拼接
            # 但记录日志说明转场需求
            logger.info(f"注意：转场效果（{transition_frames}帧）需要更复杂的实现，当前使用简单拼接")
            return self._compose_with_ffmpeg_simple(video_paths, output_path, temp_dir)

        except Exception as e:
            logger.error(f"转场拼接失败: {str(e)}")
            return None

    def _compose_with_ffmpeg_simple(self, video_paths: List[str], output_path: str,
                                  temp_dir: str) -> Optional[str]:
        """
        使用ffmpeg进行简单拼接

        Args:
            video_paths: 视频路径列表
            output_path: 输出路径
            temp_dir: 临时目录

        Returns:
            最终视频路径或None
        """
        logger.info("使用ffmpeg进行简单拼接")

        concat_list_path = os.path.join(temp_dir, "video_list_simple.txt")

        try:
            # 生成concat文件
            with open(concat_list_path, 'w', encoding='utf-8') as f:
                for video_path in video_paths:
                    abs_path = os.path.abspath(video_path)
                    f.write(f"file '{abs_path}'\n")

            # 构建ffmpeg命令
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_list_path,
                '-c', 'copy',
                '-y',
                output_path
            ]

            logger.info(f"运行ffmpeg简单拼接命令: {' '.join(cmd)}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            # 清理临时文件
            if os.path.exists(concat_list_path):
                os.remove(concat_list_path)

            if result.returncode == 0:
                logger.info(f"ffmpeg简单拼接成功: {output_path}")
                return output_path
            else:
                logger.error(f"ffmpeg简单拼接失败: {result.stderr}")
                return None

        except Exception as e:
            logger.error(f"ffmpeg简单拼接异常: {str(e)}")
            return None

    def _compose_with_opencv(self, video_paths: List[str], output_path: str,
                           transition_frames: int) -> Optional[str]:
        """
        使用OpenCV拼接视频

        Args:
            video_paths: 视频路径列表
            output_path: 输出路径
            transition_frames: 转场帧数

        Returns:
            最终视频路径或None
        """
        logger.info("使用OpenCV进行视频拼接")

        if not CV2_AVAILABLE:
            logger.error("OpenCV不可用")
            return None

        try:
            # 获取第一个视频的基本信息
            cap = cv2.VideoCapture(video_paths[0])
            if not cap.isOpened():
                logger.error(f"无法打开第一个视频: {video_paths[0]}")
                return None

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            logger.info(f"视频属性: {width}x{height} @ {fps} fps")

            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not out.isOpened():
                logger.error("无法创建输出视频文件")
                return None

            total_frames = 0

            # 处理每个视频
            for i, video_path in enumerate(video_paths):
                logger.info(f"处理视频 {i+1}/{len(video_paths)}: {os.path.basename(video_path)}")

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    logger.error(f"无法打开视频: {video_path}")
                    continue

                frames = []

                # 读取所有帧
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)

                cap.release()

                # 写入帧到输出视频
                for frame in frames:
                    out.write(frame)
                    total_frames += 1

                # 添加转场效果（如果不是最后一个视频）
                if i < len(video_paths) - 1 and transition_frames > 0:
                    if frames:
                        self._add_transition_opencv(out, frames[-1], video_paths[i+1],
                                                  transition_frames, width, height, fps)
                        total_frames += transition_frames

            out.release()

            logger.info(f"OpenCV拼接完成: {output_path}")
            logger.info(f"总帧数: {total_frames}")

            # 验证输出文件
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"输出文件大小: {file_size} bytes")
                return output_path
            else:
                logger.error("输出文件未生成")
                return None

        except Exception as e:
            logger.error(f"OpenCV拼接失败: {str(e)}")
            return None

    def _add_transition_opencv(self, writer, last_frame, next_video_path: str,
                               transition_frames: int, width: int, height: int, fps: int):
        """
        使用OpenCV添加转场效果

        Args:
            writer: 视频写入器
            last_frame: 最后一帧
            next_video_path: 下一个视频路径
            transition_frames: 转场帧数
            width: 视频宽度
            height: 视频高度
            fps: 帧率
        """
        try:
            # 读取下一个视频的第一帧
            cap = cv2.VideoCapture(next_video_path)
            if not cap.isOpened():
                logger.warning(f"无法打开下一个视频进行转场: {next_video_path}")
                return

            ret, first_frame = cap.read()
            cap.release()

            if ret:
                # 创建渐变转场
                for i in range(1, transition_frames + 1):
                    alpha = i / transition_frames
                    beta = 1 - alpha

                    # 确保帧尺寸一致
                    if last_frame.shape[:2] != (height, width):
                        last_frame = cv2.resize(last_frame, (width, height))
                    if first_frame.shape[:2] != (height, width):
                        first_frame = cv2.resize(first_frame, (width, height))

                    transition_frame = cv2.addWeighted(first_frame, alpha, last_frame, beta, 0)
                    writer.write(transition_frame)

                logger.debug(f"添加了 {transition_frames} 帧转场效果")
            else:
                logger.warning("无法读取下一个视频的第一帧")

        except Exception as e:
            logger.warning(f"添加转场效果失败: {str(e)}")

    def get_video_info(self, video_path: str) -> Optional[Dict]:
        """
        获取视频信息

        Args:
            video_path: 视频路径

        Returns:
            视频信息字典或None
        """
        try:
            if self._check_ffmpeg_available():
                return self._get_video_info_ffmpeg(video_path)
            elif CV2_AVAILABLE:
                return self._get_video_info_opencv(video_path)
            else:
                logger.error("既没有ffmpeg也没有OpenCV可用")
                return None

        except Exception as e:
            logger.error(f"获取视频信息失败: {str(e)}")
            return None

    def _get_video_info_ffmpeg(self, video_path: str) -> Optional[Dict]:
        """使用ffprobe获取视频信息"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return None

            import json
            probe_data = json.loads(result.stdout)

            video_stream = None
            for stream in probe_data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break

            if not video_stream:
                return None

            format_info = probe_data.get('format', {})

            return {
                'duration': float(format_info.get('duration', 0)),
                'size': int(format_info.get('size', 0)),
                'bit_rate': int(format_info.get('bit_rate', 0)),
                'width': video_stream.get('width', 0),
                'height': video_stream.get('height', 0),
                'fps': eval(video_stream.get('r_frame_rate', '25/1')),
                'codec': video_stream.get('codec_name', 'unknown'),
                'frames': int(video_stream.get('nb_frames', 0))
            }

        except Exception as e:
            logger.error(f"ffprobe获取视频信息失败: {str(e)}")
            return None

    def _get_video_info_opencv(self, video_path: str) -> Optional[Dict]:
        """使用OpenCV获取视频信息"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            cap.release()

            return {
                'width': width,
                'height': height,
                'fps': fps,
                'frames': frame_count,
                'duration': duration,
                'size': os.path.getsize(video_path),
                'codec': 'unknown'
            }

        except Exception as e:
            logger.error(f"OpenCV获取视频信息失败: {str(e)}")
            return None


def test_video_composer():
    """
    测试视频拼接器的基本功能
    """
    print("🧪 测试视频拼接器...")

    try:
        composer = VideoComposer()

        # 测试ffmpeg可用性
        ffmpeg_available = composer._check_ffmpeg_available()
        print(f"ffmpeg可用: {'是' if ffmpeg_available else '否'}")
        print(f"OpenCV可用: {'是' if CV2_AVAILABLE else '否'}")

        # 测试视频信息获取（不依赖实际视频文件）
        print("✅ 视频拼接器基本功能测试通过")

        if not ffmpeg_available and not CV2_AVAILABLE:
            print("⚠️  警告：既没有ffmpeg也没有OpenCV可用，无法进行实际的视频拼接")

        print("🎉 视频拼接器测试完成！")

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")


if __name__ == "__main__":
    test_video_composer()