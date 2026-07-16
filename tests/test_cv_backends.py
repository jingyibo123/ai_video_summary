"""
CV 后端对比测试脚本 (CV Backends Benchmark)
包含三种不同的视频关键帧提取实现方案，用于在不同机器上测试兼容性和性能。

用法:
  python test_cv_backends.py [视频路径] [后端: pure_python|ffmpeg_interval|ffmpeg_smart]
"""

import sys
import time
import json
import shutil
import cv2
import subprocess
import numpy as np
from pathlib import Path
import re
import threading
from typing import List, Optional

def extract_single_frame(v_path: str, t_sec: float, o_path: Path):
    """通用的高清截帧辅助函数"""
    cmd_extract = [
        "ffmpeg", "-y", "-ss", f"{t_sec:.3f}", "-i", str(v_path),
        "-vframes", "1", "-q:v", "2", str(o_path)
    ]
    subprocess.run(cmd_extract, capture_output=True, check=True)


def extract_pure_python(video_path: str, output_dir: str, max_seconds: Optional[int] = None, 
                        target_size: tuple = (256, 144), diff_threshold: int = 850, 
                        sample_interval: float = 1.0) -> List[dict]:
    """
    方案一：纯 Python/OpenCV 实现。
    优点：兼容性最好，不依赖 FFmpeg 复杂的 select 过滤器，不涉及管道通信。
    缺点：Python 层面的视频解码和跳转（seek）相对较慢。
    """
    print(">>> 启动纯 Python (OpenCV) 后端")
    output_dir_path = Path(output_dir)
    cands_dir = output_dir_path / "candidates_pure_python"
    cands_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件 {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_sec = total_frames / fps if fps > 0 else 0
    
    if max_seconds and total_sec > max_seconds:
        total_sec = max_seconds

    t_start = time.time()
    
    results = []
    last_gray = None
    last_time_sec = 0.0
    current_slide_start_sec = 0.0
    slide_intervals = []
    
    current_sec = 0.0
    processed_count = 0
    
    while current_sec <= total_sec:
        # OpenCV 跳转到指定时间
        cap.set(cv2.CAP_PROP_POS_MSEC, current_sec * 1000.0)
        ret, frame = cap.read()
        if not ret:
            break
            
        # 转灰度并缩放
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, target_size)
        
        if last_gray is None:
            last_gray = gray
            last_time_sec = current_sec
            current_slide_start_sec = current_sec
        else:
            mse = np.sum((last_gray.astype("float") - gray.astype("float")) ** 2) / float(gray.size)
            if mse > diff_threshold:
                slide_intervals.append((current_slide_start_sec, last_time_sec))
                current_slide_start_sec = current_sec
            last_gray = gray
            last_time_sec = current_sec
            
        processed_count += 1
        current_sec += sample_interval
        
    cap.release()
    
    if processed_count > 0:
        slide_intervals.append((current_slide_start_sec, last_time_sec))
        
    for start_sec, end_sec in slide_intervals:
        t_str = lambda t: f"{int(t)//3600:02d}-{int(t)%3600//60:02d}-{int(t)%60:02d}"
        fname = f"{t_str(start_sec)}_{t_str(end_sec)}.jpg"
        out_path = cands_dir / fname
        try:
            extract_single_frame(video_path, end_sec, out_path)
        except Exception:
            extract_single_frame(video_path, start_sec, out_path)
        results.append({"start_time": start_sec, "end_time": end_sec, "image": str(out_path.relative_to(output_dir_path))})
        
    elapsed = time.time() - t_start
    print(f"完成! 耗时: {elapsed:.2f}s, 采样帧数: {processed_count}, 候选帧数: {len(results)}")
    return results


def extract_ffmpeg_interval(video_path: str, output_dir: str, max_seconds: Optional[int] = None, 
                            target_size: tuple = (256, 144), diff_threshold: int = 850, 
                            sample_interval: float = 1.0) -> List[dict]:
    """
    方案二：FFmpeg 纯定时提取。
    优点：兼容所有版本的 FFmpeg，不使用复杂的 select 条件，直接通过 fps 过滤器强制输出固定间隔的帧。
    缺点：无法智能捕获翻页瞬间的 I 帧，可能错过短促的动画。
    """
    print(">>> 启动 FFmpeg 定时提取 (fps 过滤器) 后端")
    output_dir_path = Path(output_dir)
    cands_dir = output_dir_path / "candidates_ffmpeg_interval"
    cands_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_sec = total_frames / fps_video if fps_video > 0 else 0
    cap.release()
    
    if max_seconds and total_sec > max_seconds:
        total_sec = max_seconds

    t_start = time.time()
    w, h = target_size
    
    # 使用简单的 fps 过滤器
    cmd = ["ffmpeg"]
    if max_seconds:
        cmd.extend(["-t", str(max_seconds)])
    cmd.extend([
        "-i", str(video_path),
        "-vf", f"fps=1/{sample_interval},scale={w}:{h}",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "pipe:1"
    ])
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**6)
    
    frames = []
    frame_bytes = w * h
    while True:
        data = proc.stdout.read(frame_bytes)
        if len(data) < frame_bytes:
            break
        frames.append(data)
        
    proc.wait()
    n_frames = len(frames)
    
    results = []
    last_gray = None
    last_time_sec = 0.0
    current_slide_start_sec = 0.0
    slide_intervals = []
    
    for idx in range(n_frames):
        gray_bytes = frames[idx]
        sec = idx * sample_interval # 因为是固定 fps，时间可以直接算
        gray = np.frombuffer(gray_bytes, dtype=np.uint8).reshape((h, w))
        
        if last_gray is None:
            last_gray = gray
            last_time_sec = sec
            current_slide_start_sec = sec
        else:
            mse = np.sum((last_gray.astype("float") - gray.astype("float")) ** 2) / float(gray.size)
            if mse > diff_threshold:
                slide_intervals.append((current_slide_start_sec, last_time_sec))
                current_slide_start_sec = sec
            last_gray = gray
            last_time_sec = sec
            
    if n_frames > 0:
        slide_intervals.append((current_slide_start_sec, last_time_sec))
        
    for start_sec, end_sec in slide_intervals:
        t_str = lambda t: f"{int(t)//3600:02d}-{int(t)%3600//60:02d}-{int(t)%60:02d}"
        fname = f"{t_str(start_sec)}_{t_str(end_sec)}.jpg"
        out_path = cands_dir / fname
        try:
            extract_single_frame(video_path, end_sec, out_path)
        except Exception:
            extract_single_frame(video_path, start_sec, out_path)
        results.append({"start_time": start_sec, "end_time": end_sec, "image": str(out_path.relative_to(output_dir_path))})
        
    elapsed = time.time() - t_start
    print(f"完成! 耗时: {elapsed:.2f}s, 采样帧数: {n_frames}, 候选帧数: {len(results)}")
    return results


def extract_ffmpeg_smart(video_path: str, output_dir: str, max_seconds: Optional[int] = None, 
                         target_size: tuple = (256, 144), diff_threshold: int = 850, 
                         sample_interval: float = 1.0) -> List[dict]:
    """
    方案三：FFmpeg 定时 + 关键帧提取（当前线上方案）。
    优点：速度极快，智能抓取 I 帧。
    缺点：依赖特定的 select 过滤器语法，部分老旧 FFmpeg 或不同系统下转义可能出问题。
    """
    print(">>> 启动 FFmpeg 智能提取 (select 过滤器) 后端")
    output_dir_path = Path(output_dir)
    cands_dir = output_dir_path / "candidates_ffmpeg_smart"
    cands_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_sec = total_frames / fps_video if fps_video > 0 else 0
    cap.release()
    
    if max_seconds and total_sec > max_seconds:
        total_sec = max_seconds

    t_start = time.time()
    w, h = target_size
    
    min_gap = sample_interval / 2.0
    select_expr = f"select='isnan(prev_selected_t)+eq(pict_type\\,I)*gte(t-prev_selected_t\\,{min_gap})+gte(t-prev_selected_t\\,{sample_interval})'"
    
    cmd = ["ffmpeg"]
    if max_seconds:
        cmd.extend(["-t", str(max_seconds)])
    cmd.extend([
        "-i", str(video_path),
        "-fps_mode", "passthrough",
        "-vf", f"{select_expr},scale={w}:{h},showinfo",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "pipe:1"
    ])
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**6)
    
    frames = []
    timestamps = []
    
    def read_stderr():
        pts_regex = re.compile(r"pts_time:([\d\.]+)")
        for line in proc.stderr:
            line_str = line.decode("utf-8", errors="ignore")
            if "pts_time:" in line_str:
                match = pts_regex.search(line_str)
                if match:
                    timestamps.append(float(match.group(1)))
                    
    stderr_thread = threading.Thread(target=read_stderr)
    stderr_thread.start()
    
    frame_bytes = w * h
    while True:
        data = proc.stdout.read(frame_bytes)
        if len(data) < frame_bytes:
            break
        frames.append(data)
        
    proc.stdout.close()
    proc.wait()
    stderr_thread.join()
    
    n_frames = min(len(frames), len(timestamps))
    frames = frames[:n_frames]
    timestamps = timestamps[:n_frames]
    
    results = []
    last_gray = None
    last_time_sec = 0.0
    current_slide_start_sec = 0.0
    slide_intervals = []
    
    for idx in range(n_frames):
        gray_bytes = frames[idx]
        sec = timestamps[idx]
        gray = np.frombuffer(gray_bytes, dtype=np.uint8).reshape((h, w))
        
        if last_gray is None:
            last_gray = gray
            last_time_sec = sec
            current_slide_start_sec = sec
        else:
            mse = np.sum((last_gray.astype("float") - gray.astype("float")) ** 2) / float(gray.size)
            if mse > diff_threshold:
                slide_intervals.append((current_slide_start_sec, last_time_sec))
                current_slide_start_sec = sec
            last_gray = gray
            last_time_sec = sec
            
    if n_frames > 0:
        slide_intervals.append((current_slide_start_sec, last_time_sec))
        
    for start_sec, end_sec in slide_intervals:
        t_str = lambda t: f"{int(t)//3600:02d}-{int(t)%3600//60:02d}-{int(t)%60:02d}"
        fname = f"{t_str(start_sec)}_{t_str(end_sec)}.jpg"
        out_path = cands_dir / fname
        try:
            extract_single_frame(video_path, end_sec, out_path)
        except Exception:
            extract_single_frame(video_path, start_sec, out_path)
        results.append({"start_time": start_sec, "end_time": end_sec, "image": str(out_path.relative_to(output_dir_path))})
        
    elapsed = time.time() - t_start
    print(f"完成! 耗时: {elapsed:.2f}s, 采样帧数: {n_frames}, 候选帧数: {len(results)}")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test_cv_backends.py [视频路径] [后端: pure_python|ffmpeg_interval|ffmpeg_smart|all]")
        sys.exit(1)
        
    video_path = sys.argv[1]
    backend = sys.argv[2] if len(sys.argv) > 2 else "all"
    
    # 明确将输出目录设置在当前 py 文件同级的 cv_benchmark_output 文件夹内
    output_dir = str(Path(__file__).parent / "cv_benchmark_output")
    max_sec = 60  # 为加快测试，只处理前60秒
    
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
        
    print(f"测试视频: {video_path}")
    print(f"测试时长: {max_sec}秒")
    print("-" * 50)
    
    funcs = []
    if backend in ("pure_python", "all"): funcs.append(("Pure Python", extract_pure_python))
    if backend in ("ffmpeg_interval", "all"): funcs.append(("FFmpeg Interval", extract_ffmpeg_interval))
    if backend in ("ffmpeg_smart", "all"): funcs.append(("FFmpeg Smart (Current)", extract_ffmpeg_smart))
    
    all_results = {}
    for name, func in funcs:
        try:
            res = func(video_path, output_dir, max_seconds=max_sec)
            all_results[name] = len(res)
            
            # 把结果存下来到本地供检查
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
            res_file = Path(output_dir) / f"{safe_name}_results.json"
            with open(res_file, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            print(f"[{name}] 结果已保存至: {res_file}")
            
        except Exception as e:
            print(f"{name} 失败: {e}")
            all_results[name] = "Failed"
        print("-" * 50)
        
    print(">>> 测试总结 <<<")
    for name, count in all_results.items():
        print(f"{name}: {count} 帧候选")
