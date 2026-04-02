"""
AI 智能代理集 (Core AI Agents).

本模块为流水线的核心感知层，集成了三大核心能力：
1. 计算机视觉 (CV): 基于帧差 MSE 的快速跳帧算法，实现 170x+ 的视频处理速率。
2. 视觉大模型 (VLM): 统一的 `vlm_task` 接口，负责 OCR、幻灯片验证、语义去重与内容摘录。
3. 语音转录 (ASR): 灵活的音频处理引擎，支持本地 Faster-Whisper 与 OpenAI 兼容的远程/本地 API 切换。

设计哲学：极简接口，高内聚低耦合，所有识别任务均内置指数退避重试机制。
"""

import os
import re
import cv2
import time
import base64
import logging
import subprocess
import numpy as np
from typing import List, Optional
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# --- Pydantic Models for Structured Output ---

class VisualVocabulary(BaseModel):
    items: List[str] = Field(description="从幻灯片中发现的所有核心技术词汇、组件名或英文缩写。")

# --- 1. 计算机视觉 (CV) 代理 ---

def extract_key_frames(video_path: str, output_dir: str, 
                       max_seconds: Optional[int] = None, 
                       target_size: tuple = (256, 144), 
                       diff_threshold: int = 850) -> List[dict]:
    """
    使用 OpenCV 极速提取视频关键帧。
    
    Args:
        video_path: 视频源文件路径。
        output_dir: 候选帧保存目录。
        max_seconds: 最大处理时长（秒），None 则处理全片。
        target_size: 比较时的缩略图尺寸，建议保持小尺寸以提升速度。
        diff_threshold: 画面差异阈值，MSE 超过此值则认为发生翻页。
        
    Returns:
        List[dict]: 包含 'start_time', 'end_time', 'image' 的列表。
    """
    logger.info(f"CV: 开始分析视频流 {video_path}")
    cands_dir = os.path.join(output_dir, "candidates")
    os.makedirs(cands_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = max(1, int(fps))
    
    results = []
    frame_idx, current_slide_start_sec = 0.0, 0.0
    last_gray, last_full_frame, last_time_sec = None, None, 0.0
    
    t_start = time.time()
    success, frame = cap.read()
    
    while success:
        sec = frame_idx / fps
        if max_seconds and sec > max_seconds: break
            
        small = cv2.resize(frame, target_size, interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        if last_gray is None:
            last_gray, last_full_frame, last_time_sec = gray, frame.copy(), sec
        else:
            mse = np.sum((last_gray.astype("float") - gray.astype("float")) ** 2) / float(gray.size)
            if mse > diff_threshold:
                # 内联时间格式化
                t_str = lambda t: f"{int(t)//3600:02d}-{int(t)%3600//60:02d}-{int(t)%60:02d}"
                fname = f"{t_str(current_slide_start_sec)}_{t_str(last_time_sec)}.png"
                out_path = os.path.join(cands_dir, fname)
                cv2.imwrite(out_path, last_full_frame)
                results.append({"start_time": current_slide_start_sec, "end_time": last_time_sec, "image": out_path.replace("\\", "/")})
                current_slide_start_sec = sec
            last_gray, last_full_frame, last_time_sec = gray, frame.copy(), sec
            
        frame_idx += skip_frames
        for _ in range(skip_frames - 1):
            if not cap.grab(): success = False; break
        if not success: break
        success, frame = cap.read()
        
    if last_full_frame is not None:
        t_str = lambda t: f"{int(t)//3600:02d}-{int(t)%3600//60:02d}-{int(t)%60:02d}"
        fname = f"{t_str(current_slide_start_sec)}_{t_str(last_time_sec)}.png"
        out_path = os.path.join(cands_dir, fname)
        cv2.imwrite(out_path, last_full_frame)
        results.append({"start_time": current_slide_start_sec, "end_time": last_time_sec, "image": out_path.replace("\\", "/")})
        
    cap.release()
    elapsed = time.time() - t_start
    logger.info(f"CV: 处理完成，共切分 {len(results)} 个候选帧，速率 { (max_seconds or last_time_sec)/elapsed:.1f}x")
    return results

# --- 2. 视觉大模型 (VLM) 代理 ---

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.5, min=2, max=10))
def vlm_task(base_url: str, model: str, task_type: str, images: List[str]) -> any:
    """
    多功能 VLM 任务处理器，支持幻灯片校验、去重、摘要生成及热词 OCR。
    
    Args:
        base_url: OpenAI 兼容接口地址。
        model: VLM 模型名称。
        task_type: 任务类型 ('validate'|'dedup'|'caption'|'terms')。
        images: 涉及的图片本地路径列表。
        
    Returns:
        any: 校验/去重返回 bool，摘要返回 str，OCR 返回 List[str]。
    """
    client = OpenAI(api_key="none", base_url=base_url)
    prompts = {
        "validate": "Is this a presentation slide? Return TRUE or FALSE.",
        "dedup": "Are these two images the SAME slide? Return TRUE or FALSE.",
        "caption": "请用一句20字内的中文描述此幻灯片内容。",
        "terms": "Extract all technical terms from this slide."
    }
    content = [{"type": "text", "text": prompts.get(task_type, task_type)}]
    for img in images:
        with open(img, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            
    if task_type == "terms":
        try:
            resp = client.beta.chat.completions.parse(model=model, messages=[{"role": "user", "content": content}], response_format=VisualVocabulary)
            return resp.choices[0].message.parsed.items[:20] if resp.choices[0].message.parsed else []
        except: return []
        
    resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}], max_tokens=100)
    res = re.sub(r'<think>.*?</think>', '', resp.choices[0].message.content or "", flags=re.DOTALL).strip().upper()
    if task_type in ["validate", "dedup"]: return "FALSE" not in res
    return res

# --- 3. 语音转录 (ASR) 代理 ---

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.5, min=2, max=10))
def transcribe_with_whisper(audio_path: str, prompt: str, model_size: str = "base", api_base: Optional[str] = None, device: str = "cpu", compute_type: str = "int8") -> List[dict]:
    """
    核心语音转录引擎，根据 api_base 自动分发至本地 Faster-Whisper 或远程 API。
    
    Args:
        audio_path: 待处理音频路径（建议 16k mono）。
        prompt: ASR 引导语，支持热词注入。
        model_size: 本地模型尺寸或 API 指定模型名。
        api_base: 若提供，则使用 OpenAI 兼容 API 模式。
        device: 本地模型运行设备 (cpu/cuda)。
        compute_type: 本地模型计算精度 (int8/float16等)。
        
    Returns:
        List[dict]: 包含 'start', 'end', 'text', 'speaker' 的分段列表。
    """
    if api_base:
        client = OpenAI(api_key="none", base_url=api_base)
        with open(audio_path, "rb") as f:
            resp = client.audio.transcriptions.create(model=model_size, file=f, prompt=prompt, response_format="verbose_json")
        raw = getattr(resp, "segments", [])
        if not raw: return [{"start": 0.0, "end": 0.0, "text": getattr(resp, "text", ""), "speaker": "讲者"}]
        return [{"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2), "text": s["text"].strip(), "speaker": "讲者"} for s in raw]

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(audio_path, language="zh", initial_prompt=prompt, vad_filter=True)
    return [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(), "speaker": "讲者"} for s in segments]

def extract_audio(video_path: str, output_path: str, max_seconds: Optional[int] = None) -> None:
    """
    使用 FFmpeg 提取 16k 单声道 PCM 音频。
    
    Args:
        video_path: 源视频路径。
        output_path: 输出 Wav 路径。
        max_seconds: 提取的最大时长（秒）。
    """
    if os.path.exists(output_path): return
    cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", output_path]
    if max_seconds: cmd.insert(-1, "-t"); cmd.insert(-1, str(max_seconds))
    subprocess.run(cmd, capture_output=True)
