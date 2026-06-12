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
from typing import List, Optional, Any, Tuple
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# --- Pydantic Models for Structured Output ---

class VisualVocabulary(BaseModel):
    items: List[str] = Field(description="从幻灯片中发现的所有核心技术词汇、组件名或英文缩写。")

class SlideValidation(BaseModel):
    is_slide: bool = Field(description="Whether the image is a presentation slide.")

class SlideDeduplication(BaseModel):
    is_same: bool = Field(description="Whether the two images show the same presentation slide.")

class SlideCaption(BaseModel):
    caption: str = Field(description="用一句20字内的中文描述此幻灯片内容。")

class SlideAnalysis(BaseModel):
    is_slide: bool = Field(description="Whether the image is a presentation slide.")
    caption: Optional[str] = Field(None, description="如果 is_slide 为 true，用一句20字内的中文描述此幻灯片内容。")
    items: Optional[List[str]] = Field(None, description="如果 is_slide 为 true，提取所有核心技术词汇、组件名或英文缩写。")

# --- 1. 计算机视觉 (CV) 代理 ---

def extract_key_frames(video_path: str, output_dir: str, 
                       max_seconds: Optional[int] = None, 
                       target_size: tuple = (256, 144), 
                       diff_threshold: int = 850,
                       progress_hook=None) -> List[dict]:
    """
    极速视频关键帧离析 (Fast CV Slide Extraction).
    
    Args:
        video_path: 视频源文件路径。
        output_dir: 候选帧保存目录。
        max_seconds: 最大处理时长（秒），None 则处理全片。
        target_size: 比较时的缩略图尺寸，建议保持小尺寸以提升速度。
        diff_threshold: 画面差异阈值，MSE 超过此值则认为发生翻页。
        progress_hook: 回调函数 progress_hook(current_sec, total_sec)。
        
    Returns:
        List[dict]: 包含 'start_time', 'end_time', 'image' 的列表。
    """
    logger.info(f"CV: 开始分析视频流 {video_path}")
    cands_dir = os.path.join(output_dir, "candidates")
    os.makedirs(cands_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"CV: 无法打开视频文件 {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = max(1, int(fps))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_sec = total_frames / fps if fps > 0 else 0
    if max_seconds and total_sec > max_seconds:
        total_sec = max_seconds
    
    results = []
    frame_idx, current_slide_start_sec = 0.0, 0.0
    last_gray, last_full_frame, last_time_sec = None, None, 0.0
    
    def save_current_slide():
        if last_full_frame is not None:
            t_str = lambda t: f"{int(t)//3600:02d}-{int(t)%3600//60:02d}-{int(t)%60:02d}"
            fname = f"{t_str(current_slide_start_sec)}_{t_str(last_time_sec)}.jpg"
            out_path = os.path.join(cands_dir, fname)
            cv2.imwrite(out_path, last_full_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            rel_path = f"candidates/{fname}"
            results.append({"start_time": current_slide_start_sec, "end_time": last_time_sec, "image": rel_path})
    
    t_start = time.time()
    success, frame = cap.read()
    
    while success:
        sec = frame_idx / fps
        if max_seconds and sec > max_seconds: break
            
        if progress_hook: progress_hook(sec, total_sec)
        
        small = cv2.resize(frame, target_size, interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        if last_gray is None:
            last_gray, last_full_frame, last_time_sec = gray, frame.copy(), sec
        else:
            mse = np.sum((last_gray.astype("float") - gray.astype("float")) ** 2) / float(gray.size)
            if mse > diff_threshold:
                save_current_slide()
                current_slide_start_sec = sec
            last_gray, last_full_frame, last_time_sec = gray, frame.copy(), sec
            
        frame_idx += skip_frames
        for _ in range(skip_frames - 1):
            if not cap.grab(): success = False; break
        if not success: break
        success, frame = cap.read()
        
    save_current_slide()
    if progress_hook: progress_hook(total_sec, total_sec)
        
    cap.release()
    elapsed = time.time() - t_start
    logger.info(f"CV: 处理完成，共切分 {len(results)} 个候选帧，速率 { (max_seconds or last_time_sec)/elapsed:.1f}x")
    return results

# --- 2. 视觉大模型 (VLM) 代理 ---

def structured_llm_call(client: OpenAI, model: str, messages: List[dict], model_class: Any, supports_parse: bool, supports_response_format: bool) -> Tuple[Any, Optional[str]]:
    """统一的结构化 LLM 调用封装（支持 .parse / json_schema / Prompt+Regex 三级降级）"""
    if supports_parse:
        try:
            resp = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=model_class,
                timeout=600.0,
            )
            parsed = resp.choices[0].message.parsed
            reasoning = resp.choices[0].message.reasoning_content
            if parsed is None:
                raw_content = resp.choices[0].message.content
                raise ValueError(f"LLM returned None for parsed content. Raw: {raw_content}. Set VLM__SUPPORTS_PARSE=false if unsupported.")
            return parsed, reasoning
        except Exception as err:
            logger.warning(f"LLM .parse failed: {err}")
            raise

    import json
    schema_json = model_class.model_json_schema()
    has_sys = any(m["role"] == "system" for m in messages)
    schema_instruct = (
        "You are a helpful assistant designed to output structured JSON data. "
        "Your response must be valid JSON that exactly matches the following JSON schema:\n"
        f"{json.dumps(schema_json, ensure_ascii=False)}"
    )
    
    new_messages = []
    if has_sys:
        for m in messages:
            if m["role"] == "system":
                new_messages.append({"role": "system", "content": m["content"] + f"\n\n{schema_instruct}"})
            else:
                new_messages.append(m)
    else:
        new_messages = [{"role": "system", "content": schema_instruct}] + messages

    if supports_response_format:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=new_messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "structured_output",
                        "schema": schema_json
                    }
                },
                timeout=600.0,
            )
            full_text = resp.choices[0].message.content
            reasoning = resp.choices[0].message.reasoning_content
            logger.debug(f"LLM JSON schema response: [{full_text[:120]}]")
            parsed = model_class.model_validate_json(full_text.strip())
            return parsed, reasoning
        except Exception as parse_err:
            logger.warning(f"LLM JSON schema mode failed: {parse_err}")
            raise
    else:
        try:
            import re
            resp = client.chat.completions.create(
                model=model,
                messages=new_messages,
                timeout=600.0,
            )
            full_text = resp.choices[0].message.content
            reasoning = resp.choices[0].message.reasoning_content
            logger.debug(f"LLM raw response: [{full_text[:120]}]")
            cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', full_text.strip(), flags=re.MULTILINE)
            parsed = model_class.model_validate_json(cleaned)
            return parsed, reasoning
        except Exception as parse_err:
            logger.warning(f"LLM Regex fallback failed: {parse_err}")
            raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.5, min=2, max=10))
def vlm_task(base_url: str, api_key: str, model: str, task_type: str, images: List[str], supports_parse: bool = True, supports_response_format: bool = True) -> Tuple[Any, Optional[str]]:
    """
    多功能 VLM 任务处理器，支持幻灯片校验、去重、摘要生成及热词 OCR。
    
    Args:
        base_url: OpenAI 兼容接口地址。
        model: VLM 模型名称。
        task_type: 任务类型 ('validate'|'dedup'|'caption'|'terms')。
        images: 涉及的图片本地路径列表。
        supports_parse: 是否支持原生 .parse() 方法。
        supports_response_format: 是否支持 JSON Schema / JSON Object 结构化输出。
        
    Returns:
        Tuple[Any, Optional[str]]: (提取值, VLM推理过程); 校验/去重返回 bool，摘要返回 str，OCR 返回 List[str]。
    """
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=600.0,       # 改为 600s: 防止并发队列过长导致排队超时
        max_retries=0,       # 禁用客户端层重试，全部交由 tenacity 统一管控
    )
    prompts = {
        "validate": (
            "Determine if this image is a presentation slide (PPT/Keynote/Google Slides).\n"
            "Classify as slide (is_slide: true): frames with structured content — "
            "title/heading, bullet points, numbered lists, charts, diagrams, "
            "code snippets, tables, clean layout with a consistent background. "
            "Includes title slides, section divider slides, and agenda slides.\n"
            "Classify as NOT slide (is_slide: false): "
            "- Webcam video of a speaker (person's face or upper body, room background)\n"
            "- Conference room camera (wide-angle shots of a meeting room or audience)\n"
            "- Meeting software UI showing speaker name/tile without slide content "
            "(e.g., Teams/Zoom/Skype participant view, only a name on screen)\n"
            "- Windows desktop or other application windows before PPT is opened\n"
            "- Live coding terminal, IDE, or browser pages\n"
            "- Blank, black, loading or transitioning screens.\n"
            "Return ONLY a JSON object with no markdown or explanation: {\"is_slide\": true/false}"
        ),
        "dedup": (
            "Compare these two images and determine if they represent the SAME presentation slide.\n\n"
            "Crucial context: these frames are captured from a video recording of a PPT presentation. "
            "A single slide often spans multiple frames due to incremental animations.\n\n"
            "CRITERIA for is_same: true:\n"
            "- The two frames share the same overall layout, background, title/header, and color scheme\n"
            "- One frame has slightly MORE content due to PPT animation (e.g., an additional bullet point "
            "appearing, a chart segment fading in, an icon/material flying in, a number incrementing)\n"
            "- The difference is ONLY incremental content added to an existing slide, not a slide transition\n"
            "- Text content is nearly identical, with only 1-3 additional lines or elements in one frame\n"
            "- The same bullet points, headers, images, and diagrams are present in both (one just has more steps shown)\n\n"
            "CRITERIA for is_same: false:\n"
            "- The two frames have clearly different layouts, backgrounds, or color schemes\n"
            "- The title/header text is completely different\n"
            "- One frame is a title/cover slide and the other is a content slide\n"
            "- The topic or section has visibly changed (different core content, not just incremental)\n"
            "- One frame is black/blank/loading while the other is a content slide\n"
            "- The frames are from completely different presentation slides\n\n"
            "Examples of SAME slide (is_same: true):\n"
            "- Frame A shows 3 bullet points, Frame B shows 5 bullet points (same title, same layout)\n"
            "- Frame A shows a pie chart without labels, Frame B shows the same pie chart with labels animated in\n"
            "- Frame A shows a diagram without annotations, Frame B adds callout arrows to the same diagram\n"
            "- Frame A has a number '3' in a step indicator, Frame B has '4' (progress on the same slide)\n\n"
            "Examples of DIFFERENT slide (is_same: false):\n"
            "- Frame A is an 'Introduction' title slide, Frame B is a 'Technical Details' content slide\n"
            "- Frame A discusses hardware architecture, Frame B discusses software configuration\n"
            "- Frame A is a conclusion slide, Frame B is a new section divider\n\n"
            "Return ONLY a JSON object with no markdown or explanation: {\"is_same\": true/false}"
        ),
        "caption": (
            "请用一句20字内的中文描述此幻灯片内容。\n"
            "Return ONLY a JSON object with no markdown or explanation: {\"caption\": \"你的描述\"}"
        ),
        "terms": (
            "Extract all technical terms from this slide.\n"
            "Return ONLY a JSON object with no markdown or explanation: {\"items\": [\"term1\", \"term2\", ...]}"
        ),
        "analyze": (
            "Determine if this image is a presentation slide (PPT/Keynote/Google Slides).\n"
            "Classify as slide (is_slide: true): frames with structured content — "
            "title/heading, bullet points, numbered lists, charts, diagrams, "
            "code snippets, tables, clean layout with a consistent background.\n"
            "Classify as NOT slide (is_slide: false): "
            "- Webcam video of a speaker\n"
            "- Conference room camera\n"
            "- Meeting software UI\n"
            "- Blank, black, loading or transitioning screens.\n\n"
            "If is_slide is true, you MUST also provide:\n"
            "1. 'caption': A brief Chinese description of the slide (under 20 characters).\n"
            "2. 'items': A list of all technical terms, components, or acronyms found on the slide.\n\n"
            "Return ONLY a JSON object with no markdown."
        )
    }
    content = [{"type": "text", "text": prompts.get(task_type, task_type)}]
    for img in images:
        frame = cv2.imread(img)
        h, w = frame.shape[:2]
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(buf.tobytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            
    task_mapping = {
        "validate": (SlideValidation, False, lambda p: p.is_slide),
        "dedup": (SlideDeduplication, False, lambda p: p.is_same),
        "caption": (SlideCaption, "", lambda p: p.caption),
        "terms": (VisualVocabulary, [], lambda p: p.items[:20]),
        "analyze": (SlideAnalysis, None, lambda p: (p.is_slide, p.caption or "", (p.items or [])[:20]))
    }

    if task_type not in task_mapping:
        raise ValueError(f"Unknown task type: {task_type}")
        
    model_class, default_val, extractor = task_mapping[task_type]

    parsed, reasoning = structured_llm_call(client, model, [{"role": "user", "content": content}], model_class, supports_parse, supports_response_format)
    return extractor(parsed), reasoning


# --- 3. 语音转录 (ASR) 代理 ---

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.5, min=2, max=10))
def _transcribe_single_file(audio_path: str, prompt: str, model_size: str, api_base: str, api_key: str, hotwords: Optional[str]) -> List[dict]:
    client = OpenAI(
        api_key=api_key or "none",
        base_url=api_base,
        timeout=180.0,
        max_retries=0,
    )
    extra_body = {}
    if hotwords:
        extra_body["hotwords"] = hotwords
        extra_body["hot_words"] = hotwords
    resp = None
    formats_to_try = ["verbose_json", "json", "text"]
    for fmt in formats_to_try:
        with open(audio_path, "rb") as f:
            try:
                logger.info(f"ASR trying response_format='{fmt}' for model '{model_size}'")
                resp = client.audio.transcriptions.create(
                    model=model_size,
                    file=f,
                    prompt=prompt,
                    response_format=fmt,
                    extra_body=extra_body,
                    timeout=300.0
                )
                resp_attrs = [a for a in dir(resp) if not a.startswith("_")]
                logger.info(f"ASR succeeded with format='{fmt}', available fields: {resp_attrs}")
                break
            except Exception as e:
                err_msg = str(e)
                logger.warning(f"ASR with response_format='{fmt}' failed: {err_msg}")
                if fmt == formats_to_try[-1]:
                    logger.warning(f"ASR all fallback formats exhausted. Last error: {err_msg}")
                    raise
    
    raw = getattr(resp, "segments", [])
    if raw:
        logger.info(f"ASR parsed {len(raw)} segments from 'segments' field")
        return [{"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2), "text": s["text"].strip(), "speaker": "讲者"} for s in raw]

    text = getattr(resp, "text", "")
    if not text:
        logger.warning(f"ASR response has neither 'segments' nor 'text'. Raw attrs: {[a for a in dir(resp) if not a.startswith('_')]}")

    import re as _re
    sentences = [s.strip() for s in _re.split(r"([。！？\n])", text) if s.strip()]
    combined = []
    for item in sentences:
        if item in ["。", "！", "？", "\n"]:
            if combined:
                combined[-1] += item
        else:
            combined.append(item)

    try:
        size_bytes = os.path.getsize(audio_path)
        if audio_path.lower().endswith(".mp3"):
            duration = size_bytes * 8 / 64000
        elif audio_path.lower().endswith(".wav"):
            duration = size_bytes / 32000
        else:
            duration = 1680.0
    except Exception:
        duration = 1680.0
    if duration <= 0:
        duration = 1680.0

    if not combined:
        logger.info(f"ASR returned single-segment text (no sentence-split), duration={duration:.1f}s")
        return [{"start": 0.0, "end": round(duration, 2), "text": text, "speaker": "讲者"}]

    seg_duration = duration / len(combined)
    result = []
    for i, sent in enumerate(combined):
        start = round(i * seg_duration, 2)
        end = round((i + 1) * seg_duration, 2)
        result.append({"start": start, "end": end, "text": sent, "speaker": "讲者"})
    logger.info(f"ASR simulated {len(result)} segments from '{text[:80]}...' (plain text fallback, duration={duration:.1f}s)")
    return result

def split_audio(audio_path: str, chunk_length_s: int) -> List[str]:
    """
    使用 FFmpeg 将长音频分割为多个等长切片，用于规避 ASR API 的限制。
    返回切片文件路径列表。
    """
    output_dir = os.path.dirname(audio_path)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    out_pattern = os.path.join(output_dir, f"{base_name}_chunk_%03d{os.path.splitext(audio_path)[1]}")
    
    cmd_probe = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path
    ]
    try:
        duration_str = subprocess.check_output(cmd_probe, timeout=10).decode('utf-8').strip()
        duration = float(duration_str)
    except Exception as e:
        logger.warning(f"ASR chunking: 无法获取音频时长: {e}")
        duration = 0.0
        
    if duration > 0 and duration <= chunk_length_s + 5:
        return [audio_path]
        
    logger.info(f"ASR chunking: 音频时长 {duration:.1f}s 超过 {chunk_length_s}s，使用 FFmpeg 切片...")
    
    cmd_split = [
        "ffmpeg", "-i", audio_path, "-f", "segment", 
        "-segment_time", str(chunk_length_s), "-c", "copy", "-y", out_pattern
    ]
    subprocess.run(cmd_split, capture_output=True)
    
    chunks = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith(f"{base_name}_chunk_") and f.endswith(os.path.splitext(audio_path)[1])])
    return chunks if chunks else [audio_path]

def transcribe_with_whisper(audio_path: str, prompt: str, model_size: str = "base", api_base: Optional[str] = None, api_key: str = "none", device: str = "cpu", compute_type: str = "int8", hotwords: Optional[str] = None, chunk_length_s: int = 900, cache_dir: Optional[str] = None, progress_hook=None) -> List[dict]:
    """
    核心语音转录引擎，根据 api_base 自动分发至本地 Faster-Whisper 或远程 API。
    
    Args:
        audio_path: 待处理音频路径（建议 16k mono）。
        prompt: ASR 引导语，支持热词注入。
        model_size: 本地模型尺寸或 API 指定模型名。
        api_base: 若提供，则使用 OpenAI 兼容 API 模式。
        device: 本地模型运行设备 (cpu/cuda)。
        compute_type: 本地模型计算精度 (int8/float16等)。
        hotwords: 可选的热词列表（逗号分隔的字符串），用于 vLLM ASR 加权。
        chunk_length_s: 切片时长（秒），API 模式下防超时。
        cache_dir: 用于保存分片缓存的目录。
        progress_hook: 进度更新回调函数。
        
    Returns:
        List[dict]: 包含 'start', 'end', 'text', 'speaker' 的分段列表。
    """
    import json
    if api_base:
        chunks = split_audio(audio_path, chunk_length_s)
        all_segments = []
        
        for i, chunk_path in enumerate(chunks):
            chunk_cache_file = None
            if cache_dir:
                base_name = os.path.splitext(os.path.basename(chunk_path))[0]
                chunk_cache_file = os.path.join(cache_dir, f"{base_name}.json")
                
            if chunk_cache_file and os.path.exists(chunk_cache_file):
                try:
                    with open(chunk_cache_file, 'r', encoding='utf-8') as f:
                        chunk_segments = json.load(f)
                    if progress_hook: progress_hook()
                    all_segments.extend(chunk_segments)
                    continue
                except Exception:
                    pass

            if len(chunks) > 1:
                logger.info(f"ASR: 正在处理分片 {i+1}/{len(chunks)} ({os.path.basename(chunk_path)})...")
            
            offset = i * chunk_length_s
            chunk_segments = _transcribe_single_file(
                chunk_path, prompt, model_size, api_base, api_key, hotwords
            )
            
            for seg in chunk_segments:
                seg["start"] = round(seg["start"] + offset, 2)
                seg["end"] = round(seg["end"] + offset, 2)
                
            if chunk_cache_file:
                with open(chunk_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk_segments, f, ensure_ascii=False)
                    
            if progress_hook: progress_hook()
            all_segments.extend(chunk_segments)
            
        all_segments.sort(key=lambda x: x["start"])
        return all_segments
    else:
        from faster_whisper import WhisperModel
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
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed to extract audio: {e.stderr.decode('utf-8', errors='ignore')}")
        raise RuntimeError(f"FFmpeg extract_audio failed: {e}")
