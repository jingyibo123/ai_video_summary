import os
import time
import json
import shutil
import logging
import argparse
import concurrent.futures
import subprocess

from . import agents
from . import processor
from .config import AppConfig
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

# 配置根日志器使用 RichHandler，并共享 console
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(console=console, show_path=False)]
)

# 屏蔽 httpx 和 httpcore 的详细请求日志
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# --- Main Entry ---

def main() -> None:
    """
    自动化流水线入口：协调 CV、VLM、ASR 及数据渲染的完整流程。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="context.yaml", help="会议上下文 YAML 文件路径")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--output", default=None, help="输出目录 (默认在视频同级目录下的 ai_summary)")
    parser.add_argument("--max-time", type=int, default=None, help="最大处理时长（秒）")
    parser.add_argument("--skip-vlm", action="store_true", help="跳过 CV 截帧和 VLM 分析阶段")
    parser.add_argument("--skip-asr", action="store_true", help="跳过音频提取和 ASR 语音转录阶段")
    args = parser.parse_args()
    
    # 1. 确定输出目录
    video_abs = os.path.abspath(args.video)
    if not os.path.exists(video_abs):
        raise FileNotFoundError(f"输入视频文件不存在: {video_abs}")
    video_dir = os.path.dirname(video_abs)
    output_dir = args.output or os.path.join(video_dir, "ai_summary")
    os.makedirs(output_dir, exist_ok=True)
    
    # 新增：建立局部缓存目录
    cache_dir = os.path.join(output_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 2. 加载与备份配置
    config = AppConfig.load(args.config)
    if os.path.exists(args.config):
        src_abs = os.path.abspath(args.config)
        dst_abs = os.path.abspath(os.path.join(output_dir, "context.yaml"))
        if src_abs != dst_abs:
            shutil.copyfile(args.config, dst_abs)
    
    max_time = args.max_time if args.max_time else None

    t_start = time.time()
    logger.info(f"============== 开始流水线 V2.1 (Pydantic Config 版) Output: {output_dir} ==============")
    
    def _load_json(path):
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def _save_json(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 全局进度条 UI
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        # 1. 媒体离析
        aud_path = os.path.join(output_dir, "audio.wav")
        if not args.skip_asr:
            agents.extract_audio(args.video, aud_path, max_seconds=max_time)
        
        # 2. VLM 密集识别（三步：validate → dedup → enrich）
        meta_path = os.path.join(output_dir, "slide_metadata.json")
        vlm_progress_path = os.path.join(output_dir, "vlm_progress.json")
        dedup_path = os.path.join(output_dir, "vlm_deduped.json")
        enriched_path = os.path.join(output_dir, "vlm_enriched.json")

        if (slides_info := _load_json(meta_path)) is not None:
            pass
        elif (slides_info := _load_json(enriched_path)) is not None:
            for s in slides_info:
                s.setdefault("vlm_reasoning", None)
                s.setdefault("vlm_reasoning_description", None)
                s.setdefault("vlm_reasoning_terms", None)
            logger.info(f"VLM: 从 {enriched_path} 恢复了全部进度，跳过VLM步骤")
        elif args.skip_vlm:
            logger.info("VLM: --skip-vlm 被设置，跳过视觉分析阶段。")
            slides_info = []
        else:
            # --- Step 1: 验证 ---
            if (candidates := _load_json(vlm_progress_path)) is not None:
                logger.info(f"VLM: 恢复了 {len(candidates)} 帧的进度")
            else:
                task_cv = progress.add_task("[cyan]CV 视频截帧...", total=100)
                def _cv_hook(curr, total):
                    progress.update(task_cv, completed=curr, total=total)

                candidates = agents.extract_key_frames(
                    args.video, 
                    output_dir, 
                    max_seconds=max_time,
                    target_size=config.cv.target_size,
                    diff_threshold=config.cv.diff_threshold,
                    progress_hook=_cv_hook
                )
                progress.update(task_cv, completed=100, total=100)
                _save_json(candidates, vlm_progress_path)

            unvalidated = [c for c in candidates if "is_slide" not in c]
            if unvalidated:
                task_val = progress.add_task("[cyan]VLM 验证幻灯片...", total=len(unvalidated))
                with concurrent.futures.ThreadPoolExecutor(max_workers=config.vlm.max_workers) as pool:
                    futures = {
                        pool.submit(agents.vlm_task, config.vlm.base_url, config.vlm.api_key, config.vlm.model, "validate", [c['image']], config.vlm.supports_parse, config.vlm.supports_response_format): c
                        for c in unvalidated
                    }
                    completed = 0
                    for future in concurrent.futures.as_completed(futures):
                        c = futures[future]
                        c["is_slide"], c["vlm_reasoning"] = future.result()
                        completed += 1
                        progress.advance(task_val)
                        if completed % 10 == 0:
                            _save_json(candidates, vlm_progress_path)
                    _save_json(candidates, vlm_progress_path)

            valid = [c for c in candidates if c.get("is_slide")]
            
            # --- Step 2: 去重 ---
            if (dedup_data := _load_json(dedup_path)) is not None:
                deduped = dedup_data.get("slides", dedup_data) if isinstance(dedup_data, dict) else dedup_data
                dedup_decisions = dedup_data.get("dedup_decisions", []) if isinstance(dedup_data, dict) else []
                for s in deduped:
                    s.setdefault("vlm_reasoning", None)
                    s.setdefault("vlm_reasoning_description", None)
                    s.setdefault("vlm_reasoning_terms", None)
            else:
                deduped = []
                dedup_decisions = []
                task_dedup = progress.add_task("[magenta]VLM 去重处理...", total=len(valid))
                for v in valid:
                    if not deduped:
                        deduped.append(v)
                    else:
                        is_same, dedup_reasoning = agents.vlm_task(config.vlm.base_url, config.vlm.api_key, config.vlm.model, "dedup", [deduped[-1]['image'], v['image']], config.vlm.supports_parse, config.vlm.supports_response_format)
                        dedup_decisions.append({
                            "a": deduped[-1]['image'],
                            "b": v['image'],
                            "is_same": is_same,
                            "reasoning": dedup_reasoning
                        })
                        if not is_same:
                            deduped.append(v)
                        else:
                            deduped[-1]['end_time'] = v['end_time']
                        _save_json(deduped, dedup_path)
                    progress.advance(task_dedup)

                dedup_data = {"slides": deduped, "dedup_decisions": dedup_decisions}
                _save_json(dedup_data, dedup_path)

            # --- Step 3: 增强 ---
            if (slides_info := _load_json(enriched_path)) is not None:
                for s in slides_info:
                    s.setdefault("vlm_reasoning", None)
                    s.setdefault("vlm_reasoning_description", None)
                    s.setdefault("vlm_reasoning_terms", None)
            else:
                def _enrich_task(s):
                    desc, desc_reasoning = s.get("description"), None
                    kw, kw_reasoning = s.get("keywords"), None
                    if not desc:
                        desc, desc_reasoning = agents.vlm_task(config.vlm.base_url, config.vlm.api_key, config.vlm.model, "caption", [s["image"]], config.vlm.supports_parse, config.vlm.supports_response_format)
                    if not kw:
                        kw, kw_reasoning = agents.vlm_task(config.vlm.base_url, config.vlm.api_key, config.vlm.model, "terms", [s["image"]], config.vlm.supports_parse, config.vlm.supports_response_format)
                    return desc, kw, desc_reasoning, kw_reasoning
                
                task_enrich = progress.add_task("[green]VLM 提炼术语...", total=len(deduped))
                with concurrent.futures.ThreadPoolExecutor(max_workers=config.vlm.max_workers) as pool:
                    futures = {pool.submit(_enrich_task, s): s for s in deduped}
                    completed = 0
                    for future in concurrent.futures.as_completed(futures):
                        s = futures[future]
                        desc, kw, desc_reasoning, kw_reasoning = future.result()
                        s["description"] = desc
                        s["vlm_reasoning_description"] = desc_reasoning
                        s["keywords"] = kw
                        s["vlm_reasoning_terms"] = kw_reasoning
                        completed += 1
                        progress.advance(task_enrich)
                        if completed % 10 == 0:
                            _save_json(deduped, enriched_path)
                    _save_json(deduped, enriched_path)

                slides_info = deduped

            debug_path = os.path.join(output_dir, "vlm_dedup_debug.json")
            _save_json({
                "decisions": dedup_decisions,
                "summary": {
                    "total_valid": len(valid),
                    "deduped_count": len(deduped),
                    "merged_count": len(valid) - len(deduped)
                }
            }, debug_path)

            _save_json(slides_info, meta_path) 
        
        # 如果没有幻灯片信息（比如被 skip 了），创建一个全局的虚拟占位幻灯片，否则 Processor 无法生成任何内容
        if not slides_info:
            slides_info = [{
                "start_time": 0.0, 
                "end_time": max_time or 999999.0, 
                "image": "", 
                "description": "未提取视觉画面", 
                "keywords": []
            }]

        # 3. ASR
        ts_path = os.path.join(output_dir, "transcript.json")
        if args.skip_asr:
            logger.info("ASR: --skip-asr 被设置，跳过语音转录阶段。")
            transcript = []
        elif (transcript := _load_json(ts_path)) is not None:
            pass
        else:
            vul = []
            for s in slides_info: vul.extend(s.get("keywords") or [])
            all_terms = list(set(config.context.custom_terms + vul))
            prompt = f"这是一段技术讲座记录，请输出简体中文并带标点！主题: {config.context.meeting_title}。议程: {'，'.join(config.context.agenda[:3])}。"
            if all_terms: prompt += "包含术语：" + "，".join(all_terms)
            
            target_aud_path = aud_path
            if config.asr.api_base:
                mp3_path = os.path.join(output_dir, "audio.mp3")
                if not os.path.exists(mp3_path):
                    cmd = ["ffmpeg", "-i", aud_path, "-codec:a", "libmp3lame", "-b:a", "64k", "-y", mp3_path]
                    subprocess.run(cmd, capture_output=True)
                if os.path.exists(mp3_path):
                    target_aud_path = mp3_path
            
            # 计算切片大概数量来设置进度条
            from .agents import split_audio
            # 虽然 split_audio 会生成文件，但这里我们可以先预估或直接利用返回的文件数
            # 为了准确，让 agents.py 的 transcribe_with_whisper 自己利用 progress_hook 更新
            # 但我们需要知道 total，一种做法是传入未初始化的 task，或通过推断
            # 这里我们简单粗估音频长度或由 `transcribe_with_whisper` 内部自己建立进度，但由于是在 `with Progress` 里，我们建立一个 task
            task_asr = progress.add_task("[yellow]ASR 语音转录...", total=100) # 用一个假的 100%，然后在内部按切片推进
            # 改进：我们传入一个 advance 钩子，由于未知总数，暂时设为不确定或简单按段
            # 更好的办法是：transcribe_with_whisper 先获取切片数，但这需要改变它的签名
            # 简化版：由于 API base 切片是在函数里做的，我们将进度条交由钩子，每次处理完切片调用钩子并传递当前进度。但 progress_hook 只能 advance。
            # 直接在外部先分片以知道总数？ 不，我们先默认总进度不显示百分比，或者在 agents.py 里用 rich。
            # 这里我们仅仅传入一个简单的 hook: progress.advance(task_asr, 1) 但 total 需要是 len(chunks)
            # 为了优雅，让 transcribe_with_whisper 自行处理？不，我们保留 progress_hook。
            # 在没有 total 的情况下，设置 total=None 会变成 spinner。
            progress.update(task_asr, total=None) 
            
            hotwords_str = ",".join(all_terms) if all_terms else None
            transcript = agents.transcribe_with_whisper(
                target_aud_path, 
                prompt, 
                model_size=config.asr.model_size, 
                api_base=config.asr.api_base,
                api_key=config.asr.api_key,
                device=config.asr.local_device,
                compute_type=config.asr.local_compute_type,
                hotwords=hotwords_str,
                chunk_length_s=config.asr.chunk_length_s,
                cache_dir=cache_dir,
                progress_hook=lambda: progress.advance(task_asr, 1) if progress.tasks[task_asr].total else None
            )
            _save_json(transcript, ts_path)
            progress.update(task_asr, completed=100, total=100) # 完成后变成实心
     
        # 4. 提纯与渲染
        final_path = os.path.join(output_dir, "final_data.json")
        if (final_data := _load_json(final_path)) is None:
            task_proc = progress.add_task("[blue]生成技术博客...", total=len(slides_info))
            final_data = processor.build_final_json(
                config.vlm.base_url, 
                config.vlm.api_key,
                config.vlm.model, 
                slides_info, 
                transcript, 
                config.context.model_dump(),
                supports_parse=config.vlm.supports_parse,
                supports_response_format=config.vlm.supports_response_format,
                max_workers=config.vlm.max_workers,
                cache_dir=cache_dir,
                progress_hook=lambda: progress.advance(task_proc, 1)
            )
            _save_json(final_data, final_path)
     
        processor.render_minutes(final_data, os.path.join(output_dir, "format_a_minutes.md"))
        processor.render_blog(final_data, os.path.join(output_dir, "format_b_blog.md"))
    
    print()  # 修复 rich 进度条结束后 logger 输出在同一行的问题
    logger.info(f"============== 任务完成, 耗时 {time.time()-t_start:.1f}s ==============")
 
if __name__ == "__main__":
    main()
