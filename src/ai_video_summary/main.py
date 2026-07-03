from pathlib import Path
import sys
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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from loguru import logger

console = Console()

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

# Intercept all standard logging messages and redirect to loguru
logging.basicConfig(handlers=[InterceptHandler()], level=logging.DEBUG, force=True)

# Mute detailed HTTP and API client logs from standard logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Remove default loguru handler and configure console handler to show warnings/errors only
logger.remove()
logger.add(sys.stderr, level="WARNING", format="<red>{level: <8}</red> | <level>{message}</level>")

# --- Helper functions ---

def parse_list_arg(arg_val) -> list[str]:
    if not arg_val:
        return []
    result = []
    for item in arg_val:
        parts = [p.strip() for p in item.split(",") if p.strip()]
        result.extend(parts)
    return result

# --- Main Entry ---

def main() -> None:
    """
    自动化流水线入口：协调 CV、VLM、ASR 及数据渲染的完整流程。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--output", default=None, help="输出目录 (默认在视频同级目录下的 {video_name}.ai_summary)")
    parser.add_argument("--max-time", type=int, default=None, help="最大处理时长（秒）")
    parser.add_argument("--skip-vlm", action="store_true", help="跳过 CV 截帧和 VLM 分析阶段")
    parser.add_argument("--skip-asr", action="store_true", help="跳过音频提取和 ASR 语音转录阶段")
    
    # ProjectContext CLI overrides
    parser.add_argument("--meeting-title", help="会议标题")
    parser.add_argument("--date", help="会议日期")
    parser.add_argument("--attendees", nargs="*", help="参会人列表 (空格或逗号分隔)")
    parser.add_argument("--agenda", nargs="*", help="会议议程列表 (空格或逗号分隔)")
    parser.add_argument("--custom-terms", nargs="*", help="自定义ASR术语列表 (空格或逗号分隔)")
    
    args = parser.parse_args()
    
    # 1. 确定输入和输出目录
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"输入视频文件不存在: {video_path}")
        
    output_dir = Path(args.output).resolve() if args.output else video_path.parent / f"{video_path.stem}.ai_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure Loguru to save all logs to summary.log inside the output directory
    log_file_path = output_dir / "summary.log"
    logger.add(
        str(log_file_path),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        encoding="utf-8"
    )
    
    # 新增：建立局部缓存目录
    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 2. 加载与备份配置
    config = AppConfig.load()
    
    # CLI Overrides for context fields
    update_data = {}
    if args.meeting_title is not None:
        update_data["meeting_title"] = args.meeting_title
    if args.date is not None:
        update_data["date"] = args.date
    if args.attendees is not None:
        update_data["attendees"] = parse_list_arg(args.attendees)
    if args.agenda is not None:
        update_data["agenda"] = parse_list_arg(args.agenda)
    if args.custom_terms is not None:
        update_data["custom_terms"] = parse_list_arg(args.custom_terms)

    # Auto-infer title/date from filename when not explicitly provided
    needs_title = "meeting_title" not in update_data
    needs_date = "date" not in update_data
    if needs_title or needs_date:
        try:
            inferred_date, inferred_title = agents.infer_from_filename(
                video_path.stem,
                base_url=config.get_llm_base_url(),
                api_key=config.get_llm_api_key(),
                model=config.llm.model,
                supports_response_format=config.llm.supports_response_format,
            )
            if needs_title and inferred_title:
                update_data["meeting_title"] = inferred_title
            if needs_date and inferred_date:
                update_data["date"] = inferred_date
        except Exception as e:
            logger.warning(f"文件名推断失败（非致命）: {e}")

    if update_data:
        config.context = config.context.model_copy(update=update_data)
        
    
    # 保存配置到输出文件夹
    # 2.1 完整 AppConfig dump 到 json (脱敏 api_key)
    config_dump = config.model_dump()
    for service in ["vlm", "llm", "asr"]:
        if service in config_dump and "api_key" in config_dump[service]:
            if config_dump[service]["api_key"] and config_dump[service]["api_key"] != "none":
                config_dump[service]["api_key"] = "********"
    config_run_path = output_dir / "config_run.json"
    with config_run_path.open("w", encoding="utf-8") as f:
        json.dump(config_dump, f, ensure_ascii=False, indent=2)
        
    max_time = args.max_time if args.max_time else None

    t_start = time.time()
    logger.info(f"============== 开始流水线 V2.1 (Pydantic Config 版) Output: {output_dir} ==============")
    
    # 自动检测音频轨，避免无音频视频崩溃且静默生成静音文件的错误
    if not args.skip_asr:
        try:
            if not agents.has_audio_track(str(video_path)):
                logger.error("🚨🚨🚨 [CRITICAL WARNING] 检测到输入视频文件没有音频轨！自动开启 --skip-asr 跳过语音转录与文本生成阶段！ 🚨🚨🚨")
                args.skip_asr = True
        except Exception as e:
            logger.warning(f"检测视频音频轨失败: {e}")
    
    def _load_json(path: Path):
        if path.exists():
            try:
                with path.open('r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def _save_json(data, path: Path):
        with path.open('w', encoding='utf-8') as f:
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
        aud_path = output_dir / "audio.wav"
        if not args.skip_asr:
            agents.extract_audio(str(video_path), str(aud_path), max_seconds=max_time)
        
        # 2. VLM 密集识别（两步：analyze → dedup）
        meta_path = output_dir / "slide_metadata.json"
        vlm_progress_path = output_dir / "vlm_progress.json"
        dedup_path = output_dir / "vlm_deduped.json"

        if (slides_info := _load_json(meta_path)) is not None:
            pass
        elif (dedup_data := _load_json(dedup_path)) is not None:
            slides_info = dedup_data.get("slides", dedup_data) if isinstance(dedup_data, dict) else dedup_data
            logger.info(f"VLM: 从 {dedup_path} 恢复了全部进度，跳过VLM步骤")
        elif args.skip_vlm:
            logger.info("VLM: --skip-vlm 被设置，跳过视觉分析阶段。")
            slides_info = []
        else:
            # --- Step 1: 验证与提炼 ---
            if (candidates := _load_json(vlm_progress_path)) is not None:
                logger.info(f"VLM: 恢复了 {len(candidates)} 帧的进度")
            else:
                task_cv = progress.add_task("[cyan]CV 视频截帧...", total=100)
                def _cv_hook(curr, total):
                    progress.update(task_cv, completed=curr, total=total)

                candidates = agents.extract_key_frames(
                    str(video_path), 
                    str(output_dir), 
                    max_seconds=max_time,
                    target_size=config.cv.target_size,
                    diff_threshold=config.cv.diff_threshold,
                    sample_interval=config.cv.sample_interval,
                    progress_hook=_cv_hook
                )
                progress.update(task_cv, completed=100, total=100)
                _save_json(candidates, vlm_progress_path)

            unvalidated = [c for c in candidates if "is_slide" not in c]
            if unvalidated:
                task_val = progress.add_task("[cyan]VLM 分析幻灯片...", total=len(unvalidated))
                with concurrent.futures.ThreadPoolExecutor(max_workers=config.vlm.max_workers) as pool:
                    futures = {
                        pool.submit(agents.vlm_task, config.vlm.base_url, config.vlm.api_key, config.vlm.model, "analyze", [str(output_dir / c['image']) if not Path(c['image']).is_absolute() else c['image']], config.vlm.supports_parse, config.vlm.supports_response_format, config.vlm.disable_thinking, config.vlm.max_thinking_tokens): c
                        for c in unvalidated
                    }
                    completed = 0
                    for future in concurrent.futures.as_completed(futures):
                        c = futures[future]
                        result, reasoning = future.result()
                        c["is_slide"], c["description"], c["keywords"] = result
                        c["vlm_reasoning"] = reasoning
                        completed += 1
                        progress.advance(task_val)
                        if completed % 10 == 0:
                            _save_json(candidates, vlm_progress_path)
                    _save_json(candidates, vlm_progress_path)

            valid = [c for c in candidates if c.get("is_slide")]
            
            # --- Step 2: 去重与归并 ---
            deduped = []
            dedup_decisions = []
            
            if not valid:
                pass
            elif len(valid) == 1:
                deduped.append(valid[0].copy())
            else:
                task_dedup = progress.add_task("[magenta]VLM 并发去重处理...", total=len(valid) - 1)
                
                def _dedup_pair(i):
                    a = valid[i]
                    b = valid[i+1]
                    img_a_abs = output_dir / a['image'] if not Path(a['image']).is_absolute() else Path(a['image'])
                    img_b_abs = output_dir / b['image'] if not Path(b['image']).is_absolute() else Path(b['image'])
                    is_same, dedup_reasoning = agents.vlm_task(
                        config.vlm.base_url, config.vlm.api_key, config.vlm.model, 
                        "dedup", [str(img_a_abs), str(img_b_abs)], 
                        config.vlm.supports_parse, config.vlm.supports_response_format,
                        config.vlm.disable_thinking, config.vlm.max_thinking_tokens
                    )
                    return i, is_same, dedup_reasoning, a['image'], b['image']

                results_map = {}
                partial_dedup_path = cache_dir / "vlm_dedup_partial.json"
                if (partial_saved := _load_json(partial_dedup_path)) is not None:
                    for k, v in partial_saved.items():
                        results_map[int(k)] = v

                def _dedup_wrapper(i):
                    if i in results_map:
                        return i, results_map[i]["is_same"], results_map[i]["reasoning"], results_map[i]["a"], results_map[i]["b"]
                    return _dedup_pair(i)

                completed = 0
                with concurrent.futures.ThreadPoolExecutor(max_workers=config.vlm.max_workers) as pool:
                    futures = {pool.submit(_dedup_wrapper, i): i for i in range(len(valid) - 1)}
                    for future in concurrent.futures.as_completed(futures):
                        i, is_same, reason, img_a, img_b = future.result()
                        if i not in results_map:
                            results_map[i] = {
                                "is_same": is_same,
                                "reasoning": reason,
                                "a": img_a,
                                "b": img_b
                            }
                        completed += 1
                        progress.advance(task_dedup)
                        if completed % 10 == 0:
                            _save_json(results_map, partial_dedup_path)
                _save_json(results_map, partial_dedup_path)
                
                # 组装结果
                groups = []
                current_group = [valid[0]]
                for i in range(len(valid) - 1):
                    decision = results_map[i]
                    dedup_decisions.append({
                        "a": decision["a"],
                        "b": decision["b"],
                        "is_same": decision["is_same"],
                        "reasoning": decision["reasoning"]
                    })
                    
                    if decision["is_same"]:
                        current_group.append(valid[i+1])
                    else:
                        groups.append(current_group)
                        current_group = [valid[i+1]]
                groups.append(current_group)

                for g in groups:
                    rep = g[-1].copy()
                    rep['start_time'] = g[0]['start_time']
                    rep['end_time'] = g[-1]['end_time']
                    
                    # 把这组连续动画里的所有提取术语做并集，防止中间动画遮挡
                    merged_keywords = []
                    for frame in g:
                        for kw in frame.get("keywords", []):
                            if kw not in merged_keywords:
                                merged_keywords.append(kw)
                    rep['keywords'] = merged_keywords
                    
                    deduped.append(rep)

                dedup_data = {"slides": deduped, "dedup_decisions": dedup_decisions}
                _save_json(dedup_data, dedup_path)

            slides_info = deduped

            debug_path = output_dir / "vlm_dedup_debug.json"
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
        
        # 建立 assets 目录并复制最终保留的图片
        assets_dir = output_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        for s in slides_info:
            img_path = s.get("image", "")
            if img_path:
                img_path_obj = Path(img_path)
                img_abs = output_dir / img_path_obj if not img_path_obj.is_absolute() else img_path_obj
                if img_abs.exists():
                    fname = img_abs.name
                    new_rel = f"assets/{fname}"
                    new_abs = output_dir / new_rel
                    if img_abs != new_abs:
                        shutil.copyfile(str(img_abs), str(new_abs))
                    s["image"] = new_rel

        # 3. ASR
        ts_path = output_dir / "transcript.json"
        if args.skip_asr:
            logger.info("ASR: --skip-asr 被设置，跳过语音转录阶段。")
            transcript = []
        elif (transcript := _load_json(ts_path)) is not None:
            pass
        else:
            vul = []
            for s in slides_info: vul.extend(s.get("keywords") or [])
            raw_terms = list(set(config.context.custom_terms + vul))
            
            # Use LLM to filter and refine technical terms if there are too many (keeps ASR prompt focused)
            if len(raw_terms) > 20:
                logger.info(f"ASR: 候选热词过多 ({len(raw_terms)} 个)，使用 LLM 进行精炼筛选...")
                all_terms = agents.filter_asr_hotwords(
                    config.get_llm_base_url(),
                    config.get_llm_api_key(),
                    config.llm.model,
                    raw_terms,
                    config.context.agenda,
                    config.context.meeting_title
                )
            else:
                all_terms = raw_terms
            
            prompt = f"这是一段技术讲座记录，请输出简体中文并带标点！主题: {config.context.meeting_title}。议程: {'，'.join(config.context.agenda[:3])}。"
            if all_terms: prompt += "包含术语：" + "，".join(all_terms)
            
            target_aud_path = aud_path
            if config.asr.api_base:
                mp3_path = output_dir / "audio.mp3"
                if not mp3_path.exists():
                    cmd = ["ffmpeg", "-i", str(aud_path), "-codec:a", "libmp3lame", "-b:a", "64k", "-y", str(mp3_path)]
                    subprocess.run(cmd, capture_output=True)
                if mp3_path.exists():
                    target_aud_path = mp3_path
            
            # Pre-split chunks beforehand to set accurate progress bar total
            from .agents import split_audio
            if config.asr.api_base:
                chunks = split_audio(str(target_aud_path), config.asr.chunk_length_s)
                task_asr = progress.add_task("[yellow]ASR 语音转录...", total=len(chunks))
            else:
                chunks = None
                task_asr = progress.add_task("[yellow]ASR 语音转录...", total=1)
            
            hotwords_str = ",".join(all_terms) if all_terms else None
            transcript = agents.transcribe_with_whisper(
                str(target_aud_path), 
                prompt, 
                model_size=config.asr.model_size, 
                api_base=config.asr.api_base,
                api_key=config.asr.api_key,
                device=config.asr.local_device,
                compute_type=config.asr.local_compute_type,
                hotwords=hotwords_str,
                chunk_length_s=config.asr.chunk_length_s,
                cache_dir=cache_dir,
                progress_hook=lambda: progress.advance(task_asr, 1),
                max_workers=config.asr.max_workers,
                chunks=chunks
            )
            _save_json(transcript, ts_path)
            progress.update(task_asr, completed=len(chunks) if chunks else 1, total=len(chunks) if chunks else 1)
     
        # 4. 提纯与渲染
        final_path = output_dir / "final_data.json"
        if (final_data := _load_json(final_path)) is None:
            task_proc = progress.add_task("[blue]生成技术博客...", total=len(slides_info))
            final_data = processor.build_final_json(
                config.get_llm_base_url(), 
                config.get_llm_api_key(),
                config.llm.model, 
                slides_info, 
                transcript, 
                config.context.model_dump(),
                supports_parse=config.llm.supports_parse,
                supports_response_format=config.llm.supports_response_format,
                max_workers=config.llm.max_workers,
                cache_dir=cache_dir,
                progress_hook=lambda: progress.advance(task_proc, 1),
                disable_thinking=config.llm.disable_thinking,
                max_thinking_tokens=config.llm.max_thinking_tokens
            )
            _save_json(final_data, final_path)
     
        processor.render_minutes(final_data, output_dir / "format_a_minutes.md")
        processor.render_blog(final_data, output_dir / "format_b_blog.md")
    
    print()  # 修复 rich 进度条结束后 logger 输出在同一行的问题
    logger.info(f"============== 任务完成, 耗时 {time.time()-t_start:.1f}s ==============")
 
if __name__ == "__main__":
    main()
