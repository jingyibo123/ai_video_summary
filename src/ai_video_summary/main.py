import os
import time
import json
import shutil
import logging
import argparse
import concurrent.futures

from . import agents
from . import processor
from .config import AppConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- Main Entry ---

def main() -> None:
    """
    自动化流水线入口：协调 CV、VLM、ASR 及数据渲染的完整流程。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--output", default=None, help="输出目录 (默认在视频同级目录下的 ai_summary)")
    parser.add_argument("--max-time", type=int, default=None, help="最大处理时长（秒）")
    args = parser.parse_args()
    
    # 1. 确定输出目录
    video_abs = os.path.abspath(args.video)
    video_dir = os.path.dirname(video_abs)
    output_dir = args.output or os.path.join(video_dir, "ai_summary")
    os.makedirs(output_dir, exist_ok=True)

    # 2. 加载与备份配置
    config = AppConfig.from_yaml(args.config)
    shutil.copy2(args.config, os.path.join(output_dir, "config.yaml"))
    
    # 手动覆盖部分关键参数（如果 CLI 提供了）
    max_time = args.max_time if args.max_time else None

    t_start = time.time()
    logger.info(f"============== 开始流水线 V2.1 (Pydantic Config 版) Output: {output_dir} ==============")
    
    # 1. 媒体离析与 CV
    aud_path = os.path.join(output_dir, "audio.wav")
    agents.extract_audio(args.video, aud_path, max_seconds=max_time)
    
    # 2. VLM 密集识别
    meta_path = os.path.join(output_dir, "slide_metadata.json")
    if os.path.exists(meta_path):
        slides_info = json.load(open(meta_path, 'r', encoding='utf-8'))
    else:
        candidates = agents.extract_key_frames(
            args.video, 
            output_dir, 
            max_seconds=max_time,
            target_size=config.cv.target_size,
            diff_threshold=config.cv.diff_threshold
        )
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            is_slide = list(pool.map(
                lambda c: agents.vlm_task(config.vlm.base_url, config.vlm.model, "validate", [c['image']]), 
                candidates
            ))
        valid = [c for c, m in zip(candidates, is_slide) if m]
        
        deduped = []
        for v in valid:
            if not deduped or not agents.vlm_task(config.vlm.base_url, config.vlm.model, "dedup", [deduped[-1]['image'], v['image']]):
                deduped.append(v)
            else: deduped[-1]['end_time'] = v['end_time']
                
        def enrich(s):
            s["description"] = agents.vlm_task(config.vlm.base_url, config.vlm.model, "caption", [s["image"]])
            s["keywords"] = agents.vlm_task(config.vlm.base_url, config.vlm.model, "terms", [s["image"]])
            return s
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            slides_info = list(pool.map(enrich, deduped))
        json.dump(slides_info, open(meta_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
 
    # 3. ASR
    ts_path = os.path.join(output_dir, "transcript.json")
    if os.path.exists(ts_path):
        transcript = json.load(open(ts_path, 'r', encoding='utf-8'))
    else:
        vul = []
        for s in slides_info: vul.extend(s.get("keywords", []))
        all_terms = list(set(config.context.custom_terms + vul))
        prompt = f"这是一段技术讲座记录，请输出简体中文并带标点！主题: {config.context.meeting_title}。议程: {'，'.join(config.context.agenda[:3])}。"
        if all_terms: prompt += "包含术语：" + "，".join(all_terms)
        
        transcript = agents.transcribe_with_whisper(
            aud_path, 
            prompt, 
            model_size=config.asr.model_size, 
            api_base=config.asr.api_base,
            device=config.asr.local_device,
            compute_type=config.asr.local_compute_type
        )
        json.dump(transcript, open(ts_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
 
    # 4. 提纯与渲染
    final_path = os.path.join(output_dir, "final_data.json")
    if os.path.exists(final_path):
        final_data = json.load(open(final_path, 'r', encoding='utf-8'))
    else:
        final_data = processor.build_final_json(
            config.vlm.base_url, 
            config.vlm.model, 
            slides_info, 
            transcript, 
            config.context.model_dump()
        )
        json.dump(final_data, open(final_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
 
    processor.render_minutes(final_data, os.path.join(output_dir, "format_a_minutes.md"))
    processor.render_blog(final_data, os.path.join(output_dir, "format_b_blog.md"))
    logger.info(f"============== 任务完成, 耗时 {time.time()-t_start:.1f}s ==============")
 
if __name__ == "__main__":
    main()
