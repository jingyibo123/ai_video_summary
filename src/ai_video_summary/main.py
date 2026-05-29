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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
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
    args = parser.parse_args()
    
    # 1. 确定输出目录
    video_abs = os.path.abspath(args.video)
    video_dir = os.path.dirname(video_abs)
    output_dir = args.output or os.path.join(video_dir, "ai_summary")
    os.makedirs(output_dir, exist_ok=True)

    # 2. 加载与备份配置
    config = AppConfig.load(args.config)
    if os.path.exists(args.config):
        src_abs = os.path.abspath(args.config)
        dst_abs = os.path.abspath(os.path.join(output_dir, "context.yaml"))
        if src_abs != dst_abs:
            shutil.copy2(args.config, dst_abs)
    
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
        vlm_progress_path = os.path.join(output_dir, "vlm_progress.json")
        if os.path.exists(vlm_progress_path):
            with open(vlm_progress_path, 'r', encoding='utf-8') as f:
                candidates = json.load(f)
            logger.info(f"VLM: 从 {vlm_progress_path} 恢复了 {len(candidates)} 帧的进度")
        else:
            candidates = agents.extract_key_frames(
                args.video, 
                output_dir, 
                max_seconds=max_time,
                target_size=config.cv.target_size,
                diff_threshold=config.cv.diff_threshold
            )
            with open(vlm_progress_path, 'w', encoding='utf-8') as f:
                json.dump(candidates, f, ensure_ascii=False, indent=2)
        
        # 1. 验证是否为幻灯片
        unvalidated = [c for c in candidates if "is_slide" not in c]
        if unvalidated:
            logger.info(f"VLM: 正在验证 {len(unvalidated)} 帧是否为幻灯片...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
                futures = {
                    pool.submit(agents.vlm_task, config.vlm.base_url, config.vlm.api_key, config.vlm.model, "validate", [c['image']]): c 
                    for c in unvalidated
                }
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    c = futures[future]
                    c["is_slide"] = future.result()
                    completed += 1
                    # 主线程安全保存细粒度进度，每10个批处理一次缓解IO
                    if completed % 10 == 0:
                        with open(vlm_progress_path, 'w', encoding='utf-8') as f:
                            json.dump(candidates, f, ensure_ascii=False, indent=2)
                with open(vlm_progress_path, 'w', encoding='utf-8') as f:
                    json.dump(candidates, f, ensure_ascii=False, indent=2)

        valid = [c for c in candidates if c.get("is_slide")]
        
        # 2. 串行去重
        dedup_path = os.path.join(output_dir, "vlm_deduped.json")
        if os.path.exists(dedup_path):
            with open(dedup_path, 'r', encoding='utf-8') as f:
                deduped = json.load(f)
            logger.info(f"VLM: 从 {dedup_path} 恢复了去重进度，共 {len(deduped)} 帧")
        else:
            logger.info(f"VLM: 正在对 {len(valid)} 个有效帧进行去重...")
            deduped = []
            for v in valid:
                if not deduped or not agents.vlm_task(config.vlm.base_url, config.vlm.api_key, config.vlm.model, "dedup", [deduped[-1]['image'], v['image']]):
                    deduped.append(v)
                else: 
                    deduped[-1]['end_time'] = v['end_time']
                
                with open(dedup_path, 'w', encoding='utf-8') as f:
                    json.dump(deduped, f, ensure_ascii=False, indent=2)
                
        # 3. 增强描述与术语
        unenriched = [s for s in deduped if "description" not in s or "keywords" not in s]
        if unenriched:
            logger.info(f"VLM: 正在为 {len(unenriched)} 帧生成描述和关键词...")
            def _enrich_task(s):
                desc = s.get("description") or agents.vlm_task(config.vlm.base_url, config.vlm.api_key, config.vlm.model, "caption", [s["image"]])
                kw = s.get("keywords") or agents.vlm_task(config.vlm.base_url, config.vlm.api_key, config.vlm.model, "terms", [s["image"]])
                return desc, kw
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(_enrich_task, s): s for s in unenriched}
                completed = 0
                for future in concurrent.futures.as_completed(futures):
                    s = futures[future]
                    desc, kw = future.result()
                    s["description"] = desc
                    s["keywords"] = kw
                    completed += 1
                    if completed % 10 == 0:
                        with open(dedup_path, 'w', encoding='utf-8') as f:
                            json.dump(deduped, f, ensure_ascii=False, indent=2)
                with open(dedup_path, 'w', encoding='utf-8') as f:
                    json.dump(deduped, f, ensure_ascii=False, indent=2)
                        
        slides_info = deduped
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(slides_info, f, ensure_ascii=False, indent=2)
 
    # 3. ASR
    ts_path = os.path.join(output_dir, "transcript.json")
    if os.path.exists(ts_path):
        transcript = json.load(open(ts_path, 'r', encoding='utf-8'))
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
                logger.info(f"API-first ASR: 正在将音频 {aud_path} 压缩为 MP3 ({mp3_path}) 以优化 API 传输...")
                cmd = ["ffmpeg", "-i", aud_path, "-codec:a", "libmp3lame", "-b:a", "64k", "-y", mp3_path]
                subprocess.run(cmd, capture_output=True)
            if os.path.exists(mp3_path):
                logger.info(f"API-first ASR: 使用压缩后的 MP3 音频进行转录")
                target_aud_path = mp3_path
        
        hotwords_str = ",".join(all_terms) if all_terms else None
        transcript = agents.transcribe_with_whisper(
            target_aud_path, 
            prompt, 
            model_size=config.asr.model_size, 
            api_base=config.asr.api_base,
            api_key=config.asr.api_key,
            device=config.asr.local_device,
            compute_type=config.asr.local_compute_type,
            hotwords=hotwords_str
        )
        json.dump(transcript, open(ts_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
 
    # 4. 提纯与渲染
    final_path = os.path.join(output_dir, "final_data.json")
    if os.path.exists(final_path):
        final_data = json.load(open(final_path, 'r', encoding='utf-8'))
    else:
        final_data = processor.build_final_json(
            config.vlm.base_url, 
            config.vlm.api_key,
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
