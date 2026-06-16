"""
数据重组与多模态渲染引擎 (Data Synthesis & Renderer).

本模块充当流水线的“大脑”与“画笔”：
1. 数据整合: 利用 LLM 将分散的 ASR 片段与 VLM 视觉特征通过语义关联，聚合成结构化 JSON。
2. 多模态渲染: 解构 JSON 元数据，输出面向不同场景的 Format A (纪要) 与 Format B (博客)。

设计哲学：数据语义驱动，解耦底层识别与上层展示。
"""

import concurrent.futures
import os
import json
from typing import List, Optional, Callable
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry
from loguru import logger
from .config import dynamic_stop, dynamic_wait

# --- Structured Data Models ---

class SectionData(BaseModel):
    agenda_topic: str = Field(description="大议程标题，例如'DMA核心特性'")
    section_title: str = Field(description="三级标题，不得带'Slide'字样")
    image_caption: str = Field(description="图注说明")
    blog_text: str = Field(description="书面化的技术博文段落")

# --- 1. 数据合成 (Data Agent) ---

def build_final_json(base_url: str, api_key: str, model: str, slides: List[dict], transcript: List[dict], context: dict, supports_parse: bool = True, supports_response_format: bool = True, max_workers: int = 2, cache_dir: Optional[str] = None, progress_hook: Optional[Callable] = None, disable_thinking: bool = False) -> dict:
    """
    通过 LLM 聚合跨模态特征（图像描述、关键词、转录文本）生成结构化 JSON。
    
    Args:
        base_url: OpenAI 兼容接口地址。
        model: LLM 模型名称。
        slides: 经过 VLM 验证和增强的幻灯片列表。
        transcript: ASR 转录片段列表。
        context: 会议上下文（标题、议程等）。
        disable_thinking: 是否禁用思考/推理过程。
        
    Returns:
        dict: 完整的结构化会议数据。
    """
    logger.info("Processor: 开始构建结构化大纲...")
    agenda_str = ", ".join(context.get('agenda', [])) or "无明确议程"
    final_data = {
        "title": context.get("meeting_title", "会议纪要"),
        "date": context.get("date", "未知"),
        "location": context.get("location", "无"),
        "attendees": context.get("attendees", []),
        "sections": []
    }
    
    sys_prompt = f"你是一名为技术讲座进行深度博文提炼的专家。总议程: [{agenda_str}]。请将以下片段转化为严肃的技术干货章节。"

    from .agents import structured_llm_call, get_openai_client

    @retry(stop=dynamic_stop, wait=dynamic_wait)
    def call_llm(user_info_str: str) -> Optional[dict]:
        client = get_openai_client(api_key, base_url)
        try:
            parsed, _ = structured_llm_call(
                client=client,
                model=model,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_info_str}],
                model_class=SectionData,
                supports_parse=supports_parse,
                supports_response_format=supports_response_format,
                disable_thinking=disable_thinking
            )
            return parsed.model_dump() if parsed else None
        except Exception as e:
            logger.error(f"Structured content generation failed: {e}")
            raise

    def _process_one(i: int, s: dict) -> dict:
        cache_file = os.path.join(cache_dir, f"section_{i}.json") if cache_dir else None
        if cache_file and os.path.exists(cache_file):
            try:
                node = json.load(open(cache_file, 'r', encoding='utf-8'))
                if progress_hook: progress_hook()
                return node
            except Exception:
                pass
                
        raw = [seg["text"] for seg in transcript if seg["start"] < s["end_time"] and seg["end"] > s["start_time"]]
        speech = " ".join(raw)
        if len(speech) > 4000: speech = speech[:4000] + "..."
        user_info = f"时间: {s['start_time']}s-{s['end_time']}s\n画面: {s.get('description','')}\n原音: {speech or '无语音'}"
        
        try:
            node = call_llm(user_info)
        except Exception as e:
            logger.error(f"Structured content generation failed: {e}")
            node = None
            
        if not node:
            node = SectionData(
                agenda_topic="讲座内容", 
                section_title=f"分享 {i+1}", 
                image_caption=s.get('description','图片'), 
                blog_text=speech or "无内容"
            ).model_dump()
            
        node.update({"slide_index": i+1, "image_path": s["image"], "start_time": s["start_time"], "end_time": s["end_time"], 
                     "minutes_content": [seg for seg in transcript if seg["start"] < s["end_time"] and seg["end"] > s["start_time"]]})
                     
        if cache_file:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(node, f, ensure_ascii=False)
                
        if progress_hook: progress_hook()
        return node

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_process_one, i, s) for i, s in enumerate(slides)]
        final_data["sections"] = [f.result() for f in futures]
    return final_data

# --- 2. Markdown 渲染 (Markdown Agent) ---

def render_minutes(data: dict, out_path: str) -> None:
    """
    将结构化数据渲染为 Format A: 时间轴驱动的实录纪要。
    
    Args:
        data: 结构化会议数据。
        out_path: 输出 Markdown 文件路径。
    """
    lines = [f"# {data['title']} (实录纪要)\n", f"- **日期**: {data['date']}", f"- **地点**: {data['location']}", 
             f"- **与会人**: {', '.join(data['attendees'])}\n", "---"]
    
    current_agenda = None
    for sec in data["sections"]:
        if sec["agenda_topic"] != current_agenda:
            current_agenda = sec["agenda_topic"]
            lines.append(f"\n## {current_agenda}")
        
        lines.append(f"\n### {sec['section_title']}")
        lines.append(f"![{sec['image_caption']}]({sec['image_path']})\n> {sec['image_caption']}\n")
        
        last_speaker = None
        for seg in sec["minutes_content"]:
            time_tag = f"[{int(seg['start'])//60:02d}:{int(seg['start'])%60:02d}]"
            if seg["speaker"] != last_speaker:
                lines.append(f"\n**{seg['speaker']}** {time_tag}: {seg['text']}")
                last_speaker = seg["speaker"]
            else:
                lines[-1] += f" {seg['text']}"
    
    with open(out_path, 'w', encoding='utf-8') as f: f.write("\n".join(lines))

def render_blog(data: dict, out_path: str) -> None:
    """
    将结构化数据渲染为 Format B: 叙事风格的技术博客。
    
    Args:
        data: 结构化会议数据。
        out_path: 输出 Markdown 文件路径。
    """
    lines = [f"# {data['title']}\n", "> 会议总结与深度技术解析\n", "---"]
    current_agenda = None
    for sec in data["sections"]:
        if sec["agenda_topic"] != current_agenda:
            current_agenda = sec["agenda_topic"]
            lines.append(f"\n## {current_agenda}")
        lines.append(f"\n### {sec['section_title']}")
        lines.append(f"![{sec['image_caption']}]({sec['image_path']})\n> {sec['image_caption']}")
        lines.append(f"\n{sec['blog_text']}")
        
    with open(out_path, 'w', encoding='utf-8') as f: f.write("\n".join(lines))
