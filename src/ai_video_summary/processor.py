"""
数据重组与多模态渲染引擎 (Data Synthesis & Renderer).

本模块充当流水线的“大脑”与“画笔”：
1. 数据整合: 利用 LLM 将分散的 ASR 片段与 VLM 视觉特征通过语义关联，聚合成结构化 JSON。
2. 多模态渲染: 解构 JSON 元数据，输出面向不同场景的 Format A (纪要) 与 Format B (博客)。

设计哲学：数据语义驱动，解耦底层识别与上层展示。
"""

import logging
import concurrent.futures
from typing import List, Optional
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# --- Structured Data Models ---

class SectionData(BaseModel):
    agenda_topic: str = Field(description="大议程标题，例如'DMA核心特性'")
    section_title: str = Field(description="三级标题，不得带'Slide'字样")
    image_caption: str = Field(description="图注说明")
    blog_text: str = Field(description="书面化的技术博文段落")

# --- 1. 数据合成 (Data Agent) ---

def build_final_json(base_url: str, model: str, slides: List[dict], transcript: List[dict], context: dict) -> dict:
    """
    通过 LLM 聚合跨模态特征（图像描述、关键词、转录文本）生成结构化 JSON。
    
    Args:
        base_url: OpenAI 兼容接口地址。
        model: LLM 模型名称。
        slides: 经过 VLM 验证和增强的幻灯片列表。
        transcript: ASR 转录片段列表。
        context: 会议上下文（标题、议程等）。
        
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

    def _process_one(i, s):
        raw = [seg["text"] for seg in transcript if seg["start"] < s["end_time"] and seg["end"] > s["start_time"]]
        speech = " ".join(raw)
        if len(speech) > 4000: speech = speech[:4000] + "..."
        user_info = f"时间: {s['start_time']}s-{s['end_time']}s\n画面: {s.get('description','')}\n原音: {speech or '无语音'}"
        
        try:
            client = OpenAI(api_key="none", base_url=base_url)
            # 内联 LLM 请求
            resp = client.beta.chat.completions.parse(
                model=model, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_info}],
                response_format=SectionData, temperature=0.3
            )
            node = resp.choices[0].message.parsed.model_dump() if resp.choices[0].message.parsed else None
        except: node = None
        
        if not node:
            node = {"agenda_topic": "讲座内容", "section_title": f"分享 {i+1}", "image_caption": s.get('description','图片'), "blog_text": speech or "无内容"}
            
        node.update({"slide_index": i+1, "image_path": s["image"], "start_time": s["start_time"], "end_time": s["end_time"], 
                     "minutes_content": [seg for seg in transcript if seg["start"] < s["end_time"] and seg["end"] > s["start_time"]]})
        return node

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
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
