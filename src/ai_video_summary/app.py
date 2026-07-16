"""
Gradio Web UI for ai-video-summary pipeline.

Parameter schema drives both Gradio widgets and CLI flags with zero duplication.
Runtime/infrastructure settings (API keys, models, etc.) stay in .env.
"""

from __future__ import annotations

import json
import re
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Generator

import gradio as gr
from ai_video_summary import agents
from ai_video_summary.config import AppConfig


# ---------------------------------------------------------------------------
# Context parameter schema
# Each entry → one Gradio widget + one CLI flag, built automatically.
# ADVANCED pipeline flags (skip-vlm, skip-asr, max-time) are intentionally
# omitted from the UI — they remain available as CLI args for debugging.
# ---------------------------------------------------------------------------

CONTEXT_SCHEMA = [
    {
        "flag": "--meeting-title",
        "label": "视频 / 会议标题 (选填)",
        "info": "视频/会议的标题。若留空，则使用 LLM 从文件名自动提取。",
        "default": "",
        "type": "text",
        "placeholder": "若留空，则自动从文件名推断",
    },
    {
        "flag": "--date",
        "label": "日期 (选填)",
        "info": "视频/会议的日期（格式: YYYY-MM-DD）。若留空，则从文件名提取。",
        "default": "",
        "type": "text",
        "placeholder": "格式: YYYY-MM-DD，若留空则从文件名推断",
    },
    {
        "flag": "--attendees",
        "label": "参会人 (选填)",
        "info": "参会人列表，用英文逗号分隔。例如: 张三, 李四。",
        "default": "",
        "type": "text",
        "placeholder": "英文逗号分隔。例: 张三, 李四",
    },
    {
        "flag": "--agenda",
        "label": "议程 (选填)",
        "info": "会议议程列表，用英文逗号分隔。例如: DMA 介绍, 演示。",
        "default": "",
        "type": "text",
        "placeholder": "英文逗号分隔。例: DMA 介绍, 演示",
    },
    {
        "flag": "--custom-terms",
        "label": "专业术语 / 关键词 (选填)",
        "info": "专业术语或关键词列表，用英文逗号分隔。强烈建议填写以减少语音识别错误。例如: DMA, MCU。",
        "default": "",
        "type": "text",
        "placeholder": "强烈建议填写以减少语音识别错误。例: DMA, MCU",
    },
]

ALL_SCHEMA = CONTEXT_SCHEMA  # only content fields; pipeline flags stay CLI-only


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_widget(s: dict) -> gr.components.Component:
    """Instantiate a single Gradio widget from a schema entry."""
    info = s.get("info")
    if s["type"] == "text":
        return gr.Textbox(
            label=s["label"],
            info=info,
            value=s["default"],
            placeholder=s.get("placeholder", ""),
        )
    elif s["type"] == "number":
        return gr.Number(label=s["label"], info=info, value=s["default"], precision=0)
    elif s["type"] == "checkbox":
        return gr.Checkbox(label=s["label"], info=info, value=s["default"])
    return gr.Textbox(label=s["label"], info=info, value=s["default"])


def _build_cli_args(param_values: list) -> list[str]:
    """Convert ordered widget values → CLI argument list."""
    args = []
    for schema, value in zip(ALL_SCHEMA, param_values):
        flag = schema["flag"]
        ptype = schema["type"]
        if ptype == "checkbox":
            if value:
                args.append(flag)
        elif ptype == "number":
            val = int(value) if value else 0
            if val > 0:
                args.extend([flag, str(val)])
        else:
            text = str(value).strip()
            if text:
                args.extend([flag, text])
    return args


def _zip_output(output_dir: Path) -> str:
    zip_base = str(output_dir) + "_result"
    return shutil.make_archive(zip_base, "zip", str(output_dir.parent), output_dir.name)


# ---------------------------------------------------------------------------
# Core processing generator (real-time log streaming via Gradio queue)
# ---------------------------------------------------------------------------

def process_video(
    video_file: str | None,
    meeting_title: str = "",
    date: str = "",
    attendees: str = "",
    agenda: str = "",
    custom_terms: str = "",
) -> Generator[tuple[str, str | None], None, None]:
    """
    运行视频智能摘要流水线，提取视频内容并生成技术博客。

    Args:
        video_file (str | None): 上传视频文件的本地路径。
        meeting_title (str): 视频/会议的标题。若留空，则使用 LLM 从文件名自动提取。
        date (str): 视频/会议的日期（格式: YYYY-MM-DD）。若留空，则从文件名提取。
        attendees (str): 参会人列表，用英文逗号分隔。例如: "张三, 李四"。
        agenda (str): 会议议程列表，用英文逗号分隔。例如: "DMA 介绍, 演示"。
        custom_terms (str): 专业术语或关键词列表，用英文逗号分隔。强烈建议填写以减少语音识别错误。例如: "DMA, MCU"。

    Yields:
        Tuple[str, str | None]: 包含实时输出的日志行，以及处理完成后的结果 ZIP 文件路径。
    """
    log_buf = ""

    def _emit(line: str) -> str:
        nonlocal log_buf
        log_buf += line + "\n"
        return log_buf

    if not video_file:
        yield _emit("❌ 请先上传视频文件！"), None
        return

    video_path = Path(video_file)
    if not video_path.exists():
        yield _emit(f"❌ 视频文件不存在: {video_path}"), None
        return

    output_dir = video_path.parent / f"{video_path.stem}.ai_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct parameter values list for CLI converter
    param_values = [meeting_title, date, attendees, agenda, custom_terms]
    extra_args = _build_cli_args(param_values)
    cmd = [
        sys.executable, "-m", "ai_video_summary.main",
        "--video", str(video_path),
        "--output", str(output_dir),
        "--no-progress",
        *extra_args,
    ]

    yield _emit(f"🚀 启动流水线...\n命令: {' '.join(cmd)}\n{'─' * 60}"), None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=str(Path(__file__).parent.parent.parent),  # project root
        )

        for line in proc.stdout:
            yield _emit(line.rstrip()), None

        proc.wait()

        if proc.returncode == 0:
            yield _emit(f"\n{'─' * 60}\n✅ 处理完成！正在打包结果..."), None
            zip_path = _zip_output(output_dir)
            yield _emit(f"📦 结果已打包: {zip_path}"), zip_path
        else:
            yield _emit(f"\n{'─' * 60}\n❌ 流水线异常退出，返回码: {proc.returncode}"), None

    except FileNotFoundError as e:
        yield _emit(f"❌ 无法启动进程，请检查 Python 环境: {e}"), None
    except Exception as e:
        yield _emit(f"❌ 未知错误: {e}"), None


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

def build_ui() -> gr.Interface:
    video_input = gr.File(
        label="上传视频文件",
        file_types=["video", ".mp4", ".mov", ".avi", ".mkv", ".webm"],
        type="filepath",
    )
    all_widgets = [_make_widget(s) for s in ALL_SCHEMA]

    demo = gr.Interface(
        fn=process_video,
        inputs=[video_input, *all_widgets],
        outputs=[
            gr.Textbox(label="实时日志", lines=22, max_lines=22, interactive=False),
            gr.File(label="下载结果 ZIP")
        ],
        title="🎥 AI Video Summary — 视频自动转技术博客",
        description="上传视频 → 自动截帧 · 视觉分析 · 语音转录 · 生成技术博客 → 下载 ZIP\n\n**视频内容信息（全部选填）** — 若留空，标题与日期将由大模型从视频文件名中自动提取。",
        flagging_mode="never",
        submit_btn="▶ 开始处理",
    )
    return demo


def main() -> None:
    demo = build_ui()
    # Queue is automatically enabled for generator functions in Gradio 4+,
    # but explicitly defining it ensures robust SSE streaming.
    demo.queue()
    import os
    server_port = int(os.environ.get("PORT", 7860))
    inbrowser = os.environ.get("IN_BROWSER", "True").lower() == "true"
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=False,
        inbrowser=inbrowser,
        theme=gr.themes.Ocean(),
        mcp_server=True
    )


if __name__ == "__main__":
    main()
