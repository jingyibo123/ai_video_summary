# 📹 AI Conference Summary: 视频转技术博客全自动流水线 (V2.2)

这是一个基于多模态 AI (VLM + ASR + LLM) 的全自动视频转换工具。它能毫秒级定位关键帧、自动提取视觉热词、并利用 ASR 与 LLM 协同提纯出排版精美的技术博客与会议纪要。

## 🌟 核心亮点

-   **跨模态词汇注入**: VLM 自动从幻灯片中抓取技术术语，动态注入 ASR 引导词空间，显著消除 ASR 对专业名词的语焉不详。
-   **结构化输出引擎**: 全链接入 OpenAI **Structured Outputs** 协议，基于 Pydantic 强约定 Schema，提供生产级的稳定性。
-   **百倍速 CV 初筛**: 采用高效帧差 MSE 算法，万帧视频可在秒级完成候选帧切分。
-   **标准化 src 布局**: 遵循 Python 最佳实践，提供便捷的 CLI 入口。
-   **全环境适配**: 支持**全本地 ASR (Faster-Whisper)** 与 **OpenAI 兼容 API** 混合调用，无惧断网或资源受限。

## 🏗 架构模块 (Standard Package)

1.  **`ai_video_summary.agents`**: 感知层。封装 CV 跳帧、ASR 转录与统一的 `vlm_task`。
2.  **`ai_video_summary.processor`**: 理解层。LLM 数据语义合成与双格式 Markdown 渲染。
3.  **`ai_video_summary.config`**: 配置层。基于 Pydantic 的类型安全配置管理。
4.  **`ai_video_summary.main`**: 调度层。全局工作流调度、配置加载与持久化控制。

## 🚀 快速开始

### 1. 环境准备 (使用 uv)
```bash
# 安装基础依赖
uv sync

# 如果需要本地 Whisper 支持
uv sync --extra local-whisper
```
*注：需确保 `ffmpeg` 已安装。*

### 2. 配置说明

首先，复制环境配置模板并填入你的 API 密钥：
```bash
cp .env.example .env
```
（编辑 `.env` 文件，填入 VLM 和 ASR 的 API 地址、模型名称和密钥）

然后，编辑 `context.yaml` 填入会议背景信息：
```yaml
context:
  meeting_title: "英飞凌 MCU 技术培训"
  attendees: ["英飞凌讲师", "学生"]
  custom_terms: ["AURIX", "TriCore", "DMA"]
```

### 3. 一键运行
```bash
# 使用内置 CLI 命令
ai-vsummarize --video "samples/demo.mp4"

# 限制处理时间 (仅测试前 10 分钟)
ai-vsummarize --video "samples/demo.mp4" --max-time 600
```

## 📂 交付物说明
运行后将在视频所在目录下生成 `ai_summary/` 文件夹：
- `ai_summary/format_a_minutes.md`: **实录纪要** (时间轴驱动)。
- `ai_summary/format_b_blog.md`: **深度博文** (叙事驱动)。
- `ai_summary/config.yaml`: **配置备份** (用于复现)。

