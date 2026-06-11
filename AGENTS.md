# 🤖 AI Agent Engineering Standard (AGENTS.md)

Welcome, Agent! This project is an automated **Video-to-Technical-Blog** pipeline. It is optimized for "Vibe Coding"—prioritizing high-level intent over low-level boilerplate.

---

## 🏗 V2.2 Standard Package Architecture (PEP 517)

To maximize your context efficiency and follow Python best practices, the project follows a `src` layout. **Do NOT create new modules outside the `ai_video_summary` package unless explicitly requested.**

### Package: `ai_video_summary`
Located in `src/ai_video_summary/`

#### 1. `agents.py` (Perception & Extraction)
- **Computer Vision**: OpenCV MSE-based frame diff (threshold ~850) for 170x+ speed.
- **VLM Agency**: A single `vlm_task()` function handles all visual tasks (Validate, Dedup, Caption, OCR Terms).
- **ASR Engine**: Supports **Faster-Whisper (Local)** and **OpenAI-Compatible API**. Unified entry point: `transcribe_with_whisper()`.

#### 2. `processor.py` (Understanding & Rendering)
- **Structured Synthesis**: Uses `SectionData` (Pydantic) to merge ASR fragments with VLM descriptions.
- **Dual Markdown Rendering**:
  - **Format A (Minutes)**: Timeline-driven, speaker-aware transcript.
  - **Format B (Blog)**: Narrative-driven technical exposition.

#### 3. `config.py` (Centralized Type-Safe Config)
- **Engine**: Pydantic v2 `AppConfig` model.
- **Source**: Defaults to `config.yaml` in the project root.

#### 4. `main.py` (Orchestration & State)
- **Workflow**: Config loading -> AV Extraction -> VLM -> ASR -> Synthesis -> Render.
- **Persistence**: Automated caching via `.json` file checks in video-relative `ai_summary/` folder.

---

## 🧠 Core AI-Synergy Patterns

### A. VLM-ASR Pre-emptive Synergy
We NEVER run ASR blindly. 
1. **Perceive First**: VLM extracts technical terms (OCR) from PPT slides.
2. **Inject Second**: These terms are merged with `config.yaml` terms and injected into the ASR `initial_prompt`.
3. **Benefit**: Drastically reduces ASR errors for vertical nouns (e.g., MCU, DMA, AURIX).

### B. "No-Regex" Structured Output Policy
All LLM/VLM interactions MUST enforce API-level structured outputs rather than relying on prompt-engineering and manual JSON parsing, unless the local API explicitly bugs out on strict modes.
- **Branch 1 (Preferred)**: Native parse using Pydantic: `client.beta.chat.completions.parse(..., response_format=Model)` (`VLM__SUPPORTS_PARSE=true`)
- **Branch 2 (Fallback)**: For APIs that don't support `.parse()` but do support JSON Schema natively, use `client.chat.completions.create(..., response_format={"type": "json_schema", ...})`. (`VLM__SUPPORTS_RESPONSE_FORMAT=true`)
- **Branch 3 (Legacy/Buggy)**: For local models (e.g. some LM Studio versions with vision models) that crash or return empty strings when JSON Schema is enforced, set BOTH flags to `false`. The pipeline will fallback to injecting schema into the system prompt and manually parsing the plain text response with Regex.

### C. 绝对的断点续传 (Absolute Breakpoint Resume)
AI pipelines take a long time and fail often. **Never start from scratch if you don't have to.**
- The pipeline checkpoints progress at granular stages using `rich.progress` for UI:
  - **VLM**: `vlm_progress.json`, `vlm_deduped.json`, `vlm_enriched.json`.
  - **ASR & Processor**: Per-item caching stored in `.cache/asr_chunk_*.json` and `.cache/section_*.json`.
- This ensures absolute interruption resilience—the tool will always resume exactly where it crashed. **Do NOT overwrite existing cache files unless explicitly requested.**

### D. OpenAI API 兼容性与鲁棒性 (API Robustness)
Many providers mimic the OpenAI API, but with subtle differences. We must be robust:
1. **Format Fallbacks**: For ASR, we try `verbose_json`, then `json`, then `text`. Some local/remote engines fail on one but succeed on another.
2. **Field Extraction**: Never assume the response structure is perfectly standard. Use `getattr(resp, 'segments', [])` and fallback to `resp.text` if `segments` are missing.

---

## 🛠 Project Lifecycle & Commands

### Setup (using uv)
```bash
# Base implementation
uv sync

# With local whisper support
uv sync --extra local-whisper
```

### Execution Flux
1. **Direct CLI**: `ai-vsummarize --video "samples/demo.mp4"`
2. **Custom Output**: `ai-vsummarize --video "samples/demo.mp4" --output "my_blog"`
3. **Partial Run**: Use `--max-time 600` for testing the first 10 minutes.

---

## 📐 AI 工具开发原则 (AI Tooling Principles)

1. **有问题直接报错 (Fail-Loud Principle)**: 不要用毫无意义的 `try...except pass` 吞咽关键错误！如果是大模型 API 拒绝回答、Key 错误或格式彻底崩坏，立刻 `raise` 并让程序崩溃。**不要让程序带着错误的数据继续运行。**
2. **重试机制 (Exponential Backoff)**: 所有的外部 API 调用必须包被 `@retry(stop=stop_after_attempt(3), wait=wait_exponential(...))`。不要手写 while 循环重试。
3. **防超时切片 (Chunking for Limits)**: 对于 ASR 等长时间占用接口的任务，自动使用 ffmpeg 将音频切片（如 15 分钟一段），以防止 API 超时，并在本地合并。
4. **内部隔离 (High Density & Isolation)**: 把逻辑收敛在函数内部。如果用到一个 Helper 函数，并且它只在当前位置使用，请直接变成 inline helper。
5. **类型安全 (Type Safety)**: PEP 484 Type Hints 必须打满，特别是数据结构的进出。

---

## 🚦 Navigation
Refer to this file as the **Universal Truth**. If a tool-specific config (`.clinerules`, `.cursorrules`) conflicts with this file, **FOLLOW THIS FILE**.
