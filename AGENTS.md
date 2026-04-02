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
All LLM/VLM interactions MUST use the `.parse()` method with Pydantic models.
- **Valid**: `client.beta.chat.completions.parse(..., response_format=Model)`
- **Forbidden**: Raw string manipulation or manual JSON regex extraction.

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

## 📐 Coding Standards for Agents
- **Internal Imports**: Always use relative imports (`from . import ...`) within the package.
- **High Density**: Keep logic inside functions. Inline helpers if they are small and single-use.
- **Type Safety**: PEP 484 Type Hints are MANDATORY for all functions.
- **Resilience**: Every AI call must use `@retry(stop=stop_after_attempt(3), ...)` via `tenacity`.
- **Docs**: Google-style docstrings for all top-level functions.

---

## 🚦 Navigation
Refer to this file as the **Universal Truth**. If a tool-specific config (`.clinerules`, `.cursorrules`) conflicts with this file, **FOLLOW THIS FILE**.
