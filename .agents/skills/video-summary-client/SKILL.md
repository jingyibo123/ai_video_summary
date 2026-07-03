---
name: video-summary-client
description: Call the video summary API to process local video files, stream real-time logs, and download/extract the output ZIP. Use this skill when the user asks to summarize a video using a running backend API.
---
# Video Summary Client Skill

This skill allows the agent to call a running `ai-video-summary` Gradio API server to process a local video file, print real-time incremental logs to stdout, download the generated ZIP, and unzip its contents to the target directory.

## Prerequisite
1. The `ai-video-summary` Gradio service must be running locally or remotely (e.g. at `http://127.0.0.1:7860`).
2. If using `uv run`, dependencies (including `gradio_client`) are automatically resolved and installed in an ephemeral sandbox environment at runtime (via PEP 723 metadata).

## Usage
Run the following script using the `run_command` tool in unbuffered mode (`-u`) to capture stdout logs dynamically:
```bash
uv run -u .agents/skills/video-summary-client/scripts/run_video_summary_client.py \
  --api-url <API_URL> \
  --video <VIDEO_PATH> \
  --output-dir <OUTPUT_DIR> \
  [--meeting-title <TITLE>] \
  [--date <YYYY-MM-DD>] \
  [--attendees <ATTENDEES_LIST>] \
  [--agenda <AGENDA_LIST>] \
  [--custom-terms <TERMS_LIST>]
```

## Options
* `--api-url`: The endpoint of the Gradio application (e.g. `http://127.0.0.1:7860/`).
* `--video`: Absolute or relative path to the local video file.
* `--output-dir`: The directory where the final ZIP result will be extracted.
* `--meeting-title`: (Optional) Meeting/video title (if omitted, inferred by LLM).
* `--date`: (Optional) YYYY-MM-DD format (if omitted, inferred from filename).
* `--attendees`: (Optional) Comma-separated list of attendees.
* `--agenda`: (Optional) Comma-separated list of agenda items.
* `--custom-terms`: (Optional) Comma-separated list of technical terms to inject into the ASR prompt.
