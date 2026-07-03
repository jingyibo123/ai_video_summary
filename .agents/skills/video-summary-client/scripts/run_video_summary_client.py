#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gradio_client",
# ]
# ///
"""
Video Summary Client Skill Script
Usage:
    uv run .agents/skills/video-summary-client/scripts/run_video_summary_client.py --api-url http://127.0.0.1:7860 --video path/to/video.mp4 --output-dir path/to/output

This script connects to a running ai-video-summary Gradio app, uploads the video,
streams progress logs in real-time, downloads the result zip, and extracts it to the specified location.
"""

import os
import sys
import argparse
import zipfile
import shutil
from pathlib import Path

# Ensure gradio_client is installed
try:
    import gradio_client
except ImportError:
    print("⚠️  'gradio_client' is not installed. Attempting to install it now...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio_client"])
        import gradio_client
        print("✅ 'gradio_client' successfully installed!")
    except Exception as e:
        print(f"❌ Failed to install 'gradio_client': {e}", file=sys.stderr)
        print("Please run: pip install gradio_client", file=sys.stderr)
        sys.exit(1)

from gradio_client import Client


def get_file_arg(video_path: str):
    """
    Handle passing files to gradio_client across different versions.
    Modern versions (>= 0.2.0) recommend handle_file, older versions take strings.
    """
    try:
        from gradio_client import handle_file
        return handle_file(video_path)
    except ImportError:
        return video_path


def main():
    parser = argparse.ArgumentParser(description="Call ai-video-summary API to process a video and extract results.")
    parser.add_argument("--api-url", required=True, help="API URL of the Gradio service (e.g. http://127.0.0.1:7860)")
    parser.add_argument("--video", required=True, help="Path to the local video file to be processed")
    parser.add_argument("--output-dir", required=True, help="Directory to extract the result ZIP file")
    parser.add_argument("--meeting-title", default="", help="Video/Meeting title (optional)")
    parser.add_argument("--date", default="", help="Date (YYYY-MM-DD) (optional)")
    parser.add_argument("--attendees", default="", help="Attendees separated by commas (optional)")
    parser.add_argument("--agenda", default="", help="Agenda items separated by commas (optional)")
    parser.add_argument("--custom-terms", default="", help="Custom technical terms separated by commas (optional)")
    
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not video_path.exists():
        print(f"❌ Error: Local video file not found at '{video_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"🔌 Connecting to Gradio service at: {args.api_url}...")
    try:
        client = Client(args.api_url)
    except Exception as e:
        print(f"❌ Connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"📤 Preparing file upload for '{video_path.name}'...")
    video_arg = get_file_arg(str(video_path))

    print("🚀 Submitting processing job to the server...")
    try:
        # Submit task using the explicit api_name='/process_video'
        job = client.submit(
            video_arg,
            args.meeting_title,
            args.date,
            args.attendees,
            args.agenda,
            args.custom_terms,
            api_name="/process_video"
        )
    except Exception as e:
        print(f"❌ Failed to submit job: {e}", file=sys.stderr)
        sys.exit(1)

    print("📥 Job started. Streaming real-time logs:\n" + "─" * 60)
    
    last_len = 0
    final_zip_path = None
    
    try:
        for output in job:
            # The output of process_video is a tuple: (log_buffer_string, zip_file_path_or_dict)
            if not isinstance(output, (list, tuple)) or len(output) < 2:
                # If structure is unexpected, print raw output
                print(f"⚠️  Unexpected stream output format: {output}")
                continue
                
            log_buf, zip_data = output[0], output[1]
            
            # Print new log output incrementally
            if log_buf and len(log_buf) > last_len:
                new_logs = log_buf[last_len:]
                print(new_logs, end="", flush=True)
                last_len = len(log_buf)
                
            # Track the zip file output
            if zip_data:
                final_zip_path = zip_data

        # Ensure last remaining logs are printed if any
        status = job.status()
        if status.code.name == "FINISHED":
            print(f"\n{'─' * 60}\n✅ Server processing completed successfully.")
        else:
            print(f"\n{'─' * 60}\n❌ Server job ended with status: {status.code.name}")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error during execution: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract the resulting zip
    zip_file_path = None
    if isinstance(final_zip_path, dict) and "path" in final_zip_path:
        zip_file_path = final_zip_path["path"]
    elif isinstance(final_zip_path, dict) and "name" in final_zip_path:
        zip_file_path = final_zip_path["name"]
    elif isinstance(final_zip_path, str):
        zip_file_path = final_zip_path

    if zip_file_path and os.path.exists(zip_file_path):
        print(f"📦 Found result ZIP at: {zip_file_path}")
        print(f"📂 Extracting files to: {output_dir}")
        try:
            if output_dir.exists():
                print(f"⚠️  Target directory {output_dir} already exists. Merging/Overwriting files.")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            print(f"✨ Extraction complete! All files placed at: {output_dir}")
        except Exception as e:
            print(f"❌ Failed to extract ZIP file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("❌ Error: No output ZIP file was received or found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
