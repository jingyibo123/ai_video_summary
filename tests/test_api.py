import os
import sys
import logging
import time

# 将 src 目录加入 sys.path，以便导入 ai_video_summary 模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ai_video_summary.config import AppConfig
from ai_video_summary.agents import vlm_task, transcribe_with_whisper

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def create_fake_audio(filepath, duration_s=2, sample_rate=16000):
    import wave
    import struct
    import math
    if os.path.exists(filepath):
        return
    with wave.open(filepath, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        # 生成一段 440Hz 的正弦波
        for i in range(int(duration_s * sample_rate)):
            value = int(32767.0 * math.cos(440.0 * math.pi * float(i) / float(sample_rate)))
            data = struct.pack('<h', value)
            f.writeframesraw(data)

def test_api():
    logger.info("============== 开始测试 API (VLM & ASR) ==============")
    # 1. 加载配置 (这会自动读取项目根目录的 .env 文件)
    config = AppConfig.load()
    
    logger.info(f"当前 VLM 配置信息:")
    logger.info(f"  Base URL: {config.vlm.base_url}")
    logger.info(f"  Model: {config.vlm.model}")
    logger.info(f"  Supports Parse (.parse): {config.vlm.supports_parse}")
    logger.info(f"  Supports Response Format (json_schema): {config.vlm.supports_response_format}")
    
    # 2. 准备图片路径
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    img1 = os.path.join(data_dir, "00-22-14_00-22-50.jpg")
    img2 = os.path.join(data_dir, "00-22-52_00-23-03.jpg")
    
    if not os.path.exists(img1) or not os.path.exists(img2):
        logger.error(f"找不到测试图片，请确保 {img1} 和 {img2} 存在。")
    else:
        try:
            # 3. 测试单图处理 (Validate) - 检验模型是否支持基础图片读取以及结构化输出
            logger.info("\n>>> [测试 1/3] 单图结构化感知能力 (Validate)")
            t0 = time.time()
            is_slide, reasoning = vlm_task(
                base_url=config.vlm.base_url,
                api_key=config.vlm.api_key,
                model=config.vlm.model,
                task_type="validate",
                images=[img1],
                supports_parse=config.vlm.supports_parse,
                supports_response_format=config.vlm.supports_response_format
            )
            logger.info(f"✅ 单图测试成功! 耗时: {time.time() - t0:.2f}s")
            logger.info(f"  -> 分析结果 (是否为幻灯片): {is_slide}")
            logger.info(f"  -> 推理过程: {reasoning}")
            
            # 4. 测试多图比对 (Dedup) - 检验模型是否支持在一个 prompt 中传入多张图片
            logger.info("\n>>> [测试 2/3] 多图复杂比对与去重能力 (Dedup)")
            logger.info("正在验证两张连续帧是否被判定为同一张幻灯片（动画渐进）...")
            t0 = time.time()
            is_same, dedup_reasoning = vlm_task(
                base_url=config.vlm.base_url,
                api_key=config.vlm.api_key,
                model=config.vlm.model,
                task_type="dedup",
                images=[img1, img2],
                supports_parse=config.vlm.supports_parse,
                supports_response_format=config.vlm.supports_response_format
            )
            logger.info(f"✅ 多图测试成功! 耗时: {time.time() - t0:.2f}s")
            logger.info(f"  -> 分析结果 (是否为同一张图): {is_same}")
            logger.info(f"  -> 推理过程: {dedup_reasoning}")
            
            logger.info("\n🎉 VLM 测试通过！")
            
        except Exception as e:
            logger.error(f"\n❌ VLM API 测试失败！错误信息:\n{e}")
            logger.error("\n可能的原因：\n"
                         "1. 模型本身不支持图片输入，或者不支持同时传入多张图片。\n"
                         "2. API 地址 (Base URL) 或 Key 配置有误。\n"
                         "3. 对应模型厂商未完全适配 OpenAI 的严格结构化输出协议，请尝试在 .env 中设置 VLM__SUPPORTS_PARSE=false 以及 VLM__SUPPORTS_RESPONSE_FORMAT=false\n")
    
    # 5. 测试 ASR
    logger.info("\n============== 开始测试 ASR API ==============")
    logger.info(f"当前 ASR 配置信息:")
    logger.info(f"  API Base: {config.asr.api_base}")
    logger.info(f"  Model Size: {config.asr.model_size}")
    
    audio_path = os.path.join(data_dir, "fake_audio.wav")
    logger.info("正在生成模拟音频用于测试...")
    create_fake_audio(audio_path)
    
    try:
        t0 = time.time()
        logger.info("\n>>> [测试 3/3] 音频转录感知能力 (Transcribe)")
        transcript = transcribe_with_whisper(
            audio_path=audio_path,
            prompt="测试音频",
            model_size=config.asr.model_size,
            api_base=config.asr.api_base,
            api_key=config.asr.api_key,
            device=config.asr.local_device,
            compute_type=config.asr.local_compute_type,
            hotwords="测试",
            chunk_length_s=config.asr.chunk_length_s
        )
        logger.info(f"✅ ASR 测试成功! 耗时: {time.time() - t0:.2f}s")
        logger.info(f"  -> 识别到的片段数量: {len(transcript)}")
        if transcript:
            logger.info(f"  -> 片段示例: {transcript[0]}")
            
        logger.info("\n🎉 恭喜！当前配置的 API 可以完美支持本项目所需的视觉和音频处理能力。")
    except Exception as e:
        logger.error(f"\n❌ ASR API 测试失败！错误信息:\n{e}")
        logger.error("\n可能的原因：\n"
                     "1. API 地址或模型名有误。\n"
                     "2. 缺少必要依赖（如本地模式下缺少 faster-whisper）。\n"
                     "3. 该模型后端暂不支持您选择的 ASR 格式（我们在代码中依次尝试了 verbose_json, json, text）。\n")

if __name__ == "__main__":
    test_api()
