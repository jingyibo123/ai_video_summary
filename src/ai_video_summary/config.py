"""
应用配置层 (Centralized Config).

职责分层：
- `.env` / 环境变量：API 地址、模型名、密钥（敏感信息，不入库）
- `context.yaml`：会议元信息、自定义术语（项目专属，可入库）

Pydantic-Settings 自动处理 .env 读取与环境变量映射，
使用双下划线分隔符映射嵌套字段（如 VLM__BASE_URL → config.vlm.base_url）。
"""

from pathlib import Path
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class VLMConfig(BaseModel):
    base_url: str = Field(default="http://localhost:1234/v1", description="VLM API base URL")
    model: str = Field(default="qwen2-vl-7b", description="VLM model name")
    api_key: str = Field(default="none", description="VLM API key")
    supports_parse: bool = Field(default=True, description="Whether the VLM API supports beta.chat.completions.parse")
    supports_response_format: bool = Field(default=True, description="Whether the VLM API supports response_format={'type': 'json_schema'}")
    max_workers: int = Field(default=1, description="Max concurrent workers for VLM API requests")
    disable_thinking: bool = Field(default=False, description="Whether to disable thinking/reasoning process (e.g. for DeepSeek/Qwen APIs)")
    max_thinking_tokens: Optional[int] = Field(default=None, description="Max thinking/reasoning tokens for VLM")


class LLMConfig(BaseModel):
    model: str = Field(default="qwen2.5-7b-instruct", description="LLM model name for text tasks")
    base_url: Optional[str] = Field(default=None, description="LLM API base URL (falls back to VLM base URL if None)")
    api_key: Optional[str] = Field(default=None, description="LLM API key (falls back to VLM API key if None)")
    supports_parse: bool = Field(default=True, description="Whether the LLM API supports beta.chat.completions.parse")
    supports_response_format: bool = Field(default=True, description="Whether the LLM API supports response_format={'type': 'json_schema'}")
    max_workers: int = Field(default=1, description="Max concurrent workers for LLM API requests")
    disable_thinking: bool = Field(default=False, description="Whether to disable thinking/reasoning process (e.g. for DeepSeek/Qwen APIs)")
    max_thinking_tokens: Optional[int] = Field(default=None, description="Max thinking/reasoning tokens for LLM")


class ASRConfig(BaseModel):
    model_size: str = Field(default="whisper-1", description="ASR model name (API) or size (local)")
    api_base: Optional[str] = Field(default="http://localhost:8000/v1", description="Whisper-compatible ASR API base URL; set to null to use local Faster-Whisper")
    api_key: str = Field(default="none", description="ASR API key")
    local_device: str = Field(default="cpu", description="Device for local Whisper (cpu/cuda)")
    local_compute_type: str = Field(default="int8", description="Compute type for local Whisper")
    chunk_length_s: int = Field(default=900, description="Chunk length in seconds for ASR API (to avoid timeouts on long files)")
    max_workers: int = Field(default=4, description="Max concurrent workers for ASR API requests")


class CVConfig(BaseModel):
    diff_threshold: int = Field(default=850, description="MSE threshold for frame difference")
    target_size: Tuple[int, int] = Field(default=(256, 144), description="Image size for comparison")
    sample_interval: float = Field(default=1.0, description="Frame sampling interval in seconds")

class RetryConfig(BaseModel):
    max_attempts: int = Field(default=3, description="Maximum number of retry attempts")
    multiplier: float = Field(default=1.5, description="Multiplier for exponential backoff")
    min_seconds: float = Field(default=2.0, description="Minimum wait time in seconds")
    max_seconds: float = Field(default=10.0, description="Maximum wait time in seconds")


class ProjectContext(BaseModel):
    meeting_title: str = Field(default="会议纪要", description="Meeting title")
    date: str = Field(default="未知", description="Meeting date")
    location: str = Field(default="无", description="Meeting location")
    attendees: List[str] = Field(default_factory=list, description="List of attendees")
    agenda: List[str] = Field(default_factory=list, description="Meeting agenda")
    custom_terms: List[str] = Field(default_factory=list, description="Custom terms for ASR prompt")

    @field_validator("attendees", "agenda", "custom_terms", mode="before")
    @classmethod
    def _none_to_empty_list(cls, v: object) -> object:
        """YAML 中全部注释掉的列表会被解析为 None，此处自动转为空列表。"""
        return v if v is not None else []


class AppConfig(BaseSettings):
    """
    顶级配置模型，自动从 .env 和环境变量加载。
    env_nested_delimiter="__" 使得 VLM__BASE_URL 自动映射 to vlm.base_url。
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    vlm: VLMConfig = VLMConfig()
    llm: LLMConfig = LLMConfig()
    asr: ASRConfig = ASRConfig()
    cv: CVConfig = CVConfig()
    retry: RetryConfig = RetryConfig()
    context: ProjectContext = ProjectContext()

    def get_llm_base_url(self) -> str:
        return self.llm.base_url if self.llm.base_url else self.vlm.base_url

    def get_llm_api_key(self) -> str:
        return self.llm.api_key if self.llm.api_key else self.vlm.api_key

    @classmethod
    def load(cls) -> "AppConfig":
        """
        加载配置：BaseSettings 自动从 .env 和环境变量读取，
        context 字段通过 CLI 参数或 Gradio UI 输入提供。
        """
        config = cls()
        set_global_config(config)
        return config


# Global Configuration Singleton reference for tenacity integration
_global_config: Optional["AppConfig"] = None

def get_global_config() -> Optional["AppConfig"]:
    return _global_config

def set_global_config(config: "AppConfig") -> None:
    global _global_config
    _global_config = config

def dynamic_stop(retry_state) -> bool:
    cfg = get_global_config()
    max_attempts = cfg.retry.max_attempts if (cfg and cfg.retry) else 3
    return retry_state.attempt_number >= max_attempts

def dynamic_wait(retry_state) -> float:
    cfg = get_global_config()
    if cfg and cfg.retry:
        multiplier = cfg.retry.multiplier
        min_val = cfg.retry.min_seconds
        max_val = cfg.retry.max_seconds
    else:
        multiplier = 1.5
        min_val = 2.0
        max_val = 10.0
    
    attempt = retry_state.attempt_number
    delay = multiplier * (2.0 ** (attempt - 1))
    return min(max(delay, min_val), max_val)
