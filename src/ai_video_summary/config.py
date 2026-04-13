"""
应用配置层 (Centralized Config).

职责分层：
- `.env` / 环境变量：API 地址、模型名、密钥（敏感信息，不入库）
- `context.yaml`：会议元信息、自定义术语（项目专属，可入库）

Pydantic-Settings 自动处理 .env 读取与环境变量映射，
使用双下划线分隔符映射嵌套字段（如 VLM__BASE_URL → config.vlm.base_url）。
"""

import os
import yaml
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VLMConfig(BaseModel):
    base_url: str = Field(default="http://localhost:1234/v1", description="VLM API base URL")
    model: str = Field(default="qwen2-vl-7b", description="VLM model name")
    api_key: str = Field(default="none", description="VLM API key")


class ASRConfig(BaseModel):
    model_size: str = Field(default="whisper-1", description="ASR model name (API) or size (local)")
    api_base: Optional[str] = Field(default="http://localhost:8000/v1", description="Whisper-compatible ASR API base URL; set to null to use local Faster-Whisper")
    api_key: str = Field(default="none", description="ASR API key")
    local_device: str = Field(default="cpu", description="Device for local Whisper (cpu/cuda)")
    local_compute_type: str = Field(default="int8", description="Compute type for local Whisper")


class CVConfig(BaseModel):
    diff_threshold: int = Field(default=850, description="MSE threshold for frame difference")
    target_size: Tuple[int, int] = Field(default=(256, 144), description="Image size for comparison")


class ProjectContext(BaseModel):
    meeting_title: str = Field(default="会议纪要", description="Meeting title")
    date: str = Field(default="未知", description="Meeting date")
    location: str = Field(default="无", description="Meeting location")
    attendees: List[str] = Field(default_factory=list, description="List of attendees")
    agenda: List[str] = Field(default_factory=list, description="Meeting agenda")
    custom_terms: List[str] = Field(default_factory=list, description="Custom terms for ASR prompt")


class AppConfig(BaseSettings):
    """
    顶级配置模型，自动从 .env 和环境变量加载。
    env_nested_delimiter="__" 使得 VLM__BASE_URL 自动映射到 vlm.base_url。
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    vlm: VLMConfig = VLMConfig()
    asr: ASRConfig = ASRConfig()
    cv: CVConfig = CVConfig()
    context: ProjectContext = ProjectContext()

    @classmethod
    def load(cls, context_yaml: str = "context.yaml") -> "AppConfig":
        """
        加载配置：先由 BaseSettings 自动从 .env 和环境变量填充，
        再从 context.yaml 覆盖会议元信息部分。

        Args:
            context_yaml: 会议上下文 YAML 文件路径（仅含 context 字段）。

        Returns:
            完整填充的 AppConfig 实例。
        """
        config = cls()  # BaseSettings 在此自动读取 .env 和环境变量
        if os.path.exists(context_yaml):
            with open(context_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            ctx_data = data.get("context", data)  # 兼容带/不带顶层 context 键的格式
            config = config.model_copy(update={"context": ProjectContext(**ctx_data)})
        return config
