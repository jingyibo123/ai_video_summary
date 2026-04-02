import os
import yaml
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

class VLMConfig(BaseModel):
    base_url: str = Field(default="http://198.18.0.1:1234/v1", description="VLM API base URL")
    model: str = Field(default="qwen2-vl-7b", description="VLM model name")

class ASRConfig(BaseModel):
    model_size: str = Field(default="base", description="Whisper model size or API model name")
    api_base: Optional[str] = Field(default=None, description="OpenAI-compatible ASR API base URL")
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

class AppConfig(BaseModel):
    vlm: VLMConfig = VLMConfig()
    asr: ASRConfig = ASRConfig()
    cv: CVConfig = CVConfig()
    context: ProjectContext = ProjectContext()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "AppConfig":
        if not os.path.exists(yaml_path):
            return cls()
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        
        # Flattened logic for backward compatibility or simple config.yaml
        # If the YAML is flat (like the old context.yaml), we put it into 'context'
        if "meeting_title" in data and "vlm" not in data:
            return cls(context=ProjectContext(**data))
        
        return cls(**data)

    def save_template(self, yaml_path: str):
        # Create a template with comments and no sensitive info
        template_data = self.model_dump()
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(template_data, f, allow_unicode=True, sort_keys=False)
