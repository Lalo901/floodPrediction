"""Modular AI assistant package for Flood Mapper MVP (UI-agnostic, extensible, secure)."""

from ai.agent import AssistantReply, FloodAssistant, ToolTraceItem
from ai.providers import (
    PROVIDER_PRESETS,
    BaseLLMProvider,
    LLMResponse,
    ProviderConfig,
    ProviderPreset,
    ToolCallRequest,
    create_provider,
    list_available_presets,
    list_ollama_models,
)
from ai.security import redact_secrets, sanitize_user_prompt, validate_endpoint_url
from ai.tools import (
    AnalysisContext,
    ToolRegistry,
    ToolSpec,
    build_default_tool_registry,
    list_tool_calls,
    log_tool_call,
)

__all__ = [
    "PROVIDER_PRESETS",
    "AnalysisContext",
    "AssistantReply",
    "BaseLLMProvider",
    "FloodAssistant",
    "LLMResponse",
    "ProviderConfig",
    "ProviderPreset",
    "ToolCallRequest",
    "ToolRegistry",
    "ToolSpec",
    "ToolTraceItem",
    "build_default_tool_registry",
    "create_provider",
    "list_available_presets",
    "list_ollama_models",
    "list_tool_calls",
    "log_tool_call",
    "redact_secrets",
    "sanitize_user_prompt",
    "validate_endpoint_url",
]
