"""Extensible multi-provider LLM client layer (Anthropic, OpenAI-compat, Gemini, Ollama local).

Zero Streamlit dependencies: designed to grow cleanly into CLI, API, or plugin backends.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import requests

from ai.security import redact_secrets, validate_endpoint_url


@dataclass(frozen=True)
class ProviderPreset:
    """Metadata describing a supported LLM service provider."""

    id: str
    label: str
    protocol: str  # "anthropic" | "openai_compat"
    default_base_url: str
    default_model: str
    suggested_models: tuple[str, ...]
    is_local: bool = False
    requires_api_key: bool = True
    notes: str = ""


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "ollama": ProviderPreset(
        id="ollama",
        label="Ollama (Local — sin envío a terceros)",
        protocol="openai_compat",
        default_base_url="http://localhost:11434/v1",
        default_model="gemma4-ajustado:latest",
        suggested_models=("gemma4-ajustado:latest", "qwen2.5:7b", "llama3.1:8b"),
        is_local=True,
        requires_api_key=False,
        notes="Corre 100% en tu equipo (solo APP_MODE=local). Detecta modelos instalados automáticamente.",
    ),
    "lmstudio": ProviderPreset(
        id="lmstudio",
        label="LM Studio / vLLM (Local OpenAI-compat)",
        protocol="openai_compat",
        default_base_url="http://localhost:1234/v1",
        default_model="local-model",
        suggested_models=("local-model",),
        is_local=True,
        requires_api_key=False,
        notes="Servidor local compatible con OpenAI (solo APP_MODE=local).",
    ),
    "anthropic": ProviderPreset(
        id="anthropic",
        label="Anthropic Claude (BYOK)",
        protocol="anthropic",
        default_base_url="https://api.anthropic.com",
        default_model="claude-3-5-haiku-latest",
        suggested_models=("claude-3-5-haiku-latest", "claude-3-7-sonnet-latest", "claude-sonnet-4-20250514"),
        is_local=False,
        requires_api_key=True,
        notes="Requiere API key propia de Anthropic Console (las suscripciones Pro no incluyen créditos API).",
    ),
    "gemini": ProviderPreset(
        id="gemini",
        label="Google Gemini (BYOK)",
        protocol="openai_compat",
        default_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_model="gemini-2.5-flash",
        suggested_models=("gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"),
        is_local=False,
        requires_api_key=True,
        notes="Usa el endpoint oficial OpenAI-compatible de Google AI Studio (API key BYOK).",
    ),
    "openai": ProviderPreset(
        id="openai",
        label="OpenAI (BYOK)",
        protocol="openai_compat",
        default_base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        suggested_models=("gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"),
        is_local=False,
        requires_api_key=True,
        notes="API oficial de OpenAI (API key BYOK).",
    ),
    "openrouter": ProviderPreset(
        id="openrouter",
        label="OpenRouter (BYOK — multimodelo)",
        protocol="openai_compat",
        default_base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-2.5-flash",
        suggested_models=("google/gemini-2.5-flash", "anthropic/claude-3.5-haiku", "openai/gpt-4o-mini"),
        is_local=False,
        requires_api_key=True,
        notes="Acceso unificado a múltiples proveedores con una sola API key.",
    ),
    "groq": ProviderPreset(
        id="groq",
        label="Groq (BYOK — alta velocidad)",
        protocol="openai_compat",
        default_base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        suggested_models=("llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
        is_local=False,
        requires_api_key=True,
        notes="Inferencia rápida vía API OpenAI-compatible.",
    ),
    "custom_openai": ProviderPreset(
        id="custom_openai",
        label="Otro proveedor (OpenAI-compatible personalizado)",
        protocol="openai_compat",
        default_base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        suggested_models=(),
        is_local=False,
        requires_api_key=True,
        notes="Compatible con DeepSeek, Mistral, Together, Fireworks o cualquier endpoint OpenAI HTTPS.",
    ),
}


@dataclass
class ProviderConfig:
    """Runtime configuration for an LLM provider; masks api_key in repr."""

    provider_id: str
    model: str
    api_key: str = field(default="", repr=False)
    base_url: str = ""
    temperature: float = 0.2
    max_tokens: int = 1200
    timeout_s: float = 60.0
    disable_thinking: bool = False

    def __repr__(self) -> str:
        return (
            f"ProviderConfig(provider_id={self.provider_id!r}, model={self.model!r}, "
            f"base_url={self.base_url!r}, api_key='***')"
        )


@dataclass
class ToolCallRequest:
    """Normalized tool invocation requested by an LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalized response returned by any LLM provider."""

    text: str
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    raw_assistant_message: dict[str, Any] | None = None


def list_available_presets(app_mode: str = "local") -> list[ProviderPreset]:
    """Return provider presets permitted under the active APP_MODE."""
    presets = list(PROVIDER_PRESETS.values())
    if app_mode != "local":
        presets = [p for p in presets if not p.is_local]
    return presets


def list_ollama_models(
    base_url: str = "http://localhost:11434",
    *,
    app_mode: str = "local",
    timeout_s: float = 2.5,
) -> list[str]:
    """Query a local Ollama daemon (`/api/tags`) and return installed model names."""
    if app_mode != "local":
        return []
    clean_root = base_url.rstrip("/").removesuffix("/v1")
    validated = validate_endpoint_url(clean_root, app_mode=app_mode, allow_local=True)
    try:
        resp = requests.get(f"{validated}/api/tags", timeout=timeout_s)
        resp.raise_for_status()
        payload = resp.json()
        return [m["name"] for m in payload.get("models", []) if "name" in m]
    except Exception:  # noqa: BLE001
        return []


class BaseLLMProvider(ABC):
    """Abstract base class so new providers can be plugged in without touching agent logic."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @abstractmethod
    def chat(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Execute a single chat completion turn, optionally with tool schemas."""


class OpenAICompatibleProvider(BaseLLMProvider):
    """Supports OpenAI, Gemini, OpenRouter, Groq, DeepSeek, Ollama, LM Studio, and custom URLs."""

    def __init__(self, config: ProviderConfig, *, validated_base_url: str) -> None:
        super().__init__(config)
        self.validated_base_url = validated_base_url

    def chat(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        from openai import OpenAI

        api_key = self.config.api_key or "local-no-key-required"
        client = OpenAI(
            api_key=api_key,
            base_url=self.validated_base_url,
            timeout=self.config.timeout_s,
        )

        full_messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}, *messages]
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": full_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"],
                    },
                }
                for t in tools
            ]
            kwargs["tool_choice"] = "auto"

        if self.config.disable_thinking and self.config.provider_id == "ollama":
            kwargs["extra_body"] = {"think": False}

        try:
            completion = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            safe_msg = redact_secrets(str(exc), (self.config.api_key,))
            raise RuntimeError(f"Error del proveedor `{self.config.provider_id}`: {safe_msg}") from None

        choice = completion.choices[0].message
        text = (choice.content or "").strip()
        if not text:
            # Thinking models (e.g. Gemma 4, DeepSeek-R1) may populate reasoning_content/reasoning
            # when max_tokens is reached before final output.
            fallback_reasoning = (
                getattr(choice, "reasoning_content", None)
                or getattr(choice, "reasoning", None)
                or ""
            )
            if isinstance(fallback_reasoning, str) and fallback_reasoning.strip():
                text = fallback_reasoning.strip()
        tool_calls: list[ToolCallRequest] = []
        raw_tc_list: list[dict[str, Any]] = []

        for tc in getattr(choice, "tool_calls", None) or []:
            raw_args = tc.function.arguments or "{}"
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                parsed_args = {}
            tool_calls.append(
                ToolCallRequest(id=str(tc.id), name=str(tc.function.name), arguments=parsed_args)
            )
            raw_tc_list.append(
                {
                    "id": str(tc.id),
                    "type": "function",
                    "function": {"name": str(tc.function.name), "arguments": raw_args},
                }
            )

        raw_msg: dict[str, Any] = {"role": "assistant", "content": text}
        if raw_tc_list:
            raw_msg["tool_calls"] = raw_tc_list

        return LLMResponse(text=text, tool_calls=tool_calls, raw_assistant_message=raw_msg)


class AnthropicProvider(BaseLLMProvider):
    """Native Anthropic Claude Messages API provider with structured tool_use."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)

    def chat(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        from anthropic import Anthropic

        client = Anthropic(api_key=self.config.api_key, timeout=self.config.timeout_s)
        anth_messages = _convert_messages_to_anthropic(messages)
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "system": system_prompt,
            "messages": anth_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["parameters"],
                }
                for t in tools
            ]

        try:
            resp = client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            safe_msg = redact_secrets(str(exc), (self.config.api_key,))
            raise RuntimeError(f"Error de Anthropic: {safe_msg}") from None

        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        raw_blocks: list[dict[str, Any]] = []

        for block in resp.content:
            btype = getattr(block, "type", "")
            if btype == "text":
                text_parts.append(block.text)
                raw_blocks.append({"type": "text", "text": block.text})
            elif btype == "tool_use":
                args = dict(block.input) if isinstance(block.input, dict) else {}
                tool_calls.append(ToolCallRequest(id=str(block.id), name=str(block.name), arguments=args))
                raw_blocks.append(
                    {"type": "tool_use", "id": str(block.id), "name": str(block.name), "input": args}
                )

        return LLMResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            raw_assistant_message={"role": "assistant", "content": raw_blocks},
        )


def _convert_messages_to_anthropic(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert normalized OpenAI-style tool loop messages into Anthropic Messages format."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg["role"]
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": msg["tool_call_id"],
                "content": str(msg["content"]),
            }
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif role == "assistant" and isinstance(msg.get("content"), list):
            out.append({"role": "assistant", "content": msg["content"]})
        elif role == "user" and isinstance(msg.get("content"), list):
            # Support optional vision image blocks alongside text
            anth_blocks: list[dict[str, Any]] = []
            for part in msg["content"]:
                if part.get("type") == "text":
                    anth_blocks.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:image/png;base64,"):
                        anth_blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": url.split(",", 1)[1],
                                },
                            }
                        )
            out.append({"role": "user", "content": anth_blocks})
        else:
            out.append({"role": role, "content": str(msg.get("content", ""))})
    return out


def create_provider(config: ProviderConfig, *, app_mode: str = "local") -> BaseLLMProvider:
    """Validate security policies and instantiate the requested LLM provider."""
    preset = PROVIDER_PRESETS.get(config.provider_id)
    if preset is None:
        raise ValueError(f"Proveedor no soportado: `{config.provider_id}`.")

    if preset.is_local and app_mode != "local":
        raise PermissionError(
            f"El proveedor local `{preset.label}` no está disponible en APP_MODE={app_mode}. "
            "Usa un proveedor BYOK externo en Hugging Face Space."
        )

    if not config.model.strip():
        raise ValueError("Debes especificar el nombre del modelo.")

    if preset.requires_api_key and not config.api_key.strip():
        raise ValueError(f"El proveedor `{preset.label}` requiere tu API key (BYOK).")

    if preset.protocol == "anthropic":
        return AnthropicProvider(config)

    base_url = config.base_url.strip() or preset.default_base_url
    validated = validate_endpoint_url(base_url, app_mode=app_mode, allow_local=preset.is_local)
    return OpenAICompatibleProvider(config, validated_base_url=validated)
