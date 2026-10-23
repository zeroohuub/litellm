"""
Constants and helpers for ChatGPT subscription OAuth.
"""

import os
import platform
from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException

PARAMS_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
LOOSE_MAPPING_ADAPTER: Final[TypeAdapter[Mapping[object, object]]] = TypeAdapter(Mapping[object, object])

# OAuth + API constants (derived from openai/codex)
CHATGPT_AUTH_BASE: Final = "https://auth.openai.com"
CHATGPT_DEVICE_CODE_URL: Final = f"{CHATGPT_AUTH_BASE}/api/accounts/deviceauth/usercode"
CHATGPT_DEVICE_TOKEN_URL: Final = f"{CHATGPT_AUTH_BASE}/api/accounts/deviceauth/token"
CHATGPT_OAUTH_TOKEN_URL: Final = f"{CHATGPT_AUTH_BASE}/oauth/token"
CHATGPT_DEVICE_VERIFY_URL: Final = f"{CHATGPT_AUTH_BASE}/codex/device"
CHATGPT_API_BASE: Final = "https://chatgpt.com/backend-api/codex"
CHATGPT_CLIENT_ID: Final = "app_EMoamEEZ73f0CkXaXp7hrann"

DEFAULT_ORIGINATOR: Final = "codex_cli_rs"
DEFAULT_USER_AGENT: Final = "codex_cli_rs/0.0.0 (Unknown 0; unknown) unknown"
PINNED_CODEX_VERSION: Final = "0.163.0"


class ChatGPTAuthError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        request: httpx.Request | None = None,
        response: httpx.Response | None = None,
        headers: Mapping[str, str] | None = None,
        body: Mapping[str, object] | None = None,
    ):
        super().__init__(  # pyright: ignore[reportUnknownMemberType]  # BaseLLMException annotates dict without type args
            status_code=status_code,
            message=message,
            request=request,
            response=response,
            headers=dict(headers) if headers else None,
            body=dict(body) if body else None,
        )


class GetDeviceCodeError(ChatGPTAuthError):
    pass


class GetAccessTokenError(ChatGPTAuthError):
    pass


class RefreshAccessTokenError(ChatGPTAuthError):
    pass


def _safe_header_value(value: str) -> str:
    if not value:
        return ""
    return "".join(ch if 32 <= ord(ch) <= 126 else "_" for ch in value)


def _sanitize_user_agent_token(value: str) -> str:
    if not value:
        return ""
    return "".join(ch if (ch.isalnum() or ch in "-_./") else "_" for ch in value)


def _env_set(name: str) -> bool:
    return os.getenv(name) is not None


def _env_non_empty(name: str) -> str | None:
    value: Final = os.getenv(name)
    if value is None or not value.strip():
        return None
    return value


def _terminal_user_agent() -> str:
    term_program: Final = _env_non_empty("TERM_PROGRAM")
    if term_program is not None and term_program.lower() != "tmux":
        term_program_version: Final = _env_non_empty("TERM_PROGRAM_VERSION")
        term_program_token: Final = f"{term_program}/{term_program_version}" if term_program_version else term_program
        return _sanitize_user_agent_token(term_program_token) or "unknown"

    if _env_non_empty("GHOSTTY_RESOURCES_DIR"):
        return "Ghostty"

    if _env_set("WEZTERM_VERSION"):
        wezterm_version: Final = _env_non_empty("WEZTERM_VERSION")
        wezterm_token: Final = f"WezTerm/{wezterm_version}" if wezterm_version else "WezTerm"
        return _sanitize_user_agent_token(wezterm_token) or "WezTerm"

    if _env_set("ITERM_SESSION_ID") or _env_set("ITERM_PROFILE") or _env_set("ITERM_PROFILE_NAME"):
        return "iTerm.app"

    if _env_set("KITTY_WINDOW_ID") or "kitty" in (os.getenv("TERM") or ""):
        return "kitty"

    if _env_set("ALACRITTY_SOCKET") or os.getenv("TERM") == "alacritty":
        return "Alacritty"

    if _env_set("KONSOLE_VERSION"):
        konsole_version: Final = _env_non_empty("KONSOLE_VERSION")
        konsole_token: Final = f"Konsole/{konsole_version}" if konsole_version else "Konsole"
        return _sanitize_user_agent_token(konsole_token) or "Konsole"

    if _env_set("GNOME_TERMINAL_SCREEN"):
        return "gnome-terminal"

    if _env_set("VTE_VERSION"):
        vte_version: Final = _env_non_empty("VTE_VERSION")
        vte_token: Final = f"VTE/{vte_version}" if vte_version else "VTE"
        return _sanitize_user_agent_token(vte_token) or "VTE"

    if _env_set("WT_SESSION"):
        return "WindowsTerminal"

    term: Final = _env_non_empty("TERM")
    if term:
        return _sanitize_user_agent_token(term) or "unknown"

    return "unknown"


def _normalize_os_version(value: str) -> str:
    trimmed: Final = value.strip()
    if not trimmed:
        return "Unknown"
    parts: Final = trimmed.split(".")
    if len(parts) > 3 or not all(part.isdigit() for part in parts):
        return trimmed
    numbers: Final = tuple(int(part) for part in parts)
    major, minor, patch = (*numbers, 0, 0)[:3]
    return f"{major}.{minor}.{patch}"


def _linux_os_identity() -> tuple[str, str]:
    try:
        release: Final = platform.freedesktop_os_release()
    except OSError:
        return ("Linux", "Unknown")
    name: Final = (release.get("NAME") or "").removesuffix(" GNU/Linux").strip() or "Linux"
    return (name, _normalize_os_version(release.get("VERSION_ID") or ""))


def _os_identity() -> tuple[str, str]:
    system: Final = platform.system()
    if system == "Linux":
        return _linux_os_identity()
    if system == "Darwin":
        return ("Mac OS", _normalize_os_version(platform.mac_ver()[0]))
    if system == "Windows":
        return ("Windows", _normalize_os_version(platform.version()))
    return (system or "Unknown", _normalize_os_version(platform.release()))


_WINDOWS_ARCHITECTURES: Final = {
    "AMD64": "x86_64",
    "ARM64": "aarch64",
    "IA64": "ia64",
    "ARM": "arm",
    "X86": "i386",
}


def _os_architecture() -> str:
    machine: Final = platform.machine()
    if platform.system() == "Windows":
        return _WINDOWS_ARCHITECTURES.get(machine.upper(), machine) or "unknown"
    return machine or "unknown"


def get_chatgpt_codex_version() -> str:
    override: Final = os.getenv("CHATGPT_CODEX_VERSION")
    if override and override.strip():
        return _sanitize_user_agent_token(override.strip()) or PINNED_CODEX_VERSION
    return PINNED_CODEX_VERSION


def get_chatgpt_originator() -> str:
    originator: Final = os.getenv("CHATGPT_ORIGINATOR") or DEFAULT_ORIGINATOR
    return _safe_header_value(originator) or DEFAULT_ORIGINATOR


def get_chatgpt_api_base() -> str:
    return os.getenv("CHATGPT_API_BASE") or os.getenv("OPENAI_CHATGPT_API_BASE") or CHATGPT_API_BASE


def get_chatgpt_user_agent(originator: str) -> str:
    override: Final = os.getenv("CHATGPT_USER_AGENT")
    if override:
        return _safe_header_value(override) or DEFAULT_USER_AGENT
    version: Final = get_chatgpt_codex_version()
    os_type, os_version = _os_identity()
    arch: Final = _os_architecture()
    terminal_ua: Final = _terminal_user_agent()
    raw_suffix: Final = os.getenv("CHATGPT_USER_AGENT_SUFFIX", "").strip()
    suffix: Final = f" ({raw_suffix})" if raw_suffix else ""
    candidate: Final = f"{originator}/{version} ({os_type} {os_version}; {arch}) {terminal_ua}{suffix}"
    return _safe_header_value(candidate) or DEFAULT_USER_AGENT


def get_chatgpt_default_headers(
    access_token: str,
    account_id: str | None,
    session_id: str | None = None,
) -> dict[str, str]:
    originator: Final = get_chatgpt_originator()
    user_agent: Final = get_chatgpt_user_agent(originator)
    headers: Final = {
        "Authorization": f"Bearer {access_token}",
        "content-type": "application/json",
        "accept": "text/event-stream",
        "originator": originator,
        "user-agent": user_agent,
    }
    if session_id:
        headers["session-id"] = session_id
        headers["thread-id"] = session_id
        headers["x-client-request-id"] = session_id
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    return headers


def get_chatgpt_default_instructions() -> str:
    return os.getenv("CHATGPT_DEFAULT_INSTRUCTIONS", "").strip()


def _normalize_litellm_params(litellm_params: object | None) -> Mapping[str, object]:
    if litellm_params is None:
        return {}
    dumped: Final = litellm_params.model_dump() if isinstance(litellm_params, BaseModel) else litellm_params
    try:
        return PARAMS_ADAPTER.validate_python(dumped)
    except ValidationError:
        return {}


def get_chatgpt_session_id(litellm_params: object) -> str | None:
    params: Final = _normalize_litellm_params(litellm_params)
    for key in ("litellm_session_id", "session_id"):
        value = params.get(key)
        if value:
            return str(value)
    metadata: Final = params.get("metadata")
    if isinstance(metadata, Mapping):
        session_id: Final = LOOSE_MAPPING_ADAPTER.validate_python(metadata).get("session_id")
        if isinstance(session_id, str) and session_id:
            return session_id
    return None
