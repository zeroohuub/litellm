import platform
import re
from importlib.metadata import version

import pytest

from litellm.llms.chatgpt.common_utils import (
    PINNED_CODEX_VERSION,
    _normalize_os_version,
    get_chatgpt_codex_version,
    get_chatgpt_user_agent,
)

TERMINAL_ENV_VARS = (
    "TERM",
    "TERM_PROGRAM",
    "TERM_PROGRAM_VERSION",
    "TERM_SESSION_ID",
    "GHOSTTY_RESOURCES_DIR",
    "WEZTERM_VERSION",
    "ITERM_SESSION_ID",
    "ITERM_PROFILE",
    "ITERM_PROFILE_NAME",
    "KITTY_WINDOW_ID",
    "ALACRITTY_SOCKET",
    "KONSOLE_VERSION",
    "GNOME_TERMINAL_SCREEN",
    "VTE_VERSION",
    "WT_SESSION",
)

USER_AGENT_PATTERN = re.compile(
    r"^(?P<originator>[^/]+)/(?P<version>[0-9A-Za-z_.+-]+) "
    r"\((?P<os_type>[^;()]+) (?P<os_version>[^;()]+); (?P<arch>[0-9A-Za-z_.-]+)\) "
    r"(?P<terminal>[0-9A-Za-z_./+-]+)$"
)


def _user_agent(monkeypatch, env: dict[str, str], originator: str = "codex_cli_rs") -> str:
    for name in TERMINAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return get_chatgpt_user_agent(originator)


def _terminal_token(user_agent: str) -> str:
    return user_agent.rsplit(") ", 1)[-1]


def test_user_agent_version_is_the_codex_version_not_the_litellm_version(monkeypatch):
    monkeypatch.delenv("CHATGPT_CODEX_VERSION", raising=False)

    user_agent = _user_agent(monkeypatch, {})
    match = USER_AGENT_PATTERN.match(user_agent)

    assert match is not None, user_agent
    assert match["originator"] == "codex_cli_rs"
    assert match["version"] == PINNED_CODEX_VERSION
    assert match["version"] != version("litellm")


def test_user_agent_version_is_configurable(monkeypatch):
    monkeypatch.setenv("CHATGPT_CODEX_VERSION", "0.999.0")

    assert get_chatgpt_codex_version() == "0.999.0"
    assert "codex_cli_rs/0.999.0 " in _user_agent(monkeypatch, {})


def test_user_agent_version_override_cannot_break_the_header(monkeypatch):
    monkeypatch.setenv("CHATGPT_CODEX_VERSION", "0.1\nInjected: yes")

    user_agent = _user_agent(monkeypatch, {})

    assert "\n" not in user_agent
    assert USER_AGENT_PATTERN.match(user_agent) is not None, user_agent


def test_user_agent_os_segment_is_the_distro_not_the_kernel(monkeypatch):
    user_agent = _user_agent(monkeypatch, {})

    assert platform.release() not in user_agent


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("13", "13.0.0"),
        ("13.1", "13.1.0"),
        ("24.04", "24.4.0"),
        ("15.6.1", "15.6.1"),
        ("", "Unknown"),
        ("  ", "Unknown"),
        ("n/a", "n/a"),
        ("1.2.3.4", "1.2.3.4"),
    ],
)
def test_normalize_os_version(value, expected):
    assert _normalize_os_version(value) == expected


def test_headless_environment_reports_unknown_terminal(monkeypatch):
    assert _terminal_token(_user_agent(monkeypatch, {})) == "unknown"


def test_tmux_term_program_falls_through_to_term(monkeypatch):
    user_agent = _user_agent(
        monkeypatch,
        {"TERM_PROGRAM": "tmux", "TERM_PROGRAM_VERSION": "3.4", "TERM": "screen-256color"},
    )

    assert _terminal_token(user_agent) == "screen-256color"


def test_term_session_id_alone_is_not_apple_terminal(monkeypatch):
    user_agent = _user_agent(monkeypatch, {"TERM_SESSION_ID": "w0t0p0"})

    assert _terminal_token(user_agent) == "unknown"
    assert "Apple_Terminal" not in user_agent


def test_ghostty_is_detected_from_its_resources_dir(monkeypatch):
    assert _terminal_token(_user_agent(monkeypatch, {"GHOSTTY_RESOURCES_DIR": "/opt/ghostty"})) == "Ghostty"


@pytest.mark.parametrize(
    ("wezterm_version", "expected"),
    [("", "WezTerm"), ("20240101", "WezTerm/20240101")],
)
def test_wezterm_version_is_optional(monkeypatch, wezterm_version, expected):
    assert _terminal_token(_user_agent(monkeypatch, {"WEZTERM_VERSION": wezterm_version})) == expected


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"TERM": "dumb"}, "dumb"),
        ({"TERM": "xterm-256color"}, "xterm-256color"),
        ({"KITTY_WINDOW_ID": "1"}, "kitty"),
        ({"TERM": "xterm-kitty"}, "kitty"),
        ({"ALACRITTY_SOCKET": "/s"}, "Alacritty"),
        ({"KONSOLE_VERSION": "2408"}, "Konsole/2408"),
        ({"GNOME_TERMINAL_SCREEN": "/org/x"}, "gnome-terminal"),
        ({"VTE_VERSION": "7800"}, "VTE/7800"),
        ({"WT_SESSION": "abc"}, "WindowsTerminal"),
        ({"ITERM_SESSION_ID": "x"}, "iTerm.app"),
    ],
)
def test_terminal_detection_branches(monkeypatch, env, expected):
    assert _terminal_token(_user_agent(monkeypatch, env)) == expected


def test_term_program_wins_over_later_probes(monkeypatch):
    user_agent = _user_agent(
        monkeypatch,
        {"TERM_PROGRAM": "iTerm.app", "TERM_PROGRAM_VERSION": "3.5.0", "WT_SESSION": "abc"},
    )

    assert _terminal_token(user_agent) == "iTerm.app/3.5.0"


def test_terminal_token_is_ascii_safe(monkeypatch):
    user_agent = _user_agent(monkeypatch, {"TERM_PROGRAM": "evil\nInjected: yes"})

    assert all(32 <= ord(char) <= 126 for char in user_agent)
    assert "Injected: yes" not in user_agent


def test_user_agent_override_short_circuits(monkeypatch):
    monkeypatch.setenv("CHATGPT_USER_AGENT", "custom-agent/1.0")

    assert get_chatgpt_user_agent("codex_cli_rs") == "custom-agent/1.0"
