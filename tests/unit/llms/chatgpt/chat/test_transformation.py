import json
import time
from pathlib import Path

from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig


def _auth_file(tmp_path: Path) -> str:
    path = tmp_path / "auth.json"
    path.write_text(
        json.dumps(
            {
                "access_token": "access-123",
                "account_id": "acct-123",
                "expires_at": time.time() + 3600,
            }
        )
    )
    return str(path)


def _headers(auth_file: str, litellm_params: dict) -> dict:
    return ChatGPTConfig().validate_environment(
        headers={},
        model="gpt-6.1-sol",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params=litellm_params,
        api_key=auth_file,
        api_base=None,
    )


def test_validate_environment_omits_session_headers_without_identity(tmp_path: Path):
    auth_file: str = _auth_file(tmp_path)

    headers: dict = _headers(auth_file, {"api_key": auth_file})

    assert "session-id" not in headers
    assert "thread-id" not in headers
    assert "x-client-request-id" not in headers


def test_validate_environment_does_not_turn_a_trace_id_into_a_session_id(tmp_path: Path):
    auth_file: str = _auth_file(tmp_path)

    headers: dict = _headers(auth_file, {"api_key": auth_file, "litellm_trace_id": "trace-1"})

    assert "session-id" not in headers
    assert "trace-1" not in headers.values()


def test_validate_environment_forwards_a_conversation_identity(tmp_path: Path):
    auth_file: str = _auth_file(tmp_path)

    headers: dict = _headers(auth_file, {"api_key": auth_file, "metadata": {"session_id": "conv-1"}})

    assert headers["session-id"] == "conv-1"
    assert headers["thread-id"] == "conv-1"
    assert headers["x-client-request-id"] == "conv-1"
