import copy
from threading import Event, Lock, Thread
from unittest.mock import Mock, patch

import pytest

import litellm
from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig, clear_empty_reasoning_warnings


def _function_tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object"}},
    }


def test_drop_unsupported_tools_keeps_function_tools_only():
    optional_params = {
        "tools": [
            _function_tool("shell"),
            {"type": "namespace", "name": "container.exec"},
            _function_tool("apply_patch"),
        ],
        "tool_choice": "auto",
    }

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert [tool["function"]["name"] for tool in result["tools"]] == [
        "shell",
        "apply_patch",
    ]
    assert all(tool["type"] == "function" for tool in result["tools"])
    assert result["tool_choice"] == "auto"


def test_drop_unsupported_tools_drops_dangling_tool_choice_when_none_survive():
    optional_params = {
        "tools": [{"type": "namespace", "name": "container.exec"}],
        "tool_choice": "required",
        "parallel_tool_calls": True,
        "temperature": 0.2,
    }

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert "tools" not in result
    assert "tool_choice" not in result
    assert "parallel_tool_calls" not in result
    assert result["temperature"] == 0.2


def test_drop_unsupported_tools_is_noop_for_function_only():
    optional_params = {
        "tools": [_function_tool("shell")],
        "tool_choice": "auto",
    }

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert result is optional_params


def test_drop_unsupported_tools_is_noop_without_tools():
    optional_params = {"temperature": 0.7}

    result = DeepSeekChatConfig._drop_unsupported_tools(optional_params)

    assert result is optional_params


def test_transform_request_strips_unsupported_tools_from_body():
    config = DeepSeekChatConfig()
    body = config.transform_request(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "tools": [
                _function_tool("shell"),
                {"type": "namespace", "name": "container.exec"},
            ],
            "tool_choice": "auto",
        },
        litellm_params={},
        headers={},
    )

    assert [tool["type"] for tool in body["tools"]] == ["function"]
    assert body["tools"][0]["function"]["name"] == "shell"


async def test_async_transform_request_strips_unsupported_tools_from_body():
    config = DeepSeekChatConfig()
    body = await config.async_transform_request(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={
            "tools": [
                _function_tool("shell"),
                {"type": "namespace", "name": "container.exec"},
            ],
            "tool_choice": "auto",
        },
        litellm_params={},
        headers={},
    )

    assert [tool["type"] for tool in body["tools"]] == ["function"]
    assert body["tools"][0]["function"]["name"] == "shell"


def test_thinking_mode_active_bool_thinking_returns_false_without_crashing():
    config = DeepSeekChatConfig()
    assert config._thinking_mode_active(model="deepseek-reasoner", optional_params={"thinking": True}) is False


class TestDeepSeekVisionMultimodalContent:
    """Image content lists are forwarded only for user messages on vision models."""

    VISION_MODEL = "deepseek/deepseek-v4-flash-vision-exp"
    NON_VISION_MODEL = "deepseek/deepseek-chat"

    def setup_method(self):
        self.config = DeepSeekChatConfig()
        prior_entry = litellm.model_cost.get(self.VISION_MODEL)
        self._prior_registry_entry = dict(prior_entry) if prior_entry is not None else None
        litellm.register_model(
            {
                "deepseek/deepseek-v4-flash-vision-exp": {
                    "litellm_provider": "deepseek",
                    "mode": "chat",
                    "input_cost_per_token": 4.4e-07,
                    "output_cost_per_token": 1.32e-06,
                    "supports_vision": True,
                }
            }
        )

    def teardown_method(self):
        if self._prior_registry_entry is None:
            litellm.model_cost.pop(self.VISION_MODEL, None)
        else:
            litellm.model_cost[self.VISION_MODEL] = self._prior_registry_entry

    @staticmethod
    def _image_message(role="user"):
        return {
            "role": role,
            "content": [
                {"type": "text", "text": "what is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg", "detail": "auto"},
                },
            ],
        }

    def test_user_image_list_forwarded_on_vision_model(self):
        result = self.config._transform_messages([self._image_message()], model=self.VISION_MODEL)

        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "image_url"
        assert result[0]["content"][1]["image_url"]["url"] == "https://example.com/image.jpg"

    def test_image_list_collapsed_on_non_vision_model(self):
        result = self.config._transform_messages([self._image_message()], model=self.NON_VISION_MODEL)

        assert result[0]["content"] == "what is in this image?"

    def test_image_list_collapsed_on_non_user_roles_even_on_vision_model(self):
        for role in ("assistant", "system"):
            result = self.config._transform_messages([self._image_message(role=role)], model=self.VISION_MODEL)

            assert result[0]["content"] == "what is in this image?"

    def test_audio_block_collapsed_even_on_vision_model(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe this"},
                    {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}},
                ],
            }
        ]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        assert result[0]["content"] == "transcribe this"

    def test_typeless_image_block_collapses(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"image_url": {"url": "https://example.com/image.jpg"}},
                ],
            }
        ]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        assert result[0]["content"] == "what is this"

    def test_text_only_content_list_collapses(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world"},
                ],
            }
        ]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        assert isinstance(result[0]["content"], str)
        assert result[0]["content"] == "Hello world"

    def test_search_results_text_appended_on_forwarded_message(self):
        message = self._image_message()
        message["search_results"] = [{"source": "kb", "content": [{"text": "article body"}]}]

        result = self.config._transform_messages([message], model=self.VISION_MODEL)

        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[-1] == {"type": "text", "text": "kbarticle body"}
        assert any(block.get("type") == "image_url" for block in content)
        assert "search_results" not in result[0]

    def test_search_results_text_kept_on_collapse(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "context: "}],
                "search_results": [{"source": "kb", "content": [{"text": "article body"}]}],
            }
        ]

        result = self.config._transform_messages(messages, model=self.NON_VISION_MODEL)

        assert result[0]["content"] == "context: kbarticle body"

    def test_responses_shape_blocks_collapse_even_on_vision_model(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                ],
            }
        ]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        assert result[0]["content"] == "what is this?"

    def test_image_block_missing_payload_collapses(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi"}, {"type": "image_url"}],
            }
        ]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        assert result[0]["content"] == "hi"

    def test_image_block_empty_payload_object_collapses(self):
        for payload in ({}, {"url": ""}, {"detail": "auto"}, None, 42):
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": payload}],
                }
            ]

            result = self.config._transform_messages(messages, model=self.VISION_MODEL)

            assert result[0]["content"] == "hi"

    def test_image_block_string_payload_forwarded(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": "https://example.com/image.jpg"},
                ],
            }
        ]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[1]["image_url"] == {"url": "https://example.com/image.jpg"}

    def test_text_block_missing_text_field_collapses(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "text"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                ],
            }
        ]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        assert result[0]["content"] == "hi"

    def test_string_content_search_results_folded_into_string(self):
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "summarize the docs",
                "search_results": [{"source": "kb", "content": [{"text": "article body"}]}],
            }
        ]

        result = self.config._transform_messages(messages, model=self.NON_VISION_MODEL)

        assert result[0]["content"] == "summarize the docskbarticle body"

    def test_plain_string_content_message_unchanged(self):
        messages = [{"role": "user", "content": "hello"}]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        assert result[0] is messages[0]

    def test_empty_content_list_untouched(self):
        messages = [{"role": "user", "content": []}]

        result = self.config._transform_messages(messages, model=self.NON_VISION_MODEL)

        assert result[0]["content"] == []

    def test_later_messages_still_collapsed_after_forwarded_one(self):
        messages = [
            self._image_message(),
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "and "},
                    {"type": "text", "text": "then?"},
                ],
            },
            self._image_message(),
        ]

        result = self.config._transform_messages(messages, model=self.VISION_MODEL)

        assert isinstance(result[0]["content"], list)
        assert result[1]["content"] == "and then?"
        assert isinstance(result[2]["content"], list)

    def test_transform_request_preserves_image_url_block(self):
        body = self.config.transform_request(
            model=self.VISION_MODEL,
            messages=[self._image_message()],
            optional_params={},
            litellm_params={},
            headers={},
        )

        content = body["messages"][0]["content"]
        assert isinstance(content, list)
        assert any(block.get("type") == "image_url" for block in content)

    async def test_async_transform_request_preserves_image_url_block(self):
        body = await self.config.async_transform_request(
            model=self.VISION_MODEL,
            messages=[self._image_message()],
            optional_params={},
            litellm_params={},
            headers={},
        )

        content = body["messages"][0]["content"]
        assert isinstance(content, list)
        assert any(block.get("type") == "image_url" for block in content)


class TestDeepSeekThinkingParams:
    """Test thinking and reasoning_effort parameter handling for DeepSeek."""

    def setup_method(self):
        self.config = DeepSeekChatConfig()
        self.model = "deepseek-reasoner"

    def test_get_supported_openai_params_includes_thinking(self):
        """Test that thinking and reasoning_effort are in supported params."""
        params = self.config.get_supported_openai_params(self.model)
        assert "thinking" in params
        assert "reasoning_effort" in params

    def test_map_thinking_enabled(self):
        """Test that thinking={"type": "enabled"} is passed through correctly."""
        non_default_params = {"thinking": {"type": "enabled"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_thinking_with_budget_tokens_strips_budget(self):
        """Test that budget_tokens is stripped from thinking param (DeepSeek doesn't support it)."""
        non_default_params = {"thinking": {"type": "enabled", "budget_tokens": 2048}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # Should strip budget_tokens, only pass type
        assert result["thinking"] == {"type": "enabled"}
        assert "budget_tokens" not in result.get("thinking", {})

    def test_map_reasoning_effort_medium(self):
        """Test that reasoning_effort='medium' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "medium"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_low(self):
        """Test that reasoning_effort='low' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "low"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_high(self):
        """Test that reasoning_effort='high' maps to thinking enabled."""
        non_default_params = {"reasoning_effort": "high"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "enabled"}

    def test_map_reasoning_effort_none_does_not_enable_thinking(self):
        """Test that reasoning_effort='none' does not enable thinking."""
        non_default_params = {"reasoning_effort": "none"}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert result["thinking"] == {"type": "disabled"}

    def test_map_reasoning_effort_null_does_not_enable_thinking(self):
        """Test that reasoning_effort=None does not enable thinking."""
        non_default_params = {"reasoning_effort": None}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_thinking_takes_precedence_over_reasoning_effort(self):
        """Test that thinking param takes precedence when both are provided."""
        non_default_params = {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
        }
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        # thinking should be set, reasoning_effort should not override
        assert result["thinking"] == {"type": "enabled"}

    def test_invalid_thinking_type_ignored(self):
        """Test that invalid thinking type values are ignored."""
        non_default_params = {"thinking": {"type": "invalid"}}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_thinking_none_value_ignored(self):
        """Test that thinking=None is ignored."""
        non_default_params = {"thinking": None}
        optional_params = {}

        result = self.config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=self.model,
            drop_params=False,
        )

        assert "thinking" not in result

    def test_drop_unsupported_tools_removes_dangling_tool_choice(self):
        optional_params = {
            "tools": [
                {"type": "namespace", "name": "local_shell"},
                {"type": "function", "function": {"name": "get_weather"}},
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "local_shell"},
            },
            "parallel_tool_calls": True,
        }

        result = self.config._drop_unsupported_tools(optional_params)

        assert result["tools"] == [{"type": "function", "function": {"name": "get_weather"}}]
        assert "tool_choice" not in result
        assert result["parallel_tool_calls"] is True


def _empty_reasoning_message(call_id: str, content: str | list[dict] | None = None) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "reasoning_content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "shell", "arguments": "{}"},
            }
        ],
    }


def _tool_result(call_id: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": "ok"}


def _transform(config: DeepSeekChatConfig, messages: list[dict]) -> dict:
    return config.transform_request(
        model="deepseek-reasoner",
        messages=copy.deepcopy(messages),
        optional_params={"thinking": {"type": "enabled"}},
        litellm_params={},
        headers={},
    )


def _reasoning_warnings(mock_warning: Mock) -> list:
    return [call for call in mock_warning.call_args_list if "DeepSeek thinking mode" in str(call.args[0])]


class TestDeepSeekEmptyReasoningWarning:
    """An explicit empty chain is normal for the model, so warn once per turn, not per replay."""

    @pytest.fixture(autouse=True)
    def _reset_warning_cache(self):
        clear_empty_reasoning_warnings()
        yield
        clear_empty_reasoning_warnings()

    def test_replayed_history_warns_once_per_message(self):
        config = DeepSeekChatConfig()
        messages = [
            {"role": "user", "content": "Read replay.txt"},
            _empty_reasoning_message("call_replay"),
            _tool_result("call_replay"),
        ]

        with patch.object(litellm.verbose_logger, "warning") as mock_warning:
            first = _transform(config, messages)
            second = _transform(config, messages)

        assert len(_reasoning_warnings(mock_warning)) == 1
        assert first["messages"][1]["reasoning_content"] == " "
        assert second["messages"][1]["reasoning_content"] == " "

    def test_new_empty_message_warns_again(self):
        config = DeepSeekChatConfig()
        first_messages = [
            {"role": "user", "content": "Read first.txt"},
            _empty_reasoning_message("call_first"),
            _tool_result("call_first"),
        ]
        second_messages = [
            {"role": "user", "content": "Read second.txt"},
            _empty_reasoning_message("call_second"),
            _tool_result("call_second"),
        ]

        with patch.object(litellm.verbose_logger, "warning") as mock_warning:
            _transform(config, first_messages)
            _transform(config, second_messages)

        assert len(_reasoning_warnings(mock_warning)) == 2

    def test_dropped_chain_warns_when_message_turns_empty(self):
        config = DeepSeekChatConfig()
        assistant = _empty_reasoning_message("call_dropped")
        assistant["reasoning_content"] = "I should read the file first."
        messages = [
            {"role": "user", "content": "Read dropped.txt"},
            assistant,
            _tool_result("call_dropped"),
        ]

        with patch.object(litellm.verbose_logger, "warning") as mock_warning:
            _transform(config, messages)
            assistant["reasoning_content"] = ""
            _transform(config, messages)

        assert len(_reasoning_warnings(mock_warning)) == 1

    def test_absent_field_still_warns_on_every_request(self):
        config = DeepSeekChatConfig()
        assistant = _empty_reasoning_message("call_absent")
        del assistant["reasoning_content"]
        messages = [
            {"role": "user", "content": "Read absent.txt"},
            assistant,
            _tool_result("call_absent"),
        ]

        with patch.object(litellm.verbose_logger, "warning") as mock_warning:
            _transform(config, messages)
            _transform(config, messages)

        assert len(_reasoning_warnings(mock_warning)) == 2

    def test_stored_reasoning_is_promoted_without_warning(self):
        config = DeepSeekChatConfig()
        assistant = _empty_reasoning_message("call_stored")
        assistant["provider_specific_fields"] = {"reasoning_content": "stored chain"}
        messages = [
            {"role": "user", "content": "Read stored.txt"},
            assistant,
            _tool_result("call_stored"),
        ]

        with patch.object(litellm.verbose_logger, "warning") as mock_warning:
            body = _transform(config, messages)

        assert len(_reasoning_warnings(mock_warning)) == 0
        assert body["messages"][1]["reasoning_content"] == "stored chain"
        assert "reasoning_content" not in body["messages"][1]["provider_specific_fields"]

    def test_nested_transport_metadata_does_not_change_identity(self):
        config = DeepSeekChatConfig()
        plain = _empty_reasoning_message("call_metadata", content=[{"type": "text", "text": "checking"}])
        decorated = copy.deepcopy(plain)
        decorated["content"][0]["cache_control"] = {"type": "ephemeral"}
        decorated["tool_calls"][0]["provider_specific_fields"] = {"trace": "abc"}
        base = {"role": "user", "content": "Read metadata.txt"}

        with patch.object(litellm.verbose_logger, "warning") as mock_warning:
            _transform(config, [base, plain, _tool_result("call_metadata")])
            _transform(config, [base, decorated, _tool_result("call_metadata")])

        assert len(_reasoning_warnings(mock_warning)) == 1

    def test_concurrent_calls_warn_once(self):
        config = DeepSeekChatConfig()
        messages = [
            {"role": "user", "content": "Read concurrent.txt"},
            _empty_reasoning_message("call_concurrent"),
            _tool_result("call_concurrent"),
        ]
        entered_log = Event()
        release_log = Event()
        second_done = Event()
        records: list[str] = []
        record_lock = Lock()

        def record(message: str, *args: object) -> None:
            with record_lock:
                records.append(message)
                is_first = len(records) == 1
            if is_first:
                entered_log.set()
                release_log.wait(timeout=5)

        def run_once() -> None:
            _transform(config, messages)

        def run_second() -> None:
            try:
                run_once()
            finally:
                second_done.set()

        first_thread = Thread(target=run_once, daemon=True)
        second_thread: Thread | None = None

        with patch.object(litellm.verbose_logger, "warning", record):
            first_thread.start()
            try:
                assert entered_log.wait(timeout=5), "first call never reached the logger"
                second_thread = Thread(target=run_second, daemon=True)
                second_thread.start()
                assert second_done.wait(timeout=5), "second call blocked behind the first"
            finally:
                release_log.set()
                first_thread.join(timeout=5)
                if second_thread is not None:
                    second_thread.join(timeout=5)

        assert len([message for message in records if "DeepSeek thinking mode" in message]) == 1
