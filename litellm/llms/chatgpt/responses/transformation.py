from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

import httpx
from pydantic import TypeAdapter

from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _safe_convert_created_field,
)
from litellm.llms.openai.common_utils import OpenAIError
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.responses.sse_output_recovery import (
    parse_sse_json_chunk,
    record_output_item_chunk,
    record_output_text_chunk,
)
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

from ..authenticator import resolve_chatgpt_credentials
from ..common_utils import (
    ChatGPTAuthError,
    ensure_chatgpt_session_id,
    get_chatgpt_api_base,
    get_chatgpt_default_headers,
    get_chatgpt_default_instructions,
    get_chatgpt_session_id,
)

UNSUPPORTED_CHATGPT_RESPONSES_PARAMS: Final = frozenset(
    {
        "user",
        "temperature",
        "top_p",
        "context_management",
        "metadata",
        "max_output_tokens",
        "stream_options",
    }
)

PROMPT_CACHE_KEY_MAX_LENGTH: Final = 64

INPUT_ADAPTER: Final[TypeAdapter[str | list[object]]] = TypeAdapter(str | list[object])


def _input_items(input_value: str | Sequence[object]) -> Sequence[object]:
    if isinstance(input_value, str):
        return [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": input_value}]}]
    return input_value


if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class ChatGPTResponsesAPIConfig(OpenAIResponsesAPIConfig):
    def __init__(self) -> None:
        super().__init__()

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.CHATGPT

    def validate_environment(
        self,
        headers: dict,
        model: str,
        litellm_params: GenericLiteLLMParams | None,
    ) -> dict:
        configured_api_key: Final = litellm_params.api_key if litellm_params else None
        access_token, account_id = self._resolve_credentials(configured_api_key=configured_api_key, model=model)
        session_id: Final = ensure_chatgpt_session_id(litellm_params)
        default_headers: Final = get_chatgpt_default_headers(access_token, account_id, session_id)
        return {**default_headers, **headers}

    def _resolve_credentials(self, configured_api_key: str | None, model: str) -> tuple[str, str | None]:
        try:
            credentials: Final = resolve_chatgpt_credentials(configured_api_key)
        except ChatGPTAuthError as e:
            raise AuthenticationError(
                model=model,
                llm_provider="chatgpt",
                message=str(e),
            )
        return credentials.access_token, credentials.account_id

    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        request: Final = super().transform_responses_api_request(
            model,
            input,
            response_api_optional_request_params,
            litellm_params,
            headers,
        )
        request["store"] = False
        request["stream"] = True
        request.setdefault("tool_choice", "auto")
        request.setdefault("parallel_tool_calls", True)
        request.setdefault("text", {"verbosity": "low"})
        session_id: Final = get_chatgpt_session_id(litellm_params)
        if session_id:
            request.setdefault("prompt_cache_key", session_id[:PROMPT_CACHE_KEY_MAX_LENGTH])
        include: Final = list(request.get("include") or [])
        if "reasoning.encrypted_content" not in include:
            include.append("reasoning.encrypted_content")
        request["include"] = include

        base_instructions: Final = get_chatgpt_default_instructions()
        raw_instructions: Final = request.get("instructions")
        caller_instructions: Final = raw_instructions if isinstance(raw_instructions, str) else None
        if caller_instructions is not None:
            del request["instructions"]
        base_text: Final = "\n\n".join(part for part in (base_instructions, caller_instructions) if part)
        if base_text:
            request["input"] = [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": base_text}],
                    "internal_chat_message_metadata_passthrough": {"content_item_kinds": ["model.base_instructions"]},
                },
                *_input_items(INPUT_ADAPTER.validate_python(request.get("input"))),
            ]

        return {k: v for k, v in request.items() if k not in UNSUPPORTED_CHATGPT_RESPONSES_PARAMS}

    def transform_response_api_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> ResponsesAPIResponse:
        body_text: Final = raw_response.text or ""
        if not self._should_parse_as_sse(raw_response=raw_response, body_text=body_text):
            return super().transform_response_api_response(
                model=model,
                raw_response=raw_response,
                logging_obj=logging_obj,
            )

        logging_obj.post_call(
            original_response=raw_response.text,
            additional_args={"complete_input_dict": {}},
        )

        completed_response, error_message = self._extract_completed_response_from_sse(body_text=body_text)
        if completed_response is None:
            raise OpenAIError(
                message=error_message or raw_response.text,
                status_code=raw_response.status_code,
            )

        self._attach_response_headers(completed_response=completed_response, raw_response=raw_response)
        return completed_response

    def _should_parse_as_sse(self, raw_response: httpx.Response, body_text: str) -> bool:
        content_type: Final = (raw_response.headers or {}).get("content-type", "")
        if "text/event-stream" in content_type.lower():
            return True
        trimmed_body: Final = body_text.lstrip()
        return bool(
            trimmed_body.startswith("event:")
            or trimmed_body.startswith("data:")
            or "\nevent:" in body_text
            or "\ndata:" in body_text
        )

    def _extract_completed_response_from_sse(self, body_text: str) -> tuple[ResponsesAPIResponse | None, str | None]:
        completed_response = None
        error_message = None
        streamed_output_items: Final[dict[int, dict[str, object]]] = {}
        text_only_output_items: Final[dict[int, dict[str, object]]] = {}
        for chunk in body_text.splitlines():
            parsed_chunk = parse_sse_json_chunk(chunk)
            if parsed_chunk is None:
                continue

            event_type = parsed_chunk.get("type")
            if event_type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE:
                record_output_item_chunk(
                    parsed_chunk=parsed_chunk,
                    output_items=streamed_output_items,
                )
                continue

            if event_type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE:
                record_output_text_chunk(
                    parsed_chunk=parsed_chunk,
                    output_items=streamed_output_items,
                    text_only_items=text_only_output_items,
                )
                continue

            if event_type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED:
                # Real OUTPUT_ITEM_DONE events take precedence at any given
                # output_index, but text-only items at indices without a
                # matching OUTPUT_ITEM_DONE must still be preserved (e.g.
                # providers that emit only OUTPUT_TEXT_DONE for some indices).
                merged_items: dict[int, dict[str, object]] = {**text_only_output_items}
                merged_items.update(streamed_output_items)
                completed_response = self._build_completed_response_from_chunk(
                    parsed_chunk=parsed_chunk,
                    streamed_output_items=merged_items,
                )
                break

            if event_type in (
                ResponsesAPIStreamEvents.RESPONSE_FAILED,
                ResponsesAPIStreamEvents.ERROR,
            ):
                extracted_error = self._extract_error_message(parsed_chunk)
                if extracted_error is not None:
                    error_message = extracted_error

        return completed_response, error_message

    def _build_completed_response_from_chunk(
        self, parsed_chunk: Mapping[str, object], streamed_output_items: Mapping[int, dict[str, object]]
    ) -> ResponsesAPIResponse | None:
        response_payload = parsed_chunk.get("response")
        if not isinstance(response_payload, dict):
            return None
        response_payload = dict(response_payload)
        if not response_payload.get("output") and streamed_output_items:
            response_payload["output"] = [item for _, item in sorted(streamed_output_items.items())]
        if "created_at" in response_payload:
            response_payload["created_at"] = _safe_convert_created_field(response_payload["created_at"])
        try:
            return ResponsesAPIResponse(**response_payload)
        except Exception:
            return ResponsesAPIResponse.model_construct(**response_payload)

    def _extract_error_message(self, parsed_chunk: dict[str, Any]) -> str | None:
        error_obj: Final = parsed_chunk.get("error") or (parsed_chunk.get("response") or {}).get("error")
        if error_obj is None:
            return None
        if isinstance(error_obj, dict):
            return error_obj.get("message") or str(error_obj)
        return str(error_obj)

    def _attach_response_headers(
        self,
        completed_response: ResponsesAPIResponse,
        raw_response: httpx.Response,
    ) -> None:
        raw_headers: Final = dict(raw_response.headers)
        processed_headers: Final = process_response_headers(raw_headers)
        if not hasattr(completed_response, "_hidden_params"):
            setattr(completed_response, "_hidden_params", {})
        completed_response._hidden_params["additional_headers"] = processed_headers
        completed_response._hidden_params["headers"] = raw_headers

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        api_base = api_base or get_chatgpt_api_base()
        api_base = api_base.rstrip("/")
        return f"{api_base}/responses"

    def supports_native_websocket(self) -> bool:
        """ChatGPT does not support native WebSocket for Responses API"""
        return False
