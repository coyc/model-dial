"""Tests for Google ↔ OpenAI format conversion."""

import json
import pytest

from convert_google import GoogleConverter


# ===========================================================================
# Tests: convert_google_to_openai_response (non-streaming)
# ===========================================================================
class TestConvertGoogleToOpenaiResponse:
    def test_basic_response(self):
        google_resp = {
            "candidates": [
                {"content": {"parts": [{"text": "Hello!"}], "role": "model"}, "index": 0}
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 2,
                "totalTokenCount": 12,
            },
            "modelVersion": "gemini-2.0-flash",
            "responseId": "abc-123",
        }
        result = GoogleConverter.provider_to_openai(google_resp)

        assert result["id"] == "abc-123"
        assert result["model"] == "gemini-2.0-flash"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 2
        assert result["usage"]["total_tokens"] == 12

    def test_empty_candidates(self):
        google_resp = {"candidates": [], "usageMetadata": {"promptTokenCount": 5}}
        result = GoogleConverter.provider_to_openai(google_resp)

        assert result["choices"] == []
        assert result["usage"]["prompt_tokens"] == 5

    def test_finish_reason_max_tokens(self):
        google_resp = {
            "candidates": [
                {"content": {"parts": [{"text": "truncated"}], "role": "model"},
                 "finishReason": "MAX_TOKENS", "index": 0}
            ],
            "usageMetadata": {},
        }
        result = GoogleConverter.provider_to_openai(google_resp)
        assert result["choices"][0]["finish_reason"] == "length"

    def test_no_usage_metadata(self):
        google_resp = {
            "candidates": [
                {"content": {"parts": [{"text": "ok"}], "role": "model"}, "index": 0}
            ],
        }
        result = GoogleConverter.provider_to_openai(google_resp)
        assert result["usage"] == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_function_call_with_args(self):
        """Google functionCall with args should convert to OpenAI tool_calls."""
        google_resp = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "Paris", "unit": "celsius"},
                                    "id": "call_123"
                                }
                            }
                        ],
                        "role": "model"
                    },
                    "index": 0,
                    "finishReason": "STOP"
                }
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
            "responseId": "resp-1",
        }
        result = GoogleConverter.provider_to_openai(google_resp)

        assert result["choices"][0]["finish_reason"] == "tool_calls"
        assert "tool_calls" in result["choices"][0]["message"]
        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["type"] == "function"
        assert tool_calls[0]["id"] == "call_123"
        assert tool_calls[0]["function"]["name"] == "get_weather"
        args = json.loads(tool_calls[0]["function"]["arguments"])
        assert args == {"location": "Paris", "unit": "celsius"}

    def test_function_call_without_id(self):
        """Function call without id should still convert (id is optional in older API versions)."""
        google_resp = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "test_func", "args": {"a": 1}}}
                        ],
                        "role": "model"
                    },
                    "index": 0
                }
            ],
        }
        result = GoogleConverter.provider_to_openai(google_resp)

        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert "id" not in tool_calls[0]
        assert tool_calls[0]["function"]["name"] == "test_func"

    def test_function_call_with_text(self):
        """Response can contain both text and function calls."""
        google_resp = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Let me check the weather."},
                            {"functionCall": {"name": "get_weather", "args": {"location": "NYC"}, "id": "c1"}}
                        ],
                        "role": "model"
                    },
                    "index": 0
                }
            ],
        }
        result = GoogleConverter.provider_to_openai(google_resp)

        assert result["choices"][0]["message"]["content"] == "Let me check the weather."
        assert len(result["choices"][0]["message"]["tool_calls"]) == 1

    def test_multiple_function_calls(self):
        """Multiple function calls in one response."""
        google_resp = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "func1", "args": {"x": 1}, "id": "c1"}},
                            {"functionCall": {"name": "func2", "args": {"y": 2}, "id": "c2"}}
                        ],
                        "role": "model"
                    },
                    "index": 0
                }
            ],
        }
        result = GoogleConverter.provider_to_openai(google_resp)

        tool_calls = result["choices"][0]["message"]["tool_calls"]
        assert len(tool_calls) == 2
        assert tool_calls[0]["function"]["name"] == "func1"
        assert tool_calls[1]["function"]["name"] == "func2"
        assert tool_calls[0]["index"] == 0
        assert tool_calls[1]["index"] == 1


# ===========================================================================
# Tests: convert_google_sse_chunk_to_openai (streaming)
# ===========================================================================
class TestConvertGoogleSseChunkToOpenai:
    def test_text_chunk(self):
        google_data = {
            "candidates": [
                {"content": {"parts": [{"text": "He"}], "role": "model"}, "index": 0}
            ],
            "responseId": "chunk-1",
            "modelVersion": "gemini-2.0-flash",
        }
        result = GoogleConverter.provider_sse_to_openai(google_data)

        assert result is not None
        assert result["object"] == "chat.completion.chunk"
        assert result["choices"][0]["delta"]["content"] == "He"
        assert "finish_reason" not in result["choices"][0]

    def test_finish_chunk(self):
        google_data = {
            "candidates": [
                {"content": {"parts": [], "role": "model"},
                 "finishReason": "STOP", "index": 0}
            ],
            "responseId": "chunk-2",
            "modelVersion": "gemini-2.0-flash",
        }
        result = GoogleConverter.provider_sse_to_openai(google_data)

        assert result is not None
        assert result["choices"][0]["finish_reason"] == "stop"
        assert "content" not in result["choices"][0]["delta"]

    def test_empty_chunk_returns_none(self):
        """Empty chunk with no text and no finish → None (skip)."""
        google_data = {
            "candidates": [
                {"content": {"parts": [], "role": "model"}, "index": 0}
            ],
        }
        assert GoogleConverter.provider_sse_to_openai(google_data) is None

    def test_no_candidates_returns_none(self):
        assert GoogleConverter.provider_sse_to_openai({"candidates": []}) is None
        assert GoogleConverter.provider_sse_to_openai({}) is None

    def test_chunk_with_usage(self):
        google_data = {
            "candidates": [
                {"content": {"parts": [{"text": "done"}], "role": "model"},
                 "finishReason": "STOP", "index": 0}
            ],
            "usageMetadata": {
                "promptTokenCount": 20,
                "candidatesTokenCount": 5,
                "totalTokenCount": 25,
            },
        }
        result = GoogleConverter.provider_sse_to_openai(google_data)
        assert result["usage"]["prompt_tokens"] == 20
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 25

    def test_function_call_chunk(self):
        """Streaming chunk with functionCall should convert to tool_calls."""
        google_data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "search", "args": {"query": "test"}, "id": "fc1"}}
                        ],
                        "role": "model"
                    },
                    "index": 0
                }
            ],
            "responseId": "chunk-fc",
        }
        result = GoogleConverter.provider_sse_to_openai(google_data)

        assert result is not None
        assert "tool_calls" in result["choices"][0]["delta"]
        tool_calls = result["choices"][0]["delta"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "search"
        assert tool_calls[0]["id"] == "fc1"

    def test_function_call_with_finish_reason(self):
        """Function call chunk with finishReason should use 'tool_calls' as finish_reason."""
        google_data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "test", "args": {}}}
                        ],
                        "role": "model"
                    },
                    "finishReason": "STOP",
                    "index": 0
                }
            ],
        }
        result = GoogleConverter.provider_sse_to_openai(google_data)

        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_function_call_chunk_not_empty(self):
        """Chunk with only functionCall (no text) should NOT return None."""
        google_data = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "func", "args": {"x": 1}}}
                        ],
                        "role": "model"
                    },
                    "index": 0
                }
            ],
        }
        result = GoogleConverter.provider_sse_to_openai(google_data)

        assert result is not None
        assert "tool_calls" in result["choices"][0]["delta"]
        assert len(result["choices"][0]["delta"]["tool_calls"]) == 1


# ===========================================================================
# Tests: convert_openai_to_google (request conversion)
# ===========================================================================
class TestConvertOpenaiToGoogle:
    def test_basic_messages(self):
        """Convert basic messages from OpenAI to Google format."""
        openai_body = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "temperature": 0.5,
        }
        result = GoogleConverter.openai_to_provider(openai_body, "gemini-2.0-flash")

        assert "contents" in result
        assert len(result["contents"]) == 2
        assert result["contents"][0]["role"] == "user"
        assert result["contents"][0]["parts"][0]["text"] == "Hello"
        assert result["contents"][1]["role"] == "model"
        assert result["contents"][1]["parts"][0]["text"] == "Hi there!"
        assert result["generationConfig"]["temperature"] == 0.5

    def test_tools_conversion(self):
        """Convert tools from OpenAI format to Google function_declarations."""
        openai_body = {
            "messages": [{"role": "user", "content": "Create a file"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "description": "Write content to a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "File path"},
                                "content": {"type": "string", "description": "File content"},
                            },
                            "required": ["path", "content"],
                        },
                    },
                }
            ],
        }
        result = GoogleConverter.openai_to_provider(openai_body, "gemini-2.0-flash")

        assert "tools" in result
        assert len(result["tools"]) == 1
        assert "function_declarations" in result["tools"][0]
        
        declarations = result["tools"][0]["function_declarations"]
        assert len(declarations) == 1
        assert declarations[0]["name"] == "write_file"
        assert declarations[0]["description"] == "Write content to a file"
        assert "parameters" in declarations[0]
        assert declarations[0]["parameters"]["type"] == "object"
        assert "path" in declarations[0]["parameters"]["properties"]
        assert "content" in declarations[0]["parameters"]["properties"]

    def test_multiple_tools_conversion(self):
        """Convert multiple tools from OpenAI format to Google."""
        openai_body = {
            "messages": [{"role": "user", "content": "Help me"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "func1",
                        "description": "First function",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "func2",
                        "description": "Second function",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        }
        result = GoogleConverter.openai_to_provider(openai_body, "gemini-2.0-flash")

        declarations = result["tools"][0]["function_declarations"]
        assert len(declarations) == 2
        assert declarations[0]["name"] == "func1"
        assert declarations[1]["name"] == "func2"

    def test_no_tools_in_request(self):
        """When no tools provided, result should not have tools field."""
        openai_body = {
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = GoogleConverter.openai_to_provider(openai_body, "gemini-2.0-flash")

        assert "tools" not in result
        assert "contents" in result

    def test_empty_tools_array(self):
        """When tools array is empty, result should not have tools field."""
        openai_body = {
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [],
        }
        result = GoogleConverter.openai_to_provider(openai_body, "gemini-2.0-flash")

        assert "tools" not in result

    def test_system_role_maps_to_user(self):
        """System role should be converted to user role for Google."""
        openai_body = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
        }
        result = GoogleConverter.openai_to_provider(openai_body, "gemini-2.0-flash")

        assert result["contents"][0]["role"] == "user"
        assert result["contents"][0]["parts"][0]["text"] == "You are a helpful assistant."

    def test_tool_role_maps_to_user(self):
        """Tool role (tool results) should be converted to user role for Google."""
        openai_body = {
            "messages": [
                {"role": "tool", "content": "File created successfully"},
            ],
        }
        result = GoogleConverter.openai_to_provider(openai_body, "gemini-2.0-flash")

        assert result["contents"][0]["role"] == "user"
        assert result["contents"][0]["parts"][0]["text"] == "File created successfully"

    def test_multimodal_content_with_text_and_image(self):
        """Convert multimodal content (text + inline image) from OpenAI to Google."""
        openai_body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo"}},
                    ],
                }
            ],
        }
        result = GoogleConverter.openai_to_provider(openai_body, "gemini-2.0-flash")

        parts = result["contents"][0]["parts"]
        assert len(parts) == 2
        assert parts[0]["text"] == "What's in this image?"
        assert "inline_data" in parts[1]
        assert parts[1]["inline_data"]["mime_type"] == "image/png"
        assert parts[1]["inline_data"]["data"] == "iVBORw0KGgo"

    def test_image_url_http(self):
        """Convert HTTP image URL to Google file_data format."""
        openai_body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                    ],
                }
            ],
        }
        result = GoogleConverter.openai_to_provider(openai_body, "gemini-2.0-flash")

        parts = result["contents"][0]["parts"]
        assert len(parts) == 1
        assert "file_data" in parts[0]
        assert parts[0]["file_data"]["file_uri"] == "https://example.com/image.jpg"

    def test_default_temperature(self):
        """When no temperature provided, should default to 0.7."""
        openai_body = {
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = GoogleConverter.openai_to_provider(openai_body, "gemini-2.0-flash")

        assert result["generationConfig"]["temperature"] == 0.7

    def test_empty_messages(self):
        """Handle empty messages array gracefully."""
        openai_body = {
            "messages": [],
        }
        result = GoogleConverter.openai_to_provider(openai_body, "gemini-2.0-flash")

        assert result["contents"] == []
        assert "generationConfig" in result

    def test_json_schema_sanitization(self):
        """Remove unsupported JSON Schema fields (e.g., $schema, exclusiveMinimum) from parameters."""
        openai_body = {
            "messages": [{"role": "user", "content": "Test"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_func",
                        "description": "Test function",
                        "parameters": {
                            "$schema": "http://json-schema.org/draft-07/schema#",
                            "type": "object",
                            "title": "TestSchema",
                            "properties": {
                                "timeout": {
                                    "type": "integer",
                                    "description": "Timeout in ms",
                                    "exclusiveMinimum": 0,
                                    "maximum": 9007199254740991,
                                    "minimum": -9007199254740991,
                                    "default": 5000,
                                    "examples": [1000, 3000],
                                },
                                "name": {
                                    "type": "string",
                                    "description": "Name field",
                                },
                            },
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
        }
        result = GoogleConverter.openai_to_provider(openai_body, "gemini-2.0-flash")

        params = result["tools"][0]["function_declarations"][0]["parameters"]
        
        # Unsupported fields should be removed
        assert "$schema" not in params
        assert "title" not in params
        assert "additionalProperties" not in params
        
        # Supported fields should be preserved
        assert params["type"] == "object"
        assert "properties" in params
        assert params["required"] == ["name"]
        
        # Check nested property sanitization
        timeout_prop = params["properties"]["timeout"]
        assert timeout_prop["type"] == "integer"
        assert timeout_prop["description"] == "Timeout in ms"
        assert timeout_prop["minimum"] == -9007199254740991
        assert timeout_prop["maximum"] == 9007199254740991
        
        # Unsupported nested fields should be removed
        assert "exclusiveMinimum" not in timeout_prop
        assert "default" not in timeout_prop
        assert "examples" not in timeout_prop
