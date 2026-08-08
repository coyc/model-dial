"""Google ↔ OpenAI format conversion.

Implements ProviderConverter protocol for Google AI API.
Pure functions — no GatewayState or network dependencies.
"""

import json
from convert_base import ProviderConverter


class GoogleConverter(ProviderConverter):
    """Converter for Google AI API format.
    
    Implements the ProviderConverter interface for Google Gemini models.
    Handles conversion between OpenAI format and Google-native format.
    """
    
    @staticmethod
    def openai_to_provider(body: dict, model_id: str) -> dict:
        """Convert OpenAI format to Google-native format.
        
        Args:
            body: OpenAI format request (messages, tools, temperature, etc.)
            model_id: Target Google model identifier
            
        Returns:
            Google-native request format (contents, generationConfig, tools)
        """
        return _convert_openai_to_google(body, model_id)
    
    @staticmethod
    def provider_to_openai(response: dict) -> dict:
        """Convert Google response to OpenAI format.
        
        Args:
            response: Google API response
            
        Returns:
            OpenAI format response
        """
        return _convert_google_to_openai_response(response)
    
    @staticmethod
    def provider_sse_to_openai(chunk: dict) -> dict | None:
        """Convert Google SSE chunk to OpenAI streaming format.
        
        Args:
            chunk: Google SSE data chunk
            
        Returns:
            OpenAI format streaming chunk, or None if chunk should be skipped
        """
        return _convert_google_sse_chunk_to_openai(chunk)


def _sanitize_json_schema(schema: dict) -> dict:
    """Remove JSON Schema fields not supported by Google API.
    
    Google only supports a subset of JSON Schema:
    - type, description, enum, format, items, properties, required
    
    Removes: $schema, $id, title, examples, default, exclusiveMinimum, 
             exclusiveMaximum, additionalProperties, etc.
    """
    if not isinstance(schema, dict):
        return schema
    
    # Fields that Google API supports
    allowed_fields = {
        "type", "description", "enum", "format", "items", "properties", "required",
        "minimum", "maximum"
    }
    
    sanitized = {}
    for key, value in schema.items():
        if key not in allowed_fields:
            continue
        
        # Recursively sanitize nested objects
        if key == "properties" and isinstance(value, dict):
            sanitized[key] = {k: _sanitize_json_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            sanitized[key] = _sanitize_json_schema(value)
        else:
            sanitized[key] = value
    
    return sanitized


def _convert_openai_to_google(body: dict, model_id: str) -> dict:
    """Convert OpenAI format to Google-native format (only for Google providers).

    Google API supports only 'user' and 'model' roles.
    Mapping:
    - system → user (prepended as user message)
    - user → user
    - assistant → model
    - tool → user (tool results sent as user messages)
    """
    messages = body.get("messages", [])
    contents = []
    for msg in messages:
        openai_role = msg.get("role", "user")
        content = msg.get("content", "")

        # Map OpenAI roles to Google roles
        if openai_role == "assistant":
            google_role = "model"
        elif openai_role in ("system", "tool", "user"):
            google_role = "user"
        else:
            google_role = "user"

        parts = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    parts.append({"text": item.get("text", "")})
                elif item.get("type") == "image_url":
                    url_data = item.get("image_url", {}).get("url", "")
                    if url_data.startswith("data:"):
                        parts.append({
                            "inline_data": {
                                "mime_type": url_data.split(";")[0].split(":")[1],
                                "data": url_data.split(",", 1)[1],
                            }
                        })
                    else:
                        parts.append({"file_data": {"file_uri": url_data}})

        contents.append({"role": google_role, "parts": parts})

    result = {"contents": contents, "generationConfig": {"temperature": body.get("temperature", 0.7)}}
    
    # Convert tools from OpenAI format to Google format
    if "tools" in body:
        function_declarations = []
        for tool in body["tools"]:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                declaration = {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                }
                if "parameters" in func:
                    declaration["parameters"] = _sanitize_json_schema(func["parameters"])
                function_declarations.append(declaration)
        
        if function_declarations:
            result["tools"] = [{"function_declarations": function_declarations}]
    
    return result


def _convert_google_to_openai_response(google_resp: dict) -> dict:
    """Convert Google non-streaming response to OpenAI format.
    
    Supports both text content and function calls (tool_calls).
    """
    candidates = google_resp.get("candidates", [])
    usage = google_resp.get("usageMetadata", {})

    choices = []
    for c in candidates:
        content_parts = c.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in content_parts if "text" in p)
        finish = c.get("finishReason", "STOP")
        finish_map = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter", "OTHER": "stop"}
        
        # Check for function calls in parts
        tool_calls = []
        for idx, part in enumerate(content_parts):
            if "functionCall" in part:
                fc = part["functionCall"]
                args = fc.get("args", {})
                tool_call = {
                    "index": idx,
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(args) if args else "{}",
                    }
                }
                if "id" in fc:
                    tool_call["id"] = fc["id"]
                tool_calls.append(tool_call)
        
        message = {"role": "assistant", "content": text}
        if tool_calls:
            message["tool_calls"] = tool_calls
        
        choice = {
            "index": c.get("index", 0),
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else finish_map.get(finish, "stop"),
        }
        choices.append(choice)

    return {
        "id": google_resp.get("responseId", ""),
        "object": "chat.completion",
        "model": google_resp.get("modelVersion", ""),
        "choices": choices,
        "usage": {
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
        },
    }


def _convert_google_sse_chunk_to_openai(google_data: dict) -> dict | None:
    """Convert a single Google SSE data chunk to OpenAI streaming format.

    Supports both text content and function calls (tool_calls).
    Returns None if the chunk has no usable content (empty text, no function calls, no finish).
    """
    candidates = google_data.get("candidates", [])
    if not candidates:
        return None

    c = candidates[0]
    content_parts = c.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in content_parts if "text" in p)
    finish = c.get("finishReason")
    finish_map = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter", "OTHER": "stop"}

    delta = {}
    if text:
        delta["content"] = text

    # Check for function calls in parts and convert to OpenAI tool_calls format
    tool_calls = []
    for idx, part in enumerate(content_parts):
        if "functionCall" in part:
            fc = part["functionCall"]
            # Google format: {"name": "...", "args": {...}, "id": "..."}
            # OpenAI format: {"index": 0, "type": "function", "function": {"name": "...", "arguments": "{...}"}, "id": "..."}
            args = fc.get("args", {})
            tool_call = {
                "index": idx,
                "type": "function",
                "function": {
                    "name": fc.get("name", ""),
                    "arguments": json.dumps(args) if args else "{}",
                }
            }
            if "id" in fc:
                tool_call["id"] = fc["id"]
            tool_calls.append(tool_call)

    if tool_calls:
        delta["tool_calls"] = tool_calls

    choice = {"index": c.get("index", 0), "delta": delta}
    
    # Set finish_reason: if function calls present, use "tool_calls", otherwise use mapped reason
    if finish:
        if tool_calls:
            choice["finish_reason"] = "tool_calls"
        else:
            choice["finish_reason"] = finish_map.get(finish, "stop")
    elif not text and not tool_calls:
        return None  # skip empty chunks with no finish

    chunk = {
        "id": google_data.get("responseId", ""),
        "object": "chat.completion.chunk",
        "model": google_data.get("modelVersion", ""),
        "choices": [choice],
    }

    usage = google_data.get("usageMetadata")
    if usage:
        chunk["usage"] = {
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
        }

    return chunk
