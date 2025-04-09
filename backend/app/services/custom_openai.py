from langchain_openai.chat_models.base import (
    _convert_delta_to_message_chunk as original_convert_delta_to_message_chunk,
)
from langchain_openai import ChatOpenAI
from typing import Any, Dict, Mapping, Type, Optional, Union
from langchain_core.messages import BaseMessageChunk, AIMessageChunk
from langchain_core.outputs import ChatResult
import openai


def patched_convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    result = original_convert_delta_to_message_chunk(_dict, default_class)

    if isinstance(result, AIMessageChunk):
        if "model_extra" in _dict and "reasoning_content" in _dict["model_extra"]:
            if not result.additional_kwargs:
                result.additional_kwargs = {}
            result.additional_kwargs["reasoning_content"] = _dict["model_extra"][
                "reasoning_content"
            ]
        elif "reasoning_content" in _dict:
            if not result.additional_kwargs:
                result.additional_kwargs = {}
            result.additional_kwargs["reasoning_content"] = _dict["reasoning_content"]

    return result


class ChatOpenAIWithReasoning(ChatOpenAI):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "model_kwargs" not in kwargs or not kwargs["model_kwargs"]:
            self.model_kwargs = {"extra_body": {"enable_enhanced_generation": True}}
        elif "extra_body" not in self.model_kwargs:
            self.model_kwargs["extra_body"] = {"enable_enhanced_generation": True}
        elif "enable_enhanced_generation" not in self.model_kwargs["extra_body"]:
            self.model_kwargs["extra_body"]["enable_enhanced_generation"] = True

    def _create_chat_result(
        self,
        response: Union[dict, openai.BaseModel],
        generation_info: Optional[Dict] = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)

        if isinstance(response, dict):
            choices = response.get("choices", [])
            if choices and "message" in choices[0]:
                message = choices[0]["message"]
                if (
                    "model_extra" in message
                    and "reasoning_content" in message["model_extra"]
                ):
                    result.generations[0].message.additional_kwargs[
                        "reasoning_content"
                    ] = message["model_extra"]["reasoning_content"]
                elif "reasoning_content" in message:
                    result.generations[0].message.additional_kwargs[
                        "reasoning_content"
                    ] = message["reasoning_content"]

        return result


import langchain_openai.chat_models.base

langchain_openai.chat_models.base._convert_delta_to_message_chunk = (
    patched_convert_delta_to_message_chunk
)
