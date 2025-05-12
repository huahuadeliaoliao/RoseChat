"""Customizations for LangChain OpenAI Chat Model integration."""

from langchain_openai.chat_models.base import (
    _convert_delta_to_message_chunk as original_convert_delta_to_message_chunk,
)
from langchain_openai import ChatOpenAI
from typing import Any, Mapping, Type, Dict
from langchain_core.messages import BaseMessageChunk, AIMessageChunk
import langchain_openai.chat_models.base


def patched_convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    """Patched version to extract 'reasoning_content' into additional_kwargs."""
    result = original_convert_delta_to_message_chunk(_dict, default_class)

    if isinstance(result, AIMessageChunk):
        reasoning_content = None
        if "model_extra" in _dict and isinstance(_dict["model_extra"], dict):
            reasoning_content = _dict["model_extra"].get("reasoning_content")
        elif "reasoning_content" in _dict:
            reasoning_content = _dict["reasoning_content"]

        if reasoning_content:
            if not result.additional_kwargs:
                result.additional_kwargs = {}
            result.additional_kwargs["reasoning_content"] = reasoning_content

    return result


class ChatOpenAIWithReasoning(ChatOpenAI):
    """Custom ChatOpenAI class that enables enhanced generation by default."""

    def __init__(self, **kwargs: Any):
        """Initialize the custom chat model.

        Ensures 'enable_enhanced_generation' is set in model_kwargs['extra_body']
        to potentially receive reasoning or other extended outputs.

        Args:
            **kwargs: Keyword arguments passed to the parent ChatOpenAI class.
        """
        super().__init__(**kwargs)

        if "model_kwargs" not in kwargs or not isinstance(kwargs["model_kwargs"], dict):
            self.model_kwargs = {"extra_body": {"enable_enhanced_generation": True}}
        else:
            if "extra_body" not in self.model_kwargs or not isinstance(
                self.model_kwargs.get("extra_body"), dict
            ):
                self.model_kwargs["extra_body"] = {"enable_enhanced_generation": True}
            elif "enable_enhanced_generation" not in self.model_kwargs["extra_body"]:
                self.model_kwargs["extra_body"]["enable_enhanced_generation"] = True


langchain_openai.chat_models.base._convert_delta_to_message_chunk = (
    patched_convert_delta_to_message_chunk
)
