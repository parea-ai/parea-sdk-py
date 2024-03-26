from __future__ import annotations

from types import TracebackType
from typing import Callable

from anthropic import AsyncMessageStreamManager, MessageStreamManager, Stream
from anthropic.types import Message


class AnthropicStreamWrapper:
    def __init__(self, stream: Stream, accumulator, info_from_response, update_accumulator_streaming, final_processing_and_logging):
        self._stream = stream
        self._final_processing_and_logging = final_processing_and_logging
        self._update_accumulator_streaming = update_accumulator_streaming
        self._accumulator = accumulator
        self._info_from_response = info_from_response

    def __getattr__(self, attr):
        # delegate attribute access to the original async_stream
        return getattr(self._async_stream, attr)

    def __iter__(self):
        for chunk in self._stream:
            self._update_accumulator_streaming(self._accumulator, self._info_from_response, chunk)
            yield chunk

        self._final_processing_and_logging(self._accumulator, self._info_from_response)


class AnthropicAsyncStreamWrapper:
    def __init__(self, stream: Stream, accumulator, info_from_response, update_accumulator_streaming, final_processing_and_logging):
        self._stream = stream
        self._final_processing_and_logging = final_processing_and_logging
        self._update_accumulator_streaming = update_accumulator_streaming
        self._accumulator = accumulator
        self._info_from_response = info_from_response

    def __getattr__(self, attr):
        # delegate attribute access to the original async_stream
        return getattr(self._async_stream, attr)

    async def __aiter__(self):
        async for chunk in self._stream:
            self._update_accumulator_streaming(self._accumulator, self._info_from_response, chunk)
            yield chunk

        self._final_processing_and_logging(self._accumulator, self._info_from_response)


class MessageStreamManagerWrapper(MessageStreamManager):
    def __init__(self, msm_instance: MessageStreamManager, resolve_and_log: Callable):
        self._msm_instance = msm_instance
        self._resolve_and_log = resolve_and_log

    def __getattr__(self, attr):
        if attr != "_private_stream":
            return getattr(self._msm_instance, attr)
        else:
            return self._private_stream

    def __enter__(self):
        self._private_stream = self._msm_instance.__enter__()
        return self._private_stream

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        m: Message = self._private_stream.get_final_message()
        self._resolve_and_log(m)
        return super().__exit__(exc_type, exc, exc_tb)


class MessageAsyncStreamManagerWrapper(AsyncMessageStreamManager):
    def __init__(self, msm_instance: AsyncMessageStreamManager, resolve_and_log: Callable):
        self._msm_instance = msm_instance
        self._resolve_and_log = resolve_and_log

    def __getattr__(self, attr):
        if attr != "_private_stream":
            return getattr(self._msm_instance, attr)
        else:
            return self._private_stream

    async def __aenter__(self):
        self._private_stream = await self._msm_instance.__aenter__()
        return self._private_stream

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        m: Message = await self._private_stream.get_final_message()
        self._resolve_and_log(m)
        return await super().__aexit__(exc_type, exc, exc_tb)
