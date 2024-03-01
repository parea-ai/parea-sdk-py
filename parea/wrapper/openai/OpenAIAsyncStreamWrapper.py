from openai import AsyncStream


class OpenAIAsyncStreamWrapper:
    def __init__(
        self,
        async_stream: AsyncStream,
        accumulator,
        model_from_response,
        update_accumulator_streaming,
        final_processing_and_logging
    ):
        self._async_stream = async_stream
        self._final_processing_and_logging = final_processing_and_logging
        self._update_accumulator_streaming = update_accumulator_streaming
        self._accumulator = accumulator
        self._model_from_response = model_from_response

    def __getattr__(self, attr):
        # delegate attribute access to the original async_stream
        return getattr(self._async_stream, attr)

    async def __aiter__(self):
        async for chunk in self._async_stream:
            self._update_accumulator_streaming(self._accumulator, self._model_from_response, chunk)
            yield chunk

        self._final_processing_and_logging(self._accumulator, self._model_from_response)