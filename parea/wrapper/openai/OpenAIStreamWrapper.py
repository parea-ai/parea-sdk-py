from openai import Stream


class OpenAIStreamWrapper:
    def __init__(
        self,
        stream: Stream,
        accumulator,
        model_from_response,
        update_accumulator_streaming,
        final_processing_and_logging
    ):
        self._stream = stream
        self._final_processing_and_logging = final_processing_and_logging
        self._update_accumulator_streaming = update_accumulator_streaming
        self._accumulator = accumulator
        self._model_from_response = model_from_response

    def __getattr__(self, attr):
        # delegate attribute access to the original async_stream
        return getattr(self._stream, attr)

    def __iter__(self):
        for chunk in self._stream:
            self._update_accumulator_streaming(self._accumulator, self._model_from_response, chunk)
            yield chunk

        self._final_processing_and_logging(self._accumulator, self._model_from_response)