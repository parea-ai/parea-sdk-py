from attrs import asdict, define, field

from parea.api_client import HTTPClient
from parea.schemas.models import Completion, CompletionResponse, UseDeployedPrompt, UseDeployedPromptResponse

COMPLETION_ENDPOINT = "/completion"
DEPLOYED_PROMPT_ENDPOINT = "/deployed-prompt"


@define
class Parea:
    api_key: str = field(init=True, default="")
    _client: HTTPClient = field(init=False, default=HTTPClient())

    def completion(self, data: Completion) -> CompletionResponse:
        r = self._client.request(
            "POST",
            COMPLETION_ENDPOINT,
            data=asdict(data),
            authorization=self.api_key,
        )
        return CompletionResponse(**r.json())

    async def acompletion(self, data: Completion) -> CompletionResponse:
        r = await self._client.request_async(
            "POST",
            COMPLETION_ENDPOINT,
            data=asdict(data),
            authorization=self.api_key,
        )
        return CompletionResponse(**r.json())

    def get_prompt(self, data: UseDeployedPrompt) -> UseDeployedPromptResponse:
        r = self._client.request(
            "POST",
            DEPLOYED_PROMPT_ENDPOINT,
            data=asdict(data),
            authorization=self.api_key,
        )
        return UseDeployedPromptResponse(**r.json())

    async def aget_prompt(self, data: UseDeployedPrompt) -> UseDeployedPromptResponse:
        r = await self._client.request_async(
            "POST",
            DEPLOYED_PROMPT_ENDPOINT,
            data=asdict(data),
            authorization=self.api_key,
        )
        return UseDeployedPromptResponse(**r.json())
