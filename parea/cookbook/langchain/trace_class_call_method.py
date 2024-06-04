import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from parea import Parea, trace
from parea.utils.trace_integrations.langchain import PareaAILangchainTracer

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


class LangChainModule:
    handler = PareaAILangchainTracer()

    def __init__(self):
        self.llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

    def get_chain(self):
        prompt = ChatPromptTemplate.from_messages([("user", "{input}")])
        chain = prompt | self.llm | StrOutputParser()
        return chain

    @trace(name="langchain_caller_call")
    def __call__(self, query: str) -> str:
        chain = self.get_chain()
        return chain.invoke({"input": query}, config={"callbacks": [self.handler]})


class LLMCaller:
    def __init__(self, query: str):
        self.client = LangChainModule()
        self.query = query

    @trace(name="llm_caller_call")
    def __call__(self) -> str:
        return self.client(query=self.query)


@trace
def main(query: str) -> str:
    caller = LLMCaller(query=query)
    return caller()


if __name__ == "__main__":
    result = main("Write a Hello World program in Python using FastAPI.")
    print(result)
