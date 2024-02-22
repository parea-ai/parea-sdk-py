import asyncio
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from parea import Parea
from parea.utils.trace_integrations.langchain import PareaAILangchainTracer

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
handler = PareaAILangchainTracer()

llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
prompt = ChatPromptTemplate.from_messages([("user", "{input}")])
chain = prompt | llm | StrOutputParser()


def main():
    return chain.invoke(
        {"input": "Write a Hello World program in Python using FastAPI."},
        config={"callbacks": [handler]},
    )


def amain():
    return chain.ainvoke(
        {"input": "Write a Hello World program in Python using FastAPI."},
        config={"callbacks": [handler]},
    )


if __name__ == "__main__":
    print(main())
    print(asyncio.run(amain()))
