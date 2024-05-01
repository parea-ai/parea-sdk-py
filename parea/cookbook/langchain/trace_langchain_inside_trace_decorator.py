import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI

from parea import Parea, trace
from parea.utils.trace_integrations.langchain import PareaAILangchainTracer

load_dotenv()

oai_client = OpenAI()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
handler = PareaAILangchainTracer()
p.wrap_openai_client(oai_client)

llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
prompt = ChatPromptTemplate.from_messages([("user", "{input}")])
chain = prompt | llm | StrOutputParser()


@trace
def main():
    programming_language = (
        oai_client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Suggest one programming languages"}]).choices[0].message.content
    )

    return chain.invoke(
        {"input": f"Write a Hello World program in {programming_language}."},
        config={"callbacks": [handler]},
    )


if __name__ == "__main__":
    print(main())
