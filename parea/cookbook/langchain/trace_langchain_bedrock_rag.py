import os

import boto3
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.llms.bedrock import Bedrock
from langchain.output_parsers import XMLOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

from parea import Parea
from parea.utils.trace_integrations.langchain import PareaAILangchainTracer

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
handler = PareaAILangchainTracer()


def get_docs():
    loader = TextLoader("../data/2022-letter.txt")
    letter = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=4000, chunk_overlap=100)
    return text_splitter.split_documents(letter)


xml_parser = XMLOutputParser(tags=["insight"])
str_parser = StrOutputParser()

insight_prompt = PromptTemplate(
    template="""

    Human:
    {instructions} : \"{document}\"
    Format help: {format_instructions}.
    Assistant:""",
    input_variables=["instructions", "document"],
    partial_variables={"format_instructions": xml_parser.get_format_instructions()},
)

summary_prompt = PromptTemplate(
    template="""

    Human:
    {instructions} : \"{document}\"
    Assistant:""",
    input_variables=["instructions", "document"],
)

docs = get_docs()
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
bedrock_llm = Bedrock(
    client=bedrock_client,
    model_id="amazon.titan-text-express-v1",
    model_kwargs={"maxTokenCount": 4096, "stopSequences": [], "temperature": 0, "topP": 1},
)

insight_chain = insight_prompt | bedrock_llm | StrOutputParser()
summary_chain = summary_prompt | bedrock_llm | StrOutputParser()


def get_insights(docs):
    insights = []
    for i in range(len(docs)):
        insight = insight_chain.invoke(
            {"instructions": "Provide Key insights from the following text", "document": {docs[i].page_content}}, config={"callbacks": [PareaAILangchainTracer()]}
        )
        insights.append(insight)
    return insights


def main():
    print("Starting")
    insights = get_insights(docs)
    print(insights)
    summary = summary_chain.invoke(
        {
            "instructions": "You will be provided with multiple sets of insights. Compile and summarize these "
            "insights and provide key takeaways in one concise paragraph. Do not use the original xml "
            "tags. Just provide a paragraph with your compiled insights.",
            "document": {"\n".join(insights)},
        },
        config={"callbacks": [PareaAILangchainTracer()]},
    )
    print(summary)
    print("Done")


if __name__ == "__main__":
    main()
