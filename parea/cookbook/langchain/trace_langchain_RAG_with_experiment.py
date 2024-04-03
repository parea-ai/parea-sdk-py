import os
from datetime import datetime
from operator import itemgetter

from dotenv import load_dotenv
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.vectorstores.pinecone import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone as PineconeClient

from parea import Parea, trace, trace_insert
from parea.evals.general import answer_matches_target_llm_grader_factory
from parea.evals.rag import (
    answer_context_faithfulness_binary_factory,
    answer_context_faithfulness_statement_level_factory,
    context_query_relevancy_factory,
    percent_target_supported_by_context_factory,
)
from parea.utils.trace_integrations.langchain import PareaAILangchainTracer

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
handler = PareaAILangchainTracer()

pinecone = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))


class DocumentRetriever:
    def __init__(self, url: str):
        api_loader = RecursiveUrlLoader(url)
        raw_documents = api_loader.load()

        # Transformer
        doc_transformer = Html2TextTransformer()
        transformed = doc_transformer.transform_documents(raw_documents)

        # Splitter
        text_splitter = TokenTextSplitter(
            model_name="gpt-3.5-turbo",
            chunk_size=2000,
            chunk_overlap=200,
        )
        documents = text_splitter.split_documents(transformed)

        # Define vector store based
        embeddings = OpenAIEmbeddings()
        vectorstore = Pinecone.from_documents(documents, embeddings, index_name=os.getenv("PINECONE_INDEX_NAME"))
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def get_retriever(self):
        return self.retriever


class DocumentationChain:
    def __init__(self, url):
        retriever = DocumentRetriever(url).get_retriever()
        model = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful documentation Q&A assistant, trained to answer questions from the provided context."
                    "\nThe current time is {time}.\n\nRelevant documents will be retrieved in the following messages.",
                ),
                ("system", "{context}"),
                ("human", "{question}"),
            ]
        ).partial(time=str(datetime.now()))

        response_generator = prompt | model | StrOutputParser()

        self.chain = {
            "context": itemgetter("question") | retriever | self._format_docs,
            "question": itemgetter("question"),
        } | response_generator

    def get_context(self) -> str:
        """Helper to get the context from a retrieval chain, so we can use it for evaluation metrics."""
        return self.context

    def _format_docs(self, docs) -> str:
        context = "\n\n".join(doc.page_content for doc in docs)
        # set context as an attribute, so we can access it later
        self.context = context
        return context

    def get_chain(self):
        return self.chain


# EXAMPLE EVALUATION TEST CASES
eval_questions = [
    "What is the population of New York City as of 2020?",
    "Which borough of New York City has the highest population? Only respond with the name of the borough.",
    "What is the economic significance of New York City?",
    "How did New York City get its name?",
    "What is the significance of the Statue of Liberty in New York City?",
]

eval_answers = [
    "8,804,190",
    "Brooklyn",
    """New York City's economic significance is vast, as it serves as the global financial capital, housing Wall
    Street and major financial institutions. Its diverse economy spans technology, media, healthcare, education,
    and more, making it resilient to economic fluctuations. NYC is a hub for international business, attracting
    global companies, and boasts a large, skilled labor force. Its real estate market, tourism, cultural industries,
    and educational institutions further fuel its economic prowess. The city's transportation network and global
    influence amplify its impact on the world stage, solidifying its status as a vital economic player and cultural
    epicenter.""",
    """New York City got its name when it came under British control in 1664. King Charles II of England granted the
    lands to his brother, the Duke of York, who named the city New York in his own honor.""",
    """The Statue of Liberty in New York City holds great significance as a symbol of the United States and its
    ideals of liberty and peace. It greeted millions of immigrants who arrived in the U.S. by ship in the late 19th
    and early 20th centuries, representing hope and freedom for those seeking a better life. It has since become an
    iconic landmark and a global symbol of cultural diversity and freedom.""",
]
# create a dataset of questions and targets
dataset = [{"question": q, "target": t} for q, t in zip(eval_questions, eval_answers)]


@trace(
    eval_funcs=[
        # these are factory functions that return the actual evaluation functions, so we need to call them
        answer_matches_target_llm_grader_factory(),
        answer_context_faithfulness_binary_factory(),
        answer_context_faithfulness_statement_level_factory(),
        context_query_relevancy_factory(context_fields=["context"]),
        percent_target_supported_by_context_factory(context_fields=["context"]),
    ]
)
def main(question: str) -> str:
    dc = DocumentationChain(url="https://en.wikipedia.org/wiki/New_York_City")
    output = dc.get_chain().invoke(
        {"question": question},
        config={"callbacks": [handler]},  # pass the Parea callback handler to the chain
    )
    # insert the context into the trace as an input so that it can be referenced in the evaluation functions
    # context needs to be retrieved after the chain is invoked
    trace_insert({"inputs": {"context": dc.get_context()}})
    print(output)
    return output


if __name__ == "__main__":
    p.experiment(
        name="NYC_Wiki_RAG",
        data=dataset,
        func=main,
    ).run()
