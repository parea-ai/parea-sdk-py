import os
from datetime import datetime
from operator import itemgetter

import boto3
from dotenv import load_dotenv

# LangChain libs
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import RecursiveUrlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.llms.bedrock import Bedrock

# Parea libs
from parea import Parea
from parea.evals import EvalFuncTuple, run_evals_in_thread_and_log
from parea.evals.general import answer_matches_target_llm_grader_factory
from parea.evals.rag import (
    answer_context_faithfulness_binary_factory,
    answer_context_faithfulness_statement_level_factory,
    context_query_relevancy_factory,
    percent_target_supported_by_context_factory,
)
from parea.schemas import LLMInputs, Log
from parea.utils.trace_integrations.langchain import PareaAILangchainTracer

load_dotenv()

# Need to instantiate Parea for tracing and evals
p = Parea(api_key=os.getenv("PAREA_API_KEY"))


bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
bedrock_model = Bedrock(client=bedrock_client, model_id="amazon.titan-text-express-v1", model_kwargs={"maxTokenCount": 4096, "stopSequences": [], "temperature": 0, "topP": 1})
bedrock = {"model_name": "amazon.titan-text-express-v1", "model": bedrock_model, "provider": "bedrock"}
openai_model = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
openai = {"model_name": "gpt-3.5-turbo-16k", "model": openai_model, "provider": "openai"}


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
        vectorstore = Chroma.from_documents(documents, embeddings)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def get_retriever(self):
        return self.retriever


class DocumentationChain:
    def __init__(self, retriever, model, is_bedrock=False):
        self.is_bedrock = is_bedrock
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

        self.chain = (
            # The runnable map here routes the original inputs to a context and a question dictionary to pass to the response generator
            {"context": itemgetter("question") | retriever | self._format_docs, "question": itemgetter("question")}
            | response_generator
        )

    def get_context(self) -> str:
        """Helper to get the context from a retrieval chain, so we can use it for evaluation metrics."""
        return self.context

    def _format_docs(self, docs) -> str:
        if self.is_bedrock:
            docs = docs[0::2]  # avoid context window limit, get every 2nd doc
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

# set up evaluation functions we want to test, provide the name of the relevant fields needed for each eval function
EVALS = [
    # question field only
    EvalFuncTuple(name="matches_target", func=answer_matches_target_llm_grader_factory(question_field="question")),
    # questions field and single context field
    EvalFuncTuple(name="faithfulness_binary", func=answer_context_faithfulness_binary_factory(question_field="question", context_field="context")),
    # questions field and accepts multiple context fields
    EvalFuncTuple(name="faithfulness_statement", func=answer_context_faithfulness_statement_level_factory(question_field="question", context_fields=["context"])),
    EvalFuncTuple(name="relevancy", func=context_query_relevancy_factory(question_field="question", context_fields=["context"])),
    EvalFuncTuple(name="supported_by_context", func=percent_target_supported_by_context_factory(question_field="question", context_fields=["context"])),
]


def create_log(model, question, context, output, target, provider="openai") -> Log:
    """Creates a log object for evaluation metric functions."""
    inputs = {"question": question, "context": context}  # fields named question and context to align w/ eval functions
    log_config = LLMInputs(model=model, provider=provider)  # provider and model needed for precision eval metric
    return Log(configuration=log_config, inputs=inputs, output=output, target=target)


def main():
    model = openai["model"]
    model_name = openai["model_name"]
    provider = openai["provider"]

    # instantiate tracer integration
    handler = PareaAILangchainTracer()
    # set up retriever
    retriever = DocumentRetriever("https://en.wikipedia.org/wiki/New_York_City").get_retriever()
    # set up chain
    dc = DocumentationChain(retriever, model, is_bedrock=provider == "bedrock")

    # iterate through questions and answers
    for question, answer in zip(eval_questions, eval_answers):
        # call chain and attached parea tracer as a callback for logs
        output = dc.get_chain().invoke({"question": question}, config={"callbacks": [handler]})
        print(f"Question: {question}\nAnswer: {output} \n")

        # get parent trace id from the tracer
        parent_trace_id = handler.get_parent_trace_id()
        # after chain is called, get the context for evaluation metric functions
        context = dc.get_context()
        # build log component needed for evaluation metric functions
        log = create_log(model_name, question, context, output, answer, provider)

        # helper function to run evaluation metrics in a thread to avoid blocking return of chain
        run_evals_in_thread_and_log(trace_id=str(parent_trace_id), log=log, eval_funcs=EVALS, verbose=True)


if __name__ == "__main__":
    print("Running evals...")
    main()
    print("Done!")
