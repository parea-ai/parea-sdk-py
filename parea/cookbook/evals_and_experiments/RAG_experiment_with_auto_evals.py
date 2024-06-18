from typing import Any

import json
import os
import re
from functools import lru_cache

import markdownify
import requests
from dotenv import load_dotenv
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from parea import Parea, trace, trace_insert
from parea.evals.general import answer_matches_target_llm_grader_factory
from parea.evals.rag import (
    answer_context_faithfulness_binary_factory,
    answer_context_faithfulness_statement_level_factory,
    context_query_relevancy_factory,
    percent_target_supported_by_context_factory,
)

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
p = Parea(api_key=os.getenv("PAREA_API_KEY"))
p.wrap_openai_client(client)

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
text_splitter = TokenTextSplitter(model_name="gpt-3.5-turbo", chunk_size=CHUNK_SIZE, chunk_overlap=200)
embeddings = OpenAIEmbeddings()

MODEL = "gpt-4o"
TOPK = 4
NUM_SECTIONS = 20
CODA_QA_FILE_LOC = "https://gist.githubusercontent.com/wong-codaio/b8ea0e087f800971ca5ec9eef617273e/raw/39f8bd2ebdecee485021e20f2c1d40fd649a4c77/articles.json"
CODA_QA_PAIRS_LOC = "https://gist.githubusercontent.com/nelsonauner/2ef4d38948b78a9ec2cff4aa265cff3f/raw/c47306b4469c68e8e495f4dc050f05aff9f997e1/qa_pairs_coda_data.jsonl"


@lru_cache()
def get_coda_qa_content(CODA_QA_FILE_LOC) -> list[Document]:
    coda_qa_content_data = requests.get(CODA_QA_FILE_LOC).json()
    return [
        Document(page_content=section.strip(), metadata={"doc_id": row["id"], "markdown": section.strip()})
        for row in coda_qa_content_data
        for section in re.split(r"(.*\n=+\n)", markdownify.markdownify(row["body"]))
        if section.strip() and not re.match(r".*\n=+\n", section)
    ]


@lru_cache()
def get_coda_qa_pairs_raw(CODA_QA_PAIRS_LOC):
    coda_qa_pairs = requests.get(CODA_QA_PAIRS_LOC)
    qa_pairs = [json.loads(line) for line in coda_qa_pairs.text.split("\n") if line]
    return [{"question": qa_pair["input"], "doc_metadata": qa_pair["metadata"], "target": qa_pair["expected"]} for qa_pair in qa_pairs]


class DocumentRetriever:
    def __init__(self):
        coda_qa_content_data = get_coda_qa_content(CODA_QA_FILE_LOC)
        documents = text_splitter.split_documents(coda_qa_content_data)
        vectorstore = FAISS.from_documents(documents, embeddings)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": TOPK})

    @trace
    def retrieve_top_k(self, question: str) -> list[Document]:
        trace_insert({"metadata": {"source_file": CODA_QA_FILE_LOC}})
        return self.retriever.invoke(question)


@trace(
    eval_funcs=[
        # Evals that do not need a target
        answer_context_faithfulness_binary_factory(),
        answer_context_faithfulness_statement_level_factory(),
        context_query_relevancy_factory(context_fields=["context"]),
        # Eval that need a target
        answer_matches_target_llm_grader_factory(model="gpt-4o"),
        percent_target_supported_by_context_factory(context_fields=["context"]),
    ]
)
def generate_answer_from_docs(question: str, context: str) -> str:
    return (
        client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": f"""Use the following pieces of context to answer the question.
                    Do not make up an answer if no context is provided to help answer it.
                    \n\nContext:\n---------\n{context}\n\n---------\nQuestion: {question}\n---------\n\nAnswer:
                    """,
                }
            ],
        )
        .choices[0]
        .message.content
    )


@trace
def main(question: str, doc_metadata: dict[str, Any]) -> str:
    relevant_sections = DocumentRetriever().retrieve_top_k(question)
    context = "\n\n".join(doc.page_content for doc in relevant_sections)
    trace_insert({"metadata": doc_metadata})
    return generate_answer_from_docs(question, context)


if __name__ == "__main__":
    metadata = dict(model=MODEL, topk=str(TOPK), num_sections=str(NUM_SECTIONS), chunk_size=str(CHUNK_SIZE), chunk_overlap=str(CHUNK_OVERLAP))
    qa_pairs = get_coda_qa_pairs_raw(CODA_QA_PAIRS_LOC)
    p.experiment(
        name="Coda_RAG",
        data=qa_pairs[:NUM_SECTIONS],
        func=main,
        metadata=metadata,
    ).run()
