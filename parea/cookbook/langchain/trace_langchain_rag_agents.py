import os

from dotenv import load_dotenv
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent, create_retriever_tool
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from parea import Parea
from parea.utils.trace_integrations.langchain import PareaAILangchainTracer

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))

loader = TextLoader("../data/state_of_the_union.txt")


documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()
tool = create_retriever_tool(
    retriever,
    "search_state_of_union",
    "Searches and returns documents regarding the state-of-the-union.",
)
tools = [tool]


llm = ChatOpenAI(temperature=0)

agent_executor = create_conversational_retrieval_agent(llm, tools)


def main():
    result = agent_executor({"input": "what did the president say about kentaji brown jackson in the most recent state of the union?"}, callbacks=[PareaAILangchainTracer()])
    print(result)


if __name__ == "__main__":
    main()
