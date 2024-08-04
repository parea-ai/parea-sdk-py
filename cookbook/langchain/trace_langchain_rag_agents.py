import os

from dotenv import load_dotenv
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from parea import Parea
from parea.utils.trace_integrations.langchain import PareaAILangchainTracer2

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))

loader = TextLoader("../assets/data/state_of_the_union.txt")


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
    result = agent_executor.invoke(
        {"input": "what did the president say about kentaji brown jackson in the most recent state of the union?"}, config={"callbacks": [PareaAILangchainTracer2()]}
    )
    print(result)


if __name__ == "__main__":
    main()
