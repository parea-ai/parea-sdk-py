import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

from parea import Parea
from parea.schemas import UseDeployedPrompt, UseDeployedPromptResponse
from parea.utils.trace_integrations.langchain import PareaAILangchainTracer

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
handler = PareaAILangchainTracer()


CONTEXT = """Company: Nike. 2023
FORM 10-K 35
OPERATING SEGMENTS
As discussed in Note 15 2014 Operating Segments and Related Information in the accompanying Notes to the Consolidated Financial Statements, our operating segments are evidence of the structure of the Company's internal organization. The NIKE Brand segments are defined by geographic regions for operations participating in NIKE Brand sales activity.
The breakdown of Revenues is as follows:
\n\n(Dollars in millions)
\n\nFISCAL 2023 FISCAL 2022
\n\n% CHANGE\n\n% CHANGE EXCLUDING CURRENCY (1) CHANGES FISCAL 2021\n\n% CHANGE\n\n
North America Europe, Middle East & Africa Greater China\n\n$\n\n21,608 $ 13,418 7,248\n\n18,353 12,479 7,547\n\n18 % 8 % -4 %\n\n18 % $ 21 % 4 %\n\n17,179 11,456 8,290\n\n7 % 9 % -9 %\n\nAsia Pacific & Latin America Global Brand Divisions\n\n(3)\n\n(2)\n\n6,431 58\n\n5,955 102\n\n8 % -43 %\n\n17 % -43 %\n\n5,343 25\n\n11 % 308 %\n\nTOTAL NIKE BRAND Converse\n\n$\n\n48,763 $ 2,427\n\n44,436 2,346\n\n10 % 3 %\n\n16 % $ 8 %\n\n42,293 2,205\n\n5 % 6 %\n\n(4)\n\nCorporate TOTAL NIKE, INC. REVENUES\n\n$\n\n27\n\n51,217 $\n\n(72) 46,710\n\n— 10 %\n\n— 16 % $\n\n40 44,538\n\n— 5 %"""


def get_answer_prompt() -> ChatPromptTemplate:
    # fetched.prompt.raw_messages = [
    #     {
    #         "content": "Use the following pieces of context from Nike's financial 10k filings dataset to answer the question. "
    #                    "Do not make up an answer if no context is provided to help answer it."
    #                    "\n\nContext:\n---------\n{context}\n\n---------\nQuestion: {question}\n---------\n\nAnswer:",
    #         "role": "user",
    #     }
    # ]
    fetched: UseDeployedPromptResponse = p.get_prompt(UseDeployedPrompt(deployment_id="p-JTDYylldIrMbMisT70DJZ"))
    # use the raw messages since it has the templated variables which will be filled in when we invoke the prompt
    answer_prompt = ChatPromptTemplate.from_messages([(message["role"], message["content"]) for message in fetched.prompt.raw_messages])
    return answer_prompt


def get_summary_prompt() -> PromptTemplate:
    # fetched.prompt.raw_messages = [{'content': 'Compile and summarize the following content: {content}', 'role': 'user'}]
    fetched: UseDeployedPromptResponse = p.get_prompt(UseDeployedPrompt(deployment_id="p-OGWAo6yvVKr1hUBY6bmHw"))
    # use the raw messages since it has the templated variables which will be filled in when we invoke the prompt
    summary_prompt = PromptTemplate(
        template=fetched.prompt.raw_messages[0]["content"],
        input_variables=["content"],
    )
    return summary_prompt


def main(question):
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
    answer_prompt = get_answer_prompt()
    summary_prompt = get_summary_prompt()
    answer_chain = answer_prompt | llm | StrOutputParser()
    summary_chain = summary_prompt | llm | StrOutputParser()
    answer = answer_chain.invoke(
        {
            "context": CONTEXT,
            "question": question,
        },
        config={"callbacks": [handler]},
    )
    summary = summary_chain.invoke(
        {"content": answer},
        config={"callbacks": [handler]},
    )
    return summary


if __name__ == "__main__":
    response = main(question="Which operating segment contributed least to total Nike brand revenue in fiscal 2023?")
    print(response)
