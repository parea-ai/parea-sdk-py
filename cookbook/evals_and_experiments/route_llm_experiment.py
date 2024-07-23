import json
import os

from dotenv import load_dotenv
from routellm.controller import Controller

from parea import Parea, trace, trace_insert
from parea.schemas import EvaluationResult, Log, LLMInputs, Completion, Message, Role, ModelParams

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

ROUTER = "mf"
COST_THRESHOLD = 0.11593
# This tells RouteLLM to use the MF router with a cost threshold of 0.11593
RMODEL = f"router-{ROUTER}-{COST_THRESHOLD}"
STRONG_MODEL = "gpt-4o"
WEAK_MODEL = "groq/llama3-70b-8192"
client = Controller(
    routers=[ROUTER],
    strong_model=STRONG_MODEL,
    weak_model=WEAK_MODEL,
)
p = Parea(api_key=os.getenv("PAREA_API_KEY"))
p.wrap_openai_client(client)

questions = [
    {"question": "Write a function that takes a string as input and returns the string reversed."},
    {"question": "Write a haiku about a sunset."},
    {"question": "Write a cold email to a VP of Eng selling them on OpenAI's API."},
    {"question": "What's the largest city in Germany?"},
]


def llm_judge(log: Log) -> EvaluationResult:
    try:
        response = p.completion(
            data=Completion(
                llm_configuration=LLMInputs(
                    model="gpt-4o-mini",
                    messages=[
                        Message(
                            role=Role.user,
                            content=f"""[Instruction]\nPlease act as an impartial judge and evaluate the quality and
                    correctness of the response provided. Be as objective as possible. Respond in JSON with two fields: \n
                    \t 1. score: int = a number from a scale of 0 to 5; 5 being great and 0 being bad.\n
                    \t 2. reason: str =  explain your reasoning for the selected score.\n\n
                    This is this question asked: QUESTION:\n{log.inputs['question']}\n
                    This is the response you are judging, RESPONSE:\n{log.output}\n\n""",
                        )
                    ],
                    model_params=ModelParams(response_format={"type": "json_object"}),
                ),
            )
        )
        r = json.loads(response.content)
        return EvaluationResult(name="LLMJudge", score=int(r["score"]) / 5, reason=r["reason"])
    except Exception as e:
        return EvaluationResult(name="error-LLMJudge", score=0, reason=f"Error in grading: {e}")


@trace(eval_funcs=[llm_judge])
def answer_llm(question: str) -> str:
    r = client.chat.completions.create(
        model=RMODEL,
        messages=[{"role": "user", "content": f"Answer this question: {question}\n"}],
    )
    trace_insert({"metadata": {"selected_model": r.model}})
    return r.choices[0].message.content


if __name__ == "__main__":
    p.experiment(
        name="RouteLLM",
        data=questions,
        func=answer_llm,
        metadata={
            "router": ROUTER,
            "cost_threshold": str(COST_THRESHOLD),
            "strong_model": STRONG_MODEL,
            "weak_model": WEAK_MODEL,
        },
    ).run()
