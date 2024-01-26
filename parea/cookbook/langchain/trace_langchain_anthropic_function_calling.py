import os

from dotenv import load_dotenv
from langchain.chains import create_extraction_chain
from langchain.schema import HumanMessage
from langchain_experimental.llms.anthropic_functions import AnthropicFunctions

from parea import Parea
from parea.utils.trace_integrations.langchain import PareaAILangchainTracer

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))

model = AnthropicFunctions(model="claude-2")

functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]


schema = {
    "properties": {
        "name": {"type": "string"},
        "height": {"type": "integer"},
        "hair_color": {"type": "string"},
    },
    "required": ["name", "height"],
}
inp = """Alex is 5 feet tall. Claudia is 1 feet taller Alex and jumps higher than him. Claudia is a brunette and Alex
is blonde."""

chain = create_extraction_chain(schema, model)


def main():
    response = model.predict_messages([HumanMessage(content="whats the weater in boston?")], functions=functions, callbacks=[PareaAILangchainTracer()])
    print(response)
    result = chain.run(inp, callbacks=[PareaAILangchainTracer()])
    print(result)


if __name__ == "__main__":
    main()
