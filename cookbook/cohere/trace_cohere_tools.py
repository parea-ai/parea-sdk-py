import os

import cohere
from dotenv import load_dotenv

from parea import Parea
from parea.utils.universal_encoder import json_dumps

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
p.wrap_cohere_client(co)


def web_search(query: str) -> list[dict]:
    # your code for performing a web search goes here
    return [{"url": "https://en.wikipedia.org/wiki/Ontario", "text": "The capital of Ontario is Toronto, ..."}]


web_search_tool = {
    "name": "web_search",
    "description": "performs a web search with the specified query",
    "parameter_definitions": {"query": {"description": "the query to look up", "type": "str", "required": True}},
}

message = "Who is the mayor of the capital of Ontario?"
model = "command-r-plus"

# STEP 2: Check what tools the model wants to use and how

res = co.chat(model=model, message=message, force_single_step=False, tools=[web_search_tool])

# as long as the model sends back tool_calls,
# keep invoking tools and sending the results back to the model
while res.tool_calls:
    print(res.text)  # This will be an observation and a plan with next steps
    tool_results = []
    for call in res.tool_calls:
        # use the `web_search` tool with the search query the model sent back
        web_search_results = {"call": call, "outputs": web_search(call.parameters["query"])}
        tool_results.append(web_search_results)

    # call chat again with tool results
    res = co.chat(model="command-r-plus", chat_history=res.chat_history, message="", force_single_step=False, tools=[web_search_tool], tool_results=tool_results)

print(res.text)  # "The mayor of Toronto, the capital of Ontario is Olivia Chow"


# tool descriptions that the model has access to
tools = [
    {
        "name": "query_daily_sales_report",
        "description": "Connects to a database to retrieve overall sales volumes and sales information for a given day.",
        "parameter_definitions": {"day": {"description": "Retrieves sales data for this day, formatted as YYYY-MM-DD.", "type": "str", "required": True}},
    },
    {
        "name": "query_product_catalog",
        "description": "Connects to a a product catalog with information about all the products being sold, including categories, prices, and stock levels.",
        "parameter_definitions": {"category": {"description": "Retrieves product information data for all products in this category.", "type": "str", "required": True}},
    },
]

# preamble containing instructions about the task and the desired style for the output.
preamble = """
## Task & Context
You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.
"""

# user request
message = "Can you provide a sales summary for 29th September 2023, and also give me some details about the products in the 'Electronics' category, for example their prices and stock levels?"

response = co.chat(message=message, force_single_step=True, tools=tools, preamble=preamble, model="command-r")
print("The model recommends doing the following tool calls:")
print("\n".join(str(tool_call) for tool_call in response.tool_calls))

tool_results = []
# Iterate over the tool calls generated by the model
for tool_call in response.tool_calls:
    # here is where you would call the tool recommended by the model, using the parameters recommended by the model
    output = {"output": f"functions_map[{tool_call.name}]({tool_call.parameters})"}
    # store the output in a list
    outputs = [output]
    # store your tool results in this format
    tool_results.append({"call": tool_call, "outputs": outputs})


print("Tool results that will be fed back to the model in step 4:")
print(json_dumps(tool_results, indent=4))

response = co.chat(message=message, tools=tools, tool_results=tool_results, preamble=preamble, model="command-r", temperature=0.3, force_single_step=True)


print("Final answer:")
print(response.text)
