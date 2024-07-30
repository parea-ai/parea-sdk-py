import os

import cohere
from dotenv import load_dotenv

from parea import Parea

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
p.wrap_cohere_client(co)

response = co.chat(
    model="command-r-plus",
    preamble="You are a helpful assistant talking in JSON.",
    message="Generate a JSON describing a person, with the fields 'name' and 'age'",
    response_format={"type": "json_object"},
)
print(response)
print("\n\n")

response = co.chat(message="Who discovered gravity?")
print(response)
print("\n\n")
#
docs = [
    "Carson City is the capital city of the American state of Nevada.",
    "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
    "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages.",
    "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
    "Capital punishment (the death penalty) has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
]
response = co.rerank(
    model="rerank-english-v3.0",
    query="What is the capital of the United States?",
    documents=docs,
    top_n=3,
)
print(response)
print("\n\n")


response = co.chat(
    model="command-r-plus",
    message="Where do the tallest penguins live?",
    documents=[
        {"title": "Tall penguins", "snippet": "Emperor penguins are the tallest."},
        {"title": "Penguin habitats", "snippet": "Emperor penguins only live in Antarctica."},
        {"title": "What are animals?", "snippet": "Animals are different from plants."},
    ],
)
print(response)
print("\n\n")

response = co.chat(model="command-r-plus", message="Who is more popular: Nsync or Backstreet Boys?", search_queries_only=True)
print(response)
print("\n\n")

response = co.chat(model="command-r-plus", message="Who is more popular: Nsync or Backstreet Boys?", connectors=[{"id": "web-search"}])
print(response)
print("\n\n")

for event in co.chat_stream(message="Who discovered gravity?"):
    print(event)
