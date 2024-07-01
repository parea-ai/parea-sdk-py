# Evaluate Your AI Application with Parea's Python SDK

<div align="center">

[![Build status](https://github.com/parea-ai/parea-sdk/workflows/build/badge.svg?branch=master&event=push)](https://github.com/parea-ai/parea-sdk/actions?query=workflow%3Abuild)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/parea-ai/parea-sdk/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/parea-ai/parea-sdk/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/parea-ai/parea-sdk/releases)
[![License](https://img.shields.io/github/license/parea-ai/parea-sdk)](https://github.com/parea-ai/parea-sdk/blob/main/LICENSE)

</div>

[Parea AI](https://www.parea.ai) provides a SDK to evaluate & monitor your AI applications.

[Python SDK Docs](https://docs.parea.ai/api-reference/sdk/python)

## Installation

```bash
pip install -U parea-ai
```

or install with `Poetry`

```bash
poetry add parea-ai
```


## Evaluating Your LLM App

Testing your AI app means to execute it over a dataset and score it with an evaluation function.
This is done in Parea by defining & running experiments.
Below you can see can example of how to test a greeting bot with the Levenshtein distance metric. 

```python
from parea import Parea, trace
from parea.evals.general import levenshtein

p = Parea(api_key="<<PAREA_API_KEY>>")  # replace with Parea AI API key

# use the trace decorator to score the output with the Levenshtein distance  
@trace(eval_funcs=[levenshtein])
def greeting(name: str) -> str:
    return f"Hello {name}"

data = [
    {"name": "Foo", "target": "Hi Foo"},
    {"name": "Bar", "target": "Hello Bar"},
]

p.experiment(
    name="Greeting",
    data=data,
    func=greeting,
).run()
```

In the snippet above, we used the `trace` decorator to capture any inputs & outputs of the function.
This decorator also enables to score the output by executing the `levenshtein` eval in the background.
Then, we defined an experiment via `p.experiment` to evaluate our function (`greeting`) over  a dataset (here a list of dictionaries).
Calling `run` will execute the experiment, and create a report of outputs, scores & traces for any sample of the dataset.
You can find a link to the executed experiment [here](). (todo: fill-in experiment) 



### More Resources

Read more about how to run & analyze experiments.

### Running Evals


### Writing Evals


## Evaluating Your LLM App

You can evaluate any step of your LLM app by wrapping it with a decorator, called `trace`, and specifying the evaluation
function(s).
The scores associated with the traces will be logged to the Parea [dashboard](https://app.parea.ai/logs) and/or in a
local CSV file if you don't have a Parea API key.

Evaluation functions receive an argument `log` (of type [Log](parea/schemas/models.py)) and should return a
float. You don't need to start from scratch, there are pre-defined evaluation
functions for [general purpose](parea/evals/general),
[chat](parea/evals/chat), [RAG](parea/evals/rag), and [summarization](parea/evals/summary) apps :)

You can define evaluation functions locally or use the ones you have deployed to
Parea's [Test Hub](https://app.parea.ai/test-hub).
If you choose the latter option, the evaluation happens asynchronously and non-blocking.

A fully locally working cookbook can be found [here](cookbook/openai/tracing_and_evaluating_openai_endpoint.py).
Alternatively, you can add the following code to your codebase to get started:

```python
import os
from parea import Parea, InMemoryCache, trace
from parea.schemas.log import Log

Parea(api_key=os.getenv("PAREA_API_KEY"), cache=InMemoryCache())  # use InMemoryCache if you don't have a Parea API key


def locally_defined_eval_function(log: Log) -> float:
  ...


@trace(eval_func_names=['deployed_eval_function_name'], eval_funcs=[locally_defined_eval_function])
def function_to_evaluate(*args, **kwargs) -> ...:
  ...
```



## Logging & Observability

### Automatically log all your LLM call traces

You can automatically log all your LLM traces to the Parea dashboard by setting the `PAREA_API_KEY` environment variable
or specifying it in the `Parea` initialization.
This will help you debug issues your customers are facing by stepping through the LLM call traces and recreating the
issue
in your local setup & code.

```python
from parea import Parea

Parea(
  api_key=os.getenv("PAREA_API_KEY"),  # default value
  cache=...
)
```

### Logging results from LLM providers [Example]

```python
import os

import openai
from dotenv import load_dotenv

from parea import Parea

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

p = Parea(api_key=os.getenv("PAREA_API_KEY"))

x = "Golang"
y = "Fiber"
messages = [{
  "role": "user",
  "content": f"Write a hello world program using {x} and the {y} framework."
}]
model = "gpt-3.5-turbo"
temperature = 0.0


# define your OpenAI call as you would normally and we'll automatically log the results
def main():
  openai.chat.completions.create(model=model, temperature=temperature, messages=messages).choices[0].message.content
```

## Deploying Prompts

Deployed prompts enable collaboration with non-engineers such as product managers & subject-matter experts.
Users can iterate, refine & test prompts on Parea's playground.
After tinkering, you can deploy that prompt which means that it is exposed via an API endpoint to integrate it into your application.

```python
import os

from dotenv import load_dotenv

from parea import Parea
from parea.schemas.models import Completion, UseDeployedPrompt, CompletionResponse, UseDeployedPromptResponse

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))

# You will find this deployment_id in the Parea dashboard
deployment_id = '<DEPLOYMENT_ID>'

# Assuming your deployed prompt's message is:
# {"role": "user", "content": "Write a hello world program using {{x}} and the {{y}} framework."}
inputs = {"x": "Golang", "y": "Fiber"}

# You can easily unpack a dictionary into an attrs class
test_completion = Completion(
  **{
    "deployment_id": deployment_id,
    "llm_inputs": inputs,
    "metadata": {"purpose": "testing"}
  }
)

# By passing in my inputs, in addition to the raw message with unfilled variables {{x}} and {{y}}, 
# you we will also get the filled-in prompt:
# {"role": "user", "content": "Write a hello world program using Golang and the Fiber framework."}
test_get_prompt = UseDeployedPrompt(deployment_id=deployment_id, llm_inputs=inputs)


def main():
  completion_response: CompletionResponse = p.completion(data=test_completion)
  print(completion_response)
  deployed_prompt: UseDeployedPromptResponse = p.get_prompt(data=test_get_prompt)
  print("\n\n")
  print(deployed_prompt)


async def main_async():
  completion_response: CompletionResponse = await p.acompletion(data=test_completion)
  print(completion_response)
  deployed_prompt: UseDeployedPromptResponse = await p.aget_prompt(data=test_get_prompt)
  print("\n\n")
  print(deployed_prompt)
```    



## 🛡 License

[![License](https://img.shields.io/github/license/parea-ai/parea-sdk)](https://github.com/parea-ai/parea-sdk/blob/master/LICENSE)

This project is licensed under the terms of the `Apache Software License 2.0` license.
See [LICENSE](https://github.com/parea-ai/parea-sdk/blob/master/LICENSE) for more details.

## 📃 Citation

```bibtex
@misc{parea-sdk,
  author = {joel-parea-ai,joschkabraun},
  title = {Parea python sdk},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/parea-ai/parea-sdk}}
}
```
