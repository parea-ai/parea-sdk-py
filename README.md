# Parea Python SDK

<div align="center">

[![Build status](https://github.com/parea-ai/parea-sdk/workflows/build/badge.svg?branch=master&event=push)](https://github.com/parea-ai/parea-sdk/actions?query=workflow%3Abuild)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/parea-ai/parea-sdk/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/parea-ai/parea-sdk/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/parea-ai/parea-sdk/releases)
[![License](https://img.shields.io/github/license/parea-ai/parea-sdk)](https://github.com/parea-ai/parea-sdk/blob/master/LICENSE)

Parea python sdk

</div>

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

A fully locally working cookbook can be found [here](parea/cookbook/tracing_and_evaluating_openai_endpoint.py).
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

### Run Experiments

You can run an experiment for your LLM application by defining the `Experiment` class and passing it the name, the data and the
function you want to run. You need annotate the function with the `trace` decorator to trace its inputs, outputs, latency, etc.
as well as to specify which evaluation functions should be applied to it (as shown above).

```python
from parea import Experiment

Experiment(
    name="Experiment Name",        # Name of the experiment (str)
    data=[{"n": "10"}],            # Data to run the experiment on (list of dicts)
    func=function_to_evaluate,     # Function to run (callable)
)
```

Then you can run the experiment by using the `experiment` command and give it the path to the python file.
This will run your experiment with the specified inputs and create a report with the results which can be viewed under
the [Experiments tab](https://app.parea.ai/experiments).

```bash
parea experiment <path/to/experiment_file.py>
```

Full working example in our [docs](https://docs.parea.ai/evaluation/offline/experiments).

## Debugging Chains & Agents

You can iterate on your chains & agents much faster by using a local cache. This will allow you to make changes to your
code & prompts without waiting for all previous, valid LLM responses. Simply add these two lines to the beginning your
code and start
[a local redis cache](https://redis.io/docs/getting-started/install-stack/):

```python
from parea import Parea, RedisCache

Parea(cache=RedisCache())
```

Above will use the default redis cache at `localhost:6379` with no password. You can also specify your redis database
by:

```python
from parea import Parea, RedisCache

cache = RedisCache(
  host=os.getenv("REDIS_HOST", "localhost"),  # default value
  port=int(os.getenv("REDIS_PORT", 6379)),  # default value
  password=os.getenv("REDIS_PASSWORT", None)  # default value
)
Parea(cache=cache)
```

If you set `cache = None` for `Parea`, no cache will be used.

### Benchmark your LLM app across many inputs

You can benchmark your LLM app across many inputs by using the `benchmark` command. This will run your the entry point
of your app with the specified inputs and create a report with the results.

```bash
parea benchmark --func app:main --csv_path benchmark.csv
```

The CSV file will be used to fill in the arguments to your function. The report will be a CSV file of all the traces. If
you
set your Parea API key, the traces will also be logged to the Parea dashboard. Note, for this feature you need to have a
redis cache running. Please, raise a GitHub issue if you would like to use this feature without a redis cache.

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

## Use a deployed prompt

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

### Open source community features

Ready-to-use [Pull Requests templates](https://github.com/parea-ai/parea-sdk/blob/master/.github/PULL_REQUEST_TEMPLATE.md)
and several [Issue templates](https://github.com/parea-ai/parea-sdk/tree/master/.github/ISSUE_TEMPLATE).

- Files such as: `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `SECURITY.md` are generated automatically.
- [Semantic Versions](https://semver.org/) specification
  with [`Release Drafter`](https://github.com/marketplace/actions/release-drafter).

## 🛡 License

[![License](https://img.shields.io/github/license/parea-ai/parea-sdk)](https://github.com/parea-ai/parea-sdk/blob/master/LICENSE)

This project is licensed under the terms of the `Apache Software License 2.0` license.
See [LICENSE](https://github.com/parea-ai/parea-sdk/blob/master/LICENSE) for more details.

## 📃 Citation

```bibtex
@misc{parea-sdk,
  author = {joel-parea-ai},
  title = {Parea python sdk},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/parea-ai/parea-sdk}}
}
```
