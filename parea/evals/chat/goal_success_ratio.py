from typing import Callable, Optional

import json

from parea.evals.utils import call_openai
from parea.schemas.log import Log


def goal_success_ratio_factory(
    use_output: Optional[bool] = False, message_field: Optional[str] = None, model: Optional[str] = "gpt-4", is_azure: Optional[bool] = False
) -> Callable[[Log], float]:
    """
    This factory creates an evaluation function that measures the success ratio of a goal-oriented conversation.
    Typically, a user interacts with a chatbot or AI assistant to achieve specific goals.
    This motivates to measure the quality of a chatbot by counting how many messages a user has to send before they reach their goal.
    One can further break this down by successful and unsuccessful goals to analyze user & LLM behavior.

    Concretely:
    1. Delineate the conversation into segments by splitting them by the goals the user wants to achieve.
    2. Assess if every goal has been reached.
    3. Calculate the average number of messages sent per segment.

    Args:
        is_azure: Whether to use Azure as the model. Defaults to False.
        model: The model which should be used for grading.
        use_output (Optional[bool], optional): Whether to use the output of the log to access the messages. Defaults to False.
        message_field (Optional[str], optional): The name of the field in the log that contains the messages.
            Defaults to None. If None, the messages are taken from the configuration attribute.

    # noqa: DAR201
    # noqa: DAR401
    """
    if use_output and message_field:
        raise ValueError("Only one of use_output and message_field can be set.")

    def goal_success_ratio(log: Log) -> float:
        """Returns the average amount of turns the user had to converse with the AI to reach their goals."""
        if use_output:
            output_list_dicts = json.loads(log.output)
            messages = [m for m in output_list_dicts]
        elif message_field:
            messages = [m for m in log.inputs[message_field]]
        else:
            messages = [m.to_dict() for m in log.configuration.messages]
            if log.output:
                messages.append({"role": "assistant", "content": log.output})

        # need to determine where does a new goal start
        conversation_segments = []
        start_index = 0
        end_index = 3
        while end_index < len(messages):
            user_follows_same_goal = call_openai(
                [
                    {
                        "role": "system",
                        "content": "Look at the conversation and to determine if the user is still following the same goal "
                        "or if they are following a new goal. If they are following the same goal, respond "
                        "SAME_GOAL. Otherwise, respond NEW_GOAL. In any case do not answer the user request!",
                    }
                ]
                + messages[start_index:end_index],
                model=model,
                is_azure=is_azure,
            )

            if user_follows_same_goal == "SAME_GOAL":
                end_index += 2
            else:
                conversation_segments.append(messages[start_index : end_index - 1])
                start_index = end_index - 1
                end_index += 2

        if start_index < len(messages):
            conversation_segments.append(messages[start_index:])

        # for now assume that the user reached their goal in every segment
        # return the average amount of turns the user had to converse with the AI to reach their goals
        return sum([2 / len(segment) for segment in conversation_segments]) / len(conversation_segments)

    return goal_success_ratio
