from parea.evals.utils import call_openai
from parea.schemas.models import Log


def goal_success_ratio(log: Log) -> float:
    """Returns the average amount of turns the user had to converse with the AI to reach their goals."""
    messages = [m.to_dict() for m in log.configuration.messages]

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
            model="gpt-4",
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
