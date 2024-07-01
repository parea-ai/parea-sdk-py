tool_calling_example = {
    "model": "gpt-3.5-turbo-0125",
    "messages": [{"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}],
    "tools": [
        {
            "type": "function",
            "function": {
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
            },
        }
    ],
    "tool_choice": "auto",
}

functions_example = {
    "model": "gpt-3.5-turbo-0125",
    "messages": [
        {
            "role": "system",
            "content": f"You are a sophisticated AI assistant, "
            f"a specialist in user intent detection and interpretation. "
            f"Your task is to perceive and respond to the user's needs, even when they're expressed "
            f"in an indirect or direct manner. You excel in recognizing subtle cues: for example, "
            f"if a user states they are 'hungry', you should assume they are seeking nearby dining "
            f"options such as a restaurant or a cafe. If they indicate feeling 'tired', 'weary', "
            f"or mention a long journey, interpret this as a request for accommodation options like "
            f"hotels or guest houses. However, remember to navigate the fine line of interpretation "
            f"and assumption: if a user's intent is unclear or can be interpreted in multiple ways, "
            f"do not hesitate to politely ask for additional clarification. Make sure to tailor your "
            f"responses to the user based on their preferences and past experiences which can "
            f"be found here: Name: John Doe",
        },
        {"role": "user", "content": "I'm hungry"},
    ],
    "functions": [
        {
            "name": "call_google_places_api",
            "description": f"""
            This function calls the Google Places API to find the top places of a specified type near
            a specific location. It can be used when a user expresses a need (e.g., feeling hungry or tired) or wants to
            find a certain type of place (e.g., restaurant or hotel).
        """,
            "parameters": {"type": "object", "properties": {"place_type": {"type": "string", "description": "The type of place to search for."}}},
            "result": {"type": "array", "items": {"type": "string"}},
        }
    ],
}

simple_example = {
    "model": "gpt-3.5-turbo-0125",
    "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello!"}],
}

simple_example_json = {
    "model": "gpt-3.5-turbo-0125",
    "messages": [{"role": "system", "content": "You are a helpful assistant talking JSON."}, {"role": "user", "content": "Hello!"}],
    "response_format": {"type": "json_object"},
}
