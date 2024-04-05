single_tool_use = {
    "model": "claude-3-opus-20240229",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "What's the weather like in San Francisco?"}],
    "tools": [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        }
    ],
}


multiple_tool_use = {
    "model": "claude-3-opus-20240229",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "What is the weather like right now in New York? Also what time is it there?"}],
    "tools": [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit of temperature, either 'celsius' or 'fahrenheit'"},
                },
                "required": ["location"],
            },
        },
        {
            "name": "get_time",
            "description": "Get the current time in a given time zone",
            "input_schema": {
                "type": "object",
                "properties": {"timezone": {"type": "string", "description": "The IANA time zone name, e.g. America/Los_Angeles"}},
                "required": ["timezone"],
            },
        },
    ],
}


missing_information = {
    "model": "claude-3-opus-20240229",
    "max_tokens": 1024,
    "tools": [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": 'The unit of temperature, either "celsius" or "fahrenheit"'},
                },
                "required": ["location"],
            },
        }
    ],
    "messages": [
        {"role": "user", "content": "What is the weather like in San Francisco?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "<thinking>I need to use get_weather, and the user wants SF, which is likely San Francisco, CA.</thinking>"},
                {"type": "tool_use", "id": "toolu_01A09q90qw90lq917835lq9", "name": "get_weather", "input": {"location": "San Francisco, CA", "unit": "celsius"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01A09q90qw90lq917835lq9", "content": "65 degrees"}]},
    ],
}
