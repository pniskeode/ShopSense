from langchain_core.messages import AIMessage


def model_response(content: str = "", tool_calls=None) -> AIMessage:
    calls = [
        {"id": call_id, "name": name, "args": arguments, "type": "tool_call"}
        for call_id, name, arguments in tool_calls or []
    ]
    return AIMessage(content=content, tool_calls=calls)


class FakeChatModel:
    def __init__(self, responses) -> None:
        self.responses = iter(responses)
        self.calls = []
        self.bound_tools = []
        self.bind_count = 0

    def bind_tools(self, tools):
        self.bind_count += 1
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages):
        self.calls.append(list(messages))
        return next(self.responses)
