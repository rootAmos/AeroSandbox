# The function name, type hints, and docstring are all part of the tool
# schema that's passed to the model. Defining good, descriptive schemas
# is an extension of prompt engineering and is an important part of
# getting models to perform well.
from typing_extensions import Annotated, TypedDict


class add(TypedDict):
    """Add two integers."""

    # Annotations must have the type and can optionally include a default value and description (in that order).
    a: Annotated[int, ..., "First integer"]
    b: Annotated[int, ..., "Second integer"]


class multiply(TypedDict):
    """Multiply two integers."""

    a: Annotated[int, ..., "First integer"]
    b: Annotated[int, ..., "Second integer"]



if __name__ == "__main__":
    from google import genai
    from google.genai import types

    import getpass
    import os

    if not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
    from langchain.chat_models import init_chat_model

    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

    tools = [add, multiply]

    llm_with_tools = llm.bind_tools(tools)

    query = "What is 3 * 12?"

    llm_with_tools.invoke(query)
    
    