import os
import openai
import json
from src.ace.core.implementation import Context

def get_openai_api_key():
    """Retrieves the OpenAI API key from an environment variable."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return api_key

def format_prompt(context: Context) -> str:
    """Formats the content of a Context object into a simple string prompt."""
    # This is a basic implementation. It can be made more sophisticated.
    content_str = json.dumps(context.content)
    prompt = (
        f"Processing context for domain: {context.domain}\n"
        f"Context content: {content_str}\n\n"
        "Please provide a summary or analysis of the above content."
    )
    return prompt

def process_context_with_llm(prompt: str) -> str:
    """
    Processes a prompt string with an LLM, returning the LLM's response.
    """
    try:
        api_key = get_openai_api_key()
        openai.api_key = api_key

        # This is a basic call to the chat completions endpoint.
        # It can be customized with different models, parameters, etc.
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )

        return response.choices[0].message['content'].strip()

    except Exception as e:
        print(f"An error occurred while processing with LLM: {e}")
        return f"Error: Could not process context with LLM. Details: {e}"
