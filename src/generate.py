"""Build the grounded prompt and call Azure OpenAI (one chat call, no retry logic here)."""
from functools import lru_cache
from pathlib import Path

from openai import AzureOpenAI

from .config import API_VERSION, MAX_TOKENS, TEMPERATURE, require_env

# The prompt lives in prompts/system_prompt.txt, found relative to this file.
# rstrip("\n") removes the final newline an editor may add, so the text stays exact.
PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8").rstrip("\n")


@lru_cache(maxsize=None)
def get_client():
    # max_retries=0: retries are handled once, in src/retry.py, so they cannot stack.
    return AzureOpenAI(
        azure_endpoint=require_env("AZURE_OPENAI_ENDPOINT"),
        api_key=require_env("AZURE_OPENAI_API_KEY"),
        api_version=API_VERSION,
        max_retries=0,
    )


def build_messages(question, chunks):
    context = "\n\n".join(f"[chunk {c['id']}]\n{c['content']}" for c in chunks)
    user_content = f"Chunks:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def chat(messages):
    # model= takes the deployment NAME chosen in the Azure portal, not the model family.
    return get_client().chat.completions.create(
        model=require_env("AZURE_OPENAI_DEPLOYMENT"),
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
