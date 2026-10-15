from dotenv import load_dotenv
from google import genai
from google.genai import types

from readback_guard.config import ROOT

_client = None


def get_client(cfg):
    global _client
    if _client is None:
        load_dotenv(ROOT / ".env")
        retry = types.HttpRetryOptions(attempts=5, initial_delay=10, http_status_codes=[429, 500, 503, 504])
        _client = genai.Client(http_options=types.HttpOptions(timeout=cfg["llm"]["timeout_ms"], retry_options=retry))
    return _client


def generate_json(prompt, schema, cfg):
    resp = get_client(cfg).models.generate_content(
        model=cfg["llm"]["model"],
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": schema,
            "temperature": cfg["llm"]["temperature"],
        },
    )
    return resp.text
