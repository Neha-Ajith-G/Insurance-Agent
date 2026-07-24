"""
DocClient: thin wrapper around a local Ollama model that tracks token
usage and inference time for every call.
"""
from __future__ import annotations

import os
import logging
import time

import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DocClientError(Exception):
    """Raised when the LLM backend can't be reached or errors out."""


class LLMResponse(BaseModel):
    model: str
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    inference_time: float  # seconds
    done: bool
    done_reason: str | None = None


class DocClient:
    """
    Wraps Ollama's /api/generate endpoint.

    Token counts come from Ollama's own prompt_eval_count / eval_count
    fields rather than re-tokenizing client-side with tiktoken -- those
    are the exact counts the local model actually saw and produced,
    which tiktoken (a different tokenizer) would only approximate.
    """

    def __init__(
        self,
        model: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"),
        base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        num_ctx: int = 4096,
        timeout: float = 300.0,
        temperature: float = 0.0,
        keep_alive: str = "10m",
        top_p: float = 0.9,
        seed: int | None = 42,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.top_p = top_p
        self.seed = seed

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": self.seed,
            },
        }
        if system:
            payload["system"] = system

        logger.debug(
            "DocClient.generate model=%s prompt_chars=%d", self.model, len(prompt)
        )

        start = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise DocClientError(f"Ollama request failed: {e}") from e
        elapsed = time.perf_counter() - start

        data = resp.json()
        text = data.get("response", "")
        if not text:
            raise DocClientError(
                f"Model returned an empty response (done_reason="
                f"{data.get('done_reason')!r}). This usually means the "
                f"backend crashed or was interrupted mid-generation."
            )

        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        logger.info(
            "DocClient.generate model=%s in=%d out=%d time=%.3fs done_reason=%s",
            self.model, input_tokens, output_tokens, elapsed, data.get("done_reason"),
        )

        return LLMResponse(
            model=data.get("model", self.model),
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            inference_time=elapsed,
            done=data.get("done", True),
            done_reason=data.get("done_reason"),
        )

    def health(self) -> bool:
        """Check whether the Ollama backend is reachable and responsive."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.warning("DocClient.health check failed: %s", e)
            return False