import os
import time
import random
from openai import OpenAI,RateLimitError, APIError, APITimeoutError, APIConnectionError


from greek_adaptive_rewriter import LLMClient


class OpenAILLMClient(LLMClient):
    def __init__(self, model: str | None = None):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment variables")
        self.client = OpenAI(api_key=api_key)
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
        self._last_call_ts = 0.0
        self._min_interval_s = float(os.environ.get("OPENAI_MIN_INTERVAL_S", "3.0"))

    def generate(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        max_retries = 8
        base = 1.0
        cap = 20.0

        for attempt in range(1, max_retries + 1):
            # client-side throttling per attempt
            now = time.time()
            wait = self._min_interval_s - (now - self._last_call_ts)
            if wait > 0:
                print(f"[openai] throttling: sleep {wait:.1f}s")
                time.sleep(wait)

            try:
                print(
                    f"[openai] attempt {attempt}/{max_retries} "
                    f"model={self.model} temp={temperature} sys={len(system)} user={len(user)}"
                )

                # record just before the call
                self._last_call_ts = time.time()

                resp = self.client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    timeout=60,  # if unsupported by your SDK version, remove this line
                )
                text = (resp.output_text or "").strip()
                print(f"[openai] ok, chars={len(text)}")
                return text

            except RateLimitError as e:
                msg = str(e)
                print(f"[openai] 429 details: {msg}")

                # This is your real current failure mode. Don't retry.
                if "insufficient_quota" in msg:
                    raise RuntimeError(
                        "OpenAI API quota is exhausted / billing not enabled for this key. "
                        "Enable billing on platform.openai.com or use a key from a billed org/project."
                    ) from e

                # Real rate limit: honor Retry-After if present
                retry_after = None
                try:
                    headers = getattr(getattr(e, "response", None), "headers", None)
                    if headers:
                        ra = headers.get("retry-after") or headers.get("Retry-After")
                        if ra:
                            retry_after = float(ra)
                except Exception:
                    retry_after = None

                if retry_after is not None:
                    sleep_s = min(60.0, retry_after + random.random() * 0.5)
                    print(f"[openai] honoring Retry-After {retry_after}s -> sleep {sleep_s:.1f}s")
                else:
                    sleep_s = min(cap, base * (2 ** (attempt - 1))) + random.random()
                    print(f"[openai] rate limited, sleeping {sleep_s:.1f}s")

                time.sleep(sleep_s)

            except (APITimeoutError, APIConnectionError) as e:
                sleep_s = min(cap, 1.0 + random.random() * 2.0)
                print(f"[openai] transient network/timeout ({type(e).__name__}), sleeping {sleep_s:.1f}s")
                time.sleep(sleep_s)

            except APIError as e:
                sleep_s = min(cap, 1.5 + random.random() * 2.0)
                print(f"[openai] api error ({type(e).__name__}), sleeping {sleep_s:.1f}s")
                time.sleep(sleep_s)

        raise RuntimeError("OpenAI request failed after retries (rate limit or transient errors).")