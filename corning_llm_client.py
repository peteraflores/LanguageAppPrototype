# corning_llm_client.py - Updated with better error handling
import os
import time
import random
from openai import OpenAI, RateLimitError, APIError, APITimeoutError, APIConnectionError, PermissionDeniedError
from greek_adaptive_rewriter import LLMClient

class CorningLLMClient(LLMClient):
    def __init__(self, model: str | None = None):
        # Get API key - you'll need to generate this from Open WebUI
        api_key = os.environ.get("CORNING_API_KEY","sk-9de69d9b18b049b38b4efd6cfae97a38")
        if not api_key:
            raise RuntimeError(
                "CORNING_API_KEY is not set in environment variables. "
                "Generate an API key from Open WebUI: Settings -> Account -> API Keys"
            )
        
        # Configure the client to use Corning's internal endpoint
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://gatewai.svcpr.aws.corning.com/openai"
        )
        
        # Default to Claude Sonnet 4.5 as recommended in the docs
        # You can override this with the exact model name from Open WebUI
        self.model = model or os.environ.get("CORNING_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        self._last_call_ts = 0.0
        self._min_interval_s = float(os.environ.get("CORNING_MIN_INTERVAL_S", "3.0"))
        
        # Test the connection on initialization
        self._test_connection()
    
    def _test_connection(self):
        """Test if we can list models to verify API key and access"""
        try:
            models = self.client.models.list()
            available_models = [m.id for m in models]
        except Exception as e:
            print(f"[corning-llm] Failed to list models: {type(e).__name__}: {e}")
            # Don't fail here, just warn
    
    def generate(self, *, system: str, user: str, temperature: float = 0.2) -> str:
        max_retries = 8
        base = 1.0
        cap = 20.0
        
        for attempt in range(1, max_retries + 1):
            # client-side throttling per attempt
            now = time.time()
            wait = self._min_interval_s - (now - self._last_call_ts)
            if wait > 0:
                time.sleep(wait)
            
            try:
                # record just before the call
                self._last_call_ts = time.time()
                
                # Using standard OpenAI chat completions format
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    timeout=60,
                )
                
                text = resp.choices[0].message.content.strip()
                return text
                
            except PermissionDeniedError as e:
                # Permission errors are not retryable
                error_msg = str(e)
                print(f"[corning-llm] Permission denied: {error_msg}")
                
                # Try to extract more details
                if hasattr(e, 'response') and hasattr(e.response, 'json'):
                    try:
                        error_details = e.response.json()
                        print(f"[corning-llm] Error details: {error_details}")
                    except:
                        pass
                
                raise RuntimeError(
                    f"Permission denied accessing Corning LLM API. "
                    f"Check your API key and model access. Error: {error_msg}"
                ) from e
                
            except RateLimitError as e:
                msg = str(e)
                print(f"[corning-llm] 429 details: {msg}")
                
                if "quota" in msg.lower() or "limit" in msg.lower():
                    raise RuntimeError(
                        "Internal API quota exceeded. Contact your IT department or check Open WebUI."
                    ) from e
                
                # Rate limit handling
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
                    print(f"[corning-llm] honoring Retry-After {retry_after}s -> sleep {sleep_s:.1f}s")
                else:
                    sleep_s = min(cap, base * (2 ** (attempt - 1))) + random.random()
                    print(f"[corning-llm] rate limited, sleeping {sleep_s:.1f}s")
                time.sleep(sleep_s)
                
            except (APITimeoutError, APIConnectionError) as e:
                sleep_s = min(cap, 1.0 + random.random() * 2.0)
                print(f"[corning-llm] transient network/timeout ({type(e).__name__}), sleeping {sleep_s:.1f}s")
                time.sleep(sleep_s)
                
            except APIError as e:
                # Don't retry permission errors
                if "permission" in str(e).lower() or "forbidden" in str(e).lower():
                    raise RuntimeError(
                        f"Permission error accessing Corning LLM API: {e}"
                    ) from e
                    
                sleep_s = min(cap, 1.5 + random.random() * 2.0)
                print(f"[corning-llm] api error ({type(e).__name__}), sleeping {sleep_s:.1f}s")
                time.sleep(sleep_s)
        
        raise RuntimeError("Corning LLM request failed after retries.")