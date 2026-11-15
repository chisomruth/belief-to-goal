"""models.py details the model communication pipleine
"""

import os
import subprocess
import threading
import time
import urllib.request
import anthropic
from openai import OpenAI
from .open_model import LocalModel


local = LocalModel()
LOCAL_PORT = os.environ["PORT"]
LOCAL_URL = f"http://localhost:{LOCAL_PORT}/v1"
LOCAL_OPTIONS = {"reasoning_effort": "low", "top_p": 0.95, "top_k": 20}

_clients = {}               
_lock = threading.Lock()

def set_local_options(reasoning_effort=LOCAL_OPTIONS["reasoning_effort"],
                      top_p=LOCAL_OPTIONS["top_p"], top_k=LOCAL_OPTIONS["top_k"]) -> None:
    """This chooses how the local model is sampled."""
    LOCAL_OPTIONS.update(reasoning_effort=reasoning_effort, top_p=top_p, top_k=top_k)




def parse_spec(spec: str):
    if spec == "local":
        return "local", os.environ.get("LOCAL_MODEL")
    provider, sep, name = spec.partition(":")
    if not sep or not name or provider not in ("anthropic", "openai", "local"):
        raise ValueError(f"Bad model spec {spec!r}. Use 'local', 'anthropic:<name>' or 'openai:<name>'.")
    return provider, name


def _client(provider: str):
    with _lock:
        if provider not in _clients:
            if provider == "anthropic":
                _clients[provider] = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            elif provider == "openai":
                _clients[provider] = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            else:
                _clients[provider] = OpenAI(base_url=LOCAL_URL, api_key="None", timeout=1800)
        return _clients[provider]



def split_thinking(text: str):
    """Returns (final_answer_text, thinking_text)"""
    text = text or ""
    if "</think>" in text:
        thinking, _, answer = text.partition("</think>")
        return answer.strip(), thinking.replace("<think>", "").strip()
    return text.strip(), ""

def _extra_field(message, name):
    """vLLM returns the thinking in a separate field (when started with a reasoning parser)."""
    value = getattr(message, name, None)
    if value is None and getattr(message, "model_extra", None):
        value = message.model_extra.get(name)
    return value or ""




def call_model_full(spec, messages, system=None, temperature=0.0, max_tokens=3000, retries=5):
    """Send a chat, get (answer_text, thinking_text)."""
    provider, name = parse_spec(spec)
    last_error = None
    for attempt in range(retries):
        try:
            if provider == "anthropic":
                kwargs = dict(model=name, max_tokens=max_tokens, messages=messages)
                if system:
                    kwargs["system"] = system
                reply = _client(provider).messages.create(**kwargs)
                return "".join(b.text for b in reply.content if b.type == "text").strip(), ""
            full = ([{"role": "system", "content": system}] if system else []) + list(messages)
            kwargs = dict(model=name, messages=full)
            
            if provider == "openai":
                kwargs["max_completion_tokens"] = max_tokens

            else:                                                             
                kwargs["max_tokens"] = max_tokens
                kwargs["temperature"] = 1.0
                kwargs["top_p"] = LOCAL_OPTIONS["top_p"]
                kwargs["reasoning_effort"] = LOCAL_OPTIONS["reasoning_effort"]
                kwargs["extra_body"] = {"top_k": LOCAL_OPTIONS["top_k"]}      

            message = _client(provider).chat.completions.create(**kwargs).choices[0].message
            answer, thinking = split_thinking(message.content)
            thinking = thinking or _extra_field(message, "reasoning_content") or _extra_field(message, "reasoning")
            return answer, thinking
        except Exception as e:                
            last_error = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{spec} failed after {retries} tries: {last_error}")

def call_model(spec, messages, system=None, temperature=0.0, max_tokens=3000) -> str:
    """Same as call_model_full but returns only the answer text."""
    return call_model_full(spec, messages, system, temperature, max_tokens)[0]







def start_local_model(hf_name: str, reasoning_parser: str = local.reasoning_parser, max_len: int = local.max_len, extra_args=()):
    """Start vLLM (on an OpenAI-compatible server) that loads `hf_name` from Hugging Face onto the GPU."""
    os.environ["LOCAL_MODEL"] = hf_name
    cmd = ["vllm", "serve", hf_name, "--served-model-name", hf_name,
           "--host", "0.0.0.0", "--port", str(LOCAL_PORT),
           "--max-model-len", str(max_len), "--gpu-memory-utilization", "0.90"]
    if reasoning_parser:
        cmd += ["--reasoning-parser", reasoning_parser]
    cmd += list(extra_args)
    print("starting:", " ".join(cmd), flush=True)
    return subprocess.Popen(cmd)

def wait_until_ready(process=None, timeout: float = 1800, poll: float = 5.0) -> float:
    """Wait until the local server answers."""
    start = time.time()
    while time.time() - start < timeout:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"vLLM exited {process.communicate() } while loading the model." )
        try:
            with urllib.request.urlopen(f"http://localhost:{LOCAL_PORT}/health", timeout=10) as r:
                if r.status == 200:
                    return time.time() - start
        except Exception:
            pass
        time.sleep(poll)
    raise TimeoutError(f"local model not ready after {timeout:.0f}s")