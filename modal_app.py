""" The full experiment pipeline that also loads the 
opensource model(Qwen3.8 compressed version) on modal """

import threading
from bgt import experiment, models
from bgt.open_model import LocalModel
import modal
import os
import yaml
 
LOCAL = LocalModel() 
app = modal.App("bgt")

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm", "openai", "anthropic", "pyyaml", "numpy", "pandas", "datasets", "tqdm")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_ENGINE_READY_TIMEOUT_S": "3600"})
    .add_local_python_source("bgt")                     
    .add_local_dir("scenarios", remote_path="/root/scenarios")
)

hf_cache = modal.Volume.from_name("bgt-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("bgt-results", create_if_missing=True)
keys = modal.Secret.from_name("bgt-keys")

JOB = dict(image=image, gpu=LOCAL.gpu(), memory=LOCAL.host_ram_mib, secrets=[keys],
           volumes={"/root/.cache/huggingface": hf_cache, "/results": results})


def save_results_every_minute(stop: threading.Event) -> None:
    while not stop.wait(60):
        results.commit()


@app.function(timeout=24 * 60 * 60, **JOB)
def run_experiment(cfg: dict, out_name: str) -> str:
    """Runs the full experiment"""
    os.chdir("/root")                                    
    server = models.start_local_model(LOCAL.model_name, LOCAL.reasoning_parser, LOCAL.max_len, LOCAL.vllm_args())
    stop = threading.Event()
    threading.Thread(target=save_results_every_minute, args=(stop,), daemon=True).start()
    try:
        print(f"model ready after {models.wait_until_ready(server):.0f}s", flush=True)
        cfg["out"] = f"/results/{out_name}"
        models.set_local_options(cfg["reasoning_effort"], cfg["top_p"], cfg["top_k"])
        experiment.run_all(cfg)
    finally:
        stop.set()
        results.commit()
        server.terminate()
    return cfg["out"]


@app.local_entrypoint()
def main(config: str = "configs/main.yaml", out: str = "main.jsonl"):
    cfg = yaml.safe_load(open(config))
    print("results saved in volume bgt-results as:", run_experiment.remote(cfg, out))
