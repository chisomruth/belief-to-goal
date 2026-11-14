""" The settings for the open-source model we 
load on the Modal GPUs. """


class LocalModel:
    def __init__(self, model_name: str = "Qwen/Qwen3.8-Flash-Next-FP8",   
                 reasoning_parser: str = "qwen3", max_len: int = 32768,                       
                 n_gpus: int = 4, gpu_type: str = "H100", host_ram_mib: int = 131072):                      
        self.model_name = model_name
        self.reasoning_parser = reasoning_parser
        self.max_len = max_len
        self.n_gpus = n_gpus
        self.gpu_type = gpu_type
        self.host_ram_mib = host_ram_mib

    def gpu(self) -> str:
        return f"{self.gpu_type}:{self.n_gpus}"

    def vllm_args(self) -> list:
        """Extra vLLM flags (from the vLLM recipe's Hopper configuration for this model)."""
        return [
            "--tensor-parallel-size", str(self.n_gpus),
            "--enable-expert-parallel",          
            "--moe-backend", "triton",
            "--max-num-seqs", "256",              
            "--enable-prefix-caching",
            "--kv-cache-dtype", "fp8",
            "--attention-config.indexer_kv_dtype", "fp8",
            "--limit-mm-per-prompt", '{"image":0,"video":0}']