import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("vLLMEngine")

class VLLMClient:
    """
    Offline/Local Inference Engine using vLLM (with fallback to Transformers/CPU if no CUDA GPU).
    Key requirements from PDF:
    - Load Qwen model ONCE into memory. Do not reload model weights for each chunk.
    - Create a fresh prompt/context for each task and clear conversation history.
    - Structured JSON generation.
    - Zero external API keys. All models local or locally cached from Hugging Face.
    """
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        use_vllm: bool = True,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1
    ):
        self.model_name_or_path = model_name_or_path
        self.use_vllm = use_vllm
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        
        self.engine = None
        self.tokenizer = None
        self._init_engine()

    def _init_engine(self):
        if self.use_vllm:
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available. Falling back to Transformers pipeline on CPU.")
                    self._init_hf_transformers()
                    return

                from vllm import LLM, SamplingParams
                logger.info(f"Initializing vLLM engine for model: {self.model_name_or_path}")
                self.engine = LLM(
                    model=self.model_name_or_path,
                    tensor_parallel_size=self.tensor_parallel_size,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=self.max_model_len,
                    trust_remote_code=True
                )
                self.tokenizer = self.engine.get_tokenizer()
                logger.info("vLLM engine initialized successfully.")
                return
            except Exception as e:
                logger.warning(f"Could not initialize vLLM ({e}). Falling back to HuggingFace Transformers.")

        self._init_hf_transformers()

    def _init_hf_transformers(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch

            logger.info(f"Loading model with HuggingFace Transformers: {self.model_name_or_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
            
            device_map = "auto" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
            self.engine = pipeline("text-generation", model=model, tokenizer=self.tokenizer)
            logger.info("Transformers pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to load Transformers model: {e}")
            self.engine = None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 2048
    ) -> str:
        """
        Runs generation in a FRESH context for each chunk/task.
        Zero conversational carry-over.
        """
        if self.engine is None:
            raise RuntimeError("Inference engine is not initialized!")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if hasattr(self.engine, "chat"):  # vLLM LLM.chat support
            from vllm import SamplingParams
            sp = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9
            )
            outputs = self.engine.chat(messages=[messages], sampling_params=sp, use_tqdm=False)
            return outputs[0].outputs[0].text

        elif hasattr(self.engine, "generate"):  # vLLM generate with formatted prompt
            from vllm import SamplingParams
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            sp = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9
            )
            outputs = self.engine.generate([formatted_prompt], sampling_params=sp, use_tqdm=False)
            return outputs[0].outputs[0].text

        else:  # HuggingFace pipeline
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            res = self.engine(
                formatted_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.0),
                return_full_text=False
            )
            return res[0]["generated_text"]

    def repair_json_format(self, expected_schema: str, invalid_output: str, parser_error: str) -> str:
        """
        Prompt G implementation from PDF:
        Repair JSON syntax and schema formatting only. Do not add facts or change meaning.
        """
        system = "Repair JSON syntax and schema formatting only. Do not add facts or change meaning. Return JSON only."
        user = (
            f"EXPECTED_SCHEMA:\n{expected_schema}\n\n"
            f"INVALID_OUTPUT:\n{invalid_output}\n\n"
            f"PARSER_ERROR:\n{parser_error}\n\n"
            "TASK: Repair quotes, commas, brackets and unambiguous field types. "
            "If content is missing or ambiguous, return an error status. Do not guess missing values."
        )
        return self.generate(system, user, temperature=0.0, max_tokens=2048)
