import os
import gc
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("vLLMEngine")

def check_gpu_resources(required_vram_gb: float = 8.0) -> Dict[str, Any]:
    """
    Hardware inspection before inference:
    Detects CUDA availability, device count, device name, and free VRAM.
    Returns status dict with recommendations for precision and memory utilization.
    """
    info = {
        "cuda_available": False,
        "device_count": 0,
        "device_name": None,
        "total_vram_gb": 0.0,
        "free_vram_gb": 0.0,
        "sufficient_for_3b": False,
        "recommended_dtype": "float32",
        "warning": None
    }
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            
            # Query memory stats
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            info["total_vram_gb"] = round(total_bytes / (1024 ** 3), 2)
            info["free_vram_gb"] = round(free_bytes / (1024 ** 3), 2)
            
            if info["free_vram_gb"] >= required_vram_gb:
                info["sufficient_for_3b"] = True
                info["recommended_dtype"] = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            else:
                info["sufficient_for_3b"] = False
                info["warning"] = (
                    f"Available VRAM ({info['free_vram_gb']} GB) is below the recommended threshold "
                    f"({required_vram_gb} GB) for Qwen 3B models. Out-Of-Memory (OOM) error possible."
                )
        else:
            info["warning"] = "CUDA is not available. System will run on CPU (inference will be slow)."
    except Exception as e:
        info["warning"] = f"Error querying GPU info: {e}"
        
    return info

def get_offline_qwen_tokenizer(model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct"):
    """
    Safely attempts to load the exact Qwen tokenizer from local cache only.
    - Does NOT download any files from HuggingFace Hub.
    - Does NOT load any LLM model weights.
    - Does NOT substitute any other model's tokenizer.
    Returns the tokenizer if locally cached, or None if unavailable offline.
    """
    try:
        from transformers import AutoTokenizer
        logger.info(f"Attempting offline load of exact Qwen tokenizer: {model_name_or_path} (local_files_only=True)")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=True,
            trust_remote_code=True
        )
        logger.info(f"Exact Qwen tokenizer ({model_name_or_path}) loaded successfully from local offline cache.")
        return tokenizer
    except Exception as e:
        logger.warning(
            f"Exact Qwen tokenizer ({model_name_or_path}) is NOT available in local offline cache ({e}). "
            "Pipeline will continue safely using conservative character-based estimation fallback. "
            "No incompatible tokenizer will be substituted."
        )
        return None

class VLLMVisionClient:
    """
    Dedicated Qwen2.5-VL-3B-Instruct vision client.
    Loaded only during Task 4 (Image Understanding) and unloaded immediately
    to preserve GPU VRAM for the primary text LLM.
    """
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        use_vllm: bool = True,
        gpu_memory_utilization: float = 0.80,
        max_model_len: int = 2048
    ):
        self.model_name_or_path = model_name_or_path
        self.use_vllm = use_vllm
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.engine = None
        self.processor = None
        self._init_vision_engine()

    def _init_vision_engine(self):
        hw = check_gpu_resources(required_vram_gb=6.0)
        logger.info(f"Vision model hardware check: {hw}")
        if not hw["cuda_available"]:
            logger.warning("CUDA not available for Vision LLM. Attempting Transformers CPU fallback...")
            self._init_hf_vision()
            return

        if self.use_vllm:
            try:
                from vllm import LLM
                logger.info(f"Initializing vLLM Vision engine for: {self.model_name_or_path}")
                self.engine = LLM(
                    model=self.model_name_or_path,
                    gpu_memory_utilization=self.gpu_memory_utilization,
                    max_model_len=self.max_model_len,
                    trust_remote_code=True
                )
                logger.info("vLLM Vision engine loaded successfully.")
                return
            except Exception as e:
                logger.warning(f"Could not load vLLM Vision engine ({e}). Falling back to HF Transformers.")

        self._init_hf_vision()

    def _init_hf_vision(self):
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

            logger.info(f"Loading HF Vision Model: {self.model_name_or_path} on {device}")
            self.processor = AutoProcessor.from_pretrained(self.model_name_or_path, trust_remote_code=True)
            self.engine = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name_or_path,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True
            )
            logger.info("HF Vision Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load HF Vision Model: {e}")
            self.engine = None

    def describe_image(self, image_path: str, prompt_hint: Optional[str] = None) -> str:
        """
        Describes chart, diagram, photograph or visual labels from the image.
        """
        if self.engine is None:
            return ""

        if not os.path.exists(image_path):
            logger.warning(f"Image file not found: {image_path}")
            return ""

        prompt_text = prompt_hint or (
            "Analyze this corporate annual report figure/chart/diagram. "
            "Describe the visual data, key metrics, financial figures, labels, and title precisely. "
            "Do not guess missing information."
        )

        try:
            from PIL import Image
            img = Image.open(image_path).convert("RGB")

            # Check if using HF
            if hasattr(self, "processor") and self.processor is not None:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ]
                text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text_prompt], images=[img], return_tensors="pt", padding=True)
                if hasattr(self.engine, "device"):
                    inputs = {k: v.to(self.engine.device) for k, v in inputs.items()}
                
                output_ids = self.engine.generate(**inputs, max_new_tokens=512)
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)
                ]
                desc = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                return desc.strip()
            
            # vLLM Vision generation
            elif hasattr(self.engine, "generate"):
                from vllm import SamplingParams
                sp = SamplingParams(temperature=0.1, max_tokens=512)
                res = self.engine.generate(
                    {
                        "prompt": f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt_text}<|im_end|>\n<|im_start|>assistant\n",
                        "multi_modal_data": {"image": img}
                    },
                    sampling_params=sp
                )
                return res[0].outputs[0].text.strip()
        except Exception as e:
            logger.warning(f"Vision inference error for {image_path}: {e}")
            return ""

    def unload(self):
        """
        Safely unloads the vision model and frees VRAM for the primary text LLM.
        """
        logger.info("Unloading Vision model to free GPU VRAM...")
        self.engine = None
        self.processor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("CUDA memory cache cleared successfully.")
        except Exception:
            pass

class VLLMClient:
    """
    Offline/Local Inference Engine using vLLM for Qwen/Qwen2.5-3B-Instruct
    (with fallback to Transformers/CPU if no CUDA GPU).
    Key requirements:
    - Load Qwen model ONCE into memory when inference begins. Do not reload model weights for each chunk.
    - Inspect GPU resources and configure safe memory utilization.
    - Structured JSON generation.
    - Zero external API keys. All models local or locally cached from Hugging Face.
    """
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
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
        hw = check_gpu_resources(required_vram_gb=8.0)
        logger.info(f"Hardware check before text model load: {hw}")
        if hw["warning"]:
            logger.warning(hw["warning"])

        if self.use_vllm:
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available. Falling back to Transformers pipeline on CPU.")
                    self._init_hf_transformers()
                    return

                from vllm import LLM, SamplingParams
                logger.info(f"Initializing vLLM engine for model: {self.model_name_or_path} (VRAM alloc: {self.gpu_memory_utilization})")
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
