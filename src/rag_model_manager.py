import os
import threading
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, pipeline, AutoModel


class LLaVaModelManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LLaVaModelManager, cls).__new__(cls)
                cls._instance.model = None
                cls._instance.tokenizer = None
                cls._instance.initialize_model()
            return cls._instance

    def initialize_model(self):
        print(f"Initializing llava model and processor")
        llava_temp = 0.8
        llava_processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        llava_model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",
                                                                        bnb_4bit_quant_type="nf4",
                                                                        bnb_4bit_compute_dtype=torch.float16,
                                                                        low_cpu_mem_usage=True,
                                                                        load_in_4bit=True,
                                                                        temperature=llava_temp,
                                                                        do_sample=True)
        self.model = llava_model
        self.processor = llava_processor
        print("llava model initialized.")

    def get_model(self):
        return self.model

    def get_processor(self):
        return self.processor


class FalconAIModelManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, model_path):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FalconAIModelManager, cls).__new__(cls)
                cls._instance.model = None
                cls._instance.tokenizer = None
                cls._instance.initialize_model(model_path)
            return cls._instance

    def initialize_model(self, model_path):
        print(f"Initializing falcon ai summarizer model")
        self.model = pipeline("summarization", model=model_path)
        print("falcon ai model initialized.")

    def get_model(self):
        return self.model

class JINAModelManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, device):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(JINAModelManager, cls).__new__(cls)
                cls._instance.model = None
                cls._instance.tokenizer = None
                cls._instance.initialize_model(device)
            return cls._instance

    def initialize_model(self, device):
        print(f"Initializing jina ai model")
        self.model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).to(device)
        print("jina ai model initialized.")

    def get_model(self):
        return self.model
