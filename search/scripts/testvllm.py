from vllm import LLM, SamplingParams
import os
import threading
from torch.cuda import device_count
import openai
import anthropic
from vllm import LLM
from torch.cuda import device_count, is_bf16_supported

from typing import Optional, Union, Any
import os
from abc import ABC, abstractmethod
import threading
import datetime
import json
from pathlib import Path
import time
from httpcore import ReadError
import random

from coderm.prompts import Prompt
from coderm.model import Completion, logprobs_to_cumulative
from search.python_utils import autodetect_dtype_str

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", trust_remote_code=True, gpu_memory_utilization=0.9, tensor_parallel_size=device_count(), max_model_len=8192, enforce_eager=True, dtype="auto")
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("DONE")
