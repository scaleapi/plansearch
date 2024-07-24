import openai
import anthropic
from vllm import LLM
from torch.cuda import device_count
from transformers import AutoTokenizer

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
from querier_utils import generate_anthropic_completion, generate_openai_chat_completion, generate_openai_completion, generate_vllm_chat_completions, generate_vllm_completions, num_tokens_for_convo, generate_internal_completions
from python_utils import autodetect_dtype_str


MODELS_TO_METHOD = {
    "gpt-3.5-turbo-instruct": "completions",
    "gpt-3.5-turbo": "chat",
    "gpt-3.5-turbo-16k": "chat",
    "gpt-4-turbo-preview": "chat",
    "gpt-4-turbo": "chat",
    "gpt-4o": "chat",
    "gpt-4o-mini": "chat",
    "claude-2.1": "anthropicchat",
    "meta-llama/Meta-Llama-3-8B-Instruct": "vllm",
    "casperhansen/llama-3-70b-instruct-awq": "vllm",
    "Meta-Llama-3-70B-Instruct.Q4_K_S.gguf": "llama_cpp_hf",
    "Meta-Llama-3-70B-Instruct.IQ3_XS.gguf": "llama_cpp_hf",
    "together-llama-3-70b": "together"
}

SUPPORTED_CLIENT_FOR_ASSISTANT_STR = ["vllm", "llama_cpp_hf"]

MODEL_NAME_TO_CLIENT_STR = {
    "gpt-4-turbo": ("OpenAI", openai.OpenAI, {}),
    "gpt-4o": ("OpenAI", openai.OpenAI, {}),
    "gpt-4": ("OpenAI-completion", openai.OpenAI, {}),
    "gpt-4o-mini": ("OpenAI", openai.OpenAI, {}),
    "gpt-3.5-turbo-instruct": ("OpenAI-completion", openai.OpenAI, {}),
    "claude-3-5-sonnet-20240620": ("Anthropic", anthropic.Anthropic, {}),
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": ("vllm-chat", LLM, {"model": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "trust_remote_code": True, "gpu_memory_utilization": 0.975, "tensor_parallel_size": 4, "max_model_len": 4096, "dtype": "bfloat16", "enforce_eager": True}),
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Base": ("vllm-completion", LLM, {"model": "deepseek-ai/DeepSeek-Coder-V2-Lite-Base", "trust_remote_code": True, "gpu_memory_utilization": 0.975, "tensor_parallel_size": 4, "max_model_len": 4096, "dtype": "bfloat16", "enforce_eager": True}),
    "deepseek-ai/DeepSeek-Coder-V2-Instruct": ("vllm-chat", LLM, {"model": "deepseek-ai/DeepSeek-Coder-V2-Instruct", "trust_remote_code": True, "gpu_memory_utilization": 0.975, "tensor_parallel_size": 8, "max_model_len": 4096, "dtype": "bfloat16", "enforce_eager": True}),
    "deepseek-ai/DeepSeek-Coder-V2-Base": ("vllm-completion", LLM, {"model": "deepseek-ai/DeepSeek-Coder-V2-Base", "trust_remote_code": True, "gpu_memory_utilization": 0.975, "tensor_parallel_size": 8, "max_model_len": 4096, "dtype": "bfloat16", "enforce_eager": True}),
    "meta-llama/Meta-Llama-3-8B-Instruct": ("vllm-chat", LLM, {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "trust_remote_code": True, "gpu_memory_utilization": 0.9, "tensor_parallel_size": 4, "max_model_len": 8192, "dtype": "bfloat16", "enforce_eager": True}),
    "meta-llama/Meta-Llama-3.1-8B": ("vllm-completion", LLM, {"model": "meta-llama/Meta-Llama-3.1-8B", "trust_remote_code": True, "gpu_memory_utilization": 0.9, "tensor_parallel_size": 4, "max_model_len": 8192, "dtype": "bfloat16", "enforce_eager": True}),
    "meta-llama/Meta-Llama-3-8B": ("vllm-completion", LLM, {"model": "meta-llama/Meta-Llama-3-8B", "trust_remote_code": True, "gpu_memory_utilization": 0.8, "tensor_parallel_size": 4, "max_model_len": 8192, "dtype": "bfloat16", "enforce_eager": True}),
    "custom-chat": ("vllm-chat", LLM, {"trust_remote_code": True, "gpu_memory_utilization": 0.9, "tensor_parallel_size": 8, "max_model_len": 8192, "dtype": "bfloat16", "enforce_eager": True}),
    "custom-completion": ("vllm-completion", LLM, {"trust_remote_code": True, "gpu_memory_utilization": 0.9, "tensor_parallel_size": 8, "max_model_len": 8192, "dtype": "bfloat16", "enforce_eager": True}),
    "meta-llama/Meta-Llama-3-405B-Instruct": ("internal", AutoTokenizer.from_pretrained, {"pretrained_model_name_or_path": "meta-llama/Meta-Llama-3-70B-Instruct"}),
}

PRINT_EVERY_X_DOLLARS = 1e3
# (input, output) price in dollars per token
MODEL_NAME_TO_INPUT_OUTPUT_PRICE = {
    "gpt-4-turbo": (10/1e6, 30/1e6),
    "gpt-4": (10/1e6, 30/1e6),
    "gpt-4o": (5/1e6, 15/1e6),
    "gpt-4o-mini": (0.150/1e6, 0.600/1e6),
    "gpt-3.5-turbo-instruct": (0, 0),
    "claude-3-5-sonnet-20240620": (3/1e6, 15/1e6),
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": (0, 0),
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Base": (0, 0),
    "deepseek-ai/DeepSeek-Coder-V2-Instruct": (0, 0),
    "deepseek-ai/DeepSeek-Coder-V2-Base": (0, 0),
    "meta-llama/Meta-Llama-3-8B-Instruct": (0, 0),
    "meta-llama/Meta-Llama-3-405B-Instruct": (0, 0),
    "meta-llama/Meta-Llama-3.1-8B": (0, 0),
    "meta-llama/Meta-Llama-3-8B": (0, 0),
}


CLIENT_STR_TO_GENERATE_FN = {
    "OpenAI": generate_openai_chat_completion,
    "OpenAI-completion": generate_openai_completion,
    "Anthropic": generate_anthropic_completion,
    "vllm-chat": generate_vllm_chat_completions,
    "vllm-completion": generate_vllm_completions,
    "internal": generate_internal_completions,
}

BATCH_CLIENT_FNS = {
    "vllm-chat", "vllm-completion"
}

IS_COMPLETION_CLIENT_FNS = {
    "vllm-completion", "OpenAI-completion"
}

def is_chat(model_name: str):
    if model_name not in MODEL_NAME_TO_CLIENT_STR:
        return "chat" in model_name.lower() or "instruct" in model_name.lower()
    return MODEL_NAME_TO_CLIENT_STR[model_name][0] not in IS_COMPLETION_CLIENT_FNS

class LLMQuerier(ABC):
    def __init__(self, log_directory: Optional[str] = None, cache_file: Optional[str] = "cache.json") -> None:
        self.log_directory = log_directory
        self.clients = {}
        self.current_price = 0.
        self.next_print_price = PRINT_EVERY_X_DOLLARS

        self.cache_file = cache_file

        if self.cache_file is not None:
            Path(os.path.dirname(self.cache_file)).mkdir(parents=True, exist_ok=True)
            if not os.path.exists(self.cache_file):
                with open(self.cache_file, "w") as f:
                    json.dump({}, f)
            print("Loading cache")
            with open(self.cache_file, "r") as f:
                self.current_cache = json.load(f)
            print("Done loading cache")

            self.temp_cache_file = os.path.join(os.path.dirname(self.cache_file), "temp_" + os.path.basename(self.cache_file))
            if not os.path.exists(self.temp_cache_file):
                with open(self.temp_cache_file, "w") as f:
                    json.dump({}, f)

        self.next_idx_for_requery = {}


    def generate_with_info(self, model: str, prompts: list[Prompt], frequency_penalty: Optional[float] = None, logit_bias: Optional[dict[str, int]] = None, max_tokens: Optional[int] = None, presence_penalty: Optional[float] = None, seed: Optional[int] = None, stop: Union[Optional[str], list[str]] = None, temperature: Optional[float] = None, top_p: Optional[float] = None, requery: bool = False, log_name: str = "") -> list[Completion]:
        if model not in MODEL_NAME_TO_CLIENT_STR:
            if is_chat(model):
                MODEL_NAME_TO_CLIENT_STR[model] = MODEL_NAME_TO_CLIENT_STR["custom-chat"]
                MODEL_NAME_TO_CLIENT_STR[model][2]["model"] = model
                MODEL_NAME_TO_INPUT_OUTPUT_PRICE[model] = (0, 0)
            else:
                MODEL_NAME_TO_CLIENT_STR[model] = MODEL_NAME_TO_CLIENT_STR["custom-completion"]
                MODEL_NAME_TO_CLIENT_STR[model][2]["model"] = model
                MODEL_NAME_TO_INPUT_OUTPUT_PRICE[model] = (0, 0)

        if model not in self.clients:
            self.clients[model] = MODEL_NAME_TO_CLIENT_STR[model][1](**MODEL_NAME_TO_CLIENT_STR[model][2])

        print("generating completions...")
        completions, cost = _generate_completions(self.clients[model], model, messages=prompts,
            current_cache=self.current_cache if self.cache_file is not None else {},
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            seed=seed,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            next_idx_for_requery=self.next_idx_for_requery if requery else None,
        )
        self.log(prompts, completions, log_name=log_name)
        print("done")

        self.current_price += cost
        if self.current_price > self.next_print_price:
            print(f"Current spending: ${self.current_price:.2f}")
            self.next_print_price = (self.current_price // PRINT_EVERY_X_DOLLARS + 1) * PRINT_EVERY_X_DOLLARS

        if self.cache_file is not None:
            self.save_cache()

        return completions

    def generate(self, model: str, prompts: list[Prompt], frequency_penalty: Optional[float] = None, logit_bias: Optional[dict[str, int]] = None, max_tokens: Optional[int] = None, presence_penalty: Optional[float] = None, seed: Optional[int] = None, stop: Union[Optional[str], list[str]] = None, temperature: Optional[float] = None, top_p: Optional[float] = None, requery: bool = False, log_name: str = "") -> list[str]:
        generations = self.generate_with_info(model, prompts,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            seed=seed,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            requery=requery,
            log_name=log_name,)
        return [c.code for c in generations]
    
    def log(self, prompts: list[Prompt], completions: list[Completion], log_name: str = ""):
        assert len(prompts) == len(completions)
        output_list = []
        for prompt, completion in zip(prompts, completions):
            completion_dict = {"text": completion.code, "cum_logprob": completion.cumulative_logprob, "num_tokens": completion.num_tokens}
            output_list.append({"prompt": prompt, "completion": completion_dict})

        if self.log_directory is not None:
            Path(self.log_directory).mkdir(parents=True, exist_ok=True)
            output_file = os.path.join(self.log_directory, log_name + datetime.datetime.now().strftime("%m-%dT%H:%M:%S") + ".json")
            with open(output_file, "w") as f:
                json.dump(output_list, f, indent=2)
    
    def save_cache(self):
        with open(self.temp_cache_file, "w") as f:
            json.dump(self.current_cache, f, indent=2)
        os.replace(self.temp_cache_file, self.cache_file)

    def set_log_directory(self, new_log_directory: str):
        self.log_directory = new_log_directory


def cache_hash(model: str, message: Prompt, frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float]):
    if is_chat(model):
        message_str = '|'.join([f"ROLE:{message['role']}|CONTENT:{message['content']}" for message in message])
    else:
        message_str = message
    if stop is None:
        stop_list = []
    elif isinstance(stop, str):
        stop_list = [stop]
    else:
        sorted(stop)
        stop_list = stop
    logit_bias_list = sorted(logit_bias.items()) if logit_bias is not None else []
    return f"MODEL:{model}|MESSAGES:{message_str}|FREQUENCY_PENALTY:{frequency_penalty}|LOGIT_BIAS_LIST:{logit_bias_list}|MAX_TOKENS:{max_tokens}|PRESENCE_PENALTY:{presence_penalty}|SEED:{seed}|STOP_LIST:{stop_list}|TEMPERATURE:{temperature}|TOP_P:{top_p}"


# Threaded generation code adapted from Federico Cassano
def _generate_completions(client: Union[openai.OpenAI, anthropic.Anthropic], model: str, messages: Union[list[Prompt], tuple[Prompt, ...]], current_cache: dict[str, dict[str, Any]], frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], next_idx_for_requery: Optional[dict[str, int]] = None) -> tuple[list[Completion], float]:
    if not isinstance(messages, (list, tuple)):
        raise ValueError("messages must be a list or tuple")
    assert len(messages)

    is_chat = MODEL_NAME_TO_CLIENT_STR[model][0] not in IS_COMPLETION_CLIENT_FNS
    if is_chat:
        assert all(isinstance(message_seq, list) for message_seq in messages)
    else:
        assert all(isinstance(message_seq, str) for message_seq in messages)

    completion_generator_fn = CLIENT_STR_TO_GENERATE_FN[MODEL_NAME_TO_CLIENT_STR[model][0]]

    completions: list[Optional[Completion]] = [None] * len(messages)
    threads = []
    to_generate = []

    input_tokens = 0
    not_cached = []
    new_next_idx_for_requery = {}
    for i, prompt in enumerate(messages):
        cache_key = cache_hash(model, prompt, frequency_penalty, logit_bias, max_tokens, presence_penalty, seed, stop, temperature, top_p)
        cache_i = i

        # next_idx_for_requery is a dictionary mapping cache keys to the next unused index
        # Otherwise, if we started from an earlier index, we may use the cached completion instead
        if next_idx_for_requery is not None:
            cache_i = i + next_idx_for_requery.get(cache_key, 0)
            # Updates the next index that should be used for requerying on a new call of _generate_completions
            assert new_next_idx_for_requery.get(cache_key, 0) < cache_i + 1
            new_next_idx_for_requery[cache_key] = cache_i + 1

        cache_key = f"{cache_i}|{cache_key}"

        if cache_key in current_cache:
            completions[i] = Completion(current_cache[cache_key]["text"], current_cache[cache_key]["cum_logprob"], current_cache[cache_key]["num_tokens"])
        else:
            not_cached.append((i, cache_key))
            to_generate.append((prompt, i))
            input_tokens += num_tokens_for_convo(prompt, is_chat)

    if MODEL_NAME_TO_CLIENT_STR[model][0] in BATCH_CLIENT_FNS:
        to_generate_prompts = [prompt for prompt, _ in to_generate]
        if len(to_generate_prompts):
            genned_completions = completion_generator_fn(
                client=client,
                model=model,
                prompts=to_generate_prompts,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                seed=seed,
                stop=stop,
                temperature=temperature,
                top_p=top_p,
            )
            for (_, i), completion in zip(to_generate, genned_completions):
                completions[i] = completion
    else:
        def generate_completion(prompt: list[dict[str, str]], i: int):
            completions[i] = completion_generator_fn(
                    client=client,
                    model=model,
                    prompt=prompt,
                    frequency_penalty=frequency_penalty,
                    logit_bias=logit_bias,
                    max_tokens=max_tokens,
                    presence_penalty=presence_penalty,
                    seed=seed,
                    stop=stop,
                    temperature=temperature,
                    top_p=top_p,
                )
        for prompt, i in to_generate:
            thread = threading.Thread(
                target=generate_completion, args=(prompt, i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    assert all(c is not None for c in completions), "Some completions are missing -- threading bug?"

    output_tokens = 0
    for uncached_idx, key in not_cached:
        current_cache[key] = {"text": completions[uncached_idx].code, "cum_logprob": completions[uncached_idx].cumulative_logprob, "num_tokens": completions[uncached_idx].num_tokens}
        output_tokens += completions[uncached_idx].num_tokens

    for cache, i in new_next_idx_for_requery.items():
        assert next_idx_for_requery.get(cache, 0) <= i
        next_idx_for_requery[cache] = i

    total_price = input_tokens * MODEL_NAME_TO_INPUT_OUTPUT_PRICE[model][0] + output_tokens * MODEL_NAME_TO_INPUT_OUTPUT_PRICE[model][1]
    return completions, total_price


if __name__ == "__main__":
    print("starting basic query...")
    llmq = LLMQuerier("temp_logs", None)
    # print(llmq.generate("meta-llama/Meta-Llama-3-405B-Instruct", [[{"role": "user", "content": "Please count to 10."}]], max_tokens=1000, temperature=0.1, top_p=0.9))
    # print(llmq.generate("meta-llama/Meta-Llama-3-8B-Instruct", [[{"role": "user", "content": "Please count to 10."}]], max_tokens=1000, temperature=0.1, top_p=0.9))
    print(llmq.generate("claude-3-5-sonnet-20240620", [[{"role": "user", "content": "Please count to 10."}]], max_tokens=1000, temperature=0.1, top_p=0.9))
    # print(llmq.generate("claude-3-5-sonnet-20240620", ["What is up?"], max_tokens=1000, temperature=0.1, top_p=0.9))