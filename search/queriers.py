import openai
import anthropic
from vllm import LLM
from torch.cuda import device_count
from transformers import AutoTokenizer
from tqdm import tqdm

from typing import Optional, Union, Any
import os
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import datetime
import json
from pathlib import Path
import time
from httpcore import ReadError
import random
import warnings
from uuid import uuid4

from coderm.prompts import Prompt
from coderm.model import Completion, logprobs_to_cumulative
from search.query_clients import LLMClient, OpenAIClient
from search.python_utils import autodetect_dtype_str, chunk, remove_from_str


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

PRINT_EVERY_X_DOLLARS = 1e3
DEFAULT_O1_REPLACEMENT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_configs/gpt-4o.json")
O1_FILTER = ["steps", "STEPS", "Steps", "step", "STEP", "Step", "Quote"]

class CompletionCache:
    def __init__(self, cache_file: Optional[str]) -> None:
        self.disabled = cache_file is None
        self.cache_file = cache_file

        self.current_cache: dict[str, dict[str, Any]] = {}
        self.next_requery_idx: dict[str, int] = {}

        if not self.disabled:
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

    def save_cache(self):
        if self.disabled:
            return
        with open(self.temp_cache_file, "w") as f:
            json.dump(self.current_cache, f, indent=2)
        os.replace(self.temp_cache_file, self.cache_file)

    def query(self, model: str, message: Prompt, frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float]) -> tuple[Optional[Completion], str]:
        if self.disabled:
            return None, ""

        query_str = self._get_cache_query_str(
            model=model,
            convo=message,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            seed=seed,
            stop=stop,
            temperature=temperature,
            top_p=top_p
        )

        requery_idx = self.next_requery_idx.get(query_str, 0)
        self.next_requery_idx[query_str] = requery_idx + 1
        query_idx_str = f"{requery_idx}|{query_str}"

        cached_completion = self.current_cache.get(query_idx_str, None)

        if cached_completion is not None:
            cached_completion = Completion(cached_completion["text"], cached_completion["cum_logprob"], cached_completion["num_tokens"])
        
        return cached_completion, query_idx_str
   
    def update(self, query_idx_str: str, completion: Completion) -> bool:
        if self.disabled or query_idx_str == "" or query_idx_str in self.current_cache:
            return False

        self.current_cache[query_idx_str] = {"text": completion.code, "cum_logprob": completion.cumulative_logprob, "num_tokens": completion.num_tokens}
        return True
    
    def delete(self, query_idx_str: str) -> Optional[dict]:
        return self.current_cache.pop(query_idx_str, None)

    def _get_cache_query_str(self, model: str, convo: Prompt, frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float]):
        if isinstance(convo, list) or isinstance(convo, tuple):
            message_str = '|'.join([f"ROLE:{message['role']}|CONTENT:{message['content']}" for message in convo])
        else:
            assert isinstance(convo, str)
            message_str = convo
        if stop is None:
            stop_list = []
        elif isinstance(stop, str):
            stop_list = [stop]
        else:
            sorted(stop)
            stop_list = stop
        logit_bias_list = sorted(logit_bias.items()) if logit_bias is not None else []
        return f"MODEL:{model}|MESSAGES:{message_str}|FREQUENCY_PENALTY:{frequency_penalty}|LOGIT_BIAS_LIST:{logit_bias_list}|MAX_TOKENS:{max_tokens}|PRESENCE_PENALTY:{presence_penalty}|SEED:{seed}|STOP_LIST:{stop_list}|TEMPERATURE:{temperature}|TOP_P:{top_p}"


class LLMQuerier(ABC):
    def __init__(self, log_directory: Optional[str], cache_file: Optional[str], global_batch_size: Optional[int]) -> None:
        self.log_directory = log_directory
        self.clients: dict[str, LLMClient] = {}
        self.current_price = 0.
        self.next_print_price = PRINT_EVERY_X_DOLLARS

        self.cache = CompletionCache(cache_file)
        self.global_batch_size = global_batch_size

    def set_global_batch_size(self, global_batch_size: Optional[int] = None):
        self.global_batch_size = global_batch_size
    
    def add_client(self, client_path: str) -> bool:
        if client_path not in self.clients:
            self.clients[client_path] = LLMClient.from_json(client_path)
            return True
        return False

    def generate_with_info(self, client_name: str, prompts: list[Prompt], frequency_penalty: Optional[float] = None, logit_bias: Optional[dict[str, int]] = None, max_tokens: Optional[int] = None, presence_penalty: Optional[float] = None, seed: Optional[int] = None, stop: Union[Optional[str], list[str]] = None, temperature: Optional[float] = None, top_p: Optional[float] = None, requery: bool = True, log_name: str = "", timeout: Optional[float] = None, o1_retry: bool = True) -> list[Completion]:
        curr_hash = str(uuid4())[:6]
        self.add_client(client_name)
        print("generating completions...")
        
        if self.global_batch_size is not None:
            assert self.global_batch_size > 0
            chunked_prompts = list(chunk(prompts, self.global_batch_size))
        else:
            chunked_prompts = [prompts]

        all_completions = []

        with tqdm(total=len(prompts)) as pbar:
            total_failed_prompts = 0

            for i, prompt_chunk in enumerate(chunked_prompts):
                completions, cost = _generate_completions(self.clients[client_name], prompt_chunk,
                    cache=self.cache,
                    frequency_penalty=frequency_penalty,
                    logit_bias=logit_bias,
                    max_tokens=max_tokens,
                    presence_penalty=presence_penalty,
                    seed=seed,
                    stop=stop,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=timeout,
                    pbar=pbar,
                )
                self.log(prompt_chunk, completions, log_name=log_name,
                         frequency_penalty=frequency_penalty, logit_bias=logit_bias,
                         max_tokens=max_tokens, presence_penalty=presence_penalty,
                         seed=seed, stop=stop, temperature=temperature, top_p=top_p)

                self.current_price += cost
                if self.current_price > self.next_print_price:
                    print(f"Current spending: ${self.current_price:.2f}")
                    self.next_print_price = (self.current_price // PRINT_EVERY_X_DOLLARS + 1) * PRINT_EVERY_X_DOLLARS


                if self.clients[client_name].model_is_o1:
                    failed_prompts = []
                    orig_idxs = []
                    for j, completion in enumerate(completions):
                        if completion.code == self.clients[client_name].BAD_REQUEST_FLAG:

                            # used_prompt = prompt_chunk[j]
                            # assert isinstance(used_prompt, (list, tuple))
                            # if used_prompt[0]["role"] == "system":
                            #     used_prompt = used_prompt[1:]

                            # _, cache_key = self.cache.query(model=self.clients[client_name].model_name, message=used_prompt, frequency_penalty=frequency_penalty, logit_bias=logit_bias, max_tokens=max_tokens, presence_penalty=presence_penalty, seed=seed, stop=stop, temperature=temperature, top_p=top_p)
                            # assert self.cache.delete(cache_key)["text"] == self.clients[client_name].BAD_REQUEST_FLAG

                            failed_prompts.append(prompt_chunk[j])
                            orig_idxs.append(j)
                    

                    if len(failed_prompts):
                        if o1_retry:
                            print(f"{len(failed_prompts)} invalid prompt errors... Requerying with filter {O1_FILTER}! {curr_hash}")
                            filtered_prompts = []
                            for prompt in failed_prompts:
                                new_prompt = []
                                for msg in prompt:
                                    new_prompt.append({"role": msg["role"], "content": remove_from_str(msg["content"], O1_FILTER)})
                                filtered_prompts.append(new_prompt)
                            new_completions = self.generate_with_info(client_name=client_name, prompts=filtered_prompts, frequency_penalty=frequency_penalty, logit_bias=logit_bias, max_tokens=max_tokens, presence_penalty=presence_penalty, seed=seed, stop=stop, temperature=temperature, top_p=top_p, requery=requery, log_name=log_name, timeout=timeout, o1_retry=False)
                            print(f"Done with filter {curr_hash}")
                        else:
                            total_failed_prompts += len(failed_prompts)
                            print(f"{len(failed_prompts)} invalid prompt errors... Requerying with {DEFAULT_O1_REPLACEMENT}! {curr_hash}")
                            new_completions = self.generate_with_info(client_name=DEFAULT_O1_REPLACEMENT, prompts=failed_prompts, frequency_penalty=frequency_penalty, logit_bias=logit_bias, max_tokens=max_tokens, presence_penalty=presence_penalty, seed=seed, stop=stop, temperature=temperature, top_p=top_p, requery=requery, log_name=log_name, timeout=timeout, o1_retry=False)
                            print(f"Done requerying with {DEFAULT_O1_REPLACEMENT}. {curr_hash}")

                        assert len(orig_idxs) == len(new_completions)
                        for orig_idx, completion in zip(orig_idxs, new_completions):
                            completions[orig_idx] = completion

                print(f"done with {i+1}/{len(chunked_prompts)}, saving... {curr_hash}")
                self.cache.save_cache()
                all_completions.extend(completions)

        print(f"done {curr_hash}")
        if total_failed_prompts:
            print(f"{total_failed_prompts} failed invalid prompt out of {len(all_completions)}. {curr_hash}")
            print(f"an example completion: {all_completions[0].code[:200]}\n...\n{all_completions[0].code[-200:]} {curr_hash}")
        assert len(all_completions) == len(prompts)
        return all_completions

    def generate(self, client_name: str, prompts: list[Prompt], frequency_penalty: Optional[float] = None, logit_bias: Optional[dict[str, int]] = None, max_tokens: Optional[int] = None, presence_penalty: Optional[float] = None, seed: Optional[int] = None, stop: Union[Optional[str], list[str]] = None, temperature: Optional[float] = None, top_p: Optional[float] = None, requery: bool = True, log_name: str = "", timeout: Optional[float] = None) -> list[str]:
        generations = self.generate_with_info(client_name, prompts,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            seed=seed,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            requery=requery,
            log_name=log_name,
            timeout=timeout)
        return [c.code for c in generations]
    
    def log(self, prompts: list[Prompt], completions: list[Completion], log_name: str = "", frequency_penalty: Optional[float] = None, logit_bias: Optional[dict[str, int]] = None, max_tokens: Optional[int] = None, presence_penalty: Optional[float] = None, seed: Optional[int] = None, stop: Union[Optional[str], list[str]] = None, temperature: Optional[float] = None, top_p: Optional[float] = None):
        assert len(prompts) == len(completions)
        output_dict = {
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "max_tokens": max_tokens,
            "presence_penalty": presence_penalty,
            "seed": seed,
            "stop": stop,
            "temperature": temperature,
            "top_p": top_p,
            "queries": []
        }
        for prompt, completion in zip(prompts, completions):
            completion_dict = {"text": completion.code, "cum_logprob": completion.cumulative_logprob, "num_tokens": completion.num_tokens}
            output_dict["queries"].append({"prompt": prompt, "completion": completion_dict})

        if self.log_directory is not None:
            Path(self.log_directory).mkdir(parents=True, exist_ok=True)
            output_file = os.path.join(self.log_directory, log_name + datetime.datetime.now().strftime("%m-%dT%H:%M:%S") + ".json")
            with open(output_file, "w") as f:
                json.dump(output_dict, f, indent=2)
    
    def set_log_directory(self, new_log_directory: str):
        self.log_directory = new_log_directory

# Threaded generation code adapted from Federico Cassano
def _generate_completions(client: LLMClient, messages: Union[list[Prompt], tuple[Prompt, ...]], cache: CompletionCache, frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], timeout: Optional[float] = None, pbar: Optional[tqdm] = None) -> tuple[list[Completion], float]:
    if not isinstance(messages, (list, tuple)):
        raise ValueError("messages must be a list or tuple")
    assert len(messages)

    completions: list[Optional[Completion]] = [None] * len(messages)

    to_generate = []
    cache_keys = []
    for i, prompt in enumerate(messages):
        if client.model_is_o1:
            assert isinstance(prompt, (list, tuple))
            if prompt[0]["role"] == "system":
                prompt = prompt[1:]

        completion, cache_key = cache.query(model=client.model_name, message=prompt, frequency_penalty=frequency_penalty, logit_bias=logit_bias, max_tokens=max_tokens, presence_penalty=presence_penalty, seed=seed, stop=stop, temperature=temperature, top_p=top_p)

        if completion is not None:
            assert isinstance(completion, Completion)
            completions[i] = completion
            if pbar is not None:
                pbar.update(1)
        else:
            cache_keys.append((i, cache_key))
            to_generate.append(prompt)
    
    generated_completions, cost = client.generate_completions(messages=to_generate,
                                                        frequency_penalty=frequency_penalty,
                                                        logit_bias=logit_bias,
                                                        max_tokens=max_tokens,
                                                        presence_penalty=presence_penalty,
                                                        seed=seed,
                                                        stop=stop,
                                                        temperature=temperature,
                                                        top_p=top_p,
                                                        timeout=timeout,
                                                        pbar=pbar,
                                                        )

    assert len(cache_keys) == len(generated_completions)

    for (orig_idx, key), completion in zip(cache_keys, generated_completions):
        completions[orig_idx] = completion
        cache.update(key, completion)

    return completions, cost


if __name__ == "__main__":
    print("starting basic query...")
    llmq = LLMQuerier("temp_logs", None, 10)
    # print(llmq.generate("meta-llama/Meta-Llama-3-405B-Instruct", [[{"role": "user", "content": "Please count to 10."}]], max_tokens=1000, temperature=0.1, top_p=0.9))
    # print(llmq.generate("meta-llama/Meta-Llama-3-8B-Instruct", [[{"role": "user", "content": "Please count to 10."}]], max_tokens=1000, temperature=0.1, top_p=0.9))
    # print(llmq.generate("model_configs/deepseek-coder.json", [[{"role": "user", "content": "Please count to 10."}], [{"role": "user", "content": "keeeeeey"}]] * 4, max_tokens=100, temperature=0.1, top_p=0.9))
    # print(llmq.generate("model_configs/llama318bi_sglang.json", [[{"role": "user", "content": "Please count to 10."}], [{"role": "user", "content": "keeeeeey"}]] * 4, max_tokens=100, temperature=0.1, top_p=0.9))
    # print(llmq.generate("model_configs/model_configs/hsg_1.json", [[{"role": "user", "content": "Please count to 10."}], [{"role": "user", "content": "keeeeeey"}]] * 4, max_tokens=1000, temperature=0.1, top_p=0.9))
    # print(llmq.generate("claude-3-5-sonnet-20240620", ["What is up?"], max_tokens=1000, temperature=0.1, top_p=0.9))
    # print(llmq.generate("model_configs/sonnet-3-5.json", [[{"role": "user", "content": "Please count to 10."}], [{"role": "user", "content": "Please count to 100, backwards."}]], max_tokens=1000, temperature=0.1, top_p=0.9))
    # print(llmq.generate("model_configs/llama31405bi_fire.json", [[{"role": "user", "content": "Please count to 10."}], [{"role": "user", "content": "Please count to 100, backwards."}]], max_tokens=1000, temperature=0.1, top_p=0.9))
    # print(llmq.generate("model_configs/llama31405bi.json", [[{"role": "user", "content": "Please count to 10."}], [{"role": "user", "content": "keeeeeey"}]] * 4, max_tokens=100, temperature=0.1, top_p=0.9))
    print(llmq.generate("model_configs/o1-preview.json", [[{"role": "system", "content": "You are an expert counter."}, {"role": "user", "content": "Please count to 10."}], [{"role": "user", "content": "Please count to 100, backwards."}]], max_tokens=1000, temperature=1, top_p=1))
