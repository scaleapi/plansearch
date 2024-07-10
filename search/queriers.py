import openai
import anthropic

from typing import Optional, Union, Any
import os
from abc import ABC, abstractmethod
import threading
import datetime
import json
from pathlib import Path
import time
from httpcore import ReadError

from coderm.prompts import Prompt
from coderm.model import Completion, logprobs_to_cumulative


MODELS_TO_METHOD = {
    "gpt-3.5-turbo-instruct": "completions",
    "gpt-3.5-turbo": "chat",
    "gpt-3.5-turbo-16k": "chat",
    "gpt-4-turbo-preview": "chat",
    "gpt-4-turbo": "chat",
    "claude-2.1": "anthropicchat",
    "meta-llama/Meta-Llama-3-8B-Instruct": "vllm",
    "casperhansen/llama-3-70b-instruct-awq": "vllm",
    "Meta-Llama-3-70B-Instruct.Q4_K_S.gguf": "llama_cpp_hf",
    "Meta-Llama-3-70B-Instruct.IQ3_XS.gguf": "llama_cpp_hf",
    "together-llama-3-70b": "together"
}

SUPPORTED_ASSISTANT_STR = ["vllm", "llama_cpp_hf"]


MODEL_NAME_TO_CLIENT_STR = {
    "gpt-4-turbo": "OpenAI",
    "claude-3-5-sonnet-20240620": "Anthropic",
}
CLIENT_STR_TO_CLIENT = {
    "OpenAI": openai.OpenAI,
    "Anthropic": anthropic.Anthropic
}

MAX_TIMEOUT = 32 + 1e-4
START_TIMEOUT = 1/8


class LLMQuerier(ABC):
    def __init__(self, log_directory: Optional[str] = None, cache_file: Optional[str] = "cache.json") -> None:
        self.log_directory = log_directory
        self.clients = {}

        self.cache_file = cache_file
        if self.cache_file is not None:
            if not os.path.exists(self.cache_file):
                with open(self.cache_file, "w") as f:
                    json.dump({}, f)
            with open(self.cache_file, "r") as f:
                self.current_cache = json.load(f)

            self.temp_cache_file = os.path.join(os.path.dirname(self.cache_file), "temp_" + os.path.basename(self.cache_file))
            if not os.path.exists(self.temp_cache_file):
                with open(self.temp_cache_file, "w") as f:
                    json.dump({}, f)

        self.requery_cache = {}


    def generate_with_info(self, model: str, prompts: list[Prompt], frequency_penalty: Optional[float] = None, logit_bias: Optional[dict[str, int]] = None, max_tokens: Optional[int] = None, presence_penalty: Optional[float] = None, seed: Optional[int] = None, stop: Union[Optional[str], list[str]] = None, temperature: Optional[float] = None, top_p: Optional[float] = None, requery: bool = False, log_name: str = "") -> list[Completion]:
        assert model in MODEL_NAME_TO_CLIENT_STR
        if MODEL_NAME_TO_CLIENT_STR[model] not in self.clients:
            self.clients[MODEL_NAME_TO_CLIENT_STR[model]] = CLIENT_STR_TO_CLIENT[MODEL_NAME_TO_CLIENT_STR[model]]()
        print("generating completions...")
        completions = _generate_completions(self.clients[MODEL_NAME_TO_CLIENT_STR[model]], model, messages=prompts,
            current_cache=self.current_cache if self.cache_file is not None else {},
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            seed=seed,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            requery_cache=self.requery_cache if requery else None,
            )

        self.log(prompts, completions, log_name=log_name)
        print("done")
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


def _generate_anthropic_completion(client: anthropic.Anthropic, model: str, prompt: list[dict[str, str]], max_tokens: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], **kwargs) -> Completion:
    curr_backoff = START_TIMEOUT
    while 1:
        try:
            response = client.messages.create(
                model=model,
                messages=prompt,
                max_tokens=max_tokens,
                stop_sequences=stop,
                temperature=temperature,
                top_p=top_p,
            )
            break
        except anthropic.RateLimitError:
            print("Anthropic rate limit.")
        except anthropic.APITimeoutError:
            print("Anthropic API timeout.")
        except ReadError:
            print("httpcore ReadError.")
        except anthropic.APIConnectionError:
            print("Anthropic API connection error.")
        except anthropic.InternalServerError:
            print("Anthropic internal server error.")
        except json.JSONDecodeError:
            print("(Anthropic) JSON decode error.")
        except UnicodeDecodeError:
            print("(Anthropic) Unicode decode error.")
        
        curr_backoff = min(MAX_TIMEOUT, curr_backoff * 2)
        print(f"Requerying in {curr_backoff} seconds.")
        time.sleep(curr_backoff)
    
    o = response.content[0].text
    assert o is not None, "Anthropic returned a null response"
    num_tokens = response.usage.output_tokens
    if response.stop_reason == "max_tokens":
        print("Warning, output clipped.")

    return Completion(o, -1, num_tokens)


def _generate_openai_completion(client: openai.OpenAI, model: str, prompt: list[dict[str, str]], frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], **kwargs) -> Completion:
    curr_backoff = START_TIMEOUT
    while 1:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=prompt,
                frequency_penalty=frequency_penalty,
                logit_bias=logit_bias,
                logprobs=True,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                seed=seed,
                stop=stop,
                temperature=temperature,
                top_p=top_p,
            )
            break
        except openai.RateLimitError:
            print("OpenAI rate limit.")
        except openai.APITimeoutError:
            print("OpenAI API timeout.")
        except ReadError:
            print("httpcore ReadError.")
        except openai.APIConnectionError:
            print("OpenAI API connection error.")
        except openai.InternalServerError:
            print("OpenAI internal server error.")
        except json.JSONDecodeError:
            print("(OpenAI) JSON decode error.")
        except UnicodeDecodeError:
            print("(OpenAI) Unicode decode error.")
        
        curr_backoff = min(MAX_TIMEOUT, curr_backoff * 2)
        print(f"Requerying in {curr_backoff} seconds.")
        time.sleep(curr_backoff)
    
    choice = response.choices[0]
    o = choice.message.content
    logprobs = choice.logprobs.content  # type: ignore
    assert o is not None, "OpenAI returned a null response"
    assert logprobs is not None, "OpenAI returned a null logprobs"
    logprobs = [l.logprob for l in logprobs]
    num_tokens = len(logprobs)
    if choice.finish_reason == "length":
        print("Warning, output clipped.")

    cumulative_logprob = logprobs_to_cumulative(logprobs)
    return Completion(o, cumulative_logprob, num_tokens)


MODEL_NAME_TO_COMPLETION_FN = {
    "gpt-4-turbo": _generate_openai_completion,
    "claude-3-5-sonnet-20240620": _generate_anthropic_completion,
}


def cache_hash(model: str, messages: list[dict[str, str]], frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float]):
    message_str = [f"ROLE:{message['role']}|CONTENT:{message['content']}" for message in messages]
    if stop is None:
        stop_list = []
    elif isinstance(stop, str):
        stop_list = [stop]
    else:
        sorted(stop)
        stop_list = stop
    logit_bias_list = sorted(logit_bias.items()) if logit_bias is not None else []
    return f"MODEL:{model}|MESSAGES:{'|'.join(message_str)}|FREQUENCY_PENALTY:{frequency_penalty}|LOGIT_BIAS_LIST:{logit_bias_list}|MAX_TOKENS:{max_tokens}|PRESENCE_PENALTY:{presence_penalty}|SEED:{seed}|STOP_LIST:{stop_list}|TEMPERATURE:{temperature}|TOP_P:{top_p}"


# Threaded generation code adapted from Federico Cassano
def _generate_completions(client: Union[openai.OpenAI, anthropic.Anthropic], model: str, messages: Union[list[Prompt], tuple[Prompt, ...]], current_cache: dict[str, dict[str, Any]], frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], requery_cache: Optional[dict[str, int]] = None) -> list[Completion]:
    if not isinstance(messages, (list, tuple)):
        raise ValueError("messages must be a list or tuple")
    
    completion_generator_fn = MODEL_NAME_TO_COMPLETION_FN[model]

    completions: list[Optional[Completion]] = [None] * len(messages)
    threads = []

    messages = [[{"role": "user", "content": message}] if isinstance(message, str) else message for message in messages]
    assert all(isinstance(message_seq, list) for message_seq in messages)

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

    not_cached = []
    new_requery_cache = {}
    for i, prompt in enumerate(messages):
        cache_key = cache_hash(model, prompt, frequency_penalty, logit_bias, max_tokens, presence_penalty, seed, stop, temperature, top_p)
        cache_i = i

        if requery_cache is not None:
            cache_i += requery_cache.get(cache_key, 0)
            new_requery_cache[cache_key] = max(cache_i, new_requery_cache.get(cache_key, 0))

        cache_key = f"{cache_i}|{cache_key}"

        if cache_key in current_cache:
            completions[i] = Completion(current_cache[cache_key]["text"], current_cache[cache_key]["cum_logprob"], current_cache[cache_key]["num_tokens"])
        else:
            not_cached.append((i, cache_key))
            thread = threading.Thread(
                target=generate_completion, args=(prompt, i))
            threads.append(thread)
            thread.start()

    for thread in threads:
        thread.join()
    
    assert all(c is not None for c in completions), "Some completions are missing -- threading bug?"

    for uncached_idx, key in not_cached:
        current_cache[key] = {"text": completions[uncached_idx].code, "cum_logprob": completions[uncached_idx].cumulative_logprob, "num_tokens": completions[uncached_idx].num_tokens}

    for cache, i in new_requery_cache.items():
        requery_cache[cache] = max(i, requery_cache.get(cache, 0))
    
    return completions 


if __name__ == "__main__":
    print("starting basic query...")
    llmq = LLMQuerier("temp_logs", None)
    print(llmq.generate("gpt-4-turbo", ["Please count to 10."], max_tokens=1000, temperature=0.1, top_p=0.9))
    # print(llmq.generate("claude-3-5-sonnet-20240620", ["What is up?"], max_tokens=1000, temperature=0.1, top_p=0.9))