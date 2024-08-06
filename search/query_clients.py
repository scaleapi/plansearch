from tqdm import tqdm
import tiktoken
from concurrent.futures import ThreadPoolExecutor
import openai
import anthropic
from httpcore import ReadError
from vllm import SamplingParams, LLM
from transformers import AutoTokenizer

from pathlib import Path
import random
from typing import Optional, Any, Union
import json
from abc import ABC, abstractmethod
import os
import warnings
import time

from coderm.prompts import Prompt
from coderm.model import Completion, logprobs_to_cumulative
from search.python_utils import chunk, random_print

PRINT_P = 0.02


class LLMClient:
    def __init__(self, model_name: str, is_chat: bool, is_batched: bool, batch_size: Optional[int] = None, num_workers: Optional[int] = None, price_per_input_output: tuple[float, float] = (0, 0)) -> None:
        self.model_name = model_name
        self.is_chat = is_chat
        self.is_batched = is_batched

        self.batch_size = batch_size
        if self.batch_size is not None and not is_batched:
            warnings.warn("LLMClient: Setting batch_size does not do anything when is_batched is False.")
            
        self.num_workers = num_workers
        if self.num_workers is not None and is_batched:
            warnings.warn("LLMClient: Setting num_workers does not do anything when is_batched is True.")

        self.is_loaded = False
        self.input_price = price_per_input_output[0]
        self.output_price = price_per_input_output[1]

    def num_tokens_for_convo(self, convo: Prompt, encoding: tiktoken.Encoding = tiktoken.get_encoding("cl100k_base")) -> int:
        num_tokens = 0
        if self.is_chat:
            for turn in convo:
                num_tokens += 1 + len(encoding.encode(turn["content"]))
        else:
            num_tokens = len(encoding.encode(convo))
        return num_tokens

    def load_model(self):
        if self.is_loaded:
            return
        self.is_loaded = True

    def generate_completions(self, messages: Union[list[Prompt], tuple[Prompt, ...]], frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], timeout: Optional[float] = None, pbar: Optional[tqdm] = None,) -> tuple[list[Completion], float]:
        completions: list[Optional[Completion]] = [None] * len(messages)
        if self.is_chat:
            assert all(isinstance(message_seq, list) for message_seq in messages)
        else:
            assert all(isinstance(message_seq, str) for message_seq in messages)

        indexed_messages = list(enumerate(messages))

        input_tokens = 0
        for message in messages:
            input_tokens += self.num_tokens_for_convo(message)
        
        if self.is_batched:
            to_generate_chunked = chunk(indexed_messages, self.batch_size)
            for chunk_to_generate in to_generate_chunked:
                to_generate_prompts = [prompt for _, prompt in chunk_to_generate]
                if len(to_generate_prompts):
                    genned_completions = self.batch_generate(
                        messages=to_generate_prompts,
                        frequency_penalty=frequency_penalty,
                        logit_bias=logit_bias,
                        max_tokens=max_tokens,
                        presence_penalty=presence_penalty,
                        seed=seed,
                        stop=stop,
                        temperature=temperature,
                        top_p=top_p,
                        timeout=timeout,
                    )
                    for (i, _), completion in zip(chunk_to_generate, genned_completions):
                        completions[i] = completion
                        if pbar is not None:
                            pbar.update(1)
        else:
            def generate_completion_util(prompt: Prompt, i: int):
                completions[i] = self.generate(
                        message=prompt,
                        frequency_penalty=frequency_penalty,
                        logit_bias=logit_bias,
                        max_tokens=max_tokens,
                        presence_penalty=presence_penalty,
                        seed=seed,
                        stop=stop,
                        temperature=temperature,
                        top_p=top_p,
                        timeout=timeout,
                    )
                if pbar is not None:
                    pbar.update(1)

            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                to_generate_prompts = [prompt for _, prompt in indexed_messages]
                to_generate_idxs = [idx for idx, _ in indexed_messages]
                list(executor.map(generate_completion_util, to_generate_prompts, to_generate_idxs))

        assert all(c is not None for c in completions), "Some completions are missing -- threading bug?"

        output_tokens = 0
        for completion in completions:
            output_tokens += completion.num_tokens

        total_price = input_tokens * self.input_price + output_tokens * self.output_price
        return completions, total_price

    def generate(self, message: Prompt, frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], timeout: Optional[float] = None) -> Completion:
        raise NotImplementedError("LLMClient: generate not supported")

    def batch_generate(self, messages: list[Prompt], frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], timeout: Optional[float] = None) -> list[Completion]:
        raise NotImplementedError("LLMCLient: batch_generate not supported")

    @staticmethod
    def from_json(file_name: str) -> "LLMClient":
        CLIENT_TYPE_TO_CLASS = {
            "OpenAI": OpenAIClient,
            "Anthropic": AnthropicClient,
            "vLLM": vLLMClient,
        }
        assert Path(file_name).exists(), f"Path {file_name} doesn't exist!"
        assert file_name.endswith(".json")
        with open(file_name, "r") as f:
            data = json.load(f)
        client_class = CLIENT_TYPE_TO_CLASS[data["client_type"]]
        del data["client_type"]
        return client_class(**data)


class OpenAIClient(LLMClient):
    TIMEOUT_FLAG = "__OPENAI_TIMEOUT__"
    JITTER_FACTOR = 3/4
    BACKOFF_FACTOR = 2
    def __init__(self, model_name: str, is_chat: bool, is_batched: bool, batch_size: Optional[int] = None, num_workers: Optional[int] = None, price_per_input_output: tuple[float, float] = (0, 0), start_backoff: float = 30, max_backoff: float = 3 * 60) -> None:
        super().__init__(model_name=model_name, is_chat=is_chat, is_batched=is_batched, batch_size=batch_size, num_workers=num_workers, price_per_input_output=price_per_input_output)
        self.start_backoff = start_backoff
        self.max_backoff = max_backoff
        self.client = openai.OpenAI()
        self.is_loaded = True
    
    def generate(self, message: Prompt, frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], timeout: Optional[float] = None) -> Completion:
        if not self.is_chat:
            raise NotImplementedError("OpenAIClient completion format not implemented.")

        curr_backoff = self.start_backoff
        while 1:
            try:
                start = time.time()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message,
                    frequency_penalty=frequency_penalty,
                    logit_bias=logit_bias,
                    logprobs=True,
                    max_tokens=max_tokens,
                    presence_penalty=presence_penalty,
                    seed=seed,
                    stop=stop,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=timeout,
                )
                total = time.time() - start
                if total >= 120:
                    print(f"Warning: Request took {int(total)} seconds")
                break
            except openai.BadRequestError as e:
                print(f"Bad request: {e}")
                print(message, "\n^ the bad prompt")
                return Completion("No output returned.", -1, 3)
            except openai.RateLimitError:
                random_print("OpenAI rate limit.", p=PRINT_P)
            except openai.APITimeoutError:
                if timeout is not None:
                    print(f"OpenAI API exceeded timeout of {timeout}.")
                    return Completion(self.TIMEOUT_FLAG, -1, 0)
                random_print("OpenAI API timeout.", p=PRINT_P)
            except ReadError:
                random_print("httpcore ReadError.", p=PRINT_P)
            except openai.APIConnectionError:
                random_print("OpenAI API connection error.", p=PRINT_P)
            except openai.InternalServerError:
                random_print("OpenAI internal server error.", p=PRINT_P)
            except json.JSONDecodeError:
                random_print("(OpenAI) JSON decode error.", p=PRINT_P)
            except UnicodeDecodeError:
                random_print("(OpenAI) Unicode decode error.", p=PRINT_P)
            
            curr_backoff = min(self.max_backoff, curr_backoff * self.BACKOFF_FACTOR + random.random() * self.JITTER_FACTOR * curr_backoff)
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


class AnthropicClient(LLMClient):
    JITTER_FACTOR = 3/4
    BACKOFF_FACTOR = 2
    def __init__(self, model_name: str, is_chat: bool, is_batched: bool, batch_size: Optional[int] = None, num_workers: Optional[int] = None, price_per_input_output: tuple[float, float] = (0, 0), start_backoff: float = 30, max_backoff: float = 3 * 60) -> None:
        super().__init__(model_name=model_name, is_chat=is_chat, is_batched=is_batched, batch_size=batch_size, num_workers=num_workers, price_per_input_output=price_per_input_output)
        self.start_backoff = start_backoff
        self.max_backoff = max_backoff
        self.client = anthropic.Anthropic()
        self.is_loaded = True
    
    def generate(self, message: Prompt, frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], timeout: Optional[float] = None) -> Completion:
        if not self.is_chat:
            raise NotImplementedError("AnthropicClient completion format not implemented.")
        curr_backoff = self.start_backoff
        while 1:
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=message,
                    max_tokens=max_tokens,
                    stop_sequences=stop,
                    temperature=temperature,
                    top_p=top_p,
                )
                break
            except anthropic.RateLimitError:
                random_print("Anthropic rate limit.", p=PRINT_P)
            except anthropic.APITimeoutError:
                random_print("Anthropic API timeout.", p=PRINT_P)
            except ReadError:
                random_print("httpcore ReadError.", p=PRINT_P)
            except anthropic.APIConnectionError:
                random_print("Anthropic API connection error.", p=PRINT_P)
            except anthropic.InternalServerError:
                random_print("Anthropic internal server error.", p=PRINT_P)
            except json.JSONDecodeError:
                random_print("(Anthropic) JSON decode error.", p=PRINT_P)
            except UnicodeDecodeError:
                random_print("(Anthropic) Unicode decode error.", p=PRINT_P)
            
            curr_backoff = min(self.max_backoff, curr_backoff * 2)
            random_print(f"Requerying in {curr_backoff} seconds.", p=PRINT_P)
            time.sleep(curr_backoff)
        
        o = response.content[0].text
        assert o is not None, "Anthropic returned a null response"
        num_tokens = response.usage.output_tokens
        if response.stop_reason == "max_tokens":
            print("Warning, output clipped.")

        return Completion(o, -1, num_tokens)
    
class vLLMClient(LLMClient):
    def __init__(self, model_name: str, is_chat: bool, is_batched: bool, batch_size: Optional[int] = None, num_workers: Optional[int] = None, price_per_input_output: tuple[float, float] = (0, 0), tensor_parallel_size: int = 8, gpu_memory_utilization: float = 0.9, max_model_len: int = 4096, dtype: str = "bfloat16", enforce_eager: bool = True) -> None:
        super().__init__(model_name, is_chat, is_batched, batch_size, num_workers, price_per_input_output)
        self.model = None
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self.enforce_eager = enforce_eager

    def load_model(self):
        super().load_model()
        self.model = LLM(self.model_name, trust_remote_code=True, tensor_parallel_size=self.tensor_parallel_size, gpu_memory_utilization=self.gpu_memory_utilization, dtype=self.dtype, enforce_eager=self.enforce_eager)
    
    def batch_generate(self, messages: list[Prompt], frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], timeout: Optional[float] = None) -> tuple[list[Completion], float]:
        if not self.is_loaded:
            self.load_model()
        assert self.model is not None
        if self.is_chat:
            return self._generate_vLLM_chat_completions(
                prompts=messages,
                frequency_penalty=frequency_penalty,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                seed=seed,
                stop=stop,
                temperature=temperature,
                top_p=top_p
            )
        else:
            return self._generate_vLLM_completions(
                prompts=messages,
                frequency_penalty=frequency_penalty,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                seed=seed,
                stop=stop,
                temperature=temperature,
                top_p=top_p
            )

    def _generate_vLLM_chat_completions(self, prompts: list[list[dict[str, str]]], frequency_penalty: Optional[float], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], **kwargs) -> list[Completion]:
        tokenizer = self.model.get_tokenizer()
        chat_msgs_as_str = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
        return self._generate_vLLM_completions(chat_msgs_as_str, frequency_penalty=frequency_penalty, max_tokens=max_tokens, presence_penalty=presence_penalty, seed=seed, stop=stop, temperature=temperature, top_p=top_p)

    def _generate_vLLM_completions(self, prompts: list[str], frequency_penalty: Optional[float], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], **kwargs) -> list[Completion]:
        sampling_params = SamplingParams(
            n=1,
            frequency_penalty=0 if frequency_penalty is None else frequency_penalty,
            presence_penalty=0 if presence_penalty is None else presence_penalty,
            seed=seed,
            max_tokens=max_tokens,
            stop=stop,
            temperature=0.0 if temperature is None else temperature,
            top_p=0.9 if top_p is None else top_p ,
        )
        outputs = self.model.generate(prompts, sampling_params=sampling_params)
        return [Completion(output.outputs[0].text, output.outputs[0].cumulative_logprob, len(output.outputs[0].token_ids)) for output in outputs]


if __name__ == "__main__":
    lc = LLMClient.from_json("model_configs/gpt-4o-mini.json")
    print(lc.generate_completions([[{"role": "user", "content": "what is up my dude?"}]] * 100, frequency_penalty=None, logit_bias=None, max_tokens=1000, presence_penalty=None, seed=None, stop=None, temperature=0.5, top_p=1))
