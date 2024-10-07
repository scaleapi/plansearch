from tqdm import tqdm
import tiktoken
import openai
import anthropic
import together
from httpcore import ReadError
from urllib.error import URLError
from vllm import SamplingParams, LLM
from transformers import AutoTokenizer
import sglang as sgl
import llmengine

import requests
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
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

PRINT_P = 1

def check_if_o1(model_name: str) -> bool:
    return model_name.startswith("o1-")

class LLMClient:
    PARAMS = ["model_name", "is_chat", "is_batched", "batch_size", "num_workers", "price_per_input_output"]
    BAD_REQUEST_FLAG = "No output returned."
    def __init__(self, model_name: str, is_chat: bool, is_batched: bool, batch_size: Optional[int] = None, num_workers: Optional[int] = None, price_per_input_output: tuple[float, float] = (0., 0.)) -> None:
        assert isinstance(model_name, str)
        assert isinstance(is_chat, bool)
        assert isinstance(is_batched, bool)

        self.model_name = model_name
        self.is_chat = is_chat
        self.is_batched = is_batched
        self.model_is_o1 = check_if_o1(model_name)

        assert batch_size is None or isinstance(batch_size, int)
        self.batch_size = batch_size
        if self.batch_size is not None and not is_batched:
            warnings.warn("LLMClient: Setting batch_size does not do anything when is_batched is False.")
            
        assert num_workers is None or isinstance(num_workers, int)
        self.num_workers = num_workers
        if self.num_workers is not None and is_batched:
            warnings.warn("LLMClient: Setting num_workers does not do anything when is_batched is True.")

        self.is_loaded = False

        self.input_price = float(price_per_input_output[0])
        self.output_price = float(price_per_input_output[1])
        assert isinstance(self.input_price, float) and isinstance(self.output_price, float)

    def num_tokens_for_convo(self, convo: Prompt, encoding: tiktoken.Encoding = tiktoken.get_encoding("cl100k_base"), approx: bool = False) -> int:
        num_tokens = 0
        content_str = []
        if self.is_chat:
            for turn in convo:
                num_tokens += 1
                content_str.append(turn["content"])
        else:
            content_str.append(convo)

        for content in content_str:
            if not approx:
                num_tokens += len(encoding.encode(content))
            else:
                num_tokens += len(content) / 4

        return int(num_tokens)

    def load_model(self):
        if self.is_loaded:
            return
        self.is_loaded = True

    def generate_completions(self, messages: Union[list[Prompt], tuple[Prompt, ...]], frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], timeout: Optional[float] = None, pbar: Optional[tqdm] = None,) -> tuple[list[Completion], float]:
        completions: list[Optional[Completion]] = [None] * len(messages)
        if self.is_chat:
            assert all((isinstance(message_seq, list) or isinstance(message_seq, tuple)) for message_seq in messages)
        else:
            assert all(isinstance(message_seq, str) for message_seq in messages)

        indexed_messages = list(enumerate(messages))

        input_tokens = 0
        for message in messages:
            input_tokens += self.num_tokens_for_convo(message, approx=True)
        
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
        assert Path(file_name).exists(), f"Path {file_name} doesn't exist!"
        assert file_name.endswith(".json")
        with open(file_name, "r") as f:
            data = json.load(f)
        client_class = CLIENT_TYPE_TO_CLASS[data["client_type"]]
        del data["client_type"]
        return client_class(**data)


class OpenAIClient(LLMClient):
    TIMEOUT_FLAG = "__OPENAI_TIMEOUT__"
    JITTER_FACTOR = 2/5
    BACKOFF_FACTOR = 2
    PARAMS = LLMClient.PARAMS + ["start_backoff", "max_backoff"]
    def __init__(self, model_name: str, is_chat: bool, is_batched: bool, batch_size: Optional[int] = None, num_workers: Optional[int] = None, price_per_input_output: tuple[float, float] = (0., 0.), start_backoff: float = 45., max_backoff: float = 3. * 60) -> None:
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
                # # Create a unique filename based on the current time and a random number
                # unique_number = random.randint(100000, 999999)
                # log_directory = "long_logs"
                # if not os.path.exists(log_directory):
                #     os.makedirs(log_directory)
                # log_filename = os.path.join(log_directory, f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_number}.json")

                # # Log the query details before making the request
                # query_log = {
                #     "model": self.model_name,
                #     "messages": message,
                #     "frequency_penalty": frequency_penalty,
                #     "logit_bias": logit_bias,
                #     "max_tokens": max_tokens,
                #     "presence_penalty": presence_penalty,
                #     "seed": seed,
                #     "stop": stop,
                #     "temperature": temperature,
                #     "top_p": top_p,
                #     "timeout": timeout
                # }

                # with open(log_filename, "w") as log_file:
                #     json.dump(query_log, log_file)

                start = time.time()
                if self.model_is_o1:
                    assert stop is None or stop == []
                    assert temperature == 1
                    assert top_p == 1
                    assert frequency_penalty is None or frequency_penalty == 0
                    assert presence_penalty is None or presence_penalty == 0
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=message,
                        frequency_penalty=0,
                        logit_bias=logit_bias,
                        logprobs=False,
                        max_completion_tokens=max_tokens,
                        presence_penalty=0,
                        seed=seed,
                        temperature=temperature,
                        top_p=top_p,
                        timeout=timeout,
                    )
                else:
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
                    print(f"Warning: Request took {int(total)} seconds.")
                #     response_log_filename = os.path.join(log_directory, f"response_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_number}.json")
                #     print(f"Warning: Request took {int(total)} seconds, saving to {response_log_filename}")
                #     # Save the log with the response text to a different file
                #     with open(response_log_filename, "w") as response_log_file:
                #         query_log["response"] = response.choices[0].message.content
                #         json.dump(query_log, response_log_file)
                #     print(f"Response log saved to: {response_log_filename}")
                # else:
                #     # Delete the log file if the request time is less than 120 seconds
                #     print(log_filename)
                #     if os.path.exists(log_filename):
                #         print("EXIST")
                #         try:
                #             os.remove(log_filename)
                #         except FileNotFoundError:
                #             print("whoops not found")
                #             pass

                break
            except openai.BadRequestError as e:
                print(f"Bad request: {e}")
                print(message, "\n^ the bad prompt")
                return Completion(self.BAD_REQUEST_FLAG, -1, 3)
            except openai.RateLimitError:
                random_print("OpenAI rate limit.", p=PRINT_P)
            except openai.APITimeoutError:
                total = time.time() - start
                if timeout is not None:
                    print(f"OpenAI API exceeded timeout of {timeout}. ({total} seconds)")
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
            
            curr_backoff = curr_backoff + random.random() * self.JITTER_FACTOR * curr_backoff
            random_print(f"OpenAIClient: requerying in {curr_backoff} seconds.", p=PRINT_P)
            time.sleep(curr_backoff)
            curr_backoff = min(self.max_backoff, curr_backoff * self.BACKOFF_FACTOR)

        choice = response.choices[0]
        o = choice.message.content
        if o is None:
            breakpoint()

        if not self.model_is_o1:
            if choice.logprobs is None:
                breakpoint()
                print("Warning null logprobs")
                cumulative_logprob = -1
            else:
                logprobs = choice.logprobs.content  # type: ignore
                logprobs = [l.logprob for l in logprobs]
                cumulative_logprob = logprobs_to_cumulative(logprobs)
        else:
            cumulative_logprob = -1
            
        assert o is not None, "OpenAI returned a null response"
        num_tokens = response.usage.completion_tokens
        if self.model_is_o1:
            pass
            # print("NUM TOKENS:", num_tokens)
        if choice.finish_reason == "length":
            print("Warning, output clipped.")

        return Completion(o, cumulative_logprob, num_tokens)


class AnthropicClient(LLMClient):
    PARAMS = LLMClient.PARAMS + ["start_backoff", "max_backoff"]
    JITTER_FACTOR = 3/5
    BACKOFF_FACTOR = 2
    def __init__(self, model_name: str, is_chat: bool, is_batched: bool, batch_size: Optional[int] = None, num_workers: Optional[int] = None, price_per_input_output: tuple[float, float] = (0., 0.), start_backoff: float = 35., max_backoff: float = 3. * 60) -> None:
        super().__init__(model_name=model_name, is_chat=is_chat, is_batched=is_batched, batch_size=batch_size, num_workers=num_workers, price_per_input_output=price_per_input_output)
        self.start_backoff = start_backoff
        self.max_backoff = max_backoff
        self.client = anthropic.Anthropic()
        self.is_loaded = True
    
    def generate(self, message: Prompt, frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], timeout: Optional[float] = None) -> Completion:
        if not self.is_chat:
            raise NotImplementedError("AnthropicClient completion format not implemented.")

        assert isinstance(message, (list, tuple))
        system_prompt = None
        for i, individual_msg in enumerate(message):
            if individual_msg["role"] == "system":
                assert i == 0, "Anthropic only supports system prompt at first message"
                system_prompt = individual_msg["content"]
        if system_prompt is not None:
            message = message[1:]
        
        curr_backoff = self.start_backoff
        while 1:
            try:
                if system_prompt is not None:
                    response = self.client.messages.create(
                        model=self.model_name,
                        messages=message,
                        system=system_prompt,
                        max_tokens=max_tokens,
                        stop_sequences=stop,
                        temperature=temperature,
                        top_p=top_p,
                    )
                else:
                    response = self.client.messages.create(
                        model=self.model_name,
                        messages=message,
                        max_tokens=max_tokens,
                        stop_sequences=stop,
                        temperature=temperature,
                        top_p=top_p,
                    )
                break
            except anthropic.RateLimitError as e:
                random_print("Anthropic rate limit." + str(e), p=PRINT_P)
            except anthropic.APITimeoutError:
                random_print("Anthropic API timeout.", p=PRINT_P)
            except anthropic.BadRequestError as e:
                print(f"Anthropic Bad request: {e}")
                print(message, "\n^ the bad prompt")
                return Completion(self.BAD_REQUEST_FLAG, -1, 3)
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
            
            curr_backoff = curr_backoff + random.random() * self.JITTER_FACTOR * curr_backoff
            random_print(f"AnthropicClient: requerying in {curr_backoff} seconds.", p=PRINT_P)
            time.sleep(curr_backoff)
            curr_backoff = min(self.max_backoff, curr_backoff * self.BACKOFF_FACTOR)
       
        o = response.content[0].text
        assert o is not None, "Anthropic returned a null response"
        num_tokens = response.usage.output_tokens
        if response.stop_reason == "max_tokens":
            print("Warning, output clipped.")

        return Completion(o, -1, num_tokens)
    
class vLLMClient(LLMClient):
    PARAMS = LLMClient.PARAMS + ["tensor_parallel_size", "gpu_memory_utilization", "max_model_len", "dtype", "enforce_eager"]
    def __init__(self, model_name: str, is_chat: bool, is_batched: bool, batch_size: Optional[int] = None, num_workers: Optional[int] = None, price_per_input_output: tuple[float, float] = (0., 0.), tensor_parallel_size: int = 8, gpu_memory_utilization: float = 0.9, max_model_len: int = 4096, dtype: str = "bfloat16", enforce_eager: bool = True) -> None:
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
    
    def batch_generate(self, messages: list[Prompt], frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], timeout: Optional[float] = None) -> list[Completion]:
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
        assert isinstance(self.model, LLM)
        tokenizer = self.model.get_tokenizer()
        chat_msgs_as_str = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
        return self._generate_vLLM_completions(chat_msgs_as_str, frequency_penalty=frequency_penalty, max_tokens=max_tokens, presence_penalty=presence_penalty, seed=seed, stop=stop, temperature=temperature, top_p=top_p)

    def _generate_vLLM_completions(self, prompts: list[str], frequency_penalty: Optional[float], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], **kwargs) -> list[Completion]:
        assert isinstance(self.model, LLM)
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

        completions = []
        for output in outputs:
            completions.append(Completion(output.outputs[0].text, output.outputs[0].cumulative_logprob, len(output.outputs[0].token_ids)))
            if output.outputs[0].finish_reason == "length":
                warnings.warn("vLLMClient: output clipped.")
                print("WARNING: vLLMClient output clipped.")
        return completions


class SGLangClient(LLMClient):
    BACKOFF = 15
    PARAMS = LLMClient.PARAMS + ["base_url", "bac"]
    def __init__(self, model_name: str, base_url: str, is_chat: bool, is_batched: bool, batch_size: Optional[int] = None, num_workers: Optional[int] = None, price_per_input_output: tuple[float, float] = (0., 0.)) -> None:
        super().__init__(model_name=model_name, is_chat=is_chat, is_batched=is_batched, batch_size=batch_size, num_workers=num_workers, price_per_input_output=price_per_input_output)

        self.base_url = base_url

    def load_model(self):
        while 1:
            try:
                sgl.set_default_backend(sgl.RuntimeEndpoint(self.base_url))
                break
            except URLError as e:
                print(f"SGLangClient: OpenAI API connection error. Make sure SGLang server is running at {self.base_url}")
            
            curr_backoff = self.BACKOFF
            random_print(f"SGLangClient: requerying in {curr_backoff} seconds.")
            time.sleep(curr_backoff)

        self.is_loaded = True
   
    @sgl.function
    def sgl_chat_generate(s, convo: list[dict[str, str]], frequency_penalty: Optional[float], max_tokens: Optional[int], presence_penalty: Optional[float], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float]):
        assert convo[-1]["role"] == "user", f"SGLangClient last role must be user. Got {convo[-1]['role']}"
        for c in convo:
            if c["role"] == "system":
                s += sgl.system(c["content"])
            elif c["role"] == "user":
                s += sgl.user(c["content"])
            elif c["role"] == "assistant":
                s += sgl.assistant(c["content"])
            else:
                raise NotImplementedError(f'Role c["role"] not supported in SGLangClient')
        
        s += sgl.assistant(sgl.gen("response", max_tokens=max_tokens, stop=stop, temperature=temperature, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, return_logprob=True))

    @sgl.function
    def sgl_completion_generate(s, prompt: str, frequency_penalty: Optional[float], max_tokens: Optional[int], presence_penalty: Optional[float], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float]):
        s += prompt + sgl.gen("response", max_tokens=max_tokens, stop=stop, temperature=temperature, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, return_logprob=True)

    def batch_generate(self, messages: list[Prompt], frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], timeout: Optional[float] = None) -> list[Completion]:
        if not self.is_loaded:
            self.load_model()
        assert self.is_loaded

        if not self.is_chat:
            assert(all(isinstance(msg, str) for msg in messages))
            states = self.sgl_completion_generate.run_batch(
                [{"prompt": message,
                    "frequency_penalty": frequency_penalty,
                    "max_tokens": max_tokens,
                    "presence_penalty": presence_penalty,
                    "stop": stop,
                    "temperature": temperature,
                    "top_p": top_p}
                    for message in messages],
                    progress_bar=True
            )
        else:
            states = self.sgl_chat_generate.run_batch(
                [{"convo": convo,
                    "frequency_penalty": frequency_penalty,
                    "max_tokens": max_tokens,
                    "presence_penalty": presence_penalty,
                    "stop": stop,
                    "temperature": temperature,
                    "top_p": top_p}
                    for convo in messages],
                    progress_bar=True
            )
        
        completions = []
        for state in states:
            output_text = state["response"]
            if self.is_chat:
                assert state.messages()[-1]["role"] == "assistant"
                assert output_text.strip() == state.messages()[-1]["content"].strip()
                output_text = state.messages()[-1]["content"]

            meta = state.get_meta_info("response")
            logprobs = [l[0] for l in meta["output_token_logprobs"]]
            num_tok = meta["completion_tokens"]
            cum_lp = logprobs_to_cumulative(logprobs)

            if "FINISH_LENGTH" in meta["finish_reason"]:
                warnings.warn("SGLangClient: output clipped.")
                print("WARNING: SGLangClient output clipped.")
            
            completions.append(Completion(output_text, cum_lp, num_tok))
        return completions


class DeepSeekClient(LLMClient):
    TIMEOUT_FLAG = "__DEEPSEEK_TIMEOUT__"
    JITTER_FACTOR = 2/5
    BACKOFF_FACTOR = 2
    PARAMS = LLMClient.PARAMS + ["start_backoff", "max_backoff"]
    def __init__(self, model_name: str, is_chat: bool, is_batched: bool, batch_size: Optional[int] = None, num_workers: Optional[int] = None, price_per_input_output: tuple[float, float] = (0., 0.), start_backoff: float = 45., max_backoff: float = 3. * 60) -> None:
        super().__init__(model_name=model_name, is_chat=is_chat, is_batched=is_batched, batch_size=batch_size, num_workers=num_workers, price_per_input_output=price_per_input_output)
        self.start_backoff = start_backoff
        self.max_backoff = max_backoff

        api_key = os.getenv("DEEPSEEK_API_KEY")
        assert api_key is not None

        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.is_loaded = True
    
    def generate(self, message: Prompt, frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], timeout: Optional[float] = None) -> Completion:
        if not self.is_chat:
            raise NotImplementedError("DeepSeekClient completion format not implemented.")

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
                    print(f"Warning: Request took {int(total)} seconds.")
                break
            except openai.BadRequestError as e:
                print(f"Bad request: {e}")
                print(message, "\n^ the bad prompt")
                return Completion(self.BAD_REQUEST_FLAG, -1, 3)
            except openai.RateLimitError:
                random_print("DeepSeekAI rate limit.", p=PRINT_P)
            except openai.APITimeoutError:
                total = time.time() - start
                if timeout is not None:
                    print(f"DeepSeekAI API exceeded timeout of {timeout}. ({total} seconds)")
                    return Completion(self.TIMEOUT_FLAG, -1, 0)
                random_print("DeepSeekAI API timeout.", p=PRINT_P)
            except ReadError:
                random_print("httpcore ReadError.", p=PRINT_P)
            except openai.APIConnectionError:
                random_print("DeepSeekAI API connection error.", p=PRINT_P)
            except openai.InternalServerError:
                random_print("DeepSeekAI internal server error.", p=PRINT_P)
            except json.JSONDecodeError:
                random_print("(DeepSeekAI) JSON decode error.", p=PRINT_P)
            except UnicodeDecodeError:
                random_print("(DeepSeekAI) Unicode decode error.", p=PRINT_P)
            
            curr_backoff = curr_backoff + random.random() * self.JITTER_FACTOR * curr_backoff
            random_print(f"DeepSeekAI: requerying in {curr_backoff} seconds.", p=PRINT_P)
            time.sleep(curr_backoff)
            curr_backoff = min(self.max_backoff, curr_backoff * self.BACKOFF_FACTOR)

        choice = response.choices[0]
        o = choice.message.content
        if choice.logprobs is None:
            print("A5151")
            return Completion("N/A", -1, -1)
            breakpoint()
        logprobs = choice.logprobs.content  # type: ignore
        assert o is not None, "DeepSeekAI returned a null response"
        assert logprobs is not None, "DeepSeekAI returned a null logprobs"
        logprobs = [l.logprob for l in logprobs]
        num_tokens = len(logprobs)
        if choice.finish_reason == "length":
            print("Warning, output clipped.")

        cumulative_logprob = logprobs_to_cumulative(logprobs)
        return Completion(o, cumulative_logprob, num_tokens)


class LLMEngineClient(LLMClient):
    JITTER_FACTOR = 2/5
    BACKOFF_FACTOR = 2
    PARAMS = LLMClient.PARAMS + ["start_backoff", "max_backoff", "tokenizer_model"]
    def __init__(self, model_name: str, is_chat: bool, is_batched: bool, batch_size: Optional[int] = None, num_workers: Optional[int] = None, price_per_input_output: tuple[float, float] = (0., 0.), start_backoff: float = 45., max_backoff: float = 3. * 60, tokenizer_model: Optional[str] = None) -> None:
        super().__init__(model_name=model_name, is_chat=is_chat, is_batched=is_batched, batch_size=batch_size, num_workers=num_workers, price_per_input_output=price_per_input_output)
        self.start_backoff = start_backoff
        self.max_backoff = max_backoff
        self.tokenizer_model = tokenizer_model if tokenizer_model is not None else self.model_name
        self.tokenizer = None
        self.is_loaded = False
    
    def load_model(self):
        if self.is_chat:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model)
        self.is_loaded = True

    def generate(self, message: Prompt, frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], timeout: Optional[float] = None) -> Completion:
        if not self.is_loaded:
            self.load_model()

        if self.is_chat:
            return self._generate_llm_engine_chat_completion(
                prompt=message,
                frequency_penalty=frequency_penalty,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                seed=seed,
                stop=stop,
                temperature=temperature,
                top_p=top_p
            )
        else:
            return self._generate_llm_engine_completion(
                prompt=message,
                frequency_penalty=frequency_penalty,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                seed=seed,
                stop=stop,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout
            )

    def _generate_llm_engine_chat_completion(self, prompt: list[dict[str, str]], frequency_penalty: Optional[float], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], **kwargs) -> Completion:
        assert self.tokenizer is not None
        chat_msgs_as_str = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        assert isinstance(chat_msgs_as_str, str)
        return self._generate_llm_engine_completion(chat_msgs_as_str, frequency_penalty=frequency_penalty, max_tokens=max_tokens, presence_penalty=presence_penalty, seed=seed, stop=stop, temperature=temperature, top_p=top_p)

    def _generate_llm_engine_completion(self, prompt: str, frequency_penalty: Optional[float], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], **kwargs) -> Completion:
        if seed is not None:
            print("Warning: seed is not used")
        curr_backoff = self.start_backoff       
        while 1:
            try:
                start = time.time()
                completion = llmengine.Completion.create(model=self.model_name,
                                                         prompt=prompt,
                                        frequency_penalty=frequency_penalty,
                                        max_new_tokens=max_tokens,
                                        presence_penalty=presence_penalty,
                                        stop_sequences=stop,
                                        temperature=temperature,
                                        top_p=top_p,
                                        return_token_log_probs=True,
                                        )
                total = time.time() - start
                if total >= 120:
                    print(f"Warning: Request took {int(total)} seconds.")
                break
            except llmengine.errors.UnknownError as e:
                random_print(f"Unknown LLMEngine Error: {e}", p=PRINT_P)
            except llmengine.errors.BadRequestError as e:
                print(f"Bad Request Error (LLMEngine): {e}")
                print("^ Prompt:", prompt)
            except requests.exceptions.ReadTimeout as e:
                random_print(f"LLMEngineClient: Read Timeout Error: {e}", p=PRINT_P)
            except requests.exceptions.ConnectionError as e:
                random_print(f"LLMEngineClient: Connection Error: {e}", p=PRINT_P)
            except requests.exceptions.RequestException as e:
                random_print(f"LLMEngineClient: Other Request Exception: {e}", p=PRINT_P)

            curr_backoff = curr_backoff + random.random() * self.JITTER_FACTOR * curr_backoff
            random_print(f"LLMEngineClient: requerying in {curr_backoff} seconds.", p=PRINT_P)
            time.sleep(curr_backoff)
            curr_backoff = min(self.max_backoff, curr_backoff * self.BACKOFF_FACTOR)


        o = completion.output.text
        logprobs = [lp.log_prob for lp in completion.output.tokens]
        num_tok = len(logprobs)
        cumulative_logprob = logprobs_to_cumulative(logprobs)
        return Completion(o, cumulative_logprob, num_tok)

class TogetherClient(LLMClient):
    JITTER_FACTOR = 2/5
    BACKOFF_FACTOR = 1.5
    PARAMS = LLMClient.PARAMS + ["start_backoff", "max_backoff"]
    def __init__(self, model_name: str, is_chat: bool, is_batched: bool, batch_size: Optional[int] = None, num_workers: Optional[int] = None, price_per_input_output: tuple[float, float] = (0., 0.), start_backoff: float = 21., max_backoff: float = 1.5 * 60) -> None:
        super().__init__(model_name=model_name, is_chat=is_chat, is_batched=is_batched, batch_size=batch_size, num_workers=num_workers, price_per_input_output=price_per_input_output)
        self.start_backoff = start_backoff
        self.max_backoff = max_backoff

        api_key = os.getenv("TOGETHER_API_KEY")
        assert api_key is not None

        self.client = together.Together()
        self.is_loaded = True
    
    def generate(self, message: Prompt, frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], timeout: Optional[float] = None) -> Completion:
        if not self.is_chat:
            raise NotImplementedError("Together completion format not implemented.")

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
                    stop=stop,
                    temperature=temperature,
                    top_p=top_p,
                )
                total = time.time() - start

                if total >= 120:
                    print(f"Warning: Request took {int(total)} seconds.")
                break
            # except openai.BadRequestError as e:
            #     print(f"Bad request: {e}")
            #     print(message, "\n^ the bad prompt")
            #     return Completion("No output returned.", -1, 3)
            except together.error.RateLimitError:
                random_print("Together rate limit.", p=PRINT_P)
           
            curr_backoff = curr_backoff + random.random() * self.JITTER_FACTOR * curr_backoff
            random_print(f"TogetherClient: requerying in {curr_backoff} seconds.", p=PRINT_P)
            time.sleep(curr_backoff)
            curr_backoff = min(self.max_backoff, curr_backoff * self.BACKOFF_FACTOR)

        choice = response.choices[0]
        o = choice.message.content
        logprobs = choice.logprobs.token_logprobs  # type: ignore
        assert o is not None, "Together returned a null response"
        assert logprobs is not None, "Together returned a null logprobs"
        num_tokens = len(logprobs)
        if choice.finish_reason == "length":
            print("Warning, output clipped.")

        if any(l is None for l in logprobs):
            print("Warning: None in logprobs...")
            cumulative_logprob = -1
            print(f"^ output: {o}")
        else:
            cumulative_logprob = logprobs_to_cumulative(logprobs)

        return Completion(o, cumulative_logprob, num_tokens)

class FireworksClient(LLMClient):
    JITTER_FACTOR = 2/5
    BACKOFF_FACTOR = 1.5
    PARAMS = LLMClient.PARAMS + ["start_backoff", "max_backoff"]
    def __init__(self, model_name: str, is_chat: bool, is_batched: bool, batch_size: Optional[int] = None, num_workers: Optional[int] = None, price_per_input_output: tuple[float, float] = (0., 0.), start_backoff: float = 30., max_backoff: float = 1.5 * 60) -> None:
        super().__init__(model_name=model_name, is_chat=is_chat, is_batched=is_batched, batch_size=batch_size, num_workers=num_workers, price_per_input_output=price_per_input_output)
        self.start_backoff = start_backoff
        self.max_backoff = max_backoff

        api_key = os.getenv("FIREWORKS_API_KEY")
        assert api_key is not None
        assert api_key.endswith("veh")

        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.fireworks.ai/inference/v1")
        self.is_loaded = True
    
    def generate(self, message: Prompt, frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], timeout: Optional[float] = None) -> Completion:
        if not self.is_chat:
            raise NotImplementedError("Fireworks completion format not implemented.")

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
                    stop=stop,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=timeout,
                )
                total = time.time() - start

                if total >= 120:
                    print(f"Warning: Request took {int(total)} seconds.")
                break
            except openai.BadRequestError as e:
                print(f"Bad request: {e}")
                print(message, "\n^ the bad prompt")
                return Completion(self.BAD, -1, 3)
            except openai.RateLimitError:
                random_print("Fireworks rate limit.", p=PRINT_P)
            except openai.APITimeoutError:
                total = time.time() - start
                if timeout is not None:
                    print(f"Fireworks API exceeded timeout of {timeout}. ({total} seconds)")
                    return Completion(self.TIMEOUT_FLAG, -1, 0)
                random_print("Fireworks API timeout.", p=PRINT_P)
            except ReadError:
                random_print("httpcore ReadError.", p=PRINT_P)
            except openai.APIConnectionError:
                random_print("Fireworks API connection error.", p=PRINT_P)
            except openai.InternalServerError:
                random_print("Fireworks internal server error.", p=PRINT_P)
            except json.JSONDecodeError:
                random_print("(Fireworks) JSON decode error.", p=PRINT_P)
            except UnicodeDecodeError:
                random_print("(Fireworks) Unicode decode error.", p=PRINT_P)
            
            curr_backoff = curr_backoff + random.random() * self.JITTER_FACTOR * curr_backoff
            random_print(f"FireworksClient: requerying in {curr_backoff} seconds.", p=PRINT_P)
            time.sleep(curr_backoff)
            curr_backoff = min(self.max_backoff, curr_backoff * self.BACKOFF_FACTOR)

        choice = response.choices[0]
        o = choice.message.content
        logprobs = choice.logprobs.content  # type: ignore
        assert o is not None, "Fireworks returned a null response"
        assert logprobs is not None, "Fireworks returned a null logprobs"
        logprobs = [l.logprob for l in logprobs]
        num_tokens = len(logprobs)
        if choice.finish_reason == "length":
            print("Warning, output clipped.")

        cumulative_logprob = logprobs_to_cumulative(logprobs)
        return Completion(o, cumulative_logprob, num_tokens)


CLIENT_TYPE_TO_CLASS: dict[str, LLMClient] = {
    "OpenAI": OpenAIClient,
    "Anthropic": AnthropicClient,
    "vLLM": vLLMClient,
    "SGLang": SGLangClient,
    "DeepSeek": DeepSeekClient,
    "LLMEngine": LLMEngineClient,
    "Together": TogetherClient,
    "Fireworks": FireworksClient,
}


if __name__ == "__main__":
    # print(OpenAIClient.PARAMS)
    # lc = LLMClient.from_json("model_configs/llama31405bi.json")
    # lc = LLMClient.from_json("model_configs/llama31405bi_tog.json")
    lc = LLMClient.from_json("model_configs/llama31405bi_fire.json")
    print(lc.generate_completions([[{"role": "user", "content": "what is up my dude?"}]] * 100, frequency_penalty=None, logit_bias=None, max_tokens=1000, presence_penalty=None, seed=None, stop=None, temperature=0.5, top_p=1))
