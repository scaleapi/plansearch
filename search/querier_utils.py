import anthropic
import openai
from vllm import SamplingParams, LLM
import tiktoken
from transformers import AutoTokenizer
from llmengine import Completion as LLMEngineCompletion
from llmengine.errors import UnknownError, BadRequestError

import requests
from typing import Optional, Union, Any
import json
import time
from httpcore import ReadError
import random

from coderm.prompts import Prompt
from coderm.model import Completion, logprobs_to_cumulative
from python_utils import random_print


MAX_TIMEOUT = 4 * 60 + 1e-4
START_TIMEOUT = 30
JITTER_FACTOR = 3/4
PRINT_P = 1e-2


def num_tokens_for_convo(convo: Union[str, list[dict[str, str]]], is_chat: bool, encoding: tiktoken.Encoding = tiktoken.get_encoding("cl100k_base")) -> int:
    num_tokens = 0
    if is_chat:
        for turn in convo:
            num_tokens += 1 + len(encoding.encode(turn["content"]))
    else:
        num_tokens = len(encoding.encode(convo))
    return num_tokens

def generate_anthropic_completion(client: anthropic.Anthropic, model: str, prompt: list[dict[str, str]], max_tokens: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], **kwargs) -> Completion:
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
        
        curr_backoff = min(MAX_TIMEOUT, curr_backoff * 2)
        random_print(f"Requerying in {curr_backoff} seconds.", p=PRINT_P)
        time.sleep(curr_backoff)
    
    o = response.content[0].text
    assert o is not None, "Anthropic returned a null response"
    num_tokens = response.usage.output_tokens
    if response.stop_reason == "max_tokens":
        print("Warning, output clipped.")

    return Completion(o, -1, num_tokens)

def generate_openai_completion(client: openai.OpenAI, model: str, prompt: str, frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], **kwargs) -> Completion:
    curr_backoff = START_TIMEOUT
    while 1:
        try:
            response = client.completions.create(
                model=model,
                prompt=prompt,
                frequency_penalty=frequency_penalty,
                logit_bias={} if logit_bias is None else logit_bias,
                logprobs=1,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                seed=seed,
                stop=stop,
                temperature=temperature,
                top_p=top_p,
            )
            break
        except openai.RateLimitError:
            random_print("OpenAI rate limit.", p=PRINT_P)
        except openai.APITimeoutError:
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
        
        curr_backoff = min(MAX_TIMEOUT, curr_backoff * 2 + random.random() * JITTER_FACTOR * curr_backoff)
        random_print(f"Requerying in {curr_backoff} seconds.", p=PRINT_P)
        time.sleep(curr_backoff)
    
    choice = response.choices[0]
    o = choice.text
    logprobs = choice.logprobs.token_logprobs  # type: ignore
    assert o is not None, "OpenAI returned a null response"
    assert logprobs is not None, "OpenAI returned a null logprobs"
    num_tokens = len(logprobs)
    if choice.finish_reason == "length":
        print("Warning, output clipped.")

    cumulative_logprob = logprobs_to_cumulative(logprobs)
    return Completion(o, cumulative_logprob, num_tokens)


def generate_openai_chat_completion(client: openai.OpenAI, model: str, prompt: list[dict[str, str]], frequency_penalty: Optional[float], logit_bias: Optional[dict[str, int]], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], **kwargs) -> Completion:
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
        except openai.BadRequestError as e:
            print(f"Bad request: {e}")
            print(prompt, "\n^ the bad prompt")
            return Completion("No output returned.", -1, 3)
        except openai.RateLimitError:
            random_print("OpenAI rate limit.", p=PRINT_P)
        except openai.APITimeoutError:
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
        
        curr_backoff = min(MAX_TIMEOUT, curr_backoff * 2 + random.random() * JITTER_FACTOR * curr_backoff)
        random_print(f"Requerying in {curr_backoff} seconds.", p=PRINT_P)
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

def generate_vllm_chat_completions(client: LLM, model: str, prompts: list[list[dict[str, str]]], frequency_penalty: Optional[float], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], **kwargs) -> Completion:
    tokenizer = client.get_tokenizer()
    chat_msgs_as_str = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
    return generate_vllm_completions(client, model, chat_msgs_as_str, frequency_penalty=frequency_penalty, max_tokens=max_tokens, presence_penalty=presence_penalty, seed=seed, stop=stop, temperature=temperature, top_p=top_p)

def generate_vllm_completions(client: LLM, model: str, prompts: list[str], frequency_penalty: Optional[float], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], **kwargs) -> list[Completion]:
    sampling_params = SamplingParams(
        n=1,
        frequency_penalty=0 if frequency_penalty is None else frequency_penalty,
        presence_penalty=0 if presence_penalty is None else presence_penalty,
        seed=seed,
        max_tokens=max_tokens,
        stop=stop,
        temperature=temperature,
        top_p=top_p,
    )
    outputs = client.generate(prompts, sampling_params=sampling_params)
    return [Completion(output.outputs[0].text, output.outputs[0].cumulative_logprob, len(output.outputs[0].token_ids)) for output in outputs]

def generate_internal_completions(client: AutoTokenizer, model: str, prompt: list[list[dict[str, str]]], frequency_penalty: Optional[float], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], **kwargs) -> Completion:
    tokenizer = client
    chat_msgs_as_str = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    if stop is None:
        stop = []
    assert isinstance(stop, list)

    response = requests.post(
        "http://10.192.73.243:5005/predict",
        data=json.dumps({
            "prompt": chat_msgs_as_str,
            "frequency_penalty": 0 if frequency_penalty is None else frequency_penalty,
            "max_tokens": max_tokens,
            "presence_penalty": 0 if presence_penalty is None else presence_penalty,
            "seed": seed,
            "stop": stop + ["<|eot_id|>"],
            "temperature": temperature,
            "top_p": top_p
        })
    )

    output_dict = json.loads(response.content)
    return Completion(output_dict["text"], -1, output_dict["count_output_tokens"])

def generate_llm_engine_chat_completion(client: AutoTokenizer, model: str, prompt: list[dict[str, str]], frequency_penalty: Optional[float], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], **kwargs) -> Completion:
    tokenizer = client
    chat_msgs_as_str = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return generate_llm_engine_completion(client, model, chat_msgs_as_str, frequency_penalty=frequency_penalty, max_tokens=max_tokens, presence_penalty=presence_penalty, seed=seed, stop=stop, temperature=temperature, top_p=top_p)

def generate_llm_engine_completion(client: Any, model: str, prompt: str, frequency_penalty: Optional[float], max_tokens: Optional[int], presence_penalty: Optional[float], seed: Optional[int], stop: Union[Optional[str], list[str]], temperature: Optional[float], top_p: Optional[float], **kwargs) -> Completion:
    if seed is not None:
        print("Warning: seed is not used")
    curr_backoff = START_TIMEOUT
    while 1:
        try:
            completion = LLMEngineCompletion.create(model=model, prompt=prompt,
                                    frequency_penalty=frequency_penalty,
                                    max_new_tokens=max_tokens,
                                    presence_penalty=presence_penalty,
                                    stop_sequences=stop,
                                    temperature=temperature,
                                    top_p=top_p,
                                    return_token_log_probs=True,
                                    )
            break
        except UnknownError as e:
            print(f"Unknown LLMEngine Error: {e}")
        except BadRequestError as e:
            print(f"Bad Request Error (LLMEngine): {e}")
            print("^ Prompt:", prompt)
        except requests.exceptions.ReadTimeout as e:
            print(f"Read Timeout Error: {e}")
        except requests.exceptions.ConnectionError as e:
            print(f"Connection Error: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Other Request Exception: {e}")

        curr_backoff = min(MAX_TIMEOUT, curr_backoff * 1.75 + random.random() * JITTER_FACTOR * curr_backoff)
        print(f"Requerying in {curr_backoff} seconds.")
        time.sleep(curr_backoff)

    o = completion.output.text
    logprobs = [lp.log_prob for lp in completion.output.tokens]
    num_tok = len(logprobs)
    cumulative_logprob = logprobs_to_cumulative(logprobs)
    return Completion(o, cumulative_logprob, num_tok)
