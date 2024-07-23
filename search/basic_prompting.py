from abc import ABC, abstractmethod
from typing import List, Any, Optional, Union
import argparse
import os
import datetime
from pathlib import Path
import json
import shutil

from base_classes import Problem, SearchModel
from coderm.prompts import Prompt
from queriers import MODEL_NAME_TO_CLIENT_STR, is_chat
from parsing_utils import markdown_codeblock_extract


class BasicPromptingModel(SearchModel):
    def __init__(self, model_name: str, experiment_directory: Optional[str] = None, cache_file: Optional[str] = None, use_cot: bool = False, use_sys_prompts: bool = True, use_few_shot: bool = True, frequency_penalty: Optional[float] = None, logit_bias: Optional[dict[str, int]] = None, max_tokens: Optional[int] = None, presence_penalty: Optional[float] = None, seed: Optional[int] = None, stop: Union[Optional[str], list[str]] = None, temperature: Optional[float] = None, top_p: Optional[float] = None):
        super().__init__(model_name, experiment_directory=experiment_directory, cache_file=cache_file)

        self.is_chat = is_chat(model_name)
        self.use_cot = use_cot
        self.use_sys_prompts = use_sys_prompts
        self.use_few_shot = use_few_shot

        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.stop = stop
        self.temperature = temperature
        self.top_p = top_p

    @abstractmethod
    def format_problem_to_prompt(self, problem: Problem) -> Prompt:
        pass

    def generate_solutions(self, problems: list[Problem], *args, **kwargs) -> list[str]:
        problem_prompts = [self.format_problem_to_prompt(problem) for problem in problems]

        stop = [self.stop] if isinstance(self.stop, str) else self.stop
        if not self.is_chat:
            stop = [] if stop is None else stop
            stop += ["# START NEW CODE"]

        generated = self.querier.generate(self.model_name, 
                              problem_prompts,
                              frequency_penalty=self.frequency_penalty,
                              logit_bias=self.logit_bias,
                              max_tokens=self.max_tokens,
                              presence_penalty=self.presence_penalty,
                              seed=self.seed,
                              stop=stop,
                              temperature=self.temperature,
                              top_p=self.top_p,
                              log_name="solution_queries",
                              requery=True,
                              )
        if self.is_chat:
            return [markdown_codeblock_extract(genned).strip() for genned in generated]
        else:
            return [problem.starter_code.rstrip() + '\n' + genned if problem.has_starter_code() else genned for genned, problem in zip(generated, problems)]


class SimplePromptModel(BasicPromptingModel):
    import prompts.simple_prompts as prompts
    def format_problem_to_prompt(self, problem: Problem) -> Prompt:
        if self.is_chat:
            convo = []
            if self.use_sys_prompts:
                if self.use_cot:
                    convo.append({"role": "system", "content": self.prompts.SYSTEM_PROMPT_COT})
                else:
                    convo.append({"role": "system", "content": self.prompts.SYSTEM_PROMPT})

            convo.append({"role": "user", "content": self.prompts.user_content_chat(problem.problem_str, problem.starter_code, use_cot=self.use_cot, use_few_shot=self.use_few_shot)})
            return convo
        else:
            out_str = self.prompts.user_content_completion(problem.problem_str, problem.starter_code, use_cot=self.use_cot, use_few_shot=self.use_few_shot)
            return out_str


def add_basic_prompting_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model",
        required=True,
        help="Model to use"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.,
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling"
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Whether to use chain of thought"
    )
    parser.add_argument(
        "--no-sys-prompt",
        action="store_true",
        help="Whether to cancel sys prompts"
    )
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="Whether to do zero shot instead"
    )
    
def get_basic_prompting_model(args: argparse.Namespace) -> SearchModel:
    return SimplePromptModel(args.model, experiment_directory=args.experiment_directory, cache_file=args.cache_file, use_cot=args.cot, use_sys_prompts=not args.no_sys_prompt, use_few_shot=not args.zero_shot, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
