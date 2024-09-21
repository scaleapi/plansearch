from abc import ABC, abstractmethod
from typing import List, Any, Optional, Union
import argparse
from pathlib import Path
import random

from search.base_classes import Problem, SearchModel
from coderm.prompts import Prompt
from search.parsing_utils import markdown_codeblock_extract
from search.model_config_utils import add_model_config_args, parse_args_for_model_client


class OnePromptModel(SearchModel):
    def __init__(self, model_config_path: str, experiment_directory: Optional[str] = None, cache_file: Optional[str] = None, querier_batch_size: Optional[int] = 12_288, use_cot: bool = False, use_sys_prompts: bool = True, num_shot: int = 2, frequency_penalty: Optional[float] = None, logit_bias: Optional[dict[str, int]] = None, max_tokens: Optional[int] = None, presence_penalty: Optional[float] = None, seed: Optional[int] = None, stop: Union[Optional[str], list[str]] = None, temperature: Optional[float] = None, top_p: Optional[float] = None, num_story_words: int = 0, num_random_words: int = 0, use_mbpp: bool = False):
        super().__init__(model_config_path, experiment_directory=experiment_directory, cache_file=cache_file, querier_batch_size=querier_batch_size)

        self.querier.add_client(model_config_path)
        self.model_config_path = model_config_path

        self.is_chat = self.querier.clients[model_config_path].is_chat
        self.use_cot = use_cot
        self.use_sys_prompts = use_sys_prompts
        self.num_shot = num_shot

        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.stop = stop
        self.temperature = temperature
        self.top_p = top_p

        self.num_story_words = num_story_words
        self.num_random_words = num_random_words

        self.use_mbpp = use_mbpp

        random.seed(seed)

    @abstractmethod
    def format_problem_to_prompt(self, problem: Problem) -> Prompt:
        pass

    def generate_solutions(self, problems: list[Problem], *args, **kwargs) -> list[list[str]]:
        problem_prompts = [self.format_problem_to_prompt(problem) for problem in problems]

        stop = [self.stop] if isinstance(self.stop, str) else self.stop
        if not self.is_chat:
            stop = [] if stop is None else stop
            stop += ["# START NEW CODE"]

        generated = self.querier.generate(self.model_config_path, 
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
                              timeout=1000,
                              )
        if self.is_chat:
            return [[markdown_codeblock_extract(genned).strip()] for genned in generated]
        else:
            return [[problem.starter_code.rstrip() + '\n' + genned] if problem.has_starter_code() else [genned] for genned, problem in zip(generated, problems)]


class BasicPromptModel(OnePromptModel):
    import search.prompts.simple_prompts as prompts
    def format_problem_to_prompt(self, problem: Problem) -> Prompt:
        assert self.num_story_words == 0
        assert self.num_random_words == 0
        if self.is_chat:
            convo = []
            if self.use_sys_prompts:
                if self.use_cot:
                    convo.append({"role": "system", "content": self.prompts.SYSTEM_PROMPT_COT})
                else:
                    convo.append({"role": "system", "content": self.prompts.SYSTEM_PROMPT})

            convo.append({"role": "user", "content": self.prompts.user_content_chat(problem.problem_str, problem.starter_code, use_cot=self.use_cot, num_shot=self.num_shot, use_mbpp=self.use_mbpp)})
            return convo
        else:
            out_str = self.prompts.user_content_completion(problem.problem_str, problem.starter_code, use_cot=self.use_cot, num_shot=self.num_shot)
            return out_str


class StoryModel(OnePromptModel):
    import search.prompts.simple_prompts as prompts
    def format_problem_to_prompt(self, problem: Problem) -> Prompt:
        assert self.num_story_words > 0
        assert self.num_random_words == 0
        assert self.is_chat
        assert not self.use_cot

        convo = []
        if self.use_sys_prompts:
            convo.append({"role": "system", "content": self.prompts.STORY_SYSTEM_PROMPT})
        convo.append({"role": "user", "content": self.prompts.user_content_chat(problem.problem_str, problem.starter_code, use_cot=self.use_cot, num_shot=self.num_shot, num_story_words=self.num_story_words, use_mbpp=self.use_mbpp)})
        return convo

class RandomWordModel(OnePromptModel):
    import search.prompts.simple_prompts as prompts
    words = None
    def format_problem_to_prompt(self, problem: Problem) -> Prompt:
        assert self.num_story_words == 0
        assert self.num_random_words > 0
        assert self.is_chat
        assert not self.use_cot

        if self.words is None:
            with open("assets/words.txt", "r") as f:
                self.words = [line.strip() for line in f.readlines()]

        convo = []
        if self.use_sys_prompts:
            convo.append({"role": "system", "content": self.prompts.RANDOM_WORD_SYSTEM_PROMPT})

        random_words = random.sample(self.words, self.num_random_words)
        convo.append({"role": "user", "content": self.prompts.user_content_chat(problem.problem_str, problem.starter_code, use_cot=self.use_cot, num_shot=self.num_shot, random_words=random_words, use_mbpp=self.use_mbpp)})
        return convo


def _add_one_prompt_model_args(parser:argparse.ArgumentParser):
    add_model_config_args(parser, model_config_name="model")
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
        "--no-sys-prompt",
        action="store_true",
        help="Whether to cancel sys prompts"
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=2,
        help="Number of shots to pre-prompt model",
    )
    parser.add_argument(
        "--use-mbpp",
        action="store_true",
        help="Number of shots to pre-prompt model",
    )

def add_basic_prompting_args(parser: argparse.ArgumentParser):
    _add_one_prompt_model_args(parser)
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Whether to use chain of thought"
    )

def add_story_args(parser: argparse.ArgumentParser):
    _add_one_prompt_model_args(parser)
    parser.add_argument(
        "--num-story-words",
        type=int,
        default=500,
        help="Roughly how many words to use in the story."
    )

def add_random_word_args(parser: argparse.ArgumentParser):
    _add_one_prompt_model_args(parser)
    parser.add_argument(
        "--num-random-words",
        type=int,
        default=3,
        help="How many random words to use."
    )

def get_basic_prompting_model(args: argparse.Namespace) -> SearchModel:
    model_path = parse_args_for_model_client(args, model_config_name="model", temp_folder_base=args.experiment_directory)
    return BasicPromptModel(model_config_path=model_path, experiment_directory=args.experiment_directory, cache_file=args.cache_file, querier_batch_size=args.global_batch_size, use_cot=args.cot, use_sys_prompts=not args.no_sys_prompt, num_shot=args.num_shots, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, use_mbpp=args.use_mbpp)

def get_story_model(args: argparse.Namespace) -> SearchModel:
    model_path = parse_args_for_model_client(args, model_config_name="model", temp_folder_base=args.experiment_directory)
    return StoryModel(model_config_path=model_path, experiment_directory=args.experiment_directory, cache_file=args.cache_file, querier_batch_size=args.global_batch_size, use_sys_prompts=not args.no_sys_prompt, num_shot=args.num_shots, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, num_story_words=args.num_story_words, use_mbpp=args.use_mbpp)

def get_random_word_model(args: argparse.Namespace) -> SearchModel:
    model_path = parse_args_for_model_client(args, model_config_name="model", temp_folder_base=args.experiment_directory)
    return RandomWordModel(model_config_path=model_path, experiment_directory=args.experiment_directory, cache_file=args.cache_file, querier_batch_size=args.global_batch_size, use_sys_prompts=not args.no_sys_prompt, num_shot=args.num_shots, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, num_random_words=args.num_random_words, use_mbpp=args.use_mbpp)
