from typing import List, Any, Optional, Union
import argparse
import os

from search.base_classes import Problem, SearchModel
from search.parsing_utils import markdown_codeblock_extract
from search.model_config_utils import add_model_config_args, parse_args_for_model_client


class BackTranslateModel(SearchModel):
    import search.prompts.backtranslate_prompts as prompts
    def __init__(self, model_config_path: str, experiment_directory: Optional[str] = None, cache_file: Optional[str] = None, querier_batch_size: Optional[int] = 12_288, frequency_penalty: Optional[float] = None, logit_bias: Optional[dict[str, int]] = None, max_tokens: Optional[int] = None, presence_penalty: Optional[float] = None, seed: Optional[int] = None, stop: Union[Optional[str], list[str]] = None, temperature: Optional[float] = None, top_p: Optional[float] = None, num_words: Optional[int] = None):
        super().__init__(model_config_path, experiment_directory=experiment_directory, cache_file=cache_file, querier_batch_size=querier_batch_size)

        self.model_config_path = model_config_path
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.stop = stop
        self.temperature = temperature
        self.top_p = top_p
        self.num_words = num_words

    def solution_to_nl_prompt(self, problem: Problem) -> list[dict[str, str]]:
        assert problem.solutions is not None and len(problem.solutions)
        convo = [{"role": "system", "content": self.prompts.SYSTEM_PROMPT_TRANSLATE},
                 {"role": "user", "content": self.prompts.translate_code_solution(problem.problem_str, problem.solutions[0], problem.has_starter_code(), num_words=self.num_words)}]
        return convo
    
    def nl_to_actual_solution_prompt(self, problem: Problem, nl_solution: str) -> list[dict[str, str]]:
        convo = [{"role": "system", "content": self.prompts.SYSTEM_PROMPT_GENERATE},
                 {"role": "user", "content": self.prompts.generate_code_sol(problem.problem_str, nl_solution, problem.starter_code)}]
        return convo

    def generate_solutions(self, problems: list[Problem], *args, **kwargs) -> list[str]:
        solution_to_nl_prompts = [self.solution_to_nl_prompt(problem) for problem in problems]
        nl_solutions = self.querier.generate(self.model_config_path, 
                              solution_to_nl_prompts,
                              frequency_penalty=self.frequency_penalty,
                              logit_bias=self.logit_bias,
                              max_tokens=self.max_tokens,
                              presence_penalty=self.presence_penalty,
                              seed=self.seed,
                              stop=self.stop,
                              temperature=self.temperature,
                              top_p=self.top_p,
                              log_name="solutions",
                              requery=True,
                              )
        nl_to_acc_sol_prompts = [self.nl_to_actual_solution_prompt(problem, nl_sol) for problem, nl_sol in zip(problems, nl_solutions)]
        generated = self.querier.generate(self.model_config_path, 
                              nl_to_acc_sol_prompts,
                              frequency_penalty=self.frequency_penalty,
                              logit_bias=self.logit_bias,
                              max_tokens=self.max_tokens,
                              presence_penalty=self.presence_penalty,
                              seed=self.seed,
                              stop=self.stop,
                              temperature=self.temperature,
                              top_p=self.top_p,
                              log_name="codes",
                              requery=True,
                              )
        return [[markdown_codeblock_extract(genned).strip()] for genned in generated]


def add_backtranslate_args(parser: argparse.ArgumentParser):
    add_model_config_args(parser, "model")
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
        "--num-words",
        type=int,
        default=None,
        help="Approximate number of words to generate in backtranslation"
    )

def get_backtranslate_model(args: argparse.Namespace) -> SearchModel:
    model_path = parse_args_for_model_client(args, model_config_name="model", temp_folder_base=args.experiment_directory)
    return BackTranslateModel(model_path, args.experiment_directory, cache_file=args.cache_file, querier_batch_size=args.global_batch_size, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, num_words=args.num_words)
