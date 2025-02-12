from typing import List, Any, Optional, Union
import argparse
import os

from search.base_classes import Problem, SearchModel
from search.parsing_utils import markdown_codeblock_extract
from search.model_config_utils import add_model_config_args, parse_args_for_model_client


class PseudocodeModel(SearchModel):
    import search.prompts.pseudocode_prompts as prompts
    def __init__(self, idea_model_config_path: str, code_model_config_path: str, experiment_directory: Optional[str] = None, cache_file: Optional[str] = None, querier_batch_size: Optional[int] = 12_288, frequency_penalty: Optional[float] = None, logit_bias: Optional[dict[str, int]] = None, max_tokens: Optional[int] = None, presence_penalty: Optional[float] = None, seed: Optional[int] = None, stop: Union[Optional[str], list[str]] = None, idea_temperature: Optional[float] = None, code_temperature: Optional[float] = None, top_p: Optional[float] = None, use_few_shot: bool = True):
        super().__init__("pseudocode", experiment_directory=experiment_directory, cache_file=cache_file, querier_batch_size=querier_batch_size)

        self.idea_model = idea_model_config_path
        self.code_model = code_model_config_path

        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.stop = stop
        self.code_temperature = code_temperature
        self.idea_temperature = idea_temperature
        self.top_p = top_p
        self.use_few_shot = use_few_shot

    def get_nl_sols_prompt(self, problem: Problem) -> list[dict[str, str]]:
        convo = [{"role": "system", "content": self.prompts.SYSTEM_PROMPT_TRANSLATE},
                 {"role": "user", "content": self.prompts.get_nl_solution(problem.problem_str, problem.has_starter_code(), self.use_few_shot)}]
        return convo

    def generate_solutions(self, problems: list[Problem], *args, **kwargs) -> list[list[str]]:
        get_nl_sols_prompt = [self.get_nl_sols_prompt(problem) for problem in problems]
        nl_solutions = self.querier.generate(self.idea_model, 
                              get_nl_sols_prompt,
                              frequency_penalty=self.frequency_penalty,
                              logit_bias=self.logit_bias,
                              max_tokens=self.max_tokens,
                              presence_penalty=self.presence_penalty,
                              seed=self.seed,
                              stop=self.stop,
                              temperature=self.idea_temperature,
                              top_p=self.top_p,
                              requery=True,
                              )
        
        get_pseudocode_prompt = [self.prompts.get_pseudocode_prompt(problem.problem_str, nl_solution) for problem, nl_solution in zip(problems, nl_solutions)]
        pseudocodes = self.querier.generate(self.code_model, 
                              get_pseudocode_prompt,
                              frequency_penalty=self.frequency_penalty,
                              logit_bias=self.logit_bias,
                              max_tokens=self.max_tokens,
                              presence_penalty=self.presence_penalty,
                              seed=self.seed,
                              stop=self.stop,
                              temperature=self.code_temperature,
                              top_p=self.top_p,
                              requery=True,
                              )

        pseudocode_to_sol_prompts = [self.prompts.pseudocode_to_code_solution_prompt(problem.problem_str, problem.starter_code, pseudocode) for problem, pseudocode in zip(problems, pseudocodes)]
        generated = self.querier.generate(self.code_model, 
                              pseudocode_to_sol_prompts,
                              frequency_penalty=self.frequency_penalty,
                              logit_bias=self.logit_bias,
                              max_tokens=self.max_tokens,
                              presence_penalty=self.presence_penalty,
                              seed=self.seed,
                              stop=self.stop,
                              temperature=self.code_temperature,
                              top_p=self.top_p,
                              requery=True,
                              )
        return [[markdown_codeblock_extract(genned).strip()] for genned in generated]


def add_pseudocode_args(parser: argparse.ArgumentParser):
    add_model_config_args(parser, "idea_model")
    add_model_config_args(parser, "code_model")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max new tokens to generate"
    )
    parser.add_argument(
        "--idea-temperature",
        type=float,
        default=0.,
        help="Temperature for sampling ideas"
    )
    parser.add_argument(
        "--code-temperature",
        type=float,
        default=0.,
        help="Temperature for sampling codes"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling"
    )
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="Whether to do zero shot instead"
    )

def get_pseudocode_model(args: argparse.Namespace) -> SearchModel:
    idea_model_path = parse_args_for_model_client(args, model_config_name="idea_model", temp_folder_base=args.experiment_directory)
    code_model_path = parse_args_for_model_client(args, model_config_name="code_model", temp_folder_base=args.experiment_directory)
    return PseudocodeModel(idea_model_path, code_model_path, args.experiment_directory, cache_file=args.cache_file, querier_batch_size=args.global_batch_size, idea_temperature=args.idea_temperature, code_temperature=args.code_temperature, top_p=args.top_p, max_tokens=args.max_tokens, use_few_shot=not args.zero_shot)
