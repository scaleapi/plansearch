from typing import List, Any, Optional, Union
import argparse
import os

from base_classes import Problem, SearchModel
from queriers import LLMQuerier, MODEL_NAME_TO_CLIENT_STR
from parsing_utils import markdown_codeblock_extract


class SimpleIdeaModel(SearchModel):
    import prompts.idea_prompts as prompts
    def __init__(self, idea_model: str, code_model: str, experiment_directory: Optional[str] = None, cache_file: Optional[str] = None, frequency_penalty: Optional[float] = None, logit_bias: Optional[dict[str, int]] = None, max_tokens: Optional[int] = None, presence_penalty: Optional[float] = None, seed: Optional[int] = None, stop: Union[Optional[str], list[str]] = None, idea_temperature: Optional[float] = None, code_temperature: Optional[float] = None, top_p: Optional[float] = None):
        super().__init__("simple_idea", experiment_directory=experiment_directory, cache_file=cache_file)

        self.idea_model = idea_model
        self.code_model = code_model

        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.seed = seed
        self.stop = stop
        self.code_temperature = code_temperature
        self.idea_temperature = idea_temperature
        self.top_p = top_p

    def get_nl_sols_prompt(self, problem: Problem) -> list[dict[str, str]]:
        convo = [{"role": "system", "content": self.prompts.SYSTEM_PROMPT_TRANSLATE},
                 {"role": "user", "content": self.prompts.get_nl_solution(problem.problem_str, problem.has_starter_code())}]
        return convo
    
    def nl_to_code_solution_prompt(self, problem: Problem, nl_solution: str) -> list[dict[str, str]]:
        convo = [{"role": "system", "content": self.prompts.SYSTEM_PROMPT_GENERATE},
                 {"role": "user", "content": self.prompts.generate_code_sol(problem.problem_str, nl_solution, problem.starter_code)}]
        return convo

    def generate_solutions(self, problems: list[Problem], *args, **kwargs) -> list[str]:
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
                              requery=kwargs.get("requery", False),
                              )
        nl_to_acc_sol_prompts = [self.nl_to_code_solution_prompt(problem, nl_sol) for problem, nl_sol in zip(problems, nl_solutions)]
        generated = self.querier.generate(self.code_model, 
                              nl_to_acc_sol_prompts,
                              frequency_penalty=self.frequency_penalty,
                              logit_bias=self.logit_bias,
                              max_tokens=self.max_tokens,
                              presence_penalty=self.presence_penalty,
                              seed=self.seed,
                              stop=self.stop,
                              temperature=self.code_temperature,
                              top_p=self.top_p,
                              requery=kwargs.get("requery", False),
                              )
        return [markdown_codeblock_extract(genned).strip() for genned in generated]


def add_simple_idea_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--idea-model",
        choices=MODEL_NAME_TO_CLIENT_STR.keys(),
        required=True,
        help="Model to use for ideas"
    )
    parser.add_argument(
        "--code-model",
        choices=MODEL_NAME_TO_CLIENT_STR.keys(),
        required=True,
        help="Model to use for implementation"
    )
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

def get_simple_idea_model(args: argparse.Namespace) -> SearchModel:
    return SimpleIdeaModel(args.idea_model, args.code_model, args.experiment_directory, cache_file=args.cache_file, idea_temperature=args.idea_temperature, code_temperature=args.code_temperature, top_p=args.top_p, max_tokens=args.max_tokens)
