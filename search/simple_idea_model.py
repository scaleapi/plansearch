from typing import List, Any, Optional, Union
import argparse
import os
import datetime

from search.base_classes import Problem, SearchModel
from search.parsing_utils import markdown_codeblock_extract
from search.python_utils import log_to_dir
from search.model_config_utils import add_model_config_args, parse_args_for_model_client


class SimpleIdeaModel(SearchModel):
    import search.prompts.idea_prompts as prompts
    def __init__(self, idea_model_config_path: str, code_model_config_path: str, experiment_directory: Optional[str] = None, cache_file: Optional[str] = None, querier_batch_size: Optional[int] = 12_288, frequency_penalty: Optional[float] = None, logit_bias: Optional[dict[str, int]] = None, max_tokens: Optional[int] = None, presence_penalty: Optional[float] = None, seed: Optional[int] = None, stop: Union[Optional[str], list[str]] = None, idea_temperature: Optional[float] = None, code_temperature: Optional[float] = None, top_p: Optional[float] = None, use_few_shot: bool = True, use_sys_prompt: bool = True, num_words: Optional[int] = None, num_codes_per_idea: int = 1):
        super().__init__("simple_idea", experiment_directory=experiment_directory, cache_file=cache_file, querier_batch_size=querier_batch_size)

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
        self.use_sys_prompt = use_sys_prompt
        self.num_words = num_words
        self.num_codes_per_idea = num_codes_per_idea
        
        if self.num_codes_per_idea > 1:
            print("Warning: Simple Idea currently returns the first code per idea, so there is no relative benefit of setting num_codes_per_idea > 1.")

    def get_nl_sols_prompt(self, problem: Problem) -> list[dict[str, str]]:
        convo = []
        if self.use_sys_prompt:
            convo.append({"role": "system", "content": self.prompts.SYSTEM_PROMPT_TRANSLATE})
        convo.append({"role": "user", "content": self.prompts.get_nl_solution(problem.problem_str, problem.has_starter_code(), use_few_shot=self.use_few_shot, num_words=self.num_words)})
        return convo
    
    def nl_to_code_solution_prompt(self, problem: Problem, nl_solution: str) -> list[dict[str, str]]:
        convo = [{"role": "system", "content": self.prompts.SYSTEM_PROMPT_GENERATE},
                 {"role": "user", "content": self.prompts.generate_code_sol(problem.problem_str, nl_solution, problem.starter_code)}]
        return convo
    
    def select_code(self, problem: Problem, codes: list[str]) -> str:
        assert len(codes) == self.num_codes_per_idea
        if len(codes) == 1:
            return codes[0]
        # Later do more intelligent selection
        return codes[0]

    def generate_solutions(self, problems: list[Problem], *args, **kwargs) -> list[str]:
        log_dict = []

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

        nl_to_acc_sol_prompts = [self.nl_to_code_solution_prompt(problem, nl_sol) for problem, nl_sol in zip(problems, nl_solutions)] * self.num_codes_per_idea
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
                              requery=True,
                              )
        
        assert len(generated) == len(problems) * self.num_codes_per_idea

        extracted_codes = [markdown_codeblock_extract(genned).strip() for genned in generated]

        codes_per_problem = [[] for _ in range(len(problems))]
        for i, code in enumerate(extracted_codes):
            codes_per_problem[i % len(problems)].append(code)
        
        output_codes = [self.select_code(problem, codes) for problem, codes in zip(problems, codes_per_problem)]

        # Logging
        log_dict = [
            {"question": problem.problem_str, "nl_solution": nl_solution, "codes": codes, "selected": selected_code}
            for problem, nl_solution, codes, selected_code in zip(problems, nl_solutions, codes_per_problem, output_codes)
        ]
        log_to_dir(self.experiment_directory, {f"log_{datetime.datetime.now().strftime('%m-%dT%H:%M:%S')}.json": log_dict})

        return output_codes


def add_simple_idea_args(parser: argparse.ArgumentParser):
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
    parser.add_argument(
        "--no-sys-prompt",
        action="store_true",
        help="Whether to include system prompts"
    )
    parser.add_argument(
        "--num-words",
        type=int,
        default=None,
        help="Rough number of words to use in the idea"
    )
    parser.add_argument(
        "--num-codes-per-idea",
        type=int,
        default=1,
        help="Number of code samples to produce per idea"
    )

def get_simple_idea_model(args: argparse.Namespace) -> SearchModel:
    idea_model_path = parse_args_for_model_client(args, model_config_name="idea_model", temp_folder_base=args.experiment_directory)
    code_model_path = parse_args_for_model_client(args, model_config_name="code_model", temp_folder_base=args.experiment_directory)
    return SimpleIdeaModel(idea_model_path, code_model_path, args.experiment_directory, cache_file=args.cache_file, querier_batch_size=args.global_batch_size, idea_temperature=args.idea_temperature, code_temperature=args.code_temperature, top_p=args.top_p, max_tokens=args.max_tokens, use_few_shot=not args.zero_shot, use_sys_prompt=not args.no_sys_prompt, num_words=args.num_words, num_codes_per_idea=args.num_codes_per_idea)
