from typing import List, Any, Optional, Union
import argparse
import os
import datetime

from search.base_classes import Problem, SearchModel
from search.parsing_utils import markdown_codeblock_extract
from search.python_utils import log_to_dir, batch_map_on_nested_list, map_nary_fn_on_nested_list
from search.model_config_utils import add_model_config_args, parse_args_for_model_client


class SimpleIdeaModel(SearchModel):
    import search.prompts.idea_prompts as prompts
    COMPLETION_FROM_MODEL_SUPPORTED = True
    def __init__(self, idea_model_config_path: str, code_model_config_path: str, experiment_directory: Optional[str] = None, cache_file: Optional[str] = None, querier_batch_size: Optional[int] = 12_288, frequency_penalty: Optional[float] = None, logit_bias: Optional[dict[str, int]] = None, max_tokens: Optional[int] = None, presence_penalty: Optional[float] = None, seed: Optional[int] = None, stop: Union[Optional[str], list[str]] = None, idea_temperature: Optional[float] = None, code_temperature: Optional[float] = None, top_p: Optional[float] = None, use_few_shot: bool = True, use_sys_prompt: bool = True, num_words: Optional[int] = None, num_codes_per_idea: int = 1, add_criticism: bool = False):
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
        self.add_criticism = add_criticism
        
        if self.num_codes_per_idea > 1:
            print("Warning: Simple Idea currently returns the first code per idea, so there is no relative benefit of setting num_codes_per_idea > 1.")

    def query_model(self, model_type: str, prompts: list[list[dict[str, str]]], temperature: Optional[float] = None) -> list[str]:
        assert model_type in {"idea", "code"}
        if model_type == "idea":
            model = self.idea_model
            use_temperature = self.idea_temperature
        else:
            model = self.code_model
            use_temperature = self.code_temperature
        if temperature is not None:
            use_temperature = temperature

        outputs = self.querier.generate(model, 
                              prompts,
                              frequency_penalty=self.frequency_penalty,
                              logit_bias=self.logit_bias,
                              max_tokens=self.max_tokens,
                              presence_penalty=self.presence_penalty,
                              seed=self.seed,
                              stop=self.stop,
                              temperature=use_temperature,
                              top_p=self.top_p,
                              requery=True,
                              timeout=150,
                              )
        assert len(outputs) == len(prompts)
        return outputs

    def get_nl_sols_prompt(self, problem: Problem) -> tuple[dict[str, str]]:
        convo = []
        if self.use_sys_prompt:
            convo.append({"role": "system", "content": self.prompts.SYSTEM_PROMPT_TRANSLATE})
        convo.append({"role": "user", "content": self.prompts.get_nl_solution(problem.problem_str, problem.has_starter_code(), use_few_shot=self.use_few_shot, num_words=self.num_words)})
        return tuple(convo)
    
    def nl_to_code_solution_prompt(self, problem: Problem, nl_solution: str) -> tuple[dict[str, str]]:
        convo = [{"role": "system", "content": self.prompts.SYSTEM_PROMPT_GENERATE},
                 {"role": "user", "content": self.prompts.generate_code_sol(problem.problem_str, nl_solution, problem.starter_code)}]
        return tuple(convo)
    
    def select_code(self, problem: Problem, codes: list[str]) -> str:
        assert len(codes) == self.num_codes_per_idea
        if len(codes) == 1:
            return codes[0]
        # Later do more intelligent selection
        return codes[0]

    def generate_solutions(self, problems: list[Problem], *args, **kwargs) -> list[list[str]]:
        num_completions = kwargs.get("num_completions", 1)
        assert num_completions > 0
        if self.add_criticism:
            assert num_completions % 2 == 0
            num_completions = num_completions // 2

        get_nl_sols_prompt = [[self.get_nl_sols_prompt(problem)] * num_completions for problem in problems]
        nl_solutions: list[list[str]] = batch_map_on_nested_list(get_nl_sols_prompt, lambda li: self.query_model("idea", li))
        
        log_dict = [
            [{"problem_str": problem.problem_str, "nl_solution": nl_solution} for nl_solution in nl_solutions_for_problem]
            for problem, nl_solutions_for_problem in zip(problems, nl_solutions)
        ]

        if self.add_criticism:
            get_criticism_prompts = [[
                                    self.prompts.get_criticism_prompt(problem, nl_solution)
                                    for nl_solution in nl_solutions_for_problem
                                      ]
                                for problem, nl_solutions_for_problem in zip(problems, nl_solutions)]
            criticisms: list[list[str]] = batch_map_on_nested_list(get_criticism_prompts, lambda li: self.query_model("idea", li))
            get_fixes_prompts = map_nary_fn_on_nested_list(lambda e1, e2: tuple(list(e1) + [
                {"role": "assistant", "content": e2}, 
                {"role": "user", "content": self.prompts.FIX_CRITICISM_PROMPT}
                ]), get_criticism_prompts, criticisms)
            
            fixes: list[list[str]] = batch_map_on_nested_list(get_fixes_prompts, lambda li: self.query_model("idea", li))

            for i in range(len(problems)):
                for j in range(len(log_dict[i])):
                    log_dict[i][j]["criticism"] = criticisms[i][j]
                    log_dict[i][j]["fixed_solution"] = fixes[i][j]

            nl_solutions = [orig_sols + fixes_for_problem for orig_sols, fixes_for_problem in zip(nl_solutions, fixes)]

        nl_to_acc_sol_prompts = [[self.nl_to_code_solution_prompt(problem, nl_solution) for nl_solution in nl_solutions_for_problem]
                                 for problem, nl_solutions_for_problem in zip(problems, nl_solutions)]
        generated: list[list[str]] = batch_map_on_nested_list(nl_to_acc_sol_prompts, lambda li: self.query_model("code", li))
        assert len(generated) == len(problems) * self.num_codes_per_idea

        extracted_codes: list[list[str]] = map_nary_fn_on_nested_list(lambda s: markdown_codeblock_extract(s).strip(), generated)

        # codes_per_problem = [[] for _ in range(len(problems))]
        # for i, code in enumerate(extracted_codes):
        #     codes_per_problem[i % len(problems)].append(code)
        
        # output_codes = [self.select_code(problem, codes) for problem, codes in zip(problems, codes_per_problem)]
        
        for i in range(len(problems)):
            if self.add_criticism:
                assert len(extracted_codes[i]) == num_completions * 2
                for j in range(num_completions):
                    log_dict[i][j]["first_code"] = extracted_codes[i][j] 
                    log_dict[i][j]["fix_code"] = extracted_codes[i][j+num_completions]
            else:
                assert len(extracted_codes[i]) == num_completions
                for j in range(num_completions):
                    log_dict[i][j]["first_code"] = extracted_codes[i][j]

        log_to_dir(self.experiment_directory, {f"log_{datetime.datetime.now().strftime('%m-%dT%H:%M:%S')}.json": log_dict})
        return extracted_codes


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
    parser.add_argument(
        "--add-criticism",
        action="store_true",
        help="Whether to add 'You are wrong...' ideas"
    )

def get_simple_idea_model(args: argparse.Namespace) -> SearchModel:
    idea_model_path = parse_args_for_model_client(args, model_config_name="idea_model", temp_folder_base=args.experiment_directory)
    code_model_path = parse_args_for_model_client(args, model_config_name="code_model", temp_folder_base=args.experiment_directory)
    return SimpleIdeaModel(idea_model_path, code_model_path, args.experiment_directory, cache_file=args.cache_file, querier_batch_size=args.global_batch_size, idea_temperature=args.idea_temperature, code_temperature=args.code_temperature, top_p=args.top_p, max_tokens=args.max_tokens, use_few_shot=not args.zero_shot, use_sys_prompt=not args.no_sys_prompt, num_words=args.num_words, num_codes_per_idea=args.num_codes_per_idea, add_criticism=args.add_criticism)
