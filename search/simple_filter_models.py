from typing import List, Any, Optional, Union
import argparse
from pathlib import Path
import os

from python_utils import log_to_dir  

from base_classes import Problem, SearchModel
from basic_prompting import SimplePromptModel, add_basic_prompting_args
from simple_idea_model import SimpleIdeaModel, add_simple_idea_args
from exec_utils import run_tests_per_code


class SimpleFilteringModel(SearchModel):
    def __init__(self, model_name: str, experiment_directory: Optional[str] = None, cache_file: Optional[str] = None, base_search_model: SearchModel = None, gen_batch_size: int = 10, num_batches_to_try: int = 10, timeout: int = 30):
        super().__init__(model_name, experiment_directory, cache_file)
        assert base_search_model is not None
        self.base_search_model = base_search_model

        self.gen_batch_size = gen_batch_size
        self.num_batches_to_try = num_batches_to_try
        self.timeout = timeout
        self.querier = self.base_search_model.querier

    def generate_solutions(self, problems: list[Problem], *args, **kwargs) -> list[str]:        
        selected_codes = [None] * len(problems)
        unsolved_idxs = list(range(len(problems)))

        for iter_num in range(self.num_batches_to_try):
            unsolved_problems = [problems[i] for i in unsolved_idxs]
            tiled_problems = unsolved_problems * self.gen_batch_size
            
            self.base_search_model.querier.set_log_directory(os.path.join(self.experiment_directory, f"iter_{iter_num}"))
            generated = self.base_search_model.generate_solutions(tiled_problems)
            assert len(generated) == len(tiled_problems)

            results = run_tests_per_code(generated, [problem.public_tests for problem in tiled_problems], [self.timeout] * len(tiled_problems))

            for i, result in enumerate(results):
                original_idx = unsolved_idxs[i % len(unsolved_problems)]
                result_good, _ = result

                if result_good:
                    selected_codes[original_idx] = generated[i]

            # Logging
            result_log = []
            for i in range(len(unsolved_problems)):
                sub_dict = {"problem_str": unsolved_problems[i].problem_str, 
                            "tests": [test.to_repr_dict() for test in unsolved_problems[i].public_tests],
                            "gens": []}
                for j in range(self.gen_batch_size):
                    sub_dict["gens"].append({"code": generated[i + len(unsolved_problems) * j],
                                             "passed": results[i + len(unsolved_problems) * j][0],
                                             "error": results[i + len(unsolved_problems) * j][1]})
                result_log.append(sub_dict)
            log_to_dir(os.path.join(self.experiment_directory, f"iter_{iter_num}"), {"test_results.json": result_log})


            unsolved_idxs = [i for i, code in enumerate(selected_codes) if code is None]
            print(f"Remaining 'unsolved' problems: {len(unsolved_idxs)}")

            if len(unsolved_idxs) == 0:
                break

        selected_codes = [code if code is not None else "" for code in selected_codes]
        return selected_codes


def add_simple_filter_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--gen-batch-size",
        type=int,
        default=10,
        help="Batch size for each generation"
    )
    parser.add_argument(
        "--num-batches-to-try",
        type=int,
        default=10,
        help="Number of batches to try overall"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout for each test"
    )
    

def add_simple_prompt_filter_args(parser: argparse.ArgumentParser):
    add_basic_prompting_args(parser)
    add_simple_filter_args(parser)

def get_simple_prompt_filter_model(args: argparse.Namespace) -> SearchModel:
    sp_model = SimplePromptModel(args.model, experiment_directory=args.experiment_directory, cache_file=args.cache_file, use_cot=args.cot, use_sys_prompts=not args.no_sys_prompt, use_few_shot=not args.zero_shot, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    return SimpleFilteringModel(args.model, experiment_directory=args.experiment_directory, cache_file=args.cache_file, base_search_model=sp_model, gen_batch_size=args.gen_batch_size, num_batches_to_try=args.num_batches_to_try, timeout=args.timeout)


def add_idea_filter_args(parser: argparse.ArgumentParser):
    add_simple_idea_args(parser)
    add_simple_filter_args(parser)

def get_idea_filter_model(args: argparse.Namespace) -> SearchModel:
    si_model = SimpleIdeaModel(args.idea_model, args.code_model, args.experiment_directory, cache_file=args.cache_file, idea_temperature=args.idea_temperature, code_temperature=args.code_temperature, top_p=args.top_p, max_tokens=args.max_tokens, use_few_shot=not args.zero_shot)
    return SimpleFilteringModel("idea_filter", experiment_directory=args.experiment_directory, cache_file=args.cache_file, base_search_model=si_model, gen_batch_size=args.gen_batch_size, num_batches_to_try=args.num_batches_to_try, timeout=args.timeout)