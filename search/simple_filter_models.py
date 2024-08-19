from typing import List, Any, Optional, Union
import argparse
from pathlib import Path
import os

from python_utils import log_to_dir  

from search.base_classes import Problem, SearchModel
from search.one_prompt_models import BasicPromptModel, add_basic_prompting_args, get_basic_prompting_model
from search.simple_idea_model import SimpleIdeaModel, add_simple_idea_args, get_simple_idea_model
from search.exec_utils import run_tests_per_code


class SimpleFilteringModel(SearchModel):
    def __init__(self, model_config_path: str, base_search_model: SearchModel, experiment_directory: Optional[str] = None, cache_file: Optional[str] = None, querier_batch_size: Optional[int] = 12_288, gen_batch_size: int = 10, num_batches_to_try: int = 10, timeout: int = 30, num_workers: Optional[int] = os.cpu_count(), testbank: Optional[str] = None, executor: str = "http://127.0.0.1:8000"):
        super().__init__(model_config_path, experiment_directory, cache_file, querier_batch_size)
        self.base_search_model = base_search_model

        self.gen_batch_size = gen_batch_size
        self.num_batches_to_try = num_batches_to_try
        self.querier = self.base_search_model.querier

        self.timeout = timeout
        self.num_workers = num_workers
        self.testbank = testbank
        self.executor = executor

    def generate_solutions(self, problems: list[Problem], *args, **kwargs) -> list[str]:        
        selected_codes: list[Optional[str]] = [None] * len(problems)
        unsolved_idxs = list(range(len(problems)))

        for iter_num in range(self.num_batches_to_try):
            unsolved_problems = [problems[i] for i in unsolved_idxs]
            tiled_problems = unsolved_problems * self.gen_batch_size
            
            self.base_search_model.querier.set_log_directory(os.path.join(self.experiment_directory, f"iter_{iter_num}"))
            generated = self.base_search_model.generate_solutions(tiled_problems)
            assert len(generated) == len(tiled_problems)

            results = run_tests_per_code(generated, [problem.public_tests for problem in tiled_problems], [self.timeout] * len(tiled_problems), testbank=self.testbank, num_workers=self.num_workers, executor=self.executor)

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


def add_simple_prompt_filter_args(parser: argparse.ArgumentParser):
    add_basic_prompting_args(parser)
    add_simple_filter_args(parser)

def get_simple_prompt_filter_model(args: argparse.Namespace) -> SearchModel:
    sp_model = get_basic_prompting_model(args)
    return SimpleFilteringModel("simple_filter", experiment_directory=args.experiment_directory, cache_file=args.cache_file, querier_batch_size=args.global_batch_size, base_search_model=sp_model, gen_batch_size=args.gen_batch_size, num_batches_to_try=args.num_batches_to_try, timeout=args.timeout, num_workers=args.exec_batch_size, testbank=args.testbank, executor=args.executor)


def add_idea_filter_args(parser: argparse.ArgumentParser):
    add_simple_idea_args(parser)
    add_simple_filter_args(parser)

def get_idea_filter_model(args: argparse.Namespace) -> SearchModel:
    si_model = get_simple_idea_model(args)
    return SimpleFilteringModel("idea_filter", experiment_directory=args.experiment_directory, cache_file=args.cache_file, querier_batch_size=args.global_batch_size, base_search_model=si_model, gen_batch_size=args.gen_batch_size, num_batches_to_try=args.num_batches_to_try, timeout=args.timeout, num_workers=args.exec_batch_size, testbank=args.testbank, executor=args.executor)
