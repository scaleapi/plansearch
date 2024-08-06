import argparse
from copy import deepcopy
import os


from search.parsel.translate_to_parsel import compute_parsel_output
from search.base_classes import Problem, SearchModel


CONVERT_FN_TO_STDIO = """
import sys
print({root}(sys.stdin.read()))
"""

CONVERT_FN_TO_CLASS = """
class Solution:
    def {fn_name}(self, *args):
        return {root}(*args)
"""


class ParselModel(SearchModel):
    def __init__(self, model_name: str, args: argparse.Namespace):
        super().__init__(model_name, args.experiment_directory, cache_file=args.cache_file)
        self.args = deepcopy(args)
    
    def generate_solutions(self, problems: list[Problem], *args, **kwargs) -> list[str]:
        outputs = []
        for i, problem in enumerate(problems):
            output = compute_parsel_output(problem, self.args, self.querier, os.path.join(self.experiment_directory, f"prob-{i:03}"))
            if output is None:
                outputs.append("")
                continue

            root_name, impl = output
            if problem.has_starter_code():
                impl += CONVERT_FN_TO_CLASS.format(fn_name=problem.get_starter_code_fn_name(), root=root_name)
            else:
                impl += CONVERT_FN_TO_STDIO.format(root=root_name)
            outputs.append(impl)

        return outputs


def add_parsel_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--max-tries-of-combo", 
        type=int, 
        default=100000, 
        help="Maximum number of attempts for implementations"
    )
    parser.add_argument(
        "--parsel-exec-batch-size", 
        type=int, 
        default=32, 
        help="Batch size for computing tests"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=30, 
        help="Timeout for each test run (seconds)"
    )
    parser.add_argument(
        "--max-time", 
        type=float, 
        default=35, 
        help="Maximum time to run the implementation (seconds)"
    )
    parser.add_argument(
        "--num-completions", 
        type=int, 
        default=8,
        help="Number of completions to try for each function"
    )
    parser.add_argument(
        "--generate-tests", 
        action="store_true", 
        help="Generate tests for the functions"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num-solutions-to-generate", 
        type=int, 
        default=4, 
        help="Number of solutions sketches to generate"
    )
    parser.add_argument(
        "--num-translation-attempts", 
        type=int, 
        default=4, 
        help="Number of translation attempts to generate"
    )
    parser.add_argument(
        "--parsel-to-code-model-name", 
        type=str, 
        default="gpt-4-turbo", 
        help="Name of the model to convert Parsel to code when implementing functions"
    )
    parser.add_argument(
        "--gen-parsel-model-name", 
        type=str, 
        default="gpt-4-turbo", 
        help="Name of the model to generate Parsel code"
    )
    parser.add_argument(
        "--text-model-name", 
        type=str, 
        default="gpt-4-turbo", 
        help="Name of the model to generate text (solution sketches)"
    )
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=1500, 
        help="Maximum number of tokens in function implementation"
    )

def get_parsel_model(args: argparse.Namespace) -> SearchModel:
    return ParselModel("parsel", args)


if __name__ == "__main__":
    pass
