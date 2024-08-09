import argparse
import os
import shutil
import json
import datetime
from pathlib import Path


from search.dataset_utils import parse_dataset, write_dataset, process_problem_results
from search.generate_from_model import do_full_run

from search.basic_prompting import add_basic_prompting_args, get_basic_prompting_model
from search.backtranslate import add_backtranslate_args, get_backtranslate_model
from search.simple_filter_models import (
    add_simple_prompt_filter_args, get_simple_prompt_filter_model,
    add_idea_filter_args, get_idea_filter_model,
)
from search.simple_idea_model import add_simple_idea_args, get_simple_idea_model 
from search.simple_observation_model import add_simple_observation_args, get_simple_observation_model
from search.pseudocode_model import add_pseudocode_args, get_pseudocode_model

from search.parsel_model import add_parsel_args, get_parsel_model 
from search.combo_observation_model import add_combo_observation_args, get_combo_observation_model


def noop_add_args(parser: argparse.ArgumentParser):
    pass

SEARCH_ALGS_TO_GET_ARGS_MODEL = {
    "basic_prompting": (add_basic_prompting_args, get_basic_prompting_model),
    "backtranslate": (add_backtranslate_args, get_backtranslate_model),
    "simple_filter": (add_simple_prompt_filter_args, get_simple_prompt_filter_model),
    "simple_idea": (add_simple_idea_args, get_simple_idea_model),
    "idea_filter": (add_idea_filter_args, get_idea_filter_model),
    "simple_observation": (add_simple_observation_args, get_simple_observation_model),
    "pseudocode": (add_pseudocode_args, get_pseudocode_model),
    "parsel": (add_parsel_args, get_parsel_model),
    "combo_observation": (add_combo_observation_args, get_combo_observation_model),
}


def add_executor_args(parser: argparse.ArgumentParser):
    cpu_count = os.cpu_count()
    if cpu_count is None:
        cpu_count = 1
    else:
        cpu_count = int(cpu_count * 0.8)  # lower for stability
    parser.add_argument(
        "--exec-batch-size",
        type=int,
        default=cpu_count,
        help="Total batch size for execution (defaults to os.cpu_count())"
    )
    parser.add_argument(
        "--executor",
        type=str,
        default="http://127.0.0.1:8000",
        help="Server URL for executing the code"
    )
    parser.add_argument(
        "--testbank",
        type=str,
        default=None,
        help="Testbank name, which is a detaset of hash -> test that gets sent to the server for caching purposes"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout for each test"
    )


def add_universal_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--experiment-directory",
        type=str,
        default=None,
        help="Directory to save experiment logs"
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default=None,
        help="Directory to cache generations. (Default: None, to not cache generations.)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "both"],
        required=True,
        help="Split of the dataset to evaluate/generate from"
    )
    parser.add_argument(
        "--exec-public",
        action="store_true",
        help="Whether to execute public tests as well",
        )
    parser.add_argument(
        "--output-dataset",
        type=str,
        default=None,
        help="Where to output dataset of generations (will upload to Huggingface). None otherwise."
    )
    parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Where to output results of eval"
            )
    parser.add_argument(
        "--completion-limit",
        type=int,
        default=1,
        help="Number of completions to generate per problem"
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=12_288,
        help="Global querier batch size for generations. (Will save into cache every global_batch_size queries.)"
    )


def generate_and_eval(search_alg: str):
    assert search_alg in SEARCH_ALGS_TO_GET_ARGS_MODEL

    parser = argparse.ArgumentParser()
    add_universal_args(parser)
    add_executor_args(parser)
    SEARCH_ALGS_TO_GET_ARGS_MODEL[search_alg][0](parser)
    args = parser.parse_args()

    args.experiment_directory = (args.experiment_directory if args.experiment_directory is not None
                            else f"logs/{search_alg}_{datetime.datetime.now().strftime('%m-%dT%H:%M:%S')}")
    
    # Make the directory if it doesn't exist
    Path(args.experiment_directory).mkdir(parents=True, exist_ok=True)
    
    # Save a copy of args inside experiment directory
    args_file = os.path.join(args.experiment_directory, "args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=2)

    splits = []
    if args.split == "both":
        print(f"Since `args.split` is 'both', only outputting `test` split results to {args.output}. ")
        splits = ["train", "test"]
    elif args.split == "train":
        splits = ["train"]
    else:
        splits = ["test"]

    output_result_path = args.output
    experiment_dir_output = os.path.join(args.experiment_directory, "output_results")

    model = SEARCH_ALGS_TO_GET_ARGS_MODEL[search_alg][1](args)

    codes_results_dict = {}
    for split in splits:
        codes, results = do_full_run(model, args.dataset, split, args.completion_limit, experiment_dir_output + split, exec_public=args.exec_public, testbank=args.testbank, num_workers=args.exec_batch_size, executor=args.executor)
        codes_results_dict[split] = (codes, results)

    if output_result_path is not None:
        Path(os.path.dirname(output_result_path)).mkdir(parents=True, exist_ok=True)
        print(f"Copying output file from {experiment_dir_output + splits[-1]} to {args.output}")
        shutil.copyfile(experiment_dir_output + splits[-1], output_result_path)


    if args.output_dataset is not None:
        problems_dict = {}
        for split in splits:
            problems = parse_dataset(args.dataset, split)

            assert len(problems) == len(codes_results_dict[split][0]) == len(codes_results_dict[split][1])
            problems = process_problem_results(problems, codes_results_dict[split][0], codes_results_dict[split][1])
            problems_dict[split] = problems

        write_dataset(args.output_dataset, problems_dict, private=True)


    print(f"Total spending: ${model.querier.current_price:.2f}")


if __name__ == "__main__":
    assert os.environ["EXECUTOR_URL"] == '***REMOVED***', "Check your EXECUTOR_URL"
    search_alg = os.environ.get("SEARCH_ALG", None)
    if search_alg is None:
        raise ValueError("SEARCH_ALG environment variable unset")
    if search_alg not in SEARCH_ALGS_TO_GET_ARGS_MODEL:
        raise ValueError(f"SEARCH_ALG of `{search_alg}` undefined. (Select from {set(SEARCH_ALGS_TO_GET_ARGS_MODEL)})")

    generate_and_eval(search_alg)

