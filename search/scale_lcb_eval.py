import argparse
import os
import shutil
import json
import datetime
from pathlib import Path

from coderm.eval import livecodebench_eval
from coderm.eval.generic import get_generic_coderm_argparser

from basic_prompting import add_basic_prompting_args, get_basic_prompting_model
from backtranslate import add_backtranslate_args, get_backtranslate_model
from simple_filter_models import (
    add_simple_prompt_filter_args, get_simple_prompt_filter_model,
    add_idea_filter_args, get_idea_filter_model,
)
from simple_idea_model import add_simple_idea_args, get_simple_idea_model 
from simple_observation_model import add_simple_observation_args, get_simple_observation_model
from pseudocode_model import add_pseudocode_args, get_pseudocode_model

from parsel_model import add_parsel_args, get_parsel_model 
from combo_observation_model import add_combo_observation_args, get_combo_observation_model


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
        "--num-gpus",
        type=int,
        default=8,
        help="Number of GPUs to use"
    )


def add_useless_args(args: argparse.Namespace):
    args.max_tokens = -1
    args.model = "unexpected_string"
    args.model_kind = "unexpected_string"
    args.evolver_e = -1
    args.rm = "unexpected_string"
    args.num_gpus = -100
    args.top_p = -1.
    args.temperature = -1.


def eval_main(search_alg: str):
    assert search_alg in SEARCH_ALGS_TO_GET_ARGS_MODEL

    parser = get_generic_coderm_argparser("codegenning/livecodebench_lite_filtered")
    add_universal_args(parser)
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

    output_result_path = args.output
    args.output = os.path.join(args.experiment_directory, "output_results")

    model = SEARCH_ALGS_TO_GET_ARGS_MODEL[search_alg][1](args)
    add_useless_args(args)
    livecodebench_eval.main(args, model)

    Path(os.path.dirname(output_result_path)).mkdir(parents=True, exist_ok=True)
    print(f"Copying output file from {args.output} to {output_result_path}")
    shutil.copyfile(args.output, output_result_path)

    print(f"Total spending: ${model.querier.current_price:.2f}")


if __name__ == "__main__":
    search_alg = os.environ.get("SEARCH_ALG", None)
    if search_alg is None:
        raise ValueError("SEARCH_ALG environment variable unset")
    if search_alg not in SEARCH_ALGS_TO_GET_ARGS_MODEL:
        raise ValueError(f"SEARCH_ALG of `{search_alg}` undefined. (Select from {set(SEARCH_ALGS_TO_GET_ARGS_MODEL)})")

    eval_main(search_alg)