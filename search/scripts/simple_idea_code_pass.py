import datasets

from pathlib import Path
import json
import os
import argparse

from search.dataset_utils import parse_dataset
from search.base_classes import Problem
from search.exec_utils import run_tests_per_code

def get_log_files(log_dir):
    return [f for f in os.listdir(log_dir) if f.startswith(f"log_")]

def main(args: argparse.Namespace):
    log_directory = args.log_directory
    dataset_name = args.dataset
    split = args.split
    
    problems = parse_dataset(dataset_name, split)
    orig_num_problems = len(problems)
    log_files = get_log_files(log_directory)

    num_outputs = 0

    for file in log_files:
        with open(os.path.join(log_directory, file), "r") as f:
            obs = json.load(f)
        num_outputs += len(obs)

    assert num_outputs % orig_num_problems == 0
    n_completions = num_outputs // orig_num_problems
    print("n_completions:", n_completions)

    codes = []
    for file in log_files:
        with open(os.path.join(log_directory, file), "r") as f:
            code_results = json.load(f)
        for code in code_results:
            codes.append((code["question"], code["codes"]))

    problem_hashes: dict[str, Problem] = {}
    for problem in problems:
        problem_hashes[problem.problem_str] = problem 

    codes_to_run = []
    tests_to_run = []
    expanded_to_orig_idxs = []

    for i, (phash, code) in enumerate(codes):
        codes_to_run.extend(code)
        tests_to_run.extend([problem_hashes[phash].private_tests] * len(code))
        expanded_to_orig_idxs.extend([i] * len(code))

    timeouts = [args.timeout] * len(tests_to_run)

    results = run_tests_per_code(codes_to_run, tests_to_run, timeouts)
    write_data = [{"results": []} for _ in range(len(codes))]

    for orig_idx, (status, _) in zip(expanded_to_orig_idxs, results):
        write_data[orig_idx]["results"].append({"passing": status})
        
    if args.output_results is None:
        path_to_save = os.path.join(log_directory, "results_per_code_group.json")
    else:
        path_to_save = args.output_results
        if path_to_save is not None:
            Path(path_to_save).parent.mkdir(parents=True, exist_ok=True)
    with open(path_to_save, "w") as f:
        json.dump(write_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--log-directory', type=str, required=True, help="Where logs of the run were stored")
    parser.add_argument("--output-results", type=str, default=None, help="Where to output code execution results (if None, saves into log-directory)")
    parser.add_argument('--split', type=str, default="test", help='Split of dataset, only "test" supported for now')
    parser.add_argument('--timeout', type=int, default=60, help="Timeout")

    args = parser.parse_args()
    assert args.split == "test"
    main(args)
