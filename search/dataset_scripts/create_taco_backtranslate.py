import os
import json
import argparse
from datasets import load_dataset, DatasetDict, Dataset
from backtranslate import BackTranslateModel
from base_classes import Problem, Test
from exec_utils import run_tests_per_code
from typing import Any, Optional
import datetime
from python_utils import chunk
from tqdm import tqdm
import random

def filter_prompt(prompt: str, delimiter: str = '"""') -> str:
    return prompt.split(delimiter)[1].split(delimiter)[0].strip()

def convert_test_list(tests: list[Test]) -> str:
    output_dict = {"inputs": [], "outputs": []}
    assert len(tests)
    for test in tests:
        assert test.fn_name == tests[0].fn_name, "All tests must have the same fn_name"

    fn_name = tests[0].fn_name
    if fn_name is not None and fn_name != "":
        output_dict["fn_name"] = fn_name

    for test in tests:
        output_dict["inputs"].append(test.get_input_no_kwargs())
        output_dict["outputs"].append(test.output)
    
    return json.dumps(output_dict)


def try_implementing_problems(problems: list[Problem], model: BackTranslateModel, gen_batch_size: int, num_batches: int, path_to_result: str) -> tuple[list[Optional[str]], list[Optional[str]]]:
    selected_codes = [None] * len(problems)
    selected_nl_sols = [None] * len(problems)
    unsolved_idxs = list(range(len(problems)))

    for iter_num in range(num_batches):
        unsolved_problems = [problems[i] for i in unsolved_idxs]
        tiled_problems = unsolved_problems * gen_batch_size
        
        model.querier.set_log_directory(os.path.join(path_to_result, f"iter_{iter_num}"))
        generated = model.generate_solutions(tiled_problems, requery=True)
        assert len(generated) == len(tiled_problems)

        results = run_tests_per_code(generated, [problem.private_tests for problem in tiled_problems], [30] * len(tiled_problems))


        query_path = os.path.join(path_to_result, f"iter_{iter_num}")
        solution_files = [f for f in os.listdir(query_path) if f.startswith("solution")]

        solution_paths = []
        for solution_file in solution_files:
            solution_path = os.path.join(query_path, solution_file)
            print(f"Found solution file: {solution_path}")
            solution_paths.append(solution_path)
        assert len(solution_paths)

        nl_solutions = []
        for solution_path in solution_paths:
            with open(solution_path, "r") as solution_file:
                nl_sub_solutions = json.load(solution_file)
                nl_solutions.extend([e["completion"]["text"] for e in nl_sub_solutions])

        if not (len(nl_solutions) == len(results) == len(generated)):
            breakpoint()
        assert len(nl_solutions) == len(results) == len(generated)

        for i, (result, gen_code, gen_nl_sol) in enumerate(zip(results, generated, nl_solutions)):
            original_idx = unsolved_idxs[i % len(unsolved_problems)]
            result_good, _ = result
            if result_good:
                assert (gen_code is None) == (gen_nl_sol is None)
                selected_codes[original_idx] = gen_code
                selected_nl_sols[original_idx] = gen_nl_sol

        unsolved_idxs = [i for i, code in enumerate(selected_codes) if code is None]
        print(f"Remaining 'unsolved' problems: {len(unsolved_idxs)}")

        if len(unsolved_idxs) == 0:
            break
    
    return selected_codes, selected_nl_sols


def main(args: argparse.Namespace):
    PATH_TO_RESULT = f"temp_backtranslate_gen_logs/{datetime.datetime.now().strftime('%m-%dT%H-%M-%S')}"

    NUM_BATCHES_TO_TRY = args.num_batches
    GEN_BATCH_SIZE = args.gen_batch_size
    NUM_WORDS = args.num_words
    DATASET_NAME = args.dataset_name

    btm = BackTranslateModel(model_config_path="gpt-4o-mini", experiment_directory=PATH_TO_RESULT, cache_file="caches/temp_backtranslate_cache.json", num_words=NUM_WORDS)
    data = load_dataset(DATASET_NAME)


    prompts = list(data["train"]["prompt"])
    actual_prompts = [filter_prompt(prompt) for prompt in prompts]
    actual_starter_code = list(data["train"]["starter_code"]) 

    tests = list(data["train"]["input_output"])
    actual_tests = [json.loads(test_set) for test_set in tests]

    actual_solutions = list(data["train"]["solutions"])


    problems: list[Problem] = []
    random.seed(42)
    for i, (prompt, starter_code, test, solutions) in list(enumerate(zip(actual_prompts, actual_starter_code, actual_tests, actual_solutions))):
        try:
            problems.append(Problem.from_coderm_item(prompt, starter_code, None, tests=test, solutions=solutions))
        except:
            new_outputs = []
            for t in test["outputs"]:
                if isinstance(t, str):
                    new_outputs.append(t)
                else:
                    assert isinstance(t, list)
                    new_outputs.append('\n'.join(t) + '\n')
            test["outputs"] = new_outputs
            problems.append(Problem.from_coderm_item(prompt, starter_code, None, tests=test, solutions=solutions))


    expanded_problems: list[Problem] = []
    problem_to_expand_idx: list[list[int]] = []
    expand_to_problem_idx: list[int] = []
    for i, problem in enumerate(problems):
        problem_to_expand_idx.append([])
        
        if args.max_solutions_to_try is not None:
            selected_solutions = random.sample(problem.solutions, min(args.max_solutions_to_try, len(problem.solutions)))
        else:
            selected_solutions = problem.solutions

        for solution in selected_solutions:
            problem_to_expand_idx[i].append(len(expanded_problems))
            expanded_problems.append(Problem(problem.problem_str, problem.starter_code, problem.public_tests, problem.private_tests, [solution]))
            expand_to_problem_idx.append(i)


    chunked_problems = chunk(expanded_problems, args.batch_size)

    batched_selected_sols = [try_implementing_problems(chunked_problem, btm, GEN_BATCH_SIZE, NUM_BATCHES_TO_TRY, os.path.join(PATH_TO_RESULT, f"batch_{i}"))
                        for i, chunked_problem in enumerate(tqdm(chunked_problems))]

    selected_nl_sols = [nl_sol_attempt
                        for _, nl_batch_sols in batched_selected_sols
                        for nl_sol_attempt in nl_batch_sols]
    selected_code_sols = [code_sol_attempt
                          for code_batch_sols, _ in batched_selected_sols
                          for code_sol_attempt in code_batch_sols]
    assert len(selected_nl_sols) == len(selected_code_sols) == len(expanded_problems)

    new_problems_dataset = {"problem_str": [], "starter_code": [], "tests": [], "orig_code_solutions": [], "nl_solutions": [], "code_from_nl_solutions": []}

    for orig_idx, expand_idxs in enumerate(problem_to_expand_idx):
        filtered_orig_code_solutions = []
        filtered_nl_solutions = []
        filtered_code_from_nl_solutions = []
        for idx in expand_idxs:
            if not ((selected_nl_sols[idx] is None) == (selected_code_sols[idx] is None)):
                breakpoint()
            assert (selected_nl_sols[idx] is None) == (selected_code_sols[idx] is None)
            if selected_nl_sols[idx] is not None:
                filtered_orig_code_solutions.append(expanded_problems[idx].solutions[0])
                filtered_nl_solutions.append(selected_nl_sols[idx])
                filtered_code_from_nl_solutions.append(selected_code_sols[idx])
        
        if len(filtered_nl_solutions) == 0:
            continue

        new_problems_dataset["problem_str"].append(problems[orig_idx].problem_str)
        new_problems_dataset["starter_code"].append(problems[orig_idx].starter_code)
        new_problems_dataset["tests"].append(convert_test_list(problems[orig_idx].private_tests))
        new_problems_dataset["orig_code_solutions"].append(filtered_orig_code_solutions)
        new_problems_dataset["nl_solutions"].append(filtered_nl_solutions)
        new_problems_dataset["code_from_nl_solutions"].append(filtered_code_from_nl_solutions)

    new_problems_dataset = Dataset.from_dict(new_problems_dataset)
    ds = DatasetDict({"train": new_problems_dataset})
    ds.push_to_hub(DATASET_NAME + "_with_nlsols", commit_message="With NL solutions")


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Backtranslate Taco dataset")
    parser.add_argument('--num-batches', type=int, default=4, help='Number of batches to try')
    parser.add_argument('--gen-batch-size', type=int, default=1, help='Size of the batches to try generating solutions')
    parser.add_argument('--num-words', type=int, default=100, help='Number of words for backtranslation')
    parser.add_argument('--dataset-name', type=str, default="codegenning/taco-rl-tests10-withpassingsolutions_v3", help='Dataset name')
    parser.add_argument('--batch-size', type=int, default=32768, help='Batch size to chunk dataset into')
    parser.add_argument('--max-solutions-to-try', type=int, default=None, help='Maximum number of code solutions to try backtranslating')
    args = parser.parse_args()
    main(args)
