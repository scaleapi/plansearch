import os
import json
import argparse
from datasets import load_dataset, DatasetDict, Dataset
from backtranslate import BackTranslateModel
from basic_prompting import SimplePromptModel
from simple_idea_model import SimpleIdeaModel
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


def try_implementing_problems(problems: list[Problem], model: BackTranslateModel, num_codes: int) -> tuple[list[list[str]], list[list[str]]]:
    selected_codes = [[] for _ in range(len(problems))]
    selected_fails = [[] for _ in range(len(problems))]

    generated = model.generate_solutions(problems * num_codes, requery=True)
    results = run_tests_per_code(generated, [problem.private_tests for problem in problems] * num_codes, [30] * len(problems) * num_codes)

    for i, (result, gen_code) in enumerate(zip(results, generated)):
        orig_idx = i % len(problems)
        result_good, _ = result
        if result_good:
            selected_codes[orig_idx].append(gen_code)
        else:
            selected_fails[orig_idx].append(gen_code)
    
    return selected_codes, selected_fails


def main(args: argparse.Namespace):
    PATH_TO_RESULT = f"temp_create_general_sol_logs/{datetime.datetime.now().strftime('%m-%dT%H-%M-%S')}"
    NUM_CODES = args.num_codes
    DATASET_NAME = args.dataset_name


    # btm = SimpleIdeaModel(idea_model="gpt-4o-mini", code_model="gpt-4o-mini", idea_temperature=0.8, code_temperature=0.8, experiment_directory=PATH_TO_RESULT, cache_file="temp_create_taco_sol_logs/cache.json")
    btm = SimplePromptModel(model_name=args.model_name, temperature=0.8, experiment_directory=PATH_TO_RESULT, cache_file="temp_create_general_sol_logs/cache.json",
                            num_shot=1, top_p=0.95)
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

    chunked_problems = list(chunk(problems, args.batch_size))

    all_code_sols = []
    all_code_fails = []
    
    for chunked_problem in tqdm(chunked_problems):
        good_codes, fail_codes = try_implementing_problems(chunked_problem, btm, NUM_CODES)
        for good_code, fail_code in zip(good_codes, fail_codes):
            all_code_sols.append(good_code)
            all_code_fails.append(fail_code)
        
    assert len(all_code_sols) == len(all_code_fails) == len(problems)

    new_problems_dataset = {"problem_str": [], "starter_code": [], "tests": [], "correct_solutions": [], "fail_solutions": []}

    for problem, code_sols_per_prob, code_fails_per_prob in zip(problems, all_code_sols, all_code_fails):
        new_problems_dataset["problem_str"].append(problem.problem_str)
        new_problems_dataset["starter_code"].append(problem.starter_code)
        new_problems_dataset["tests"].append(convert_test_list(problem.private_tests))
        new_problems_dataset["correct_solutions"].append(code_sols_per_prob)
        new_problems_dataset["fail_solutions"].append(code_fails_per_prob)

    new_problems_dataset = Dataset.from_dict(new_problems_dataset)
    ds = DatasetDict({"train": new_problems_dataset})
    ds.push_to_hub(f"{DATASET_NAME}_{(args.model_name).replace('/', '-')}_sols", commit_message=f"{args.model_name} codes on {DATASET_NAME}", private=True)


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Backtranslate Taco dataset")
    parser.add_argument('--model-name', type=str, default="Which model to query")
    parser.add_argument('--dataset-name', type=str, default="codegenning/taco-rl", help='Dataset name')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size to chunk dataset into')
    parser.add_argument('--num-codes', type=int, default=120, help='Number of code solutions to output')
    args = parser.parse_args()
    main(args)
