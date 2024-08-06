import os
from IPython import embed
import json
import argparse
from datasets import load_dataset, DatasetDict, Dataset
from backtranslate import BackTranslateModel
from simple_idea_model import SimpleIdeaModel
from basic_prompting import SimplePromptModel
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


def try_implementing_problems(problems: list[Problem], model: BackTranslateModel, gen_batch_size: int, num_batches: int, max_num_codes: int) -> list[list[str]]:
    assert max_num_codes > 1000, "Increase max num codes"
    selected_codes = [[] for _ in range(len(problems))]
    selected_failures = [[] for _ in range(len(problems))]
    unsolved_idxs = list(range(len(problems)))

    for __ in range(num_batches):
        unsolved_problems = [problems[i] for i in unsolved_idxs]
        tiled_problems = unsolved_problems * gen_batch_size
        
        generated = model.generate_solutions(tiled_problems, requery=True)
        assert len(generated) == len(tiled_problems)

        results = run_tests_per_code(generated, [problem.private_tests for problem in tiled_problems], [30] * len(tiled_problems))
        assert len(results) == len(generated)

        for i, (result, gen_code) in enumerate(zip(results, generated)):
            original_idx = unsolved_idxs[i % len(unsolved_problems)]
            result_good, _ = result
            if result_good:
                if len(selected_codes[original_idx]) < max_num_codes:
                    selected_codes[original_idx].append(gen_code)
            else:
                if len(selected_failures[original_idx]) < max_num_codes:
                    selected_failures[original_idx].append(gen_code)

        unsolved_idxs = [i for i, code in enumerate(selected_codes) if len(code) < max_num_codes]
        print(f"Remaining 'unsolved' problems: {len(unsolved_idxs)}")

        if len(unsolved_idxs) == 0:
            break
    
    return selected_codes, selected_failures


def main(args: argparse.Namespace):
    PATH_TO_RESULT = f"temp_create_taco_sol_logs/{datetime.datetime.now().strftime('%m-%dT%H-%M-%S')}"
    MAX_NUM_CODES = args.max_num_codes
    NUM_BATCHES_TO_TRY = args.num_batches
    GEN_BATCH_SIZE = args.gen_batch_size
    DATASET_NAME = args.dataset_name


    # btm = SimplePromptModel(idea_model="gpt-4o-mini", code_model="gpt-4o-mini", idea_temperature=0.8, code_temperature=0.8, experiment_directory=PATH_TO_RESULT, cache_file="temp_create_taco_sol_logs/cache.json")
    btm = SimplePromptModel("llama-3-1-8b-instruct", PATH_TO_RESULT, cache_file="temp_create_taco_sol_logs/cache.json", max_tokens=1024, temperature=0.8, top_p=0.95)
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

    selected_code_sols = []
    selected_code_fails = []
    for chunked_problem in tqdm(chunked_problems):
        correct, wrong = try_implementing_problems(chunked_problem, btm, GEN_BATCH_SIZE, NUM_BATCHES_TO_TRY, MAX_NUM_CODES)

        for c, r in zip(correct, wrong):
            selected_code_sols.append(c)
            selected_code_fails.append(r)

    assert len(selected_code_sols) == len(problems) == len(selected_code_fails)

    new_problems_dataset = {"problem_str": [], "starter_code": [], "tests": [], "generated_solutions": [], "generated_failures": []}

    for problem, selected_codes, selected_fails in zip(problems, selected_code_sols, selected_code_fails):
        new_problems_dataset["problem_str"].append(problem.problem_str)
        new_problems_dataset["starter_code"].append(problem.starter_code)
        new_problems_dataset["tests"].append(convert_test_list(problem.private_tests))
        new_problems_dataset["generated_solutions"].append(selected_codes)
        new_problems_dataset["generated_failures"].append(selected_fails)

    new_problems_dataset = Dataset.from_dict(new_problems_dataset)
    ds = DatasetDict({"train": new_problems_dataset})
    embed()
    # ds.push_to_hub(DATASET_NAME + "_with_gennedsolstest", commit_message="With generated GPT-4o-mini solutions")


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Backtranslate Taco dataset")
    parser.add_argument('--num-batches', type=int, default=4, help='Number of batches to try')
    parser.add_argument('--gen-batch-size', type=int, default=30, help='Size of the batches to try generating solutions')
    parser.add_argument('--dataset-name', type=str, default="codegenning/taco-rl-tests10-withpassingsolutions_v3", help='Dataset name')
    parser.add_argument('--batch-size', type=int, default=1028, help='Batch size to chunk dataset into')
    parser.add_argument('--max-num-codes', type=int, default=1001, help='Maximum number of code solutions to output')
    args = parser.parse_args()
    main(args)
