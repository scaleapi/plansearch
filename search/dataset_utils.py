from datasets import DatasetDict, load_dataset, Dataset

import json

from coderm.eval.generic import CompletionResult

from search.base_classes import Test, Problem


def filter_prompt(prompt: str, delimiter: str = '"""') -> str:
    return prompt.split(delimiter)[1].split(delimiter)[0].strip()


def convert_test_list_to_json(tests: list[Test]) -> str:
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


def process_problem_results(problems: list[Problem], codes_per_prob: list[list[str]], results_per_prob: list[list[CompletionResult]]) -> list[Problem]:
    assert len(problems) == len(codes_per_prob) == len(results_per_prob)
    for problem, codes, results in zip(problems, codes_per_prob, results_per_prob):
        assert len(codes) == len(results)

        if not (problem.solutions is None or len(problem.solutions) == 0):
            print("Warning: problem solutions previously filled.")
        if not (problem.fail_codes is None or len(problem.fail_codes) == 0):
            print("Warning: problem fail_codes previously filled.")

        problem.solutions = []
        problem.fail_codes = []

        for code, result in zip(codes, results):
            if result.passing:
                problem.solutions.append(code)
            else:
                problem.fail_codes.append(code)

    return problems
   

def write_dataset(output_dataset_name: str, problems_dict: dict[str, list[Problem]], private: bool = True) -> None:
    output_dataset = {}
    for split, problems in problems_dict.items():
        new_dataset = {"question": [], "starter_code": [], "input_output": [], "public_input_output": [], "solutions": [], "fail_codes": []}
        for problem in problems:
            prob_dict = problem.to_dict()
            for key in new_dataset:
                new_dataset[key].append(prob_dict[key])
            
        output_dataset[split] = Dataset.from_dict(new_dataset)

    output_dd = DatasetDict(output_dataset)
    output_dd.push_to_hub(output_dataset_name, private=private)


def parse_dataset(dataset_name: str, split: str) -> list[Problem]:
    data = load_dataset(dataset_name)
    assert isinstance(data, DatasetDict)

    problems = []
    for row in data[split]:
        assert isinstance(row, dict)

        public_tests = row.get("public_input_output", None)
        if public_tests is not None:
            public_tests = json.loads(public_tests)
        
        if isinstance(json.loads(row["input_output"])["inputs"][0], list) and row["starter_code"] == "":
            assert False
        problems.append(Problem.from_coderm_item(row["question"], row["starter_code"], public_tests, json.loads(row["input_output"]), row.get("solutions", None), row.get("fail_codes", None)))

    return problems


if __name__ == "__main__":
    parse_dataset("codegenning/F_taco_execclean", "train")

