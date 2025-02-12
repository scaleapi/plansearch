import os
from pathlib import Path
from datasets import load_dataset
from typing import Optional
from coderm.eval.generic import make_items_from_ds, Completion, CompletionResult, CompletionItem
from coderm.utils import gunzip_json_write

from search.base_classes import SearchModel
from search.exec_utils import run_tests_per_code, run_individual_tests_per_code
from search.dataset_utils import parse_dataset
from search.python_utils import batch_map_on_nested_list


def do_full_run(model: SearchModel, dataset_name: str, split: str, num_repeats: int, num_completions_from_model: int, output_path: str, exec_type: str, testbank: Optional[str] = None, num_workers: Optional[int] = os.cpu_count(), total_num_concurrent: int = 1000, executor: str = "http://127.0.0.1:8000", timeout: int = 60, run_individual_tests: bool = False) -> tuple[list[list[str]], list[list[CompletionResult]]]:
    assert num_completions_from_model != 0
    assert num_repeats >= 1

    problems = parse_dataset(dataset_name, split)

    optional_features = {
        "public_tests_col": "public_input_output",
        "difficulty_col": "difficulty",
        "unique_name_col": "id",
        "solutions_col": "solutions",
    }
    ds = load_dataset(dataset_name, split=split)
    for feature in optional_features:
        if feature not in ds.features.keys():
            optional_features[feature] = None

    completion_items = make_items_from_ds(ds, "question", "input_output", public_tests_col=optional_features["public_tests_col"], starter_code_col="starter_code", difficulty_col=optional_features["difficulty_col"], unique_name_col=optional_features["unique_name_col"], solutions_col=optional_features["solutions_col"])
    

    if num_completions_from_model != 1:
        assert model.COMPLETION_FROM_MODEL_SUPPORTED
    expanded_problems = problems * num_repeats

    num_empty_tests = sum([(len(problem.private_tests) == 0) and (problem.private_exec_string is None) for problem in problems])
    num_empty_public_tests = sum([(len(problem.public_tests) == 0) and (problem.public_exec_string is None) for problem in problems])

    unsorted_codes = model.generate_solutions(expanded_problems, num_completions=num_completions_from_model)

    assert len(unsorted_codes) == len(problems) * num_repeats
    if num_completions_from_model != -1:
        assert all(len(code) == num_completions_from_model for code in unsorted_codes)
    
    codes = [[] for _ in range(len(problems))]
    for i, output_code in enumerate(unsorted_codes):
        codes[i % len(problems)].extend(output_code)
    assert len(codes) == len(problems)

    flattened_codes = []
    flattened_fn_names = []
    flattened_private_tests = []
    flattened_public_tests = []
    orig_problem_idxs = []
    for i, codes_for_problem in enumerate(codes):
        flattened_codes.extend(codes_for_problem)
        flattened_fn_names.extend([expanded_problems[i].fn_name] * len(codes_for_problem))
        flattened_private_tests.extend([expanded_problems[i].get_test_private()] * len(codes_for_problem))
        flattened_public_tests.extend([expanded_problems[i].get_test_public()] * len(codes_for_problem))
        orig_problem_idxs.extend([i] * len(codes_for_problem))

    num_total_to_eval = len(flattened_codes)
    if num_completions_from_model != -1:
        assert num_total_to_eval == len(problems) * num_completions_from_model * num_repeats

    if num_empty_tests > 0 and exec_type != "none":
        print(f"Warning: {num_empty_tests} problems with no private tests." )
    if num_empty_public_tests > 0 and exec_type == "both":
        print(f"Warning: {num_empty_public_tests} problems with no public tests." )

    private_results = [(None, None)] * num_total_to_eval
    public_results = [(None, None)] * num_total_to_eval 

    if exec_type != "none":
        if run_individual_tests:
            private_results = run_individual_tests_per_code(flattened_codes, flattened_private_tests, [timeout] * num_total_to_eval, fn_names_pc=flattened_fn_names, testbank=testbank, num_workers=num_workers, total_num_concurrent=total_num_concurrent, executor=executor)
        else:
            private_results = run_tests_per_code(flattened_codes, flattened_private_tests, [timeout] * num_total_to_eval, fn_names_pc=flattened_fn_names, testbank=testbank, num_workers=num_workers, total_num_concurrent=total_num_concurrent, executor=executor)
        assert len(private_results) == num_total_to_eval

    if exec_type == "both":
        public_results = run_tests_per_code(flattened_codes, flattened_public_tests, [timeout] * num_total_to_eval, fn_names_pc=flattened_fn_names, testbank=testbank, num_workers=num_workers, total_num_concurrent=total_num_concurrent, executor=executor)
        assert len(public_results) == num_total_to_eval

    all_results = [[] for _ in range(len(problems))]
    for orig_idx, code, private_result, public_result in zip(orig_problem_idxs, flattened_codes, private_results, public_results):
        completion_items[orig_idx].completions.append(Completion(code, -1, -1))

        if run_individual_tests:
            score = problems[orig_idx].calculate_score([res[0] for res in private_result])
            report_result = (True, "")
            for result in private_result:
                if not result[0]:
                    report_result = result
                    break
            
            all_results[orig_idx].append(CompletionResult(report_result[0], report_result[1], public_result[0], public_result[1], score))

        else:
            all_results[orig_idx].append(CompletionResult(private_result[0], private_result[1], public_result[0], public_result[1], None))

        if exec_type != "none":
            completion_items[orig_idx].results.append(all_results[orig_idx][-1])
        

    save_completions(completion_items, output_path, model=model.model_name, completion_limit=num_total_to_eval, dataset_name=dataset_name)

    return codes, all_results 


# From Federico Cassano's CodeRM
def save_completions(
        items: list[CompletionItem],
        output_path: str,
        verbose: bool = True,
        model: str = "model",
        completion_limit: int = -1,
        dataset_name: str = "dataset",
):
    outpath = Path(output_path)
    if outpath.exists():
        outpath.unlink()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Saving completions to {outpath}")

    d = {
        "model": model,
        "max_tokens": -1,
        "top_p": -1,
        "temperature": -1,
        "completion_limit": completion_limit,
        "dataset_name": dataset_name,
        "items": [item.to_dict() for item in items],
    }

    gunzip_json_write(outpath, d)

