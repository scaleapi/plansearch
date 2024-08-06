import os
from pathlib import Path
from datasets import load_dataset
from typing import Optional
from coderm.eval.generic import make_items_from_ds, Completion, CompletionResult, CompletionItem
from coderm.utils import gunzip_json_write

from search.base_classes import SearchModel
from search.exec_utils import run_tests_per_code
from search.dataset_utils import parse_dataset


def do_full_run(model: SearchModel, dataset_name: str, split: str, num_completions: int, output_path: str, exec_public: bool = False, testbank: Optional[str] = None, num_workers: Optional[int] = os.cpu_count(), executor: str = "http://127.0.0.1:8000") -> tuple[list[list[str]], list[list[CompletionResult]]]:
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
    
    expanded_problems = problems * num_completions

    num_empty_tests = sum([len(problem.private_tests) == 0 for problem in problems])
    num_empty_public_tests = sum([len(problem.public_tests) == 0 for problem in problems])

    codes = model.generate_solutions(expanded_problems)
    results = run_tests_per_code(codes, [problem.private_tests for problem in expanded_problems], [60] * len(expanded_problems), testbank=testbank, num_workers=num_workers, executor=executor)

    if num_empty_tests > 0:
        print(f"Warning: {num_empty_tests} empty private tests." )
    if num_empty_public_tests > 0 and exec_public:
        print(f"Warning: {num_empty_public_tests} empty public tests." )

    public_results = [(None, None)] * len(expanded_problems)
    if exec_public:
        public_results = run_tests_per_code(codes, [problem.public_tests for problem in expanded_problems], [60] * len(expanded_problems), testbank=testbank, num_workers=num_workers, executor=executor)

    assert len(codes) == len(results) == len(problems) * num_completions

    all_codes = [[] for _ in range(len(problems))]
    all_results = [[] for _ in range(len(problems))]
    for i, (code, result, public_result) in enumerate(zip(codes, results, public_results)):
        orig_idx = i % len(problems)
        all_codes[orig_idx].append(code)
        completion_items[orig_idx].completions.append(Completion(code, -1, -1))

        all_results[orig_idx].append(CompletionResult(result[0], result[1], public_result[0], public_result[1]))
        completion_items[orig_idx].results.append(all_results[orig_idx][-1])
        

    save_completions(completion_items, output_path, model=model.model_name, completion_limit=num_completions, dataset_name=dataset_name)

    return all_codes, all_results 


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

