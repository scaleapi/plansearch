import argparse
import os
import shutil

from search.exec_utils import run_tests_per_code
from coderm.utils import gunzip_json_read, gunzip_json_write
from search.dataset_utils import parse_dataset
from search.eval import add_executor_args


def main(args: argparse.Namespace):
    data = gunzip_json_read(args.results_path)
    problems = parse_dataset(data["dataset_name"], args.split)

    if args.test_type == "public":
        test_types = ["public"]
    elif args.test_type == "private":
        test_types = ["private"]
    else:
        test_types = ["public", "private"]

    if args.testbank is not None:
        print(f"Overwriting testbank provided in results file with {args.testbank}")
        testbank = args.testbank
    else:
        testbank = data["dataset_name"].replace("codegenning/F_", "codegenning/B_")

    all_codes = []
    for i, item in enumerate(data["items"]):
        codes = []
        for result in item["results"]:
            codes.append(result["code"])
        all_codes.append(codes)
        assert i == int(item["unique_name"])

    tests = {
        "public": [problem.get_test_public() for problem in problems],
        "private": [problem.get_test_private() for problem in problems]
    }

    for test_type in test_types:
        specific_tests = tests[test_type]
        num_empty_tests = sum(len(test) == 0 if isinstance(test, list) else test is None for test in specific_tests)
        if num_empty_tests > 0:
            print(f"Warning: {num_empty_tests} problems with no {test_type} tests.")

    flattened_codes = []
    og_idxs = []
    og_specific_idxs = []

    for i, codes in enumerate(all_codes):
        og_idxs.extend([i] * len(codes))
        og_specific_idxs.extend(range(len(codes)))
        flattened_codes.extend(codes)

    for test_type in test_types:
        flattened_tests = [tests[test_type][og_idx] for og_idx in og_idxs]
        assert len(flattened_tests) == len(flattened_codes)

        results = run_tests_per_code(flattened_codes, flattened_tests, [args.timeout] * len(flattened_codes), fn_names_pc=[problems[og_idx].fn_name for og_idx in og_idxs], num_workers=args.exec_batch_size, testbank=testbank, executor=args.executor)
        
        if test_type == "public":
            bool_name = "passing_public"
            str_name = "output_public"
        else:
            bool_name = "passing"
            str_name = "output"

        for og_idx, og_specific_idx, result in zip(og_idxs, og_specific_idxs, results):
            data["items"][og_idx]["results"][og_specific_idx][bool_name] = result[0]
            data["items"][og_idx]["results"][og_specific_idx][str_name] = result[1]

    print(f"Copying {args.results_path} to {args.backup_file}...")
    shutil.copyfile(args.results_path, args.backup_file)
    print(f"Writing new data to {args.results_path}...")
    gunzip_json_write(args.results_path, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", required=True, type=str, help="The json.gz file with results")
    parser.add_argument("--split", default="test", type=str, help="The split of the dataset to use. Usually is 'test'")
    parser.add_argument("--test-type", default="public", type=str, choices=["public", "private", "both"], help="Which type of tests to run")
    parser.add_argument("--backup-file", default=None, type=str, help="Where to save the backup file. Default is to save to prefixed old_<results_file>")
    add_executor_args(parser)

    args = parser.parse_args()
    if args.backup_file is None:
        results_dir, results_file = os.path.split(args.results_path)
        args.backup_file = os.path.join(results_dir, f"old_{results_file}")

    main(args)
