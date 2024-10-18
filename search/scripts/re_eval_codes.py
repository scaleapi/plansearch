import argparse
import os
import datetime
import tqdm

from search.exec_utils import run_tests_per_code
from coderm.utils import gunzip_json_read, gunzip_json_write
from search.dataset_utils import parse_dataset
from search.eval import add_executor_args
from search.python_utils import chunk


def save_data(data: dict, args: argparse.Namespace, og: bool = False):
    current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    if og:
        filename = f"og_{args.results_filename}_{current_time}"
    else:
        filename = f"{args.results_filename}_{current_time}"
    save_path = os.path.join(args.backup_directory, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving to {save_path}...")
    gunzip_json_write(save_path, data)


def main(args: argparse.Namespace):
    data = gunzip_json_read(args.results_path)
    problems = parse_dataset(data["dataset_name"], args.split)
    save_data(data, args, og=True)

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
        try:
            unique_id = int(item["unique_name"])
        except:
            unique_id = None
        assert i == unique_id or problems[i].problem_str.strip() == item["prompt"].strip()

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
        if test_type == "public":
            bool_name = "passing_public"
            str_name = "output_public"
        else:
            bool_name = "passing"
            str_name = "output"

        flattened_tests = [tests[test_type][og_idx] for og_idx in og_idxs]
        assert len(flattened_tests) == len(flattened_codes)

        if args.overwrite:
            write_idxs = list(range(len(flattened_tests)))
        else:
            write_idxs = []
            for i, (og_idx, og_specific_idx) in enumerate(zip(og_idxs, og_specific_idxs)):
                passed = data["items"][og_idx]["results"][og_specific_idx].get(bool_name, None)
                output = data["items"][og_idx]["results"][og_specific_idx].get(str_name, None)
                assert (passed is None) == (output is None)
                if passed is None:
                    write_idxs.append(i)

        if len(write_idxs) == 0:
            print(f"0 tests... Skipping {test_type} test execution.")
            continue

        print(f"Running {len(write_idxs)} {test_type} tests...")
        
        codes_to_run = list(chunk([flattened_codes[i] for i in write_idxs], args.save_every))
        tests_to_run = list(chunk([flattened_tests[i] for i in write_idxs], args.save_every))
        og_idxs_to_run = list(chunk([og_idxs[i] for i in write_idxs], args.save_every))
        og_specific_idxs_to_run = list(chunk([og_specific_idxs[i] for i in write_idxs], args.save_every))

        total = 0
        for i, (chunk_codes, chunk_tests, chunk_og_idxs, chunk_og_specific_idxs) in enumerate(tqdm.tqdm(list(zip(codes_to_run, tests_to_run, og_idxs_to_run, og_specific_idxs_to_run)))):
            chunk_results = run_tests_per_code(
                chunk_codes, chunk_tests, 
                [args.timeout] * len(chunk_codes),
                fn_names_pc=[problems[og_idx].fn_name for og_idx in chunk_og_idxs],
                num_workers=args.exec_num_processes,
                total_num_concurrent=args.exec_batch_size,
                testbank=testbank,
                executor=args.executor,
                return_none=True
            )
            total += len(chunk_results)

            for og_idx, og_specific_idx, result in zip(chunk_og_idxs, chunk_og_specific_idxs, chunk_results):
                if result is not None:
                    data["items"][og_idx]["results"][og_specific_idx][bool_name] = result[0]
                    data["items"][og_idx]["results"][og_specific_idx][str_name] = result[1]

            print(f"Finished chunk {i+1}/{len(codes_to_run)} with {len(chunk_results)} tests...")
            save_data(data, args)
            gunzip_json_write(args.results_path, data)
            print("Done saving for iteration.")
        assert total == len(write_idxs)

    print(f"Final: Writing new data to {args.results_path}...")
    gunzip_json_write(args.results_path, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", required=True, type=str, help="The json.gz file with results")
    parser.add_argument("--split", default="test", type=str, help="The split of the dataset to use. Usually is 'test'")
    parser.add_argument("--test-type", default="public", type=str, choices=["public", "private", "both"], help="Which type of tests to run")
    parser.add_argument("--backup-directory", default=None, type=str, help="Where to save the backup file. Default is to save to old_results")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing execution results")
    parser.add_argument("--save-every", default=None, type=int, help="Save to results file every X tests run. Does not save if default.")
    add_executor_args(parser)

    args = parser.parse_args()
    if args.backup_directory is None:
        results_dir, results_file = os.path.split(args.results_path)
        args.backup_directory = os.path.join(results_dir, "old_results")
        args.results_filename = results_file

    main(args)
