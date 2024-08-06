from datasets import load_dataset, Dataset, DatasetDict

import json
import os
from typing import Any
import argparse

from search.queriers import LLMQuerier
from search.query_clients import OpenAIClient
from search.python_utils import fn_arg_join


SYS_PROMPT = ("You are an expert problem-setter for competitive programming competitions. " + 
"You will be given a competitive programming problem, which is evaluated on a private suite of tests, not shown to the participant. " + 
"The problem statement may have sample tests in the statement itself, which are shown to the user. " + 
"You will also be given a test case from the test suite. Determine whether it is present in the problem statement.")

def get_prompt(question: str, inp: str, out: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": 
        f"Here is the competitive programming problem in its entirety:\n\n{question}\n\n" + 
        "Here is a test in the test suite:\n\n" +
        f"Input:\n{inp}\n\nOutput:\n{out}\n\n" + 
        "Is the above test included exactly in the problem statement as shown, or is it in the private test suite? You must be completely correct. Before outputting your answer, recite the relevant test cases in the problem statement (if there are any) exactly."
    }]

SECONDARY_PROMPT = 'Using the reasoning above, answer "Yes." if the test is included exactly in the problem statement, or "No." if the test is not included exactly in the problem statement. Do not output anything else.'

UNIQUE_DELIM = "@,~z"

def get_delimed_string(s: str) -> str:
    return UNIQUE_DELIM.join(s.strip().split())

def wrap_newline(s: str) -> str:
    return s.strip() + '\n'

def check_in_dict(d: dict, inp: str, out: str):
    if (inp, out) in d:
        return
    d[(inp, out)] = None

def filter_question(question: str) -> str:
    WORD_TO_NUM = {
        "One\n": "1\n",
        "Two\n": "2\n",
        "Three\n": "3\n",
        "Four\n": "4\n",
        "Five\n": "5\n",
        "Six\n": "6\n",
        "Seven\n": "7\n",
        "Eight\n": "8\n",
        "Nine\n": "9\n",
        "Ten\n": "10\n",
    }
    for word, num in WORD_TO_NUM.items():
        question = question.replace(word, num)
    return question

def reduce_tests(data) -> list[str]:
    all_tests = []
    for i, row in enumerate(data):
        tests = json.loads(row["input_output"])
        check_dict = {}
        for inp, out in zip(tests["inputs"], tests["outputs"]):
            if row["starter_code"] == "":
                inp = wrap_newline(inp)
                out = wrap_newline(out)
            check_in_dict(check_dict, json.dumps(inp), json.dumps(out))

        new_inps = []
        new_outs = []
        for inp, out in check_dict.keys():
            new_inps.append(json.loads(inp))
            new_outs.append(json.loads(out))

            if row["starter_code"] == "":
                assert new_inps[-1].endswith("\n")
                assert new_outs[-1].endswith("\n")

        tests["inputs"] = new_inps
        tests["outputs"] = new_outs
        all_tests.append(json.dumps(tests))
    return all_tests

def is_multi_solution(row) -> bool:
    if "print any" in row["question"] or "output any" in row["question"]:
        return True
    if "I Ching" in row["question"] or "any of them may be output" in row["question"] or "provide any" in row["question"] or "permutation of the array A" in row["question"] or "interactive problem" in row["question"]:
        # Manual selection
        return True
    return False

def filter_first(all_parsed_tests: list[dict[str, Any]], data: Dataset) -> tuple[list[list[dict[str, str]]], list[tuple[int, int]]]:
    queries = []
    query_to_prob_test_idx = []
    for i, (test_dict, row) in enumerate(zip(all_parsed_tests, data)):
        for j, (inp, out) in enumerate(zip(test_dict["inputs"], test_dict["outputs"])):
            if test_dict.get("fn_name", None) is None:
                string_inp = str(inp)
                string_out = str(out)
                
                delimed_q = get_delimed_string(row["question"])
                problem_add = ""
                if get_delimed_string(string_inp) not in delimed_q or get_delimed_string(string_out) not in delimed_q:
                    continue
            
            else:
                assert isinstance(inp, list)
                string_inp = test_dict["fn_name"] + '(' + fn_arg_join(inp) + ')'
                string_out = repr(out)

                problem_add = f"\n\nTests are called by running `{row['starter_code'].split(':')[0].split('def ')[1]}` and the output is the return value. Do not worry about how the function call is structured when outputting your response; only consider the content of the test."
            
                if len(string_inp) > 700 or len(string_out) > 700:
                    continue

            first_p = get_prompt(row["question"] + problem_add, string_inp, string_out)
            queries.append(first_p)
            query_to_prob_test_idx.append((i, j))
    return queries, query_to_prob_test_idx

def filter_second(all_parsed_tests: list[dict[str, Any]], data: Dataset) -> tuple[list[list[dict[str, str]]], list[tuple[int, int]]]:
    queries = []
    query_to_prob_test_idx = []
    for i, (test_dict, row) in enumerate(zip(all_parsed_tests, data)):
        for j, (inp, out) in enumerate(zip(test_dict["inputs"], test_dict["outputs"])):
            if test_dict.get("fn_name", None) is None:
                string_inp = str(inp)
                string_out = str(out)
                problem_add = "\n\nThe problem statement does not have to exactly match the test, but it must be evaluated as CORRECT given the test and the instructions in the problem. For example, floating point outputs that are within the accepted precision should be marked as yes, included in the problem statement."
                if len(string_inp) > 500 or len(string_out) > 500:
                    continue
            else:
                assert isinstance(inp, list)
                string_inp = test_dict["fn_name"] + '(' + fn_arg_join(inp) + ')'
                string_out = repr(out)

                problem_add = f"\n\nNote: tests are ran by calling `{row['starter_code'].split(':')[0].split('def ')[1]}` and the output is the return value. Do not worry about how the function call is structured when outputting your response; only consider the content within the test."
            
                if len(string_inp) > 500 or len(string_out) > 500:
                    continue

            first_p = get_prompt(row["question"] + problem_add, string_inp, string_out)
            queries.append(first_p)
            query_to_prob_test_idx.append((i, j))
    return queries, query_to_prob_test_idx

def get_publics(all_parsed_tests: list[dict[str, Any]], querier: LLMQuerier, queries: list[list[dict[str, str]]], query_to_prob_test_idx: list[tuple[int, int]], args: argparse.Namespace, model: str = "gpt-4o-mini") -> list[dict[str, Any]]:
    outputs = querier.generate(model, queries, temperature=0, top_p=0.9, timeout=args.timeout)
    
    new_queries = []
    new_idx_to_old_idxs = []
    ultimate_outputs = [None for _ in range(len(queries))]
    for i, (existing_query, out) in enumerate(zip(queries, outputs)):
        if out != OpenAIClient.TIMEOUT_FLAG:
            new_queries.append(existing_query + [{"role": "assistant", "content": out}, {"role": "user", "content": SECONDARY_PROMPT}])
            new_idx_to_old_idxs.append(i)
        else:
            ultimate_outputs[i] = "No."

    new_outputs = querier.generate(model, new_queries, temperature=0, top_p=0.9)

    for old_idx, new_out in zip(new_idx_to_old_idxs, new_outputs):
        ultimate_outputs[old_idx] = new_out
    assert all(ultimate_output is not None for ultimate_output in ultimate_outputs)

    all_public_tests = []
    for parsed_tests in all_parsed_tests:
        all_public_tests.append({"inputs": [], "outputs": []})
        if parsed_tests.get("fn_name", None) is not None:
            all_public_tests[-1]["fn_name"] = parsed_tests["fn_name"]

    for (orig_prob_idx, orig_test_idx), output in zip(query_to_prob_test_idx, ultimate_outputs):
        if output != "Yes." and output != "No.":
            print(f"Warning: {output} on {orig_prob_idx, orig_test_idx}")
            output = "No."

        if output.strip() == "Yes." or output.strip() == "Yes":
            all_public_tests[orig_prob_idx]["inputs"].append(all_parsed_tests[orig_prob_idx]["inputs"][orig_test_idx])
            all_public_tests[orig_prob_idx]["outputs"].append(all_parsed_tests[orig_prob_idx]["outputs"][orig_test_idx])
    
    return all_public_tests


def main(args: argparse.Namespace):
    querier = LLMQuerier(None, args.cache_file, batch_size=4096)
    full_data = load_dataset(args.dataset)

    splits = []
    if args.split == "both":
        splits = ["train", "test"]
    elif args.split == "train":
        splits = ["train"]
    else:
        splits = ["test"]

    new_dd = {}

    for split in splits:
        data = full_data[split]
        data = data.map(lambda row, idx: {"question": filter_question(row["question"])}, with_indices=True)

        to_keep = []
        for i, row in enumerate(data):
            if not is_multi_solution(row):
                to_keep.append(i)
        data = data.select(to_keep)

        all_tests = reduce_tests(data)
        all_parsed_tests = []
        for tests in all_tests:
            test_dict = json.loads(tests)
            all_parsed_tests.append(test_dict)

        queries, query_to_prob_test_idx = filter_first(all_parsed_tests, data)
        print(f"{len(queries)} queries.")

        all_public_tests = get_publics(all_parsed_tests, querier, queries, query_to_prob_test_idx, args)

        zero_public_problem_idx = [i for i, public_tests in enumerate(all_public_tests) if len(public_tests["inputs"]) == 0]
        filt_all_parsed_tests = [parsed_tests for i, parsed_tests in enumerate(all_parsed_tests) if i in zero_public_problem_idx]
        filt_data = data.select(zero_public_problem_idx)

        queries, query_to_prob_test_idx = filter_second(filt_all_parsed_tests, filt_data)
        print(f"{len(queries)} queries.")
        filt_all_public_tests = get_publics(filt_all_parsed_tests, querier, queries, query_to_prob_test_idx, args, model="gpt-4o")
        for orig_idx, public_tests in zip(zero_public_problem_idx, filt_all_public_tests):
            all_public_tests[orig_idx] = public_tests

        new_data = {}
        for feat in data.features:
            new_data[feat] = data[feat]
        new_data["input_output"] = all_tests
        new_data["public_input_output"] = [json.dumps(public_test) for public_test in all_public_tests]

        new_dd[split] = data.map(lambda row, idx: {"input_output": all_tests[idx], "public_input_output": json.dumps(all_public_tests[idx])}, with_indices=True)
    
    print(f"costed: ${querier.current_price}")
    new_dd = DatasetDict(new_dd)
    new_dd.push_to_hub(args.output, private=True)
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset and output paths.")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the input Huggingface dataset')
    parser.add_argument('--output', type=str, required=True, help='Path to the output Huggingface dataset')
    parser.add_argument('--cache-file', type=str, default="caches/add_public_tests_cache.json", help='Path to the cache file')
    parser.add_argument('--timeout', type=float, default=25, help='Timeout (s) per query. (If tests are repetitive, queries can take 5 minutes.)')
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "both"],
        required=True,
        help="Split of the dataset to evaluate/generate from"
    )
    args = parser.parse_args()

    main(args)
