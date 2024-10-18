import numpy as np

from typing import Optional, Union, Any
import argparse
import random
import os
from datetime import datetime

from search.base_classes import Problem
from search.queriers import LLMQuerier
from search.query_clients import OpenAIClient
from search.python_utils import batch_map_on_nested_list, log_to_dir, map_nary_fn_on_nested_list, merge_nested_lists
from coderm.utils import gunzip_json_read


SYSTEM_PROMPT_IDEA = (
    "You are an expert Python programmer. "
    "You will be given a question (problem specification) and a Python "
    "program which matches the specification. You will return a "
    "high-level, natural language description of how the code works and the idea behind it, "
    "like an editorial. You will NOT return any code."
)

SYSTEM_PROMPT_YN = (
    "You are an expert Python programmer. You will be given a competitive programming problem "
    "and two pieces of code which are attempts to solve the problem. For your convenience, you will also be given "
    "the idea for each code, summarized in natural language. You will be asked to answer whether the ideas behind the code are the same. "
    "You must ONLY output 'Yes.' or 'No.'"
)

def get_summarize_prompt(problem_str: str, code: str) -> tuple[dict[str, str]]:
    return (
        {"role": "system",
         "content": SYSTEM_PROMPT_IDEA},
        {"role": "user",
         "content": (
             f"Here is the competitive programming question:\n\n{problem_str}\n\n"
             f"Here is an attempted solution to the question:\n\n```python\n{code}\n```\n\n"
             "Provide a high-level, natural language description of how the code works and the idea behind it, like an editorial."
         )}
    )

def get_yes_no_prompt(problem_str: str, code1: str, code_idea1: str, code2: str, code_idea2: str) -> tuple[dict[str, str]]:
    return (
        {"role": "system",
         "content": SYSTEM_PROMPT_YN},
        {"role": "user",
         "content": (
             f"Here is the competitive programming question:\n\n{problem_str}\n\n"
             f"Here is the first attempted solution, followed by the natural language description:\n\n"
             f"```python\n{code1}\n```\n\n{code_idea1}\n\n"
             f"Here is the second attempted solution, followed by the natural language description:\n\n"
             f"```python\n{code2}\n```\n\n{code_idea2}\n\n"
             "Using the natural language description of both codes, determine whether: yes, the codes share the same idea, or no, they do not share the same idea. "
             "You MUST answer with 'Yes.' or 'No.' ONLY."
         )}
    )

def is_similar(output: str) -> float:
    og_output = output
    output = output.lower()
    if "yes" in output:
        return 1.
    else:
        if "no" not in output:
            print(f"Warning: output is {og_output}")
        return 0.

def select_idea_codes(querier: LLMQuerier, model: str, log_directory: str, problem_strs: list[str], all_codes: list[list[str]], max_to_select: Optional[int], max_fill_attempts: int = 1) -> list[list[tuple[str, str]]]:
    all_selected_codes = [[] for _ in range(len(problem_strs))]
    all_code_ideas = [[] for _ in range(len(problem_strs))]
    start_at_idxs = [0] * len(problem_strs)

    for iter_num in range(max_fill_attempts):
        is_empty = True
        get_idea_logs = []
        code_idea_prompts = []
        temp_selected_codes = []

        for i, (problem_str, codes) in enumerate(zip(problem_strs, all_codes)):
            num_needed = max_to_select - len(all_code_ideas[i])
            if num_needed <= 0 or start_at_idxs[i] >= len(codes):
                get_idea_logs.append([])
                code_idea_prompts.append([])
                temp_selected_codes.append([])
                continue
            end_idx = start_at_idxs[i] + num_needed
            codes_to_use = codes[start_at_idxs[i]:end_idx]

            code_idea_prompts.append([get_summarize_prompt(problem_str, code) for code in codes_to_use])
            temp_selected_codes.append(codes_to_use)

            get_idea_logs.append([{"problem": problem_str, "code": code, "prompt": prompt} for code, prompt in zip(codes_to_use, code_idea_prompts[-1])])

            start_at_idxs[i] = end_idx
            is_empty = False

        if is_empty:
            assert iter_num != 0
            break

        temp_code_ideas: list[list[str]] = batch_map_on_nested_list(code_idea_prompts, lambda li: querier.generate(model, li, temperature=0, top_p=0.9, timeout=120))
        map_nary_fn_on_nested_list(lambda log_dict, output: log_dict.update({"idea": output}), get_idea_logs, temp_code_ideas)
        log_to_dir(log_directory, {f"get_ideas_{iter_num}.json": get_idea_logs,})

        for i, (temp_codes, temp_ideas) in enumerate(zip(temp_selected_codes, temp_code_ideas)):
            for code, idea in zip(temp_codes, temp_ideas):
                if idea == OpenAIClient.TIMEOUT_FLAG or idea == OpenAIClient.BAD_REQUEST_FLAG:
                    continue
                else:
                    all_code_ideas[i].append(idea)
                    all_selected_codes[i].append(code)

        assert all(len(codes) == len(ideas) for codes, ideas in zip(all_selected_codes, all_code_ideas))
        assert all(len(codes) <= start_idx for codes, start_idx in zip(all_selected_codes, start_at_idxs))
 
    return merge_nested_lists(all_code_ideas, all_selected_codes)

def self_cross_prompts(problem_str: str, idea_codes: list[tuple[str, str]]) -> tuple[list[list[dict[str, str]]], list[dict[str, Any]]]:
    log_list = []
    prompts = []

    for i in range(len(idea_codes)):
        for j in range(i+1, len(idea_codes)):
            idea1, code1 = idea_codes[i]
            idea2, code2 = idea_codes[j]

            if code1 == "" or code2 == "":
                continue

            prompt = get_yes_no_prompt(problem_str, code1, idea1, code2, idea2)
            prompts.append(prompt)

            log_list.append({
                "problem": problem_str,
                "code1": code1,
                "code_idea1": idea1,
                "code2": code2,
                "code_idea2": idea2,
                "prompt": prompt
            })

    return prompts, log_list

def cross_code_prompts(problem_str: str, idea_codes1: list[tuple[str, str]], idea_codes2: list[tuple[str, str]]) -> tuple[list[list[dict[str, str]]], list[dict[str, Any]]]:
    log_list = []
    prompts = []

    for idea1, code1 in idea_codes1:
        for idea2, code2 in idea_codes2:
            if code1 == "" or code2 == "":
                continue

            prompt = get_yes_no_prompt(problem_str, code1, idea1, code2, idea2)
            prompts.append(prompt)

            log_list.append({
                "problem": problem_str,
                "code1": code1,
                "code_idea1": idea1,
                "code2": code2,
                "code_idea2": idea2,
                "prompt": prompt
            })

    return prompts, log_list


def main(args: argparse.Namespace):
    random.seed(args.seed)

    output_gunzip = gunzip_json_read(args.results_file)
    querier = LLMQuerier(log_directory=os.path.join(args.log_directory, "queries"), cache_file=args.cache_file, global_batch_size=12288)

    all_generated_codes = []
    problem_strs = []
    correct_flags = []

    for problem_items in output_gunzip["items"]:
        codes = [result["code"] for result in problem_items["results"]]
        correct = [result["passing"] for result in problem_items["results"]]
        # I don't shuffle here because we did sample before and I don't want to mess up cached
        code_idxs = random.sample(range(len(codes)), len(codes))

        codes = [codes[i] for i in code_idxs]
        all_generated_codes.append(codes)
        correct = [correct[i] for i in code_idxs]
        correct_flags.append(correct)

        problem_strs.append(problem_items["prompt"])

    yes_no_prompts = []
    scoring_logs = []
  
    if args.comparison_type == "default":
        selected_idea_codes = select_idea_codes(querier, args.model_config_path, args.log_directory,
                                                problem_strs, all_generated_codes, args.max_to_consider, args.max_fill_idea_attempts)

        for problem_str, idea_codes in zip(problem_strs, selected_idea_codes):
            prompts, logs = self_cross_prompts(problem_str, idea_codes)
            yes_no_prompts.append(prompts)
            scoring_logs.append(logs)
    elif args.comparison_type == "right-wrong":
        wrong_codes = [[code for flag, code in zip(flags, codes) if not flag]
                    for flags, codes in zip(correct_flags, all_generated_codes)]
        selected_wrong_idea_codes = select_idea_codes(querier, args.model_config_path, args.log_directory,
                                                problem_strs, wrong_codes, args.max_to_consider, args.max_fill_idea_attempts)

        right_codes = [[code for flag, code in zip(flags, codes) if flag]
                    for flags, codes in zip(correct_flags, all_generated_codes)]
        selected_right_idea_codes = select_idea_codes(querier, args.model_config_path, args.log_directory,
                                                problem_strs, right_codes, args.max_to_consider, args.max_fill_idea_attempts)

        for problem_str, right_idea_codes, wrong_idea_codes in zip(problem_strs, selected_right_idea_codes, selected_wrong_idea_codes):
            prompts, logs = cross_code_prompts(problem_str, right_idea_codes, wrong_idea_codes)
            yes_no_prompts.append(prompts)
            scoring_logs.append(logs)

    elif args.comparison_type == "right":
        right_codes = [[code for flag, code in zip(flags, codes) if flag]
                    for flags, codes in zip(correct_flags, all_generated_codes)]
        selected_right_idea_codes = select_idea_codes(querier, args.model_config_path, args.log_directory,
                                                problem_strs, right_codes, args.max_to_consider, args.max_fill_idea_attempts)
        for problem_str, right_idea_codes in zip(problem_strs, selected_right_idea_codes):
            prompts, logs = self_cross_prompts(problem_str, right_idea_codes)
            yes_no_prompts.append(prompts)
            scoring_logs.append(logs)

    elif args.comparison_type == "wrong":
        wrong_codes = [[code for flag, code in zip(flags, codes) if not flag]
                    for flags, codes in zip(correct_flags, all_generated_codes)]
        selected_wrong_idea_codes = select_idea_codes(querier, args.model_config_path, args.log_directory,
                                                problem_strs, wrong_codes, args.max_to_consider, args.max_fill_idea_attempts)
        for problem_str, wrong_idea_codes in zip(problem_strs, selected_wrong_idea_codes):
            prompts, logs = self_cross_prompts(problem_str, wrong_idea_codes)
            yes_no_prompts.append(prompts)
            scoring_logs.append(logs)
       

    prompt_outputs = batch_map_on_nested_list(yes_no_prompts, lambda li: querier.generate(args.model_config_path, li, temperature=0, top_p=0.9, timeout=120))
    parsed_prompt_outputs = map_nary_fn_on_nested_list(is_similar, prompt_outputs)

    def temp(log_dict: dict[str], score: float):
        log_dict["output_score"] = score
    map_nary_fn_on_nested_list(temp, scoring_logs, parsed_prompt_outputs)

    log_to_dir(args.log_directory, {"scoring.json": scoring_logs})

    fractions = []

    for outputs_for_problem in parsed_prompt_outputs:
        total_similar = 0
        possible = len(outputs_for_problem) 

        for output in outputs_for_problem:
            total_similar += output

        print(f"TOTAL SIMILAR: {total_similar} out of {possible} possible")
        if possible != 0:
            fractions.append(total_similar / possible)
        else: 
            print(f"Omitting since only {len(outputs_for_problem)} completions.")

    np.save(os.path.join(args.log_directory, "results.npy"), np.array(fractions))
    print(fractions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check similarity of code solutions.")
    parser.add_argument("--model-config-path", type=str, required=True, help="Path to the model JSON configuration.")
    parser.add_argument("--results-file", type=str, required=True, help="Path to the results file for gunzip_json_read.")
    parser.add_argument("--comparison-type", type=str, choices=["default", "right-wrong", "right", "wrong"], default="default", help="Whether to partition along correct vs wrong")

    parser.add_argument("--log-directory", type=str, default=None, help="Directory to log to. Will default-fill to other_logs/similar_logs/...")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--max-to-consider", type=int, default=25, help="Maximum number of codes to consider.")
    parser.add_argument("--max-fill-idea-attempts", type=int, default=3, help="Maximum times to fill code ideas if missing.")
    parser.add_argument("--cache-file", type=str, default="caches/check_similar_cache.json", help="Cache file to use for queries")

    args = parser.parse_args()

    if args.log_directory is None:
        current_time = datetime.now().strftime("%m-%dT%H:%M:%S")
        args.log_directory = f"other_logs/similar_logs/logs_{current_time}"

    log_to_dir(args.log_directory, {"args.json": vars(args)})

    main(args)