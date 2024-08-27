import numpy as np

import argparse
import random
import os
from datetime import datetime

from search.queriers import LLMQuerier
from search.query_clients import OpenAIClient
from search.python_utils import batch_map_on_nested_list, log_to_dir, map_nary_fn_on_nested_list
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


def main(args: argparse.Namespace):
    random.seed(args.seed)

    output_gunzip = gunzip_json_read(args.results_file)
    querier = LLMQuerier(log_directory=os.path.join(args.log_directory, "queries"), cache_file=args.cache_file, global_batch_size=12288)

    code_idea_prompts = []
    all_generated_codes = []
    problem_strs = []

    for problem_items in output_gunzip["items"]:
        codes = [result["code"] for result in problem_items["results"]]
        # I don't shuffle here because we did sample before and I don't want to mess up cached
        codes = random.sample(codes, len(codes))
        all_generated_codes.append(codes)
        problem_strs.append(problem_items["prompt"])


    all_selected_codes = [[] for _ in range(len(problem_strs))]
    all_code_ideas = [[] for _ in range(len(problem_strs))]
    start_at_idxs = [0] * len(problem_strs)

    for iter_num in range(args.max_fill_idea_attempts):
        is_empty = True
        get_idea_logs = []
        code_idea_prompts = []
        temp_selected_codes = []

        logs = []
        for i, (problem_str, codes) in enumerate(zip(problem_strs, all_generated_codes)):
            num_needed = args.max_to_consider - len(all_code_ideas[i])
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

        temp_code_ideas: list[list[str]] = batch_map_on_nested_list(code_idea_prompts, lambda li: querier.generate(args.model_config_path, li, temperature=0, top_p=0.9, timeout=120))
        map_nary_fn_on_nested_list(lambda log_dict, output: log_dict.update({"idea": output}), get_idea_logs, temp_code_ideas)
        log_to_dir(args.log_directory, {f"get_ideas_{iter_num}.json": get_idea_logs,})

        for i, (temp_codes, temp_ideas) in enumerate(zip(temp_selected_codes, temp_code_ideas)):
            for code, idea in zip(temp_codes, temp_ideas):
                if idea == OpenAIClient.TIMEOUT_FLAG or idea == OpenAIClient.BAD_REQUEST_FLAG:
                    continue
                else:
                    all_code_ideas[i].append(idea)
                    all_selected_codes[i].append(code)

        assert all(len(codes) == len(ideas) for codes, ideas in zip(all_selected_codes, all_code_ideas))
        assert all(len(codes) <= start_idx for codes, start_idx in zip(all_selected_codes, start_at_idxs))
    
    yes_no_prompts = []
    scoring_logs = []
    for problem_str, codes, code_ideas in zip(problem_strs, all_selected_codes, all_code_ideas):
        prompts = []
        logs = []

        assert len(codes) == len(code_ideas)
        for i in range(len(codes)):
            for j in range(i + 1, len(codes)):
                if codes[i] == "" or codes[j] == "":
                    continue

                prompt = get_yes_no_prompt(problem_str, codes[i], code_ideas[i], codes[j], code_ideas[j])
                prompts.append(prompt)

                logs.append({
                    "problem": problem_str,
                    "code1": codes[i],
                    "code_idea1": code_ideas[i],
                    "code2": codes[j],
                    "code_idea2": code_ideas[j],
                    "prompt": prompt
                })

        yes_no_prompts.append(prompts)
        scoring_logs.append(logs)


    prompt_outputs = batch_map_on_nested_list(yes_no_prompts, lambda li: querier.generate(args.model_config_path, li, temperature=0, top_p=0.9, timeout=120))
    parsed_prompt_outputs = map_nary_fn_on_nested_list(is_similar, prompt_outputs)

    def temp(log_dict: dict[str], score: float):
        log_dict["output_score"] = score
    map_nary_fn_on_nested_list(temp, scoring_logs, parsed_prompt_outputs)

    log_to_dir(args.log_directory, {"get_ideas.json": get_idea_logs, "scoring.json": scoring_logs})

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