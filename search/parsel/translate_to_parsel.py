import os
from parsel.parsel import parsel_graph
import argparse
from prompts.parsel_prompts import (
    SOLUTION_PROMPT_FOR_COMPLETION,
    QUESTION_PREFIX,
    QUESTION_SUFFIX,
    SOLUTION_PROMPT_FOR_CHAT,
    TRANSLATION_PROMPT_FOR_COMPLETION,
    SHORTER_TRANSLATION_FOR_CHAT,
    translation_prompt_for_chat
)
from typing import Any, Union, Optional, Generator

from python_utils import (
    log_to_dir,
    safe_iter,
)
from parsing_utils import extract_code
import traceback
from copy import deepcopy
from fn import get_all_descendant_impls, Function
from base_classes import Test
from parsel.construct_graph import get_root, get_graph

from base_classes import Problem
from queriers import LLMQuerier, MODELS_TO_METHOD


def compute_parsel_output(problem: Problem, args: argparse.Namespace, querier: LLMQuerier, subdirectory: str) -> Optional[tuple[str, str]]:
    if args.text_model_name in MODELS_TO_METHOD[args.text_model_name] and MODELS_TO_METHOD[args.text_model_name] == "completions":
        raise NotImplementedError("not yet integrated")
        solution_prompt = QUESTION_PREFIX + problem.problem_str + SOLUTION_PROMPT_FOR_COMPLETION
        stop = ['"""']
    else:
        solution_prompt = QUESTION_PREFIX + problem.problem_str + SOLUTION_PROMPT_FOR_CHAT

    log_to_dir(subdirectory, {"problem.txt": problem.problem_str})
    querier.set_log_directory(os.path.join(subdirectory, "queries"))
    sketch_solutions = querier.generate(
        model=args.text_model_name,
        prompts=[solution_prompt] * args.num_solutions_to_generate,
        max_tokens=3500,
        temperature=0.6,
        log_name="solution",
    )

    for i, sketch_solution in enumerate(sketch_solutions):
        parsel_solution = translation_prompt_for_chat(solution_text=sketch_solution.lstrip(), has_starter_code=problem.has_starter_code(), starter_code_fn=problem.get_starter_code_fn())
        translated_solutions = querier.generate(
            model=args.gen_parsel_model_name,
            prompts=[QUESTION_PREFIX + problem.problem_str + parsel_solution] * args.num_translation_attempts,
            max_tokens=3500,
            temperature=0.2,
            presence_penalty=0.1,
            log_name=f"parsel_{i}",
        )
        translated_solutions = extract_code(
            translated_solutions,
            return_orig=False,
            add_markers=[('"""', '"""')],
        )

        for j, parsel_code in enumerate(translated_solutions):
            parsel_code_splitline = parsel_code.splitlines()
            if any(line.startswith("# ") for line in parsel_code_splitline):
                continue
            if len("".join(parsel_code_splitline).strip()) == 0:
                continue

            # Remove blank lines
            parsel_code_splitline = [line for line in parsel_code_splitline if line.strip() != ""]
            new_parsel = [parsel_code_splitline[0]]
            for line in parsel_code_splitline[1:]:
                if line.lstrip() != line:
                    new_parsel.append(line)
                else:
                    break
            parsel_code = "\n".join(new_parsel)

            try:
                root, defined_fns = get_graph(parsel_code)
            except Exception as e:
                print("error (before parsel):")
                traceback.print_exc()
                continue

            if len(defined_fns) > 7:
                continue
            else:
                print(f"Translation Attempt number {j}")
                print(f"Implementing {len(defined_fns)} functions")

            root.tests = deepcopy(problem.public_tests)
            for test in root.tests:
                test.switch_fn_name(root.name)

            for fn in defined_fns.values():
                fn.prefix_for_prompts = QUESTION_PREFIX + problem.problem_str + QUESTION_SUFFIX
                fn.problem_context = problem.problem_str

            specific_parsel_path = os.path.join(
                subdirectory,
                f"translation-{i}_parsel-{j}",
            )

            log_to_dir(
                specific_parsel_path,
                {
                    "natlang_solution.txt": sketch_solution,
                    "parsel_solution.txt": parsel_code,
                },
            )

            querier.set_log_directory(os.path.join(specific_parsel_path, "queries"))

            _, status = parsel_graph(
                defined_fns=defined_fns,
                querier=querier,
                args=args,
                log_directory=specific_parsel_path,
            )

            if not status:
                continue

            root_name = get_root(defined_fns)
            assert root_name == root.name
            return root_name, get_all_descendant_impls(
                [defined_fns[root_name]],
            ) + defined_fns[root_name].fixed_implementation + '\n'

    return None
