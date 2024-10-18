from typing import Optional

from search.prompts.simple_prompts import LCB_IO_FEWSHOT, LCB_FN_FEWSHOT

SYSTEM_PROMPT_TRANSLATE = ("You are an expert Python programmer. " +
                 "You will be given a question (problem specification) and a Python " +
                 "program which matches the specification. You will return a " + 
                 "high-level, natural language description of how the code works, " +
                 "like an editorial. You will NOT return any code.")

SYSTEM_PROMPT_GENERATE = ("You are an expert Python programmer. " +
                 "You will be given a question (problem specification) and a natural language " +
                 "solution/tutorial that describes how to solve the problem. You will " +
                 "generate a correct Python program that matches said specification and tutorial " +
                 "and passes all tests. You will NOT return anything except for " +
                 "the program inside markdown codeblocks")


def translate_code_solution(question: str, code_solution: str, has_starter_code: bool, num_words: Optional[int] = None) -> str:
    assert num_words is None or isinstance(num_words, int), "num_words must be an integer"
    assert num_words is None or num_words > 0, "num_words must be positive"

    og_q = question.replace('"""', r'\"""')
    if not has_starter_code: 
        shots_arr = LCB_IO_FEWSHOT
    else:
        shots_arr = LCB_FN_FEWSHOT

    out_str = f"You will given a competitive programming problem and correct Python code that solves it; please output a high-level description of what the code does, in natural language. Below are examples:\n\n\n"

    for shot in shots_arr:
        out_str += f"Example input:\n\n{shot['question']}\n\nCorrect code:\n\n```\n{shot['code']}\n```\n\nExample output:\n\n"
        out_str += f"{shot['cot']}\n\n\n"

    out_str += f"Here is the competitive programming problem:\n\n{og_q}\n\n"
    out_str += f"Correct code:\n\n```\n{code_solution}\n```\n\n"
    if num_words is None:
        out_str += "Output a natural language translation of the Python program solving the problem above."
    else:
        out_str += f"Output a natural language translation of the Python program solving the problem above, summarized in roughly {num_words} words."

    return out_str


# Adapted from CodeRM's prompting code
def generate_code_sol(question: str, nL_solution: str, starter_code: str = "") -> str:
    og_q = question.replace('"""', r'\"""')
    if starter_code == "" or starter_code is None:
        shots_arr = LCB_IO_FEWSHOT
    else:
        shots_arr = LCB_FN_FEWSHOT

    out_str = ("You will given a competitive programming problem and a natural language " +
               "description for how to solve it; please output correct Python code that " +
               "follows the description and solves the problem. Below are examples:\n\n\n")

    for shot in shots_arr:
        out_str += f"Example input:\n\n{shot['question']}\n\nNatural language tutorial:\n\n{shot['cot']}\n\nExample output:\n\n"
        out_str += f"```python\n{shot['code']}\n```\n\n\n"

    out_str += f"Here is the competitive programming problem:\n\n{og_q}\n\n"
    out_str += f"Natural language tutorial:\n\n{nL_solution}\n\n"
    out_str += "Please write a Python program that solves this problem."

    if not (starter_code == "" or starter_code is None):
        out_str += "\n\nYour solution should also utilize the following starter code:\n\n```python\n" + starter_code + "\n```"
    else:
        out_str += "\n\nYour solution should use standard input/output and will be run directly without any edits."

    return out_str
