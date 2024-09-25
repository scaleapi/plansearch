from search.prompts.simple_prompts import LCB_IO_FEWSHOT, LCB_FN_FEWSHOT
from search.prompts.idea_prompts import SYSTEM_PROMPT_TRANSLATE, get_nl_solution

SYSTEM_PROMPT_PSEUDOCODE = ("You are an expert Python programmer. " +
                 "You will be given a question (problem specification) and a natural language " +
                 "solution/tutorial that describes how to solve the problem. You will " +
                 "generate correct pseudocode that follows the above solution exactly. " +
                 "You will NOT return anything for the pseudocode inside Markdown codeblocks."
)

SYSTEM_PROMPT_GENERATE = ("You are an expert Python programmer. " +
                 "You will be given a question (problem specification) and high-level pseudocode " +
                 "that describes how to solve the problem. You will " +
                 "generate a correct Python program that matches said pseudocode " +
                 "and passes all tests. You will NOT return anything except for " +
                 "the program inside Markdown codeblocks.")


def get_pseudocode(question: str, nl_solution) -> str:
    og_q = question.replace('"""', r'\"""')

    out_str = f"Here is the competitive programming problem\n\n{og_q}\n\n"

    out_str += f"Natural language tutorial:\n\n{nl_solution}\n\n"

    out_str += ("Carefully write pseudocode that follows the above solution EXACTLY and solves the problem. " +
                "The pseudocode does not need to go into low-level details, and should be high-level.")
    
    return out_str


# Adapted from CodeRM's prompting code
def generate_code_sol(question: str, pseudocode: str, starter_code: str = "") -> str:
    og_q = question.replace('"""', r'\"""')
    if starter_code == "" or starter_code is None:
        shots_arr = LCB_IO_FEWSHOT
    else:
        shots_arr = LCB_FN_FEWSHOT

    out_str = ("You will given a competitive programming problem and high-level pseudocode " +
               "for the solution to the problem; please output correct Python code that " +
               "follows said pseudocode, while solving the problem. Below are examples:\n\n\n")

    for shot in shots_arr:
        out_str += f"Example input:\n\n{shot['question']}\n\nExample code output:\n\n"
        out_str += f"```python\n{shot['code']}\n```\n\n\n"

    out_str += f"Here is the competitive programming problem:\n\n{og_q}\n\n"
    out_str += f"Associated pseudocode:\n\n{pseudocode}\n\n"
    out_str += "Please write a Python program that follows the pseudocode and solves the competitive programming problem."

    if not (starter_code == "" or starter_code is None):
        out_str += "\n\nYour solution should also utilize the following starter code:\n\n```python\n" + starter_code + "\n```"

    return out_str

def get_pseudocode_prompt(problem_str: str, nl_solution: str) -> list[dict[str, str]]:
    convo = [{"role": "system", "content": SYSTEM_PROMPT_PSEUDOCODE},
                {"role": "user", "content": get_pseudocode(problem_str, nl_solution)}]
    return convo

def pseudocode_to_code_solution_prompt(problem_str: str, starter_code: str, pseudocode: str) -> list[dict[str, str]]:
    convo = [{"role": "system", "content": SYSTEM_PROMPT_GENERATE},
                {"role": "user", "content": generate_code_sol(problem_str, pseudocode, starter_code)}]
    return convo

