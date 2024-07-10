from prompts.simple_prompts import LCB_IO_FEWSHOT, LCB_FN_FEWSHOT
from prompts.backtranslate_prompts import generate_code_sol, SYSTEM_PROMPT_GENERATE

SYSTEM_PROMPT_TRANSLATE = ("You are an expert Python programmer. " +
                 "You will be given an algorithmic question (problem specification). " +
                 " You will return a high-level, natural language solution to the question, " +
                 "like an editorial. You will NOT return any code.")

def get_nl_solution(question: str, has_starter_code: bool) -> str:
    og_q = question.replace('"""', r'\"""')
    if not has_starter_code: 
        shots_arr = LCB_IO_FEWSHOT
    else:
        shots_arr = LCB_FN_FEWSHOT

    out_str = f"You will given a competitive programming problem; please output a high-level description of how to solve the problem in natural language. Below are examples:\n\n\n"

    for shot in shots_arr:
        out_str += f"Example input:\n\n{shot['question']}\n\nExample output:\n\n"
        out_str += f"{shot['cot']}\n\n\n"

    out_str += f"Here is the competitive programming problem:\n\n{og_q}\n\n"
    out_str += "Output a high-level, natural language solution to the problem above."

    return out_str
