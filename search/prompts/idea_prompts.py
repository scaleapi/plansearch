from typing import Optional

from prompts.simple_prompts import LCB_IO_FEWSHOT, LCB_FN_FEWSHOT
from prompts.backtranslate_prompts import generate_code_sol, SYSTEM_PROMPT_GENERATE

SYSTEM_PROMPT_TRANSLATE = ("You are an expert Python programmer. " +
                 "You will be given an algorithmic question (problem specification). " +
                 " You will return a high-level, natural language solution to the question, " +
                 "like an editorial. You will NOT return any code. Be as creative as possible, " +
                 "going beyond what you think is intuitively correct.")

def get_nl_solution(question: str, has_starter_code: bool, use_few_shot: bool, num_words: Optional[int] = None) -> str:
    assert num_words is None or isinstance(num_words, int), "num_words must be an integer"
    assert num_words is None or num_words > 0, "num_words must be positive"

    og_q = question.replace('"""', r'\"""')

    out_str = ""
    if use_few_shot:
        if not has_starter_code: 
            shots_arr = LCB_IO_FEWSHOT
        else:
            shots_arr = LCB_FN_FEWSHOT

        out_str += f"You will given a competitive programming problem; please output a high-level description of how to solve the problem in natural language. Below are examples:\n\n\n"

        for shot in shots_arr:
            out_str += f"Example input:\n\n{shot['question']}\n\nExample output:\n\n"
            out_str += f"{shot['cot']}\n\n\n"

    out_str += f"Here is the competitive programming problem:\n\n{og_q}\n\n"
    if num_words is None:
        out_str += "Brainstorm a high-level, natural language solution to the problem above. "
    else:
        out_str += f"Brainstorm a high-level, natural language solution to the problem above, summarized in roughly {num_words} words. "

    out_str += ("Note that your intuition may lead you astray, so come up with simple, creative ideas " +
                "that go beyond what you would usually come up with and go beyond your narrow intuition. " +
                "Brainstorming solutions that do not seem intuitively correct IS CRUCIAL.")

    return out_str
