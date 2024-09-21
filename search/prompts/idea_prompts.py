from typing import Optional

from search.base_classes import Problem
from search.prompts.simple_prompts import LCB_IO_FEWSHOT, LCB_FN_FEWSHOT
from search.prompts.backtranslate_prompts import generate_code_sol, SYSTEM_PROMPT_GENERATE

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

SYSTEM_PROMPT_CRITIC = (
    "You are an expert Python programmer and competitive programming coach. "
    "You will be given a competitive programming question (problem specification) "
    "and a proposed idea to solve the problem by a student. "
    "You will provide a criticism of said idea and a simple counter-example which would cause "
    "the proposed idea to fail. "
    "You MUST NOT output any solution code."
)
def get_criticism_from_nl_sol(problem_str: str, nl_solution: str) -> str:
    prompt = f"Here is the competitive programming problem:\n\n{problem_str}\n\n"
    prompt += f"Here is a proposed natural language solution:\n\n{nl_solution}\n\n"
    prompt += (
            "The above solution may not be correct. Construct a criticism of this solution. "
            "Remember that this should be a solution for a competitive programming setting, so the solution must be absolutely correct, "
            "and issues commonly found in the real-world may not apply.\n\n"
            "Thus, also give a counter-example (an input) to the solution which causes this solution to fail.\n"
               )
    return prompt

def get_criticism_prompt(problem: Problem, nl_solution: str) -> tuple[dict[str, str]]:
    convo = ({"role": "system", "content": SYSTEM_PROMPT_CRITIC},
                {"role": "user", "content": get_criticism_from_nl_sol(problem.problem_str, nl_solution)})
    return convo
 
FIX_CRITICISM_PROMPT = "Given your criticisms, how would you fix the proposed idea? DO NOT output any code."

SYSTEM_PROMPT_MERGE_FIXES = ("You are an expert Python programmer. "
                             "You will be given a competitive programming question (problem specification). "
                             "A student has come up with a proposed idea to solve the problem, but it is incorrect. "
                             "You will also be given correct fixes to the idea. "
                             "Incorporate the fixes into the original idea to make it correct. "
                             "MAKE SURE not to output any code."
                             )
def get_merge_orig_fix(problem_str: str, nl_solution: str, fixes: str) -> str:
    prompt = f"Here is the competitive programming problem:\n\n{problem_str}\n\n"
    prompt += f"Here is the proposed natural language solution:\n\n{nl_solution}\n\n"
    prompt += f"Unfortunately, the above solution is not entirely correct. Thus, there have been criticisms and fixes as such:\n\n{fixes}\n\n"
    prompt += "Fix the original solution using the fixes as described above. YOU MUST FOLLOW THE FIXES EXACTLY, EVEN IF THEY ARE NOT INTUITIVELY CORRECT. DO NOT output code."
    return prompt

