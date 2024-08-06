from search.prompts.simple_prompts import LCB_IO_FEWSHOT, LCB_FN_FEWSHOT
from search.prompts.idea_prompts import generate_code_sol, SYSTEM_PROMPT_GENERATE

SYSTEM_PROMPT_OBSERVATION = ("You are an expert Python programmer and programming instructor. " +
                             "You will be given an algorithmic question (problem specification). " +
                             "You will return a high-level, list of observations in natural " +
                             "language that are may be helpful to solving the question. However, " +
                             "you will NOT give progress in solving the problem, and you will NOT return any code.")

SYSTEM_PROMPT_TRANSLATE = ("You are an expert Python programmer. " +
                 "You will be given an algorithmic question (problem specification) and some observations" +
                 "which may be helpful in solving the problem. You will return a high-level, natural " +
                 "language solution to the question, " +
                 "like an editorial. You will NOT return any code.")

def get_observation(question: str) -> str:
    og_q = question.replace('"""', r'\"""')

    out_str = f"Here is the competitive programming problem\n\n{og_q}\n\n"

    out_str += ("Please output a few helpful observations that may solve the problem. " +
                "Do not make significant progress in solving the problem, but output observations that may help.")
    
    return out_str

def get_nl_solution(question: str, has_starter_code: bool, observations: str) -> str:
    og_q = question.replace('"""', r'\"""')
    out_str = ""
    # if not has_starter_code: 
    #     shots_arr = LCB_IO_FEWSHOT
    # else:
    #     shots_arr = LCB_FN_FEWSHOT

    # out_str += f"You will given a competitive programming problem; please output a high-level description of how to solve the problem in natural language. Below are examples:\n\n\n"

    # for shot in shots_arr:
    #     out_str += f"Example input:\n\n{shot['question']}\n\nExample output:\n\n"
    #     out_str += f"{shot['cot']}\n\n\n"

    out_str += f"Here is the competitive programming problem:\n\n{og_q}\n\n"
    out_str += f"Here are some observations that may help you solve the problem:\n\n{observations}\n\n"
    out_str += "Output a high-level, natural language solution to the problem above."

    return out_str
