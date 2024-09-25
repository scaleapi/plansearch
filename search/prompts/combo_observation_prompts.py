from search.prompts.pseudocode_prompts import (
    get_pseudocode_prompt,
    pseudocode_to_code_solution_prompt, 
    LCB_FN_FEWSHOT,
    LCB_IO_FEWSHOT
)

from search.prompts.idea_prompts import (
    SYSTEM_PROMPT_CRITIC, 
    FIX_CRITICISM_PROMPT, 
    SYSTEM_PROMPT_MERGE_FIXES, 
    get_criticism_from_nl_sol, 
    get_merge_orig_fix, 
    get_criticism_prompt,
    nl_to_code_solution_prompt as idea_to_code_solution_prompt
)

NL = '\n'

SYSTEM_PROMPT_OBSERVATION = ("You are an expert Python programmer. " +
                             "You will be given an competitive programming question (problem specification). " +
                             "You will return several useful, non-obvious, and correct observations " +
                             "about the problem, like hints to solve the problem. You will NOT return any code." +
                             "Be as creative as possible, going beyond what you think is intuitively correct.")

SYSTEM_PROMPT_OBSERVATION2 = ("You are an expert Python programmer. " +
                             "You will be given an competitive programming question (problem specification) " +
                             "and several correct observations about the problem. "
                             "You will brainstorm several new, useful, and correct observations " +
                             "about the problem, derived from the given observations. You will NOT return any code. " +
                             "Be as creative as possible, going beyond what you think is intuitively correct.")


def get_observation(problem_str: str, num_observations: int) -> str:
    prompt = f"Here is the competitive programming problem:\n\n{problem_str}\n\n"
    prompt += (f"Brainstorm a list of {num_observations} non-obvious observations about " + 
              "properties of the problem, each a few words long. The observations should " +
              "be simple and creative observations which you normally would not think of, " + 
              "and are not intuitively correct.\n\n")
    
    prompt += """Reason before outputting each observation like so:\n
Observation 1:\n
A paragraph or two of reasoning\n
"your observation" (for example, 'Greedy approaches should be considered'. or 'DP can be useful in computing XYZ.')\n
Make sure the observations are useful in solving the problem, not just a restatement of the obvious."""

    return prompt

def combine_observations(problem_str: str, obs_combo: tuple[str]) -> str:
    prompt = f"Here is the competitive programming problem:\n\n{problem_str}\n\n"
    
    at_least_two = len(obs_combo) >= 2
    has_observation = len(obs_combo) >= 1

    if at_least_two:
        observation_str = "Here are several correct observations which will help in solving the problem:"
        observation_str += "- " + "\n\n- ".join(obs_combo) + "\n\n"
    elif has_observation:
        observation_str = "Here is a correct observation which will help in solving the problem:"
        observation_str += f"- {obs_combo[0]}\n\n"
    else:
        observation_str = "No observations are necessary to solve this problem.\n\n"

    prompt = prompt + observation_str + f"""First reason about implications of the {'observations' if has_observation else 'problem'} above. Include the most critical parts of the problem statement.

Then creatively {'combine the observations' if at_least_two else 'use the implications'} above to brainstorm more non-obvious observations. Before outputting EACH new observation, quote relevant parts of the implications EXACTLY.

Follow the rough format below:

## Implications:

[Quotes from critical parts of the problem statement]

{'[Quotes of the observations]' + NL * 2 if has_observation else ''}### Reasoned Implications:

[Your step-by-step reasoned-through implications{' of joining the observations above' if at_least_two else ('of the observation above' if has_observation else '')}, referencing the quotes above.]

## New Observations: 

### Observation 1:

[Quotes of relevant pieces of text from the Reasoned Implications section.]

[A paragraph or two of step-by-step reasoning.]

[Your new final observation]

### Observation 2:

...

Make sure the new observations are creative and useful in solving the problem, not just a restatement of the obvious. Also make sure that the new observations are derived from the implications{' of the old observations' if has_observation else ''}.
"""
    return prompt

def get_first_observations_prompt(problem_str: str, num_observations_to_generate: int) -> tuple[dict[str, str]]:
    convo = ({"role": "system", "content": SYSTEM_PROMPT_OBSERVATION},
                {"role": "user", "content": get_observation(problem_str, num_observations_to_generate)})
    return convo

def get_combined_observations_prompt(problem_str: str, observation_combo: tuple[str, ...]) -> tuple[dict[str, str]]:
    convo = ({"role": "system", "content": SYSTEM_PROMPT_OBSERVATION2},
                {"role": "user", "content": combine_observations(problem_str, observation_combo)})
    return convo
 

FORMAT_INTO_LIST_PROMPT = "List all the observations. Do not output anything else."
FILTER_TO_USEFUL_LIST_PROMPT = "Filter these observations to the most useful."
PARSE_INTO_PYTHON_LIST_PROMPT = """Format these observations as a Python list of strings. Below is an example:

List of observations:
1. Use a greedy algorithm on XYZ.
2. Dynamic programming can be used on part A with transition B.

Python output:
```python
[
    "Use a greedy algorithm on XYZ.",
    "Dynamic programming can be used on part A with transition B."
]
```

Output the Python output only."""


SYSTEM_PROMPT_NL_SOL_FROM_OBS_COMBO = ("You are an expert Python programmer. " +
                 "You will be given an algorithmic question (problem specification) and " +
                 "some observations which are necessary to solve the problem."
                 " You will return a high-level, natural language solution to the question, " +
                 "like an editorial, which uses the observations given. You will NOT return any code. Be as creative as possible, " +
                 "going beyond what you think is intuitively correct.")

def get_nl_solution_from_obs_combo(problem_str: str, obs_combo: tuple[str, ...]) -> str:
    prompt = f"Here is the competitive programming problem:\n\n{problem_str}\n\n"
    if len(obs_combo) == 0:
        prompt += "No observations are necessary to solve this problem.\n\n"
    else:
        prompt += "Here are the intelligent observations to help solve the problem:\n\n"
        prompt += " - " + "\n\n - ".join(obs_combo) + "\n\n"
    
    prompt += ("Use these observations above to brainstorm a natural language solution to the problem above. " + 
               "Note that your intuition may lead you astray, so come up with simple, creative ideas " +
               "that go beyond what you would usually come up with and exceeds your narrow intuition. " +
               "Quote relevant parts of the observations EXACTLY before each step of the solution. QUOTING IS CRUCIAL. ")
    
    return prompt

def get_nl_sols_prompt(problem_str: str, observation_combo: tuple[str, ...]) -> list[dict[str, str]]:
    convo = [{"role": "system", "content": SYSTEM_PROMPT_NL_SOL_FROM_OBS_COMBO},
                {"role": "user", "content": get_nl_solution_from_obs_combo(problem_str, observation_combo)}]
    return convo

def get_merge_fixes_prompt(problem_str: str, nl_solution: str, fixes: str) -> list[dict[str, str]]:
    convo = [{"role": "system", "content": SYSTEM_PROMPT_MERGE_FIXES},
                {"role": "user", "content": get_merge_orig_fix(problem_str, nl_solution, fixes)}]
    return convo


SYSTEM_PROMPT_CODE_FROM_OBS_COMBO = ("You are an expert Python programmer. " +
                 "You will be given an algorithmic question (problem specification) and " +
                 "some observations which are necessary to solve the problem. "
                 "You will generate a correct Python program that uses the observations and "
                 "passes all tests. You will NOT return anything except for the program inside Markdown codeblocks.")

def get_code_solution_from_obs_combo(question: str, obs_combo: tuple[str, ...], starter_code: str = "") -> str:
    og_q = question.replace('"""', r'\"""')

    out_str = ("You will given a competitive programming problem and several observations " +
               "necessary to solve it; please output correct Python code that " +
               "uses the observations and solves the problem. Below are examples:\n\n\n")

    out_str += f"Here is the competitive programming problem:\n\n{og_q}\n\n"

    if len(obs_combo) == 0:
        out_str += "No observations are necessary to solve this problem.\n\n"
    else:
        out_str += "Here are the intelligent observations required to solve the problem:\n\n"
        out_str += " - " + "\n\n - ".join(obs_combo) + "\n\n"

    out_str += "Please write a Python program that solves the problem above. You must use the observations above."

    if not (starter_code == "" or starter_code is None):
        out_str += "\n\nYour solution should also utilize the following starter code:\n\n```python\n" + starter_code + "\n```"
    else:
        out_str += "\n\nYour solution should use standard input/output and will be run directly without any edits."
    return out_str

def get_code_sol_from_obs_combo_prompt(problem_str: str, starter_code: str, observation_combo: tuple[str, ...]) -> list[dict[str, str]]:
    convo = [{"role": "system", "content": SYSTEM_PROMPT_CODE_FROM_OBS_COMBO},
                {"role": "user", "content": get_code_solution_from_obs_combo(problem_str, observation_combo, starter_code=starter_code)}]
    return convo
