from search.fn import Function


def description_comment_helper(description: str) -> str:
    return f"# Description: {description}\n"

def import_statement_helper(name: str) -> str:
    return f"from helpers import {name}\n"

def signature_comment_helper(sig: str) -> str:
    return f"# Signature: {sig}\n"

def uses_comment_helper(uses: str) -> str:
    return f"# Uses: {uses}\n"

def parsel_to_code_chat_suffix(fn_signature: str, fn_description: str) -> str:
    return (f"Implement the function `{fn_signature}`. The function description is:\n" +
            f"{fn_description}\nANSWER in Python code, surrounded by triple backticks.")

def str_with_fn_children_info(fn: Function, with_impls: bool = False) -> str:
    """
    # Description: ...
    # Signature: ...
    from helpers import ...
    <if with_impls is True, instead of import, do the below>
    def <fn_name>(...) -> ...:
        etc...
    """
    base_str = ""
    already_listed = [fn.name]
    for child in fn.children:
        if child.name in already_listed:
            continue
        already_listed.append(child.name)

        base_str += description_comment_helper(child.desc)
        base_str += signature_comment_helper(child.fn_signature())
        if not with_impls:
            base_str += import_statement_helper(child.name)
        else:
            assert child.fixed_implementation is not None
            base_str += child.fixed_implementation
        base_str += '\n'

    return base_str

def str_with_fn_info(fn: Function) -> str:
    """
    # Description: ...
    # Uses: ..., ...
    """
    base_str = ""
    if fn.desc:
        base_str += description_comment_helper(fn.desc)
    if fn.ret:
        base_str += f"# Returns type of: {fn.ret}\n"
    other_children = [child for child in fn.children if child.name != fn.name]
    if other_children:
        base_str += uses_comment_helper(", ".join([child.name for child in other_children]))
    return base_str

# Constructs prompt for code generation
def completion_prompt_to_implement_fn(fn: Function, with_impls: bool = False):
    base_str = fn.prefix_for_prompts

    base_str += str_with_fn_children_info(fn, with_impls=with_impls)
    base_str += str_with_fn_info(fn)

    base_str += f"{fn.header_with_def()}:\n"
    return base_str

def chat_prompt_to_implement_fn(fn: Function, with_impls: bool = False) -> str:
    base_str = fn.prefix_for_prompts + '\n'

    if len(fn.children):
        base_str += f"Implement the function {fn.name}. Here are several helper functions which {fn.name} should use:\n\n"

        # Adds all helper function information to the prompt
        helper_functions = []
        for child in fn.children:
            helper_functions.append(f"{child.name}:\nDescription: {child.desc}\n")

            if with_impls:
                assert child.fixed_implementation is not None
                helper_functions[-1] += f"Implementation:\n```\n{child.fixed_implementation}\n```\n"
            else:
                helper_functions[-1] += f"Signature: `{child.fn_signature()}`\n"
        base_str += "\n".join(helper_functions) + '\n'

        if not with_impls:
            base_str += ("Do not implement the above helper functions. Instead, import them " + 
                        "at the beginning of your code: `from helpers import " + 
                        f"{', '.join(fn.name for fn in fn.children)}`\n\n")
        
        base_str += "Please use the above helper functions in your implementation.\n\n"

    base_str += f"""The problem above is provided for context. Implement the Python function `{fn.name}`, with signature: `{fn.fn_signature()}`.
Description: {fn.desc}

Wrap the code in triple backticks (``` [YOUR CODE HERE] ```)"""

    return base_str

# Constructs prompt for code generation
def completion_prompt_for_tests(fn: Function, with_impls: bool = False):
    base_str = fn.get_codex_input(with_impls=with_impls)
    base_str += "Do not implement the function. Implement the assert only."
    base_str += f"""  pass

# check the correctness of {fn.name}
assert"""
    return base_str
