from search.fn import Function
from search.queriers import LLMQuerier, MODELS_TO_METHOD, SUPPORTED_CLIENT_FOR_ASSISTANT_STR
from search.prompts.fn_prompts import chat_prompt_to_implement_fn, completion_prompt_to_implement_fn, parsel_to_code_chat_suffix
from search.parsing_utils import remove_helper_imports, extract_code


# Call code model and optionally filter the results
# Generate implementations for the function
def gen_fn_implementations(fns: list[Function], querier: LLMQuerier, model: str, num_completions: int, max_tokens: int = 1000, with_child_impls: bool = False) -> list[str]:
    if model in MODELS_TO_METHOD and MODELS_TO_METHOD[model] in SUPPORTED_CLIENT_FOR_ASSISTANT_STR: 
        raise NotImplementedError("not fully integrated into Scale")
        if with_child_impls:
            raise NotImplementedError("Assistant start in `fn.py` implement not implemented yet with fixed child impls.")
        in_strings = self.prefix_for_prompts
        in_strings += parsel_to_code_chat_suffix(fn_signature=self.fn_signature(), fn_description=self.desc)
        assistant_start = self.prefix_for_prompts + '\n' + self.get_code_prefix_with_helpers()
        stop = ["\ndef"]
        indented = True
    else:
        in_strings = [chat_prompt_to_implement_fn(fn=fn, with_impls=with_child_impls) for fn in fns for _ in range(num_completions)]

    implementations = querier.generate(
        model=model,
        prompts=in_strings,
        max_tokens=max_tokens,
        temperature=0.6,
        log_name="functions_impl",
    )
    
    assert len(implementations) == len(fns) * num_completions

    if not (model in MODELS_TO_METHOD and MODELS_TO_METHOD[model] in SUPPORTED_CLIENT_FOR_ASSISTANT_STR):
        fn_implementations = [remove_helper_imports(extract_code(implementations[i*num_completions:(i+1)*num_completions], fn_name=fns[i].name), remove_def=True, remove_import=True) for i in range(len(fns))]
    else:
        raise NotImplementedError("not fully integrated into Scale")

    assert len(fn_implementations) == len(fns)

    return [[fn.header_with_def() + ":\n" + impl.rstrip() + '\n' for impl in implementations] for fn, implementations in zip(fns, fn_implementations)]


# Generate tests for this function
def gen_fn_tests(fn: Function, querier: LLMQuerier, model: str, num_completions: int):
    raise NotImplementedError("not fully integrated into Scale")
    tests = querier.generate(
        codex_in=self.get_codex_test_input(),
        num_completions=num_completions * 5,
        max_tokens=100,
        temperature=0.6,
        stop="\n",
        indented=True,
        indented_after_first_line=False,
        require=None,
        cache_key=None,
        model=model,
    )
    tests = set([test[0] for test in tests if test])
    return tests
