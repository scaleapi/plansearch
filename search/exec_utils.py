from typing import Optional, Union, Any

from fn import Function, get_all_descendant_impls
from base_classes import Test
from coderm.execution import smart_exec_tests_queuebatched


def get_passed_tests_and_errors(
    test_results: list[bool, str],
    tests: list[Test],
) -> tuple[list[Test], set[str]]:
    assert len(test_results) == len(tests)

    passed_tests = []
    str_errors = set()
    for test, test_result in zip(tests, test_results):
        if test_result[0]:
            passed_tests.append(test)
        else:
            str_errors.add(test_result[1])
    return passed_tests, str_errors

def run_tests_per_code(impls: list[str], tests_per_code: list[list[Test]], timeouts: list[int]) -> list[tuple[bool, str]]:
    assert len(impls) == len(tests_per_code) == len(timeouts)

    inputs_pc = [[test.get_input_no_kwargs() for test in tests] for tests in tests_per_code]
    outputs_pc = [[test.output for test in tests] for tests in tests_per_code]
    fn_names_pc = [[test.fn_name for test in tests] for tests in tests_per_code]

    test_dicts = []
    for fn_names, inputs, outputs in zip(fn_names_pc, inputs_pc, outputs_pc):
        test_dicts.append({"inputs": inputs, "outputs": outputs})

        # Check fn_names is all the same
        if len(fn_names):
            assert len(fn_names)
            assert all(fn_name == fn_names[0] for fn_name in fn_names)

            if fn_names[0] is not None:
                test_dicts[-1]["fn_name"] = fn_names

    for inputs in inputs_pc:
        if len(inputs) == 0:
            print("Warning: empty input test case found.")
    return smart_exec_tests_queuebatched(impls, test_dicts, timeouts=timeouts)


def run_tests(impl: str, tests: list[Test], timeout: int) -> tuple[bool, str]:
    return run_tests_per_code([impl], [tests], [timeout])[0]


def check_fn(
    fn: Function,
    timeout: float,
    tests: Optional[list[Test]] = None,
    separate: bool = False
) -> Union[tuple[bool, str], list[tuple[bool, str]]]:
    """
    Evaluates `fn` with the fixed implementations, on the provided list of
    `tests`. If `tests` is None, uses all the tests defined in `fn`.
    """
    assert fn.fixed_implementation is not None

    impl_str = get_all_descendant_impls([fn])
    impl_str += fn.fixed_implementation

    if tests is None:
        tests = fn.tests

    if separate:
        impl_strs = [impl_str] * len(tests)
        timeouts = [timeout] * len(tests)
        separate_tests = [[test] for test in tests]

        output_results = run_tests_per_code(impl_strs, separate_tests, timeouts=timeouts)
        assert len(output_results) == len(tests)
        return output_results
    else:
        return run_tests(impl_str, tests, timeout)
