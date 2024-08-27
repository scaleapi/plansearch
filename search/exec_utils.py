from typing import Optional, Union
import os

from search.fn import Function, get_all_descendant_impls
from search.base_classes import Test
from coderm.execution import smart_exec_tests_queuebatched


def get_passed_tests_and_errors(
    test_results: list[tuple[bool, str]],
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


def run_tests_per_code(impls: list[str], tests_per_code: list[Union[list[Test], str]], timeouts: list[int], fn_names_pc: Optional[list[Optional[str]]] = None, num_workers: Optional[int] = os.cpu_count(), testbank: Optional[str] = None, executor: str = "http://127.0.0.1:8000") -> list[tuple[bool, str]]:
    assert len(impls) == len(tests_per_code) == len(timeouts)

    test_infos = []
    has_Solutions_pc = []
    for i, tests in enumerate(tests_per_code):
        if isinstance(tests, list):
            inputs = [test.get_input_no_kwargs() for test in tests]
            outputs = [test.output for test in tests]
            fn_names = [test.fn_name for test in tests]
            has_Solutions = tests[0].has_Solution if len(tests) else None
            test_dict = {"inputs": inputs, "outputs": outputs}

            # Check fn_names is all the same
            if len(fn_names):
                assert all(fn_name == fn_names[0] for fn_name in fn_names)
                fn_name = fn_names[0]
                if fn_names_pc is not None:
                    assert fn_name == fn_names_pc[i]
            else:
                if fn_names_pc is not None:
                    fn_name = fn_names_pc[i]
                else:
                    fn_name = None

            if fn_name is not None:
                test_dict["fn_name"] = fn_names


            has_Solutions_pc.append(has_Solutions)
            test_infos.append(test_dict)

            if len(inputs) == 0:
                print("Warning: empty input test case found.")

        else:
            assert isinstance(tests, str)
            test_infos.append(tests)
            has_Solutions_pc.append(None)

    return smart_exec_tests_queuebatched(impls, test_infos, timeouts=timeouts, has_Solution_per_code=has_Solutions_pc, workers=num_workers, executor=executor, testbank=testbank)


def run_tests(impl: str, tests: Union[list[Test], str], timeout: int, num_workers: Optional[int] = os.cpu_count(), testbank: Optional[str] = None, executor: str = "http://127.0.0.1:8000") -> tuple[bool, str]:
    return run_tests_per_code([impl], [tests], [timeout], num_workers=num_workers, testbank=testbank, executor=executor)[0]


def check_fn(
    fn: Function,
    timeout: int,
    tests: Optional[list[Test]] = None,
    separate: bool = False,
    num_workers: Optional[int] = os.cpu_count(),
    testbank: Optional[str] = None,
    executor: str = "http://127.0.0.1:8000",
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

        output_results = run_tests_per_code(impl_strs, separate_tests, timeouts=timeouts, num_workers=num_workers, testbank=testbank, executor=executor)
        assert len(output_results) == len(tests)
        return output_results
    else:
        return run_tests(impl_str, tests, timeout, num_workers=num_workers, testbank=testbank, executor=executor)

if __name__ == "__main__":
    n_codes = 200
    import random
    random_nums = [random.randint(0, 100) for _ in range(n_codes)]
    codes = [f"print({num})" for num in random_nums]
    tests = [[Test(([""], {}), str(num), None)] for num in random_nums]
    timeouts = [5 for _ in range(n_codes)]
    print(run_tests_per_code(codes, tests, timeouts))
