import argparse
import itertools
from tqdm import tqdm
import random
import time
from typing import Optional, TypeVar, Union, Any
import os
import sys
from uuid import uuid4

from search.queriers import LLMQuerier
from search.exec_utils import check_fn, run_tests_per_code, get_passed_tests_and_errors
from search.parsel.construct_graph import strongly_connected_components, get_root
from search.parsel.parsel_queries import gen_fn_implementations, gen_fn_tests
from search.fn import Function
from search.base_classes import Test
from search.python_utils import log_to_dir, chunk


P_RANDOM_SAMPLE = 0.002

# Join a set of function implementations to a string, along with
# the already-implemented dependencies of the functions
def to_implementation_str(implementation_set: list[str], dependencies_str: str) -> str:
    implementation_attempt = dependencies_str
    for fn_implementation in implementation_set:
        implementation_attempt += fn_implementation + "\n"
    return implementation_attempt


# Use multiprocessing to try to fill in the implementation of an SCC
def try_impl_combos(
    scc: set[str],
    dependencies_str: str,
    defined_fns: dict[str, Function],
    all_implementations: dict[str, list[str]],
    tests: list[Test],
    args: argparse.Namespace,
    log_directory: str,
):
    implementation_set_keys = all_implementations.keys()
    random.seed(args.seed)
    all_implementation_sets = [list(set(impls)) for impls in all_implementations.values()]
    # We save memory by only storing the index of the implementation in all_implementation_sets
    implementation_sets: list[tuple[int, ...]] = list(
        itertools.product(*[list(range(len(impls))) for impls in all_implementation_sets])
    )
    num_to_try = min(len(implementation_sets), args.max_tries_of_combo)

    if num_to_try == len(implementation_sets):
        random.shuffle(implementation_sets)
    else:
        implementation_sets = random.sample(implementation_sets, num_to_try)

    start_time = time.time()


    pbar = tqdm(total=num_to_try)

    def impl_idx_to_set(implementation_set_indices: tuple[int, ...]) -> list[str]:
        return [
            all_implementation_sets[i][impl_id]
            for i, impl_id in enumerate(implementation_set_indices)
        ]

    for chunked_impl_sets in chunk(implementation_sets, args.parsel_exec_batch_size):
        random_samples = {}
        impls_to_test = []
        for i, implementation_set_idxs in enumerate(chunked_impl_sets):
            implementation_set = impl_idx_to_set(implementation_set_idxs)
            impls_to_test.append(to_implementation_str(implementation_set, dependencies_str))
            if random.random() < P_RANDOM_SAMPLE:
                log_dict = {"impl": to_implementation_str(implementation_set, dependencies_str), "tests": [test.to_repr_dict() for test in tests]}
                random_samples[i] = log_dict

        results = run_tests_per_code(impls_to_test, [tests] * len(impls_to_test), [args.timeout] * len(impls_to_test)) 

        assert len(results) == len(chunked_impl_sets)
        
        for i, ((passed, error), implementation_set_idxs) in enumerate(zip(results, chunked_impl_sets)):
            implementation_set = impl_idx_to_set(implementation_set_idxs)
            implementation_attempt = to_implementation_str(implementation_set, dependencies_str)
            if i in random_samples:
                random_samples[i]["error"] = error
                random_samples[i]["passed"] = passed
                log_to_dir(log_directory, {f"sample-{str(uuid4())[:6]}.json": random_samples[i]})

            pbar.update(1)
            if passed:
                pbar.close()

                # We succeeded, so we can return the implementation
                print("Successfully implemented", scc)

                # Since we found a working solution, we can consider the implementation fixed
                for fn_name, implementation in zip(implementation_set_keys, implementation_set):
                    fn = defined_fns[fn_name]
                    if fn.fixed_implementation is None:
                        fn.fix_implementation(implementation)

                tests_dict = {
                    "all_tests": [test.to_repr_dict() for test in tests],
                    "passed_tests": [test.to_repr_dict() for test in tests],
                }

                log_to_dir(
                    log_directory,
                    {
                        "implementation.py": implementation_attempt,
                        "train_tests.json": tests_dict,
                    },
                )

                return implementation_attempt
       
        if time.time() - start_time > args.max_time:
            break

    print(
        f"Failed implementing {scc}."
    )
    return None


# Evaluate all the combinations of possible
# implementations of the functions in the SCC
def eval_scc(
    scc: set[str],
    dependencies_str: str,
    defined_fns: dict[str, Function],
    args: argparse.Namespace,
    log_directory: str,
):
    all_implementations = {}

    tests = []
    for fn_name in scc:
        fn = defined_fns[fn_name]
        all_implementations[fn_name] = fn.get_implementation_strs()
        tests.extend(fn.tests)

    print("Attempting to implement", scc)
    return try_impl_combos(
        scc=scc,
        dependencies_str=dependencies_str,
        defined_fns=defined_fns,
        all_implementations=all_implementations,
        tests=tests,
        args=args,
        log_directory=log_directory,
    )


# Implement the SCC and return the string
def implement_scc(
    scc_idx: int,
    sccs: list[set[str]],
    implemented_sccs: dict[int, str],
    scc_edges: list[list[int]],
    defined_fns: dict[str, Function],
    querier: LLMQuerier,
    args: argparse.Namespace,
    log_directory: str,
) -> Optional[str]:
    print("Implementing SCC", scc_idx, sccs[scc_idx])
    if scc_idx in implemented_sccs:
        return implemented_sccs[scc_idx]
    dependencies_str = ""
    for edge in scc_edges[scc_idx]:
        implement_scc_str = implement_scc(
            scc_idx=edge,
            sccs=sccs,
            implemented_sccs=implemented_sccs,
            scc_edges=scc_edges,
            defined_fns=defined_fns,
            querier=querier,
            args=args,
            log_directory=log_directory
        )
        if implement_scc_str is None:
            return None
        dependencies_str += implement_scc_str

    num_completions = args.num_completions
    print(f"Trying {num_completions} completions")

    # generating fn implementations in bulk way
    fn_list = list(defined_fns.values())
    fn_impls = gen_fn_implementations(fn_list, querier, model=args.parsel_to_code_model_name, num_completions=num_completions, max_tokens=args.max_tokens)
    for fn, impl in zip(fn_list, fn_impls):
        fn.implementations = impl
        
    log_to_dir(os.path.join(log_directory, "fn_impls"), {f"{fn.name}.json": fn.get_implementation_strs() for fn in defined_fns.values()})
    scc_implementation = eval_scc(
        scc=sccs[scc_idx],
        dependencies_str=dependencies_str,
        defined_fns=defined_fns,
        args=args,
        log_directory=log_directory,
    )

    if scc_implementation is not None:
        implemented_sccs[scc_idx] = dependencies_str + scc_implementation
        return implemented_sccs[scc_idx]

    print(f"No implementations found for scc {sccs[scc_idx]}...")


# The key function of the program, which takes a function graph
# Decomposes them to their strongly connected components
# And then implements each SCC in turn
def parsel_graph(
    defined_fns: dict[str, Function],
    querier: LLMQuerier,
    args: argparse.Namespace,
    log_directory: str,
) -> tuple[int, bool]:
    if args.generate_tests:
        raise NotImplementedError("generate tests not fully implemented yet")

    sccs, scc_edges = strongly_connected_components(defined_fns)
    implemented_sccs = {}

    for scc_idx, _ in enumerate(sccs):
        new_str = implement_scc(
            scc_idx=scc_idx,
            sccs=sccs,
            implemented_sccs=implemented_sccs,
            scc_edges=scc_edges,
            defined_fns=defined_fns,
            querier=querier,
            args=args,
            log_directory=log_directory
        )
        if new_str is None:
            break

    for fn in defined_fns.values():
        if fn.fixed_implementation is None:
            print(f"Warning: Parsel {fn.name} fixed_implementation is None.")
            return 0, False
        assert fn.fixed_implementation is not None

    test_results = check_fn(defined_fns[get_root(defined_fns)], args.timeout, separate=True)
    passed_tests, errors = get_passed_tests_and_errors(test_results, defined_fns[get_root(defined_fns)].tests)

    if len(errors) != 0:
        print("Parsel train sample test errors: ", errors)

    return len(passed_tests), True
