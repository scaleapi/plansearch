from __future__ import annotations
from typing import Any, Optional, Union
from collections import deque

from base_classes import Test
from python_utils import safe_repr


class Function:
    def __init__(
        self,
        name: str,
        arguments: list[str],
        return_type: str,
        description: str,
        parents: Optional[list[Function]],
        prefix_for_prompts: str = "",
        problem_context: str = "",
    ):
        self.name: str = name
        self.args = arguments
        self.ret = return_type
        self.desc = description
        self.prefix_for_prompts = prefix_for_prompts
        self.problem_context = problem_context
        if parents is None:
            self.parents: list[Function] = []
        else:
            self.parents: list[Function] = parents
        self.tests: list[Test] = []
        self.children: list[Function] = []
        self.implementations = []
        self.temp_impls: list[str] = []
        self.fixed_implementation: Optional[str] = None

    def call_str(
        self,
        arg_list: list,
        kwarg_dict: dict,
        no_arg_repr: bool = False,
        temp_placement_arg: Optional[str] = None,
    ):
        if temp_placement_arg is not None:
            return f"{self.name}({temp_placement_arg})"
        return (
            f"{self.name}({create_args_kwargs_str(arg_list, kwarg_dict, no_arg_repr)})"
        )

    # "name(args) -> return"
    def fn_signature(self):
        return f"{self.name}({', '.join(self.args)})" + (
            f" -> {self.ret}" if self.ret else ""
        )

    def header_with_def(self):
        header = f"def {self.fn_signature()}"
        return header

    # Get the string representation of all implementations of this function
    def get_implementation_strs(self) -> list[str]:
        return self.implementations

    # Set the implementation of this function as fixed to a particular string
    def fix_implementation(self, impl_str: str):
        self.fixed_implementation = impl_str
        self.temp_impls = []

    # We need to be careful about infinite recursion here
    # Get all functions that are descendants of this function
    def get_descendants(self, visited=None) -> dict[str, Function]:
        if visited is None:
            visited = {self.name: self}
        for child in self.children:
            if child.name not in visited:
                visited[child.name] = child
                child.get_descendants(visited)
        return visited

    # Get all functions that are ancestors of this function
    def get_ancestors(self, visited=None):
        if visited is None:
            visited = {self.name: self}
        for parent in self.parents:
            if parent.name not in visited:
                visited[parent.name] = parent
                parent.get_ancestors(visited)
        return visited

    def __repr__(self):
        parent_names = [parent.name for parent in self.parents]
        child_name = [child.name for child in self.children]
        ret_str = f" -> {self.ret}"
        return f"Function({self.name}({self.args}){ret_str}); parents: {parent_names}; children: {child_name})"


def get_all_descendant_impls(
    fn_list: list[Function]
) -> str:
    """
    Returns a string of all descendants of functions in `fn_list`.

    Requires that all descendants of functions in `fn_list` have already fixed
    their implementations.

    Does not include the implementations of functions in `fn_list`.
    """
    # TODO: recursion not accounted for
    fn_names = set([fn.name for fn in fn_list])
    dependency_str = ""
    # Gets all descendant fn names
    for fn in fn_list:
        for fn_name, new_fn in fn.get_descendants().items():
            # Adds their implementations, besides those from fn_names
            if fn_name not in fn_names:
                assert new_fn.fixed_implementation is not None

                dependency_str += new_fn.fixed_implementation + '\n'
               
                fn_names.add(fn_name)

    return dependency_str


def create_args_kwargs_str(args: list, kwargs: dict, no_arg_repr: bool = False) -> str:
    if len(args) + len(kwargs) == 0:
        return ""

    if not no_arg_repr:
        arg_str = ", ".join([safe_repr(arg) for arg in args])
    else:
        arg_str = ", ".join(args)
    kwarg_str = ", ".join([f"{k}: {safe_repr(v)}" for k, v in kwargs.items()])

    if arg_str is None:
        return kwarg_str
    if kwarg_str is None:
        return arg_str
    return ", ".join([arg_str, kwarg_str])

