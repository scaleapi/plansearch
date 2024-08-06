from search.fn import Function

def assert_check(line: str) -> bool:
    line = line.strip()
    return (
        find_str(line, ":") == -1
        and "->" in line
        and (find_str(line, "-") == find_str(line, ">") - 1)
    )

def find_str(line: str, target: str) -> int:
    # Find the first : not in parentheses
    paren_count = 0
    bracket_count = 0
    curly_count = 0
    in_string = None
    for i, c in enumerate(line):
        if c == "(":
            paren_count += 1
        elif c == ")":
            paren_count -= 1
        elif c == "[":
            bracket_count += 1
        elif c == "]":
            bracket_count -= 1
        elif c == "{":
            curly_count += 1
        elif c == "}":
            curly_count -= 1
        elif c == '"' or c == "'":
            if in_string == c:
                in_string = None
            else:
                in_string = c
        elif (
            c == target
            and paren_count == 0
            and bracket_count == 0
            and curly_count == 0
            and in_string is None
        ):
            return i
    return -1

def parse_line(line: str) -> tuple[str, list[str], str, str]:
    # Parse a function definition
    colon_idx = find_str(line, ":")
    if colon_idx == -1:
        return line, None, None, None
    fn_sig, desc = line[:colon_idx], line[colon_idx + 1 :]
    desc = desc.strip()
    if len(fn_sig.split("(", 1)) == 1:
        raise ValueError(f"Invalid function signature: {fn_sig}")
    fn_name, fn_args = fn_sig.split("(", 1)
    if "->" in fn_args:
        fn_args, fn_ret = fn_args.split("->", 1)
        fn_args, fn_ret = fn_args.strip(), fn_ret.strip()
    else:
        fn_ret = ""
    assert fn_args.endswith(")")
    fn_args = fn_args[:-1]

    fn_arg_list = []
    while True:
        comma_idx = find_str(fn_args, ",")
        if comma_idx == -1:
            fn_arg_list.append(fn_args)
            break
        fn_arg_list.append(fn_args[:comma_idx])
        fn_args = fn_args[comma_idx + 1 :]

    fn_arg_list = [arg.strip().replace("List", "list") for arg in fn_arg_list]
    return fn_name, fn_arg_list, fn_ret, desc

def parse_to_fn(
    line, parent, defined_fns, scope=None, loose_ref=False, loose_def=False
):
    if scope is None:
        scope = defined_fns
    fn_name, fn_args, fn_ret, desc = parse_line(line.strip())
    # print(f"Parsing {fn_name}({fn_args}) -> {fn_ret}")
    # print("Line:", line)
    if fn_name in scope:
        if fn_args is not None:
            if loose_ref:
                print(f"Warning: Function {fn_name} already defined")
            else:
                raise RuntimeError(f"Warning: Function {fn_name} already defined")
            new_fn = defined_fns[fn_name]
            if parent is not None:
                if parent not in new_fn.parents:
                    new_fn.parents.append(parent)
                if new_fn not in parent.children:
                    parent.children.append(new_fn)
            return new_fn
        new_fn = defined_fns[fn_name]
        if parent is not None:
            new_fn.parents.append(parent)
            parent.children.append(new_fn)
        return new_fn
    else:
        if fn_args is not None:
            new_fn = Function(
                name=fn_name,
                arguments=fn_args,
                return_type=fn_ret,
                description=desc,
                parents=[parent],
            )
            if parent is not None:
                parent.children.append(new_fn)
                defined_fns[fn_name] = new_fn
                new_fn.prefix_for_prompts = parent.prefix_for_prompts
            return new_fn
        else:
            if loose_def:
                print(f"Function {fn_name} not defined; skipped")
            else:
                raise RuntimeError(f"Function {fn_name} not defined; skipped")

def initial_node(line, cur_node):
    new_node = {
        "name": line.split("(")[0].strip(),
        "line": line,
        "children": [],
        "parent": cur_node,
        "asserts": [],
    }
    if cur_node is not None:
        cur_node["children"].append(new_node)
    return new_node


def fill_graph(node, node_equiv, defined_fns=None, scope=None):
    if defined_fns is None:
        defined_fns = {}
    if scope is None:
        scope = set()
    else:
        scope = scope.copy()
    scope.add(node["name"])
    child_equivs = []
    for child in node["children"]:
        child_node = parse_to_fn(child["line"], node_equiv, defined_fns, scope)
        scope.add(child_node.name)
        child_equivs.append(child_node)
    for child, child_equiv in zip(node["children"], child_equivs):
        fill_graph(child, child_equiv, defined_fns, scope)
    return defined_fns


# Inspired by https://stackoverflow.com/questions/45964731/how-to-parse-hierarchy-based-on-indents-with-python
def get_graph(program: str) -> tuple[Function, dict[str, Function]]:
    program = program.splitlines()
    root = initial_node("root", None)
    cur_node = root
    indentation = [-1]
    depth = -1
    buffer_line = ""
    for cur_line in program:
        # Handle line continuations
        if cur_line[-1] == "\\":
            buffer_line += cur_line[:-1] + "\n"
            continue
        line = buffer_line + cur_line
        buffer_line = ""

        indent = len(line) - len(line.lstrip())
        if not line.strip():
            continue
        if indent > indentation[-1]:
            new_node = initial_node(line, cur_node)
            cur_node = new_node
            depth += 1
            indentation.append(indent)
            continue

        if indent < indentation[-1]:
            while indent < indentation[-1]:
                depth -= 1
                indentation.pop()
                cur_node = cur_node["parent"]

            if indent != indentation[-1]:
                raise RuntimeError("Bad formatting")

        if indent == indentation[-1]:
            if assert_check(line):
                cur_node["asserts"].append(line.strip())
                print("warning, assert encountered")
            else:
                new_node = initial_node(line, cur_node["parent"])
                cur_node = new_node

    temp_root = Function(
        name="root", arguments=[], description="Main function", return_type=[], parents=None
    )
    defined_fns = {"root": temp_root}
    fill_graph(root, temp_root, defined_fns=defined_fns, scope={"root"})
    del defined_fns["root"]
    assert len(temp_root.children) == 1, "There should only be one root function"
    root_fn_graph = temp_root.children[0]
    return root_fn_graph, defined_fns


def strongly_connected_components(
    defined_fns: dict[str, Function]
) -> tuple[list[set[str]], list[list[int]]]:
    # Identify the nodes reachable from each node
    reachable = {fn_name: {fn_name} for fn_name in defined_fns}
    changed = True
    while changed:
        changed = False
        # Loop through all the pairs of fn_name and the functions reachable from it
        for fn_name, fns_reachable in reachable.items():
            # Loop through all the functions reachable from fn_name
            for fn_reachable_name in fns_reachable.copy():
                fn = defined_fns[fn_reachable_name]
                # Loop through all the children of the functions reachable from fn_name
                for child in fn.children:
                    initial_len = len(reachable[fn_name])
                    # Try to add the child to the set of functions reachable from fn_name
                    reachable[fn_name].add(child.name)
                    # If the child has no asserts, it also depends on the parent
                    initial_len_2 = len(reachable[child.name])
                    reachable[child.name].add(fn_reachable_name)
                    if len(reachable[child.name]) > initial_len_2:
                        changed = True
                    if len(reachable[fn_name]) > initial_len:
                        changed = True
                # Reachability is transitive, so add everything reachable from anything reachable from fn_name
                for fn_reachable_name_2 in fns_reachable.copy():
                    initial_len = len(reachable[fn_name])
                    reachable[fn_name].update(reachable[fn_reachable_name_2])
                    if len(reachable[fn_name]) > initial_len:
                        changed = True

    # Identify the strongly connected components
    sccs = []
    remaining_nodes = set(defined_fns)
    for fn_name in defined_fns.keys():
        if fn_name not in remaining_nodes:
            continue
        remaining_nodes.remove(fn_name)
        scc = {fn_name}
        for child_name in reachable[fn_name]:
            if fn_name in reachable[child_name]:
                if child_name in remaining_nodes:
                    scc.add(child_name)
                    remaining_nodes.remove(child_name)
        sccs.append(scc)

    # Identify the relationships between the strongly connected components
    scc_edges = []
    for scc_1_idx, scc_1 in enumerate(sccs):
        scc_1_edges = []
        for scc_2_idx, scc_2 in enumerate(sccs):
            if scc_1_idx == scc_2_idx:
                continue
            if list(scc_2)[0] in reachable[list(scc_1)[0]]:
                scc_1_edges += [scc_2_idx]
        scc_edges.append(scc_1_edges)
    return sccs, scc_edges


def get_root(defined_fns) -> str:
    # Identify a function which is the parent of all other functions
    # We allow for cycles, so we can't use just parents
    shared_ancestors = None
    for fn in defined_fns.values():
        if shared_ancestors is None:
            shared_ancestors = set(fn.get_ancestors()) | {fn.name}
        else:
            shared_ancestors.intersection_update(fn.get_ancestors())
    shared_defined = shared_ancestors & set(defined_fns.keys())
    root_name = shared_defined.pop()
    assert isinstance(root_name, str)
    return root_name
