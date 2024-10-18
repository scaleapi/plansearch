from typing import Any, Optional, Union


# Directly from CodeRM
def markdown_codeblock_extract(response: str) -> str:
    lines = response.split("\n")
    buf = ""
    in_codeblock = False
    for ln in lines:
        if ln.strip().startswith("```"):
            if in_codeblock:
                break
            else:
                in_codeblock = True
        elif in_codeblock:
            buf += ln + "\n"
    return buf

def find_encased_strings(string: str, begin_end_pattern: Union[str, tuple[str, str]]) -> list[str]:
    """
    `string`: the string to be searched through
    `begin_end_pattern`: a string or tuple of strings containing the start and end pattern that encases
        the targets. If a string is provided, returns all possible
        strings divided by the pattern. If they are different, returns all strings
        encased between begin and end patterns.
    """
    if isinstance(begin_end_pattern, str):
        return string.split(begin_end_pattern)
    
    begin_str = begin_end_pattern[0]
    end_str = begin_end_pattern[1]

    start_from_idx = 0
    begin_idx = string.find(begin_str, start_from_idx)
    matches = []
    while begin_idx != -1:
        start_from_idx += len(begin_str)
        end_idx = string.find(end_str, start_from_idx)
        if end_idx == -1:
            break

        matches.append(string[begin_idx+len(begin_str):end_idx])
        
        start_from_idx += len(end_str)
        begin_idx = string.find(begin_str, start_from_idx)
    
    return matches


def remove_helper_imports(impls: list[str], remove_def: bool = False, remove_import: bool = False) -> list[str]:
    """
    Gets rid of all lines starting with "from helpers ..."
    If `remove_def` is `True`, removes lines starting with "def ..." also.
    If `remove_import` is `True`, removes lines starting with "from ..." and "import ..."
    """
    clean_impls = []
    for impl in impls:
        impl_lines = impl.splitlines()
        impl_lines = [line for line in impl_lines if not line.startswith("from helpers ")]
        up_to_idx = 0
        for line in impl_lines:
            skip_def = line.startswith("def ") and remove_def
            skip_import = (line.startswith("from ") or line.startswith("import ")) and remove_import
            if not (skip_import or skip_def or line.strip() == ""):
                break
            
            up_to_idx += 1

        clean_impls.append("\n".join(impl_lines[up_to_idx:]))
    return clean_impls


def filter_backticks_for_fn(string: str, fn_name: str) -> str:
    possible_codes = string.split("```")
    for possible_code in possible_codes:
        filtered_str = filter_for_fn(possible_code, fn_name)
        if filtered_str is not None:
            return filtered_str
    
    
    return extract_code([string], True)[0][1]


def filter_for_fn(string: str, fn_name: str) -> Optional[str]:
    DEF_FN = f"def {fn_name}("
    tab_level = ""
    def_present = False
    lines_after = []
    for line in string.splitlines():
        if not line.startswith(tab_level):
            # Assuming tabs are the same and there are no unindented lines in the function
            break
        line = line[len(tab_level):]

        if def_present and line.lstrip() == line:
            # Assuming there are no unindented lines in the function
            break
 
        if not def_present:
            if DEF_FN in line:
                def_present = True
                tab_level = line[:line.index(DEF_FN)]
                lines_after.append(line[len(tab_level):])
        else:
            lines_after.append(line)

    if not def_present:
        return None
    return '\n'.join(lines_after)


def extract_code(
    string_list: list[str],
    return_orig: bool = False,
    add_markers: Optional[list[tuple[str, str]]] = None,
    attempt_to_add_backticks: bool = True,
    fn_name: Optional[str] = None,
) -> Union[list[str], list[tuple[str, str]]]:
    SUBSEQS = [
        ("[ANSWER]", "[\ANSWER]"),
        ("```python", "```"),
        ("```py", "```"),
        ("```", "```"),
        ("`", "`"),
    ]
    if add_markers is not None:
        SUBSEQS = add_markers + SUBSEQS
    out_list = []
    for string in string_list:
        alt_string = string.replace("```\n```", "```")
        if attempt_to_add_backticks:
            if alt_string.count("```") % 2:
                alt_string += "```"

        ind_list = []
        for subseq in SUBSEQS:
            end_ind = alt_string.rfind(subseq[1])
            start_ind = alt_string.rfind(subseq[0], None, end_ind - 1)
            ind_list.append((start_ind, end_ind))

        max_start_ind = -1
        for start_ind, end_ind in ind_list[:-1]:
            if start_ind != -1 and end_ind != -1:
                max_start_ind = max(max_start_ind, start_ind)

        appended = False
        for i, (start_ind, end_ind) in enumerate(ind_list[:-1]):
            if start_ind != -1 and end_ind != -1 and start_ind == max_start_ind:
                to_append = alt_string[start_ind + len(SUBSEQS[i][0]) : end_ind].strip()
                out_list.append(to_append)
                appended = True
                break

        if not appended:
            start_ind, end_ind = ind_list[-1]
            if start_ind != -1 and end_ind != -1:
                to_append = alt_string[start_ind + len(SUBSEQS[-1][0]) : end_ind]
                out_list.append(to_append)
            else:
                # No matches
                out_list.append(None)
    
    # Post-processing
    for i, out_str in enumerate(out_list):
        if out_str is None:
            continue
        if len(out_str) and out_str[0] == '`' and out_str[-1] == '`':
            out_str = out_str[1:-1]
        if fn_name is not None:
            out_str = filter_for_fn(out_str, fn_name=fn_name)
        
        out_list[i] = out_str

    if return_orig:
        return [(out_str, in_str) for in_str, out_str in zip(string_list, out_list)]
    
    return [out_str for out_str in out_list if out_str is not None]

if __name__ == "__main__":
    pass
