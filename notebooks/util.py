from typing import Dict, List, Optional, Set, Tuple, TypeVar

import numpy as np
from graphviz import Digraph

T = TypeVar("T")
BRANCH, INTER, END = 0, 1, 2


def is_prnt(prnt: int, childs: List[int], parent_map) -> Optional[int]:
    return next((child for child in childs if prnt in parent_map[child]), None)


def normalize(lst: List[float]) -> List[float]:
    s = sum(lst)
    return [x / s for x in lst]


def random_pop(lst: List[T]) -> T:
    elem = np.random.choice(lst)
    lst.remove(elem)
    return elem


def choose_parent(idx, node_list, branches, inters, ends, parent_map):
    # sourcery skip: raise-specific-error
    if len(ends):
        prnt = random_pop(ends)
        inters.append(prnt)
    elif len(inters):
        prnt = random_pop(inters)
        branches.append(prnt)
    else:
        np.random.shuffle(branches)
        for prnt in branches:
            child = is_prnt(prnt, node_list[idx + 1 :], parent_map)
            if child is not None:
                break
        if child is None:
            raise Exception("No parent found")
        parent_map[child].remove(prnt)
    return prnt


def get_childs(parent_map, node: int):
    return {child for child in parent_map if node in parent_map[child]}


def get_ancestors(parent_map, node, visited=None):
    if visited is None:
        visited = set()
    if node in visited:
        return set()
    visited.add(node)
    ancestors = {node}
    for parent in parent_map[node]:
        ancestors.add(parent)
        ancestors.update(get_ancestors(parent_map, parent, visited))
    return ancestors


def get_descendants(parent_map, node, visited=None):
    if visited is None:
        visited = set()
    if node in visited:
        return set()
    visited.add(node)
    descendants = {node}
    for child, parents in parent_map.items():
        if node in parents:
            descendants.add(child)
            descendants.update(get_descendants(parent_map, child, visited))
    return descendants


def get_dag(num_nodes: int, prob_type: List[float]) -> Dict[int, Set[int]]:
    """
    [Basic]
    - Get a random graph with `num_nodes` nodes.
    - Each node is either a branch, an intermediate, or an end node.
    - To avoid trivial node, if a node has a single parent, it needs to be a branch.
    - There should be a single entry point.
    - There can be multiple exit points.

    [Acyclic]
    - To avoid cycles, the node can only point to nodes with a larger index.
    """
    branches, inters, ends = [], [], []
    node_list = list(range(num_nodes))
    parent_map: Dict[int, Set[int]] = {idx: set() for idx in node_list}
    for idx in node_list:
        if idx > 0 and not len(parent_map[idx]):
            prnt = choose_parent(
                idx, node_list, branches, inters, ends, parent_map
            )
            parent_map[idx].add(prnt)
        if (
            idx >= num_nodes - 2
            and len(parent_map[idx]) == 1
            and (len(get_childs(parent_map, parent_map[idx])) == 1)
        ):
            prnt = choose_parent(
                idx, node_list, branches, inters, ends, parent_map
            )
            parent_map[idx].add(prnt)
            # print("check", idx, prnt)

        if idx == num_nodes - 1:
            node_type = END
        elif idx == num_nodes - 2:
            node_type = np.random.choice(
                [INTER, END], p=normalize(prob_type[1:])
            )
        elif len(parent_map[idx]) <= 1:
            node_type = BRANCH
        else:
            node_type = np.random.choice([BRANCH, INTER, END], p=prob_type)
        # print(idx, node_type)

        if node_type == BRANCH and idx < num_nodes - 2:
            childs = np.random.choice(
                node_list[idx + 1 :], size=2, replace=False
            )
            for child in childs:
                parent_map[child].add(idx)
        elif node_type == INTER or (
            node_type == BRANCH and idx == num_nodes - 2
        ):
            child = np.random.choice(node_list[idx + 1 :])
            parent_map[child].add(idx)
            node_type = INTER

        if node_type == BRANCH:
            branches.append(idx)
        elif node_type == INTER:
            inters.append(idx)
        else:
            ends.append(idx)
        # print(parent_map)
    return parent_map


def get_rev_map(parent_map: Dict[int, Set[int]]) -> Dict[int, List[int]]:
    rev_map = {idx: [] for idx in parent_map}
    for node, parents in parent_map.items():
        for parent in parents:
            rev_map[parent].append(node)
    return {k: sorted(v) for k, v in rev_map.items()}


def get_branches(parent_map: Dict[int, Set[int]]) -> List[int]:
    rev_map = get_rev_map(parent_map)
    return [node for node, childs in rev_map.items() if len(childs) > 1]


def draw_graph(parent_map: Dict[int, Set[int]]) -> Digraph:
    d = Digraph()
    for node in parent_map:
        d.node(str(node))
    for node, parents in parent_map.items():
        for parent in parents:
            d.edge(str(parent), str(node))
    return d


def transition_matrix(parent_map: Dict[int, Set[int]]) -> np.ndarray:
    rev_map = get_rev_map(parent_map)
    num_nodes = len(parent_map)
    mat = np.zeros((num_nodes, num_nodes))
    for node, childs in rev_map.items():
        for child in childs:
            mat[child, node] = 1
    return mat


def get_reaching_state(parent_map) -> Dict:
    return {node: get_descendants(parent_map, node) for node in parent_map}


# refer digraph-example-small.gv
def get_small_dg() -> Dict[int, Set[int]]:  # sourcery skip: merge-dict-assign
    parent_map = {}
    parent_map[0] = {4}
    parent_map[1] = {0}
    parent_map[2] = {1}
    parent_map[3] = {1}
    parent_map[4] = {2, 3}
    parent_map[5] = {0, 3}
    return parent_map, [(0, 5)]



# refer digraph-example.gv
def get_dg() -> Dict[int, Set[int]]:
    # sourcery skip: merge-dict-assign
    parent_map = {}
    parent_map[0] = {1}
    parent_map[1] = {0}
    parent_map[2] = {0}
    parent_map[3] = {2, 5, 6}
    parent_map[4] = {3}
    parent_map[5] = {4}
    parent_map[6] = {4, 5}
    parent_map[7] = {3, 6}
    parent_map[8] = {7}
    parent_map[9] = {7}
    parent_map[10] = {2}
    parent_map[11] = {10}
    parent_map[12] = {11, 13}
    parent_map[13] = {12}
    parent_map[14] = {11, 12}
    parent_map[15] = {10}
    parent_map[16] = {15, 17}
    parent_map[17] = {16}
    parent_map[18] = {15}
    parent_map[19] = {16, 8, 9, 14}
    return parent_map, [(0, 2), (3, 7), (12, 14), (16, 19)]


def _get_dg(num_nodes: int, prob_type: List[float]) -> Dict[int, Set[int]]:
    # sourcery skip: raise-specific-error
    """
    [Basic]
    Get a random graph with `num_nodes` nodes.
    Each node is either a branch, an intermediate, or an end node.
    To avoid trivial node, if a node has a single parent, it needs to be a branch.
    There should be a single entry point.
    There can be multiple exit points.

    [Loop]
    - A branch node that has an incoming edge from one of its descendant is the entry point of the loop.
    - One of the successors of the entry point of the loop that starts the loop is the starting point of the loop.
    - The other successor of the entry point is the exit point of the loop.
    - There could be one entry point, one starting point, and one exit point for each loop.
    - Every node in the loop can only have out-going edge to the node within the loop, or the entry point (continue stmt) or the exit point of the loop (break stmt).

    [Loop construction during generating the graph]
    - If a node X has an outgoing edge to one its ancestor Y would create a loop.
    - The ancestor of X satisfying the following condition can only be the Y and create the loop.
        - Y becomes the entry point of the loop, so it should be the branch node.
        - One of the successors (Z) of Y should lie between Y and X, and it becomes the starting point of the loop.
        - The other successor (W) of Y should not be in between Y and X, and it becomes the exit point of the loop.
        - Every descendant of Z (including itself) can only have an outgoing edge to the node within the loop, or the entry point (continue stmt) or the exit point of the loop (break stmt).
        - Every descendant of Z (including itself) cannot have an incoming edge from the non-decendant of Z.
    - Also, every node that is not in the loop cannot have an outgoing to the descendant of Z (including Z, except Y).
    """
    branches, inters, ends = [], [], []
    # key: [entry, starting, exist], value: {nodes in the loop}
    loop_dict: Dict[Tuple[int, int, int], Set[int]] = {}
    node_list = list(range(num_nodes))
    parent_map: Dict[int, Set[int]] = {idx: set() for idx in node_list}

    def get_child_candidate(idx) -> Set[int]:
        cands = set(list(range(num_nodes)))
        cands.remove(idx)
        already_child = get_childs(parent_map, idx)
        assert len(already_child) <= 1
        cands -= already_child
        for loop_idx, loop_nodes in loop_dict.items():
            if idx in loop_nodes:
                cands = cands.intersection(loop_nodes.union([loop_idx[2]]))
            elif loop_idx[0] != idx:
                non_entry_loop_nodes = loop_nodes.difference([loop_idx[0]])
                cands -= non_entry_loop_nodes
            else:  # loop_idx[0] == idx
                raise Exception(
                    f"Loop entry point should already have two children. ({idx=}, {already_child=})"
                )
        already_ancs = get_ancestors(parent_map, idx)
        for anc in already_ancs:
            if anc in cands and get_childs(parent_map, anc) != 2:
                cands.remove(anc)
        return cands

    def get_parent_candidate(idx):
        cands = set(list(range(num_nodes)))
        cands.remove(idx)
        already_parent = parent_map[idx]
        cands -= already_parent
        for loop_idx, loop_nodes in loop_dict.items():
            if idx in loop_nodes and loop_idx[0] != idx:
                cands = cands.intersection(loop_nodes)
            elif idx not in loop_idx:
                non_entry_loop_nodes = loop_nodes.difference([loop_idx[0]])
                cands -= non_entry_loop_nodes
        return cands

    def update_loop_dict(loop_dict, src, dest):
        print(f"{loop_dict=}, {src=}, {dest=}")
        already_loop = False
        for loop_idx, loop_nodes in loop_dict.items():
            if src in loop_nodes:
                loop_dict[loop_idx].add(dest)
                already_loop = True
        if not already_loop and dest in get_ancestors(parent_map, src):
            print(f"{src=} is an ancestor of {dest=}, {parent_map=}")
            ancs = get_ancestors(parent_map, src)
            dest_childs = get_childs(parent_map, dest)
            assert len(dest_childs) == 2
            assert len(ancs.intersection(dest_childs)) == 1
            start = ancs.intersection(dest_childs).pop()
            exit = dest_childs.difference([start]).pop()
            in_loops = get_descendants(parent_map, start).intersection(ancs)
            in_loops.add(dest)
            loop_dict[(src, start, exit)] = in_loops
        return loop_dict

    def add_parent(idx):
        prnt = np.random.choice(list(get_parent_candidate(idx)))
        parent_map[idx].add(prnt)
        update_loop_dict(loop_dict, prnt, idx)

    for idx in node_list:
        print(idx, parent_map)
        if idx > 0 and not len(parent_map[idx]):
            print(f"{idx=} has no parent, {parent_map=}")
            add_parent(idx)
        if (
            idx == num_nodes - 1
            and len(parent_map[idx]) == 1
            and len(get_childs(parent_map, parent_map[idx])) == 1
        ):
            print(f"{idx=} has only one parent, {parent_map=}")
            # Now the one before from the last node can have
            # more than one child candidate.
            # So, we now no longer need to add the parent for it.
            # However, if there can be only one child candidate,
            # it needs to have another parent.
            add_parent(idx)

        if len(parent_map[idx]) <= 1:
            node_type = BRANCH
        elif idx == num_nodes - 1:
            node_type = END
        else:
            node_type = np.random.choice([BRANCH, INTER, END], p=prob_type)

        if (
            node_type == END
            and len(parent_map[idx]) == 1
            and len(get_childs(parent_map, parent_map[idx])) == 1
        ):
            print("END node. Its parent has only one child.")
            add_parent(idx)
        else:
            child_candidate = list(get_child_candidate(idx))
            print(f"{idx=} has {node_type=} and {child_candidate=}")
            if not child_candidate:
                print(f"{idx=} has no child candidate, {parent_map=}")
                # If there is only one parent, we need to add another parent.
                if (
                    len(parent_map[idx]) == 1
                    and get_childs(parent_map, parent_map[idx]) == 1
                ):
                    print(f"{idx=}'s parent has only one child.")
                    add_parent(idx)
                node_type = END
            else:
                print(
                    f"{idx=} has child candidates: {child_candidate=}, {parent_map=}"
                )
                child = np.random.choice(child_candidate)
                parent_map[child].add(idx)
                update_loop_dict(loop_dict, idx, child)
                if node_type == BRANCH:
                    child_candidate = list(get_child_candidate(idx))
                    print(f"Branched node. {child_candidate=}")
                    if not child_candidate:
                        print(f"No one more child for {idx=}.")
                        node_type = INTER
                        if (
                            len(parent_map[idx]) == 1
                            and len(get_childs(parent_map, parent_map[idx]))
                            == 1
                        ):
                            print(f"{idx=}'s parent has only one child.")
                            add_parent(idx)
                    else:
                        print(f"Add second child for {idx=}.")
                        child = np.random.choice(child_candidate)
                        parent_map[child].add(idx)
                        update_loop_dict(loop_dict, idx, child)

        if node_type == BRANCH:
            branches.append(idx)
        elif node_type == INTER:
            inters.append(idx)
        else:
            ends.append(idx)

    return parent_map, loop_dict
