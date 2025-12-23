import copy
import json
import time
from functools import reduce
from itertools import chain, combinations
from typing import Dict, List, Set, Tuple

import numpy as np
from graphviz import Digraph
from z3 import *

from util import get_branches, get_dag, get_rev_map, normalize


class SimpleBN:
    def __init__(
        self,
        num_nodes: int = -1,
        prob_type: Tuple[float] = (1.0, 5.0, 1.0),
        path: str = None,
    ):
        if num_nodes == -1 and path is None:
            raise ValueError("Either num_nodes or path should be specified")
        if path is not None:
            self.load(path)
        else:
            self.num_nodes = num_nodes
            self.prob_type = normalize(prob_type)
            self.parent_map = get_dag(num_nodes, self.prob_type)
            branches = get_branches(self.parent_map)
            self.chain_prob_dict = dict(
                zip(
                    branches,
                    [
                        (np.random.randint(0, 2), np.random.random())
                        for _ in branches
                    ],
                )
            )
            self.graph = get_rev_map(self.parent_map)
            self.reach_prob_dict = None
            self.compute_reachability()
            self.dominant_dict = self.get_dominant_dict()
            self.edges = [
                (parent, child)
                for parent, childs in self.graph.items()
                for child in childs
            ]
            self.edge2idx = {edge: idx for idx, edge in enumerate(self.edges)}
            self.edgeidx2edge = dict(enumerate(self.edges))
            self.solver = self.build_path_constraints()

    def dump(self, fpath: str):
        obj = {
            "num_nodes": self.num_nodes,
            "prob_type": self.prob_type,
            "parent_map": {k: list(v) for k, v in self.parent_map.items()},
            "chain_prob_dict": self.chain_prob_dict,
            "graph": self.graph,
            "reach_prob_dict": self.reach_prob_dict,
            "dominant_dict": {
                k: list(v) for k, v in self.dominant_dict.items()
            },
            "edges": self.edges,
            "edge2idx": {str(k): v for k, v in self.edge2idx.items()},
            "edgeidx2edge": self.edgeidx2edge,
            "solver": str(self.solver),
        }
        with open(fpath, "w") as f:
            json.dump(obj, f, indent=4)

    def load(self, path: str):
        with open(path, "r") as f:
            obj = json.load(f)
        self.num_nodes = obj["num_nodes"]
        self.prob_type = obj["prob_type"]
        self.parent_map = {int(k): set(v) for k, v in obj["parent_map"].items()}
        self.chain_prob_dict = {
            int(k): v for k, v in obj["chain_prob_dict"].items()
        }
        self.graph = {int(k): v for k, v in obj["graph"].items()}
        self.reach_prob_dict = {
            int(k): v for k, v in obj["reach_prob_dict"].items()
        }
        self.dominant_dict = {
            int(k): v for k, v in obj["dominant_dict"].items()
        }
        self.edges = [tuple(edge) for edge in obj["edges"]]
        self.edge2idx = {
            tuple(int(e) for e in k.strip("()").split(", ")): v
            for k, v in obj["edge2idx"].items()
        }
        self.edgeidx2edge = {
            int(k): tuple(v) for k, v in obj["edgeidx2edge"].items()
        }
        self.solver = self.build_path_constraints()

    def draw(self) -> Digraph:
        d = Digraph()
        for node in self.graph:
            d.node(str(node), xlabel=f"{self.reach_prob_dict[node]:.2e}")
        for node, childs in self.graph.items():
            if len(childs) == 1:
                d.edge(str(node), str(childs[0]))
        for prnt_idx, (true_br, prob) in self.chain_prob_dict.items():
            childs = self.graph[prnt_idx]
            d.edge(str(prnt_idx), str(childs[true_br]), label=f"{prob:.2f}")
            d.edge(
                str(prnt_idx), str(childs[1 - true_br]), label=f"{1 - prob:.2f}"
            )
        return d

    def compute_reachability(self):
        self.reach_prob_dict = {}
        for node in range(self.num_nodes):
            if node == 0:
                self.reach_prob_dict[node] = 1
            else:
                self.reach_prob_dict[node] = 0
                for parent in self.parent_map[node]:
                    if parent not in self.chain_prob_dict:
                        self.reach_prob_dict[node] += self.reach_prob_dict[
                            parent
                        ]
                    else:
                        true_br, prob = self.chain_prob_dict[parent]
                        if self.graph[parent][true_br] == node:
                            self.reach_prob_dict[node] += (
                                self.reach_prob_dict[parent] * prob
                            )
                        else:
                            self.reach_prob_dict[node] += self.reach_prob_dict[
                                parent
                            ] * (1 - prob)

    def gen_obss(
        self,
        num_obs: int,
        start_idx: int = 0,
        context: np.ndarray = None,
        debug: bool = False,
    ) -> np.ndarray:
        if debug:
            print(f"Generating {num_obs} observations")
        start_time = time.time()
        obss = []
        for _ in range(num_obs):
            obs = [0] * len(self.edges)
            if context is not None:
                obs = context.copy()
            curr_node = start_idx
            while len(self.graph[curr_node]):
                prev_node = curr_node
                childs = self.graph[curr_node]
                if len(childs) > 1:
                    true_br, prob = self.chain_prob_dict[curr_node]
                    if np.random.random() < prob:
                        curr_node = childs[true_br]
                    else:
                        curr_node = childs[1 - true_br]
                else:
                    curr_node = childs[0]
                obs[self.edge2idx[(prev_node, curr_node)]] = 1
            obss.append(obs)
        obss = np.array(obss, dtype=np.int8)
        if debug:
            print(
                f"Generated {num_obs} observations in {time.time() - start_time:.2f}s"
            )
        return obss

    def edgecov2nodecov(self, edgecov: np.ndarray) -> np.ndarray:
        nodecov = np.zeros((edgecov.shape[0], self.num_nodes), dtype=np.int8)
        if len(edgecov):
            for edgeidx, edge in self.edgeidx2edge.items():
                nodecov[:, edge[0]] += edgecov[:, edgeidx]
                nodecov[:, edge[1]] += edgecov[:, edgeidx]
            nodecov = nodecov.astype(bool).astype(np.int8)
        return nodecov

    def get_parents(self, node: int) -> Set[int]:
        return self.parent_map[node]

    def get_children(self, node: int) -> Set[int]:
        return self.graph[node]

    def get_ancestors(self, node: int) -> Set[int]:
        visited = set()
        ancestors = set()
        queue = {node}
        while queue:
            curr_node = queue.pop()
            if curr_node in visited:
                continue
            visited.add(curr_node)
            ancestors.add(curr_node)
            queue.update(self.get_parents(curr_node))
        return ancestors

    def get_descendants(self, node: int) -> Set[int]:
        visited = set()
        descendants = set()
        queue = {node}
        while queue:
            curr_node = queue.pop()
            if curr_node in visited:
                continue
            visited.add(curr_node)
            descendants.add(curr_node)
            queue.update(self.get_children(curr_node))
        return descendants

    def cf_simulate(
        self,
        num_obs: int,
        cf_idx: int,
        pred_idx: int,
        orig_obss: np.ndarray,
        weights: np.ndarray,
    ):
        raise Exception("Not updated after node obs -> edge obs change")
        # choosing random num_obs observations from orig_obss
        norm_weight = normalize(weights)
        random_idxs = np.random.choice(
            len(orig_obss), num_obs, replace=True, p=norm_weight
        )
        contexts = orig_obss[random_idxs].copy()
        # if context is one dimensional, make it two dimensional
        if len(contexts.shape) == 1:
            contexts = contexts.reshape(-1, 1)
        contexts[:, pred_idx:] = 0
        cf_obss = [
            self.gen_obss(1, start_idx=cf_idx, context=context).reshape(-1)
            for context in contexts
        ]
        return np.array(cf_obss)

    def get_dominant_dict(self) -> Dict[int, Set[int]]:
        dominant_dict = {0: {0}}
        for node in range(1, self.num_nodes):
            dominant_dict[node] = set(range(self.num_nodes))
        while True:
            prev_dict = copy.deepcopy(dominant_dict)
            for node in range(1, self.num_nodes):
                parents = list(self.get_parents(node))
                intersect_parent = reduce(
                    set.intersection,
                    [dominant_dict[p] for p in parents],
                    dominant_dict[parents[0]],
                )
                dominant_dict[node] = {node} | intersect_parent
            stop_flag = all(
                dominant_dict[node] == prev_dict[node]
                for node in range(1, self.num_nodes)
            )
            if stop_flag:
                break
        return dominant_dict

    def is_dominated(self, node1: int, node2: int) -> bool:
        """
        check node1 dominates node2
        """
        return node1 in self.dominant_dict[node2]

    def build_path_constraints(self) -> Solver:
        # sourcery skip: raise-specific-error
        s = Solver()
        # if edge e_i_j exists, then v_i and v_j
        for parent, childs in self.graph.items():
            for child in childs:
                s.add(
                    Implies(
                        Bool(f"e_{parent}_{child}"),
                        And(Bool(f"v_{parent}"), Bool(f"v_{child}")),
                    )
                )
        # if v_j, then one of e_i_j is true
        for child, parents in self.parent_map.items():
            if len(parents):
                s.add(
                    Implies(
                        Bool(f"v_{child}"),
                        Or([Bool(f"e_{p}_{child}") for p in parents]),
                    )
                )
        # if v_i, then only one of e_i_j is true
        for parent, childs in self.graph.items():
            if len(childs) == 1:
                s.add(
                    Implies(
                        Bool(f"v_{parent}"), Bool(f"e_{parent}_{childs[0]}")
                    )
                )
            elif len(childs) == 2:
                child1, child2 = childs
                s.add(
                    Implies(
                        Bool(f"v_{parent}"),
                        Or(
                            Bool(f"e_{parent}_{child1}"),
                            Bool(f"e_{parent}_{child2}"),
                        ),
                    )
                )
                s.add(
                    Or(
                        Not(Bool(f"e_{parent}_{child1}")),
                        Not(Bool(f"e_{parent}_{child2}")),
                    )
                )

        s.add(Bool("v_0"))
        if s.check() != sat:
            raise Exception(f"No solution.\n{s}")
        return s

    def solve_path(self, nodes, constraints) -> bool:
        solver = copy.deepcopy(self.solver)
        for node, constraint in zip(nodes, constraints):
            if constraint:
                solver.add(Bool(f"v_{node}"))
            else:
                solver.add(Not(Bool(f"v_{node}")))
        return solver.check() == sat

    def solve_equivalent_edge(
        self, edge1: Tuple[int, int], edge2: Tuple[int, int]
    ) -> bool:
        solver = copy.deepcopy(self.solver)
        solver.add(
            And(
                Bool(f"e_{edge1[0]}_{edge1[1]}"),
                Not(Bool(f"e_{edge2[0]}_{edge2[1]}")),
            )
        )
        return solver.check() == unsat

    def solve_satisfiable_edge(self, edges: List[Tuple[int, int]]) -> bool:
        solver = copy.deepcopy(self.solver)
        for edge in edges:
            solver.add(Bool(f"e_{edge[0]}_{edge[1]}"))
        return solver.check() == sat


class MediumBN:
    def __init__(
        self,
        num_nodes: int = -1,
        prob_type: Tuple[float] = (1.0, 5.0, 1.0),
        path: str = None,
    ):
        if num_nodes == -1 and path is None:
            raise ValueError("Either num_nodes or path should be specified")
        if path is not None:
            self.load(path)
        else:
            self.num_nodes = num_nodes
            self.prob_type = normalize(prob_type)
            self.parent_map_cont = get_dag(num_nodes, self.prob_type)
            self.graph = get_rev_map(self.parent_map_cont)
            branches = get_branches(self.parent_map_cont)
            self.parent_map_data = {}
            for branch in branches:
                # there's data dependency
                if np.random.random() < 0.5:
                    preds = sorted(self.get_pred(branch))
                    if len(preds) == 0:
                        continue
                    proportion = np.array([2**i for i in range(len(preds))])
                    self.parent_map_data[branch] = int(
                        np.random.choice(preds, p=proportion / proportion.sum())
                    )
            self.chain_prob_dict = {}
            for branch in branches:
                if branch in self.parent_map_data:
                    self.chain_prob_dict[branch] = (
                        np.random.randint(0, 2),
                        (np.random.random(), np.random.random()),
                    )
                else:
                    self.chain_prob_dict[branch] = (
                        np.random.randint(0, 2),
                        np.random.random(),
                    )
            self.reach_prob_dict = None
            self.compute_reachability()
            self.dominant_dict = self.get_dominant_dict()

    def load(self, path: str):
        with open(path, "r") as f:
            obj = json.load(f)
        self.num_nodes = obj["num_nodes"]
        self.prob_type = obj["prob_type"]
        self.parent_map_cont = {
            int(k): v for k, v in obj["parent_map_cont"].items()
        }
        self.parent_map_data = {
            int(k): v for k, v in obj["parent_map_data"].items()
        }
        self.graph = {int(k): v for k, v in obj["graph"].items()}
        self.chain_prob_dict = {
            int(k): v for k, v in obj["chain_prob_dict"].items()
        }
        self.reach_prob_dict = {
            int(k): v for k, v in obj["reach_prob_dict"].items()
        }
        self.dominant_dict = {
            int(k): v for k, v in obj["dominant_dict"].items()
        }

    def get_pred(self, node: int) -> List[int]:
        queue = [node]
        visited = set()
        while queue:
            curr_node = queue.pop()
            if curr_node in visited:
                continue
            visited.add(curr_node)
            queue.extend(self.parent_map_cont[curr_node])
        visited.remove(node)
        return list(visited)

    def compute_reachability(self):  # sourcery skip: remove-redundant-if
        path_prob_dict = {}
        for node in range(self.num_nodes):
            if node == 0:
                path_prob_dict[(0,)] = 1.0
                continue
            temp_dict = path_prob_dict.copy()
            for path in temp_dict:
                last_node = path[-1]
                if last_node in self.parent_map_cont[node]:
                    parent = last_node
                    if parent not in self.chain_prob_dict:
                        path_prob_dict[path + (node,)] = path_prob_dict[path]
                    elif parent in self.parent_map_data:
                        data_parent = self.parent_map_data[parent]
                        is_data_cov = data_parent in path
                        true_br, (
                            prob_data_false,
                            prob_data_true,
                        ) = self.chain_prob_dict[parent]
                        is_true_br = self.graph[parent][true_br] == node
                        if is_data_cov and is_true_br:
                            path_prob_dict[path + (node,)] = (
                                path_prob_dict[path] * prob_data_true
                            )
                        elif is_data_cov and not is_true_br:
                            path_prob_dict[path + (node,)] = path_prob_dict[
                                path
                            ] * (1 - prob_data_true)
                        elif not is_data_cov and is_true_br:
                            path_prob_dict[path + (node,)] = (
                                path_prob_dict[path] * prob_data_false
                            )
                        elif not is_data_cov and not is_true_br:
                            path_prob_dict[path + (node,)] = path_prob_dict[
                                path
                            ] * (1 - prob_data_false)
                    else:
                        true_br, prob = self.chain_prob_dict[parent]
                        is_true_br = self.graph[parent][true_br] == node
                        if is_true_br:
                            path_prob_dict[path + (node,)] = (
                                path_prob_dict[path] * prob
                            )
                        else:
                            path_prob_dict[path + (node,)] = path_prob_dict[
                                path
                            ] * (1 - prob)
        self.reach_prob_dict = {node: 0.0 for node in range(self.num_nodes)}
        for node in range(self.num_nodes):
            if node == 0:
                self.reach_prob_dict[node] = 1.0
                continue
            for path in path_prob_dict:
                if path[-1] == node:
                    self.reach_prob_dict[node] += path_prob_dict[path]

    def draw(self):
        d = Digraph()
        for node in self.graph:
            d.node(str(node), xlabel=f"{self.reach_prob_dict[node]:.2e}")
        for branch, data_parent in self.parent_map_data.items():
            d.edge(str(data_parent), str(branch), color="blue")
        for node, childs in self.graph.items():
            if len(childs) == 1:
                d.edge(str(node), str(childs[0]))
        for parent_idx, prob_obj in self.chain_prob_dict.items():
            childs = self.graph[parent_idx]
            if parent_idx not in self.parent_map_data:
                true_br, prob = prob_obj
                d.edge(
                    str(parent_idx), str(childs[true_br]), label=f"{prob:.2f}"
                )
                d.edge(
                    str(parent_idx),
                    str(childs[1 - true_br]),
                    label=f"{1 - prob:.2f}",
                )
            else:
                true_br, (prob_data_false, prob_data_true) = prob_obj
                d.edge(
                    str(parent_idx),
                    str(childs[true_br]),
                    label=f"0:{prob_data_false:.2f}\n1:{prob_data_true:.2f}",
                )
                d.edge(
                    str(parent_idx),
                    str(childs[1 - true_br]),
                    label=f"0:{1 - prob_data_false:.2f}\n1:{1 - prob_data_true:.2f}",
                )
        return d

    def dump(self, fpath: str) -> Dict:
        obj = {
            "num_nodes": self.num_nodes,
            "prob_type": self.prob_type,
            "parent_map_cont": {
                k: list(v) for k, v in self.parent_map_cont.items()
            },
            "parent_map_data": self.parent_map_data,
            "chain_prob_dict": self.chain_prob_dict,
            "graph": self.graph,
            "reach_prob_dict": self.reach_prob_dict,
            "dominant_dict": self.dominant_dict,
        }
        with open(fpath, "w") as f:
            json.dump(obj, f, indent=4)

    def gen_obss(
        self,
        num_obs: int,
        start_idx: int = 0,
        context: np.ndarray = None,
        debug: bool = False,
    ) -> np.ndarray:
        # sourcery skip: merge-else-if-into-elif, swap-if-else-branches
        if debug:
            print(f"Generating {num_obs} observations")
        start_time = time.time()
        obss = []
        for i in range(num_obs):
            if i % 1000000 == 0 and i > 0:
                print(f"{i=}", end=" ")
            obs = [0] * self.num_nodes
            if context is not None:
                obs = context.copy()
            obs[start_idx] = 1
            curr_node = start_idx
            while len(self.graph[curr_node]):
                childs = self.graph[curr_node]
                if len(childs) > 1:
                    if curr_node not in self.parent_map_data:
                        true_br, prob = self.chain_prob_dict[curr_node]
                        if np.random.random() < prob:
                            obs[childs[true_br]] = 1
                            curr_node = childs[true_br]
                        else:
                            obs[childs[1 - true_br]] = 1
                            curr_node = childs[1 - true_br]
                    else:
                        data_parent = self.parent_map_data[curr_node]
                        true_br, (
                            prob_data_false,
                            prob_data_true,
                        ) = self.chain_prob_dict[curr_node]
                        if not obs[data_parent]:
                            if np.random.random() < prob_data_false:
                                obs[childs[true_br]] = 1
                                curr_node = childs[true_br]
                            else:
                                obs[childs[1 - true_br]] = 1
                                curr_node = childs[1 - true_br]
                        else:
                            if np.random.random() < prob_data_true:
                                obs[childs[true_br]] = 1
                                curr_node = childs[true_br]
                            else:
                                obs[childs[1 - true_br]] = 1
                                curr_node = childs[1 - true_br]
                else:
                    obs[childs[0]] = 1
                    curr_node = childs[0]
            obss.append(obs)
        obss = np.array(obss, dtype=np.int8)
        if debug:
            print(
                f"Generated {num_obs} observations in {time.time() - start_time:.2f}s"
            )
        return obss

    def get_parents(self, node: int) -> Set[int]:
        return self.parent_map_cont[node]

    def cf_simulate(
        self,
        num_obs: int,
        cf_idx: int,
        pred_idx: int,
        orig_obss: np.ndarray,
        weights: np.ndarray,
    ):
        # choosing random num_obs observations from orig_obss
        norm_weight = normalize(weights)
        random_idxs = np.random.choice(
            len(orig_obss), num_obs, replace=True, p=norm_weight
        )
        contexts = orig_obss[random_idxs].copy()
        # if context is one dimensional, make it two dimensional
        if len(contexts.shape) == 1:
            contexts = contexts.reshape(-1, 1)
        contexts[:, pred_idx:] = 0
        cf_obss = [
            self.gen_obss(1, start_idx=cf_idx, context=context).reshape(-1)
            for context in contexts
        ]
        return np.array(cf_obss)

    def get_dominant_dict(self):
        dominant_dict = {0: {0}}
        for node in range(1, self.num_nodes):
            dominant_dict[node] = set(range(self.num_nodes))
        while True:
            prev_dict = copy.deepcopy(dominant_dict)
            for node in range(1, self.num_nodes):
                parents = list(self.get_parents(node))
                intersect_parent = reduce(
                    set.intersection,
                    [dominant_dict[p] for p in parents],
                    dominant_dict[parents[0]],
                )
                dominant_dict[node] = {node} | intersect_parent
            stop_flag = all(
                dominant_dict[node] == prev_dict[node]
                for node in range(1, self.num_nodes)
            )
            if stop_flag:
                break
        return dominant_dict

    def is_dominated(self, node1: int, node2: int) -> bool:
        """
        check node1 dominates node2
        """
        return node1 in self.dominant_dict[node2]

    def get_children(self, node: int) -> Set[int]:
        return self.graph[node]
