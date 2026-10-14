"""UCT traversal and average-reward backpropagation, extracted from the original search."""

import logging
import math

from .tree import Node


class MCTS:
    def __init__(self, agent, initial_state, logger=None, c_param=0.01,
                 value_threshold=0.0, complete_path_n=1, final_selection="greedy"):
        self.agent = agent
        self.root = Node(state=initial_state, is_root=True)
        self.logger = logger or logging.getLogger(__name__)
        self.c_param = c_param
        self.value_threshold = value_threshold
        self.complete_path_n = complete_path_n
        if final_selection not in ("greedy", "best-terminal"):
            raise ValueError("final_selection must be greedy or best-terminal")
        self.final_selection = final_selection

    def ucb_value(self, node):
        visits = max(1, node.parent().visit_count()) if node.parent() else 1
        return node.value() + self.c_param * math.sqrt(math.log(visits) / (node.visit_count() + 1))

    def selection(self, node=None):
        node = self.root if node is None else node
        if node.is_leaf():
            return None if node.is_terminated() or node.exhausted else node
        for child in sorted(node.children(), key=self.ucb_value, reverse=True):
            result = self.selection(child)
            if result is not None:
                return result
        return None

    def backpropagation(self, node, reward):
        while node is not None:
            node.update(reward)
            node = node.parent()

    def nodes(self):
        result, pending = [], [self.root]
        while pending:
            node = pending.pop()
            result.append(node)
            pending.extend(reversed(node.children()))
        return result

    def pre_expansion(self, depth):
        parents = [self.root]
        for _ in range(depth):
            parents = [node for node in parents if not node.is_terminated() and not node.exhausted]
            if not parents:
                break
            for parent in parents:
                parent.set_expand_idx()
            # The original layer agent attaches children itself.
            results = self.agent.perform_layer_reasoning(parents, self.logger)
            for node, reward in results:
                self.backpropagation(node, reward)
            for parent in parents:
                if parent.is_leaf():
                    parent.exhausted = True
            parents = [node for node, _ in results]

    def _best_terminal(self):
        terminal = [node for node in self.nodes() if node.is_terminated()]
        return max(terminal, key=lambda node: node.value()) if terminal else None

    def best_path(self):
        node = self.root
        while node.children():
            node = max(node.children(), key=lambda child: child.value())
        return node

    def _threshold_solution(self):
        best = self._best_terminal()
        count = sum(node.is_terminated() for node in self.nodes())
        if best is not None and best.value() > self.value_threshold and count >= self.complete_path_n:
            return best
        return None

    def run_search_with_pre_expansion(self, num_simulations, pre_expansion_depth=2):
        if num_simulations < 0 or pre_expansion_depth < 0:
            raise ValueError("Search budgets cannot be negative")
        self.pre_expansion(pre_expansion_depth)
        simulations = 0
        for _ in range(num_simulations):
            if self._threshold_solution() is not None:
                break
            leaf = self.selection()
            if leaf is None:
                break
            leaf.set_expand_idx()
            results = self.agent.perform_planning(leaf, self.logger)
            for child, reward in results:
                leaf.add_child(child)
                self.backpropagation(child, reward)
            if not results:
                leaf.exhausted = True
            simulations += 1
        # Check after the final expansion too, as in the original run_search.
        best = self._threshold_solution()
        if best is None:
            if self.final_selection == "best-terminal":
                best = self._best_terminal()
            if best is None:
                best = self.best_path()
        return best, self.nodes(), simulations
