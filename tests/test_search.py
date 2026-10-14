import unittest

from protinvtree.searching.search import MCTS
from protinvtree.searching.tree import Node
from protinvtree.searching.utils import node_count


class ToyAgent:
    def __init__(self, depth=2):
        self.depth = depth

    def perform_planning(self, parent, logger):
        results = []
        for reward in (0.2, 0.9):
            child = Node(parent=parent, state="candidate", initial_value=reward)
            if child.get_depth() == self.depth:
                child.set_terminated()
            results.append((child, reward))
        return results

    def perform_layer_reasoning(self, parents, logger):
        results = []
        for parent in parents:
            for child, reward in self.perform_planning(parent, logger):
                parent.add_child(child)
                results.append((child, reward))
        return results


class SearchTests(unittest.TestCase):
    def setUp(self):
        node_count.set_count(0)

    def test_backpropagation_uses_average_reward(self):
        search = MCTS(ToyAgent(), "root")
        child = Node(parent=search.root)
        search.backpropagation(child, 0.2)
        search.backpropagation(child, 0.8)
        self.assertEqual(child.visit_count(), 2)
        self.assertAlmostEqual(search.root.value(), 0.5)

    def test_preexpanded_terminals_return_with_zero_simulations(self):
        search = MCTS(ToyAgent(depth=1), "root")
        best, nodes, simulations = search.run_search_with_pre_expansion(0, 1)
        self.assertTrue(best.is_terminated())
        self.assertEqual(best.value(), 0.9)
        self.assertEqual(len(nodes), 3)
        self.assertEqual(simulations, 0)

    def test_terminal_count_is_not_accumulated_across_iterations(self):
        search = MCTS(ToyAgent(), "root", complete_path_n=3)
        best, nodes, simulations = search.run_search_with_pre_expansion(5, 0)
        self.assertTrue(best.is_terminated())
        self.assertEqual(sum(node.is_terminated() for node in nodes), 4)
        self.assertEqual(simulations, 3)

    def test_zero_budget_keeps_root_as_fallback(self):
        search = MCTS(ToyAgent(), "root")
        best, nodes, simulations = search.run_search_with_pre_expansion(0, 0)
        self.assertIs(best, search.root)
        self.assertEqual(simulations, 0)

    def test_empty_expansion_does_not_loop(self):
        class EmptyAgent(ToyAgent):
            def perform_planning(self, parent, logger):
                return []
        search = MCTS(EmptyAgent(), "root")
        _, _, simulations = search.run_search_with_pre_expansion(50, 0)
        self.assertEqual(simulations, 1)

    def exhausted_tree(self, final_selection):
        search = MCTS(ToyAgent(), "root", value_threshold=2.0, final_selection=final_selection)
        search.root.update(0.85)
        for name, branch_value, terminal_value in (("left", 0.9, 0.5), ("right", 0.8, 0.95)):
            branch = Node(parent=search.root, state=name, initial_value=branch_value)
            branch.update(branch_value)
            terminal = Node(parent=branch, state=name + "_terminal", initial_value=terminal_value)
            terminal.set_terminated()
            branch.add_child(terminal)
            search.root.add_child(branch)
        return search

    def test_budget_fallback_defaults_to_original_greedy_path(self):
        search = self.exhausted_tree("greedy")
        best, _, _ = search.run_search_with_pre_expansion(1, 0)
        self.assertEqual(best.state, "left_terminal")

    def test_best_terminal_budget_fallback_is_explicit(self):
        search = self.exhausted_tree("best-terminal")
        best, _, _ = search.run_search_with_pre_expansion(1, 0)
        self.assertEqual(best.state, "right_terminal")
