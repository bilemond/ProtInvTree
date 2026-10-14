"""Search nodes compatible with the supplied ProtInvTree agent."""

from .utils import node_count, expand_count


class Node:
    def __init__(self, parent=None, state=None, initial_value=0.0, is_root=False):
        self._parent = parent
        self._children = []
        self._visit_count = 0
        self._value_sum = 0.0
        self._initial_value = initial_value
        self._terminated = False
        self._is_root = is_root
        self.state = state
        self.idx = node_count.get_count()
        node_count.set_count(self.idx + 1)
        self.expand_idx = -1
        self.rollout_res = []
        self.exhausted = False

    def value(self):
        return self._value_sum / self._visit_count if self._visit_count else self._initial_value

    def parent(self):
        return self._parent

    def children(self):
        return self._children

    def visit_count(self):
        return self._visit_count

    def is_root(self):
        return self._is_root

    def is_leaf(self):
        return not self._children

    def is_terminated(self):
        return self._terminated

    def set_terminated(self):
        self._terminated = True

    def update(self, value):
        self._visit_count += 1
        self._value_sum += value

    def add_child(self, node):
        self._children.append(node)

    def set_expand_idx(self):
        self.expand_idx = expand_count.get_count()
        expand_count.set_count(self.expand_idx + 1)

    def get_depth(self):
        return 0 if self._parent is None else self._parent.get_depth() + 1

    def get_index_in_layer(self):
        root = self
        while root.parent() is not None:
            root = root.parent()
        layer = [root]
        for _ in range(self.get_depth()):
            layer = [child for node in layer for child in node.children()]
        return layer.index(self)
