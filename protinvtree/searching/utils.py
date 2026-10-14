"""Counters and content-based deduplication for search states."""


class NodeCount:
    def __init__(self):
        self.now_idx = 0

    def set_count(self, count):
        self.now_idx = count

    def get_count(self):
        return self.now_idx


node_count = NodeCount()
expand_count = NodeCount()


def state_key(state):
    sequence = state.sequence
    if isinstance(sequence, str):
        return sequence
    return tuple(sequence.detach().cpu().reshape(-1).tolist())


def same_state(left, right):
    return state_key(left) == state_key(right)


def clean_outputs(outputs):
    responses = []
    for parent_idx, choices in enumerate(outputs):
        seen = set()
        for choice in choices:
            if choice is None:
                continue
            key = state_key(choice)
            if key not in seen:
                responses.append((parent_idx, choice))
                seen.add(key)
    return responses


def get_unique_value(values):
    if not values or len(set(values)) != 1:
        raise ValueError("Batched rollouts must all have the same search depth")
    return values[0]
