"""Sequence metrics used in the paper; all comparisons require equal lengths."""

from itertools import combinations


def recovery(predicted, reference):
    if not reference or len(predicted) != len(reference):
        raise ValueError("Sequence recovery requires nonempty, equal-length sequences")
    return sum(a == b for a, b in zip(predicted, reference)) / len(reference)


def novelty(predicted, reference):
    return 1.0 - recovery(predicted, reference)


def diversity(sequences):
    if not sequences:
        raise ValueError("Diversity requires at least one sequence")
    if len({len(sequence) for sequence in sequences}) != 1 or not sequences[0]:
        raise ValueError("Diversity requires nonempty, equal-length sequences")
    if len(sequences) == 1:
        return None
    differences = [novelty(a, b) for a, b in combinations(sequences, 2)]
    return sum(differences) / len(differences)
