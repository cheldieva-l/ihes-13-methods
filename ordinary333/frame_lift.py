"""Rewrite a route through 24 rigid frames, charging every physical slice turn.

This is a route optimizer, not an optimal full-state solver. It keeps the exact
54-label permutation, including the final positions of all six centers.
"""
import argparse
from collections import deque
import itertools
import json
from pathlib import Path
import random
import time

import numpy as np

from . import classical as route_tools

_CONTEXTS = {}


def permutation(word, moves):
    """Input: generator names and pullbacks. Output: their exact product."""
    result = np.arange(54, dtype=np.uint8)
    for name in word:
        result = result[moves[name]]
    return result


def power_word(axis, powers):
    """Input: three mod-four slice powers. Output: minimum parallel-turn word."""
    result = []
    for layer, power in enumerate(powers):
        base = axis + str(layer)
        power %= 4
        result.extend([base] * power if power < 3 else ['-' + base])
    return result


def make_context(moves):
    """Input: dataset moves. Output: verified rotations and conjugation tables."""
    identity = np.arange(54, dtype=np.uint8)
    rotations = [identity]
    words = [[]]
    indices = {identity.tobytes(): 0}
    global_moves = []
    for axis in 'dfr':
        for sign in (1, -1):
            word = power_word(axis, [sign] * 3)
            global_moves.append((permutation(word, moves), word))
    frontier = deque([0])
    while frontier:
        index = frontier.popleft()
        for move, word in global_moves:
            rotation = rotations[index][move]
            key = rotation.tobytes()
            if key not in indices:
                indices[key] = len(rotations)
                rotations.append(rotation)
                words.append(words[index] + word)
                frontier.append(len(rotations) - 1)
                if len(rotations) > 24:
                    raise ValueError('Parallel turns do not form the expected rigid-frame group.')
    if len(rotations) != 24:
        raise ValueError('The dataset does not have exactly 24 rigid frames.')
    names = {np.asarray(p, dtype=np.uint8).tobytes(): name for name, p in moves.items()}
    conjugates = []
    for rotation in rotations:
        inverse = np.argsort(rotation)
        mapping = {}
        for name, move in moves.items():
            key = rotation[move][inverse].tobytes()
            if key not in names:
                raise ValueError('A rigid frame did not preserve the signed generator set.')
            mapping[name] = names[key]
        conjugates.append(mapping)
    transitions = {}
    for axis in 'dfr':
        for q in range(4):
            rotation_q = permutation(power_word(axis, [q] * 3), moves)
            for index, rotation in enumerate(rotations):
                transitions[axis, q, index] = indices[rotation_q[rotation].tobytes()]
    return dict(rotations=rotations, correction=words, conjugates=conjugates,
                transitions=transitions, verified_rotations=len(rotations))


def context_for(moves):
    """Input: generator dictionary. Output: cached, exact-verified frame tables."""
    key = tuple((name, np.asarray(p, dtype=np.uint8).tobytes()) for name, p in moves.items())
    if key not in _CONTEXTS:
        _CONTEXTS[key] = make_context(moves)
    return _CONTEXTS[key]


def dynamic_pass(word, context):
    """Input: route and frame tables. Output: cheapest frame-lifted rewrite.

    At each block the consumed permutation is output * carried_rotation.
    Subtracting one rigid rotation from three parallel slices may replace two
    outer turns by a middle turn. The final carried rotation is paid in full.
    """
    prefixes = {0: []}
    for _, group in itertools.groupby(word, key=lambda name: name.lstrip('-')[0]):
        block = list(group)
        following = {}
        for frame, prefix in prefixes.items():
            mapped = [context['conjugates'][frame][name] for name in block]
            axis = mapped[0].lstrip('-')[0]
            powers = [0, 0, 0]
            for name in mapped:
                powers[int(name[-1])] += -1 if name.startswith('-') else 1
            for q in range(4):
                output = power_word(axis, [value - q for value in powers])
                next_frame = context['transitions'][axis, q, frame]
                candidate = route_tools.reduce_axis_runs(prefix + output)
                previous = following.get(next_frame)
                if previous is None or len(candidate) < len(previous):
                    following[next_frame] = candidate
        prefixes = following
    candidates = [route_tools.reduce_axis_runs(prefix + context['correction'][frame])
                  for frame, prefix in prefixes.items()]
    return min(candidates, key=len)


def optimize(word, moves):
    """Input: route and moves. Output: an exactly equivalent, non-longer route.

    Invalid frame assumptions or a failed equivalence check retain the input.
    Four forward/inverse passes allow neighboring parallel blocks to simplify.
    """
    original = list(word)
    try:
        context = context_for(moves)
        best = route_tools.reduce_axis_runs(original)
        for _ in range(4):
            old_length = len(best)
            forward = dynamic_pass(best, context)
            inverse = route_tools.inverse_word(dynamic_pass(route_tools.inverse_word(best), context))
            best = min((best, forward, inverse), key=len)
            if len(best) >= old_length:
                break
        if len(best) <= len(original) and np.array_equal(permutation(best, moves), permutation(original, moves)):
            return best
    except (ValueError, KeyError, IndexError) as error:
        print(json.dumps(dict(stage='frame_lift_skipped', reason=str(error))), flush=True)
    return original


def self_test(moves, count=100):
    """Input: moves and count. Output: reproducible random-word equivalence audit."""
    rng = random.Random(333)
    names = list(moves)
    failed = []
    raw_failed = []
    improved = 0
    context = context_for(moves)
    for index in range(count):
        word = rng.choices(names, k=rng.randrange(0, 55))
        raw = dynamic_pass(word, context)
        if not np.array_equal(permutation(word, moves), permutation(raw, moves)):
            raw_failed.append(index)
        candidate = optimize(word, moves)
        if len(candidate) > len(word) or not np.array_equal(permutation(word, moves), permutation(candidate, moves)):
            failed.append(index)
        improved += len(candidate) < len(word)
    return dict(random_words=count, failed_indices=failed, raw_failed_indices=raw_failed,
                shortened_random_words=improved,
                rotations=context_for(moves)['verified_rotations'])


def main():
    """Run deterministic exact-equivalence tests; import optimize() for route processing."""
    print(json.dumps(self_test(route_tools.load_moves()),indent=2),flush=True)


if __name__ == '__main__':
    main()
