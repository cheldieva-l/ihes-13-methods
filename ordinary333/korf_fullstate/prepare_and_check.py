"""Export direct source states and run only bounded adapter/shallow-solver checks."""

import argparse
import json
import random
import re
import subprocess
import time
from pathlib import Path

from compact_adapter import (CompactAdapter, CubieState, EXE, HERE, ORBITS, PUBLIC,
                             export_sources, from_twsearch, native_replay,
                             read_source_states, to_twsearch)


def run_official(definition, extra=(), input_text=None, seconds=20):
    command = [str(EXE), "-q", "-M", "64", "-t", "2", "--nowrite",
               "--cachedir", str(HERE / "smoke_cache"), *map(str, extra), str(definition)]
    started = time.monotonic()
    try:
        process = subprocess.run(command, input=input_text, capture_output=True, text=True,
                                 cwd=HERE, timeout=seconds)
        return dict(command=command, exit_code=process.returncode, timed_out=False,
                    seconds=time.monotonic()-started, stdout=process.stdout, stderr=process.stderr)
    except subprocess.TimeoutExpired as exc:
        def text(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        return dict(command=command, exit_code=None, timed_out=True,
                    seconds=time.monotonic()-started, stdout=text(exc.stdout), stderr=text(exc.stderr))
    except OSError as exc:
        return dict(command=command, exit_code=None, timed_out=False,
                    seconds=time.monotonic()-started, stdout="", stderr=str(exc))


def parse_shown_states(stdout):
    result = []
    for block in re.findall(r"Scramble noname\s*\n(.*?)\nEnd", stdout, re.S):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        decoded = []
        try:
            for name, count, modulus in ORBITS:
                at = lines.index(name)
                permutation = tuple(int(x)-1 for x in lines[at+1].split())
                orientation = tuple(map(int, lines[at+2].split())) if modulus > 1 else (0,)*count
                decoded.append((permutation, orientation))
            result.append(CubieState(*decoded[0], *decoded[1], decoded[2][0]))
        except (ValueError, IndexError):
            result.append(None)
    return result


def adapter_checks(adapter, states):
    """Failures are ordinary report records, never assertion-driven run failures."""
    checks = []

    def record(name, ok, **details):
        checks.append(dict(name=name, ok=bool(ok), **details))

    official = json.loads((PUBLIC / "puzzle_info.json").read_text(encoding="utf-8"))
    record("geometry_matches_public_moves", adapter.moves == official["generators"])
    record("geometry_matches_public_goal", adapter.central == official["central_state"])
    record("adapter_initialization", not adapter.errors, errors=adapter.errors)
    inverse_checks = 0
    for token, transform in adapter.transforms.items():
        inverse = token[1:] if token.startswith("-") else "-" + token
        one = native_replay(adapter.central, [token], adapter.moves)
        record("single_move_roundtrip:" + token, adapter.decode(transform) == one)
        record("inverse:" + token,
               adapter.compose(transform, adapter.transforms[inverse]) == adapter.solved)
        current = adapter.solved
        for _ in range(4):
            current = adapter.compose(current, transform)
        record("fourth_power:" + token, current == adapter.solved)
        inverse_checks += 1
    rng = random.Random(734210)
    tokens = list(adapter.moves)
    transition_count = 0
    for index in range(100):
        word = [rng.choice(tokens) for _ in range(rng.randrange(0, 81))]
        native = native_replay(adapter.central, word, adapter.moves)
        compact = adapter.solved
        for token in word:
            compact = adapter.compose(compact, adapter.transforms[token])
        encoded, reason = adapter.encode(native)
        record(f"random_roundtrip:{index}", not reason and encoded == compact
               and adapter.decode(compact) == native, length=len(word), reason=reason)
        for token in tokens:
            child = adapter.compose(compact, adapter.transforms[token])
            expected = native_replay(native, [token], adapter.moves)
            record(f"random_transition:{index}:{token}", adapter.decode(child) == expected)
            transition_count += 1
    source_errors = []
    for puzzle_id, native in states.items():
        encoded, reason = adapter.encode(native)
        if reason or adapter.decode(encoded) != native:
            source_errors.append(dict(puzzle_id=puzzle_id, reason=reason or "roundtrip_failed"))
    record("all_source_roundtrips", not source_errors, count=len(states), failures=source_errors)
    moved_centers = [token for token in tokens
                     if adapter.transforms[token].centers != adapter.solved.centers]
    record("middle_turn_centers_retained", set(moved_centers) ==
           {"f1", "-f1", "r1", "-r1", "d1", "-d1"}, tokens=moved_centers)
    malformed = list(range(54))
    malformed[0], malformed[1] = malformed[1], malformed[0]
    record("nonrigid_input_reported", adapter.encode(malformed)[1] is not None)
    return dict(ok=all(row["ok"] for row in checks), check_count=len(checks),
                random_transition_count=transition_count, source_count=len(states),
                failed=[row for row in checks if not row["ok"]],
                inverse_move_count=inverse_checks)


def official_convention(adapter, definition, seconds):
    rng = random.Random(2934)
    tokens = list(adapter.moves)
    words = [[token] for token in tokens] + [
        [rng.choice(tokens) for _ in range(length)] for length in (2, 5, 17, 41)]
    receipt = run_official(definition, ["--showpositions"],
                           "\n".join(" ".join(map(to_twsearch, w)) for w in words)+"\n", seconds)
    shown = parse_shown_states(receipt["stdout"])
    checks = []
    for i, word in enumerate(words):
        obtained = adapter.decode(shown[i]) if i < len(shown) and shown[i] else None
        checks.append(dict(word=word, ok=obtained == native_replay(adapter.central, word, adapter.moves)))
    receipt.update(checks=checks, shown_count=len(shown),
                   ok=len(shown) == len(words) and all(row["ok"] for row in checks)
                   and receipt["exit_code"] == 0 and not receipt["timed_out"])
    return receipt


def shallow_solve_checks(adapter, definition, states, seconds):
    """One bounded official process; source states plus constructed targets of depth<=5."""
    cases = [(f"source_{pid}", states[pid]) for pid in (0, 3, 4, 5, 6, 7) if pid in states]
    for index, word in enumerate(([], ["r1"], ["f0", "f0"],
                                  ["r1", "-d0", "f2"],
                                  ["f1", "r0", "-d1", "r2", "d2"])):
        cases.append((f"synthetic_{index}", native_replay(adapter.central, word, adapter.moves)))
    scrambles, failures = [], []
    for name, native in cases:
        state, reason = adapter.encode(native)
        if reason:
            failures.append(dict(name=name, status="F", reason=reason))
        else:
            scrambles.append(adapter.scramble_text(name, state))
    target_file = HERE / "shallow_checks.scr"
    target_file.write_text("".join(scrambles), encoding="utf-8")
    # Official twsearch accepts position blocks on stdin when no scramble filename is supplied.
    # --scramblefile is not a supported flag, so append the verified file after the definition.
    command = [str(EXE), "-q", "-M", "64", "-t", "2", "--nowrite", "--cachedir",
               str(HERE / "smoke_cache"), "--maxdepth", "5", str(definition), str(target_file)]
    started = time.monotonic()
    try:
        process = subprocess.run(command, text=True, capture_output=True, cwd=HERE, timeout=seconds)
        receipt = dict(command=command, exit_code=process.returncode, timed_out=False,
                       seconds=time.monotonic()-started, stdout=process.stdout, stderr=process.stderr)
    except subprocess.TimeoutExpired as exc:
        text = lambda x: x.decode(errors="replace") if isinstance(x, bytes) else x or ""
        receipt = dict(command=command, exit_code=None, timed_out=True,
                       seconds=time.monotonic()-started, stdout=text(exc.stdout), stderr=text(exc.stderr))
    except OSError as exc:
        receipt = dict(command=command, exit_code=None, timed_out=False,
                       seconds=time.monotonic()-started, stdout="", stderr=str(exc))
    blocks = dict(re.findall(r"Solving (\S+)\n(.*?)(?=\nSolving |\Z)", receipt["stdout"], re.S))
    results = []
    for name, native in cases:
        solutions = []
        for line in blocks.get(name, "").splitlines():
            parts = line.split()
            word = [from_twsearch(token) for token in parts]
            if parts and all(token is not None for token in word):
                solutions.append(dict(word=word, length=len(word),
                                      valid=native_replay(native, word, adapter.moves) == adapter.central))
        if native == adapter.central and re.search(r"Found 1 solution max depth 0", blocks.get(name, "")):
            solutions.append(dict(word=[], length=0, valid=True))
        results.append(dict(name=name, solutions=solutions,
                            ok=bool(solutions) and all(s["valid"] and s["length"] <= 5 for s in solutions)))
    receipt.update(results=results, input_failures=failures,
                   ok=all(row["ok"] for row in results) and not failures
                   and receipt["exit_code"] == 0 and not receipt["timed_out"])
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-seconds", type=float, default=20)
    parser.add_argument("--skip-official", action="store_true")
    args = parser.parse_args()
    adapter = CompactAdapter()
    states, source_errors = read_source_states()
    numerical = adapter_checks(adapter, states)
    (HERE / "adapter_checks.json").write_text(json.dumps(numerical, indent=2), encoding="utf-8")
    exported = export_sources(adapter, states)
    report = dict(adapter=numerical, exported=exported, source_errors=source_errors)
    print(json.dumps(dict(stage="adapter_export", checks_ok=numerical["ok"],
                          exported=exported["exported_count"], failures=exported["failures"])), flush=True)
    if not args.skip_official:
        convention = official_convention(adapter, Path(exported["definition"]), args.smoke_seconds)
        (HERE / "official_convention.json").write_text(json.dumps(convention, indent=2), encoding="utf-8")
        report["official_convention_ok"] = convention["ok"]
        print(json.dumps(dict(stage="official_convention", ok=convention["ok"],
                              shown_count=convention["shown_count"])), flush=True)
        if convention["ok"]:
            shallow = shallow_solve_checks(adapter, Path(exported["definition"]), states, args.smoke_seconds)
            (HERE / "shallow_solver_receipt.json").write_text(json.dumps(shallow, indent=2), encoding="utf-8")
            report["shallow_solves_ok"] = shallow["ok"]
            print(json.dumps(dict(stage="shallow_solves", ok=shallow["ok"],
                                  results=shallow["results"], seconds=shallow["seconds"])), flush=True)
        else:
            report["shallow_solves_status"] = "not_run_unsafe_adapter_convention"
    (HERE / "checks_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


